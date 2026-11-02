"""The benchmark harness: suites run through `run_call`, scored by metric tasks, written as `benchmark-result@1`.

A suite (`suites/<name>.yaml`, `benchmark-suite@1.0`) lists benchmarks; each names a data manifest in `manifests/`
(`data-manifest@2.0`, fetched by `senselab_core.runtime.fetch`, the engine behind `senselab data fetch`), an adapter
that turns the fetched files into items, the task to run per item and the metric task to score with. Every item
belongs to `calibration` or `heldout` by the digest of its id. Each run writes `<out>/<run id>/result.json` and the
PROV documents of every call under `<out>/<run id>/prov/`.
"""

import hashlib
import json
import tarfile
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Protocol

import numpy
import yaml
from senselab_core.kernel._base import Frozen, NonEmptyStr
from senselab_core.kernel.formats import BENCHMARK_RESULT_FORMAT, FormatTag, Formatted
from senselab_core.kernel.identity import ExecutionContext
from senselab_core.kernel.results import Embedding, Intervals, Metric, TextSpans, Transcript, TranscriptNode, Trials
from senselab_core.kernel.signals import PlainText
from senselab_core.runtime.call import CallFailed, Result, TaskCall, run_call
from senselab_core.runtime.fetch import DataManifest, fetch
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.store import ResultStore
from senselab_core.testing.synth import tone

SUITE_FORMAT = "benchmark-suite@1.0"
SPLITS = ("calibration", "heldout")
POOLED = "pooled:"


def split_of(item_id: str) -> str:
    """Return `calibration` or `heldout` by the parity of the first byte of the sha256 of `item_id`."""
    return SPLITS[hashlib.sha256(item_id.encode()).digest()[0] % 2]


def _by_digest(ids: Sequence[str]) -> list[str]:
    """Return `ids` ordered by the sha256 of each, a selection order independent of the dataset's own."""
    return sorted(ids, key=lambda i: hashlib.sha256(i.encode()).hexdigest())


@dataclass(frozen=True)
class Item:
    """One benchmark item: its id, the task's inputs and what the adapter keeps for scoring."""

    id: str
    inputs: Mapping[str, Any]
    reference: Any = None


@dataclass(frozen=True)
class Scoring:
    """One metric call: its row `label`, the `split` it belongs to and the metric task's inputs."""

    label: str
    split: str
    inputs: Mapping[str, Any]


class Adapter(Protocol):
    """Turns fetched files into items, and a task's outputs into metric inputs."""

    def items(self, files: Mapping[str, Path], count: int, work: Path) -> list[Item]:
        """Return `count` items; any file the task reads is written under `work`."""
        ...

    def scoring(self, items: Sequence[Item], outputs: Mapping[str, Mapping[str, Any]]) -> list[Scoring]:
        """Return the metric calls over the `outputs` (by item id) of the items that ran."""
        ...


class _PerItem:
    """Scores each item's output alone: `metric_inputs(item, outputs)`."""

    def scoring(self, items: Sequence[Item], outputs: Mapping[str, Mapping[str, Any]]) -> list[Scoring]:
        return [Scoring(i.id, split_of(i.id), self.metric_inputs(i, outputs[i.id])) for i in items if i.id in outputs]

    def metric_inputs(self, item: Item, output: Mapping[str, Any]) -> Mapping[str, Any]:
        raise NotImplementedError


def _read_tar(path: Path) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Return `{utterance id: transcript}` and `[(utterance id, member name)]` of a LibriSpeech tar."""
    transcripts: dict[str, str] = {}
    audio: list[tuple[str, str]] = []
    with tarfile.open(path, "r|gz") as tar:
        for member in tar:
            if member.name.endswith(".trans.txt"):
                handle = tar.extractfile(member)
                assert handle is not None
                for line in handle.read().decode("utf-8").splitlines():
                    uid, _, text = line.partition(" ")
                    transcripts[uid] = text
            elif member.name.endswith(".flac"):
                audio.append((Path(member.name).stem, member.name))
    return transcripts, audio


def _extract(path: Path, wanted: Mapping[str, str], work: Path) -> dict[str, Path]:
    """Write the tar members named by `wanted` (`{utterance id: member}`) under `work`; return their paths."""
    work.mkdir(parents=True, exist_ok=True)
    by_member = {m: u for u, m in wanted.items()}
    out: dict[str, Path] = {}
    with tarfile.open(path, "r|gz") as tar:
        for member in tar:
            if member.name in by_member:
                handle = tar.extractfile(member)
                assert handle is not None
                target = work / f"{by_member[member.name]}.flac"
                target.write_bytes(handle.read())
                out[by_member[member.name]] = target
    return out


def _libri(files: Mapping[str, Path]) -> Path:
    [path] = files.values()
    return path


class LibriSpeechAsr(_PerItem):
    """LibriSpeech test-clean utterances, transcribed; the metric reads the reference and hypothesis transcripts."""

    def items(self, files: Mapping[str, Path], count: int, work: Path) -> list[Item]:
        tar_path = _libri(files)
        transcripts, audio = _read_tar(tar_path)
        members = dict(audio)
        chosen = _by_digest(sorted(transcripts.keys() & members.keys()))[:count]
        paths = _extract(tar_path, {u: members[u] for u in chosen}, work)
        return [Item(u, {"audio": paths[u]}, transcripts[u]) for u in chosen]

    def metric_inputs(self, item: Item, output: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "reference": Transcript(segments=(TranscriptNode(text=item.reference),)),
            "hypothesis": output["transcript"],
        }


class LibriSpeechTrials:
    """LibriSpeech test-clean utterances of a few speakers, embedded; pairs within a split are the trials."""

    per_speaker = 5

    def items(self, files: Mapping[str, Path], count: int, work: Path) -> list[Item]:
        tar_path = _libri(files)
        _, audio = _read_tar(tar_path)
        members = dict(audio)
        by_speaker: dict[str, list[str]] = defaultdict(list)
        for uid in sorted(members):
            by_speaker[uid.split("-")[0]].append(uid)
        chosen: list[str] = []
        for speaker in _by_digest(list(by_speaker)):
            if len(chosen) >= count:
                break
            chosen += _by_digest(by_speaker[speaker])[: self.per_speaker]
        chosen = chosen[:count]
        paths = _extract(tar_path, {u: members[u] for u in chosen}, work)
        return [Item(u, {"audio": paths[u]}, u.split("-")[0]) for u in chosen]

    def scoring(self, items: Sequence[Item], outputs: Mapping[str, Mapping[str, Any]]) -> list[Scoring]:
        scored: list[Scoring] = []
        for split in SPLITS:
            ran = [i for i in items if i.id in outputs and split_of(i.id) == split]
            pairs = [(a, b) for n, a in enumerate(ran) for b in ran[n + 1 :]]
            if not pairs:
                continue
            first = [outputs[a.id]["embedding"] for a, _ in pairs]
            second = [outputs[b.id]["embedding"] for _, b in pairs]
            embedding: Embedding = first[0]
            trials = Trials(
                first=numpy.stack([e.vector for e in first]),
                second=numpy.stack([e.vector for e in second]),
                same=numpy.array([a.reference == b.reference for a, b in pairs]),
                space=embedding.space,
            )
            scored.append(Scoring(f"{POOLED}{split}", split, {"trials": trials}))
        return scored


def _parquet_rows(path: Path, columns: Sequence[str]) -> list[dict[str, Any]]:
    import importlib

    table = importlib.import_module("pyarrow.parquet").read_table(path, columns=list(columns))
    return table.to_pylist()  # type: ignore[no-any-return]


class VoxConverseDiarization(_PerItem):
    """VoxConverse test recordings of the first shard, diarized; the metric reads the reference and hypothesis turns."""

    def items(self, files: Mapping[str, Path], count: int, work: Path) -> list[Item]:
        [shard] = files.values()
        work.mkdir(parents=True, exist_ok=True)
        items = []
        for n, row in enumerate(_parquet_rows(shard, ["audio", "timestamps_start", "timestamps_end", "speakers"])):
            if n >= count:
                break
            uid = Path(row["audio"].get("path") or f"recording-{n:04d}").stem
            target = work / f"{uid}.wav"
            target.write_bytes(row["audio"]["bytes"])
            turns = list(zip(row["timestamps_start"], row["timestamps_end"], row["speakers"], strict=True))
            items.append(Item(uid, {"audio": target}, turns))
        return items

    def metric_inputs(self, item: Item, output: Mapping[str, Any]) -> Mapping[str, Any]:
        hypothesis: Intervals = output["turns"]
        reference = Intervals(
            start_ns=[round(s * 1e9) for s, _, _ in item.reference],
            end_ns=[round(e * 1e9) for _, e, _ in item.reference],
            label=numpy.array([lab for _, _, lab in item.reference]),
            clock_id=hypothesis.clock_id,
        )
        return {"reference": reference, "hypothesis": hypothesis}


class GretelPii(_PerItem):
    """Gretel synthetic finance documents of the test split; the metric reads reference and hypothesis spans."""

    def items(self, files: Mapping[str, Path], count: int, work: Path) -> list[Item]:  # noqa: ARG002 - the protocol
        [parquet] = files.values()
        rows = {
            f"{r['language']}-{r['index']}": r
            for r in _parquet_rows(parquet, ["index", "language", "generated_text", "pii_spans"])
        }
        items = []
        for uid in _by_digest(list(rows))[:count]:
            text = PlainText(content=rows[uid]["generated_text"])
            spans = [json.loads(s) if isinstance(s, str) else s for s in rows[uid]["pii_spans"]]
            spans = [s for s in spans if 0 <= s["start"] <= s["end"] <= len(text.content)]
            reference = TextSpans.over(
                text,
                start=numpy.array([s["start"] for s in spans], dtype=numpy.int64),
                end=numpy.array([s["end"] for s in spans], dtype=numpy.int64),
                label=numpy.array([s["label"] for s in spans], dtype=str),
            )
            items.append(Item(uid, {"text": text}, reference))
        return items

    def metric_inputs(self, item: Item, output: Mapping[str, Any]) -> Mapping[str, Any]:
        return {"reference": item.reference, "hypothesis": output["spans"]}


class SyntheticTones(_PerItem):
    """Tones at 44.1 kHz resampled to 16 kHz; the reference is the same tone synthesised at 16 kHz."""

    frequencies_hz = (220.0, 440.0, 880.0, 1760.0, 3520.0)
    source_rate = 44_100
    target_rate = 16_000

    def items(self, files: Mapping[str, Path], count: int, work: Path) -> list[Item]:  # noqa: ARG002 - no data
        tones = [self.frequencies_hz[n % len(self.frequencies_hz)] for n in range(count)]
        return [
            Item(f"tone-{n:02d}", {"series": tone(f, duration_s=1.0, sampling_rate=self.source_rate)}, f)
            for n, f in enumerate(tones)
        ]

    def metric_inputs(self, item: Item, output: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "reference": tone(item.reference, duration_s=1.0, sampling_rate=self.target_rate),
            "hypothesis": output["series"],
        }


ADAPTERS: Mapping[str, Callable[[], Adapter]] = {
    "librispeech_asr": LibriSpeechAsr,
    "librispeech_trials": LibriSpeechTrials,
    "voxconverse_diarization": VoxConverseDiarization,
    "gretel_pii": GretelPii,
    "synthetic_tones": SyntheticTones,
}
"""The adapter classes a suite may name."""


class SuiteError(ValueError):
    """A suite file is unreadable, or names an adapter or manifest that does not exist."""


@dataclass(frozen=True)
class Benchmark:
    """One suite entry; `manifest` is None for an adapter that makes its own data."""

    name: str
    adapter: str
    items: int
    task: str
    metric: str
    manifest: str | None = None
    model: str | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    metric_params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Suite:
    """A named list of benchmarks."""

    name: str
    benchmarks: tuple[Benchmark, ...]

    @classmethod
    def load(cls, path: Path) -> "Suite":
        """Read a `benchmark-suite@1.0` file.

        Raises:
            SuiteError: the file is not a suite of that format or an entry lacks a required key.
        """
        data = yaml.safe_load(path.read_text("utf-8"))
        if not isinstance(data, dict) or data.get("format") != SUITE_FORMAT or "benchmarks" not in data:
            raise SuiteError(f"{path.name} is not a {SUITE_FORMAT} suite: it needs `format` and `benchmarks`")
        try:
            return cls(str(data["name"]), tuple(Benchmark(**entry) for entry in data["benchmarks"]))
        except (KeyError, TypeError) as err:
            raise SuiteError(f"{path.name}: a benchmark entry is malformed: {err}") from None


class CallRecord(Frozen):
    """One `run_call` of a row: the task, its logical key digest, the context fingerprint, backend and output."""

    task: NonEmptyStr
    key: NonEmptyStr
    context: NonEmptyStr
    backend: NonEmptyStr
    output: NonEmptyStr
    reused: bool


class Row(Frozen):
    """One metric value over one item (or, for a pooled metric, one split) with the calls that made it."""

    label: NonEmptyStr
    split: NonEmptyStr
    value: float | None = None
    unit: str | None = None
    items: int | None = None
    fault: str | None = None
    calls: tuple[CallRecord, ...] = ()


class Aggregate(Frozen):
    """A split's mean of its rows' values, weighted by their `items`."""

    split: NonEmptyStr
    value: float | None
    items: int
    rows: int
    faults: int


class Entry(Frozen):
    """One benchmark's rows and aggregates, with the data it ran on."""

    benchmark: NonEmptyStr
    task: NonEmptyStr
    metric: NonEmptyStr
    manifest: str | None
    manifest_digest: str | None
    rows: tuple[Row, ...]
    aggregates: tuple[Aggregate, ...]


class BenchmarkResult(Formatted):
    """A suite run: when, under which execution profile, what each benchmark scored and every context it ran in."""

    format: FormatTag = BENCHMARK_RESULT_FORMAT
    suite: NonEmptyStr
    run_id: NonEmptyStr
    started: NonEmptyStr
    execution: NonEmptyStr
    execution_digest: NonEmptyStr
    entries: tuple[Entry, ...]
    contexts: tuple[ExecutionContext, ...]


def _record(task: str, result: Result) -> CallRecord:
    return CallRecord(
        task=task,
        key=result.key.digest(),
        context=result.context.fingerprint(),
        backend=result.placement.backend.name,
        output=result.output_identity,
        reused=result.reused is not None,
    )


def _aggregate(rows: Sequence[Row]) -> tuple[Aggregate, ...]:
    out = []
    for split in SPLITS:
        mine = [r for r in rows if r.split == split]
        scored = [r for r in mine if r.value is not None and r.items]
        weight = sum(r.items or 0 for r in scored)
        value = sum((r.value or 0.0) * (r.items or 0) for r in scored) / weight if weight else None
        out.append(
            Aggregate(
                split=split, value=value, items=weight, rows=len(mine), faults=sum(r.fault is not None for r in mine)
            )
        )
    return tuple(out)


def _run_benchmark(  # noqa: PLR0913 - the benchmark and everything it runs under
    bench: Benchmark,
    files: Mapping[str, Path],
    adapter: Adapter,
    execution: ExecutionProfile,
    store: ResultStore,
    work: Path,
    contexts: dict[str, ExecutionContext],
) -> tuple[Row, ...]:
    items = adapter.items(files, bench.items, work)
    records: dict[str, CallRecord] = {}
    outputs: dict[str, Mapping[str, Any]] = {}
    rows: list[Row] = []
    for item in items:
        call = TaskCall.model_validate(
            {"task": bench.task, "inputs": item.inputs, "params": bench.params, "model": bench.model}
        )
        try:
            result = run_call(call, execution=execution, store=store)
        except CallFailed as err:
            rows.append(Row(label=item.id, split=split_of(item.id), fault=str(err.error.get("fault"))))
            continue
        outputs[item.id] = result.outputs
        records[item.id] = _record(bench.task, result)
        contexts[records[item.id].context] = result.context
    for scoring in adapter.scoring(items, outputs):
        metric_call = TaskCall.model_validate(
            {"task": bench.metric, "inputs": scoring.inputs, "params": bench.metric_params}
        )
        scored = run_call(metric_call, execution=execution, store=store)
        metric: Metric = scored.outputs["metric"]  # type: ignore[assignment]
        contexts[scored.context.fingerprint()] = scored.context
        pooled = scoring.label.startswith(POOLED)
        own = [
            records[i.id]
            for i in items
            if i.id in records and (split_of(i.id) == scoring.split if pooled else i.id == scoring.label)
        ]
        rows.append(
            Row(
                label=scoring.label,
                split=scoring.split,
                value=metric.value,
                unit=metric.unit,
                items=metric.items,
                calls=(*own, _record(bench.metric, scored)),
            )
        )
    return tuple(rows)


def run_suite(  # noqa: PLR0913 - the suite and where everything it reads and writes lives
    suite: Suite,
    *,
    execution: ExecutionProfile,
    manifests: Path,
    data_cache: Path,
    out: Path,
    run_id: str,
    store: ResultStore | None = None,
    opener: Callable[[str], BinaryIO] | None = None,
    adapters: Mapping[str, Callable[[], Adapter]] = ADAPTERS,
) -> Path:
    """Run every benchmark of `suite` and write `<out>/<run_id>/result.json` and the run's PROV documents.

    Each manifest is fetched into `data_cache` (a present, hash-checked file is not fetched again). Returns the run
    directory.

    Raises:
        SuiteError: a benchmark names an adapter or a manifest that does not exist.
        As `fetch` and `run_call`, but a call the backend fails (`CallFailed`) is recorded as a row's fault.
    """
    run = out / run_id
    run.mkdir(parents=True, exist_ok=False)
    store = store if store is not None else ResultStore(out / "results")
    started = datetime.now(timezone.utc).isoformat()
    contexts: dict[str, ExecutionContext] = {}
    entries = []
    for bench in suite.benchmarks:
        if bench.adapter not in adapters:
            raise SuiteError(f"benchmark {bench.name!r} names adapter {bench.adapter!r}; known: {sorted(adapters)}")
        manifest, files = None, {}
        if bench.manifest is not None:
            path = manifests / f"{bench.manifest}.yaml"
            if not path.is_file():
                raise SuiteError(
                    f"benchmark {bench.name!r} names manifest {bench.manifest!r}, which {manifests.name}/ lacks"
                )
            manifest = DataManifest.load(path)
            files = fetch(manifest, cache_root=data_cache, opener=opener)
        rows = _run_benchmark(
            bench, files, adapters[bench.adapter](), execution, store, run / "work" / bench.name, contexts
        )
        entries.append(
            Entry(
                benchmark=bench.name,
                task=bench.task,
                metric=bench.metric,
                manifest=None if manifest is None else f"{manifest.name}@{manifest.version}",
                manifest_digest=None if manifest is None else manifest.digest(),
                rows=rows,
                aggregates=_aggregate(rows),
            )
        )
    document = BenchmarkResult(
        suite=suite.name,
        run_id=run_id,
        started=started,
        execution=execution.name,
        execution_digest=execution.canonical_digest(),
        entries=tuple(entries),
        contexts=tuple(contexts[k] for k in sorted(contexts)),
    )
    (run / "result.json").write_text(document.model_dump_json(indent=2), "utf-8")
    store.provenance.write(run)
    return run


def load_result(path: Path) -> BenchmarkResult:
    """Read a `result.json`; a tag of another major version is refused."""
    return BenchmarkResult.model_validate_json(path.read_text("utf-8"))


__all__ = ["ADAPTERS", "Benchmark", "BenchmarkResult", "Suite", "SuiteError", "load_result", "run_suite", "split_of"]
