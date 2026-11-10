"""The benchmark harness on the fake stack: splits, provenance of every call, the shipped manifests, the CLI."""

import hashlib
import importlib.util
import io
import json
import logging
import platform
import sys
import tarfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import numpy
import pytest
from senselab.cli import main
from senselab.metrics import der, eer, span_f1, wer  # noqa: F401 - declares the `wer` task and backend in REGISTRY
from senselab_core.kernel.declare import REGISTRY, BackendSpec, Env, Registry, TaskSpec
from senselab_core.kernel.results import Embedding, Intervals, Metric, TextSpans, Transcript, TranscriptNode, Trials
from senselab_core.kernel.signals import PlainText
from senselab_core.kernel.wire import wire_form
from senselab_core.runtime import fake_backend as fake
from senselab_core.runtime.call import Packaged, use_packaged
from senselab_core.runtime.catalogue import Catalogue
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.fetch import DataManifest
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.provenance import ProvStore
from senselab_core.testing.workers import mit_lock_records

ROOT = Path(__file__).resolve().parents[4]
BENCHMARKS = ROOT / "benchmarks"


def _harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location("harness_under_test", BENCHMARKS / "harness.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


harness = _harness()

TOY_ROWS = "u1\tthe cat sat\nu2\ta dog ran\nu3\tone two three\nu4\tfour five\n"
HYPOTHESIS = Transcript(segments=(TranscriptNode(text="the cat sat"),))
TRANSCRIBE = TaskSpec(
    name="bench_test.transcribe",
    version="1",
    kind="transform",
    inputs={"text": "Text"},
    outputs={"transcript": "Transcript"},
    params=fake.EchoParams,
)


class ToyParams(fake.EchoParams):
    """Params of the toy transcribe task: the transcript the fake returns, as JSON."""

    outputs_json: str = ""


TRANSCRIBE = TRANSCRIBE.model_copy(update={"params": ToyParams})


def _registry() -> Registry:
    registry = Registry()
    registry.add_task(TRANSCRIBE)
    registry.add_backend(
        BackendSpec(
            task=TRANSCRIBE.name,
            name="fake-transcribe",
            version="1",
            stack="isolated",
            env=Env.isolated("fake"),
            devices=("cpu",),
            batch_invariant=True,
            worker=fake.WORKER_MODULE,
            load_stub={},
        )
    )
    registry.add_task(REGISTRY.spec("wer"))
    for backend in REGISTRY.backends("wer"):
        registry.add_backend(backend)
    return registry


def _this_interpreter(backend: BackendSpec, profile: ExecutionProfile) -> EnvHandle:  # noqa: ARG001 - an EnvOpener
    return EnvHandle(
        python=sys.executable,
        python_version=platform.python_version(),
        lock_sha="1" * 64,
        wheel_set="2" * 64,
        torch_index=None,
    )


@pytest.fixture(autouse=True)
def packaged() -> Iterator[None]:
    """The executor reads the toy registry (and `wer`) and opens every env as this interpreter."""
    previous = use_packaged(Packaged(Catalogue(entries=()), _this_interpreter, _registry(), mit_lock_records))
    yield
    use_packaged(previous)


class ToyAdapter:
    """Items are the lines of the one fetched file; the metric is the word error rate against the line."""

    def items(self, files: Mapping[str, Path], count: int, work: Path, seconds: float | None) -> list[Any]:  # noqa: ARG002, D102
        [path] = files.values()
        lines = [line.split("\t") for line in path.read_text("utf-8").splitlines()]
        return [harness.Item(uid, {"text": text}, text) for uid, text in lines[:count]]

    def scoring(self, items: Sequence[Any], outputs: Mapping[str, Mapping[str, Any]]) -> list[Any]:  # noqa: D102
        return [
            harness.Scoring(
                i.id,
                harness.split_of(i.id),
                {
                    "reference": Transcript(segments=(TranscriptNode(text=i.reference),)),
                    "hypothesis": outputs[i.id]["transcript"],
                },
            )
            for i in items
            if i.id in outputs
        ]


class EmptyReferenceAdapter(ToyAdapter):
    """As `ToyAdapter`, but item `u2` has no reference words, so its metric has nothing to score."""

    def scoring(self, items: Sequence[Any], outputs: Mapping[str, Mapping[str, Any]]) -> list[Any]:  # noqa: D102
        return [
            harness.Scoring(s.label, s.split, {**s.inputs, "reference": Transcript(segments=())})
            if s.label == "u2"
            else s
            for s in super().scoring(items, outputs)
        ]


def _toy_manifests(tmp_path: Path) -> Path:
    directory = tmp_path / "manifests"
    directory.mkdir()
    digest = hashlib.sha256(TOY_ROWS.encode()).hexdigest()
    (directory / "toy.yaml").write_text(
        f"""format: data-manifest@2.0
name: toy
version: "1"
license:
  expression: MIT
  commercial_use: "yes"
  copyleft: none
  attribution: false
  gated: false
  source: toy
  verified: {{date: "2026-11-01", how: manual}}
terms: toy
files:
  - {{path: rows.tsv, url: "https://example.invalid/rows.tsv", sha256: {digest}, size: {len(TOY_ROWS.encode())}}}
"""
    )
    return directory


def _toy_suite() -> Any:  # noqa: ANN401
    return harness.Suite(
        "toy",
        (
            harness.Benchmark(
                name="toy-asr",
                adapter="toy",
                items=4,
                task=TRANSCRIBE.name,
                metric="wer",
                manifest="toy",
                params={"outputs_json": json.dumps({"transcript": wire_form(HYPOTHESIS)})},
            ),
        ),
    )


def _run(tmp_path: Path, run_id: str = "r1") -> Path:
    run: Path = harness.run_suite(
        _toy_suite(),
        execution=ExecutionProfile.load("laptop", {"devices": ["cpu"]}),
        manifests=tmp_path / "manifests",
        data_cache=tmp_path / "data",
        out=tmp_path / "out",
        run_id=run_id,
        opener=lambda url: io.BytesIO(TOY_ROWS.encode()),  # noqa: ARG005
        adapters={"toy": ToyAdapter},
    )
    return run


@pytest.mark.logic
def test_harness_records_provenance(tmp_path: Path) -> None:
    """Every call of every row is an activity of the run's PROV document, and no path reaches the result."""
    _toy_manifests(tmp_path)
    run = _run(tmp_path)
    result = harness.load_result(run / "result.json")
    [entry] = result.entries
    assert entry.manifest == "toy@1"
    assert entry.manifest_digest == DataManifest.load(tmp_path / "manifests" / "toy.yaml").digest()
    assert [r.label for r in entry.rows] == ["u1", "u2", "u3", "u4"]
    assert all(r.fault is None and r.value is not None for r in entry.rows)
    assert {r.label: r.value for r in entry.rows}["u1"] == 0.0
    calls = [c for r in entry.rows for c in r.calls]
    assert [c.task for c in calls[:2]] == ["bench_test.transcribe", "wer"]
    assert {c.context for c in calls} == {c.fingerprint() for c in result.contexts}
    activities = ProvStore.read(run).activities()
    assert (
        len(activities) == len({(c.task, c.key, c.context) for c in calls}) == len(calls) - sum(c.reused for c in calls)
    )
    assert sum(a.value * a.items for a in entry.aggregates if a.value is not None) > 0
    assert sum(a.rows for a in entry.aggregates) == 4
    assert str(tmp_path) not in (run / "result.json").read_text("utf-8")
    assert result.execution == "laptop"


@pytest.mark.logic
def test_a_metric_fault_is_that_rows_fault_and_the_suite_completes(tmp_path: Path) -> None:
    """An item the metric cannot score is a fault row; the rows before and after it are kept."""
    _toy_manifests(tmp_path)
    run = harness.run_suite(
        _toy_suite(),
        execution=ExecutionProfile.load("laptop", {"devices": ["cpu"]}),
        manifests=tmp_path / "manifests",
        data_cache=tmp_path / "data",
        out=tmp_path / "out",
        run_id="r1",
        opener=lambda url: io.BytesIO(TOY_ROWS.encode()),  # noqa: ARG005
        adapters={"toy": EmptyReferenceAdapter},
    )
    [entry] = harness.load_result(run / "result.json").entries
    assert {r.label: r.fault is not None for r in entry.rows} == {"u1": False, "u2": True, "u3": False, "u4": False}
    assert "no words" in next(r.fault for r in entry.rows if r.label == "u2")


@pytest.mark.logic
def test_manifest_split_is_deterministic() -> None:
    """A split is a function of the item id alone; both splits occur; the shipped digest rule is pinned."""
    ids = [f"item-{n}" for n in range(200)]
    forward = [harness.split_of(i) for i in ids]
    assert forward == [harness.split_of(i) for i in reversed(ids[::-1])]
    assert set(forward) == {"calibration", "heldout"}
    assert 60 < forward.count("calibration") < 140
    assert [harness.split_of("u1"), harness.split_of("u2")] == [
        harness.SPLITS[hashlib.sha256(b"u1").digest()[0] % 2],
        harness.SPLITS[hashlib.sha256(b"u2").digest()[0] % 2],
    ]


@pytest.mark.logic
def test_shipped_suite_resolves_to_pinned_manifests() -> None:
    """Every manifest the smoke suite names loads with a pinned sha256 per file and an https URL."""
    suite = harness.Suite.load(BENCHMARKS / "suites" / "smoke.yaml")
    assert {b.adapter for b in suite.benchmarks} <= set(harness.ADAPTERS)
    names = {b.manifest for b in suite.benchmarks if b.manifest}
    assert names == {"librispeech-test-clean", "voxconverse-test-mirror", "gretel-synthetic-pii"}
    for name in names:
        manifest = DataManifest.load(BENCHMARKS / "manifests" / f"{name}.yaml")
        assert manifest.name == name and manifest.license.expression and manifest.terms
        assert all(f.url.startswith("https://") and f.size > 0 for f in manifest.files)


@pytest.mark.logic
def test_bench_cli_runs_toy_suite(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`senselab bench run` over a suite of synthetic tones writes a result of resampling errors."""
    use_packaged(Packaged(Catalogue(entries=()), _this_interpreter, REGISTRY, mit_lock_records))
    suite = tmp_path / "tones.yaml"
    suite.write_text(
        "format: benchmark-suite@1.0\nname: tones\nbenchmarks:\n"
        "  - {name: tones, adapter: synthetic_tones, items: 2, task: signal.resample,"
        " params: {target_rate: 16000}, metric: resample_error}\n"
    )
    code = main(
        [
            "bench",
            "run",
            str(suite),
            "--benchmarks",
            str(BENCHMARKS),
            "--out",
            str(tmp_path / "out"),
            "--execution",
            "laptop",
        ]
    )
    assert code == 0
    result = harness.load_result(Path(capsys.readouterr().out.strip()))
    [entry] = result.entries
    assert [r.label for r in entry.rows] == ["tone-00", "tone-01"]
    assert all(r.value is not None and r.value < 0.05 and r.unit for r in entry.rows)


@pytest.mark.logic
def test_bench_cli_names_each_not_placeable_benchmark_on_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A skipped benchmark is one stderr line naming it and the reason; the exit code stays 0."""
    from senselab.text.tasks.pii_spans import transformers_backend  # noqa: F401, PLC0415 - declares the backend

    use_packaged(Packaged(Catalogue(entries=()), _this_interpreter, REGISTRY, mit_lock_records))
    suite = tmp_path / "gpu.yaml"
    suite.write_text(
        "format: benchmark-suite@1.0\nname: gpu\nbenchmarks:\n"
        "  - {name: gretel-pii, adapter: gretel_pii, items: 2, task: pii_spans, metric: span_f1}\n"
    )
    args = ["bench", "run", str(suite), "--benchmarks", str(BENCHMARKS), "--out", str(tmp_path / "out")]
    code = main([*args, "--execution", str(_cpu_profile(tmp_path))])
    captured = capsys.readouterr()
    assert code == 0
    [line] = captured.err.strip().splitlines()
    assert line.startswith("senselab: not placeable: gretel-pii: ") and "cuda" in line


def _cpu_profile(tmp_path: Path) -> Path:
    path = tmp_path / "cpu.yaml"
    path.write_text(ExecutionProfile.load("laptop", {"devices": ["cpu"]}).model_dump_json())
    return path


@pytest.mark.logic
def test_bench_cli_reports_unknown_suite(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A suite file that is not a `benchmark-suite@1.0` is one error line and exit code 1."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("format: nope\n")
    assert main(["bench", "run", str(bad), "--benchmarks", str(BENCHMARKS), "--out", str(tmp_path / "o")]) == 1
    assert "benchmark-suite@1.0" in capsys.readouterr().err


@pytest.mark.logic
def test_fault_text_names_type_and_no_path() -> None:
    """A fault row carries the error type and fault class; a path in the message is replaced."""
    text = harness._fault_text(
        {"type": "ValueError", "fault": "content", "message": "bad file /tmp/x/a.wav at C:\\d\\a"}
    )
    assert text == "ValueError (content): bad file <path> at <path>"


RECORDED_PII_SPANS = (
    '[{"start": 21, "end": 66, "label": "company"}, {"start": 83, "end": 93, "label": "date"}]',
    "[]",
)


@pytest.mark.logic
def test_gretel_spans_parse_the_recorded_shape() -> None:
    """The dataset stores `pii_spans` as one JSON array text (recorded rows); elements as texts or dicts also read."""
    assert harness.pii_spans(RECORDED_PII_SPANS[0]) == [
        {"start": 21, "end": 66, "label": "company"},
        {"start": 83, "end": 93, "label": "date"},
    ]
    assert harness.pii_spans(RECORDED_PII_SPANS[1]) == []
    assert (
        harness.pii_spans(['{"start": 1, "end": 2, "label": "x"}', {"start": 3, "end": 4, "label": "y"}])[1]["start"]
        == 3
    )


@pytest.mark.logic
def test_diarization_clip_cuts_audio_and_turns() -> None:
    """A clip keeps the first seconds of audio and ends or drops the turns past them."""
    import numpy
    import soundfile

    buffer = io.BytesIO()
    soundfile.write(buffer, numpy.zeros(16000 * 5, numpy.float32), 16000, format="WAV")
    clipped = harness._cut_wav(buffer.getvalue(), 2.0)
    assert soundfile.info(io.BytesIO(clipped)).frames == 32000
    assert harness._cut_wav(buffer.getvalue(), None) == buffer.getvalue()
    turns = [(0.0, 1.0, "a"), (1.5, 3.0, "b"), (2.5, 4.0, "c")]
    assert harness._clip_turns(turns, 2.0) == [(0.0, 1.0, "a"), (1.5, 2.0, "b")]


@pytest.mark.logic
def test_repeated_warning_prints_once() -> None:
    """The log filter lets each distinct message through once."""
    once = harness._Once()
    records = [logging.LogRecord("t", logging.WARNING, "f", 1, m, None, None) for m in ("a", "a", "b", "a")]
    assert [once.filter(r) for r in records] == [True, False, True, False]


def _gpu_only_registry() -> Registry:
    registry = _registry()
    gpu_task = TRANSCRIBE.model_copy(update={"name": "bench_test.gpu_transcribe", "inputs": {"prompt": "Text"}})
    registry.add_task(gpu_task)
    registry.add_backend(
        BackendSpec(
            task=gpu_task.name,
            name="fake-gpu",
            version="1",
            stack="isolated",
            env=Env.isolated("fake"),
            devices=("cuda",),
            batch_invariant=True,
            worker=fake.WORKER_MODULE,
            load_stub={},
        )
    )
    return registry


@pytest.mark.logic
def test_cuda_only_backend_on_cpu_host_is_not_placeable_and_the_rest_runs(tmp_path: Path) -> None:
    """A benchmark whose backend declares no device the profile and host share is reported, never started."""
    _toy_manifests(tmp_path)
    use_packaged(Packaged(Catalogue(entries=()), _this_interpreter, _gpu_only_registry(), mit_lock_records))
    fetched: list[str] = []

    def opener(url: str) -> io.BytesIO:
        fetched.append(url)
        return io.BytesIO(TOY_ROWS.encode())

    gpu = harness.Benchmark(
        name="gpu-asr", adapter="toy", items=4, task="bench_test.gpu_transcribe", metric="wer", manifest="toy"
    )
    suite = harness.Suite("mixed", (gpu, *_toy_suite().benchmarks))
    run = harness.run_suite(
        suite,
        execution=ExecutionProfile.load("laptop", {"devices": ["cpu"]}),
        manifests=tmp_path / "manifests",
        data_cache=tmp_path / "data",
        out=tmp_path / "out",
        run_id="r1",
        opener=opener,
        adapters={"toy": ToyAdapter},
    )
    skipped, ran = harness.load_result(run / "result.json").entries
    assert skipped.benchmark == "gpu-asr" and skipped.rows == ()
    assert skipped.not_placeable is not None and "fake-gpu" in skipped.not_placeable and "cuda" in skipped.not_placeable
    assert ran.not_placeable is None and len(ran.rows) == 4 and all(r.fault is None for r in ran.rows)
    assert len(fetched) == 1


@pytest.mark.logic
def test_pii_spans_transformers_backend_is_cuda_only() -> None:
    """The 31B catalogue model of `pii_spans` is declared runnable on cuda alone."""
    from senselab.text.tasks.pii_spans import transformers_backend  # noqa: F401, PLC0415 - declares the backend

    [spec] = [b for b in REGISTRY.backends("pii_spans") if b.name == "transformers"]
    assert spec.devices == ("cuda",)


def _row(label: str, split: str, metric: Metric) -> Any:  # noqa: ANN401
    return harness.Row(
        label=label,
        split=split,
        value=metric.value,
        unit=metric.unit,
        items=metric.items,
        numerator=metric.numerator,
        denominator=metric.denominator,
        empty_value=metric.empty_value,
    )


def _turns(*rows: tuple[str, float, float]) -> Intervals:
    return Intervals(
        start_ns=[round(s * 1e9) for _, s, _ in rows],
        end_ns=[round(e * 1e9) for _, _, e in rows],
        label=numpy.array([lab for lab, _, _ in rows]),
        clock_id="c",
    )


@pytest.mark.logic
def test_der_aggregate_is_ratio_of_sums() -> None:
    """A split's DER is its summed error seconds over its summed reference seconds, not a mean weighted by turns."""
    params = der.DerParams()
    short = der.score({"reference": _turns(("A", 0, 1)), "hypothesis": _turns(("x", 5, 6))}, params)
    long = der.score(
        {"reference": _turns(("A", 0, 50), ("B", 50, 99)), "hypothesis": _turns(("x", 0, 50), ("y", 50, 99))}, params
    )
    rows = (_row("short", "calibration", short), _row("long", "calibration", long))
    [calibration, heldout] = harness._aggregate(rows)
    assert calibration.value == pytest.approx(2.0 / 100.0)
    assert (heldout.value, heldout.rows) == (None, 0)


def _cosine_trials(*pairs: tuple[float, bool]) -> Trials:
    return Trials(
        first=numpy.array([[1.0, 0.0]] * len(pairs)),
        second=numpy.array([[c, numpy.sqrt(1.0 - c * c)] for c, _ in pairs]),
        same=numpy.array([same for _, same in pairs]),
        space="m",
    )


@pytest.mark.logic
def test_eer_aggregates_pooled_trials() -> None:
    """EER carries no parts; a split's EER is the one of all its trials pooled, not the mean of per-group EERs."""
    separated = _cosine_trials((0.9, True), (0.8, True), (0.1, False), (0.2, False))
    inverted = _cosine_trials((0.2, True), (0.7, False))
    pooled = _cosine_trials((0.9, True), (0.8, True), (0.1, False), (0.2, False), (0.2, True), (0.7, False))
    per_group = [eer.score({"trials": t}, None) for t in (separated, inverted)]
    metric = eer.score({"trials": pooled}, None)
    assert [m.value for m in per_group] == [0.0, 1.0] and metric.value == pytest.approx(1 / 3)
    assert all((m.numerator, m.denominator) == (None, None) for m in (*per_group, metric))
    [calibration, heldout] = harness._aggregate((_row("pooled:calibration", "calibration", metric),))
    assert (calibration.value, calibration.items, heldout.rows) == (metric.value, metric.items, 0)


@pytest.mark.logic
def test_empty_span_f1_documents_add_nothing_to_the_pooled_ratio() -> None:
    """Ten documents with no entities and one with two missed spans pool to 0, and nothing to score pools to 1."""
    nothing = span_f1.score({"reference": _spans(), "hypothesis": _spans()}, None)
    missed = span_f1.score({"reference": _spans((0, 3, "p"), (4, 8, "p")), "hypothesis": _spans()}, None)
    assert (nothing.value, nothing.numerator, nothing.denominator) == (1.0, 0.0, 0.0)
    rows = tuple(_row(f"d{n}", "calibration", nothing) for n in range(10)) + (_row("m", "calibration", missed),)
    assert harness._aggregate(rows)[0].value == 0.0
    assert harness._aggregate(rows[:10])[0].value == 1.0


@pytest.mark.logic
@pytest.mark.parametrize("name", ["wer", "der", "eer"])
def test_a_split_with_nothing_to_score_is_undefined_unless_the_metric_declares_a_value(name: str) -> None:
    """Summed denominator 0 aggregates to None for a metric without an empty value; span F1 declares 1.0."""
    ratio = name != "eer"
    empty = Metric(
        name=name, value=0.0, unit="ratio", items=1, **({"numerator": 0.0, "denominator": 0.0} if ratio else {})
    )
    [calibration, _] = harness._aggregate((_row("a", "calibration", empty), _row("b", "calibration", empty)))
    assert (calibration.value, calibration.rows) == (None if ratio else 0.0, 2)
    declared = span_f1.score({"reference": _spans(), "hypothesis": _spans()}, None)
    assert harness._aggregate((_row("a", "calibration", declared),))[0].value == span_f1.EMPTY_VALUE == 1.0


@pytest.mark.logic
def test_a_split_mixing_ratio_and_plain_rows_is_refused() -> None:
    """Rows of one split either all carry ratio parts or none does."""
    ratio = Metric(name="wer", value=0.5, unit="ratio", items=2, numerator=1, denominator=2)
    plain = Metric(name="eer", value=0.5, unit="ratio", items=2)
    with pytest.raises(harness.MixedRatios, match="mix ratio parts"):
        harness._aggregate((_row("a", "calibration", ratio), _row("b", "calibration", plain)))


def _spans(*rows: tuple[int, int, str]) -> TextSpans:
    text = PlainText(content="Ada Lovelace met Charles Babbage")
    return TextSpans.over(
        text,
        start=numpy.array([s for s, _, _ in rows], dtype=numpy.int64),
        end=numpy.array([e for _, e, _ in rows], dtype=numpy.int64),
        label=numpy.array([lab for _, _, lab in rows], dtype=str),
    )


def _speaker_tar(path: Path, speakers: int, per_speaker: int) -> Path:
    with tarfile.open(path, "w:gz") as tar:
        for speaker in range(speakers):
            for n in range(per_speaker):
                data = b"flac"
                member = tarfile.TarInfo(f"LibriSpeech/test-clean/{speaker}/1/{speaker}-1-{n:04d}.flac")
                member.size = len(data)
                tar.addfile(member, io.BytesIO(data))
    return path


@pytest.mark.logic
def test_trials_split_by_speaker(tmp_path: Path) -> None:
    """The trials adapter's split unit is the speaker: one speaker's utterances share a split, and so do its trials."""
    adapter = harness.LibriSpeechTrials()
    assert adapter.split_unit == "speaker" and harness.VoxConverseDiarization().split_unit == "recording"
    tar_path = _speaker_tar(tmp_path / "a.tar.gz", speakers=12, per_speaker=5)
    items = adapter.items({"tar": tar_path}, 60, tmp_path / "work", None)
    by_speaker: dict[str, set[str]] = {}
    for item in items:
        by_speaker.setdefault(item.reference, set()).add(item.split)
        assert item.split == harness.split_of(item.reference)
    assert len(by_speaker) == 12 and all(len(splits) == 1 for splits in by_speaker.values())
    assert any(len({harness.split_of(i.id) for i in items if i.reference == sp}) == 2 for sp in by_speaker)
    outputs = {
        i.id: {"embedding": Embedding(vector=numpy.array([1.0, float(n)]), space="m")} for n, i in enumerate(items)
    }
    scored = adapter.scoring(items, outputs)
    assert {s.label for s in scored} == {"pooled:calibration", "pooled:heldout"}
    for scoring in scored:
        assert len(scoring.inputs["trials"].same) == sum(
            1 for a in items for b in items if a.id < b.id and a.split == b.split == scoring.split
        )


@pytest.mark.logic
def test_malformed_reference_is_item_fault(tmp_path: Path) -> None:
    """A reference span outside the text makes its item a counted fault row; the item is not run."""
    text = "Ada met Babbage"
    sound = harness.gretel_item("ok", text, '[{"start": 0, "end": 3, "label": "name"}]')
    assert sound.fault is None and sound.reference is not None
    for spans in (
        '[{"start": 0, "end": 99, "label": "n"}]',
        '[{"start": 5, "end": 2, "label": "n"}]',
        '[{"start": -1, "end": 2, "label": "n"}]',
    ):
        item = harness.gretel_item("bad", text, spans)
        assert item.fault == "ContentFault (content): 1 of 1 reference spans lie outside the text of 15 chars"

    unreadable = [
        '[{"start": 0, "label": "n"}]',
        '[{"start": "0", "end": 2, "label": "n"}]',
        "not json",
        '[{"start": 0.5, "end": 2, "label": "n"}]',
        "[3]",
    ]
    for spans in unreadable:
        item = harness.gretel_item("odd", text, spans)
        assert item.fault is not None and item.fault.startswith(
            "ContentFault (content): the reference spans are unreadable"
        )

    class Malformed:
        split_unit: Literal["item"] = "item"

        def items(self, files: Mapping[str, Path], count: int, work: Path, seconds: float | None) -> list[Any]:  # noqa: ARG002
            return [harness.gretel_item(f"d{n}", text, '[{"start": 0, "end": 99, "label": "n"}]') for n in range(count)]

        def scoring(self, items: Sequence[Any], outputs: Mapping[str, Mapping[str, Any]]) -> list[Any]:  # noqa: ARG002
            return []

    suite = harness.Suite("m", (harness.Benchmark(name="m", adapter="m", items=3, task=TRANSCRIBE.name, metric="wer"),))
    run = harness.run_suite(
        suite,
        execution=ExecutionProfile.load("laptop", {"devices": ["cpu"]}),
        manifests=tmp_path,
        data_cache=tmp_path / "data",
        out=tmp_path / "out",
        run_id="r1",
        adapters={"m": Malformed},
    )
    [entry] = harness.load_result(run / "result.json").entries
    assert [(r.label, r.value) for r in entry.rows] == [("d0", None), ("d1", None), ("d2", None)]
    assert all("1 of 1 reference spans" in (r.fault or "") for r in entry.rows)
    assert sum(a.faults for a in entry.aggregates) == 3


@pytest.mark.logic
def test_benchmark_result1_is_refused(tmp_path: Path) -> None:
    """A `benchmark-result@1` document is refused as retired; the current format is the one written."""
    old = tmp_path / "result.json"
    old.write_text(json.dumps({"format": "benchmark-result@1.0", "suite": "s"}), "utf-8")
    with pytest.raises(harness.BenchmarkResultRetired, match="benchmark-result@2.0"):
        harness.load_result(old)
    assert harness.BenchmarkResult.model_fields["format"].default == "benchmark-result@2.0"
