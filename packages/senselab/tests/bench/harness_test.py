"""The benchmark harness on the fake stack: splits, provenance of every call, the shipped manifests, the CLI."""

import hashlib
import importlib.util
import io
import json
import logging
import platform
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from senselab.cli import main
from senselab.metrics import wer  # noqa: F401 - declares the `wer` task and backend in REGISTRY
from senselab_core.kernel.declare import REGISTRY, BackendSpec, Env, Registry, TaskSpec
from senselab_core.kernel.results import Transcript, TranscriptNode
from senselab_core.kernel.wire import wire_form
from senselab_core.runtime import fake_backend as fake
from senselab_core.runtime.call import Packaged, use_packaged
from senselab_core.runtime.catalogue import Catalogue
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.fetch import DataManifest
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.provenance import ProvStore

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
    """`run_call` reads the toy registry (and `wer`) and opens every env as this interpreter."""
    previous = use_packaged(Packaged(Catalogue(entries=()), _this_interpreter, _registry()))
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
    use_packaged(Packaged(Catalogue(entries=()), _this_interpreter, REGISTRY))
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

    use_packaged(Packaged(Catalogue(entries=()), _this_interpreter, REGISTRY))
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
    path.write_text("devices: [cpu]\n")
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
    use_packaged(Packaged(Catalogue(entries=()), _this_interpreter, _gpu_only_registry()))
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
