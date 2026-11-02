"""The `pyannote` backend of `diarize`: its turns, faults, model and the isolated worker's contract."""

from typing import Any

import numpy
import pytest
from senselab._api.packaged import CATALOGUE
from senselab.audio.tasks.speaker_diarization import pyannote as pa
from senselab.audio.tasks.speaker_diarization import (
    pyannote_backend,  # noqa: F401 - declares the backend
)
from senselab_core.kernel.declare import REGISTRY
from senselab_core.kernel.results import Intervals
from senselab_core.kernel.signals import Audio
from senselab_core.kernel.wire import from_wire_form, wire_form
from senselab_core.runtime.hardware import HardwareInfo
from senselab_core.runtime.licenses import packaged_lock_records
from senselab_core.runtime.profiles import ExecutionProfile, resolve_execution
from senselab_core.testing.contracts import contract_cases
from senselab_core.testing.workers import RealWorkers, run_alone
from senselab_worker.faults import ContentFault, FaultClass, ResourceFault

LOCK_RECORDS = packaged_lock_records("senselab")

HOST = HardwareInfo(cpu_count=1, memory_bytes=1, gpus=(), apple_silicon=False, platform="linux-x86_64")
PARAMS: dict[str, Any] = {"num_speakers": None, "min_speakers": None, "max_speakers": None, "backend_params": {}}


def _tone(seconds: float = 1.0, rate: int = 16000, channels: int = 1, gain: float = 0.3) -> Audio:
    t = numpy.arange(max(1, round(seconds * rate))) / rate
    x = (gain * numpy.sin(2 * numpy.pi * 220 * t)).astype(numpy.float32)
    return Audio.from_array(numpy.tile(x, (channels, 1)), rate)


def _item(name: str, audio: Audio) -> dict[str, Any]:
    return {"id": name, "inputs": {"audio": wire_form(audio)}}


def _same(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Whether two results hold equal turns."""
    x, y = a["outputs"]["turns"], b["outputs"]["turns"]
    return x.keys() == y.keys() and all(numpy.array_equal(x[k], y[k]) for k in x)


def _run(items: list[dict[str, Any]], faults: dict[str, Any] | None = None, **params: Any) -> Any:  # noqa: ANN401
    state = pa.load({"stub": True, "device": "cpu", "faults": faults or {}})
    return pa.handle(state, items, {**PARAMS, **params})


@pytest.mark.contract
def test_turns_are_intervals_on_the_audios_clock() -> None:
    """The output round-trips as `intervals@1.0`, ordered, inside the audio, labelled by speaker."""
    audio = _tone(2.0)
    wired = _run([_item("a", audio)])[0]["outputs"]["turns"]
    out = from_wire_form(wired)
    assert isinstance(out, Intervals)
    assert out.clock_id == audio.anchor.clock_id and len(out.label) == 2
    assert list(out.start_ns) == sorted(out.start_ns) and int(out.end_ns.max()) <= 2_000_000_000


@pytest.mark.logic
def test_speaker_hint_sets_the_count_and_unset_hints_are_not_passed() -> None:
    """`num_speakers` reaches the model; silence-length audio gives an empty table, not a fault."""
    wired = _run([_item("a", _tone(3.0))], num_speakers=3)[0]["outputs"]["turns"]
    assert len(wired["label"]) == 3
    assert pa.intervals([], "c", 0)["label"].size == 0


@pytest.mark.logic
def test_turns_are_clamped_and_ordered() -> None:
    """Out-of-range and unordered turns become ordered integer ns inside `[0, duration]`."""
    wired = pa.intervals([("B", 0.5, 9.0), ("A", -1.0, 0.25)], "c", 1_000_000_000)
    assert list(wired["label"]) == ["A", "B"]
    assert list(wired["start_ns"]) == [0, 500_000_000] and list(wired["end_ns"]) == [250_000_000, 1_000_000_000]


@pytest.mark.logic
@pytest.mark.parametrize(
    ("audio", "text"), [(_tone(channels=2), "mono"), (_tone(rate=8000), "16000 Hz")], ids=["stereo", "rate"]
)
def test_unsupported_audio_is_a_content_fault(audio: Audio, text: str) -> None:
    """Stereo or a wrong rate fails that item as a typed content fault."""
    [result] = _run([_item("a", audio)])
    assert isinstance(result["error"], ContentFault) and text in str(result["error"])


@pytest.mark.logic
def test_faults_stay_with_their_item() -> None:
    """A content fault and an out of memory leave the third item unchanged, typed, never a gap."""
    items = [_item(i, _tone(1.0 + n / 10)) for n, i in enumerate("abc")]
    clean = _run(items)
    faulted = _run(items, faults={"fail_items": ["a"], "oom_items": ["b"]})
    assert isinstance(faulted[0]["error"], ContentFault) and isinstance(faulted[1]["error"], ResourceFault)
    assert _same(faulted[2], clean[2])


@pytest.mark.logic
def test_unknown_backend_param_is_refused() -> None:
    """A backend param the worker does not know is an error, not ignored."""
    with pytest.raises(ValueError, match="bogus"):
        _run([_item("a", _tone())], backend_params={"bogus": 1})


@pytest.mark.logic
def test_load_refuses_a_ref_an_unknown_key_and_both_stub_and_model() -> None:
    """A model loads by commit only; init takes its documented keys."""
    model = {"source": "hf", "id": pa.MODEL_ID, "identity": "main"}
    for init in ({"model": model}, {"stub": True, "bogus": 1}, {"stub": True, "model": model}, {}):
        with pytest.raises(ValueError):
            pa.load(init)


@pytest.mark.contract
def test_backend_serves_the_community_model_and_declares_its_isolated_env() -> None:
    """One isolated backend `pyannote` over the catalogue's default `diarize` model, placed by its worker sources."""
    (spec,) = REGISTRY.backends("diarize")
    assert spec.name == "pyannote" and spec.models == (pa.MODEL_ID,) and spec.batch_invariant
    assert spec.env.kind == "isolated" and spec.env.name == "pyannote" and spec.worker in spec.worker_sources
    entry = CATALOGUE.entry(pa.MODEL_ID)
    assert list(entry.backends) == ["pyannote"] and list(entry.stacks) == [spec.stack] and entry.license.gated
    placed = resolve_execution(REGISTRY.spec("diarize"), entry, ExecutionProfile.load(), HOST, REGISTRY, CATALOGUE)
    assert placed.backend.name == "pyannote"


@pytest.mark.contract
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(c, id=name)
        for name, c in contract_cases(REGISTRY, lock_records=LOCK_RECORDS)
        if name.startswith(("diarize::", "pyannote::"))
    ],
)
def test_contract(case: Any) -> None:  # noqa: ANN401 - a no-argument contract case
    """The task and backend satisfy the generated contract, the worker running on its load stub."""
    case()


@pytest.mark.env
@pytest.mark.model
def test_content_fault_isolated_with_real_weights(real_worker: RealWorkers) -> None:
    """On the pinned model, in its env, a content fault leaves batch-mates' turns unchanged."""
    items = [_item(i, _tone(2.0 + n)) for n, i in enumerate("abc")]
    clean = real_worker("diarize", "pyannote").run(items, PARAMS)
    faulted = real_worker("diarize", "pyannote", {"fail_items": ["b"]}).run(items, PARAMS)
    assert all("outputs" in r for r in clean) and faulted[1]["error"]["fault"] is FaultClass.CONTENT
    assert _same(faulted[0], clean[0]) and _same(faulted[2], clean[2])


@pytest.mark.limits
@pytest.mark.model
@pytest.mark.parametrize("exclusive", [True, False])
def test_edge_inputs_with_real_weights(real_worker: RealWorkers, exclusive: bool) -> None:
    """One sample, silence, clipping, long input and wrong rates each give turns or a typed fault."""
    worker = real_worker("diarize", "pyannote")
    params = {**PARAMS, "backend_params": {"exclusive": exclusive}}
    cases = {
        "one-sample": _tone(1 / 16000),
        "silence": _tone(2.0, gain=0.0),
        "clipping": _tone(2.0, gain=40.0),
        "long": _tone(70.0),
        **{f"rate-{r}": _tone(1.0, r) for r in (8000, 22050, 48000)},
    }
    outputs = run_alone(worker, {n: _item(n, a) for n, a in cases.items()}, params)
    assert all(isinstance(from_wire_form(o["turns"]), Intervals) for o in outputs.values())
