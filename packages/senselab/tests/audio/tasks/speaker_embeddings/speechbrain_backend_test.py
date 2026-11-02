"""The `speechbrain` backend of `speaker_embeddings`: its vectors, faults, model and the isolated worker's contract."""

from types import SimpleNamespace
from typing import Any

import numpy
import pytest
from senselab._api.packaged import CATALOGUE, LOCK_RECORDS
from senselab.audio.tasks.speaker_embeddings import speechbrain as sb
from senselab.audio.tasks.speaker_embeddings import (
    speechbrain_backend,  # noqa: F401 - declares the backend
)
from senselab.audio.tasks.speaker_embeddings.windowing import embed_windows, slice_audio, window_starts
from senselab_core.kernel.declare import REGISTRY
from senselab_core.kernel.results import Embedding
from senselab_core.kernel.signals import Audio
from senselab_core.kernel.wire import from_wire_form, wire_form
from senselab_core.runtime.hardware import HardwareInfo
from senselab_core.runtime.profiles import ExecutionProfile, resolve_execution
from senselab_core.testing.contracts import contract_cases
from senselab_worker.faults import ContentFault, ResourceFault

HOST = HardwareInfo(cpu_count=1, memory_bytes=1, gpus=(), apple_silicon=False, platform="linux-x86_64")
PARAMS: dict[str, Any] = {"backend_params": {}}


def _tone(seconds: float = 1.0, rate: int = 16000, channels: int = 1, gain: float = 0.3, hz: float = 220) -> Audio:
    t = numpy.arange(max(1, round(seconds * rate))) / rate
    x = (gain * numpy.sin(2 * numpy.pi * hz * t)).astype(numpy.float32)
    return Audio.from_array(numpy.tile(x, (channels, 1)), rate)


def _item(name: str, audio: Audio) -> dict[str, Any]:
    return {"id": name, "inputs": {"audio": wire_form(audio)}}


def _run(items: list[dict[str, Any]], faults: dict[str, Any] | None = None, **params: Any) -> Any:  # noqa: ANN401
    state = sb.load({"stub": True, "device": "cpu", "faults": faults or {}})
    return sb.handle(state, items, {**PARAMS, **params})


def _vector(result: dict[str, Any]) -> numpy.ndarray:
    out = from_wire_form(result["outputs"]["embedding"])
    assert isinstance(out, Embedding)
    return out.vector


@pytest.mark.contract
def test_output_is_an_embedding_in_the_models_space() -> None:
    """The output round-trips as `embedding@1.0`, finite, in the space the model names."""
    out = from_wire_form(_run([_item("a", _tone())])[0]["outputs"]["embedding"])
    assert isinstance(out, Embedding) and out.space.endswith(sb.MODEL_ID) and out.vector.shape == (sb.STUB_DIMENSION,)


@pytest.mark.logic
def test_a_vector_does_not_depend_on_its_batch_mates() -> None:
    """Items of different durations, alone or together, give the same vectors: nothing is padded to a peer."""
    short, long = _item("s", _tone(0.5)), _item("l", _tone(3.0, hz=330))
    together = _run([short, long])
    assert numpy.array_equal(_vector(together[0]), _vector(_run([short])[0]))
    assert numpy.array_equal(_vector(together[1]), _vector(_run([long])[0]))
    assert not numpy.array_equal(_vector(together[0]), _vector(together[1]))


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
    assert numpy.array_equal(_vector(faulted[2]), _vector(clean[2]))


@pytest.mark.logic
def test_unknown_backend_param_is_refused() -> None:
    """The backend takes no params; one given is an error, not ignored."""
    with pytest.raises(ValueError, match="bogus"):
        _run([_item("a", _tone())], backend_params={"bogus": 1})


@pytest.mark.logic
def test_load_refuses_a_ref_an_unknown_key_and_both_stub_and_model() -> None:
    """A model loads by commit only; init takes its documented keys."""
    model = {"source": "hf", "id": sb.MODEL_ID, "identity": "main"}
    for init in ({"model": model}, {"stub": True, "bogus": 1}, {"stub": True, "model": model}, {}):
        with pytest.raises(ValueError):
            sb.load(init)


@pytest.mark.logic
def test_windows_cover_the_audio_and_end_on_its_tail() -> None:
    """Windows start every hop, the last pinned to the tail; short audio has one; bad sizes none."""
    assert window_starts(2.0, 1.0, 0.5) == [0.0, 0.5, 1.0]
    assert window_starts(2.2, 1.0, 0.5) == [0.0, 0.5, 1.0, pytest.approx(1.2)]
    assert window_starts(0.4, 1.0, 0.5) == [0.0] and window_starts(1.0, 0.0, 0.5) == []
    assert slice_audio(_tone(2.0), 0.5, 1.5).frames == 16000


@pytest.mark.logic
def test_embed_windows_makes_one_task_call_per_window_in_order() -> None:
    """A 2 s audio at window 1 s, hop 0.5 s gives three windows, each embedded by its own call on the stub worker."""
    calls: list[Audio] = []

    def task(audio: Audio) -> Any:  # noqa: ANN401 - a stand-in with a Result's `outputs`
        calls.append(audio)
        out = _run([_item("w", audio)])[0]["outputs"]
        return SimpleNamespace(outputs={"embedding": from_wire_form(out["embedding"])})

    windows = embed_windows(_tone(2.0), task, window_s=1.0, hop_s=0.5)
    assert [(w.start_s, w.end_s) for w in windows] == [(0.0, 1.0), (0.5, 1.5), (1.0, 2.0)] and len(calls) == 3
    assert all(c.frames == 16000 for c in calls) and all(w.embedding.space.endswith(sb.MODEL_ID) for w in windows)
    with pytest.raises(ValueError, match="shorter"):
        embed_windows(_tone(0.5), task)


@pytest.mark.contract
def test_backend_serves_the_ecapa_model_and_declares_its_isolated_env() -> None:
    """One isolated backend `speechbrain` over the catalogue's default `speaker_embeddings` model."""
    (spec,) = REGISTRY.backends("speaker_embeddings")
    assert spec.name == "speechbrain" and spec.models == (sb.MODEL_ID,) and spec.batch_invariant
    assert spec.env.kind == "isolated" and spec.env.name == "speechbrain" and spec.worker in spec.worker_sources
    entry = CATALOGUE.entry(sb.MODEL_ID)
    assert list(entry.backends) == ["speechbrain"] and list(entry.stacks) == [spec.stack]
    placed = resolve_execution(
        REGISTRY.spec("speaker_embeddings"), entry, ExecutionProfile.load(), HOST, REGISTRY, CATALOGUE
    )
    assert placed.backend.name == "speechbrain"


@pytest.mark.contract
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(c, id=name)
        for name, c in contract_cases(REGISTRY, lock_records=LOCK_RECORDS)
        if name.startswith(("speaker_embeddings::", "speechbrain::"))
    ],
)
def test_contract(case: Any) -> None:  # noqa: ANN401 - a no-argument contract case
    """The task and backend satisfy the generated contract, the worker running on its load stub."""
    case()


def _real_state(faults: dict[str, Any] | None = None) -> Any:  # noqa: ANN401
    import importlib

    torch = importlib.import_module("torch")
    entry = CATALOGUE.entry(sb.MODEL_ID)
    model = {"source": "hf", "id": entry.id, "identity": entry.revision}
    return sb.load({"model": model, "device": "cuda" if torch.cuda.is_available() else "cpu", "faults": faults or {}})


@pytest.mark.env
@pytest.mark.model
def test_content_fault_isolated_with_real_weights() -> None:
    """On the pinned model a content fault leaves batch-mates' vectors unchanged."""
    items = [_item(i, _tone(2.0 + n)) for n, i in enumerate("abc")]
    clean = sb.handle(_real_state(), items, PARAMS)
    faulted = sb.handle(_real_state({"fail_items": ["b"]}), items, PARAMS)
    assert all("outputs" in r for r in clean) and isinstance(faulted[1]["error"], ContentFault)
    assert numpy.array_equal(_vector(faulted[0]), _vector(clean[0]))
    assert numpy.array_equal(_vector(faulted[2]), _vector(clean[2]))


@pytest.mark.limits
@pytest.mark.model
def test_short_span_keeps_its_vector_inside_a_batch_with_long_recordings() -> None:
    """A 0.5 s span embedded alone and inside a batch with 8 s and 2 s recordings gives the identical vector."""
    state = _real_state()
    short = _item("s", _tone(0.5, hz=300))
    mixed = sb.handle(state, [short, _item("l", _tone(8.0)), _item("m", _tone(2.0, hz=500))], PARAMS)
    alone = sb.handle(state, [short], PARAMS)
    assert numpy.array_equal(_vector(mixed[0]), _vector(alone[0]))


@pytest.mark.limits
@pytest.mark.model
def test_edge_inputs_with_real_weights() -> None:
    """One sample, silence, clipping, long input and wrong rates each give a vector or a typed fault."""
    state = _real_state()
    cases = {
        "one-sample": _tone(1 / 16000),
        "silence": _tone(2.0, gain=0.0),
        "clipping": _tone(2.0, gain=40.0),
        "long": _tone(70.0),
        **{f"rate-{r}": _tone(1.0, r) for r in (8000, 22050, 48000)},
    }
    for name, audio in cases.items():
        [result] = sb.handle(state, [_item(name, audio)], PARAMS)
        if "error" in result:
            assert isinstance(result["error"], (ContentFault, ResourceFault)), (name, result["error"])
        else:
            assert isinstance(from_wire_form(result["outputs"]["embedding"]), Embedding)
