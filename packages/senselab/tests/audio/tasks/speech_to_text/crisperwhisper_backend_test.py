"""The `crisperwhisper` backend of `speech_to_text`: its engines, model, faults and the isolated worker's imports."""

import json
from typing import Any

import numpy
import pytest
from senselab._api.packaged import CATALOGUE
from senselab.audio.tasks.speech_to_text import crisperwhisper as cw
from senselab.audio.tasks.speech_to_text import (
    crisperwhisper_backend,  # noqa: F401 - declares the backend
)
from senselab_core.kernel.declare import REGISTRY
from senselab_core.kernel.signals import Audio
from senselab_core.kernel.wire import from_wire_form, wire_form
from senselab_core.runtime.hardware import HardwareInfo
from senselab_core.runtime.licenses import packaged_lock_records
from senselab_core.runtime.profiles import ExecutionProfile, NoBackend, resolve_execution
from senselab_core.testing.contracts import contract_cases
from senselab_core.testing.workers import RealWorkers, run_alone
from senselab_worker.faults import ContentFault, FaultClass, ResourceFault

LOCK_RECORDS = packaged_lock_records("senselab")

HOST = HardwareInfo(cpu_count=1, memory_bytes=1, gpus=(), apple_silicon=False, platform="linux-x86_64")
PARAMS: dict[str, Any] = {"language": None, "timestamps": "word", "backend_params": {}}
WHISPER = "openai/whisper-large-v3-turbo"


def _tone(seconds: float = 1.0, rate: int = 16000, channels: int = 1, gain: float = 0.3) -> Audio:
    t = numpy.arange(max(1, round(seconds * rate))) / rate
    x = (gain * numpy.sin(2 * numpy.pi * 220 * t)).astype(numpy.float32)
    return Audio.from_array(numpy.tile(x, (channels, 1)), rate)


def _item(name: str, audio: Audio) -> dict[str, Any]:
    return {"id": name, "inputs": {"audio": wire_form(audio)}}


def _run(items: list[dict[str, Any]], engine: str = "transformers", faults: dict[str, Any] | None = None) -> Any:  # noqa: ANN401
    state = cw.load({"stub": True, "device": "cpu"})
    params = {**PARAMS, "backend_params": {"engine": engine}, **({"_fault": faults} if faults else {})}
    return cw.handle(state, items, params)


@pytest.mark.contract
def test_two_engines_one_output_schema() -> None:
    """Both engines answer the same `transcript@2.0` shape on the audio's own clock."""
    audio = _tone()
    wires = {e: _run([_item("a", audio)], e)[0]["outputs"]["transcript"] for e in cw.ENGINES}
    for wired in wires.values():
        out: Any = from_wire_form(wired)
        assert json.loads(json.dumps(wire_form(out))) == wired
        assert out.segments[0].span.clock_id == audio.anchor.clock_id
        assert [w.text for w in out.segments[0].children] and all(
            type(w.span.start_ns) is int for w in out.segments[0].children
        )
    assert wires["ct2"].keys() == wires["transformers"].keys()


@pytest.mark.logic
def test_engine_param_selects_the_engine_that_decodes() -> None:
    """The engine named in backend params is the one asked, and only that one."""

    class Engine:
        def __init__(self, name: str, log: list[str]) -> None:
            self.name, self.log, self.stub = name, log, cw.StubModel()

        def transcribe(self, samples: numpy.ndarray, language: str, max_new_tokens: int) -> cw.Decoded:
            self.log.append(f"{self.name}:{language}:{max_new_tokens}")
            return self.stub.transcribe(samples, language, max_new_tokens)

    log: list[str] = []
    state = cw.State(None, "cpu", {e: Engine(e, log) for e in cw.ENGINES})
    params = {"language": "de", "timestamps": "none", "backend_params": {"engine": "ct2", "max_new_tokens": 9}}
    cw.handle(state, [_item("a", _tone())], params)
    assert log == ["ct2:de:9"]


@pytest.mark.logic
@pytest.mark.parametrize(
    "backend_params", [{"engine": "onnx"}, {"engine": "ct2", "beam": 5}], ids=["engine", "unknown-param"]
)
def test_unknown_engine_or_backend_param_is_refused(backend_params: dict[str, Any]) -> None:
    """An engine outside `ct2 | transformers` or an unknown backend param raises before any item runs."""
    state = cw.load({"stub": True})
    with pytest.raises(ValueError):
        cw.handle(state, [_item("a", _tone())], {**PARAMS, "backend_params": backend_params})


@pytest.mark.logic
def test_timestamp_levels() -> None:
    """`word` has children, `segment` one spanned segment, `none` one untimed segment."""
    decoded = cw.Decoded([("[UM]", 0.0, 0.4), ("hi", 0.4, 2.0)], [0.5, None], 0.9)
    word = cw.transcript(decoded, "word", "c", 1_000_000_000)["segments"][0]
    assert [c["text"] for c in word["children"]] == ["[UM]", "hi"] and word["children"][1]["span"]["end_ns"] == 10**9
    assert word["children"][0]["confidence"] == 0.5 and word["confidence"] == 0.9
    assert cw.transcript(decoded, "segment", "c", 10**9)["segments"][0]["children"] == []
    assert cw.transcript(decoded, "none", "c", 10**9)["segments"][0]["span"] is None
    assert cw.transcript(cw.Decoded([], [], None), "word", "c", 10**9)["segments"] == []


@pytest.mark.logic
@pytest.mark.parametrize(
    ("audio", "text"), [(_tone(channels=2), "mono"), (_tone(rate=8000), "16000 Hz")], ids=["stereo", "rate"]
)
def test_unsupported_audio_is_a_content_fault(audio: Audio, text: str) -> None:
    """Stereo or a wrong rate fails that item as a typed content fault."""
    [result] = _run([_item("a", audio)])
    assert isinstance(result["error"], ContentFault) and text in str(result["error"])


@pytest.mark.logic
def test_input_below_the_declared_minimum_is_a_content_fault() -> None:
    """One sample under the declared minimum fails as a content fault; the minimum itself transcribes."""
    n = cw.MIN_INPUT_NS * cw.EXPECTED_RATE // 1_000_000_000
    short, ok = _run([_item("s", _tone((n - 1) / 16000)), _item("o", _tone(n / 16000))])
    assert n == 400 and cw.MIN_INPUT_NS == 25_000_000 and isinstance(short["error"], ContentFault)
    assert "outputs" in ok


@pytest.mark.logic
def test_faults_stay_with_their_item() -> None:
    """A content fault and an out of memory leave the third item unchanged, typed, never a gap."""
    items = [_item(i, _tone(1.0 + n / 10)) for n, i in enumerate("abc")]
    clean = _run(items)
    faulted = _run(items, faults={"fail_items": ["a"], "oom_items": ["b"]})
    assert isinstance(faulted[0]["error"], ContentFault) and isinstance(faulted[1]["error"], ResourceFault)
    assert faulted[2] == clean[2]


@pytest.mark.logic
def test_load_refuses_a_ref_an_unknown_key_and_both_stub_and_model() -> None:
    """A model loads by commit only; init takes its documented keys."""
    model = {"source": "hf", "id": cw.MODEL_ID, "identity": "main"}
    for init in ({"model": model}, {"stub": True, "bogus": 1}, {"stub": True, "model": model}, {}):
        with pytest.raises(ValueError):
            cw.load(init)


@pytest.mark.contract
def test_backend_serves_only_the_crisperwhisper_model_and_declares_its_isolated_env() -> None:
    """One backend, isolated, over the one catalogue model that names it; Whisper models are not served by it."""
    (spec,) = [b for b in REGISTRY.backends("speech_to_text") if b.name == "crisperwhisper"]
    assert spec.models == (cw.MODEL_ID,) and spec.env.kind == "isolated" and spec.env.name == "crisperwhisper"
    assert spec.min_input_ns == cw.MIN_INPUT_NS
    assert spec.params.model_fields["engine"].default == cw.DEFAULT_ENGINE and spec.batch_invariant
    entry = CATALOGUE.entry(cw.MODEL_ID)
    assert list(entry.backends) == ["crisperwhisper"] and list(entry.stacks) == [spec.stack]
    profile = ExecutionProfile.load()
    placed = resolve_execution(REGISTRY.spec("speech_to_text"), entry, profile, HOST, REGISTRY, CATALOGUE)
    assert placed.backend.name == "crisperwhisper"
    assert "crisperwhisper" not in CATALOGUE.entry(WHISPER).backends
    whisper = resolve_execution(
        REGISTRY.spec("speech_to_text"), CATALOGUE.entry(WHISPER), profile, HOST, REGISTRY, CATALOGUE
    )
    assert whisper.backend.name == "transformers"
    only = profile.model_copy(update={"stack_preferences": {"speech_to_text": ("crisperwhisper",)}})
    with pytest.raises(NoBackend):
        resolve_execution(REGISTRY.spec("speech_to_text"), CATALOGUE.entry(WHISPER), only, HOST, REGISTRY, CATALOGUE)


@pytest.mark.contract
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(c, id=name)
        for name, c in contract_cases(REGISTRY, lock_records=LOCK_RECORDS)
        if name.startswith(("speech_to_text::", "crisperwhisper::"))
    ],
)
def test_contract(case: Any) -> None:  # noqa: ANN401 - a no-argument contract case
    """The task and backend satisfy the generated contract, the worker running on its load stub."""
    case()


@pytest.mark.env
@pytest.mark.model
@pytest.mark.parametrize("engine", cw.ENGINES)
def test_content_fault_isolated_with_real_weights(real_worker: RealWorkers, engine: str) -> None:
    """On the pinned model, in its env, each engine isolates a content fault: batch-mates' transcripts are unchanged."""
    items = [_item(i, _tone(1.0 + n)) for n, i in enumerate("abc")]
    params = {**PARAMS, "backend_params": {"engine": engine}}
    worker = real_worker("speech_to_text", "crisperwhisper")
    clean = worker.run(items, params)
    faulted = worker.run(items, {**params, "_fault": {"fail_items": ["b"]}})
    assert all("outputs" in r for r in clean)
    assert faulted[1]["error"]["fault"] is FaultClass.CONTENT
    assert (faulted[0], faulted[2]) == (clean[0], clean[2])


@pytest.mark.limits
@pytest.mark.model
@pytest.mark.parametrize("engine", cw.ENGINES)
def test_edge_inputs_with_real_weights(real_worker: RealWorkers, engine: str) -> None:
    """One sample, silence, clipping, long input and wrong rates each give a transcript or a typed fault."""
    worker = real_worker("speech_to_text", "crisperwhisper")
    params = {**PARAMS, "backend_params": {"engine": engine}}
    cases = {
        "one-sample": _tone(1 / 16000),
        "silence": _tone(2.0, gain=0.0),
        "clipping": _tone(2.0, gain=40.0),
        "long": _tone(70.0),
        **{f"rate-{r}": _tone(1.0, r) for r in (8000, 22050, 48000)},
    }
    outputs = run_alone(worker, {n: _item(n, a) for n, a in cases.items()}, params)
    assert all(from_wire_form(o["transcript"]) is not None for o in outputs.values())
