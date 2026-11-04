"""The `crisperwhisper` backend of `speech_to_text`: its engines, model, faults and the isolated worker's imports."""

import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy
import pytest
from senselab._api.packaged import CATALOGUE
from senselab.audio.tasks.speech_to_text import crisperwhisper as cw
from senselab.audio.tasks.speech_to_text import (
    crisperwhisper_backend,  # noqa: F401 - declares the backend
)
from senselab.audio.tasks.speech_to_text.crisperwhisper_backend import CrisperWhisperParams
from senselab.audio.tasks.speech_to_text.whisper_frontend import MIN_INPUT_NS
from senselab_core.kernel.declare import REGISTRY, BackendSpec, Env, TaskParams, TaskSpec
from senselab_core.kernel.identity import ExecutionContext
from senselab_core.kernel.results import Transcript
from senselab_core.kernel.signals import Audio
from senselab_core.kernel.wire import from_wire_form, wire_form
from senselab_core.runtime.fetch import DataManifest, default_cache_root, fetch
from senselab_core.runtime.hardware import HardwareInfo
from senselab_core.runtime.keys import Call, Placement, build_key
from senselab_core.runtime.licenses import packaged_lock_records
from senselab_core.runtime.profiles import ExecutionProfile, NoBackend, resolve_execution
from senselab_core.testing.contracts import BASE_CONTEXT, contract_cases
from senselab_core.testing.workers import RealWorkers, run_alone
from senselab_worker.fake import FAULT_KEY
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
    params = {**PARAMS, "backend_params": {"engine": engine}, **({FAULT_KEY: faults} if faults else {})}
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

        def transcribe(
            self, samples: numpy.ndarray, language: str, max_new_tokens: int, confidence: bool
        ) -> cw.Decoded:
            self.log.append(f"{self.name}:{language}:{max_new_tokens}:{confidence}")
            return self.stub.transcribe(samples, language, max_new_tokens, confidence)

    log: list[str] = []
    state = cw.State(None, "cpu", {e: Engine(e, log) for e in cw.ENGINES})
    params = {"language": "de", "timestamps": "none", "backend_params": {"engine": "ct2", "max_new_tokens": 9}}
    cw.handle(state, [_item("a", _tone())], params)
    assert log == ["ct2:de:9:True"]


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
    n = MIN_INPUT_NS * cw.EXPECTED_RATE // 1_000_000_000
    short, ok = _run([_item("s", _tone((n - 1) / 16000)), _item("o", _tone(n / 16000))])
    assert n == 400 and MIN_INPUT_NS == 25_000_000 and isinstance(short["error"], ContentFault)
    assert "outputs" in ok


@pytest.mark.logic
def test_diagnostics_are_an_output_only_when_requested() -> None:
    """`diagnostics` (windows, chunks, worker seconds) is added by the reserved request param and absent otherwise."""
    state = cw.load({"stub": True, "device": "cpu"})
    item = _item("a", _tone())
    [plain] = cw.handle(state, [item], PARAMS)
    [asked] = cw.handle(state, [item], {**PARAMS, cw.DIAGNOSTICS_KEY: True})
    assert set(plain["outputs"]) == {"transcript"}
    assert set(asked["outputs"]["diagnostics"]) == {
        "windows",
        "chunks",
        "decode_s",
        "confidence_s",
        "engine_build_s",
        "encode_reused",
    }
    assert asked["outputs"]["diagnostics"]["windows"] == 0 and asked["outputs"]["diagnostics"]["chunks"] is None


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
    assert spec.min_input_ns == MIN_INPUT_NS
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
    faulted = worker.run(items, {**params, FAULT_KEY: {"fail_items": ["b"]}})
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


MANIFEST = Path(__file__).parents[6] / "benchmarks" / "manifests" / "librispeech-test-clean.yaml"
BENCHMARK = MANIFEST.parents[1] / "crisperwhisper_confidence_cost.py"


def _long_speech() -> Audio:
    """Return the cost benchmark's long-form item, fetched through the data cache (`$SENSELAB_DATA_CACHE`)."""
    spec = importlib.util.spec_from_file_location("confidence_cost", BENCHMARK)
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    archive = fetch(DataManifest.load(MANIFEST), cache_root=default_cache_root())["test-clean.tar.gz"]
    return Audio.from_array(script.long_form(archive, script.LONG_FORM_SECONDS)[None, :], cw.EXPECTED_RATE)


@pytest.mark.env
@pytest.mark.model
@pytest.mark.parametrize("engine", cw.ENGINES)
def test_long_form_decode_is_scored_with_real_weights(real_worker: RealWorkers, engine: str) -> None:
    """A real item over 30 s has several windows, as many chunks, every confidence, and no second encoder pass."""
    params = {**PARAMS, "backend_params": {"engine": engine}, cw.DIAGNOSTICS_KEY: True}
    [result] = real_worker("speech_to_text", "crisperwhisper").run([_item("long", _long_speech())], params)
    diagnostics = result["outputs"]["diagnostics"]
    assert diagnostics["windows"] >= 2 and diagnostics["chunks"] == diagnostics["windows"], diagnostics
    assert 0 < diagnostics["encode_reused"] <= diagnostics["windows"], diagnostics
    transcript = from_wire_form(result["outputs"]["transcript"])
    assert isinstance(transcript, Transcript)
    [segment] = transcript.segments
    assert segment.confidence is not None and len(segment.children) > 10, diagnostics
    assert all(word.confidence is not None for word in segment.children), diagnostics


@pytest.mark.logic
def test_confidence_is_a_declared_backend_param_that_defaults_on() -> None:
    """`confidence` is in the param schema, default true, and the worker accepts it."""
    assert CrisperWhisperParams.model_fields["confidence"].default is True
    state = cw.load({"stub": True, "device": "cpu"})
    params = {**PARAMS, "backend_params": {"confidence": False}}
    assert "outputs" in cw.handle(state, [_item("a", _tone())], params)[0]


@pytest.mark.logic
def test_confidence_off_and_on_never_share_a_logical_key() -> None:
    """Everything else equal, `confidence` false and true build different keys."""
    task = TaskSpec(name="t", version="1", kind="transform", inputs={"audio": "Audio"}, outputs={}, params=TaskParams)
    backend = BackendSpec(
        task="t",
        name="crisperwhisper",
        version="1",
        stack="crisperwhisper",
        env=Env.host(),
        devices=("cpu",),
        batch_invariant=True,
        worker=cw.WORKER_MODULE,
        load_stub={},
        params=CrisperWhisperParams,
    )
    call = Call(task=task, params=TaskParams(), inputs={"audio": "content:" + "a" * 64})
    context = ExecutionContext.model_validate(BASE_CONTEXT)

    def digest(confidence: bool) -> str:
        placement = Placement(backend=backend, backend_params=CrisperWhisperParams(confidence=confidence), device="cpu")
        return build_key(call, placement, context).digest()

    assert digest(True) != digest(False)
