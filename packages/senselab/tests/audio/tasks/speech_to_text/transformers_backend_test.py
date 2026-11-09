"""The `transformers` backend of `speech_to_text`: transcript construction, typed faults and batch isolation."""

import importlib
from typing import Any

import numpy
import pytest
from senselab.audio.tasks.speech_to_text import huggingface as hf
from senselab.audio.tasks.speech_to_text.token_confidence import (
    capture_token_confidence,
    merge_confidence_blocks,
    word_confidences,
)
from senselab.audio.tasks.speech_to_text.whisper_frontend import MIN_INPUT_NS
from senselab_core.kernel.declare import REGISTRY
from senselab_core.kernel.signals import Audio
from senselab_core.kernel.wire import from_wire_form, wire_form
from senselab_core.runtime.licenses import packaged_lock_records
from senselab_core.testing.contracts import contract_cases
from senselab_worker.fake import FAULT_KEY
from senselab_worker.faults import ContentFault, ResourceFault, error_record
from senselab_worker.serve import call_handle

LOCK_RECORDS = packaged_lock_records("senselab")

CLOCK = "content:" + "a" * 64
PIPE_OUT = {
    "text": " hello there.",
    "chunks": [{"text": " hello", "timestamp": (0.0, 0.5)}, {"text": " there.", "timestamp": (0.5, None)}],
}
PARAMS: dict[str, Any] = {"language": None, "timestamps": "word", "backend_params": {}}


def _bounds(node: Any) -> tuple[int, int, str]:  # noqa: ANN401 - a TranscriptNode
    """Return the node's span as `(start_ns, end_ns, clock_id)`, integers."""
    assert node.span is not None
    assert type(node.span.start_ns) is int and type(node.span.end_ns) is int
    return node.span.start_ns, node.span.end_ns, node.span.clock_id


def _audio(samples: int = 1600, rate: int = 16000, channels: int = 1) -> Audio:
    x = numpy.sin(numpy.arange(samples) / 7.0).astype(numpy.float32)
    return Audio.from_array(numpy.tile(x, (channels, 1)), rate)


def _run(items: list[dict[str, Any]], params: dict[str, Any] = PARAMS, faults: dict[str, Any] | None = None) -> Any:  # noqa: ANN401
    state = hf.load({"stub": True, "device": "cpu"})
    return hf.handle(state, items, {**params, **({FAULT_KEY: faults} if faults else {})})


@pytest.mark.logic
def test_word_transcript_has_integer_ns_spans_and_children() -> None:
    """A word transcript is one segment over its words, spans in integer ns on the audio's clock."""
    t = hf.transcript(PIPE_OUT, "word", CLOCK, 1_000_000_000)
    (segment,) = t.segments
    assert segment.text == "hello there." and [w.text for w in segment.children] == ["hello", "there."]
    assert _bounds(segment) == (0, 1_000_000_000, CLOCK)
    assert [_bounds(w)[:2] for w in segment.children] == [(0, 500_000_000), (500_000_000, 1_000_000_000)]
    assert all(_bounds(w)[2] == CLOCK for w in segment.children)


@pytest.mark.logic
def test_segment_and_none_levels() -> None:
    """`segment` keeps timed chunks without children; `none` is one untimed segment; empty text has no segment."""
    seg = hf.transcript(PIPE_OUT, "segment", CLOCK, 10**9)
    assert [s.text for s in seg.segments] == ["hello", "there."] and all(not s.children for s in seg.segments)
    (plain,) = hf.transcript(PIPE_OUT, "none", CLOCK, 10**9).segments
    assert plain.span is None and plain.text == "hello there."
    assert hf.transcript({"text": "  ", "chunks": []}, "word", CLOCK, 10**9).segments == ()


@pytest.mark.logic
def test_negative_and_inverted_timestamps_are_clamped() -> None:
    """A tiny negative start is 0 and an end before its start is its start."""
    out = {"text": "a", "chunks": [{"text": "a", "timestamp": (-0.02, -0.5)}]}
    (word,) = hf.transcript(out, "word", CLOCK, 10**9).segments[0].children
    assert _bounds(word)[:2] == (0, 0)


@pytest.mark.logic
def test_word_confidence_is_joint_token_probability_when_words_align() -> None:
    """Tokens group into words by leading space; a count that differs from the pipeline's gives no confidence."""
    pieces = {1: " he", 2: "llo", 3: " there", 4: ".", 9: "<|0.00|>"}
    block: dict[str, Any] = {
        "tokens": [9, 1, 2, 3, 4],
        "token_logprob": [-1.0, -0.5, -0.5, -2.0, 0.0],
        "avg_logprob": -0.75,
    }
    got = word_confidences(block["tokens"], block["token_logprob"], pieces.__getitem__)
    assert got == pytest.approx([numpy.exp(-1.0), numpy.exp(-2.0)])
    t = hf.transcript(PIPE_OUT, "word", CLOCK, 10**9, block, pieces.__getitem__, None)
    assert [w.confidence for w in t.segments[0].children] == pytest.approx(got)
    assert t.segments[0].confidence == pytest.approx(numpy.exp(-0.75))
    short = {**block, "tokens": [1, 2], "token_logprob": [-0.5, -0.5]}
    t = hf.transcript(PIPE_OUT, "word", CLOCK, 10**9, short, pieces.__getitem__, None)
    assert [w.confidence for w in t.segments[0].children] == [None, None]


@pytest.mark.logic
def test_merge_concatenates_tokens_in_window_order() -> None:
    """Windows of one transcript concatenate tokens and log-probabilities and average `avg_logprob`."""
    a = {"tokens": [1], "token_logprob": [-1.0], "avg_logprob": -1.0}
    b = {"tokens": [2], "token_logprob": [-3.0], "avg_logprob": -3.0}
    merged = merge_confidence_blocks([a, b])
    assert merged["tokens"] == [1, 2] and merged["token_logprob"] == [-1.0, -3.0]
    assert merged["avg_logprob"] == -2.0


@pytest.mark.logic
def test_capture_leaves_a_generate_that_refuses_the_flags_untouched() -> None:
    """The wrapper falls back to the plain call and is removed on exit."""

    class Model:
        def generate(self, x: int) -> int:
            return x + 1

    class Pipe:
        model = Model()

    pipe = Pipe()
    with capture_token_confidence(pipe) as blocks:
        assert pipe.model.generate(1) == 2
    assert blocks == [] and "generate" not in vars(pipe.model)


@pytest.mark.logic
def test_stub_handle_returns_a_transcript_wire_form_per_item() -> None:
    """The result is the wire form of a transcript on the audio's own clock."""
    audio = _audio()
    [result] = _run([{"id": "a", "inputs": {"audio": audio}}])
    out = from_wire_form(result["outputs"]["transcript"])
    assert out.segments[0].span.clock_id == audio.anchor.clock_id  # type: ignore[attr-defined]
    assert wire_form(out) == result["outputs"]["transcript"]


@pytest.mark.logic
@pytest.mark.parametrize(
    ("audio", "fault", "text"),
    [(_audio(channels=2), ContentFault, "mono"), (_audio(rate=8000), ContentFault, "16000 Hz")],
)
def test_unsupported_audio_is_a_content_fault_without_a_gap(audio: Audio, fault: type, text: str) -> None:
    """Stereo or a wrong rate fails that item as a typed content fault."""
    [result] = _run([{"id": "a", "inputs": {"audio": audio}}])
    assert isinstance(result["error"], fault) and text in str(result["error"])


@pytest.mark.logic
def test_input_below_the_declared_minimum_is_a_content_fault() -> None:
    """The front end shared with crisperwhisper sets the minimum: one sample under 25 ms fails, 25 ms transcribes."""
    n = MIN_INPUT_NS * hf.EXPECTED_RATE // 1_000_000_000
    short, ok = _run([{"id": "s", "inputs": {"audio": _audio(n - 1)}}, {"id": "o", "inputs": {"audio": _audio(n)}}])
    assert n == 400 and isinstance(short["error"], ContentFault) and "outputs" in ok


@pytest.mark.logic
def test_faults_stay_with_their_item() -> None:
    """A content fault and an out of memory on two items leave the third unchanged and are typed."""
    items = [{"id": i, "inputs": {"audio": _audio(1600 + n)}} for n, i in enumerate("abc")]
    clean = _run(items)
    faulted = _run(items, faults={"fail_items": ["a"], "oom_items": ["b"]})
    assert isinstance(faulted[0]["error"], ContentFault) and isinstance(faulted[1]["error"], ResourceFault)
    assert faulted[2] == clean[2]


@pytest.mark.logic
def test_a_pipeline_failure_is_isolated_by_rerunning_items_alone() -> None:
    """When a batch call raises, only the item whose own run raises carries the fault, with `gap: true`."""

    class Picky:
        feature_extractor = hf.StubPipeline.feature_extractor

        def __call__(self, inputs: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            if any(len(i["array"]) == 1700 for i in inputs):
                raise RuntimeError("bad window")
            return hf.StubPipeline()(inputs, **kwargs)

    state = hf.State(Picky(), 16000)
    items = [{"id": i, "inputs": {"audio": _audio(n)}} for i, n in (("a", 1600), ("b", 1700), ("c", 1800))]
    out = hf.handle(state, items, PARAMS)
    assert "outputs" in out[0] and "outputs" in out[2]
    assert (error_record(out[1]["error"])["fault"], error_record(out[1]["error"])["gap"]) == ("infrastructure", True)


class _Recording:
    """A pipeline that records the `batch_size` of each call and raises `error` when it is given one."""

    feature_extractor = hf.StubPipeline.feature_extractor

    def __init__(self, error: BaseException | None = None) -> None:
        self.batch_sizes: list[int] = []
        self.error = error

    def __call__(self, inputs: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        self.batch_sizes.append(kwargs["batch_size"])
        if self.error is not None and len(inputs) > 1:
            raise self.error
        return hf.StubPipeline()(inputs, **kwargs)


def _three() -> list[dict[str, Any]]:
    return [{"id": i, "inputs": {"audio": _audio(1600 + n)}} for n, i in enumerate("abc")]


@pytest.mark.logic
def test_the_placed_batch_is_the_pipeline_batch() -> None:
    """A request of three items is one pipeline call of batch three."""
    pipe = _Recording()
    out = hf.handle(hf.State(pipe, 16000), _three(), PARAMS)
    assert pipe.batch_sizes == [3] and all("outputs" in o for o in out)


@pytest.mark.logic
@pytest.mark.parametrize(("cap", "expected"), [(2, 2), (1, 1), (8, 8)])
def test_a_declared_batch_size_passes_to_the_pipeline_unchanged(cap: int, expected: int) -> None:
    """An explicit `batch_size` counts windows and reaches the pipeline as given, whatever the item count."""
    pipe = _Recording()
    hf.handle(hf.State(pipe, 16000), _three(), {**PARAMS, "backend_params": {"batch_size": cap}})
    assert pipe.batch_sizes == [expected]


@pytest.mark.logic
def test_a_batched_resource_fault_reaches_the_pool_through_call_handle() -> None:
    """A torch-style out of memory in the batched call is a resource failure of the request, not per-item ok."""
    torch_oom = type("OutOfMemoryError", (RuntimeError,), {})("CUDA out of memory")
    pipe = _Recording(torch_oom)
    state: object = hf.State(pipe, 16000)
    out = call_handle(lambda _, items, params: hf.handle(hf.State(pipe, 16000), items, params), state, _three(), PARAMS)
    assert pipe.batch_sizes == [3]
    assert [error_record(o["error"])["fault"] for o in out] == ["resource"] * 3


@pytest.mark.logic
@pytest.mark.parametrize("error", [ResourceFault("device lost memory"), MemoryError(), TimeoutError()])
def test_every_resource_class_fault_propagates_from_handle(error: BaseException) -> None:
    """The worker's classification decides: a resource-class exception is not re-run per item."""
    pipe = _Recording(error)
    with pytest.raises(type(error)):
        hf.handle(hf.State(pipe, 16000), _three(), PARAMS)
    assert pipe.batch_sizes == [3]


@pytest.mark.logic
def test_load_refuses_a_ref_an_unknown_key_and_both_stub_and_model() -> None:
    """A model loads by commit only; init takes its documented keys."""
    model = {"source": "hf", "id": "openai/whisper-tiny", "identity": "main"}
    for init in ({"model": model}, {"stub": True, "bogus": 1}, {"stub": True, "model": model}, {}):
        with pytest.raises(ValueError):
            hf.load(init)


CASES = {
    name: c
    for name, c in contract_cases(REGISTRY, lock_records=LOCK_RECORDS)
    if name.startswith(("speech_to_text::", "speech_to_text/transformers::"))
}


@pytest.mark.contract
@pytest.mark.parametrize("case", [pytest.param(c, id=name) for name, c in CASES.items()])
def test_contract(case: Any) -> None:  # noqa: ANN401 - a no-argument contract case
    """The task and backend satisfy the generated contract, the worker running on its load stub."""
    case()


@pytest.mark.contract
def test_contract_selects_the_backend() -> None:
    """The backend's cases, `output_affecting_produced` among them, are selected, not filtered away."""
    assert "speech_to_text/transformers::output_affecting_produced" in CASES


@pytest.mark.contract
def test_backend_is_a_host_backend_the_senselab_lock_covers() -> None:
    """The backend declares its requirements and the root lock covers them."""
    from senselab._api.packaged import host_lock
    from senselab_core.runtime.envs import host_lock_gaps

    (spec,) = [b for b in REGISTRY.backends("speech_to_text") if b.name == "transformers"]
    assert spec.env.kind == "host" and spec.requirements and not spec.batch_invariant
    lock = host_lock()
    assert lock is not None and host_lock_gaps(lock, spec.requirements) == []


def _real_state() -> Any:  # noqa: ANN401
    """Load the catalogue's default `speech_to_text` model by its pinned commit on CUDA; skip the test without it.

    The default is Whisper large, which a CPU does not run in bounded time.
    """
    from senselab._api.packaged import CATALOGUE

    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        pytest.skip("the default speech_to_text model is Whisper large: real weights need a GPU")

    entry = CATALOGUE.default("speech_to_text")
    model = {"source": "hf", "id": entry.id, "identity": entry.revision}
    return hf.load({"model": model, "device": "cuda"})


def _timed(result: dict[str, Any]) -> list[tuple[Any, ...]]:
    """Return a result's segments and words as `(text, span, child texts)`, without confidences."""
    segments = result["outputs"]["transcript"]["segments"]
    return [(g["text"], g["span"], [(w["text"], w["span"]) for w in g["children"]]) for g in segments]


def _tone(seconds: float, rate: int = 16000, *, gain: float = 0.3) -> Audio:
    t = numpy.arange(max(1, round(seconds * rate))) / rate
    return Audio.from_array((gain * numpy.sin(2 * numpy.pi * 220 * t)).astype(numpy.float32), rate)


@pytest.mark.env
@pytest.mark.model
def test_transformers_backend_content_fault_isolated_with_real_weights() -> None:
    """On the pinned model a content fault on one item is typed and leaves its batch-mates' text and spans unchanged.

    The backend is declared non-batch-invariant, so confidences are not compared.
    """
    items = [{"id": i, "inputs": {"audio": _tone(1.0 + n)}} for n, i in enumerate("abc")]
    state = _real_state()
    clean = hf.handle(state, items, PARAMS)
    faulted = hf.handle(state, items, {**PARAMS, FAULT_KEY: {"fail_items": ["b"]}})
    assert all("outputs" in r for r in clean)
    assert isinstance(faulted[1]["error"], ContentFault)
    assert [_timed(faulted[i]) for i in (0, 2)] == [_timed(clean[i]) for i in (0, 2)]


@pytest.mark.limits
@pytest.mark.model
def test_transformers_backend_edge_inputs_with_real_weights() -> None:
    """One sample, silence, clipping, long input and every rate each give a transcript or a typed fault, never a gap."""
    state = _real_state()
    cases = {
        "one-sample": _tone(1 / 16000),
        "silence": _tone(2.0, gain=0.0),
        "clipping": _tone(2.0, gain=40.0),
        "long": _tone(70.0),
        **{f"rate-{r}": _tone(1.0, r) for r in (8000, 11025, 22050, 32000, 44100, 48000)},
    }
    for name, audio in cases.items():
        [result] = hf.handle(state, [{"id": name, "inputs": {"audio": audio}}], PARAMS)
        if "error" in result:
            assert isinstance(result["error"], (ContentFault, ResourceFault)), (name, result["error"])
            assert audio.sampling_rate != 16000 or name == "one-sample", name
        else:
            assert from_wire_form(result["outputs"]["transcript"]) is not None


@pytest.mark.env
def test_token_confidence_from_logits_scores_every_step() -> None:
    """A uniform step gives the chosen token's id and log-probability."""
    torch = pytest.importorskip("torch")
    from senselab.audio.tasks.speech_to_text.token_confidence import token_confidence_from_logits

    [block] = token_confidence_from_logits(logits=[torch.zeros(1, 4)], sequences=torch.tensor([[7, 2]]))
    assert block["tokens"] == [2] and block["token_logprob"] == pytest.approx([float(numpy.log(0.25))])


@pytest.mark.contract
def test_torch_extra_is_the_backends_requirements() -> None:
    """The root `torch` extra lists only requirements the host backend declares, so the two cannot drift."""
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[6]
    extra = tomllib.loads((root / "pyproject.toml").read_text("utf-8"))["project"]["optional-dependencies"]["torch"]
    (spec,) = [b for b in REGISTRY.backends("speech_to_text") if b.name == "transformers"]
    assert set(extra) <= set(spec.requirements)


@pytest.mark.logic
def test_a_chunk_ending_past_the_audio_is_clamped_to_its_extent() -> None:
    """A pipeline chunk (1.0 s, 9.0 s) of a 2 s recording ends at 2 s, as the other backend's spans do."""
    out = {"text": "late", "chunks": [{"text": "late", "timestamp": (1.0, 9.0)}]}
    (word,) = hf.transcript(out, "word", CLOCK, 2 * 10**9).segments[0].children
    assert (word.span.start_ns, word.span.end_ns) == (10**9, 2 * 10**9)  # type: ignore[union-attr]


@pytest.mark.logic
def test_the_backend_takes_the_wire_form_of_audio_too() -> None:
    """The one input check reads an `Audio` and its wire form alike."""
    audio = _audio()
    [kernel] = _run([{"id": "a", "inputs": {"audio": audio}}])
    [wire] = _run([{"id": "a", "inputs": {"audio": wire_form(audio)}}])
    assert kernel == wire
