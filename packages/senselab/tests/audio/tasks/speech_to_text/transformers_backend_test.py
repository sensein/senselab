"""The `transformers` backend of `speech_to_text`: transcript construction, typed faults and batch isolation."""

from typing import Any

import numpy
import pytest
from senselab.audio.tasks.speech_to_text import huggingface as hf
from senselab.audio.tasks.speech_to_text.token_confidence import (
    capture_token_confidence,
    merge_confidence_blocks,
    word_confidences,
)
from senselab_core.kernel.declare import REGISTRY
from senselab_core.kernel.signals import Audio
from senselab_core.kernel.wire import from_wire_form, wire_form
from senselab_core.testing.contracts import contract_cases
from senselab_worker.fake import FakeFaults
from senselab_worker.faults import ContentFault, ResourceFault

CLOCK = "content:" + "a" * 64
PIPE_OUT = {
    "text": " hello there.",
    "chunks": [{"text": " hello", "timestamp": (0.0, 0.5)}, {"text": " there.", "timestamp": (0.5, None)}],
}
PARAMS: dict[str, Any] = {"language": None, "timestamps": "word", "backend_params": {}}


def _audio(samples: int = 1600, rate: int = 16000, channels: int = 1) -> Audio:
    x = numpy.sin(numpy.arange(samples) / 7.0).astype(numpy.float32)
    return Audio.from_array(numpy.tile(x, (channels, 1)), rate)


def _run(items: list[dict[str, Any]], params: dict[str, Any] = PARAMS, faults: dict[str, Any] | None = None) -> Any:  # noqa: ANN401
    state = hf.load({"stub": True, "device": "cpu", "faults": faults or {}})
    return hf.handle(state, items, params)


@pytest.mark.logic
def test_word_transcript_has_integer_ns_spans_and_children() -> None:
    """A word transcript is one segment over its words, spans in integer ns on the audio's clock."""
    t = hf.transcript(PIPE_OUT, "word", CLOCK, 1_000_000_000)
    (segment,) = t.segments
    assert segment.text == "hello there." and [w.text for w in segment.children] == ["hello", "there."]
    assert (segment.span.start_ns, segment.span.end_ns) == (0, 1_000_000_000)  # type: ignore[union-attr]
    assert [(w.span.start_ns, w.span.end_ns) for w in segment.children] == [  # type: ignore[union-attr]
        (0, 500_000_000),
        (500_000_000, 1_000_000_000),
    ]
    assert all(type(w.span.start_ns) is int and w.span.clock_id == CLOCK for w in segment.children)  # type: ignore[union-attr]


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
    assert (word.span.start_ns, word.span.end_ns) == (0, 0)  # type: ignore[union-attr]


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
def test_faults_stay_with_their_item() -> None:
    """A content fault and an out of memory on two items leave the third unchanged and are typed."""
    items = [{"id": i, "inputs": {"audio": _audio(1600 + n)}} for n, i in enumerate("abc")]
    clean = _run(items)
    faulted = _run(items, faults={"fail_items": ["a"], "oom_items": ["b"]})
    assert isinstance(faulted[0]["error"], ContentFault) and isinstance(faulted[1]["error"], ResourceFault)
    assert faulted[2] == clean[2]


@pytest.mark.logic
def test_a_pipeline_failure_is_isolated_by_rerunning_items_alone() -> None:
    """When a batch call raises, only the item whose own run raises carries the fault, typed, never a gap."""

    class Picky(hf.StubPipeline):
        def __call__(self, inputs: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            if any(len(i["array"]) == 1700 for i in inputs):
                raise RuntimeError("bad window")
            return super().__call__(inputs, **kwargs)

    state = hf.State(Picky(), 16000, FakeFaults())
    items = [{"id": i, "inputs": {"audio": _audio(n)}} for i, n in (("a", 1600), ("b", 1700), ("c", 1800))]
    out = hf.handle(state, items, PARAMS)
    assert "outputs" in out[0] and "outputs" in out[2]
    assert isinstance(out[1]["error"], ContentFault)


@pytest.mark.logic
def test_load_refuses_a_ref_an_unknown_key_and_both_stub_and_model() -> None:
    """A model loads by commit only; init takes its documented keys."""
    model = {"source": "hf", "id": "openai/whisper-tiny", "identity": "main"}
    for init in ({"model": model}, {"stub": True, "bogus": 1}, {"stub": True, "model": model}, {}):
        with pytest.raises(ValueError):
            hf.load(init)


@pytest.mark.contract
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(c, id=name)
        for name, c in contract_cases(REGISTRY)
        if name.startswith(("speech_to_text::", "transformers::"))
    ],
)
def test_contract(case: Any) -> None:  # noqa: ANN401 - a no-argument contract case
    """The task and backend satisfy the generated contract, the worker running on its load stub."""
    case()


@pytest.mark.contract
def test_backend_is_a_host_backend_the_senselab_lock_covers() -> None:
    """The backend declares its requirements and the root lock covers them."""
    from senselab._api.packaged import host_lock
    from senselab_core.runtime.envs import host_lock_gaps

    (spec,) = [b for b in REGISTRY.backends("speech_to_text") if b.name == "transformers"]
    assert spec.env.kind == "host" and spec.requirements and not spec.batch_invariant
    lock = host_lock()
    assert lock is not None and host_lock_gaps(lock, spec.requirements) == []


def _real_state(faults: dict[str, Any] | None = None) -> Any:  # noqa: ANN401
    """Load the catalogue's default `speech_to_text` model by its pinned commit, on CUDA when present."""
    import torch
    from senselab._api.packaged import CATALOGUE

    entry = CATALOGUE.default("speech_to_text")
    model = {"source": "hf", "id": entry.id, "identity": entry.revision}
    return hf.load({"model": model, "device": "cuda" if torch.cuda.is_available() else "cpu", "faults": faults or {}})


def _tone(seconds: float, rate: int = 16000, *, gain: float = 0.3) -> Audio:
    t = numpy.arange(max(1, round(seconds * rate))) / rate
    return Audio.from_array((gain * numpy.sin(2 * numpy.pi * 220 * t)).astype(numpy.float32), rate)


@pytest.mark.env
@pytest.mark.model
def test_transformers_backend_content_fault_isolated_with_real_weights() -> None:
    """On the pinned model a content fault on one item leaves its batch-mates' transcripts unchanged and is typed."""
    items = [{"id": i, "inputs": {"audio": _tone(1.0 + n)}} for n, i in enumerate("abc")]
    clean = hf.handle(_real_state(), items, PARAMS)
    faulted = hf.handle(_real_state({"fail_items": ["b"]}), items, PARAMS)
    assert all("outputs" in r for r in clean)
    assert isinstance(faulted[1]["error"], ContentFault)
    assert (faulted[0], faulted[2]) == (clean[0], clean[2])


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


@pytest.mark.contract
def test_no_lock_checks_installed_versions_and_refuses_a_gap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,  # noqa: ANN401
) -> None:
    """With no `uv.lock` above the package, `open_env` checks the backend's requirements against what is installed."""
    from senselab._api import packaged
    from senselab_core.runtime.envs import HostLockLacksBackend
    from senselab_core.runtime.profiles import ExecutionProfile

    monkeypatch.setattr(packaged, "host_lock", lambda: None)
    (spec,) = [b for b in REGISTRY.backends("speech_to_text") if b.name == "transformers"]
    gap = spec.model_copy(update={"requirements": ("no-such-dist-xyz>=1",)})
    with pytest.raises(HostLockLacksBackend, match="no-such-dist-xyz"):
        packaged.open_env(gap, ExecutionProfile.load())
