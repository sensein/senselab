"""Backend `transformers` of `speech_to_text`: the transformers ASR pipeline over a Whisper-family model, in-process.

`load` builds the pipeline from the pinned model in `init["model"]`, or the stub pipeline under `init["stub"]`;
`handle` transcribes a batch of mono 16 kHz audio and answers one `Transcript` wire form, or one typed fault, per item.
`transformers` and `torch` are imported only inside `load` and the confidence engine's scoring.
"""

import functools
import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy
from pydantic import Field
from senselab_core.kernel.declare import BackendParams, Env, backend
from senselab_core.kernel.results import Transcript, TranscriptNode
from senselab_core.kernel.time import Span
from senselab_core.kernel.wire import wire_form
from senselab_core.runtime.audio_io import READER
from senselab_worker.fake import FakeFaults, injected, without_fault

from senselab._api.declarations import SpeechToTextParams
from senselab.audio.tasks.speech_to_text.token_confidence import (
    capture_token_confidence,
    merge_confidence_blocks,
    special_token_ids,
    word_confidences,
)
from senselab.item_faults import (
    NS_PER_S,
    WHISPER_MIN_INPUT_NS,
    clamp_ns,
    init_model,
    mono_check,
    raise_injected,
    wire_audio,
)

WORKER_MODULE = "senselab.audio.tasks.speech_to_text.huggingface"
EXPECTED_RATE = 16_000
MIN_INPUT_NS = WHISPER_MIN_INPUT_NS
_INIT_KEYS = frozenset({"stub", "model", "device", "array_dir"})


class TransformersParams(BackendParams):
    """Backend params of `transformers`: tokens per window, window length in seconds, windows per batch."""

    max_new_tokens: int = Field(default=128, ge=1)
    chunk_length_s: int = Field(default=30, ge=1)
    batch_size: int = Field(default=1, ge=1)


@backend(
    task="speech_to_text",
    name="transformers",
    version="1",
    stack="transformers",
    worker=WORKER_MODULE,
    load_stub={"stub": True, "device": "cpu"},
    batch_invariant=False,
    min_input_ns=MIN_INPUT_NS,
    output_affecting=("batch_formation",),
    devices=("cuda", "cpu"),
    env=Env.host(),
    requirements=("transformers>=5.8", "torch>=2.4", "numpy"),
    params=TransformersParams,
)
class Transformers:
    """The transformers ASR pipeline over the pinned catalogue model it is called with, one model per process."""


@dataclass(frozen=True)
class State:
    """A loaded backend: the pipeline and the sampling rate it takes."""

    pipe: Any
    sampling_rate: int


class StubPipeline:
    """The fake stack's model: deterministic words, evenly spaced over the audio, derived from its samples."""

    class _Extractor:
        sampling_rate = EXPECTED_RATE

    feature_extractor = _Extractor()

    def __call__(self, inputs: Sequence[Mapping[str, Any]], **kwargs: Any) -> list[dict[str, Any]]:  # noqa: ANN401
        """Return one `{"text", "chunks"}` per input, as the transformers ASR pipeline does."""
        words = kwargs.get("return_timestamps") == "word"
        return [self._one(numpy.asarray(i["array"]), words) for i in inputs]

    @staticmethod
    def _one(samples: numpy.ndarray, words: bool) -> dict[str, Any]:
        digest = hashlib.sha256(samples.astype(numpy.float32).tobytes()).hexdigest()
        seconds = samples.size / EXPECTED_RATE
        texts = [f" w{digest[4 * i : 4 * i + 4]}" for i in range(3)]
        step = seconds / len(texts)
        chunks = [{"text": t, "timestamp": (i * step, (i + 1) * step)} for i, t in enumerate(texts)]
        return {"text": "".join(texts), "chunks": chunks}


def state_bytes(state: State) -> int:
    """Return the bytes of the loaded model's parameters, 0 for the stub."""
    model = getattr(state.pipe, "model", None)
    if model is None:
        return 0
    return int(sum(p.numel() * p.element_size() for p in model.parameters()))


def load(init: dict[str, Any]) -> State:
    """Return the worker's state from `init`.

    `init` holds `model` (`{"source": "hf", "id", "identity"}`, `identity` a 40-hex commit) and `device`, or
    `"stub": True` for the stub pipeline; it may hold `array_dir`.

    Raises:
        ValueError: on an unknown key, a model that is not a pinned hub commit, or both `stub` and `model`.
    """
    model = init_model(init, _INIT_KEYS, "transformers")
    if model is None:
        return State(StubPipeline(), EXPECTED_RATE)
    from transformers import pipeline  # noqa: PLC0415 - the framework loads in `load` only

    pipe = pipeline(
        "automatic-speech-recognition",
        model=str(model["id"]),
        revision=str(model["identity"]),
        device=init.get("device", "cpu"),
    )
    return State(pipe, int(pipe.feature_extractor.sampling_rate))  # type: ignore[union-attr]


def handle(state: State, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one `{"outputs": {"transcript": <transcript@2.0 wire form>}}` per item, or its typed fault.

    Items that pass the checks run in one pipeline call; if it raises, each runs alone so a fault stays with its item.
    """
    task = SpeechToTextParams.model_validate({k: v for k, v in without_fault(params).items() if k != "backend_params"})
    faults = injected(params)
    tuning = TransformersParams.model_validate(params.get("backend_params", {}))
    results: list[dict[str, Any] | None] = [None] * len(items)
    ready: list[tuple[int, _Prepared]] = []
    for index, item in enumerate(items):
        try:
            ready.append((index, _prepare(state, item, faults)))
        except Exception as exc:  # noqa: BLE001 - every per-item failure is reported as that item's typed fault
            results[index] = {"error": exc}
    try:
        for (index, _), transcript in zip(ready, _transcribe(state, [p for _, p in ready], task, tuning)):
            results[index] = {"outputs": {"transcript": transcript}}
    except Exception:  # noqa: BLE001 - re-run each item alone to find the one at fault
        for index, prepared in ready:
            try:
                results[index] = {"outputs": {"transcript": _transcribe(state, [prepared], task, tuning)[0]}}
            except Exception as exc:  # noqa: BLE001 - as above
                results[index] = {"error": exc}
    return [r for r in results if r is not None]


@dataclass(frozen=True)
class _Prepared:
    samples: numpy.ndarray
    clock_id: str
    duration_ns: int


def _prepare(state: State, item: dict[str, Any], faults: FakeFaults) -> _Prepared:
    raise_injected(item["id"], faults)
    value = item["inputs"]["audio"]
    if isinstance(value, dict):
        samples, rate, clock_id = wire_audio(value)
    else:
        samples, rate, clock_id = value.read(reader=READER), float(value.sampling_rate), str(value.anchor.clock_id)
    samples, clock_id = mono_check(samples, rate, clock_id, state.sampling_rate, "whisper", MIN_INPUT_NS)
    return _Prepared(samples, clock_id, round(samples.size / state.sampling_rate * NS_PER_S))


def _transcribe(
    state: State, prepared: list[_Prepared], task: SpeechToTextParams, tuning: TransformersParams
) -> list[dict[str, Any]]:
    if not prepared:
        return []
    kwargs: dict[str, Any] = {
        "chunk_length_s": tuning.chunk_length_s,
        "batch_size": tuning.batch_size,
        "generate_kwargs": {"max_new_tokens": tuning.max_new_tokens, "num_beams": 1}
        | ({"language": task.language.lower()} if task.language else {}),
    }
    if task.timestamps != "none":
        kwargs["return_timestamps"] = "word" if task.timestamps == "word" else True
    inputs = [{"array": p.samples, "sampling_rate": state.sampling_rate} for p in prepared]
    special_ids = special_token_ids(state.pipe)
    with capture_token_confidence(state.pipe, special_token_ids=special_ids) as blocks:
        outputs = state.pipe(inputs, **kwargs)
    tokenizer = getattr(state.pipe, "tokenizer", None)
    decode = None if tokenizer is None else (lambda token: str(tokenizer.decode([token])))
    return [
        wire_form(
            transcript(
                out, task.timestamps, p.clock_id, p.duration_ns, _block(blocks, len(prepared), i), decode, special_ids
            )
        )
        for i, (p, out) in enumerate(zip(prepared, outputs))
    ]


def _block(blocks: list[dict[str, Any]], count: int, index: int) -> dict[str, Any] | None:
    """Return the confidence block of item `index`: one per item, or all merged for a lone item, else none."""
    if len(blocks) == count:
        return blocks[index]
    if count == 1 and blocks:
        return merge_confidence_blocks(blocks)
    return None


def _span(stamp: Sequence[float | None], clock_id: str, floor_ns: int, duration_ns: int) -> Span:
    """Return the span of a pipeline `(start, end)` in seconds; a missing start is `floor_ns`, an end the audio end."""
    start, end = clamp_ns(stamp[0], stamp[1], floor_ns, duration_ns)
    return Span(start_ns=start, end_ns=end, clock_id=clock_id)


def transcript(  # noqa: PLR0913 - the pipeline output, its level, the audio's clock and length, and the confidence
    output: Mapping[str, Any],
    timestamps: str,
    clock_id: str,
    duration_ns: int,
    block: dict[str, Any] | None = None,
    decode: Callable[[int], str] | None = None,
    special_token_ids: set[int] | None = None,
) -> Transcript:
    """Return the `Transcript` of one pipeline output at the `timestamps` level.

    `word` gives one segment dividing into words, `segment` one node per timed chunk, `none` one untimed segment; an
    empty text gives no segment. Spans are in integer ns on `clock_id`. A segment alone in the transcript carries
    `exp(avg_logprob)` as its confidence; each word carries its tokens' joint probability when the captured tokens
    divide into exactly the pipeline's words, and none otherwise.
    """
    over = functools.partial(Transcript, extent=Span(start_ns=0, end_ns=duration_ns, clock_id=clock_id))
    text = str(output.get("text", "")).strip()
    if not text:
        return over(segments=())
    chunks = [c for c in output.get("chunks") or [] if str(c["text"]).strip()]
    mean = None if block is None or block.get("avg_logprob") is None else min(1.0, math.exp(block["avg_logprob"]))
    if timestamps == "none" or not chunks:
        return over(segments=(TranscriptNode(text=text, confidence=mean),))
    timed: list[tuple[str, Span]] = []
    floor = 0
    for chunk in chunks:
        span = _span(chunk["timestamp"], clock_id, floor, duration_ns)
        timed.append((str(chunk["text"]).strip(), span))
        floor = span.end_ns
    if timestamps == "segment":
        segments = tuple(TranscriptNode(text=t, span=s, confidence=mean if len(timed) == 1 else None) for t, s in timed)
        return over(segments=segments)
    scores = _word_scores(block, decode, special_token_ids, len(timed))
    words = tuple(TranscriptNode(text=t, span=s, confidence=c) for (t, s), c in zip(timed, scores))
    whole = Span(start_ns=timed[0][1].start_ns, end_ns=timed[-1][1].end_ns, clock_id=clock_id)
    return over(segments=(TranscriptNode(text=text, span=whole, confidence=mean, children=words),))


def _word_scores(
    block: dict[str, Any] | None,
    decode: Callable[[int], str] | None,
    special_token_ids: set[int] | None,
    count: int,
) -> list[float | None]:
    if block is None or decode is None or not block.get("tokens"):
        return [None] * count
    scores = word_confidences(block["tokens"], block["token_logprob"], decode, special_token_ids)
    return list(scores) if len(scores) == count else [None] * count
