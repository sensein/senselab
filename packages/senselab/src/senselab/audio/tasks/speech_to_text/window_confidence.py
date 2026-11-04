"""Token confidence of a CrisperWhisper decode: each window's final ids, scored from the decode's own computation.

`capture_windows` records the ids each window finally emits. On transformers, `capture_decodes` records the raw
logits of every `generate` call from the output head and `spliced_block` reads a window's from them. On ct2,
`capture_windows` also records the encoder output of each window's decode and scores the window as soon as it is
final (`ct2_window_logits`, one decoder pass on that output, through `window_confidence`), so one or two encoder
outputs are held at a time. `torch`, `numpy` and `ctranslate2` are imported only where logits are scored, so
importing this module loads no framework.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Optional

from senselab.audio.tasks.speech_to_text.token_confidence import (
    logprob_block,
    token_confidence_from_logits,
    word_prefix,
)

__all__ = [
    "Decode",
    "Window",
    "capture_decodes",
    "capture_windows",
    "ct2_window_logits",
    "spliced_block",
    "surviving_windows",
    "window_confidence",
]

DECODES = ("generate_with_repair", "generate_with_repair_and_attention")
"""The engine methods that decode one window, each `(features, prompt_tokens, ...)` returning its ids first."""
REPLACES = "cross_attention_for_tokens"
"""The engine method `(features, prompt_tokens, gen_ids)` the library calls when it replaces a window's ids."""
SEGMENT = "generate_greedy_with_attention"
"""The ct2 engine method `(encoded, prompt, ...)` returning `(state, ids)` that a window decode calls per segment."""


@dataclass(frozen=True)
class Window:
    """One decoded window: its encoder input, its prompt, the ids it emits, and its decode's encoder output if seen.

    Once scored, `token_logprob` holds the log-probability of each id, `encoded` is released, and `reused` says
    whether the score read the decode's encoder output.
    """

    features: Any
    prompt_tokens: list[int]
    ids: list[int]
    encoded: Any = None
    token_logprob: list[float] | None = None
    reused: bool = False


@dataclass
class Decode:
    """One `generate` call: its encoder input, decoder prefix and generated ids, and the log-probabilities it saw.

    `own[k]` is the log-probability of `generated[k]`; `tail`, when the call has a step past its returned ids (an
    end token the result leaves out), is the log-probability row of that step over the vocabulary.
    """

    features: Any
    prefix: list[int]
    generated: list[int]
    own: list[float]
    tail: Any = None


@dataclass
class Capture:
    """What `capture_decodes` records: the calls, and the seconds spent scoring them."""

    decodes: list[Decode] = field(default_factory=list)
    scoring_s: float = 0.0


@contextlib.contextmanager
def capture_windows(
    engine: Any,  # noqa: ANN401 - a crisperwhisper engine
    score: Callable[[Window], list[float]] | None = None,
) -> Iterator[list[Window]]:
    """Wrap the `DECODES` and `REPLACES` of `engine` (and, with `score`, its `SEGMENT`), recording each window.

    A decode records a window (with `score`, also the encoder output of its first `SEGMENT` call on its own prompt);
    a `REPLACES` call on the same `features` and prompt replaces that window's ids and keeps its encoder output
    (coverage fallback, early-EOT recovery). With `score`, a window is scored and its encoder output released as
    soon as a decode on other features starts, and the rest at the end of the block; a window replaced after it
    was scored is scored again, on its mel. Yields the windows in decode order. The wrappers return what the caller
    would have received and are removed on exit.
    """
    recorder = _Recorder(score)
    wrappers: dict[str, Callable[[Callable[..., Any]], Callable[..., Any]]] = dict.fromkeys(DECODES, recorder.decode)
    wrappers[REPLACES] = recorder.replacement
    if score is not None and callable(getattr(engine, SEGMENT, None)):
        wrappers[SEGMENT] = recorder.segment
    with _patched(engine, wrappers):
        yield recorder.windows
        recorder.settle()


class _Recorder:
    """The state of `capture_windows`: the windows, their index by features and prompt, and those not yet scored."""

    def __init__(self, score: Callable[[Window], list[float]] | None) -> None:
        self.score = score
        self.windows: list[Window] = []
        self.index: dict[tuple[int, tuple[int, ...]], int] = {}
        self.segments: list[tuple[Any, list[int]]] = []
        self.pending: list[int] = []

    def record(self, features: Any, prompt_tokens: Sequence[int], ids: Sequence[int], encoded: Any) -> None:  # noqa: ANN401
        key = (id(features), tuple(int(t) for t in prompt_tokens))
        given = [int(t) for t in ids]
        at = self.index.setdefault(key, len(self.windows))
        if at == len(self.windows):
            self.windows.append(Window(features, list(key[1]), given, encoded))
        elif self.windows[at].token_logprob is None:
            old = self.windows[at]
            self.windows[at] = replace(old, ids=given, encoded=encoded if old.encoded is None else old.encoded)
        else:
            self.windows[at] = replace(self.windows[at], ids=given, token_logprob=None, reused=False)
        if at not in self.pending:
            self.pending.append(at)

    def settle(self, keep: Any = None) -> None:  # noqa: ANN401 - the features whose windows stay open
        if self.score is None:
            return
        for at in [at for at in self.pending if self.windows[at].features is not keep]:
            window = self.windows[at]
            logprob = self.score(window) if window.ids else []
            self.windows[at] = replace(window, encoded=None, token_logprob=logprob, reused=window.encoded is not None)
            self.pending.remove(at)

    def decode(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def call(features: Any, prompt_tokens: Sequence[int], *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            self.settle(keep=features)
            self.segments.clear()
            result = original(features, prompt_tokens, *args, **kwargs)
            prompt = [int(t) for t in prompt_tokens]
            encoded = next((enc for enc, p in self.segments if p == prompt), None)
            self.segments.clear()
            self.record(features, prompt, result[0] if isinstance(result, tuple) else result, encoded)
            return result

        return call

    def replacement(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def call(features: Any, prompt_tokens: Sequence[int], gen_ids: Sequence[int]) -> Any:  # noqa: ANN401
            result = original(features, prompt_tokens, gen_ids)
            self.record(features, prompt_tokens, gen_ids, None)
            return result

        return call

    def segment(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def call(encoded: Any, prompt: Sequence[int], *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            self.segments.append((encoded, [int(t) for t in prompt]))
            return original(encoded, prompt, *args, **kwargs)

        return call


@contextlib.contextmanager
def _patched(target: Any, wrappers: dict[str, Callable[[Callable[..., Any]], Callable[..., Any]]]) -> Iterator[None]:  # noqa: ANN401
    originals = {name: getattr(target, name) for name in wrappers}
    own = {name for name in wrappers if name in vars(target)}
    for name, wrap in wrappers.items():
        setattr(target, name, wrap(originals[name]))
    try:
        yield
    finally:
        for name, original in originals.items():
            if name in own:
                setattr(target, name, original)
            else:
                delattr(target, name)


@contextlib.contextmanager
def capture_decodes(model: Any) -> Iterator[Capture]:  # noqa: ANN401 - a transformers model
    """Record every `model.generate` call of the block as a `Decode`, leaving the call and its result unchanged.

    A forward hook on the output head (`get_output_embeddings`) keeps, during a `generate` call and outside its
    `detect_language`, the raw logits of each step's last position. A call is recorded when its batch is one and
    its steps are its returned ids, or its ids and one more (an end token the result leaves out). A model without a
    head or `generate` records nothing.
    """
    capture = Capture()
    head = model.get_output_embeddings() if callable(getattr(model, "get_output_embeddings", None)) else None
    if head is None or not callable(getattr(model, "generate", None)):
        yield capture
        return
    steps: list[Any] | None = None

    def _hook(module: Any, inputs: Any, output: Any) -> None:  # noqa: ANN401 - a torch module and tensors
        if steps is not None:
            steps.append(output[:, -1].detach().clone())

    def _generate(original: Callable[..., Any]) -> Callable[..., Any]:
        def call(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 - passthrough
            nonlocal steps
            steps = []
            try:
                result = original(*args, **kwargs)
            finally:
                rows, steps = steps, None
            started = time.perf_counter()
            decode = _decode_of(args, kwargs, result, rows)
            if decode is not None:
                capture.decodes.append(decode)
            capture.scoring_s += time.perf_counter() - started
            return result

        return call

    def _unrecorded(original: Callable[..., Any]) -> Callable[..., Any]:
        def call(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 - passthrough
            nonlocal steps
            paused, steps = steps, None
            try:
                return original(*args, **kwargs)
            finally:
                steps = paused

        return call

    wrappers: dict[str, Callable[[Callable[..., Any]], Callable[..., Any]]] = {"generate": _generate}
    if callable(getattr(model, "detect_language", None)):
        wrappers["detect_language"] = _unrecorded
    handle = head.register_forward_hook(_hook)
    try:
        with _patched(model, wrappers):
            yield capture
    finally:
        handle.remove()


def _decode_of(args: tuple[Any, ...], kwargs: dict[str, Any], result: Any, rows: list[Any]) -> Decode | None:  # noqa: ANN401
    import torch  # noqa: PLC0415 - torch loads only where logits are scored

    sequences = getattr(result, "sequences", result)
    if not rows or not isinstance(sequences, torch.Tensor) or sequences.shape[0] != 1:
        return None
    if any(row.shape[0] != 1 for row in rows):
        return None
    given = kwargs.get("decoder_input_ids")
    prefix = [] if given is None else [int(t) for t in given[0].tolist()]
    sequence = [int(t) for t in sequences[0].tolist()]
    generated = sequence[len(prefix) :] if sequence[: len(prefix)] == prefix else sequence
    if len(rows) - len(generated) not in (0, 1):
        return None
    logprobs = torch.log_softmax(torch.cat(rows).float(), dim=-1)
    at = torch.tensor(generated, dtype=torch.long, device=logprobs.device)
    own = logprobs[torch.arange(len(generated), device=logprobs.device), at].tolist()
    tail = logprobs[-1].cpu() if len(rows) > len(generated) else None
    features = args[0] if args else kwargs.get("input_features")
    return Decode(features, prefix, generated, own, tail)


def spliced_block(
    window: Window, decodes: Sequence[Decode], special_token_ids: Optional[set[int]] = None
) -> dict[str, Any] | None:
    """Return the block of `window`'s ids read from the `decodes` on its features, or None if one id has none.

    Id `j` is read from a decode whose prefix and first `k` generated ids are the window's prompt and first `j` ids:
    `own[k]` if that decode generated id `j` there, else its `tail` row when `k` is its length. A repaired window is
    the earlier call up to the rewind, then the later call.
    """
    ids, prompt = window.ids, window.prompt_tokens
    full = prompt + ids
    calls = [d for d in decodes if d.features is window.features]
    values: list[float] = []
    current: Decode | None = None
    offset = 0
    for j, token in enumerate(ids):
        if current is None or not _reads(current, offset, token):
            current, offset = _find(calls, full, len(prompt) + j, token)
            if current is None:
                return None
        values.append(current.own[offset] if offset < len(current.own) else float(current.tail[token]))
        offset += 1
    return logprob_block(ids, values, special_token_ids)


def _reads(decode: Decode, offset: int, token: int) -> bool:
    if offset < len(decode.generated):
        return decode.generated[offset] == token
    return offset == len(decode.generated) and decode.tail is not None


def _find(calls: Sequence[Decode], full: list[int], position: int, token: int) -> tuple[Decode | None, int]:
    for decode in reversed(calls):
        offset = position - len(decode.prefix)
        if (
            0 <= offset <= len(decode.generated)
            and decode.prefix + decode.generated[:offset] == full[:position]
            and _reads(decode, offset, token)
        ):
            return decode, offset
    return None, 0


def surviving_windows(
    windows: Sequence[Window],
    kept_words: Sequence[int],
    decode: Callable[[int], str],
    special_token_ids: Optional[set[int]] = None,
) -> list[Window]:
    """Return `windows` cut to the ids of the first `kept_words[i]` words of window `i`, the words the merge keeps.

    Raises:
        ValueError: if `kept_words` does not give one count per window.
    """
    if len(kept_words) != len(windows):
        raise ValueError(f"{len(kept_words)} kept-word counts for {len(windows)} windows")
    cut = [word_prefix(w.ids, kept, decode, special_token_ids) for w, kept in zip(windows, kept_words)]
    return [
        replace(w, ids=w.ids[:n], token_logprob=None if w.token_logprob is None else w.token_logprob[:n])
        for w, n in zip(windows, cut)
    ]


def _host(view: Any) -> Any:  # noqa: ANN401 - a ctranslate2 StorageView or a numpy array in, a numpy array out
    import numpy  # noqa: PLC0415 - loads only where ct2 logits are scored

    if isinstance(view, numpy.ndarray):
        return view.astype(numpy.float32, copy=False)
    import ctranslate2  # noqa: PLC0415 - as above

    return numpy.array(view.to(ctranslate2.DataType.float32).to_device(ctranslate2.Device.cpu))


def ct2_window_logits(engine: Any, window: Window) -> Any:  # noqa: ANN401 - a CT2Engine in, a numpy array out
    """Return the raw logits `(len(ids), vocab)` with which the ct2 engine predicts each of the window's ids.

    `prefill` of the prompt on the window's recorded encoder output (its mel when none was recorded) gives the row
    predicting `ids[0]`; `forward_batch` over `ids[:-1]` gives the rest.

    Raises:
        ValueError: if the model returns a row count other than `len(ids)`.
    """
    import numpy  # noqa: PLC0415 - loads only where ct2 logits are scored

    source = window.features if window.encoded is None else window.encoded
    state, first = engine.model.prefill(source, window.prompt_tokens)
    head = _host(first)
    rows = [head.reshape(-1, head.shape[-1])[-1:]]
    if len(window.ids) > 1:
        rest = _host(engine.model.forward_batch(state, window.ids[:-1]))
        rows.append(rest.reshape(-1, rest.shape[-1]))
    logits = numpy.concatenate(rows)
    if logits.shape[0] != len(window.ids):
        raise ValueError(f"{logits.shape[0]} logit rows for {len(window.ids)} generated tokens")
    return logits


def window_confidence(
    engine: Any,  # noqa: ANN401 - a crisperwhisper engine
    windows: Sequence[Window],
    window_logits: Callable[[Any, Window], Any],
    special_token_ids: Optional[set[int]] = None,
) -> list[dict[str, Any]]:
    """Return one `token_confidence_from_logits` block per window with ids, scored from `window_logits`."""
    import torch  # noqa: PLC0415 - torch loads only where logits are scored

    blocks: list[dict[str, Any]] = []
    for window in windows:
        if window.ids:
            logits = torch.as_tensor(window_logits(engine, window))
            blocks.extend(
                token_confidence_from_logits(
                    logits=list(logits), sequences=torch.tensor([window.ids]), special_token_ids=special_token_ids
                )
            )
    return blocks
