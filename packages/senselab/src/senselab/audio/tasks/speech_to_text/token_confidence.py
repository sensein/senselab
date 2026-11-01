"""Per-token ASR confidence: entropy, log-probability and no-speech probability read off the decoder's own logits.

`capture_token_confidence` wraps `pipe.model.generate` for one call and scores its raw logits;
`word_confidences` turns the scored tokens into one probability per word. `torch` is imported only where logits
are scored, so importing this module loads no framework.
"""

from __future__ import annotations

import contextlib
import math
import re
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import torch

__all__ = [
    "capture_token_confidence",
    "merge_confidence_blocks",
    "token_confidence_from_logits",
    "whisper_token_ids",
    "word_confidences",
]

_NO_SPEECH_TOKEN_SPELLINGS = ("<|nospeech|>", "<|nocaptions|>")
"""Vocabulary spellings of Whisper's no-speech marker, in preference order; a hit equal to the unk id is rejected."""
_CONTROL_TOKEN = re.compile(r"<\|.*\|>")
"""A control token's text: language, task and timestamp markers, which no word contains."""
_LOGIT_DIMS = 2
_BLOCK_KEYS = ("token_entropy", "avg_logprob", "no_speech_prob", "tokens", "token_logprob")


def whisper_token_ids(pipe: Any) -> tuple[Optional[int], Optional[set[int]]]:  # noqa: ANN401 - a transformers pipeline
    """Return `(no_speech_token_id, special_token_ids)` of a pipeline's tokenizer.

    Both are `None` for a backend without a tokenizer, which leaves the confidence fields empty.

    Args:
        pipe: A transformers pipeline, or any object exposing `tokenizer` and `model.generation_config`.

    Returns:
        The no-speech vocabulary id and the set of special-token ids to exclude from `avg_logprob`.
    """
    tokenizer = getattr(pipe, "tokenizer", None)
    if tokenizer is None:
        return None, None

    no_speech_id: Optional[int] = None
    generation_config = getattr(getattr(pipe, "model", None), "generation_config", None)
    for attr in ("no_speech_token_id", "no_speech_token"):
        candidate = getattr(generation_config, attr, None)
        if isinstance(candidate, int):
            no_speech_id = candidate
            break

    if no_speech_id is None:
        unk = getattr(tokenizer, "unk_token_id", None)
        for spelling in _NO_SPEECH_TOKEN_SPELLINGS:
            try:
                candidate = tokenizer.convert_tokens_to_ids(spelling)
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
            if isinstance(candidate, int) and candidate != unk:
                no_speech_id = candidate
                break

    special_ids: Optional[set[int]] = None
    raw_special = getattr(tokenizer, "all_special_ids", None)
    if raw_special is not None:
        try:
            candidates = {int(i) for i in raw_special if isinstance(i, int)}
        except TypeError:
            candidates = set()
        special_ids = candidates or None
    return no_speech_id, special_ids


def merge_confidence_blocks(blocks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Fold the blocks of one transcript's windows into one block.

    Entropies, token ids and token log-probabilities concatenate in window order, `avg_logprob` averages, and
    `no_speech_prob` takes the maximum.

    Args:
        blocks: Blocks as returned by `token_confidence_from_logits`.

    Returns:
        One block with the same keys.
    """
    entropies: list[float] = []
    logprobs: list[float] = []
    no_speech: list[float] = []
    tokens: list[int] = []
    token_logprob: list[float] = []
    for block in blocks:
        raw_entropy = block.get("token_entropy")
        if isinstance(raw_entropy, (list, tuple)):
            entropies.extend(float(e) for e in raw_entropy)
        elif raw_entropy is not None:
            entropies.append(float(raw_entropy))
        if block.get("avg_logprob") is not None:
            logprobs.append(float(block["avg_logprob"]))
        if block.get("no_speech_prob") is not None:
            no_speech.append(float(block["no_speech_prob"]))
        tokens.extend(int(t) for t in block.get("tokens") or [])
        token_logprob.extend(float(p) for p in block.get("token_logprob") or [])
    return {
        "token_entropy": entropies or None,
        "avg_logprob": (sum(logprobs) / len(logprobs)) if logprobs else None,
        "no_speech_prob": max(no_speech) if no_speech else None,
        "tokens": tokens or None,
        "token_logprob": token_logprob or None,
    }


def word_confidences(
    tokens: Sequence[int],
    token_logprob: Sequence[float],
    decode: Callable[[int], str],
    special_token_ids: Optional[set[int]] = None,
) -> list[float]:
    """Return one probability per word: the joint probability of the word's tokens, each given its decoded prefix.

    Control tokens and `special_token_ids` belong to no word. A token whose decoded text begins with a space starts
    a word; any other token continues the current one, so punctuation attaches to the word before it.

    Args:
        tokens: The scored token ids of one transcript, in order.
        token_logprob: The log-probability of each of `tokens`.
        decode: The text of one token id.
        special_token_ids: Ids that belong to no word.

    Returns:
        One value in `[0, 1]` per word, in order.

    Raises:
        ValueError: if `tokens` and `token_logprob` differ in length.
    """
    if len(tokens) != len(token_logprob):
        raise ValueError(f"{len(tokens)} tokens but {len(token_logprob)} log-probabilities")
    excluded = special_token_ids or set()
    sums: list[float] = []
    for token, logprob in zip(tokens, token_logprob):
        if token in excluded:
            continue
        text = decode(token)
        if _CONTROL_TOKEN.fullmatch(text.strip()):
            continue
        if text.startswith(" ") or not sums:
            sums.append(0.0)
        sums[-1] += logprob
    return [min(1.0, math.exp(total)) for total in sums]


def token_confidence_from_logits(
    *,
    logits: Sequence[torch.Tensor],
    sequences: torch.Tensor,
    no_speech_token_id: Optional[int] = None,
    special_token_ids: Optional[set[int]] = None,
) -> list[dict[str, Any]]:
    """Derive per-token entropy, log-probability and no-speech probability from raw logits.

    Args:
        logits: One tensor per decoding step, each `(batch, vocab)`, from `generate(..., output_logits=True)`.
        sequences: Generated token ids `(batch, seq_len)`; the trailing `len(logits)` ids are the scored ones.
        no_speech_token_id: Id of the no-speech token; `no_speech_prob` is read from the first step's distribution.
        special_token_ids: Ids excluded from `avg_logprob`; entropy and `token_logprob` cover every step.

    Returns:
        One dict per batch row: `token_entropy` (nats), `avg_logprob`, `no_speech_prob`, `tokens` (the scored ids)
        and `token_logprob` (each id's log-probability), each `None` when not available.
    """
    import torch  # noqa: PLC0415 - torch loads only where logits are scored

    if sequences.ndim == 1:
        sequences = sequences.unsqueeze(0)
    logits = [step.unsqueeze(0) if step.ndim == 1 else step for step in logits]

    batch_size = int(sequences.shape[0]) if sequences.ndim >= 1 else 1
    steps = len(logits)
    if steps == 0:
        return [dict.fromkeys(_BLOCK_KEYS) for _ in range(batch_size)]

    excluded = special_token_ids or set()
    no_speech: list[Optional[float]] = [None] * batch_size
    if no_speech_token_id is not None:
        first = logits[0].detach().float()
        if first.ndim == _LOGIT_DIMS and 0 <= no_speech_token_id < first.shape[-1]:
            probs = torch.softmax(first, dim=-1)[:, no_speech_token_id]
            no_speech = [float(p) for p in probs[:batch_size]]

    scored_ids = sequences[:, -steps:] if sequences.ndim == _LOGIT_DIMS and sequences.shape[1] >= steps else None

    entropies: list[list[float]] = [[] for _ in range(batch_size)]
    scored: list[list[int]] = [[] for _ in range(batch_size)]
    chosen: list[list[float]] = [[] for _ in range(batch_size)]

    for step, step_logits in enumerate(logits):
        lg = step_logits.detach().float()
        if lg.ndim != _LOGIT_DIMS:
            continue
        logprobs = torch.log_softmax(lg, dim=-1)
        step_entropy = -(logprobs.exp() * logprobs).sum(dim=-1)
        for row in range(min(batch_size, lg.shape[0])):
            entropies[row].append(float(step_entropy[row]))
            if scored_ids is None:
                continue
            token_id = int(scored_ids[row, step])
            scored[row].append(token_id)
            chosen[row].append(float(logprobs[row, token_id]))

    out: list[dict[str, Any]] = []
    for row in range(batch_size):
        content = [p for t, p in zip(scored[row], chosen[row]) if t not in excluded]
        out.append(
            {
                "token_entropy": entropies[row] or None,
                "avg_logprob": (sum(content) / len(content)) if content else None,
                "no_speech_prob": no_speech[row],
                "tokens": scored[row] or None,
                "token_logprob": chosen[row] or None,
            }
        )
    return out


@contextlib.contextmanager
def capture_token_confidence(
    pipe: Any,  # noqa: ANN401 - a transformers pipeline
    *,
    no_speech_token_id: Optional[int] = None,
    special_token_ids: Optional[set[int]] = None,
) -> Iterator[list[dict[str, Any]]]:
    """Wrap `pipe.model.generate` for the duration of the block to harvest per-token confidence.

    Yields a list that accumulates one block per generated sequence (one per transcript for a long-form call), in
    call order. The wrapper adds `output_logits` and `return_dict_in_generate` to every generate call, returns what
    the caller would have received without them, and is removed on exit. A backend that ignores those flags
    contributes nothing.

    Args:
        pipe: A transformers ASR pipeline exposing `.model.generate`.
        no_speech_token_id: Passed to `token_confidence_from_logits`.
        special_token_ids: Passed to `token_confidence_from_logits`.

    Yields:
        The accumulating list of blocks.
    """
    captured: list[dict[str, Any]] = []
    model = getattr(pipe, "model", None)
    if model is None or not callable(getattr(model, "generate", None)):
        yield captured
        return

    original = model.generate
    had_own_attr = "generate" in vars(model)

    def _score(step_logits: Any, sequences: Any) -> list[dict[str, Any]]:  # noqa: ANN401 - tensors
        return token_confidence_from_logits(
            logits=list(step_logits),
            sequences=sequences,
            no_speech_token_id=no_speech_token_id,
            special_token_ids=special_token_ids,
        )

    def _wrapped(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 - passthrough
        caller_wants_dict = bool(
            kwargs.get("return_dict_in_generate")
            or kwargs.get("return_token_timestamps")
            or kwargs.get("return_segments")
        )
        try:
            result = original(*args, **{**kwargs, "output_logits": True, "return_dict_in_generate": True})
        except TypeError:
            return original(*args, **kwargs)
        try:
            captured.extend(_harvest(result, _score))
        except (RuntimeError, ValueError, IndexError, TypeError, AttributeError, KeyError):
            pass
        return result if caller_wants_dict else _sequences_of(result)

    model.generate = _wrapped
    try:
        yield captured
    finally:
        if had_own_attr:
            model.generate = original
        else:
            with contextlib.suppress(AttributeError):
                del model.generate


def _harvest(result: Any, score: Callable[[Any, Any], list[dict[str, Any]]]) -> list[dict[str, Any]]:  # noqa: ANN401
    """Return the blocks of a generate output: its own, or one per batch item merged over that item's windows."""
    step_logits, sequences = _get(result, "logits"), _get(result, "sequences")
    if step_logits is not None and sequences is not None:
        return score(step_logits, sequences)
    merged: list[dict[str, Any]] = []
    for item in _get(result, "segments") or []:
        blocks: list[dict[str, Any]] = []
        for window in item or []:
            inner = window.get("result") if isinstance(window, dict) else None
            if inner is not None and _get(inner, "logits") is not None and _get(inner, "sequences") is not None:
                blocks.extend(score(_get(inner, "logits"), _get(inner, "sequences")))
        if blocks:
            merged.append(merge_confidence_blocks(blocks))
    return merged


def _sequences_of(result: Any) -> Any:  # noqa: ANN401 - a ModelOutput
    """Return the bare token tensor the caller would have seen without the added flags, else `result`."""
    import torch  # noqa: PLC0415 - torch loads only where logits are scored

    sequences = _get(result, "sequences")
    return sequences if isinstance(sequences, torch.Tensor) else result


def _get(obj: Any, key: str) -> Any:  # noqa: ANN401 - ModelOutput is dict-like and attr-like
    """Return `key` of a transformers `ModelOutput`, as an attribute or a mapping entry."""
    value = getattr(obj, key, None)
    if value is None and isinstance(obj, dict):
        value = obj.get(key)
    return value
