"""Per-token ASR confidence: the log-probability of each decoded token, read off the decoder's own logits.

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
    "special_token_ids",
    "word_confidences",
    "word_prefix",
    "word_texts",
]

_CONTROL_TOKEN = re.compile(r"<\|.*\|>")
"""A control token's text: language, task and timestamp markers, which no word contains."""
_LOGIT_DIMS = 2
_BLOCK_KEYS = ("avg_logprob", "tokens", "token_logprob")


def special_token_ids(pipe: Any) -> Optional[set[int]]:  # noqa: ANN401 - a transformers pipeline
    """Return the special-token ids of a pipeline's tokenizer, excluded from `avg_logprob`; None without one."""
    raw = getattr(getattr(pipe, "tokenizer", None), "all_special_ids", None)
    if raw is None:
        return None
    try:
        return {int(i) for i in raw if isinstance(i, int)} or None
    except TypeError:
        return None


def merge_confidence_blocks(blocks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Fold the blocks of one transcript's windows into one block.

    Token ids and token log-probabilities concatenate in window order and `avg_logprob` averages.

    Args:
        blocks: Blocks as returned by `token_confidence_from_logits`.

    Returns:
        One block with the same keys.
    """
    logprobs = [float(b["avg_logprob"]) for b in blocks if b.get("avg_logprob") is not None]
    tokens = [int(t) for b in blocks for t in b.get("tokens") or []]
    token_logprob = [float(p) for b in blocks for p in b.get("token_logprob") or []]
    return {
        "avg_logprob": (sum(logprobs) / len(logprobs)) if logprobs else None,
        "tokens": tokens or None,
        "token_logprob": token_logprob or None,
    }


def _word_tokens(tokens: Sequence[int], decode: Callable[[int], str], excluded: set[int]) -> Iterator[tuple[int, bool]]:
    """Yield `(index, starts_a_word)` of each token that belongs to a word."""
    started = False
    for index, token in enumerate(tokens):
        if token in excluded:
            continue
        text = decode(token)
        if _CONTROL_TOKEN.fullmatch(text.strip()):
            continue
        yield index, text.startswith(" ") or not started
        started = True


def word_texts(
    tokens: Sequence[int], decode: Callable[[int], str], special_token_ids: Optional[set[int]] = None
) -> list[str]:
    """Return the text of each word `word_confidences` groups `tokens` into."""
    words: list[str] = []
    for index, starts in _word_tokens(tokens, decode, special_token_ids or set()):
        if starts:
            words.append("")
        words[-1] += decode(tokens[index])
    return words


def word_prefix(
    tokens: Sequence[int],
    words: int,
    decode: Callable[[int], str],
    special_token_ids: Optional[set[int]] = None,
) -> int:
    """Return how many leading `tokens` make up the first `words` words, by the rule `word_confidences` groups by.

    Tokens that belong to no word after the last kept word's own tokens are not part of the prefix; `len(tokens)`
    when there are no more than `words` words.
    """
    count = 0
    for index, starts in _word_tokens(tokens, decode, special_token_ids or set()):
        count += starts
        if count > words:
            return index
    return len(tokens)


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
    sums: list[float] = []
    for index, starts in _word_tokens(tokens, decode, special_token_ids or set()):
        if starts:
            sums.append(0.0)
        sums[-1] += token_logprob[index]
    return [min(1.0, math.exp(total)) for total in sums]


def token_confidence_from_logits(
    *,
    logits: Sequence[torch.Tensor],
    sequences: torch.Tensor,
    special_token_ids: Optional[set[int]] = None,
) -> list[dict[str, Any]]:
    """Derive each scored token's log-probability from raw logits.

    Args:
        logits: One tensor per decoding step, each `(batch, vocab)`, from `generate(..., output_logits=True)`.
        sequences: Generated token ids `(batch, seq_len)`; the trailing `len(logits)` ids are the scored ones.
        special_token_ids: Ids excluded from `avg_logprob`; `tokens` and `token_logprob` cover every step.

    Returns:
        One dict per batch row: `avg_logprob`, `tokens` (the scored ids) and `token_logprob` (each id's
        log-probability), each `None` when not available.
    """
    import torch  # noqa: PLC0415 - torch loads only where logits are scored

    if sequences.ndim == 1:
        sequences = sequences.unsqueeze(0)
    logits = [step.unsqueeze(0) if step.ndim == 1 else step for step in logits]
    batch_size = int(sequences.shape[0]) if sequences.ndim >= 1 else 1
    steps = len(logits)
    if steps == 0 or sequences.ndim != _LOGIT_DIMS or sequences.shape[1] < steps:
        return [dict.fromkeys(_BLOCK_KEYS) for _ in range(batch_size)]

    excluded = special_token_ids or set()
    scored_ids = sequences[:, -steps:]
    scored: list[list[int]] = [[] for _ in range(batch_size)]
    chosen: list[list[float]] = [[] for _ in range(batch_size)]
    for step, step_logits in enumerate(logits):
        lg = step_logits.detach().float()
        if lg.ndim != _LOGIT_DIMS:
            continue
        logprobs = torch.log_softmax(lg, dim=-1)
        for row in range(min(batch_size, lg.shape[0])):
            token_id = int(scored_ids[row, step])
            scored[row].append(token_id)
            chosen[row].append(float(logprobs[row, token_id]))

    out: list[dict[str, Any]] = []
    for row in range(batch_size):
        content = [p for t, p in zip(scored[row], chosen[row]) if t not in excluded]
        out.append(
            {
                "avg_logprob": (sum(content) / len(content)) if content else None,
                "tokens": scored[row] or None,
                "token_logprob": chosen[row] or None,
            }
        )
    return out


@contextlib.contextmanager
def capture_token_confidence(
    pipe: Any,  # noqa: ANN401 - a transformers pipeline
    *,
    special_token_ids: Optional[set[int]] = None,
) -> Iterator[list[dict[str, Any]]]:
    """Wrap `pipe.model.generate` for the duration of the block to harvest per-token confidence.

    Yields a list that accumulates one block per generated sequence (one per transcript for a long-form call), in
    call order. The wrapper adds `output_logits` and `return_dict_in_generate` to every generate call, returns what
    the caller would have received without them, and is removed on exit. A backend that ignores those flags
    contributes nothing.

    Args:
        pipe: A transformers ASR pipeline exposing `.model.generate`.
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
