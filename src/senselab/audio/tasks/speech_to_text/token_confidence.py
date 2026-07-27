"""Per-token ASR confidence extraction (feature 20260722-175022, FR-017).

Whisper's decoder knows more about its own uncertainty than the transcript text
reveals: the per-step distribution over the vocabulary is peaked when the model is
sure and flat when it is guessing. That signal is what the utterance axis needs —
pairwise transcript disagreement only fires when two backends actually differ,
while token entropy registers a single model's private doubt.

Why a capture seam rather than ``generate_kwargs``:
    Transformers' ``AutomaticSpeechRecognitionPipeline._forward`` keeps only
    ``sequences`` and ``token_timestamps`` from the generate output and drops the
    scores/logits before ``postprocess`` ever sees them. Passing
    ``output_scores=True`` through ``pipe(...)`` therefore has no observable
    effect. :func:`capture_token_confidence` instead wraps ``pipe.model.generate``
    for the duration of one call, reads the logits off the returned
    ``ModelOutput``, and restores the original method afterwards. The pipeline's
    own behavior is untouched — we only observe.

We request ``output_logits`` (raw, pre-processor) rather than ``output_scores``
(post-processor) on purpose: Whisper's logits processors suppress
``<|nospeech|>`` to ``-inf``, which would drive the no-speech probability to a
constant 0, and they distort the entropy of the remaining distribution.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from typing import Any, Optional

import torch

__all__ = [
    "capture_token_confidence",
    "merge_confidence_blocks",
    "token_confidence_from_logits",
    "whisper_token_ids",
]

_NO_SPEECH_TOKEN_SPELLINGS = ("<|nospeech|>", "<|nocaptions|>")
"""Vocabulary spellings of Whisper's no-speech marker, in preference order.

The spelling is not stable across releases: ``openai/whisper-tiny`` ships
``<|nocaptions|>`` (id 50362) and maps ``<|nospeech|>`` to the *unk* id, while other
checkpoints do the reverse. Both are tried, and any hit equal to ``unk_token_id`` is
rejected — otherwise we'd read the probability of an unrelated token.
"""


def whisper_token_ids(pipe: Any) -> tuple[Optional[int], Optional[set]]:  # noqa: ANN401 — a transformers pipeline
    """Resolve ``(no_speech_token_id, special_token_ids)`` for a pipeline's tokenizer.

    Returns ``(None, None)`` for backends without a tokenizer (e.g. raw CTC models),
    which is what makes the confidence fields degrade gracefully (FR-017).

    Args:
        pipe: A transformers pipeline, or any object exposing ``tokenizer`` and
            ``model.generation_config``.

    Returns:
        The ``<|nospeech|>`` vocabulary id (or ``None`` if this backend has none) and
        the set of special-token ids to exclude from ``avg_logprob`` (or ``None``).
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

    # Guard the conversion: a stand-in/Mock tokenizer returns a non-iterable
    # attribute here, and blowing up would fail the whole transcription over a
    # purely additive signal.
    special_ids: Optional[set] = None
    raw_special = getattr(tokenizer, "all_special_ids", None)
    if raw_special is not None:
        try:
            candidates = {int(i) for i in raw_special if isinstance(i, int)}
        except TypeError:
            candidates = set()
        special_ids = candidates or None
    return no_speech_id, special_ids


def merge_confidence_blocks(blocks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Fold several confidence blocks for the *same* transcript into one.

    Used for Whisper long-form, where one transcript is produced by several
    ``generate`` calls (one per 30 s window) and each window yields its own block.
    Entropies concatenate (they're a per-token sequence), log-probabilities average,
    and the no-speech probability takes the maximum so the most silence-like window
    dominates rather than being averaged away.

    Args:
        blocks: Confidence blocks as returned by :func:`token_confidence_from_logits`.

    Returns:
        A single merged block with the same three keys.
    """
    entropies: list[float] = []
    logprobs: list[float] = []
    no_speech: list[float] = []
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
    return {
        "token_entropy": entropies or None,
        "avg_logprob": (sum(logprobs) / len(logprobs)) if logprobs else None,
        "no_speech_prob": max(no_speech) if no_speech else None,
    }


def token_confidence_from_logits(
    *,
    logits: Sequence[torch.Tensor],
    sequences: torch.Tensor,
    no_speech_token_id: Optional[int] = None,
    special_token_ids: Optional[set[int]] = None,
) -> list[dict[str, Any]]:
    """Derive per-token entropy, ``avg_logprob`` and ``no_speech_prob`` from logits.

    Args:
        logits: One tensor per decoding step, each shaped ``(batch, vocab)``, as
            returned by ``generate(..., output_logits=True)``. Raw (unprocessed)
            logits are expected.
        sequences: Generated token ids, shaped ``(batch, seq_len)``. For
            encoder-decoder models ``seq_len`` exceeds ``len(logits)`` because it
            carries the forced decoder prefix; the trailing ``len(logits)`` ids are
            the scored ones, matching HF's transition-score alignment.
        no_speech_token_id: Vocabulary id of Whisper's ``<|nospeech|>`` token. When
            given, ``no_speech_prob`` is read from the first step's distribution —
            Whisper's own definition. ``None`` for backends without the token.
        special_token_ids: Ids excluded from ``avg_logprob`` (forced language /
            task / timestamp markers). Entropy is still reported for every step.

    Returns:
        One dict per batch row with keys ``token_entropy`` (list of nats, or
        ``None``), ``avg_logprob`` (float or ``None``) and ``no_speech_prob``
        (float or ``None``).

    Example:
        >>> import torch
        >>> flat = torch.zeros(1, 4)  # uniform over 4 tokens
        >>> out = token_confidence_from_logits(logits=[flat], sequences=torch.tensor([[2]]))
        >>> round(out[0]["token_entropy"][0], 4) == round(float(torch.log(torch.tensor(4.0))), 4)
        True
    """
    # Whisper's long-form path hands back unbatched per-segment tensors —
    # logits ``(vocab,)`` and sequences ``(seq_len,)``. Normalize to the batched
    # shapes so one code path serves both. Verified against transformers 5.5.4.
    if sequences.ndim == 1:
        sequences = sequences.unsqueeze(0)
    logits = [step.unsqueeze(0) if step.ndim == 1 else step for step in logits]

    batch_size = int(sequences.shape[0]) if sequences.ndim >= 1 else 1
    steps = len(logits)
    if steps == 0:
        return [{"token_entropy": None, "avg_logprob": None, "no_speech_prob": None} for _ in range(batch_size)]

    excluded = special_token_ids or set()

    # Whisper reads the no-speech probability off the first generated position.
    no_speech: list[Optional[float]] = [None] * batch_size
    if no_speech_token_id is not None:
        first = logits[0].detach().float()
        if first.ndim == 2 and 0 <= no_speech_token_id < first.shape[-1]:
            probs = torch.softmax(first, dim=-1)[:, no_speech_token_id]
            no_speech = [float(p) for p in probs[:batch_size]]

    # Align the scored ids to the recorded steps (drop any forced prefix).
    scored_ids = sequences[:, -steps:] if sequences.ndim == 2 and sequences.shape[1] >= steps else None

    entropies: list[list[float]] = [[] for _ in range(batch_size)]
    chosen_logprobs: list[list[float]] = [[] for _ in range(batch_size)]

    for step, step_logits in enumerate(logits):
        lg = step_logits.detach().float()
        if lg.ndim != 2:
            continue
        logprobs = torch.log_softmax(lg, dim=-1)
        # -Σ p·log p, computed from logprobs to avoid a second softmax.
        step_entropy = -(logprobs.exp() * logprobs).sum(dim=-1)
        for row in range(min(batch_size, lg.shape[0])):
            entropies[row].append(float(step_entropy[row]))
            if scored_ids is None:
                continue
            token_id = int(scored_ids[row, step])
            if token_id in excluded:
                continue
            chosen_logprobs[row].append(float(logprobs[row, token_id]))

    out: list[dict[str, Any]] = []
    for row in range(batch_size):
        row_logprobs = chosen_logprobs[row]
        out.append(
            {
                "token_entropy": entropies[row] or None,
                "avg_logprob": (sum(row_logprobs) / len(row_logprobs)) if row_logprobs else None,
                "no_speech_prob": no_speech[row],
            }
        )
    return out


@contextlib.contextmanager
def capture_token_confidence(
    pipe: Any,  # noqa: ANN401 — a transformers pipeline; typed loosely to avoid a hard import
    *,
    no_speech_token_id: Optional[int] = None,
    special_token_ids: Optional[set[int]] = None,
) -> Iterator[list[dict[str, Any]]]:
    """Temporarily wrap ``pipe.model.generate`` to harvest per-token confidence.

    Yields a list that accumulates one confidence dict per generated sequence, in
    call order. The wrapper adds ``output_logits`` / ``return_dict_in_generate``
    to every generate call and is removed on exit, leaving the pipeline exactly as
    it was found. Backends that ignore those flags (returning a bare tensor)
    simply contribute nothing — the caller degrades to ``None`` fields.

    Args:
        pipe: A transformers ASR pipeline exposing ``.model.generate``.
        no_speech_token_id: Passed through to :func:`token_confidence_from_logits`.
        special_token_ids: Passed through to :func:`token_confidence_from_logits`.

    Yields:
        The accumulating list of per-sequence confidence dicts.
    """
    captured: list[dict[str, Any]] = []
    model = getattr(pipe, "model", None)
    if model is None or not callable(getattr(model, "generate", None)):
        yield captured
        return

    original = model.generate
    had_own_attr = "generate" in vars(model)

    def _score(step_logits: Any, sequences: Any) -> list[dict[str, Any]]:  # noqa: ANN401 — tensors
        return token_confidence_from_logits(
            logits=list(step_logits),
            sequences=sequences,
            no_speech_token_id=no_speech_token_id,
            special_token_ids=special_token_ids,
        )

    def _wrapped(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 — passthrough
        # Whether the *caller* wanted a dict back. Whisper's word-timestamp path
        # does (the pipeline sets return_token_timestamps / return_segments);
        # plain greedy decoding does not, and handing those callers a ModelOutput
        # where they expect a tensor breaks them downstream
        # ("'ModelOutput' object has no attribute 'dtype'").
        caller_wants_dict = bool(
            kwargs.get("return_dict_in_generate")
            or kwargs.get("return_token_timestamps")
            or kwargs.get("return_segments")
        )
        try:
            result = original(*args, **{**kwargs, "output_logits": True, "return_dict_in_generate": True})
        except TypeError:
            # Backend's generate doesn't accept these kwargs — run it untouched.
            return original(*args, **kwargs)
        try:
            step_logits = _get(result, "logits")
            sequences = _get(result, "sequences")
            if step_logits is not None and sequences is not None:
                captured.extend(_score(step_logits, sequences))
            else:
                # Whisper long-form (the `return_timestamps="word"` default) reports
                # nothing at the top level; each window's generate output lives at
                # segments[batch][window]["result"]. Merge a batch item's windows into
                # one block so the count still matches one-per-transcript.
                segments = _get(result, "segments")
                for item in segments or []:
                    blocks: list[dict[str, Any]] = []
                    for window in item or []:
                        inner = window.get("result") if isinstance(window, dict) else None
                        if inner is None:
                            continue
                        inner_logits = _get(inner, "logits")
                        inner_sequences = _get(inner, "sequences")
                        if inner_logits is None or inner_sequences is None:
                            continue
                        blocks.extend(_score(inner_logits, inner_sequences))
                    if blocks:
                        captured.append(merge_confidence_blocks(blocks))
        except (RuntimeError, ValueError, IndexError, TypeError, AttributeError, KeyError):
            # Confidence is strictly additive — never fail a transcription because
            # the logits came back in an unexpected shape.
            pass

        # Restore the return type the caller would have seen without our flags, so
        # observing the decoder stays invisible to the pipeline.
        if not caller_wants_dict:
            sequences = _get(result, "sequences")
            if isinstance(sequences, torch.Tensor):
                return sequences
        return result

    model.generate = _wrapped  # type: ignore[method-assign]
    try:
        yield captured
    finally:
        if had_own_attr:
            model.generate = original  # type: ignore[method-assign]
        else:
            with contextlib.suppress(AttributeError):
                del model.generate


def _get(obj: Any, key: str) -> Any:  # noqa: ANN401 — ModelOutput is dict-like *and* attr-like
    """Read ``key`` off a transformers ``ModelOutput`` (attribute or mapping)."""
    value = getattr(obj, key, None)
    if value is None and isinstance(obj, dict):
        value = obj.get(key)
    return value
