"""Per-task harvest helpers used by the three uncertainty axes.

These read the in-memory result objects produced by senselab's audio task pipeline
(diarization, ASR, scene classification, PPG, alignment) and project them onto a bucket
boundary so the per-axis vote harvesters can build their dicts. The functions here are
shape-tolerant: they work with both Pydantic models (in-memory) and the dict shape that
JSON-cache deserialization produces.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from senselab.audio.tasks.classification.label_scores import label_scores

if TYPE_CHECKING:
    import numpy as np


def seg_attr(seg: Any, name: str) -> Any:  # noqa: ANN401
    """Return ``seg.name`` whether ``seg`` is a Pydantic model or a JSON dict.

    Cache reads deserialize ScriptLine into plain dicts; in-memory results are Pydantic
    objects. Both shapes flow through the harvesters.
    """
    if isinstance(seg, dict):
        return seg.get(name)
    return getattr(seg, name, None)


# ── Diarization ───────────────────────────────────────────────────────


def _union_length(spans: list[tuple[float, float]]) -> float:
    """Total length covered by ``spans``, counting overlaps once.

    Shared by the diarization and transcript coverage measures because both answer "how much of
    this bucket is claimed", and summing instead of unioning would let two simultaneous speakers —
    or two aligners' word spans — report more than a bucket's worth.
    """
    if not spans:
        return 0.0
    ordered = sorted(spans)
    covered = 0.0
    cur_lo, cur_hi = ordered[0]
    for lo, hi in ordered[1:]:
        if lo > cur_hi:
            covered += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
        else:
            cur_hi = max(cur_hi, hi)
    return covered + (cur_hi - cur_lo)


def diar_covered_fraction(result: Any, win_start: float, win_end: float) -> float | None:  # noqa: ANN401
    """Fraction of ``[win_start, win_end)`` covered by any diarization segment.

    Replaces the ``speaks`` bool this used to reduce to. A segment overlapping 5% of a bucket and
    one covering all of it are not the same evidence, and a bool cannot tell them apart — which
    matters most at segment boundaries, exactly where speaker uncertainty is highest.

    Returns:
        Coverage in ``[0, 1]``, or ``None`` when the window is empty or the model produced no
        segments — an absent model is not a model reporting zero coverage.
    """
    if not result:
        return None
    span = float(win_end) - float(win_start)
    if span <= 0:
        return None
    segments = result[0] if isinstance(result, list) and result else []
    if not segments:
        return None
    # Union of overlaps, not their sum: overlapping segments from two speakers must not report
    # more than a bucket's worth of coverage.
    spans: list[tuple[float, float]] = []
    for seg in segments:
        s_attr = seg_attr(seg, "start")
        e_attr = seg_attr(seg, "end")
        if s_attr is None or e_attr is None:
            continue
        lo = max(float(win_start), float(s_attr))
        hi = min(float(win_end), float(e_attr))
        if hi > lo:
            spans.append((lo, hi))
    if not spans:
        return 0.0
    return max(0.0, min(1.0, _union_length(spans) / span))


def diar_speaker_label_in_window(result: Any, win_start: float, win_end: float) -> str | None:  # noqa: ANN401
    """Return the diarization speaker label whose segment overlaps the window most.

    When multiple segments overlap, the one with the largest temporal overlap wins.
    Equal-overlap ties are broken deterministically by the lexicographic order
    of the speaker label so different diar models (pyannote vs Sortformer)
    produce a consistent label per bucket regardless of segment-list iteration
    order. Returns None when no segment overlaps.
    """
    if not result:
        return None
    segments = result[0] if isinstance(result, list) and result else []
    best_overlap = 0.0
    best_label: str | None = None
    for seg in segments:
        s = seg_attr(seg, "start")
        e = seg_attr(seg, "end")
        if s is None or e is None:
            continue
        s_f = float(s)
        e_f = float(e)
        if s_f >= win_end or e_f <= win_start:
            continue
        overlap = min(e_f, win_end) - max(s_f, win_start)
        label = seg_attr(seg, "speaker") or "SPEAKER_UNKNOWN"
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = label
        elif overlap == best_overlap and best_label is not None and label < best_label:
            best_label = label
    return best_label


# ── ASR ───────────────────────────────────────────────────────────────


def asr_has_timestamps(result: Any) -> bool:  # noqa: ANN401
    """True if any ScriptLine actually carries a timestamp.

    Requires a non-null ``start`` on a chunk or on the line itself. The mere
    *speech_presence* of chunks is not evidence of timing: a chunked-but-untimed
    transcript has no usable times, and treating it as timestamped would make the
    alignment stage skip exactly the input it exists to fix. ``analyze_audio.py``
    carried a looser duplicate that returned True whenever ``chunks`` was
    non-empty — it disagreed with ``resolve_asr_result`` below, which has always
    used these strict semantics. Consolidated here (T051b).
    """
    if not result:
        return False
    items = result if isinstance(result, list) else [result]
    for line in items:
        chunks = seg_attr(line, "chunks") or []
        if chunks:
            for c in chunks:
                if seg_attr(c, "start") is not None:
                    return True
        if seg_attr(line, "start") is not None:
            return True
    return False


def resolve_asr_result(asr_block: dict[str, Any], align_block: dict[str, Any] | None) -> Any:  # noqa: ANN401
    """Return the ASR result that carries usable timestamps.

    For text-only ASR backends (Granite, Canary-Qwen) without per-token chunks, falls
    through to the post-MMS alignment block per FR-011. Without alignment, text without
    a time anchor produces no token overlap (asr_says_speech = false).
    """
    if not isinstance(asr_block, dict):
        return asr_block
    asr_res = asr_block.get("result")
    if asr_has_timestamps(asr_res):
        return asr_res
    if isinstance(align_block, dict) and align_block.get("status") == "ok":
        ar = align_block.get("result")
        if isinstance(ar, list) and ar and isinstance(ar[0], list):
            return ar[0]
        return ar
    return asr_res


def asr_bucket_chunk_evidence(result: Any, win_start: float, win_end: float) -> dict[str, Any]:  # noqa: ANN401
    """What an ASR model reported over one bucket, in its own units and unpooled.

    Replaces the three belief-producing readers this call site used to need
    (``token_overlaps_window``, ``whisper_bucket_confidence``, ``whisper_bucket_no_speech_prob``),
    each of which reduced before returning. In particular the confidence reader pooled per-chunk
    log-probabilities as ``mean(exp(x))``, which by Jensen's inequality is not the same statistic as
    ``exp(mean(x))`` — a choice worth making explicitly, at L2, rather than inside a getter.

    Returns:
        ``{"word_overlap_s", "n_words", "avg_logprobs", "no_speech_probs", "claim_span_s",
        "segment_span_s"}``. Coverage is a union over word spans clipped to the bucket, so
        overlapping spans cannot exceed its width. The two lists hold one entry per contributing
        chunk, in the model's own log / probability domains.

        The two span fields are **unclipped** unions, and exist so the coarseness of the claim is
        measured rather than declared: ``claim_span_s`` is how wide the transcript evidence
        reaching this bucket actually is, and ``segment_span_s`` the same for the segments whose
        scalars were pooled. A voter's window is only "coarse" relative to the grid it is reported
        on, so L2 needs the number rather than a hand-set flag.

    Scalars fall back to the segment level when a line carries no chunk overlapping the bucket
    (post-aligned text-only ASR exposes them only per segment), while coverage does not — a line
    whose chunks all fall outside the bucket has placed no word inside it, whatever its own span
    says.
    """
    spans: list[tuple[float, float]] = []
    claim_spans: list[tuple[float, float]] = []
    segment_spans: list[tuple[float, float]] = []
    n_words = 0
    avg_logprobs: list[float] = []
    no_speech_probs: list[float] = []
    empty: dict[str, Any] = {
        "word_overlap_s": 0.0,
        "n_words": 0,
        "avg_logprobs": [],
        "no_speech_probs": [],
        "claim_span_s": None,
        "segment_span_s": None,
    }
    if not result:
        return empty

    items = result if isinstance(result, list) else [result]
    for line in items:
        chunks = seg_attr(line, "chunks") or []
        chunk_seen_any = False
        for c in chunks:
            cs = seg_attr(c, "start")
            ce = seg_attr(c, "end")
            if cs is None or ce is None:
                continue
            if float(cs) < win_end and float(ce) > win_start:
                chunk_seen_any = True
                n_words += 1
                spans.append((max(float(win_start), float(cs)), min(float(win_end), float(ce))))
                claim_spans.append((float(cs), float(ce)))
                _collect_chunk_scalars(c, avg_logprobs, no_speech_probs)
        ls = seg_attr(line, "start")
        le = seg_attr(line, "end")
        if chunk_seen_any:
            if ls is not None and le is not None:
                segment_spans.append((float(ls), float(le)))
            continue
        if ls is None or le is None:
            # No timestamps at all: the scalars still describe this bucket as well as any other,
            # which is what the aligner exists to fix. No coverage and no span are claimed.
            _collect_chunk_scalars(line, avg_logprobs, no_speech_probs)
            continue
        if float(ls) < win_end and float(le) > win_start:
            _collect_chunk_scalars(line, avg_logprobs, no_speech_probs)
            segment_spans.append((float(ls), float(le)))
            if not chunks:
                n_words += 1
                spans.append((max(float(win_start), float(ls)), min(float(win_end), float(le))))
                claim_spans.append((float(ls), float(le)))
    return {
        "word_overlap_s": _union_length(spans),
        "n_words": n_words,
        "avg_logprobs": avg_logprobs,
        "no_speech_probs": no_speech_probs,
        "claim_span_s": _union_length(claim_spans) if claim_spans else None,
        "segment_span_s": _union_length(segment_spans) if segment_spans else None,
    }


def _collect_chunk_scalars(item: Any, avg_logprobs: list[float], no_speech_probs: list[float]) -> None:  # noqa: ANN401
    """Append a chunk's or segment's native scalars, skipping fields it does not expose."""
    avg = seg_attr(item, "avg_logprob")
    if avg is not None:
        try:
            avg_logprobs.append(float(avg))
        except (TypeError, ValueError):
            pass
    nsp = seg_attr(item, "no_speech_prob")
    if nsp is not None:
        try:
            no_speech_probs.append(float(nsp))
        except (TypeError, ValueError):
            pass


def asr_text_in_window(
    result: Any,  # noqa: ANN401
    win_start: float,
    win_end: float,
    *,
    fully_contained: bool = False,
) -> str:
    """Concatenated transcript tokens within ``[win_start, win_end)``.

    Args:
        result: Resolved ASR result (raw ScriptLines if natively timestamped, otherwise
            the post-MMS alignment block — see ``resolve_asr_result``).
        win_start: Window start time in seconds.
        win_end: Window end time in seconds.
        fully_contained: When ``True``, only include chunks whose ``[start, end]`` lies
            entirely within ``[win_start, win_end)``. The default ``False`` keeps the
            traditional overlap rule (chunk crosses into the window). Used by the
            asr axis (with True) so partial words straddling a window boundary
            don't pollute the WER score on either side.
    """
    if not result:
        return ""
    items = result if isinstance(result, list) else [result]
    pieces: list[str] = []

    def _included(cs: float, ce: float) -> bool:
        if fully_contained:
            return cs >= win_start and ce <= win_end
        return cs < win_end and ce > win_start

    def _walk(node: Any) -> None:  # noqa: ANN401
        # Recurse into ``.chunks`` until we hit a leaf (no inner chunks). Post-
        # MMS-aligned text-only ASR (Granite, Canary) emits a 3-level nesting:
        # outer line → asr ScriptLine → word ScriptLines. Whisper / Qwen
        # are 2-level (line → words). The leaf is what we want to bucket on.
        chunks = seg_attr(node, "chunks") or []
        if chunks:
            for c in chunks:
                _walk(c)
            return
        cs = seg_attr(node, "start")
        ce = seg_attr(node, "end")
        text = seg_attr(node, "text") or ""
        if not text:
            return
        if cs is None or ce is None:
            # No time anchor — include only when overlap rules say so (else drop).
            if not fully_contained:
                pieces.append(str(text).strip())
            return
        if _included(float(cs), float(ce)):
            pieces.append(str(text).strip())

    for line in items:
        _walk(line)
    return " ".join(p for p in pieces if p).strip()


# ── Whisper-style native confidence ───────────────────────────────────


def asr_alignment_score_in_window(
    result: Any,  # noqa: ANN401
    win_start: float,
    win_end: float,
) -> float | None:
    """Mean MMS-CTC posterior score across alignment leaf chunks overlapping the window.

    The forced-alignment dict carries a ``score`` field at every level (char →
    word → sentence → line) — the mean per-frame CTC posterior probability the
    Wav2Vec2-CTC model assigned to that token's path through its trellis. We
    aggregate at the leaf (character) level and average over leaves whose
    timestamps overlap ``[win_start, win_end)``.

    Returns the mean score in ``[0, 1]`` (higher = more confident) or ``None``
    when no alignment leaf overlaps the bucket.
    """
    if not result:
        return None
    items = result if isinstance(result, list) else [result]
    scores: list[float] = []

    def _walk(node: Any) -> None:  # noqa: ANN401
        chunks = seg_attr(node, "chunks") or []
        if chunks:
            for c in chunks:
                _walk(c)
            return
        cs = seg_attr(node, "start")
        ce = seg_attr(node, "end")
        if cs is None or ce is None:
            return
        # Overlap rule.
        if not (float(cs) < win_end and float(ce) > win_start):
            return
        s = seg_attr(node, "score")
        if s is None:
            return
        try:
            scores.append(float(s))
        except (TypeError, ValueError):
            return

    for line in items:
        _walk(line)
    if not scores:
        return None
    return sum(scores) / len(scores)


def whisper_chunk_confidence(chunk: Any) -> tuple[float | None, float | None]:  # noqa: ANN401
    """Return (confidence, no_speech_prob) from a Whisper chunk dict / ScriptLine.

    confidence = exp(avg_logprob) clipped to [0, 1]. Returns (None, None) when the
    chunk exposes no native scalar.
    """
    avg = seg_attr(chunk, "avg_logprob")
    nsp = seg_attr(chunk, "no_speech_prob")
    confidence: float | None = None
    if avg is not None:
        try:
            confidence = max(0.0, min(1.0, float(math.exp(float(avg)))))
        except (ValueError, OverflowError):
            confidence = None
    no_speech = float(nsp) if nsp is not None else None
    return confidence, no_speech


def whisper_bucket_avg_logprob(result: Any, win_start: float, win_end: float) -> float | None:  # noqa: ANN401
    """Mean of raw per-chunk ``avg_logprob`` over chunks overlapping the window.

    Returns the arithmetic mean of negative logprobs — equivalent to the geometric
    mean of per-chunk confidences when later exponentiated. This is the unbiased
    way to aggregate Whisper's native logprob to a bucket scale; the asr
    aggregator computes ``1 − exp(avg_logprob)`` once on the bucket value.
    """
    if not result:
        return None
    items = result if isinstance(result, list) else [result]
    logprobs: list[float] = []
    for line in items:
        chunks = seg_attr(line, "chunks") or []
        chunk_seen_any = False
        for c in chunks:
            cs = seg_attr(c, "start")
            ce = seg_attr(c, "end")
            if cs is None or ce is None:
                continue
            if float(cs) < win_end and float(ce) > win_start:
                chunk_seen_any = True
                avg = seg_attr(c, "avg_logprob")
                if avg is not None:
                    try:
                        logprobs.append(float(avg))
                    except (TypeError, ValueError):
                        continue
        if not chunk_seen_any:
            ls = seg_attr(line, "start")
            le = seg_attr(line, "end")
            if ls is None or le is None or (float(ls) < win_end and float(le) > win_start):
                avg = seg_attr(line, "avg_logprob")
                if avg is not None:
                    try:
                        logprobs.append(float(avg))
                    except (TypeError, ValueError):
                        continue
    if not logprobs:
        return None
    return sum(logprobs) / len(logprobs)


def _collapse_token_entropy(raw: Any) -> float | None:  # noqa: ANN401 — float | list | junk
    """Collapse a ``token_entropy`` field to a scalar mean, or ``None`` if unusable."""
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        values: list[float] = []
        for item in raw:
            try:
                values.append(float(item))
            except (TypeError, ValueError):
                continue
        return sum(values) / len(values) if values else None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def mean_token_entropy_in_window(result: Any, win_start: float, win_end: float) -> float | None:  # noqa: ANN401
    """Mean per-token softmax entropy (nats) attributable to ``[win_start, win_end)``.

    Two shapes are handled, in priority order:

    1. **Word-level** — when transcript chunks carry their own ``token_entropy``,
       each word is assigned to exactly one bucket by its timestamp *midpoint*
       (the contract's rule), so a word is never counted twice across
       overlapping buckets.
    2. **Line-level** — the Whisper capture seam attaches one entropy sequence per
       generated sequence, with no per-token timestamps to distribute it by. Those
       lines contribute their whole-sequence mean to every bucket they overlap.

    Word-level wins when present so a coarse line value can't double-count against
    the finer per-word evidence.

    Args:
        result: One ``ScriptLine``-like object or a list of them (Pydantic objects or
            cache-deserialized dicts).
        win_start: Bucket start in seconds (inclusive).
        win_end: Bucket end in seconds (exclusive).

    Returns:
        The mean entropy in nats, or ``None`` when no contributing source reported
        token entropy for this window.
    """
    if not result:
        return None
    items = result if isinstance(result, list) else [result]

    word_values: list[float] = []
    line_values: list[float] = []
    for line in items:
        for chunk in seg_attr(line, "chunks") or []:
            value = _collapse_token_entropy(seg_attr(chunk, "token_entropy"))
            if value is None:
                continue
            cs = seg_attr(chunk, "start")
            ce = seg_attr(chunk, "end")
            if cs is None or ce is None:
                continue
            midpoint = (float(cs) + float(ce)) / 2.0
            if win_start <= midpoint < win_end:
                word_values.append(value)

        value = _collapse_token_entropy(seg_attr(line, "token_entropy"))
        if value is None:
            continue
        ls = seg_attr(line, "start")
        le = seg_attr(line, "end")
        if ls is None or le is None or (float(ls) < win_end and float(le) > win_start):
            line_values.append(value)

    if word_values:
        return sum(word_values) / len(word_values)
    if line_values:
        return sum(line_values) / len(line_values)
    return None


# ── Scene classification (AST / YAMNet) ───────────────────────────────


def classification_windows(result: Any) -> list[Any]:  # noqa: ANN401
    """Unwrap classify_audios output to a flat list of per-window dict entries."""
    if not result:
        return []
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, list):
            return list(first)
        return list(result)
    return []


def classification_window_top1(window: Any) -> tuple[str | None, float | None, float | None]:  # noqa: ANN401
    """Return ``(top1_label, top1_score, entropy)`` for one classification window dict.

    Senselab's ``classify_audios`` (windowed) emits ``{"start", "end", "labels": [...],
    "scores": [...]}`` per window — labels and scores are pre-sorted descending, so
    ``labels[0]`` is the top-1.
    """
    if not isinstance(window, dict):
        return None, None, None
    pairs = label_scores(window)
    labels = [next(iter(d)) for d in pairs]
    scores = [next(iter(d.values())) for d in pairs]
    if not labels or not scores:
        return None, None, None
    label = str(labels[0])
    score = float(scores[0])
    probs = [max(float(s), 1e-12) for s in scores]
    total = sum(probs) or 1.0
    probs = [p / total for p in probs]
    entropy = -sum(p * math.log(p) for p in probs)
    return label, score, entropy


def classification_top1_in_window(result: Any, win_idx: int) -> tuple[str | None, float | None, float | None]:  # noqa: ANN401
    """Return top-1 (label, score, entropy) for the ``win_idx``-th classification window."""
    windows = classification_windows(result)
    if win_idx < 0 or win_idx >= len(windows):
        return None, None, None
    return classification_window_top1(windows[win_idx])


# ── PPG ↔ ASR phoneme error rate ──────────────────────────────────────


def g2p_phonemes(text: str) -> list[str]:
    """Run g2p_en on ``text`` and return the ARPAbet phoneme sequence.

    Lazy import + lazy NLTK resource download. Returns an empty list when text
    is empty. NLTK lookup failures (missing tagger / cmudict) trigger one
    targeted download attempt; if that download itself fails (no network), the
    exception is re-raised so the caller sees the real cause rather than a
    silent empty PPG↔ASR PER signal across the entire run.
    """
    if not text.strip():
        return []
    import nltk
    from g2p_en import G2p  # type: ignore[import-untyped]

    from senselab.audio.tasks.speech_to_text_evaluation.utils import strip_nonlexical_tokens

    # A bracketed marker is an annotation, not a word. Running G2P on "[cough]" yields
    # phonemes for something nobody said, which then get aligned against real acoustics
    # and compared as if they were a transcription disagreement.
    text = strip_nonlexical_tokens(text)
    if not text.strip():
        return []

    g = getattr(g2p_phonemes, "_cached_g2p", None)
    if g is None:
        try:
            g = G2p()
        except LookupError:
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)
            nltk.download("cmudict", quiet=True)
            g = G2p()  # if this still fails, re-raise (real config problem).
        g2p_phonemes._cached_g2p = g  # type: ignore[attr-defined]
    try:
        seq = g(text)
    except LookupError:
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        seq = g(text)
    return [str(p).strip() for p in seq if str(p).strip() and not str(p).isspace()]


def normalize_arpabet(phoneme: str) -> str:
    """Normalise a g2p_en ARPAbet phoneme: lowercase, stress markers stripped.

    g2p_en returns uppercase ARPAbet with stress markers (``"AH0"``, ``"EY1"``).
    Named for the ``ppgs`` inventory it once targeted; the normalisation outlived that consumer
    because it is what makes two ASRs' phoneme sequences comparable — ``AH0`` and ``AH1`` are the
    same phoneme, and counting them as a difference would inflate every pairwise distance.

    The format is lowercase ARPAbet without stress markers
    (``"ah"``, ``"ey"``) plus ``"<silent>"`` for non-speech frames. Mapping:
    ``.lower().rstrip("0123456789")``.
    """
    return phoneme.lower().rstrip("0123456789")


def asr_phoneme_sequence_in_window(
    asr_result: Any,  # noqa: ANN401
    win_start: float,
    win_end: float,
    *,
    fully_contained: bool = True,
) -> list[str]:
    """Return the ARPAbet phoneme sequence (PPG-format) for the ASR words in the window.

    Two modes:

    - ``fully_contained=True`` (default): keep only words whose ``[start, end]``
      lies entirely inside the window. The whole word's phoneme sequence
      contributes. Used by callers that prefer "all-or-nothing per word".
    - ``fully_contained=False``: per-phoneme overlap. Each word's phonemes are
      distributed uniformly across the word's time span (one slot per phoneme,
      ``slot_dur = word_dur / n_phonemes``). A phoneme is kept when its slot
      midpoint falls inside the bucket. This is the right rule for
      PPG-vs-ASR PER comparison: PPG argmax includes every audio frame in the
      window, so the ASR side must also reflect "phonemes that occur during
      this time" rather than "whole words that fit". Without this, boundary
      words artificially deflate the ASR sequence and inflate PER.

    All output phonemes are translated to PPG inventory format
    (``normalize_arpabet``).
    """
    if not asr_result:
        return []
    items = asr_result if isinstance(asr_result, list) else [asr_result]
    out: list[str] = []

    def _walk(node: Any) -> None:  # noqa: ANN401
        # Recurse into ``.chunks`` until we hit a leaf (post-MMS-aligned
        # text-only ASR is line → asr → words; Whisper / Qwen are
        # line → words). Apply the bucket containment rule at the leaf.
        chunks = seg_attr(node, "chunks") or []
        if chunks:
            for c in chunks:
                _walk(c)
            return
        cs = seg_attr(node, "start")
        ce = seg_attr(node, "end")
        text = seg_attr(node, "text") or ""
        if cs is None or ce is None or not text.strip():
            return
        cs_f, ce_f = float(cs), float(ce)
        if fully_contained:
            if cs_f >= win_start and ce_f <= win_end:
                out.extend(normalize_arpabet(p) for p in g2p_phonemes(text.strip()))
            return
        # Overlap mode: distribute the word's phonemes uniformly across its
        # time span and keep those whose midpoint is inside the bucket.
        if cs_f >= win_end or ce_f <= win_start:
            return
        phonemes = g2p_phonemes(text.strip())
        if not phonemes:
            return
        word_dur = max(ce_f - cs_f, 1e-9)
        slot_dur = word_dur / len(phonemes)
        for i, p in enumerate(phonemes):
            mid = cs_f + (i + 0.5) * slot_dur
            if win_start <= mid < win_end:
                out.append(normalize_arpabet(p))

    for line in items:
        _walk(line)
    return out


def _levenshtein(a: list[str], b: list[str]) -> int:
    """Phoneme-level edit distance (insertions + deletions + substitutions) between two sequences."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(
                cur[j - 1] + 1,  # insertion in a
                prev[j] + 1,  # deletion from a
                prev[j - 1] + cost,  # substitution
            )
        prev = cur
    return prev[-1]


SILENCE_LABEL = "<silent>"
"""The label a harvester uses for "nobody here".

Spelled from the PPG inventory's non-speech class, which is where it came from, and it outlived that
inventory because ``speaker_claims_from_votes`` needs it: a model reporting silence has *claimed
nothing*, and treating that as a claim would let the mask discount a model for agreeing with it.
Named once so the label and the readers that compare against it cannot drift apart."""
