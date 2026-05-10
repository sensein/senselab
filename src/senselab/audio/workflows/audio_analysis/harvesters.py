"""Per-task harvest helpers used by the three uncertainty axes.

These read the in-memory result objects produced by senselab's audio task pipeline
(diarization, ASR, scene classification, PPG, alignment) and project them onto a bucket
boundary so the per-axis vote harvesters can build their dicts. The functions here are
shape-tolerant: they work with both Pydantic models (in-memory) and the dict shape that
JSON-cache deserialization produces.
"""

from __future__ import annotations

import math
from typing import Any


def seg_attr(seg: Any, name: str) -> Any:  # noqa: ANN401
    """Return ``seg.name`` whether ``seg`` is a Pydantic model or a JSON dict.

    Cache reads deserialize ScriptLine into plain dicts; in-memory results are Pydantic
    objects. Both shapes flow through the harvesters.
    """
    if isinstance(seg, dict):
        return seg.get(name)
    return getattr(seg, name, None)


# ── Diarization ───────────────────────────────────────────────────────


def diar_speaks_in_window(result: Any, win_start: float, win_end: float) -> bool:  # noqa: ANN401
    """True if any diarization segment overlaps ``[win_start, win_end)``."""
    if not result:
        return False
    segments = result[0] if isinstance(result, list) and result else []
    for seg in segments:
        s = seg_attr(seg, "start")
        e = seg_attr(seg, "end")
        if s is None or e is None:
            continue
        if float(s) < win_end and float(e) > win_start:
            return True
    return False


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
    """True if any ScriptLine has a non-null start or non-empty chunks."""
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


def token_overlaps_window(result: Any, win_start: float, win_end: float) -> bool:  # noqa: ANN401
    """True if any transcript chunk's timestamp overlaps the window."""
    if not result:
        return False
    items = result if isinstance(result, list) else [result]
    for line in items:
        chunks = seg_attr(line, "chunks") or []
        if chunks:
            for c in chunks:
                cs = seg_attr(c, "start")
                ce = seg_attr(c, "end")
                if cs is None or ce is None:
                    continue
                if float(cs) < win_end and float(ce) > win_start:
                    return True
        else:
            ls = seg_attr(line, "start")
            le = seg_attr(line, "end")
            if ls is not None and le is not None and float(ls) < win_end and float(le) > win_start:
                return True
    return False


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
            utterance axis (with True) so partial words straddling a window boundary
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
        # outer line → utterance ScriptLine → word ScriptLines. Whisper / Qwen
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


def whisper_bucket_confidence(result: Any, win_start: float, win_end: float) -> float | None:  # noqa: ANN401
    """Mean Whisper-native confidence over chunks overlapping the window.

    Falls back to the segment-level avg_logprob when chunks are absent (e.g.
    post-aligned text-only ASR). Returns None when no native signal is available
    (FR-007 — drop, do not zero-impute).

    Note: this returns the arithmetic mean of per-chunk ``exp(avg_logprob)`` —
    appropriate for the presence axis where each chunk is treated as an independent
    "is the model confident here?" vote. Do NOT take ``log()`` of this value to
    recover an avg_logprob — by Jensen's inequality
    ``log(mean(exp(x))) > mean(x)``. Use ``whisper_bucket_avg_logprob`` instead.
    """
    if not result:
        return None
    items = result if isinstance(result, list) else [result]
    confidences: list[float] = []
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
                conf, _ = whisper_chunk_confidence(c)
                if conf is not None:
                    confidences.append(conf)
        if not chunk_seen_any:
            ls = seg_attr(line, "start")
            le = seg_attr(line, "end")
            if ls is None or le is None or (float(ls) < win_end and float(le) > win_start):
                conf, _ = whisper_chunk_confidence(line)
                if conf is not None:
                    confidences.append(conf)
    if not confidences:
        return None
    return sum(confidences) / len(confidences)


def whisper_bucket_no_speech_prob(result: Any, win_start: float, win_end: float) -> float | None:  # noqa: ANN401
    """Mean Whisper ``no_speech_prob`` over chunks overlapping the window.

    Whisper exports a per-segment ``no_speech_prob`` from its silence head —
    the cleanest single-scalar VAD signal Whisper itself produces. ``∈ [0, 1]``
    where higher means the model thinks this region is silence. Returned to
    the presence harvester as a direct voice-presence voter.
    """
    if not result:
        return None
    items = result if isinstance(result, list) else [result]
    nsps: list[float] = []
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
                _, nsp = whisper_chunk_confidence(c)
                if nsp is not None:
                    nsps.append(nsp)
        if not chunk_seen_any:
            ls = seg_attr(line, "start")
            le = seg_attr(line, "end")
            if ls is None or le is None or (float(ls) < win_end and float(le) > win_start):
                _, nsp = whisper_chunk_confidence(line)
                if nsp is not None:
                    nsps.append(nsp)
    if not nsps:
        return None
    return sum(nsps) / len(nsps)


def whisper_bucket_avg_logprob(result: Any, win_start: float, win_end: float) -> float | None:  # noqa: ANN401
    """Mean of raw per-chunk ``avg_logprob`` over chunks overlapping the window.

    Returns the arithmetic mean of negative logprobs — equivalent to the geometric
    mean of per-chunk confidences when later exponentiated. This is the unbiased
    way to aggregate Whisper's native logprob to a bucket scale; the utterance
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
    labels = window.get("labels") or []
    scores = window.get("scores") or []
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


def arpabet_to_ppg_inventory(phoneme: str) -> str:
    """Translate a g2p_en ARPAbet phoneme to the lowercase no-stress format used by ``ppgs``.

    g2p_en returns uppercase ARPAbet with stress markers (``"AH0"``, ``"EY1"``).
    The ``ppgs`` library uses lowercase ARPAbet without stress markers
    (``"ah"``, ``"ey"``) plus ``"<silent>"`` for non-speech frames. Mapping:
    ``.lower().rstrip("0123456789")``.
    """
    return phoneme.lower().rstrip("0123456789")


def ppg_argmax_per_frame(
    ppg_result: Any,  # noqa: ANN401
    phoneme_labels: list[str] | tuple[str, ...] | None,
    duration_s: float,
) -> tuple[list[str], float]:
    """Pre-compute the per-frame argmax phoneme sequence from a PPG tensor.

    Args:
        ppg_result: The output of ``extract_ppgs_from_audios`` — typically a list of
            tensors per audio. Each tensor is shaped ``(phonemes, frames)`` or
            ``(1, phonemes, frames)`` per the ``ppgs`` library convention; we
            normalize to ``(frames, phonemes)`` then argmax along the phoneme axis.
        phoneme_labels: The PPG inventory (e.g. lowercase ARPAbet + ``<silent>``).
            When ``None``, falls back to the 40-phoneme ppgs default inventory.
        duration_s: Audio duration in seconds. ``frame_hop = duration_s / n_frames``.

    Returns:
        ``(per_frame_phonemes, frame_hop_s)``. Empty list and 0.0 hop on bad input.
    """
    if not ppg_result or duration_s <= 0:
        return [], 0.0
    labels = list(phoneme_labels) if phoneme_labels else list(_DEFAULT_PPG_LABELS)
    n_phonemes = len(labels)
    # Outer list = per-audio; we always pass a single audio.
    ppg = ppg_result[0] if isinstance(ppg_result, list) and ppg_result else ppg_result
    arr = _to_2d_frame_major(ppg, n_phonemes=n_phonemes)
    if arr is None or arr.shape[0] == 0:
        return [], 0.0
    n_frames = int(arr.shape[0])
    frame_hop = duration_s / n_frames

    # Argmax along phoneme axis.
    indices = arr.argmax(axis=-1) if hasattr(arr, "argmax") else None
    if indices is None:
        return [], 0.0

    per_frame: list[str] = []
    for raw_idx in indices.tolist() if hasattr(indices, "tolist") else list(indices):
        idx = int(raw_idx)
        per_frame.append(labels[idx] if 0 <= idx < len(labels) else "<unk>")
    return per_frame, frame_hop


def ppg_argmax_confidence_per_frame(
    ppg_result: Any,  # noqa: ANN401
    phoneme_labels: list[str] | tuple[str, ...] | None,
    duration_s: float,
) -> tuple[list[float], float]:
    """Pre-compute the per-frame ARGMAX POSTERIOR (max softmax value) of the PPG.

    Parallel to ``ppg_argmax_per_frame`` but returns the value at the argmax
    instead of the label. PPG outputs are softmax probabilities per phoneme per
    frame; argmax probability is the model's confidence in its top-1 phoneme.
    Aggregating these in a bucket window gives a per-bucket PPG-confidence
    signal complementary to the ASR alignment score.

    Returns ``(per_frame_argmax_probs, frame_hop_s)``. Empty list and 0.0 hop
    on bad input.
    """
    if not ppg_result or duration_s <= 0:
        return [], 0.0
    labels = list(phoneme_labels) if phoneme_labels else list(_DEFAULT_PPG_LABELS)
    n_phonemes = len(labels)
    ppg = ppg_result[0] if isinstance(ppg_result, list) and ppg_result else ppg_result
    arr = _to_2d_frame_major(ppg, n_phonemes=n_phonemes)
    if arr is None or arr.shape[0] == 0:
        return [], 0.0
    n_frames = int(arr.shape[0])
    frame_hop = duration_s / n_frames
    if not hasattr(arr, "max"):
        return [], 0.0
    max_per_frame = arr.max(axis=-1)
    flat = max_per_frame.tolist() if hasattr(max_per_frame, "tolist") else list(max_per_frame)
    return [float(v) for v in flat], frame_hop


def ppg_mean_confidence_in_window(
    ppg_argmax_confidence_per_frame_seq: list[float],
    ppg_frame_hop: float,
    win_start: float,
    win_end: float,
) -> float | None:
    """Mean per-frame PPG argmax probability over frames inside ``[win_start, win_end)``.

    Higher value (closer to 1.0) → PPG is more confident in its phoneme decoding
    for this bucket. Lower value → many frames have flat / spread posteriors,
    which usually correlates with non-speech audio or ambiguous phonetic content.
    """
    if not ppg_argmax_confidence_per_frame_seq or ppg_frame_hop <= 0 or win_end <= win_start:
        return None
    first_frame = max(0, int(win_start / ppg_frame_hop))
    last_frame = min(int(math.ceil(win_end / ppg_frame_hop)), len(ppg_argmax_confidence_per_frame_seq))
    if last_frame <= first_frame:
        return None
    slice_ = ppg_argmax_confidence_per_frame_seq[first_frame:last_frame]
    if not slice_:
        return None
    return sum(slice_) / len(slice_)


def _to_2d_frame_major(t: Any, *, n_phonemes: int) -> Any:  # noqa: ANN401
    """Normalize a PPG tensor / array to ``(frames, phonemes)``.

    ``extract_ppgs_from_audios`` returns ``(1, phonemes, frames)`` or
    ``(phonemes, frames)``; sometimes the cache round-trip leaves a list-of-lists.
    Cached entries from analyze_audio's wrapper serialize tensors as
    ``{"_tensor_shape": [...], "_dtype": "torch.float32", "values": [...]}`` — we
    recognise that shape and rebuild the array.

    Disambiguates orientation by matching against the caller-supplied ``n_phonemes``
    (the actual inventory size from the PPG block, not a hard-coded default).

    Returns ``None`` on shapes we can't safely interpret. When neither dim matches
    ``n_phonemes`` exactly, falls back to "phoneme axis is the smaller dim" — but
    raises a ``ValueError`` if the shape is ambiguous (square, both dims tiny) so
    the caller surfaces the problem rather than silently producing garbage argmax.
    """
    import numpy as np

    if t is None:
        return None
    if isinstance(t, dict) and "_tensor_shape" in t and "values" in t:
        # Cache-restored tensor → reconstruct the numpy array.
        try:
            arr = np.asarray(t["values"]).reshape(t["_tensor_shape"])
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"PPG cached tensor reconstruction failed (shape={t.get('_tensor_shape')!r}): {exc!r}"
            ) from exc
    else:
        arr = np.asarray(t.detach().cpu()) if hasattr(t, "detach") else np.asarray(t)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        return None
    rows, cols = arr.shape
    if rows == n_phonemes and cols != n_phonemes:
        return arr.T  # (phonemes, frames) → (frames, phonemes)
    if cols == n_phonemes and rows != n_phonemes:
        return arr  # already (frames, phonemes)
    if rows == n_phonemes and cols == n_phonemes:
        # Square against the inventory size — assume the conventional
        # ppgs layout ``(phonemes, frames)`` and transpose.
        return arr.T
    # Neither dim matches the inventory exactly — heuristic fallback, but only when
    # the shape is unambiguous. ``frames`` is typically ≥10× ``phonemes`` for any
    # audio longer than ~0.4 s @ 100 Hz frame rate.
    if max(rows, cols) < 4 * min(rows, cols):
        raise ValueError(
            f"PPG tensor shape {arr.shape} is ambiguous against inventory size "
            f"{n_phonemes}; cannot determine frame vs phoneme axis."
        )
    return arr.T if rows < cols else arr


def ppg_argmax_runs_in_window(
    ppg_per_frame_phonemes: list[str],
    ppg_frame_hop: float,
    win_start: float,
    win_end: float,
) -> list[tuple[float, float, str]]:
    """Return ``[(start, end, phoneme), ...]`` for argmax runs inside the window.

    Walks the per-frame argmax sequence inside ``[win_start, win_end)``, collapses
    consecutive frames sharing the same phoneme into a single run, and reports the
    run's time span. ``<silent>`` runs are kept (they're a valid phoneme in the PPG
    inventory). Used both by the utterance edit-distance metric (just the phoneme
    sequence) and by the plot's PPG row (which renders the time-spans as bars).
    """
    if not ppg_per_frame_phonemes or ppg_frame_hop <= 0 or win_end <= win_start:
        return []
    # ``last_frame`` is exclusive. Use ``ceil`` so a window whose end lands
    # mid-frame still includes that frame; ``int()`` would silently drop the
    # final partial frame and over time elide many frames from the PER signal.
    first_frame = max(0, int(win_start / ppg_frame_hop))
    last_frame = min(int(math.ceil(win_end / ppg_frame_hop)), len(ppg_per_frame_phonemes))
    if last_frame <= first_frame:
        return []
    runs: list[tuple[float, float, str]] = []
    cur_phon = ppg_per_frame_phonemes[first_frame]
    cur_start_frame = first_frame
    for f in range(first_frame + 1, last_frame):
        if ppg_per_frame_phonemes[f] != cur_phon:
            runs.append((cur_start_frame * ppg_frame_hop, f * ppg_frame_hop, cur_phon))
            cur_phon = ppg_per_frame_phonemes[f]
            cur_start_frame = f
    runs.append((cur_start_frame * ppg_frame_hop, last_frame * ppg_frame_hop, cur_phon))
    return runs


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
    (``arpabet_to_ppg_inventory``).
    """
    if not asr_result:
        return []
    items = asr_result if isinstance(asr_result, list) else [asr_result]
    out: list[str] = []

    def _walk(node: Any) -> None:  # noqa: ANN401
        # Recurse into ``.chunks`` until we hit a leaf (post-MMS-aligned
        # text-only ASR is line → utterance → words; Whisper / Qwen are
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
                out.extend(arpabet_to_ppg_inventory(p) for p in g2p_phonemes(text.strip()))
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
                out.append(arpabet_to_ppg_inventory(p))

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


def ppg_sequence_per_in_window(
    ppg_per_frame_phonemes: list[str],
    ppg_frame_hop: float,
    asr_result: Any,  # noqa: ANN401
    win_start: float,
    win_end: float,
) -> float | None:
    """Phoneme-sequence edit-distance rate between ASR and PPG over ``[win_start, win_end)``.

    Builds two sequences:

    - **PPG side**: argmax phoneme per frame inside the window, deduped to runs
      (consecutive frames sharing a phoneme collapse to one entry). ``<silent>``
      runs are stripped — they're not real phonemes in the ASR sequence.
    - **ASR side**: g2p_en applied to each fully-contained ASR chunk in the window,
      translated to PPG inventory format.

    Returns ``edit_distance / max(len(asr), len(ppg))`` clipped to ``[0, 1]``, or
    ``None`` when both sides are empty (no signal). Less sensitive to small time
    misalignments than the per-frame approach because the deduped sequence ignores
    duration and only compares phoneme order.
    """
    if not ppg_per_frame_phonemes or ppg_frame_hop <= 0 or win_end <= win_start:
        return None
    runs = ppg_argmax_runs_in_window(ppg_per_frame_phonemes, ppg_frame_hop, win_start, win_end)
    ppg_seq = [p for _, _, p in runs if p != "<silent>"]
    # PPG argmax includes EVERY frame in the window — even frames covering
    # words that straddle the bucket boundary. To compare apples to apples,
    # the ASR phoneme sequence must include the same boundary words. Use the
    # overlap rule (not ``fully_contained``) so we don't artificially deflate
    # the ASR side and inflate the resulting PER.
    asr_seq = asr_phoneme_sequence_in_window(asr_result, win_start, win_end, fully_contained=False)
    if not asr_seq:
        # No fully-contained ASR words in this bucket. We CANNOT score
        # PPG-vs-ASR PER for this ASR model on this bucket — return None so
        # the aggregator drops the sub-signal rather than penalising the
        # model with a spurious 1.0 (used to inflate utterance uncertainty
        # whenever a text-only ASR's transcript didn't quite land inside a
        # bucket boundary).
        return None
    if not ppg_seq:
        # PPG is silent throughout the bucket but ASR has phonemes — that IS
        # a real disagreement (the audio model says no speech, the language
        # model transcribed text). Saturate at 1.0.
        return 1.0
    distance = _levenshtein(asr_seq, ppg_seq)
    denom = max(len(asr_seq), len(ppg_seq))
    return min(1.0, distance / denom) if denom > 0 else None


# Default PPG inventory — ppgs 0.0.9 lowercase ARPAbet + ``<silent>``.
# Pinned here so the harvester is callable without senselab's PPG task installed.
_DEFAULT_PPG_LABELS = (
    "aa", "ae", "ah", "ao", "aw", "ay", "b", "ch", "d", "dh",
    "eh", "er", "ey", "f", "g", "hh", "ih", "iy", "jh", "k",
    "l", "m", "n", "ng", "ow", "oy", "p", "r", "s", "sh",
    "t", "th", "uh", "uw", "v", "w", "y", "z", "zh", "<silent>",
)  # fmt: skip
