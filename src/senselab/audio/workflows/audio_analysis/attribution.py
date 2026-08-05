"""The speaker axis's composition: how sure are we *who* is speaking here.

The axis used to ask "was it the same speaker as before?", validated per (diar model × embedder) pair
against embedding cosine. On the run's 0.1 s grid that asks ten times a second against embeddings
windowed at 0.5 s, so every disagreement between a diarizer's continuity claim and the cosine
registered as doubt: it read 0.666 on a clean two-speaker conversation whose count posterior was
2 at 0.978, whose per-speaker existence uncertainty was 0.0 and 0.022, and whose per-speaker presence
doubt averaged 0.168.

It now asks about **attribution**, from three terms, each a function here:

- :func:`per_speaker_attribution_doubt` — do the diarization models agree about who is here?
- :func:`word_location_doubt` — do we know where the words are? Word boundaries are what assign a
  word to a speaker's span, so not knowing where a word starts is not knowing whose it is. This
  consumes the per-edge temporal confidences D-27 moved onto the word, which had no consumer until
  now.
- :func:`target_activity_doubt` — do we know whether the target was active at all? Not knowing that
  is not knowing whether there is anyone to attribute.

Pure functions over plain data, deliberately: the composition they define can then be checked without
running a model, which the change-detection composition could not be. The fold across them is
``fuse.fuse_axis``'s, like every other axis — there is no second fold here.

Design: ``specs/20260728-221507-per-speaker-identity-scene/speaker-axis-attribution-design.md``.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

__all__ = [
    "per_speaker_attribution_doubt",
    "target_activity_doubt",
    "word_location_doubt",
]

SILENT_CLUSTER_ID = "SIL"
"""The pseudo-cluster standing for "no speaker here" — bookkeeping, never a person."""


def _binary_entropy(p: float) -> float:
    """Normalised Shannon entropy of a two-outcome split; 0 unanimous, 1 evenly split."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)) / math.log(2.0)


def per_speaker_attribution_doubt(
    clusters: Mapping[str, str],
    *,
    silent_cluster_id: str = SILENT_CLUSTER_ID,
) -> float | None:
    """How much the diarization models disagree about *who* is in this bucket.

    Per speaker present, the share of models placing them here, read as binary entropy — the same
    quantity ``speaker.per_speaker_tracks`` publishes per speaker in
    ``final/per_speaker_presence.parquet``, so the axis and that deliverable can no longer disagree
    about how confident the run is. They did: the tracks reported mean doubt 0.168 on a recording the
    axis called 0.666.

    **Folded by ``max`` over the speakers present, not by mean.** If any speaker's presence here is
    contested, attribution here is contested; averaging a contested speaker against a confidently
    placed one lets the confident one hide the doubt.

    Models reporting silence stay in the denominator: a lone detection among four silent models is
    exactly the case that must not read as certain.

    Args:
        clusters: ``{diar model → cluster id}`` for one bucket.
        silent_cluster_id: The id standing for "no speaker", excluded from the speakers but kept in
            the denominator.

    Returns:
        Doubt in ``[0, 1]``, or ``None`` when no model placed a speaker here — which is the absence
        of a claim rather than confident attribution of nobody.
    """
    if not clusters:
        return None
    n_models = len(clusters)
    active = sorted({c for c in clusters.values() if c != silent_cluster_id})
    if not active:
        return None
    return max(_binary_entropy(sum(1 for c in clusters.values() if c == cluster) / n_models) for cluster in active)


def word_location_doubt(
    words: Sequence[Mapping[str, Any]],
    buckets: Sequence[tuple[float, float]],
) -> dict[tuple[float, float], float | None]:
    """Per bucket, how poorly localised the words reaching it are.

    ``1 - temporal_confidence`` per word, coverage-weighted over the words overlapping the bucket.
    ``temporal_confidence`` is the fused word's own agreement about its span (the per-edge
    ``onset_confidence`` / ``offset_confidence`` folded), so this is the run's own measure of "do we
    know where this word is" — projected onto the axis grid the same way ``asr.resample_word_doubt``
    projects accuracy.

    This is also where the axis's *resolution* comes from: measured on the conversation clip, the
    per-speaker term takes 3 distinct values across the recording and this one takes 79.

    Args:
        words: Fused words carrying ``start``, ``end`` and ``temporal_confidence``.
        buckets: ``(start, end)`` pairs on the axis grid.

    Returns:
        ``{bucket → doubt}``, ``None`` where no word with a measured temporal confidence reaches the
        bucket. ``None`` rather than ``0.0``: nothing was said there, so nothing localises it, and
        zero would assert that we know exactly where a word we never heard was.
    """
    out: dict[tuple[float, float], float | None] = {}
    for bucket in buckets:
        weighted = 0.0
        total = 0.0
        for word in words:
            confidence = word.get("temporal_confidence")
            if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
                continue
            try:
                start, end = float(word["start"]), float(word["end"])
            except (KeyError, TypeError, ValueError):
                continue
            overlap = min(end, bucket[1]) - max(start, bucket[0])
            if overlap > 0:
                weighted += overlap * max(0.0, min(1.0, 1.0 - float(confidence)))
                total += overlap
        out[bucket] = (weighted / total) if total > 0 else None
    return out


def target_activity_doubt(
    mask_regions: Sequence[Mapping[str, Any]],
    buckets: Sequence[tuple[float, float]],
) -> dict[tuple[float, float], tuple[float | None, str | None]]:
    """Per bucket, the mask's doubt about whether the target was active — and its verdict.

    Returns the **state** alongside the number because the number alone cannot be acted on: low
    uncertainty means the mask is sure, and "sure the target is active" and "sure the region is
    target-free" call for opposite treatment. The caller nulls the axis where the state is
    ``target_free``, because there is nobody to attribute there.

    Doubt is contributed **only where the state is not** ``target_active``. Folding the mask's
    uncertainty in unconditionally was measured and rejected: 14 coarse regions against 214 fine
    buckets collapsed the axis from 80 distinct values to 35, the coarse measurement overwriting the
    fine one.

    A bucket takes the region it overlaps most, so a bucket straddling a boundary gets one verdict
    rather than a blend of two, and the first region wins a tie so the result cannot depend on
    iteration order.

    Args:
        mask_regions: Region dicts carrying ``start``, ``end``, ``state`` and ``uncertainty`` — the
            region table (``L2/background_mask.parquet``), not the ``background_mask`` axis rows.
            Only the regions carry ``state``, and the gate needs the direction, not the magnitude.
        buckets: ``(start, end)`` pairs on the axis grid.

    Returns:
        ``{bucket → (doubt, state)}``. ``doubt`` is ``None`` where the state is ``target_active`` (the
        question is simply live) or where no region covers the bucket (the mask said nothing, which is
        not the same as saying the target was active). ``state`` is ``None`` in the latter case.
    """
    out: dict[tuple[float, float], tuple[float | None, str | None]] = {}
    for bucket in buckets:
        best_overlap = 0.0
        best: Mapping[str, Any] | None = None
        for region in mask_regions:
            try:
                start, end = float(region["start"]), float(region["end"])
            except (KeyError, TypeError, ValueError):
                continue
            overlap = min(end, bucket[1]) - max(start, bucket[0])
            if overlap > best_overlap:
                best_overlap, best = overlap, region
        if best is None:
            out[bucket] = (None, None)
            continue
        state = str(best.get("state")) if best.get("state") is not None else None
        if state == "target_active":
            out[bucket] = (None, state)
            continue
        raw = best.get("uncertainty")
        doubt = max(0.0, min(1.0, float(raw))) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else None
        out[bucket] = (doubt, state)
    return out
