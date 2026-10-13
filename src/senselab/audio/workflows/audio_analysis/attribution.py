"""The speaker axis's composition: how sure are we *who* is speaking here.

The axis used to ask "was it the same speaker as before?", validated per (diar model × embedder) pair
against embedding cosine. On the run's 0.1 s grid that asks ten times a second against embeddings
windowed at 0.5 s, so every disagreement between a diarizer's continuity claim and the cosine
registered as doubt: it read 0.666 on a clean two-speaker conversation whose count posterior was
2 at 0.978, whose per-speaker existence uncertainty was 0.0 and 0.022, and whose per-speaker presence
doubt averaged 0.168.

It now asks about **attribution**, from three terms, each a function here:

- :func:`speaker_assignment_doubt` — do the diarization models agree about who is here? Measured
  over *all* the answers they gave, with no speaker privileged, because absent a target embedding the
  question is "do we know who is talking" rather than "is this particular person here".
- :func:`word_coverage` — is there any speech here to attribute? A **gate, not a voter**: a bucket
  with no word in it gets no claim at all, because attributing speech requires speech. Word timing
  bounds *where* a speaker change can be; it is not evidence about *who*. The gate is skipped where
  the mask's region state positively reports a voice (``target_active``/``nontarget_active``, see
  ``speaker._VOCAL_ACTIVITY``): word absence is only a *proxy* for speech absence, and it holds for
  adult connected speech, not for a cry, a cough or a groan (F-165).
- :func:`target_activity_doubt` — do we know whether the target was active at all? Not knowing that
  is not knowing whether there is anyone to attribute.

Every mask-state reading above is **inert on a run today**: the per-region table never reaches the
harvester, so ``state`` is always ``None``, the gate skip never fires and ``target_activity`` never
votes (F-187 in ``specs/20260815-215106-analyze-audio-audit/register.md``).

Pure functions over plain data, deliberately: the composition they define can then be checked without
running a model, which the change-detection composition could not be. The fold across them is
``fuse.fuse_axis``'s, like every other axis — there is no second fold here.

Design: ``specs/20260728-221507-per-speaker-identity-scene/speaker-axis-attribution-design.md``.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

__all__ = [
    "speaker_assignment_doubt",
    "target_activity_doubt",
    "word_coverage",
]

SILENT_CLUSTER_ID = "SIL"
"""The pseudo-cluster standing for "no speaker here" — bookkeeping, never a person."""


def _binary_entropy(p: float) -> float:
    """Normalised Shannon entropy of a two-outcome split; 0 unanimous, 1 evenly split."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)) / math.log(2.0)


def speaker_assignment_doubt(
    clusters: Mapping[str, str],
    *,
    silent_cluster_id: str = SILENT_CLUSTER_ID,
    target: str | None = None,
) -> float | None:
    """How spread the diarization models are over **who** is speaking in this bucket.

    Normalised Shannon entropy of the models' distribution across the answers they gave, where the
    answers are the speaker ids they assigned plus ``silent_cluster_id`` — "nobody" is one of the
    answers to "who is here", so it is an outcome rather than an exclusion. Unanimity is ``0.0``, an
    even split across the claimed answers is ``1.0``.

    **No speaker is privileged.** This was ``max`` over the speakers present of each one's *binary*
    entropy, which picks the single most-contested speaker and reports that speaker's doubt — a
    *targeted* reading with no target supplied, and it read as a statement about one person when the
    question is "do we know who is talking". For two outcomes the two agree exactly (binary entropy is
    symmetric, so ``H(p) = H(1-p)``), which is why the change is invisible on a two-speaker recording
    and wrong in principle: with three or more answers in play the max silently elected one of them.

    ``target`` is the hook for the future targeted mode: given a reference embedding for a specific
    speaker, the question becomes binary — is *that* speaker here — and the right measure is that one
    speaker's binary entropy. Passing it restores exactly that, and leaving it ``None`` keeps the
    untargeted reading. The two are different questions and must not be silently interchanged.

    Args:
        clusters: ``{diar model → cluster id}`` for one bucket.
        silent_cluster_id: The id standing for "no speaker". Kept as an *outcome*, not dropped: a lone
            detection among three silent models is the case that must not read as certain, and it
            scores 0.811 here exactly as it did under the per-speaker form.
        target: A specific cluster id to measure instead of the whole assignment — the targeted mode.
            ``None`` (the default) measures the spread across every answer given.

    Returns:
        Doubt in ``[0, 1]``, or ``None`` when no model placed a speaker here — the absence of a claim
        rather than confident attribution of nobody.
    """
    if not clusters:
        return None
    n_models = len(clusters)
    if not [c for c in clusters.values() if c != silent_cluster_id]:
        return None
    if target is not None:
        # Targeted: one speaker, one binary question.
        return _binary_entropy(sum(1 for c in clusters.values() if c == target) / n_models)
    counts: dict[str, int] = {}
    for cluster in clusters.values():
        counts[cluster] = counts.get(cluster, 0) + 1
    if len(counts) == 1:
        return 0.0
    entropy = -sum((k / n_models) * math.log(k / n_models, 2.0) for k in counts.values())
    # Normalised by the number of answers actually in play, so an even split reads 1.0 whether the
    # models offered two answers or five — the same convention ``statistics.entropy_uncertainty`` uses.
    return max(0.0, min(1.0, entropy / math.log(len(counts), 2.0)))


def word_coverage(
    words: Sequence[Mapping[str, Any]],
    buckets: Sequence[tuple[float, float]],
) -> dict[tuple[float, float], float]:
    """Per bucket, the fraction of it occupied by a recognized word, in ``[0, 1]``.

    **A measurement, not a vote.** The speaker axis reads it as a *gate*: a bucket with no word in it
    has no speech to attribute, so the axis makes no claim there — *unless* the background mask
    positively reports a voice in that bucket (``target_active`` or ``nontarget_active``), which is
    measured evidence against the proxy this gate rests on. That exemption does not fire on a run
    today: the mask's regions never reach the harvester, so the state is always ``None`` (F-187).
    Either way it does not enter the fold.

    That distinction was learned the hard way. This started as ``word_location_doubt`` —
    ``1 - temporal_confidence``, coverage-weighted — and *voted* on the axis. Measured on a clean
    two-speaker conversation it contributed ~0.223 in every bucket, standing doubt that swamped the
    per-speaker term: the axis read 0.295 where the per-speaker evidence said 0.0 across 86% of the
    recording. Word-boundary jitter of a few tens of milliseconds says almost nothing about *which*
    of two speakers said a word — speakers change on the scale of seconds — so as a vote on identity
    it was measuring the wrong thing at the wrong scale. What word timing legitimately does is bound
    *where a speaker change can be*, and the sharpening that buys, for this use case, is knowing when
    there is no speech to attribute at all.

    Spans are unioned before measuring, so two recognizers' overlapping words are one span of speech
    rather than 200% coverage.

    Args:
        words: Fused words carrying ``start`` and ``end``.
        buckets: ``(start, end)`` pairs on the axis grid.

    Returns:
        ``{bucket → covered fraction}``. ``0.0`` is a genuine measurement — no word occupies any of
        this bucket — and is what the caller gates on, in the states where it gates at all: the
        caller checks the mask region first and lets a positively-reported voice override this
        reading (``speaker.harvest_speaker_votes``) — an override no production run reaches, since
        the region state never arrives there (F-187), so today this reading gates unconditionally.
    """
    spans: list[tuple[float, float]] = []
    for word in words:
        try:
            start, end = float(word["start"]), float(word["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            spans.append((start, end))
    spans.sort()
    merged: list[list[float]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    out: dict[tuple[float, float], float] = {}
    for bucket in buckets:
        width = bucket[1] - bucket[0]
        if width <= 0:
            out[bucket] = 0.0
            continue
        covered = sum(max(0.0, min(end, bucket[1]) - max(start, bucket[0])) for start, end in merged)
        out[bucket] = max(0.0, min(1.0, covered / width))
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
