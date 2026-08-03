"""Speaker occupancy and count, derived from spans across diarizers of differing capacity (D-19).

L2 derivatives over :class:`~.shapes.Spans`. Every diarization tool emits what sortformer and
``community-1`` already emit — ``(start, end, speaker_label)`` at its own boundaries, on no grid — and
occupancy or a count is derived by projecting them here.

**Why this replaces the Poisson-binomial.** ``joint.overlap_count_posterior`` built a count
distribution over ``segmentation-3.0``'s per-speaker channel probabilities, treating them as
independent Bernoullis. They are a **powerset conversion**: the classes are mutually exclusive by
construction and the per-speaker columns are derived from them, so the independence the
Poisson-binomial assumes was never there. What it produced was one model's internal confidence dressed
as a distribution over speaker count.

The honest uncertainty about "how many speakers are active here" is the same as for every other axis:
**disagreement across models.** Each diarizer's spans give a count at time *t*, and the spread across
diarizers is the uncertainty. That is measured rather than assumed, and it composes with D-19's
censoring — a tool at its capacity contributes a *lower bound*, not a point.

**What is kept from the frame-level version**, because it was right: overlap is an *instantaneous*
fact. Two speakers alternating inside a bucket average to 0.5 on each channel, which as a per-bucket
calculation reports an overlap that never occurred. :func:`count_at` evaluates at an instant, so it
cannot produce that artifact at all.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from senselab.audio.workflows.audio_analysis.shapes import Capacity, Spans
from senselab.audio.workflows.audio_analysis.statistics import entropy_uncertainty

__all__ = ["count_at", "count_posterior", "occupancy"]


def occupancy(spans: Spans, *, start: float, end: float) -> dict[str, float]:
    """Per-speaker coverage of ``[start, end)``, as a proportion of the bucket.

    Args:
        spans: One diarizer's span set.
        start: Bucket start, seconds.
        end: Bucket end, seconds.

    Returns:
        ``{speaker_label → covered_fraction}`` for the speakers this diarizer placed in the bucket.
        A speaker it did not place here is **absent from the mapping**, not present with ``0.0``:
        zero coverage is a claim the diarizer did not make, it simply drew no span.

        Coverage is a **union** per label, so two overlapping spans from one speaker are one speaker
        present rather than 1.5 of them.
    """
    width = end - start
    if width <= 0:
        return {}
    clipped: dict[str, list[tuple[float, float]]] = {}
    for span in spans.spans:
        lo, hi = max(span.start, start), min(span.end, end)
        if hi > lo:
            clipped.setdefault(span.label, []).append((lo, hi))
    return {label: _union_length(intervals) / width for label, intervals in sorted(clipped.items())}


def _union_length(intervals: list[tuple[float, float]]) -> float:
    """Total length covered by ``intervals``, counting overlap once."""
    total = 0.0
    end_so_far = float("-inf")
    for lo, hi in sorted(intervals):
        if hi <= end_so_far:
            continue
        total += hi - max(lo, end_so_far)
        end_so_far = max(end_so_far, hi)
    return total


def count_at(spans: Spans, t: float) -> int:
    """How many distinct speakers this diarizer places at instant ``t``.

    Evaluated at an instant rather than over a window, because overlap is an instantaneous fact. A
    windowed count would make two speakers taking turns indistinguishable from two speaking at once —
    the artifact the frame-level implementation existed to avoid, preserved here by construction.

    A span is treated as ``[start, end)`` so a speaker ending exactly at ``t`` is not counted twice
    with one starting there.
    """
    return len({span.label for span in spans.spans if span.start <= t < span.end})


def count_posterior(
    counts_by_source: Mapping[str, int],
    *,
    capacities: Mapping[str, Capacity],
) -> Optional[dict[str, Any]]:
    """A distribution over speaker count, from cross-tool spread with censoring (D-19).

    Args:
        counts_by_source: ``{tool → the count it reported at this instant}``.
        capacities: ``{tool → its speaker ceiling}``. Every source in ``counts_by_source`` must
            appear. A missing entry raises rather than defaulting to ``"unbounded"``: absent and
            unbounded are different claims, and guessing the permissive one hides exactly the bias
            this function exists to correct.

    Returns:
        ``{"counts", "expected_count", "p_overlap", "uncertainty", "lower_bounded",
        "censored_sources", "contributing_sources"}``, or ``None`` when no source reported — a
        distribution with no evidence behind it is a guess wearing a posterior's clothes.

    Raises:
        KeyError: For a source with no declared capacity.

    **How censoring enters.** A tool *at* its capacity cannot report one more speaker and does not say
    so, so its count is a lower bound: it corroborates every candidate count **at or above** its
    figure and is evidence against none of them. A 4-capacity tool reporting 4 while an unbounded tool
    reports 5 has not contradicted it — it had no fifth column. Counting that 4 as a vote against 5 is
    what biases a fused posterior toward the smallest-capacity tool.

    A tool *below* its capacity is full evidence, because it had a column available and did not use it.
    """
    if not counts_by_source:
        return None
    missing = [source for source in counts_by_source if source not in capacities]
    if missing:
        raise KeyError(f"no capacity declared for {sorted(missing)}; absent is not the same as unbounded")

    censored = tuple(
        sorted(source for source, count in counts_by_source.items() if _is_censored(capacities[source], count))
    )
    support = sorted(set(counts_by_source.values()))
    mass = dict.fromkeys(support, 0.0)
    for source, count in counts_by_source.items():
        if source in censored:
            # A lower bound: spread this source's weight over every candidate it does not rule out.
            eligible = [candidate for candidate in support if candidate >= count]
            for candidate in eligible:
                mass[candidate] += 1.0 / len(eligible)
        else:
            mass[count] += 1.0

    total = sum(mass.values())
    counts = {count: value / total for count, value in mass.items() if value > 0}
    return {
        "counts": counts,
        "expected_count": sum(count * p for count, p in counts.items()),
        "p_overlap": sum(p for count, p in counts.items() if count > 1),
        "uncertainty": entropy_uncertainty({str(count): p for count, p in counts.items()}),
        # Every source at its ceiling means the whole posterior is a lower bound. Without saying so,
        # a ceiling that all tools reached reads as a confident count.
        "lower_bounded": len(censored) == len(counts_by_source),
        "censored_sources": censored,
        "contributing_sources": tuple(sorted(counts_by_source)),
    }


def _is_censored(capacity: Capacity, count: int) -> bool:
    """Is this source's count at a ceiling it cannot see past?"""
    return isinstance(capacity, int) and count >= capacity
