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

from typing import Any, Final, Mapping, Optional

from senselab.audio.workflows.audio_analysis.shapes import Capacity, Span, Spans
from senselab.audio.workflows.audio_analysis.statistics import entropy_uncertainty

__all__ = [
    "SPEAKER_CAPACITY",
    "capacity_for",
    "count_at",
    "count_posterior",
    "count_posterior_in_window",
    "occupancy",
    "spans_from_diarization",
]

SPEAKER_CAPACITY: Final[Mapping[str, Capacity]] = {
    "pyannote/speaker-diarization-community-1": "unbounded",
    "nvidia/diar_sortformer_4spk-v1": 4,
    "mago-ai/ultra_diar_streaming_sortformer_8spk_v1": 8,
}
"""Each diarizer's maximum representable speaker count (D-19), declared once.

Fixed by architecture, not by the audio, and **a tool does not report when it runs out** — it simply
assigns what columns it has. So this is provenance of the same kind as units, and without it a reader
cannot distinguish "3 speakers active" from "3 active and the model had no fourth column".

A clustering pipeline with no fixed-width head is ``"unbounded"``. An **unlisted** model raises rather
than defaulting: guessing ``"unbounded"`` would silently un-censor a bounded tool, which is the exact
bias :func:`count_posterior` exists to correct.
"""


_CLUSTERING_PREFIXES: Final[tuple[str, ...]] = ("embedding_silhouette/",)
"""Prefixes identifying an embedder-plus-clusterer diarizer (D-20).

These are ``"unbounded"`` by *construction* rather than by lookup: a clusterer chooses its own
cluster count, so there is no fixed-width head to run out of. Matched by prefix because the tool id
carries the embedding model, so the set is open — one entry per *clusterer*, not per embedder.
"""


def capacity_for(model_id: str) -> Capacity:
    """The declared capacity for ``model_id``, or ``None`` when nothing declared one.

    ``None`` is **not** a permissive default. It means *unknown*, and
    :func:`spans_from_diarization` omits such a tool from the span set rather than including it: a
    tool whose capacity is unknown cannot be censored correctly, and including it uncensored is
    exactly the bias :func:`count_posterior` exists to correct. Omitting loses its evidence, which is
    worse than having it and better than having it wrong.

    Raising instead was tried and is wrong at this depth — one unlisted diarizer would kill the whole
    harvest, so a new model could not be trialled without a table edit first.
    """
    if model_id in SPEAKER_CAPACITY:
        return SPEAKER_CAPACITY[model_id]
    if any(model_id.startswith(prefix) for prefix in _CLUSTERING_PREFIXES):
        return "unbounded"
    return None


def spans_from_diarization(by_model: Mapping[str, Any]) -> dict[str, Spans]:
    """``{model → Spans}`` from a pass summary's ``diarization.by_model`` block.

    Only models whose block reports ``status == "ok"`` **and** whose speaker capacity is declared
    contribute.

    Two omissions, for two different reasons, and both are warned about rather than being silent:

    - A **failed** model is absent rather than present with an empty span set. "The diarizer crashed"
      and "the diarizer found no speakers" are different states, and only the second is evidence.
    - A model with **no declared capacity** is absent because it cannot be censored correctly, and an
      uncensored bounded tool is the bias the count posterior exists to correct. Its absence from
      ``contributing_sources`` downstream is what makes the loss visible.
    """
    import sys

    from senselab.audio.workflows.audio_analysis.harvesters import seg_attr

    out: dict[str, Spans] = {}
    for model, block in (by_model or {}).items():
        if not (isinstance(block, Mapping) and block.get("status") == "ok"):
            continue
        result = block.get("result")
        if not isinstance(result, list) or not result:
            continue
        capacity = capacity_for(model)
        if capacity is None:
            print(
                f"warn: diarizer {model!r} has no declared speaker capacity, so it cannot be "
                "censored and is omitted from the count evidence; add it to "
                "occupancy.SPEAKER_CAPACITY",
                file=sys.stderr,
            )
            continue
        segments = result[0] if isinstance(result[0], list) else result
        spans = []
        for segment in segments:
            start, end = seg_attr(segment, "start"), seg_attr(segment, "end")
            label = seg_attr(segment, "speaker") or seg_attr(segment, "label")
            if start is None or end is None or label is None or float(end) <= float(start):
                continue
            spans.append(Span(start=float(start), end=float(end), label=str(label)))
        out[model] = Spans(spans=tuple(spans), capacity=capacity)
    return out


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


def count_posterior_in_window(
    spans_by_tool: Mapping[str, Spans],
    *,
    start: float,
    end: float,
    step: float = 0.01,
) -> Optional[dict[str, Any]]:
    """Pool per-instant count posteriors across ``[start, end)`` into one bucket-level distribution.

    Args:
        spans_by_tool: ``{tool → its span set}``, each carrying its own capacity.
        start: Bucket start, seconds.
        end: Bucket end, seconds.
        step: Sampling interval. Default 10 ms — fine enough that a turn boundary lands in its own
            sample rather than being straddled, and it is the resolution the frame-level
            implementation this replaces worked at.

    Returns:
        The same fields as :func:`count_posterior` plus ``n_samples``, or ``None`` when no tool
        reported anything anywhere in the bucket.

    **Pooling distributions, not counts.** The per-instant posteriors are averaged, never the counts.
    Averaging counts first is what made two speakers alternating inside a bucket read as 0.5 speakers
    each and report an overlap that never occurred — the defect the frame-level version was written
    to avoid, and the reason a bucket-level count has to be built from instants and only then reduced.

    ``censored_sources`` and ``lower_bounded`` are the **union** over samples: a tool that hit its
    ceiling anywhere in the bucket was censored in that bucket, because the bucket's figure inherits
    the bound. Intersecting them would report an unbounded count for a bucket that contained one.
    """
    if end <= start:
        return None
    n = max(1, int(round((end - start) / step)))
    pooled: dict[int, float] = {}
    censored: set[str] = set()
    contributing: set[str] = set()
    samples = 0
    lower_bounded_anywhere = False
    for i in range(n):
        t = start + (i + 0.5) * (end - start) / n
        per_tool = {tool: count_at(spans, t) for tool, spans in spans_by_tool.items()}
        speaking = {tool: c for tool, c in per_tool.items() if c > 0}
        if not speaking:
            continue
        at_t = count_posterior(speaking, capacities={tool: spans_by_tool[tool].capacity for tool in speaking})
        if at_t is None:
            continue
        samples += 1
        for count, p in at_t["counts"].items():
            pooled[count] = pooled.get(count, 0.0) + p
        censored |= set(at_t["censored_sources"])
        contributing |= set(at_t["contributing_sources"])
        lower_bounded_anywhere = lower_bounded_anywhere or at_t["lower_bounded"]
    if not samples:
        return None
    counts = {count: mass / samples for count, mass in sorted(pooled.items())}
    return {
        "counts": counts,
        "expected_count": sum(count * p for count, p in counts.items()),
        "p_overlap": sum(p for count, p in counts.items() if count > 1),
        "uncertainty": entropy_uncertainty({str(count): p for count, p in counts.items()}),
        "lower_bounded": lower_bounded_anywhere,
        "censored_sources": tuple(sorted(censored)),
        "contributing_sources": tuple(sorted(contributing)),
        "n_samples": samples,
    }
