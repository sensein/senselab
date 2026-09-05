"""Proposing spans from an envelope and its local floor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Span:
    """One proposed span.

    Attributes:
        start: Onset in seconds.
        end: Offset in seconds.
        peak_over_floor_db: The span's peak, referenced to the recording's floor.
        merged_proposals: How many proposals this span absorbed. One for a span the merge rule left
            alone — a span is its own proposal — so zero is never a valid value, and a span covering
            several events is legible as one rather than indistinguishable from a single event.
    """

    start: float
    end: float
    peak_over_floor_db: float
    merged_proposals: int = 1


@dataclass(frozen=True)
class NoContrast:
    """Nothing anywhere rose the required amount above the local floor.

    Distinct from an empty span list: an unmeasurable recording must not read as a quiet one.

    Attributes:
        reason: What was required and what was found.
    """

    reason: str


def propose_spans(
    envelope_db: np.ndarray,
    floor_db: float,
    sampling_rate: int,
    *,
    k_db: float,
    floor_margin_db: float,
    transition_window_ms: int,
    min_duration_ms: int,
    min_separation_ms: int,
) -> list[Span] | NoContrast:
    """Propose spans from an envelope, against one floor for the whole recording.

    The threshold crossing itself generates the candidate spans: every maximal run where the
    envelope has risen ``k_db`` above the floor is one candidate, directly — not a peak later
    expanded outward. Onset and offset then walk by the identical rule, in opposite directions,
    from each candidate's own edges: neither can close for a reason the other would not also close
    for, unlike the peak-anchored/floor-fraction pair this replaced, whose asymmetry (see
    ``floor_margin_db``) let a real multi-scene recording collapse into one merged span once an
    unrelated bug stopped masking it.

    Args:
        envelope_db: Envelope in dBFS, ``nan`` at any sample that had no dB value.
        floor_db: The recording's own global floor — see
            :func:`~senselab.audio.tasks.envelope.api.global_floor_dbfs` — one value for the whole
            signal, not a local, time-varying one.
        sampling_rate: Samples per second.
        k_db: How far above the floor the envelope must cross to open a candidate span. Per reader:
            read it from ``spans.k_db.<reader>`` in the triage config.
        floor_margin_db: A walk stops once the envelope has fallen within this many dB of the floor,
            sustained for ``transition_window_ms`` — the same rule for onset (walking backward) and
            offset (walking forward). Replaces a peak-anchored onset (walk back while within a fixed
            drop of the peak) paired with a floor-fraction offset (walk forward to a threshold fixed
            once at the peak's own floor value): that pair's asymmetry is what let an offset's stale,
            peak-anchored threshold outlive the walk's own progress across a scene. Read it from
            ``spans.floor_margin_db``.
        transition_window_ms: How long the envelope must stay within ``floor_margin_db`` of the floor,
            continuously, before a walk closes. Must be shorter than the shortest event to be
            bounded. Read it from ``spans.transition_window_ms``.
        min_duration_ms: Discard spans shorter than this. Read it from ``spans.min_duration_ms``.
        min_separation_ms: Two threshold-crossing candidates closer together than this are one
            proposal, not two, before either is walked. Read it from ``spans.min_separation_ms``.

    Returns:
        Merged spans in time order, every extent bounded by measured samples, or
        :class:`NoContrast` when nothing crosses ``k_db`` or nothing was measurable at all.
    """
    above = envelope_db - floor_db
    rise = np.where(np.isfinite(above), above, -np.inf)
    measured = np.where(np.isfinite(envelope_db), envelope_db, -np.inf)
    crossings = _contiguous_true_runs(rise >= k_db)
    if not crossings:
        rises = above[np.isfinite(above)]
        if rises.size == 0:
            return NoContrast(reason="the envelope holds no sample measurable against its floor")
        return NoContrast(
            reason=f"nothing rose {k_db} dB above the floor; the largest rose {float(rises.max()):.1f} dB"
        )
    separation = int(min_separation_ms * sampling_rate / 1000)
    candidates: list[tuple[int, int, int]] = []
    for start, end in crossings:
        if candidates and start - candidates[-1][1] < separation:
            candidates[-1] = (candidates[-1][0], end, candidates[-1][2] + 1)
        else:
            candidates.append((start, end, 1))
    threshold = floor_db + floor_margin_db
    win = int(transition_window_ms * sampling_rate / 1000)
    found: list[Span] = []
    for start, end, n_absorbed in candidates:
        peak = float(measured[start:end].max())
        i = start
        while i > 0:
            window = measured[max(0, i - win) : i]
            if len(window) == 0 or window.max() <= threshold:
                break
            i -= 1
        j = end
        while j < len(envelope_db) - 1:
            window = measured[j : j + win]
            if len(window) == 0 or window.max() <= threshold:
                break
            j += 1
        if (j - i) >= min_duration_ms * sampling_rate / 1000:
            found.append(
                Span(
                    start=i / sampling_rate,
                    end=j / sampling_rate,
                    peak_over_floor_db=peak - floor_db,
                    merged_proposals=n_absorbed,
                )
            )
    found.sort(key=lambda s: s.start)
    merged: list[Span] = []
    for span in found:
        if merged and span.start <= merged[-1].end:
            last = merged[-1]
            merged[-1] = Span(
                start=last.start,
                end=max(last.end, span.end),
                peak_over_floor_db=max(last.peak_over_floor_db, span.peak_over_floor_db),
                merged_proposals=last.merged_proposals + span.merged_proposals,
            )
        else:
            merged.append(span)
    return merged


def _contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Start/end (half-open) sample index pairs for each maximal run of ``True`` in ``mask``."""
    if not mask.any():
        return []
    edges = np.flatnonzero(np.diff(np.concatenate(([False], mask, [False]))))
    return [(int(edges[i]), int(edges[i + 1])) for i in range(0, len(edges), 2)]


def _n_change_points(n_samples: int, cut_percentile: float) -> int:
    """How many of ``n_samples`` the rank cut marks, at ``cut_percentile`` percent.

    Args:
        n_samples: Length of the trace.
        cut_percentile: Percent of samples to mark, lowest first.

    Returns:
        The count, which may be zero.
    """
    return int(round(n_samples * cut_percentile / 100.0))


def rank_cut_level(trace: np.ndarray, *, cut_percentile: float) -> float | None:
    """The highest trace value the rank cut marks as a change point.

    Args:
        trace: One value per sample.
        cut_percentile: Percent of samples to mark as change points, lowest first.

    Returns:
        The level, or None when the trace is empty or the cut marks no sample. Where the trace has
        ties astride the boundary this level is reached by more samples than the cut marks, because
        the cut selects by rank and not by value; it is an annotation of where the cut fell, never
        an equivalent test.
    """
    n = len(trace)
    n_change_points = _n_change_points(n, cut_percentile)
    if n == 0 or n_change_points <= 0:
        return None
    return float(np.sort(trace, kind="stable")[n_change_points - 1])


def segments_between_change_points(
    trace: np.ndarray, sampling_rate: int, *, cut_percentile: float, min_duration_ms: float
) -> list[Span]:
    """Spans covering the runs a novelty trace leaves between its lowest-ranked samples.

    The lowest ``cut_percentile`` percent of ``trace``, **selected by rank**, are the change points;
    each maximal run of the remaining samples becomes one span. Ranking rather than comparing against
    a percentile value is what makes this total: a value comparison lands on the plateau of a flat
    trace, where ``>`` admits no samples at all and ``>=`` admits every one of them.

    Unlike :func:`propose_spans` there is no floor, no margin and no walk — a rank cut is scale-free,
    so nothing here depends on the units or spread of the trace it is given.

    Args:
        trace: One value per sample. Higher means less change.
        sampling_rate: Samples per second, used to place the runs on the timeline.
        cut_percentile: Percent of samples to mark as change points, lowest first.
        min_duration_ms: Runs shorter than this are dropped.

    Returns:
        The surviving spans, in start order. ``peak_over_floor_db`` is ``nan``: no floor was
        referenced. Empty when every run is shorter than ``min_duration_ms``.
    """
    n = len(trace)
    if n == 0:
        return []
    n_change_points = _n_change_points(n, cut_percentile)
    is_change_point = np.zeros(n, dtype=bool)
    if n_change_points > 0:
        is_change_point[np.argsort(trace, kind="stable")[:n_change_points]] = True
    minimum_samples = min_duration_ms * sampling_rate / 1000.0
    return [
        Span(start=start / sampling_rate, end=end / sampling_rate, peak_over_floor_db=float("nan"))
        for start, end in _contiguous_true_runs(~is_change_point)
        if (end - start) >= minimum_samples
    ]


def group_extents_into_runs(extents: list[tuple[float, float]], gap_ms: float) -> list[tuple[float, float, list[int]]]:
    """A run is the extent of a group of extents; a gap over ``gap_ms`` starts a new run.

    Generic over any already-timed source (consensus ASR words, or any other list of
    ``(start, end)`` extents already in seconds) — there is no envelope or floor here, unlike
    :func:`propose_spans`: the caller's own extents are the ground truth being grouped, not a
    measure needing a threshold to become one.

    Args:
        extents: ``(start, end)`` pairs in seconds, in any order.
        gap_ms: The gap that starts a new run, in milliseconds.

    Returns:
        ``[(start, end, member indices), ...]``, in start order. Member indices refer to
        positions in the input ``extents`` list.
    """
    runs: list[tuple[float, float, list[int]]] = []
    for index in sorted(range(len(extents)), key=lambda i: extents[i][0]):
        start, end = extents[index]
        if runs and (start - runs[-1][1]) * 1000.0 <= gap_ms:
            open_start, open_end, members = runs[-1]
            runs[-1] = (open_start, max(open_end, end), [*members, index])
        else:
            runs.append((start, end, [index]))
    return runs
