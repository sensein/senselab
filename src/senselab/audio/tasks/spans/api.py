"""Proposing spans from an envelope and its local floor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks


@dataclass(frozen=True)
class Span:
    """One proposed span.

    Attributes:
        start: Onset in seconds.
        end: Offset in seconds.
        peak_over_floor_db: The span's peak, referenced to the local floor.
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
    """No peak anywhere rose the required amount above the local floor.

    Distinct from an empty span list: an unmeasurable recording must not read as a quiet one.

    Attributes:
        reason: What was required and what was found.
    """

    reason: str


def propose_spans(
    envelope_db: np.ndarray,
    floor_db: np.ndarray,
    sampling_rate: int,
    *,
    k_db: float,
    onset_drop_db: float,
    offset_fraction: float,
    hangover_ms: int,
    min_duration_ms: int,
    min_separation_ms: int,
) -> list[Span] | NoContrast:
    """Propose spans from an envelope, anchoring the onset to each event's own peak.

    Args:
        envelope_db: Envelope in dBFS, ``nan`` at any sample that had no dB value.
        floor_db: Local floor, same length as ``envelope_db``, ``nan`` where the window held nothing.
        sampling_rate: Samples per second.
        k_db: How far above the local floor a peak must rise to be proposed. Per reader: read it
            from ``spans.k_db.<reader>`` in the triage config. Also floors the onset walk (see
            ``onset_drop_db``): a sample still ``k_db`` above its own local floor counts as inside
            the event even where it falls more than ``onset_drop_db`` below the chosen peak.
        onset_drop_db: Walk back from the peak while still within ``onset_drop_db`` of it, *or*
            while still ``k_db`` above the local floor at that sample — measured onsets fitted this
            peak-anchored rule alone against a labelled benchmark (5 of 6 correct against 2 of 6 for
            a floor-referenced rule), so it stays the primary criterion; the floor-relative half is
            additive, not a replacement, added after a different benchmark: a sustained event whose
            envelope dips internally (a gain curve settling too slowly across a short event, or
            ordinary two-syllable amplitude modulation) can dip more than onset_drop_db below its own
            peak while remaining far above the local floor, which peak-anchored alone read as the
            event having already ended. The floor-relative half can only extend the walk further, so
            it cannot occur where the peak-anchored rule already fits the benchmark. Read it from
            ``spans.onset_drop_db``.
        offset_fraction: Walk forward to ``peak - offset_fraction * (peak - floor)``. Read it from
            ``spans.offset_fraction``.
        hangover_ms: The offset closes only after this long continuously below threshold. Must be
            shorter than the shortest event to be bounded. Read it from ``spans.hangover_ms``.
        min_duration_ms: Discard spans shorter than this. Read it from ``spans.min_duration_ms``.
        min_separation_ms: Minimum distance between two proposed peaks. Read it from
            ``spans.min_separation_ms``.

    Returns:
        Merged spans in time order, every extent bounded by measured samples, or
        :class:`NoContrast` when no peak clears ``k_db`` or nothing was measurable at all.
    """
    above = envelope_db - floor_db
    rise = np.where(np.isfinite(above), above, -np.inf)
    measured = np.where(np.isfinite(envelope_db), envelope_db, -np.inf)
    peaks, _ = find_peaks(rise, height=k_db, distance=int(min_separation_ms * sampling_rate / 1000))
    if len(peaks) == 0:
        rises = above[np.isfinite(above)]
        if rises.size == 0:
            return NoContrast(reason="the envelope holds no sample measurable against its local floor")
        return NoContrast(
            reason=f"no peak rose {k_db} dB above the local floor; the largest rose {float(rises.max()):.1f} dB"
        )
    hang = int(hangover_ms * sampling_rate / 1000)
    found: list[Span] = []
    for p in peaks:
        peak = float(envelope_db[p])
        i = int(p)
        while i > 0 and (measured[i] > peak - onset_drop_db or rise[i] >= k_db):
            i -= 1
        threshold = peak - offset_fraction * (peak - float(floor_db[p]))
        j = int(p)
        while j < len(envelope_db) - 1:
            window = measured[j : j + hang]
            if len(window) == 0 or window.max() <= threshold:
                break
            j += 1
        if (j - i) >= min_duration_ms * sampling_rate / 1000:
            found.append(
                Span(start=i / sampling_rate, end=j / sampling_rate, peak_over_floor_db=peak - float(floor_db[p]))
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
