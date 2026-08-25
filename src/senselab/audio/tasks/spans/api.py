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
        envelope_db: Envelope in dBFS.
        floor_db: Local floor, same length as ``envelope_db``.
        sampling_rate: Samples per second.
        k_db: How far above the local floor a peak must rise to be proposed. Per reader: read it
            from ``spans.k_db.<reader>`` in the triage config.
        onset_drop_db: Walk back from the peak to ``peak - onset_drop_db``. Read it from
            ``spans.onset_drop_db``.
        offset_fraction: Walk forward to ``peak - offset_fraction * (peak - floor)``. Read it from
            ``spans.offset_fraction``.
        hangover_ms: The offset closes only after this long continuously below threshold. Must be
            shorter than the shortest event to be bounded. Read it from ``spans.hangover_ms``.
        min_duration_ms: Discard spans shorter than this. Read it from ``spans.min_duration_ms``.
        min_separation_ms: Minimum distance between two proposed peaks. Read it from
            ``spans.min_separation_ms``.

    Returns:
        Merged spans in time order, or :class:`NoContrast` when no peak clears ``k_db``.
    """
    above = envelope_db - floor_db
    peaks, _ = find_peaks(above, height=k_db, distance=int(min_separation_ms * sampling_rate / 1000))
    if len(peaks) == 0:
        return NoContrast(
            reason=f"no peak rose {k_db} dB above the local floor; the largest rose {float(above.max()):.1f} dB"
        )
    hang = int(hangover_ms * sampling_rate / 1000)
    found: list[Span] = []
    for p in peaks:
        peak = float(envelope_db[p])
        i = int(p)
        while i > 0 and envelope_db[i] > peak - onset_drop_db:
            i -= 1
        threshold = peak - offset_fraction * (peak - float(floor_db[p]))
        j = int(p)
        while j < len(envelope_db) - 1:
            window = envelope_db[j : j + hang]
            if len(window) and window.max() <= threshold:
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
