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
    """No peak anywhere rose the required amount above the local floor.

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

    Onset and offset walk by the identical rule, in opposite directions: neither can close for a
    reason the other would not also close for, unlike the peak-anchored/floor-fraction pair this
    replaced, whose asymmetry (see ``floor_margin_db``) let a real multi-scene recording collapse
    into one merged span once an unrelated bug stopped masking it.

    Args:
        envelope_db: Envelope in dBFS, ``nan`` at any sample that had no dB value.
        floor_db: The recording's own global floor — see
            :func:`~senselab.audio.tasks.envelope.api.global_floor_dbfs` — one value for the whole
            signal, not a local, time-varying one.
        sampling_rate: Samples per second.
        k_db: How far above the floor a peak must rise to be proposed. Per reader: read it from
            ``spans.k_db.<reader>`` in the triage config.
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
            return NoContrast(reason="the envelope holds no sample measurable against its floor")
        return NoContrast(
            reason=f"no peak rose {k_db} dB above the floor; the largest rose {float(rises.max()):.1f} dB"
        )
    threshold = floor_db + floor_margin_db
    win = int(transition_window_ms * sampling_rate / 1000)
    found: list[Span] = []
    for p in peaks:
        peak = float(envelope_db[p])
        i = int(p)
        while i > 0:
            window = measured[max(0, i - win) : i]
            if len(window) == 0 or window.max() <= threshold:
                break
            i -= 1
        j = int(p)
        while j < len(envelope_db) - 1:
            window = measured[j : j + win]
            if len(window) == 0 or window.max() <= threshold:
                break
            j += 1
        if (j - i) >= min_duration_ms * sampling_rate / 1000:
            found.append(Span(start=i / sampling_rate, end=j / sampling_rate, peak_over_floor_db=peak - floor_db))
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
