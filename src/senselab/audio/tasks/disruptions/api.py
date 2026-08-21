"""Detecting recording disruptions within a span.

Counts and extents, never a score. How much disruption makes a span unusable is a tolerance nobody has
derived, and it is the caller's decision rather than this module's.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from senselab.audio.data_structures import Audio


@dataclass(frozen=True)
class Disruptions:
    """What was found in one span.

    Attributes:
        start: Span onset in seconds.
        end: Span offset in seconds.
        clipped_runs: Number of runs of consecutive samples at or beyond the headroom.
        clipped_s: Total duration of those runs.
        dropout_runs: Number of runs of exact zeros at least ``min_dropout_ms`` long.
        dropout_s: Total duration of those runs.
        discontinuities: Number of sample-to-sample jumps exceeding the threshold.
        dc_offset: Mean sample value over the span.
    """

    start: float
    end: float
    clipped_runs: int
    clipped_s: float
    dropout_runs: int
    dropout_s: float
    discontinuities: int
    dc_offset: float


def _runs(mask: np.ndarray, minimum: int) -> tuple[int, int]:
    """Count runs of True at least ``minimum`` long, and their total length.

    A run touching the first or last element of ``mask`` counts, measured by the extent
    visible in ``mask``.

    Args:
        mask: Boolean array.
        minimum: Shortest run that counts.

    Returns:
        ``(run_count, total_samples)``.
    """
    if not mask.any():
        return 0, 0
    edges = np.diff(mask.astype(np.int8))
    starts = [int(i) for i in np.flatnonzero(edges == 1) + 1]
    ends = [int(i) for i in np.flatnonzero(edges == -1) + 1]
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(len(mask))
    lengths = [e - s for s, e in zip(starts, ends) if e - s >= minimum]
    return len(lengths), int(sum(lengths))


def detect_disruptions(
    audio: Audio,
    start_s: float,
    end_s: float,
    *,
    clip_headroom: float,
    min_clip_run: int,
    min_dropout_ms: float,
    discontinuity_threshold: float,
) -> Disruptions:
    """Measure disruptions inside one span.

    A clipped or zero run that touches the span's start or end counts, measured only by its
    extent inside the span; samples outside the span are never read.

    Args:
        audio: The recording. A multi-channel input is averaged.
        start_s: Span onset.
        end_s: Span offset.
        clip_headroom: A sample at or beyond this absolute value counts as clipped. Read it from
            ``disruptions.clip_headroom``.
        min_clip_run: Shortest run of clipped samples that counts as a clipping event. Read it from
            ``disruptions.min_clip_run``.
        min_dropout_ms: Shortest run of exact zeros that counts as a dropout. Read it from
            ``disruptions.min_dropout_ms``.
        discontinuity_threshold: Absolute sample-to-sample jump that counts as a discontinuity. Read
            it from ``disruptions.discontinuity_threshold``.

    Returns:
        The span's disruptions. Every count is exact; a clean span reports zeros.
    """
    x = np.asarray(audio.waveform, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=0)
    sr = audio.sampling_rate
    segment = x[max(0, int(start_s * sr)) : min(len(x), int(end_s * sr))]
    if segment.size == 0:
        return Disruptions(start_s, end_s, 0, 0.0, 0, 0.0, 0, 0.0)
    clip_runs, clip_n = _runs(np.abs(segment) >= clip_headroom, min_clip_run)
    drop_runs, drop_n = _runs(segment == 0.0, max(1, int(min_dropout_ms * sr / 1000)))
    jumps = int(np.count_nonzero(np.abs(np.diff(segment)) > discontinuity_threshold))
    return Disruptions(
        start=start_s,
        end=end_s,
        clipped_runs=clip_runs,
        clipped_s=clip_n / sr,
        dropout_runs=drop_runs,
        dropout_s=drop_n / sr,
        discontinuities=jumps,
        dc_offset=float(segment.mean()),
    )
