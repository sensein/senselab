"""Detecting recording disruptions within a span.

Counts and extents, never a score, alongside two plain readings of the span — its DC offset and its
zero-crossing rate. How much disruption makes a span unusable is a tolerance nobody has derived, and
it is the caller's decision rather than this module's. The discontinuity criterion is locally
referenced: a jump is measured against the variation of the windows flanking it, not against an
absolute number.
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
        discontinuities: Number of sample-to-sample jumps large against the local variation.
        dc_offset: Mean sample value over the span.
        zero_crossing_rate: Sign changes per second over the span.
    """

    start: float
    end: float
    clipped_runs: int
    clipped_s: float
    dropout_runs: int
    dropout_s: float
    discontinuities: int
    dc_offset: float
    zero_crossing_rate: float


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


def _local_variation(x: np.ndarray, window: int) -> np.ndarray:
    """Per jump ``i -> i+1``, the larger standard deviation of the two windows flanking it.

    Neither window contains the jump, so the reference cannot be inflated by what it is measuring,
    and a standard deviation rather than an RMS so that a constant offset — reported separately as
    ``dc_offset`` — is not mistaken for variation. Taking the larger of the two sides means an onset,
    where a quiet window abuts a loud one, is referenced to the loud side.

    Args:
        x: The span's samples.
        window: Window length in samples.

    Returns:
        One reference value per jump, of length ``len(x) - 1``.
    """
    sums = np.cumsum(np.concatenate([[0.0], x]))
    squares = np.cumsum(np.concatenate([[0.0], x * x]))

    def deviation(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        n = hi - lo
        mean = (sums[hi] - sums[lo]) / n
        return np.sqrt(np.maximum((squares[hi] - squares[lo]) / n - mean * mean, 0.0))

    i = np.arange(len(x) - 1)
    before = deviation(np.maximum(i + 1 - window, 0), i + 1)
    after = deviation(i + 1, np.minimum(i + 1 + window, len(x)))
    return np.maximum(before, after)


def detect_disruptions(
    audio: Audio,
    start_s: float,
    end_s: float,
    *,
    clip_headroom: float,
    min_clip_run: int,
    min_dropout_ms: float,
    discontinuity_local_factor: float,
    discontinuity_window_ms: float,
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
        discontinuity_local_factor: How many times the local variation a sample-to-sample jump must
            exceed to count as a discontinuity. Read it from ``disruptions.discontinuity_local_factor``.
        discontinuity_window_ms: Length of each of the two windows flanking a jump that the local
            variation is measured over. Read it from ``disruptions.discontinuity_window_ms``.

    Returns:
        The span's disruptions, with its DC offset and zero-crossing rate. Every count is exact; a
        clean span reports zeros.
    """
    x = np.asarray(audio.waveform, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=0)
    sr = audio.sampling_rate
    segment = x[max(0, int(start_s * sr)) : min(len(x), int(end_s * sr))]
    if segment.size == 0:
        return Disruptions(start_s, end_s, 0, 0.0, 0, 0.0, 0, 0.0, 0.0)
    clip_runs, clip_n = _runs(np.abs(segment) >= clip_headroom, min_clip_run)
    drop_runs, drop_n = _runs(segment == 0.0, max(1, int(min_dropout_ms * sr / 1000)))
    window = max(1, int(discontinuity_window_ms * sr / 1000))
    reference = _local_variation(segment, window) if segment.size > 1 else np.empty(0)
    jumps = int(np.count_nonzero(np.abs(np.diff(segment)) > discontinuity_local_factor * reference))
    crossings = int(np.count_nonzero(np.diff(np.signbit(segment))))
    return Disruptions(
        start=start_s,
        end=end_s,
        clipped_runs=clip_runs,
        clipped_s=clip_n / sr,
        dropout_runs=drop_runs,
        dropout_s=drop_n / sr,
        discontinuities=jumps,
        dc_offset=float(segment.mean()),
        zero_crossing_rate=crossings * sr / segment.size,
    )
