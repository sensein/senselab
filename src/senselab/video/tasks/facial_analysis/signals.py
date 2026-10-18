"""Derived nonverbal signals: head gestures, blinks, gaze proxy, and audio-visual speaker linking."""

from __future__ import annotations

import itertools
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from senselab.video.tasks.facial_analysis.results import BlinkEvent, HeadGestureEvent, SpeakerLink


def fill_short_gaps(values: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly interpolate NaN runs of at most ``max_gap`` samples that have finite neighbours.

    Args:
        values: 1-D series.
        max_gap: Longest run filled.

    Returns:
        A copy with short interior gaps filled.
    """
    out = np.asarray(values, dtype=np.float64).copy()
    bad = ~np.isfinite(out)
    if not bad.any() or bad.all():
        return out
    idx = np.arange(out.size)
    padded = np.concatenate(([False], bad, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    for start, end in zip(edges[::2], edges[1::2]):
        if end - start <= max_gap and start > 0 and end < out.size:
            out[start:end] = np.interp(idx[start:end], [start - 1, end], [out[start - 1], out[end]])
    return out


def _finite_segments(values: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    good = np.isfinite(values)
    padded = np.concatenate(([False], good, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2]) if b - a >= min_len]


def bandpass(values: np.ndarray, fps: float, band_hz: Sequence[float]) -> np.ndarray:
    """Zero-phase Butterworth band-pass over each finite segment; NaN elsewhere.

    The upper edge is clipped to 0.45 of the sampling rate so low frame rates stay valid.

    Args:
        values: 1-D series.
        fps: Sampling rate.
        band_hz: ``(low, high)`` in Hz.

    Returns:
        The filtered series.
    """
    from scipy.signal import butter, sosfiltfilt

    low, high = float(band_hz[0]), min(float(band_hz[1]), 0.45 * fps)
    out = np.full(values.shape, np.nan)
    if high <= low:
        return out
    sos = butter(2, [low, high], btype="bandpass", fs=fps, output="sos")
    min_len = max(16, int(3 * fps / low) // 2)
    for a, b in _finite_segments(values, min_len):
        out[a:b] = sosfiltfilt(sos, values[a:b])
    return out


def _alternating_extrema(x: np.ndarray) -> List[int]:
    """Indices of local maxima and minima of a finite series, strictly alternating."""
    d = np.diff(x)
    sign = np.sign(d)
    for i in range(1, sign.size):
        if sign[i] == 0:
            sign[i] = sign[i - 1]
    turns = np.flatnonzero(np.diff(sign) != 0) + 1
    return [int(i) for i in turns]


def detect_head_gestures(
    times: np.ndarray,
    pitch_deg: np.ndarray,
    yaw_deg: np.ndarray,
    fps: float,
    tile_id: str = "T0",
    band_hz: Sequence[float] = (0.5, 4.0),
    min_swing_deg: float = 3.0,
    min_extrema: int = 3,
    max_half_period_s: float = 0.8,
    dominance_ratio: float = 1.5,
    max_gap_s: float = 0.3,
) -> List[HeadGestureEvent]:
    """Find nods (pitch oscillation) and shakes (yaw oscillation).

    Each angle is gap-filled, band-passed, and split at its alternating extrema. An event is a run
    of at least ``min_extrema`` extrema whose consecutive swings each reach ``min_swing_deg`` and are
    no more than ``max_half_period_s`` apart, and whose swing exceeds the other axis's swing over the
    same span by ``dominance_ratio``.

    Args:
        times: Sample times in seconds.
        pitch_deg: Head pitch.
        yaw_deg: Head yaw.
        fps: Sampling rate.
        tile_id: Tile label carried into the events.
        band_hz: Band-pass edges.
        min_swing_deg: Smallest peak-to-trough swing counted.
        min_extrema: Fewest alternating extrema per event.
        max_half_period_s: Largest spacing of consecutive extrema.
        dominance_ratio: Required ratio of this axis's swing to the other's.
        max_gap_s: Longest missing-face gap bridged by interpolation.

    Returns:
        Events sorted by start time.
    """
    times = np.asarray(times, dtype=np.float64)
    max_gap = max(0, int(round(max_gap_s * fps)))
    filtered = {
        "pitch": bandpass(fill_short_gaps(np.asarray(pitch_deg, float), max_gap), fps, band_hz),
        "yaw": bandpass(fill_short_gaps(np.asarray(yaw_deg, float), max_gap), fps, band_hz),
    }
    events: List[HeadGestureEvent] = []
    for axis, other, kind in (("pitch", "yaw", "nod"), ("yaw", "pitch", "shake")):
        x, y = filtered[axis], filtered[other]
        for a, b in _finite_segments(x, 3):
            seg = x[a:b]
            ext = [a + i for i in _alternating_extrema(seg)]
            run: List[int] = []

            def close(run: List[int]) -> None:
                if len(run) < min_extrema:
                    return
                swings = np.abs(np.diff(x[run]))
                lo, hi = run[0], run[-1] + 1
                other_swing = np.nanmax(y[lo:hi]) - np.nanmin(y[lo:hi]) if np.isfinite(y[lo:hi]).any() else 0.0
                if np.median(swings) < dominance_ratio * other_swing:
                    return
                span = times[run[-1]] - times[run[0]]
                events.append(
                    HeadGestureEvent(
                        tile_id=tile_id,
                        kind=kind,  # type: ignore[arg-type]
                        start_s=float(times[run[0]]),
                        end_s=float(times[run[-1]]),
                        n_extrema=len(run),
                        amplitude_deg=float(np.median(swings)),
                        frequency_hz=float((len(run) - 1) / (2 * span)) if span > 0 else 0.0,
                    )
                )

            for i in ext:
                if run:
                    swing = abs(x[i] - x[run[-1]])
                    gap = times[i] - times[run[-1]]
                    if swing >= min_swing_deg and gap <= max_half_period_s:
                        run.append(i)
                        continue
                    close(run)
                    # A failed link may still start the next run from the previous extremum.
                    run = [run[-1], i] if swing >= min_swing_deg and gap <= max_half_period_s else [i]
                else:
                    run = [i]
            close(run)
    return sorted(events, key=lambda e: e.start_s)


def detect_blinks(
    times: np.ndarray,
    blink_left: np.ndarray,
    blink_right: np.ndarray,
    tile_id: str = "T0",
    high_rise: float = 0.3,
    low_rise: float = 0.15,
    baseline_window_s: float = 2.0,
    max_duration_s: float = 0.5,
) -> List[BlinkEvent]:
    """Find blinks as brief rises of the eyeBlink blendshapes above their running baseline.

    The baseline is a centred rolling median of the mean of eyeBlinkLeft/Right, so a face whose
    resting coefficient is high (glasses, hooded lids, downward gaze) is not read as one long
    closure. A run of samples above ``baseline + low_rise`` is a blink when it peaks at least
    ``high_rise`` above baseline and lasts no longer than ``max_duration_s``.

    Args:
        times: Sample times in seconds.
        blink_left: eyeBlinkLeft.
        blink_right: eyeBlinkRight.
        tile_id: Tile label carried into the events.
        high_rise: Peak rise above baseline a blink must reach.
        low_rise: Rise above baseline that extends a blink.
        baseline_window_s: Rolling-median window for the baseline.
        max_duration_s: Longer runs are eye closures, not blinks.

    Returns:
        Blink events in time order; ``peak`` is the absolute coefficient at the peak.
    """
    times = np.asarray(times, dtype=np.float64)
    if times.size < 2:
        return []
    level = (np.asarray(blink_left, dtype=np.float64) + np.asarray(blink_right, dtype=np.float64)) / 2.0
    step = float(np.median(np.diff(times)))
    window = max(3, int(round(baseline_window_s / step)) if step > 0 else 3)
    baseline = _rolling_median(level, window)
    excess = level - baseline
    above = np.where(np.isfinite(excess), excess > low_rise, False)
    padded = np.concatenate(([False], above, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    events = []
    for a, b in zip(edges[::2], edges[1::2]):
        if np.nanmax(excess[a:b]) < high_rise:
            continue
        if times[b - 1] - times[a] > max_duration_s:
            continue
        peak = float(np.nanmax(level[a:b]))
        events.append(BlinkEvent(tile_id=tile_id, start_s=float(times[a]), end_s=float(times[b - 1]), peak=peak))
    return events


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    import pandas as pd

    series = pd.Series(np.asarray(values, dtype=np.float64))
    return series.rolling(window, center=True, min_periods=max(1, window // 4)).median().to_numpy()


def gaze_proxy(
    head_pitch_deg: np.ndarray,
    head_yaw_deg: np.ndarray,
    eye_horizontal: np.ndarray,
    eye_vertical: np.ndarray,
    eye_gain_deg: float = 60.0,
    sigma_deg: float = 10.0,
    eye_vertical_neutral: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine head pose and iris offset into a toward-camera score.

    Args:
        head_pitch_deg: Head pitch, positive up.
        head_yaw_deg: Head yaw, positive toward the subject's right (image left).
        eye_horizontal: Iris offset in eye widths, positive toward image right.
        eye_vertical: Iris offset in eye widths, positive downward.
        eye_gain_deg: Degrees of eye rotation per eye width of offset.
        sigma_deg: Width of the Gaussian score.
        eye_vertical_neutral: Vertical offset of a level gaze (the iris centre sits above the
            corner line), subtracted before scaling.

    Returns:
        ``(gaze_yaw_deg, gaze_pitch_deg, toward_camera_score)`` in the head-pose conventions; the
        score is 1 when the combined angle is 0.
    """
    gaze_yaw = np.asarray(head_yaw_deg, float) - eye_gain_deg * np.asarray(eye_horizontal, float)
    gaze_pitch = np.asarray(head_pitch_deg, float) - eye_gain_deg * (
        np.asarray(eye_vertical, float) - eye_vertical_neutral
    )
    score = np.exp(-0.5 * (gaze_yaw**2 + gaze_pitch**2) / sigma_deg**2)
    return gaze_yaw, gaze_pitch, score


def _rolling(values: np.ndarray, window: int, how: str) -> np.ndarray:
    import pandas as pd

    series = pd.Series(np.asarray(values, dtype=np.float64))
    roll = series.rolling(window, center=True, min_periods=max(1, window // 2))
    return (roll.std() if how == "std" else roll.mean()).to_numpy()


def segments_to_activity(segments: Sequence[Tuple[float, float]], times: np.ndarray) -> np.ndarray:
    """Binary activity sampled at ``times`` from ``(start_s, end_s)`` segments.

    Args:
        segments: Speech segments of one speaker.
        times: Sample times.

    Returns:
        ``1.0`` where any segment covers the time, else ``0.0``.
    """
    times = np.asarray(times, dtype=np.float64)
    out = np.zeros(times.shape)
    for start, end in segments:
        out[(times >= start) & (times < end)] = 1.0
    return out


def _pearson(a: np.ndarray, b: np.ndarray) -> Tuple[float, int]:
    ok = np.isfinite(a) & np.isfinite(b)
    n = int(ok.sum())
    if n < 3:
        return float("nan"), n
    x, y = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    return (float((x * y).sum() / denom) if denom > 0 else 0.0), n


def link_tiles_to_speakers(
    mouth_series_per_tile: Mapping[str, np.ndarray],
    speaker_activity_per_speaker: Mapping[str, np.ndarray],
    fps: float,
    window_s: float = 0.5,
    n_shifts: int = 200,
    min_shift_s: float = 10.0,
    seed: int = 0,
    visibility_per_tile: Optional[Mapping[str, np.ndarray]] = None,
) -> SpeakerLink:
    """Assign tiles to diarized speakers by correlating mouth motion with speech activity.

    Mouth activity is the rolling standard deviation of the mouth series (jawOpen) over
    ``window_s``; speaker activity is the binary diarization track smoothed by a rolling mean over
    the same window. All series must share one time grid at ``fps``. The Hungarian algorithm picks
    the one-to-one assignment with the highest total correlation; with more tiles than speakers (or
    the reverse) the surplus stays unassigned.

    Args:
        mouth_series_per_tile: ``tile_id -> series`` (NaN where no face).
        speaker_activity_per_speaker: ``speaker -> 0/1 series``.
        fps: Sampling rate of every series.
        window_s: Smoothing window.
        n_shifts: Circular shifts drawn for the null distribution of the best total correlation.
        min_shift_s: Smallest shift, so the null excludes near-aligned series.
        seed: Seed of the shift draw.
        visibility_per_tile: ``tile_id -> bool series``: where the tile's face is shown (a person
            track of a speaker-view layout). Mouth activity is NaN elsewhere, and the Pearson
            correlation of each visibility series with each smoothed speaker track is reported.

    Returns:
        Correlations, the assignment, per-tile margins, the total margin over the runner-up
        assignment, the shift-null p-value and, with visibility, the visibility-activity correlations.
    """
    from scipy.optimize import linear_sum_assignment

    tiles = list(mouth_series_per_tile)
    speakers = list(speaker_activity_per_speaker)
    if not tiles or not speakers:
        return SpeakerLink(correlations={}, assignment={}, margins={}, total_margin=0.0, p_value=1.0, n_frames=0)
    length = min(len(v) for v in itertools.chain(mouth_series_per_tile.values(), speaker_activity_per_speaker.values()))
    window = max(2, int(round(window_s * fps)))
    mouth = {t: _rolling(np.asarray(mouth_series_per_tile[t], float)[:length], window, "std") for t in tiles}
    speech = {
        s: _rolling(np.asarray(speaker_activity_per_speaker[s], float)[:length], window, "mean") for s in speakers
    }
    visibility_corr: Dict[str, Dict[str, float]] = {}
    for t, shown in (visibility_per_tile or {}).items():
        if t not in mouth:
            continue
        seen = np.asarray(shown, bool)[:length]
        mouth[t] = np.where(seen, mouth[t], np.nan)
        visibility_corr[t] = {}
        for s in speakers:
            r, _ = _pearson(seen.astype(float), speech[s])
            visibility_corr[t][s] = float(r) if np.isfinite(r) else 0.0

    def correlate(shift: int) -> Tuple[np.ndarray, int]:
        matrix = np.zeros((len(tiles), len(speakers)))
        used = 0
        for i, t in enumerate(tiles):
            series = np.roll(mouth[t], shift) if shift else mouth[t]
            for j, s in enumerate(speakers):
                r, n = _pearson(series, speech[s])
                matrix[i, j] = r if np.isfinite(r) else 0.0
                used = max(used, n)
        return matrix, used

    corr, n_used = correlate(0)
    rows, cols = linear_sum_assignment(-corr)
    assignment = {tiles[i]: speakers[j] for i, j in zip(rows, cols)}
    best_total = float(corr[rows, cols].sum())
    margins: Dict[str, float] = {}
    for i, j in zip(rows, cols):
        others = [corr[i, k] for k in range(len(speakers)) if k != j]
        margins[tiles[i]] = float(corr[i, j] - max(others)) if others else float(corr[i, j])
    runner_up = _second_best_total(corr, best_total)
    min_shift = int(round(min_shift_s * fps))
    exceed = 0
    drawn = 0
    if length > 2 * min_shift and n_shifts > 0:
        rng = np.random.default_rng(seed)
        for shift in rng.integers(min_shift, length - min_shift, size=n_shifts):
            null, _ = correlate(int(shift))
            r, c = linear_sum_assignment(-null)
            exceed += float(null[r, c].sum()) >= best_total
            drawn += 1
    p_value = (exceed + 1) / (drawn + 1)
    return SpeakerLink(
        correlations={t: {s: float(corr[i, j]) for j, s in enumerate(speakers)} for i, t in enumerate(tiles)},
        assignment=assignment,
        margins=margins,
        total_margin=float(best_total - runner_up) if runner_up is not None else best_total,
        p_value=float(p_value),
        n_frames=n_used,
        visibility_activity_corr=visibility_corr,
    )


def _second_best_total(corr: np.ndarray, best_total: float) -> Optional[float]:
    """Total correlation of the best assignment that differs from the optimum (exhaustive up to 7)."""
    n_rows, n_cols = corr.shape
    k = min(n_rows, n_cols)
    if max(n_rows, n_cols) > 7:
        return None
    totals = []
    for rows in itertools.combinations(range(n_rows), k):
        for cols in itertools.permutations(range(n_cols), k):
            totals.append(float(sum(corr[r, c] for r, c in zip(rows, cols))))
    totals.sort(reverse=True)
    for total in totals[1:]:
        if total < best_total - 1e-12:
            return total
    return totals[1] if len(totals) > 1 else None
