"""Harmonicity and glottal period marks, through Praat."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.features_extraction.praat_parselmouth import (
    PARSELMOUTH_AVAILABLE,
    get_sound,
    parselmouth,
)


@dataclass(frozen=True)
class PeriodMark:
    """One glottal period, delimited by two consecutive Praat pulses.

    Attributes:
        time_s: The opening pulse, in the recording's clock.
        period_s: Time to the next pulse.
        amplitude: Peak absolute amplitude of the waveform within the period.
    """

    time_s: float
    period_s: float
    amplitude: float


def _require_parselmouth() -> None:
    """Raise when parselmouth is not installed."""
    if not PARSELMOUTH_AVAILABLE:
        raise ModuleNotFoundError(
            "`parselmouth` is not installed. Please install senselab audio dependencies using `pip install senselab`."
        )


def hnr_track(
    audio: Audio,
    *,
    f0_min_hz: float,
    hop_s: float,
    silence_threshold: float,
    periods_per_window: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Harmonics-to-noise ratio over time, in dB, via Praat's cc method.

    Args:
        audio: The recording. ``get_sound`` handles channel merging and resampling.
        f0_min_hz: Lowest F0 the analysis considers. Read it from ``voice.f0_range_hz``.
        hop_s: Praat's ``time_step``. Read it from ``phonation.hop_s``.
        silence_threshold: Praat's silence threshold. Read it from ``phonation.silence_threshold``.
        periods_per_window: Praat's analysis window length, in periods of ``f0_min_hz``. Read it
            from ``phonation.periods_per_window``.

    Returns:
        ``(times_s, hnr_db)``, one value per frame. Frames Praat judges silent carry its floor
        value rather than being dropped, so the track and its times stay aligned.

    Raises:
        ModuleNotFoundError: If parselmouth is not installed.
    """
    _require_parselmouth()
    snd = get_sound(audio)
    harmonicity = snd.to_harmonicity_cc(
        time_step=hop_s,
        minimum_pitch=f0_min_hz,
        silence_threshold=silence_threshold,
        periods_per_window=periods_per_window,
    )
    times = np.asarray(harmonicity.xs(), dtype=np.float64)
    values = np.asarray(harmonicity.values, dtype=np.float64).squeeze(0)
    return times, values


def period_marks(
    audio: Audio,
    start_s: float,
    end_s: float,
    *,
    f0_min_hz: float,
    f0_max_hz: float,
) -> list[PeriodMark]:
    """Glottal period marks inside one span, from Praat's point process.

    Args:
        audio: The recording.
        start_s: Span onset.
        end_s: Span offset.
        f0_min_hz: Lowest F0 to search. Read it from ``voice.f0_range_hz``.
        f0_max_hz: Highest F0 to search. Read it from ``voice.f0_range_hz``.

    Returns:
        One mark per pair of consecutive pulses whose gap is a plausible period — within
        ``[1/f0_max_hz, 1/f0_min_hz]``; a longer gap is a voicing break, not a period. Empty when
        Praat places no pulses — absent, not zero.

    Raises:
        ModuleNotFoundError: If parselmouth is not installed.
    """
    _require_parselmouth()
    snd = get_sound(audio)
    part = snd.extract_part(from_time=start_s, to_time=end_s, preserve_times=True)
    point_process = parselmouth.praat.call(part, "To PointProcess (periodic, cc)", f0_min_hz, f0_max_hz)
    n_points = int(parselmouth.praat.call(point_process, "Get number of points"))
    pulses = [float(parselmouth.praat.call(point_process, "Get time from index", i)) for i in range(1, n_points + 1)]
    x = np.asarray(part.values, dtype=np.float64).squeeze(0)
    t0 = float(part.xmin)
    sr = 1.0 / float(part.dx)
    marks: list[PeriodMark] = []
    for opening, closing in zip(pulses, pulses[1:]):
        period = closing - opening
        if not (1.0 / f0_max_hz <= period <= 1.0 / f0_min_hz):
            continue
        i0 = max(0, int((opening - t0) * sr))
        i1 = min(len(x), int((closing - t0) * sr))
        amplitude = float(np.abs(x[i0:i1]).max()) if i1 > i0 else 0.0
        marks.append(PeriodMark(time_s=opening, period_s=period, amplitude=amplitude))
    return marks


def f0_track(
    audio: Audio,
    *,
    f0_min_hz: float,
    f0_max_hz: float,
    hop_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """F0 and its strength per frame via Praat's cc pitch: ``(times_s, f0_hz, strength)``.

    Unvoiced frames carry NaN in ``f0_hz`` with their ``strength`` retained, so F0 always travels
    with the periodicity that placed it and a reader cannot separate them.

    Args:
        audio: The recording. ``get_sound`` handles channel merging and resampling.
        f0_min_hz: Lowest F0 to search. Read it from ``voice.f0_range_hz``.
        f0_max_hz: Highest F0 to search. Read it from ``voice.f0_range_hz``.
        hop_s: Praat's ``time_step``. Read it from ``phonation.hop_s``.

    Returns:
        ``(times_s, f0_hz, strength)``, one value per frame, all three the same length.

    Raises:
        ModuleNotFoundError: If parselmouth is not installed.
    """
    _require_parselmouth()
    snd = get_sound(audio)
    pitch = snd.to_pitch_cc(time_step=hop_s, pitch_floor=f0_min_hz, pitch_ceiling=f0_max_hz)
    times = np.asarray(pitch.xs(), dtype=np.float64)
    f0 = np.asarray(pitch.selected_array["frequency"], dtype=np.float64)
    strength = np.asarray(pitch.selected_array["strength"], dtype=np.float64)
    f0[f0 == 0.0] = np.nan
    return times, f0, strength
