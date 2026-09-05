"""Harmonicity, glottal period marks, formant tracks and phonation spans, through Praat."""

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


@dataclass(frozen=True)
class FormantTrack:
    """Formant frequencies and bandwidths over one stream, on a fixed hop.

    Attributes:
        times_s: Frame times, in seconds.
        f_hz: Four arrays, F1 to F4, each one value per frame. NaN where Praat placed none.
        bandwidth_hz: Four arrays, the corresponding 3 dB bandwidths. NaN where the formant is NaN.
    """

    times_s: np.ndarray
    f_hz: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    bandwidth_hz: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def formant_track(
    audio: Audio,
    *,
    hop_s: float,
    max_formants: int,
    formant_max_hz: float,
    window_s: float,
    preemphasis_hz: float,
) -> FormantTrack:
    """F1-F4 and their bandwidths over the whole stream, by Praat's Burg method.

    Computed once over the stream so a consumer slices the track rather than re-tracking a fragment.

    Args:
        audio: The recording. ``get_sound`` handles channel merging and resampling.
        hop_s: Praat's ``time_step``. Read it from ``phonation_spans.hop_s``.
        max_formants: Praat's ``max_number_of_formants``. Read it from ``phonation_spans.max_formants``.
        formant_max_hz: Praat's ``maximum_formant``. Read it from ``phonation_spans.formant_max_hz``.
        window_s: Praat's ``window_length``. Read it from ``phonation_spans.formant_window_s``.
        preemphasis_hz: Praat's ``pre_emphasis_from``. Read it from
            ``phonation_spans.formant_preemphasis_hz``.

    Returns:
        The track. A frame where Praat placed no formant carries NaN in both arrays, so a missing
        formant is absent rather than zero.

    Raises:
        ModuleNotFoundError: If parselmouth is not installed.
    """
    _require_parselmouth()
    snd = get_sound(audio)
    formants = snd.to_formant_burg(
        time_step=hop_s,
        max_number_of_formants=max_formants,
        maximum_formant=formant_max_hz,
        window_length=window_s,
        pre_emphasis_from=preemphasis_hz,
    )
    times = np.asarray(formants.xs(), dtype=np.float64)
    values: list[np.ndarray] = []
    bandwidths: list[np.ndarray] = []
    for order in (1, 2, 3, 4):
        values.append(
            np.asarray(
                [formants.get_value_at_time(order, t, unit=parselmouth.FormantUnit.HERTZ) for t in times],
                dtype=np.float64,
            )
        )
        bandwidths.append(
            np.asarray(
                [formants.get_bandwidth_at_time(order, t, unit=parselmouth.FormantUnit.HERTZ) for t in times],
                dtype=np.float64,
            )
        )
    return FormantTrack(
        times_s=times,
        f_hz=(values[0], values[1], values[2], values[3]),
        bandwidth_hz=(bandwidths[0], bandwidths[1], bandwidths[2], bandwidths[3]),
    )


_CENTS_PER_OCTAVE = 1200.0


def _cents(lower: float, upper: float) -> float:
    """The interval between two frequencies in cents, or NaN when either is not positive."""
    if not (lower > 0.0) or not (upper > 0.0):
        return float("nan")
    return float(_CENTS_PER_OCTAVE * np.log2(upper / lower))
