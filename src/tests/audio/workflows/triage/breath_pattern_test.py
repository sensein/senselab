"""Tests for the breathing-pattern measure."""

import numpy as np

from senselab.audio.workflows.triage.breath_pattern import (
    ALTERNATING_BREATHS,
    NO_BREATHING,
    SINGLE_BREATH,
    breath_pattern_parameters,
    measure_breath_pattern,
    measure_modulation,
    modulation_parameters,
)

HOP_S = 0.005
BIN_HZ = 50.0
BINS = 161


def _spectrogram(duration_s: float, breaths: list[tuple[float, float]], level_db: float = 30.0) -> np.ndarray:
    """A noise floor with flat broadband bursts at each (start, length), level_db above the floor."""
    rng = np.random.default_rng(0)
    frames = int(duration_s / HOP_S)
    power = rng.uniform(0.5, 1.5, size=(BINS, frames)) * 1e-6
    for start, length in breaths:
        a, b = int(start / HOP_S), int((start + length) / HOP_S)
        power[:, a:b] *= 10 ** (level_db / 10)
    return power


def _measure(power: np.ndarray, voiced: np.ndarray | None = None) -> str:
    frames = power.shape[1]
    mask = np.zeros(frames, dtype=bool) if voiced is None else voiced
    return measure_breath_pattern(
        power, hop_s=HOP_S, bin_hz=BIN_HZ, voiced=mask, parameters=breath_pattern_parameters()
    ).pattern


def test_alternating_breaths_read_as_a_pattern() -> None:
    """Five 1 s breaths every 3 s over a quiet floor are alternating breaths."""
    assert _measure(_spectrogram(18.0, [(1 + 3 * k, 1.0) for k in range(5)])) == ALTERNATING_BREATHS


def test_one_long_breath_is_a_single_breath() -> None:
    """One 1.6 s breath, as 7c169ccc held where three quick ones were asked."""
    assert _measure(_spectrogram(8.5, [(3.0, 1.6)])) == SINGLE_BREATH


def test_a_silent_recording_has_no_breathing() -> None:
    """The noise floor alone holds no breath."""
    assert _measure(_spectrogram(20.0, [])) == NO_BREATHING


def test_breaths_below_the_rise_are_no_breathing() -> None:
    """Breaths only 3 dB above the floor are too quiet to count: the owner discards them."""
    assert _measure(_spectrogram(18.0, [(1 + 3 * k, 1.0) for k in range(5)], level_db=3.0)) == NO_BREATHING


def test_a_voiced_burst_is_not_a_breath() -> None:
    """A burst voiced over most of its length is speech, not breath."""
    power = _spectrogram(10.0, [(4.0, 1.0)])
    voiced = np.zeros(power.shape[1], dtype=bool)
    voiced[int(4.0 / HOP_S) : int(5.0 / HOP_S)] = True
    assert _measure(power, voiced) == NO_BREATHING


def _modulation(power: np.ndarray):  # noqa: ANN202
    return measure_modulation(
        power,
        hop_s=HOP_S,
        bin_hz=BIN_HZ,
        smooth_s=breath_pattern_parameters().smooth_s,
        parameters=modulation_parameters(),
    )


def test_five_breath_cycles_read_as_five_breaths() -> None:
    """An inhale and an exhale per breath make two peaks each; ten peaks are five breaths."""
    bursts = [(2 + k * 5.5 + offset, length) for k in range(5) for offset, length in ((0.5, 1.0), (2.0, 1.4))]
    reading = _modulation(_spectrogram(30.0, bursts))
    assert reading is not None
    assert reading.breathing and not reading.speech_like
    assert reading.estimated_breaths == 5


def test_syllabic_bursts_trip_the_speech_guard() -> None:
    """Bursts at a syllabic rate are speech, not breathing."""
    bursts = [(2 + k * 0.25, 0.12) for k in range(100)]
    reading = _modulation(_spectrogram(30.0, bursts))
    assert reading is not None
    assert reading.speech_like and not reading.breathing


def test_silence_has_no_breathing_cycles() -> None:
    """A floor with no burst has no active span, so no cycles count."""
    reading = _modulation(_spectrogram(30.0, []))
    assert reading is not None
    assert not reading.breathing and reading.active_span_s == 0


def test_a_recording_shorter_than_the_reading_needs_has_none() -> None:
    """Under min_duration_s the modulation spectrum is not read."""
    assert _modulation(_spectrogram(2.0, [(0.5, 1.0)])) is None


def test_an_odd_peak_count_rounds_up_to_the_unpaired_breath() -> None:
    """Nine peaks are an unpaired burst beside four full breaths: five breaths."""
    bursts = [(2 + k * 5.5 + offset, length) for k in range(5) for offset, length in ((0.5, 1.0), (2.0, 1.4))][:9]
    reading = _modulation(_spectrogram(30.0, bursts))
    assert reading is not None
    assert reading.modulation_peaks == 9
    assert reading.estimated_breaths == 5
