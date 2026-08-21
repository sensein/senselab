"""The HNR track and the glottal period marks, both through Praat."""

from __future__ import annotations

import numpy as np
import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.phonation import PeriodMark, hnr_track, period_marks

SR = 16000


def _buzz(f0: float, seconds: float = 1.0) -> Audio:
    """Return a harmonic buzz at the given F0."""
    t = np.arange(int(seconds * SR)) / SR
    wave = sum((0.3 / (h + 1) * np.sin(2 * np.pi * f0 * (h + 1) * t) for h in range(6)), np.zeros_like(t))
    return Audio(waveform=wave.astype(np.float32)[None, :], sampling_rate=SR)


def _noise(seconds: float = 1.0) -> Audio:
    """Return seeded white noise."""
    rng = np.random.default_rng(0)
    return Audio(waveform=(0.1 * rng.standard_normal(int(seconds * SR))).astype(np.float32)[None, :], sampling_rate=SR)


def _hnr(audio: Audio) -> np.ndarray:
    """Run hnr_track with Praat's documented cc settings; the fixture supplies the literals."""
    _, hnr_db = hnr_track(audio, f0_min_hz=60.0, hop_s=0.01, silence_threshold=0.1, periods_per_window=4.5)
    return hnr_db


class TestHnr:
    """The gate's measurement is harmonicity in dB, from Praat's cc method."""

    def test_a_buzz_is_harmonic_and_noise_is_not(self) -> None:
        """A 100 Hz harmonic buzz reads far above 20 dB HNR; white noise reads below 0 dB."""
        assert float(np.median(_hnr(_buzz(100.0)))) > 20.0
        assert float(np.median(_hnr(_noise()))) < 0.0

    def test_the_track_carries_its_own_times(self) -> None:
        """Times and values come back paired, equally long and increasing."""
        times, hnr_db = hnr_track(
            _buzz(100.0), f0_min_hz=60.0, hop_s=0.01, silence_threshold=0.1, periods_per_window=4.5
        )
        assert times.shape == hnr_db.shape
        assert np.all(np.diff(times) > 0)

    def test_every_parameter_is_required(self) -> None:
        """No default stands in for a value the caller did not choose."""
        with pytest.raises(TypeError):
            hnr_track(_buzz(100.0))  # type: ignore[call-arg]


class TestPeriodMarks:
    """Pulse times come from Praat's point process, not from integer-lag multiples."""

    def test_marks_are_spaced_by_one_period(self) -> None:
        """A 100 Hz buzz yields pulses one 10 ms period apart across the span."""
        marks = period_marks(_buzz(100.0), 0.2, 0.8, f0_min_hz=60.0, f0_max_hz=400.0)
        assert len(marks) > 40
        gaps = np.diff([m.time_s for m in marks])
        assert float(np.median(gaps)) == pytest.approx(0.01, abs=0.001)

    def test_each_mark_carries_its_period_and_amplitude(self) -> None:
        """Jitter and shimmer read consecutive periods, so each mark carries both quantities."""
        marks = period_marks(_buzz(100.0), 0.2, 0.4, f0_min_hz=60.0, f0_max_hz=400.0)
        m = marks[len(marks) // 2]
        assert isinstance(m, PeriodMark)
        assert m.period_s == pytest.approx(0.01, abs=0.002)
        assert m.amplitude > 0.0

    def test_mark_times_stay_in_the_recordings_clock(self) -> None:
        """Praat places pulses on the waveform inside the span, in absolute time."""
        marks = period_marks(_buzz(100.0), 0.2, 0.8, f0_min_hz=60.0, f0_max_hz=400.0)
        assert 0.2 <= marks[0].time_s <= 0.25
        assert marks[-1].time_s <= 0.8

    def test_noise_yields_no_marks(self) -> None:
        """White noise gives Praat no periodic pulses, so the answer is absent, not zero."""
        assert period_marks(_noise(), 0.2, 0.8, f0_min_hz=60.0, f0_max_hz=400.0) == []
