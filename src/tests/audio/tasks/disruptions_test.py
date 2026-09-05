"""Recording disruptions: clipping, dropouts, discontinuities, DC offset."""

from __future__ import annotations

import numpy as np
import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.disruptions import Disruptions, detect_disruptions

SR = 16000


def _audio(x: np.ndarray) -> Audio:
    return Audio(waveform=x.astype("float32")[None, :], sampling_rate=SR)


def _tone(seconds: float = 1.0, amp: float = 0.5, freq: float = 200.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return amp * np.sin(2 * np.pi * freq * t)


def _detect(audio: Audio, start_s: float, end_s: float) -> Disruptions:
    """Run detect_disruptions with the config's conventional values; the fixture supplies the literals."""
    return detect_disruptions(
        audio,
        start_s,
        end_s,
        clip_headroom=0.999,
        min_clip_run=3,
        min_dropout_ms=10.0,
        discontinuity_local_factor=10.0,
        discontinuity_window_ms=20.0,
    )


class TestClipping:
    """Clipping is a run of samples at the headroom, never a single sample."""

    def test_a_clean_tone_has_no_clipping(self) -> None:
        """A half-scale tone reports zero clipped runs and zero clipped time."""
        d = _detect(_audio(_tone()), 0.0, 1.0)
        assert d.clipped_runs == 0
        assert d.clipped_s == 0.0

    def test_a_saturated_tone_is_clipped(self) -> None:
        """A tone clipped at full scale reports a run on every half cycle."""
        d = _detect(_audio(np.clip(_tone(amp=2.0), -1.0, 1.0)), 0.0, 1.0)
        assert d.clipped_runs >= 100, "200 Hz for 1 s saturates on every half cycle"
        assert d.clipped_s > 0.1

    def test_a_single_full_scale_sample_is_not_clipping(self) -> None:
        """One sample at full scale is below min_clip_run and does not count."""
        x = _tone()
        x[5000] = 1.0
        assert _detect(_audio(x), 0.0, 1.0).clipped_runs == 0


class TestDropouts:
    """A dropout is a run of exact zeros at least min_dropout_ms long."""

    def test_a_zero_run_is_a_dropout(self) -> None:
        """A 250 ms zero run is one dropout of that duration."""
        x = _tone()
        x[4000:8000] = 0.0
        d = _detect(_audio(x), 0.0, 1.0)
        assert d.dropout_runs == 1
        assert d.dropout_s == pytest.approx(4000 / SR, abs=0.002)

    def test_a_run_shorter_than_the_minimum_is_not_a_dropout(self) -> None:
        """A 20-sample zero run is shorter than 10 ms and does not count."""
        x = _tone()
        x[4000:4020] = 0.0
        assert _detect(_audio(x), 0.0, 1.0).dropout_runs == 0


class TestDiscontinuities:
    """A discontinuity is a jump large against the signal's own local variation, not against a fixed number."""

    def test_a_step_is_a_discontinuity(self) -> None:
        """A 0.9 step in an otherwise smooth tone is counted."""
        x = _tone(amp=0.1)
        x[8000:] += 0.9
        assert _detect(_audio(x), 0.0, 1.0).discontinuities >= 1

    def test_a_smooth_tone_has_none(self) -> None:
        """A continuous tone produces zero discontinuities."""
        assert _detect(_audio(_tone()), 0.0, 1.0).discontinuities == 0

    def test_a_loud_high_frequency_tone_is_not_a_field_of_discontinuities(self) -> None:
        """An absolute jump threshold measures high-frequency energy; the local reference does not.

        A full-scale 3 kHz tone at 16 kHz steps 1.18 between neighbouring samples at every zero
        crossing, so an absolute 0.5 rule calls almost every such sample a defect. Nothing about the
        tone is discontinuous: each jump is ordinary against the local variation that produced it.
        """
        x = 1.0 * np.sin(2 * np.pi * 3000.0 * np.arange(SR) / SR)
        absolute = int(np.count_nonzero(np.abs(np.diff(x)) > 0.5))
        assert absolute > 10_000, "the fixture must actually defeat the absolute rule"
        assert _detect(_audio(x), 0.0, 1.0).discontinuities == 0

    def test_the_reference_is_local_so_the_same_jump_reads_differently_by_context(self) -> None:
        """One 0.9 jump against a quiet neighbourhood counts; the same jump against a loud one does not."""
        quiet = _tone(amp=0.05)
        quiet[8000:] += 0.9
        assert _detect(_audio(quiet), 0.0, 1.0).discontinuities == 1
        loud = _tone(amp=0.9)
        loud[8000:] += 0.9
        assert _detect(_audio(loud), 0.0, 1.0).discontinuities == 0


class TestZeroCrossingRate:
    """Zero-crossing rate is a plain reading of the span, in crossings per second."""

    def test_a_tone_reports_twice_its_frequency(self) -> None:
        """A 200 Hz tone crosses zero 400 times a second."""
        assert _detect(_audio(_tone(freq=200.0)), 0.0, 1.0).zero_crossing_rate == pytest.approx(400.0, abs=2.0)

    def test_a_higher_tone_reports_a_higher_rate(self) -> None:
        """The reading tracks the signal, not the sample rate."""
        low = _detect(_audio(_tone(freq=200.0)), 0.0, 1.0).zero_crossing_rate
        high = _detect(_audio(_tone(freq=800.0)), 0.0, 1.0).zero_crossing_rate
        assert high == pytest.approx(4 * low, rel=0.02)


class TestDcOffset:
    """DC offset is the mean sample value over the span."""

    def test_a_bias_is_reported(self) -> None:
        """A +0.2 bias reads back as 0.2."""
        d = _detect(_audio(_tone() + 0.2), 0.0, 1.0)
        assert d.dc_offset == pytest.approx(0.2, abs=0.01)

    def test_a_centred_signal_reports_near_zero(self) -> None:
        """A zero-mean tone reads near zero."""
        assert abs(_detect(_audio(_tone()), 0.0, 1.0).dc_offset) < 0.01


class TestScoping:
    """Measurement is scoped to the requested span."""

    def test_only_the_requested_span_is_measured(self) -> None:
        """Clipping in the first second is invisible to a span that excludes it."""
        x = _tone(seconds=3.0)
        x[:SR] = np.clip(_tone(amp=2.0)[:SR], -1.0, 1.0)
        assert _detect(_audio(x), 1.5, 2.5).clipped_runs == 0
        assert _detect(_audio(x), 0.0, 1.0).clipped_runs > 0

    def test_a_clean_span_reports_zero_rather_than_nothing(self) -> None:
        """A clean span is zeros — a different statement from a span nobody measured."""
        d = _detect(_audio(_tone()), 0.0, 1.0)
        assert isinstance(d, Disruptions)
        assert (d.clipped_runs, d.dropout_runs, d.discontinuities) == (0, 0, 0)
