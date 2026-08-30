"""The Hilbert envelope in dBFS and its rolling local floor."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.envelope import dynamic_range_normalize, hilbert_envelope_dbfs, rolling_floor_dbfs

SR = 16000


def _tone(seconds: float, amp: float, freq: float = 200.0) -> Audio:
    t = np.arange(int(seconds * SR)) / SR
    return Audio(waveform=(amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)[None, :], sampling_rate=SR)


def _env(audio: Audio) -> np.ndarray:
    """Run the envelope with the config's measured values; the fixture supplies the literals."""
    return hilbert_envelope_dbfs(audio, lowpass_hz=40.0, filter_order=4)


class TestEnvelopeIsAbsolute:
    """The envelope is dBFS, never normalised by the input's own maximum."""

    def test_a_half_scale_tone_sits_near_minus_six_dbfs(self) -> None:
        """A 0.5-amplitude tone reads close to -6 dBFS, pinning the absolute reference."""
        env = _env(_tone(1.0, 0.5))
        mid = env[SR // 4 : -SR // 4]
        assert -7.5 < float(np.median(mid)) < -4.5

    def test_scaling_the_input_shifts_the_envelope_by_the_same_amount(self) -> None:
        """A 20 dB input change moves the envelope 20 dB — absolute, not max-normalised."""
        loud = float(np.median(_env(_tone(1.0, 0.5))[SR // 4 : -SR // 4]))
        quiet = float(np.median(_env(_tone(1.0, 0.05))[SR // 4 : -SR // 4]))
        assert loud - quiet == pytest.approx(20.0, abs=1.0), "dBFS is absolute, not max-normalised"

    def test_a_loud_click_elsewhere_does_not_move_the_rest(self) -> None:
        """One 30 ms click cannot rescale the envelope far from it."""
        quiet = _tone(2.0, 0.05)
        clicked = quiet.waveform.numpy().copy() if hasattr(quiet.waveform, "numpy") else np.array(quiet.waveform)
        clicked[0, SR : SR + 480] += 0.95
        a = _env(quiet)
        b = _env(Audio(waveform=clicked.astype(np.float32), sampling_rate=SR))
        early = slice(SR // 8, SR // 2)
        assert float(np.median(b[early])) == pytest.approx(float(np.median(a[early])), abs=0.5)


class TestRollingFloor:
    """The floor tracks the recording rather than summarising it."""

    def test_the_floor_tracks_a_level_change_rather_than_averaging_it(self) -> None:
        """A -60 to -30 dB step moves the floor to each level, not to their mean."""
        env = np.concatenate([np.full(5 * SR, -60.0), np.full(5 * SR, -30.0)])
        fl = rolling_floor_dbfs(env, SR, window_s=1.0, percentile=10.0, eval_grid_s=0.1)
        assert fl[SR] == pytest.approx(-60.0, abs=1.0)
        assert fl[9 * SR] == pytest.approx(-30.0, abs=1.0)

    def test_the_floor_is_one_value_per_sample(self) -> None:
        """The floor track has the envelope's own shape."""
        env = np.full(3 * SR, -50.0)
        assert rolling_floor_dbfs(env, SR, window_s=3.0, percentile=10.0, eval_grid_s=0.1).shape == env.shape


def _burst(seconds: float = 2.0, amp: float = 0.6, freq: float = 220.0) -> Audio:
    """A tone burst with a sharp offset, which is what makes ``filtfilt`` undershoot below zero."""
    t = np.arange(int(seconds * SR)) / SR
    x = np.zeros_like(t)
    voiced = (t >= 0.5) & (t < 1.0)
    x[voiced] = amp * np.sin(2 * np.pi * freq * t[voiced])
    return Audio(waveform=x.astype(np.float32)[None, :], sampling_rate=SR)


class TestAnUnmeasurableSampleHasNoDecibelValue:
    """A filtered envelope that undershoots to zero or below has no dB value there to report."""

    def test_an_undershooting_sample_reads_nan(self) -> None:
        """The samples where the zero-phase filter undershoots are unmeasurable, not very quiet."""
        env = _env(_burst())
        assert np.isnan(env).any(), "this signal must undershoot; the fixture is the whole test"

    def test_the_samples_that_are_measurable_stay_finite(self) -> None:
        """Only the undershoot goes; the burst itself still reads a level."""
        env = _env(_burst())
        assert np.isfinite(env[int(0.75 * SR)])
        assert np.isfinite(env).sum() > env.size // 2

    def test_no_sample_reads_the_clamps_own_value(self) -> None:
        """-240 dBFS was 20*log10(1e-12): the clamp's value, never a measurement of the signal."""
        env = _env(_burst())
        assert not np.any(env <= -240.0)

    def test_the_finite_range_is_the_signals_own(self) -> None:
        """The panel's y-scale comes from these values, so a fabricated floor destroys it."""
        env = _env(_burst())
        assert float(np.nanmin(env)) > -150.0

    def test_the_contract_is_still_one_value_per_input_sample(self) -> None:
        """A gap is a NaN in place, not a dropped sample; the time axis must stay aligned."""
        audio = _burst()
        assert _env(audio).shape == (audio.waveform.shape[1],)

    def test_filter_residue_below_the_input_resolution_is_not_a_false_floor(self) -> None:
        """Sub-resolution ringing is a missing level, rather than an extreme dBFS observation."""
        quiet = _tone(1.0, 1e-9)
        assert np.isnan(_env(quiet)[SR // 4 : -SR // 4]).all()


class TestTheFloorOverAnUnmeasurableEnvelope:
    """The floor is a percentile of what was measured, and says nothing where nothing was."""

    def test_the_floor_is_computed_over_the_finite_samples(self) -> None:
        """A scatter of NaN inside a level window leaves the floor at that level."""
        env = np.full(6 * SR, -50.0)
        env[:: SR // 100] = np.nan
        fl = rolling_floor_dbfs(env, SR, window_s=1.0, percentile=10.0, eval_grid_s=0.1)
        assert fl[3 * SR] == pytest.approx(-50.0, abs=0.5)

    def test_a_window_with_no_finite_sample_has_no_floor(self) -> None:
        """Nothing was measured there, so there is no percentile of it to report."""
        env = np.full(9 * SR, -50.0)
        env[3 * SR : 6 * SR] = np.nan
        fl = rolling_floor_dbfs(env, SR, window_s=1.0, percentile=10.0, eval_grid_s=0.1)
        assert np.isnan(fl[int(4.5 * SR)])
        assert np.isfinite(fl[SR])
        assert np.isfinite(fl[8 * SR])

    def test_an_unmeasurable_window_raises_no_warning(self) -> None:
        """A run that prints a RuntimeWarning per window is a run nobody reads the output of."""
        env = np.full(3 * SR, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fl = rolling_floor_dbfs(env, SR, window_s=1.0, percentile=10.0, eval_grid_s=0.1)
        assert np.isnan(fl).all()


def _raw_tone(seconds: float, amp: float, freq: float = 200.0) -> np.ndarray:
    """A tone as a plain array, for building multi-segment or multi-channel fixtures by hand."""
    t = np.arange(int(seconds * SR)) / SR
    return amp * np.sin(2 * np.pi * freq * t)


def _normalize(audio: Audio) -> Audio:
    """Run dynamic_range_normalize with the reference implementation's own literal values."""
    return dynamic_range_normalize(
        audio,
        macro_lowpass_hz=0.2,
        micro_lowpass_hz=20.0,
        envelope_filter_order=2,
        target_dr_db=15.0,
        compression_ratio=2.0,
        macro_target_dbfs=-6.0,
        gain_smooth_hz=10.0,
        gain_filter_order=1,
        floor_dbfs=-100.0,
        ceiling=0.95,
    )


def _rms_dbfs(waveform: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(waveform))))
    return 20.0 * np.log10(max(rms, 1e-12))


class TestDynamicRangeNormalizeEvensOutScenes:
    """A quiet scene and a loud scene of the same file are brought toward one reference level."""

    def test_a_quiet_scene_is_boosted_and_a_loud_one_is_not_boosted_further(self) -> None:
        """Two four-second tones at very different levels end up much closer together."""
        quiet = _raw_tone(4.0, amp=0.01)
        loud = _raw_tone(4.0, amp=0.5)
        x = np.concatenate([quiet, loud])
        audio = Audio(waveform=x.astype(np.float32)[None, :], sampling_rate=SR)
        processed = _normalize(audio).waveform.squeeze(0).numpy()

        before_gap = _rms_dbfs(loud) - _rms_dbfs(quiet)
        mid = SR * 4
        after_gap = _rms_dbfs(processed[mid + SR : mid + 3 * SR]) - _rms_dbfs(processed[SR : 3 * SR])
        assert abs(after_gap) < abs(before_gap)

    def test_output_never_exceeds_the_ceiling(self) -> None:
        """The final safety clamp is absolute, regardless of how much gain compression applied."""
        x = _raw_tone(2.0, amp=0.01)
        audio = Audio(waveform=x.astype(np.float32)[None, :], sampling_rate=SR)
        processed = _normalize(audio).waveform.numpy()
        assert float(np.abs(processed).max()) <= 0.95 + 1e-6

    def test_shape_rate_and_channel_count_are_preserved(self) -> None:
        """A stereo input comes back stereo, at the same length and rate."""
        x = np.stack([_raw_tone(1.0, amp=0.2), _raw_tone(1.0, amp=0.2, freq=300.0)])
        audio = Audio(waveform=x.astype(np.float32), sampling_rate=SR)
        processed = _normalize(audio)
        assert processed.waveform.shape == audio.waveform.shape
        assert processed.sampling_rate == audio.sampling_rate

    def test_a_stereo_input_applies_one_shared_gain_curve(self) -> None:
        """The gain is derived from the downmix, then applied identically to every channel."""
        base = _raw_tone(1.0, amp=0.05)
        x = np.stack([base, 0.5 * base])
        audio = Audio(waveform=x.astype(np.float32), sampling_rate=SR)
        processed = _normalize(audio).waveform.numpy()
        ratio = processed[0][SR // 4 : -SR // 4] / processed[1][SR // 4 : -SR // 4]
        assert np.allclose(ratio, 2.0, atol=1e-3)

    def test_silence_produces_a_finite_not_nan_result(self) -> None:
        """floor_dbfs stands in wherever the envelope has no measurable value."""
        x = np.zeros(SR)
        audio = Audio(waveform=x.astype(np.float32)[None, :], sampling_rate=SR)
        processed = _normalize(audio).waveform.numpy()
        assert np.isfinite(processed).all()
