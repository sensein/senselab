"""Individual smoke tests: every QC check must be False on a clean stereo sample."""

from __future__ import annotations

import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.bioacoustic_qc.checks import (
    audio_length_positive_check,
    very_low_headroom_check,
    very_high_headroom_check,
    very_low_amplitude_interquartile_range_check,
    very_high_amplitude_interquartile_range_check,
    very_low_amplitude_kurtosis_check,
    very_high_amplitude_kurtosis_check,
    very_low_amplitude_modulation_depth_check,
    low_amplitude_modulation_depth_check,
    high_proportion_clipped_check,
    clipping_present_check,
    completely_silent_check,
    mostly_silent_check,
    high_amplitude_skew_magnitude_check,
    high_crest_factor_check,
    low_crest_factor_check,
    very_low_dynamic_range_check,
    very_high_dynamic_range_check,
    very_low_mean_absolute_deviation_check,
    very_high_mean_absolute_deviation_check,
    very_low_peak_snr_from_spectral_check,
    low_peak_snr_from_spectral_check,
    very_high_peak_snr_from_spectral_check,
    low_phase_correlation_check,
    high_proportion_silence_at_beginning_check,
    high_proportion_silence_at_end_check,
    very_low_root_mean_square_energy_check,
    very_high_root_mean_square_energy_check,
    low_shannon_entropy_amplitude_check,
    high_shannon_entropy_amplitude_check,
    low_signal_variance_check,
    high_signal_variance_check,
    low_spectral_gating_snr_check,
    high_spectral_gating_snr_check,
    low_zero_crossing_rate_metric_check,
    high_zero_crossing_rate_metric_check,
    very_high_zero_crossing_rate_metric_check,
)

# ---------------------------------------------------------------------
# Helper that makes one test function per check
# ---------------------------------------------------------------------
def _assert_false(check_fn: callable, audio: Audio) -> None:
    """Assert the QC check returns False for `audio`."""
    assert not check_fn(audio), f"{check_fn.__name__} flagged stereo_audio_sample"


# ---------------------------------------------------------------------
# === Individual smoke tests (auto-generated) ===
# ---------------------------------------------------------------------

def test_very_low_headroom_check(stereo_audio_sample: Audio):
    _assert_false(very_low_headroom_check, stereo_audio_sample)


def test_very_high_headroom_check(stereo_audio_sample: Audio):
    _assert_false(very_high_headroom_check, stereo_audio_sample)


def test_very_low_amplitude_interquartile_range_check(stereo_audio_sample: Audio):
    _assert_false(very_low_amplitude_interquartile_range_check, stereo_audio_sample)


def test_very_high_amplitude_interquartile_range_check(stereo_audio_sample: Audio):
    _assert_false(very_high_amplitude_interquartile_range_check, stereo_audio_sample)


def test_very_low_amplitude_kurtosis_check(stereo_audio_sample: Audio):
    _assert_false(very_low_amplitude_kurtosis_check, stereo_audio_sample)


def test_very_high_amplitude_kurtosis_check(stereo_audio_sample: Audio):
    _assert_false(very_high_amplitude_kurtosis_check, stereo_audio_sample)


def test_very_low_amplitude_modulation_depth_check(stereo_audio_sample: Audio):
    _assert_false(very_low_amplitude_modulation_depth_check, stereo_audio_sample)


def test_low_amplitude_modulation_depth_check(stereo_audio_sample: Audio):
    _assert_false(low_amplitude_modulation_depth_check, stereo_audio_sample)


def test_high_proportion_clipped_check(stereo_audio_sample: Audio):
    _assert_false(high_proportion_clipped_check, stereo_audio_sample)


def test_clipping_present_check(stereo_audio_sample: Audio):
    _assert_false(clipping_present_check, stereo_audio_sample)


def test_completely_silent_check(stereo_audio_sample: Audio):
    _assert_false(completely_silent_check, stereo_audio_sample)


def test_mostly_silent_check(stereo_audio_sample: Audio):
    _assert_false(mostly_silent_check, stereo_audio_sample)


def test_high_amplitude_skew_magnitude_check(stereo_audio_sample: Audio):
    _assert_false(high_amplitude_skew_magnitude_check, stereo_audio_sample)


def test_high_crest_factor_check(stereo_audio_sample: Audio):
    _assert_false(high_crest_factor_check, stereo_audio_sample)


def test_low_crest_factor_check(stereo_audio_sample: Audio):
    _assert_false(low_crest_factor_check, stereo_audio_sample)


def test_very_low_dynamic_range_check(stereo_audio_sample: Audio):
    _assert_false(very_low_dynamic_range_check, stereo_audio_sample)


def test_very_high_dynamic_range_check(stereo_audio_sample: Audio):
    _assert_false(very_high_dynamic_range_check, stereo_audio_sample)


def test_very_low_mean_absolute_deviation_check(stereo_audio_sample: Audio):
    _assert_false(very_low_mean_absolute_deviation_check, stereo_audio_sample)


def test_very_high_mean_absolute_deviation_check(stereo_audio_sample: Audio):
    _assert_false(very_high_mean_absolute_deviation_check, stereo_audio_sample)


def test_very_low_peak_snr_from_spectral_check(stereo_audio_sample: Audio):
    _assert_false(very_low_peak_snr_from_spectral_check, stereo_audio_sample)


def test_low_phase_correlation_check(stereo_audio_sample: Audio):
    _assert_false(low_phase_correlation_check, stereo_audio_sample)


def test_low_spectral_gating_snr_check(stereo_audio_sample: Audio):
    _assert_false(low_spectral_gating_snr_check, stereo_audio_sample)


def test_high_spectral_gating_snr_check(stereo_audio_sample: Audio):
    _assert_false(high_spectral_gating_snr_check, stereo_audio_sample)


def test_very_high_peak_snr_from_spectral_check(stereo_audio_sample: Audio):
    _assert_false(very_high_peak_snr_from_spectral_check, stereo_audio_sample)


def test_high_proportion_silence_at_beginning_check(stereo_audio_sample: Audio):
    _assert_false(high_proportion_silence_at_beginning_check, stereo_audio_sample)


def test_high_proportion_silence_at_end_check(stereo_audio_sample: Audio):
    _assert_false(high_proportion_silence_at_end_check, stereo_audio_sample)


def test_very_low_root_mean_square_energy_check(stereo_audio_sample: Audio):
    _assert_false(very_low_root_mean_square_energy_check, stereo_audio_sample)


def test_very_high_root_mean_square_energy_check(stereo_audio_sample: Audio):
    _assert_false(very_high_root_mean_square_energy_check, stereo_audio_sample)


def test_low_shannon_entropy_amplitude_check(stereo_audio_sample: Audio):
    _assert_false(low_shannon_entropy_amplitude_check, stereo_audio_sample)


def test_high_shannon_entropy_amplitude_check(stereo_audio_sample: Audio):
    _assert_false(high_shannon_entropy_amplitude_check, stereo_audio_sample)


def test_low_signal_variance_check(stereo_audio_sample: Audio):
    _assert_false(low_signal_variance_check, stereo_audio_sample)


def test_high_signal_variance_check(stereo_audio_sample: Audio):
    _assert_false(high_signal_variance_check, stereo_audio_sample)


def test_low_zero_crossing_rate_metric_check(stereo_audio_sample: Audio):
    _assert_false(low_zero_crossing_rate_metric_check, stereo_audio_sample)


def test_high_zero_crossing_rate_metric_check(stereo_audio_sample: Audio):
    _assert_false(high_zero_crossing_rate_metric_check, stereo_audio_sample)


def test_very_high_zero_crossing_rate_metric_check(stereo_audio_sample: Audio):
    _assert_false(very_high_zero_crossing_rate_metric_check, stereo_audio_sample)
