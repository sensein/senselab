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


def test_very_low_headroom_check(stereo_audio_sample: Audio):
    assert not very_low_headroom_check(stereo_audio_sample), "very_low_headroom_check flagged stereo_audio_sample"


def test_very_high_headroom_check(stereo_audio_sample: Audio):
    assert not very_high_headroom_check(stereo_audio_sample), "very_high_headroom_check flagged stereo_audio_sample"


def test_very_low_amplitude_interquartile_range_check(stereo_audio_sample: Audio):
    assert not very_low_amplitude_interquartile_range_check(stereo_audio_sample), "very_low_amplitude_interquartile_range_check flagged stereo_audio_sample"


def test_very_high_amplitude_interquartile_range_check(stereo_audio_sample: Audio):
    assert not very_high_amplitude_interquartile_range_check(stereo_audio_sample), "very_high_amplitude_interquartile_range_check flagged stereo_audio_sample"


def test_very_low_amplitude_kurtosis_check(stereo_audio_sample: Audio):
    assert not very_low_amplitude_kurtosis_check(stereo_audio_sample), "very_low_amplitude_kurtosis_check flagged stereo_audio_sample"


def test_very_high_amplitude_kurtosis_check(stereo_audio_sample: Audio):
    assert not very_high_amplitude_kurtosis_check(stereo_audio_sample), "very_high_amplitude_kurtosis_check flagged stereo_audio_sample"


def test_very_low_amplitude_modulation_depth_check(stereo_audio_sample: Audio):
    assert not very_low_amplitude_modulation_depth_check(stereo_audio_sample), "very_low_amplitude_modulation_depth_check flagged stereo_audio_sample"


def test_low_amplitude_modulation_depth_check(stereo_audio_sample: Audio):
    assert not low_amplitude_modulation_depth_check(stereo_audio_sample), "low_amplitude_modulation_depth_check flagged stereo_audio_sample"


def test_high_proportion_clipped_check(stereo_audio_sample: Audio):
    assert not high_proportion_clipped_check(stereo_audio_sample), "high_proportion_clipped_check flagged stereo_audio_sample"


def test_clipping_present_check(stereo_audio_sample: Audio):
    assert not clipping_present_check(stereo_audio_sample), "clipping_present_check flagged stereo_audio_sample"


def test_completely_silent_check(stereo_audio_sample: Audio):
    assert not completely_silent_check(stereo_audio_sample), "completely_silent_check flagged stereo_audio_sample"


def test_mostly_silent_check(stereo_audio_sample: Audio):
    assert not mostly_silent_check(stereo_audio_sample), "mostly_silent_check flagged stereo_audio_sample"


def test_high_amplitude_skew_magnitude_check(stereo_audio_sample: Audio):
    assert not high_amplitude_skew_magnitude_check(stereo_audio_sample), "high_amplitude_skew_magnitude_check flagged stereo_audio_sample"


def test_high_crest_factor_check(stereo_audio_sample: Audio):
    assert not high_crest_factor_check(stereo_audio_sample), "high_crest_factor_check flagged stereo_audio_sample"


def test_low_crest_factor_check(stereo_audio_sample: Audio):
    assert not low_crest_factor_check(stereo_audio_sample), "low_crest_factor_check flagged stereo_audio_sample"


def test_very_low_dynamic_range_check(stereo_audio_sample: Audio):
    assert not very_low_dynamic_range_check(stereo_audio_sample), "very_low_dynamic_range_check flagged stereo_audio_sample"


def test_very_high_dynamic_range_check(stereo_audio_sample: Audio):
    assert not very_high_dynamic_range_check(stereo_audio_sample), "very_high_dynamic_range_check flagged stereo_audio_sample"


def test_very_low_mean_absolute_deviation_check(stereo_audio_sample: Audio):
    assert not very_low_mean_absolute_deviation_check(stereo_audio_sample), "very_low_mean_absolute_deviation_check flagged stereo_audio_sample"


def test_very_high_mean_absolute_deviation_check(stereo_audio_sample: Audio):
    assert not very_high_mean_absolute_deviation_check(stereo_audio_sample), "very_high_mean_absolute_deviation_check flagged stereo_audio_sample"


def test_very_low_peak_snr_from_spectral_check(stereo_audio_sample: Audio):
    assert not very_low_peak_snr_from_spectral_check(stereo_audio_sample), "very_low_peak_snr_from_spectral_check flagged stereo_audio_sample"


def test_low_phase_correlation_check(stereo_audio_sample: Audio):
    assert not low_phase_correlation_check(stereo_audio_sample), "low_phase_correlation_check flagged stereo_audio_sample"


def test_low_spectral_gating_snr_check(stereo_audio_sample: Audio):
    assert not low_spectral_gating_snr_check(stereo_audio_sample), "low_spectral_gating_snr_check flagged stereo_audio_sample"


def test_high_spectral_gating_snr_check(stereo_audio_sample: Audio):
    assert not high_spectral_gating_snr_check(stereo_audio_sample), "high_spectral_gating_snr_check flagged stereo_audio_sample"


def test_very_high_peak_snr_from_spectral_check(stereo_audio_sample: Audio):
    assert not very_high_peak_snr_from_spectral_check(stereo_audio_sample), "very_high_peak_snr_from_spectral_check flagged stereo_audio_sample"


def test_high_proportion_silence_at_beginning_check(stereo_audio_sample: Audio):
    assert not high_proportion_silence_at_beginning_check(stereo_audio_sample), "high_proportion_silence_at_beginning_check flagged stereo_audio_sample"


def test_high_proportion_silence_at_end_check(stereo_audio_sample: Audio):
    assert not high_proportion_silence_at_end_check(stereo_audio_sample), "high_proportion_silence_at_end_check flagged stereo_audio_sample"


def test_very_low_root_mean_square_energy_check(stereo_audio_sample: Audio):
    assert not very_low_root_mean_square_energy_check(stereo_audio_sample), "very_low_root_mean_square_energy_check flagged stereo_audio_sample"


def test_very_high_root_mean_square_energy_check(stereo_audio_sample: Audio):
    assert not very_high_root_mean_square_energy_check(stereo_audio_sample), "very_high_root_mean_square_energy_check flagged stereo_audio_sample"


def test_low_shannon_entropy_amplitude_check(stereo_audio_sample: Audio):
    assert not low_shannon_entropy_amplitude_check(stereo_audio_sample), "low_shannon_entropy_amplitude_check flagged stereo_audio_sample"


def test_high_shannon_entropy_amplitude_check(stereo_audio_sample: Audio):
    assert not high_shannon_entropy_amplitude_check(stereo_audio_sample), "high_shannon_entropy_amplitude_check flagged stereo_audio_sample"


def test_low_signal_variance_check(stereo_audio_sample: Audio):
    assert not low_signal_variance_check(stereo_audio_sample), "low_signal_variance_check flagged stereo_audio_sample"


def test_high_signal_variance_check(stereo_audio_sample: Audio):
    assert not high_signal_variance_check(stereo_audio_sample), "high_signal_variance_check flagged stereo_audio_sample"


def test_low_zero_crossing_rate_metric_check(stereo_audio_sample: Audio):
    assert not low_zero_crossing_rate_metric_check(stereo_audio_sample), "low_zero_crossing_rate_metric_check flagged stereo_audio_sample"


def test_high_zero_crossing_rate_metric_check(stereo_audio_sample: Audio):
    assert not high_zero_crossing_rate_metric_check(stereo_audio_sample), "high_zero_crossing_rate_metric_check flagged stereo_audio_sample"


def test_very_high_zero_crossing_rate_metric_check(stereo_audio_sample: Audio):
    assert not very_high_zero_crossing_rate_metric_check(stereo_audio_sample), "very_high_zero_crossing_rate_metric_check flagged stereo_audio_sample"
