"""Individual smoke tests: every QC check must be False on a clean stereo sample.

If a test fails it prints the associated metric value so you see *why* it triggered.
"""

from __future__ import annotations

import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.bioacoustic_qc.metrics import (
    amplitude_headroom_metric,
    amplitude_interquartile_range_metric,
    amplitude_kurtosis_metric,
    amplitude_modulation_depth_metric,
    amplitude_skew_metric,
    crest_factor_metric,
    dynamic_range_metric,
    mean_absolute_deviation_metric,
    peak_snr_from_spectral_metric,
    phase_correlation_metric,
    proportion_clipped_metric,
    proportion_silence_at_beginning_metric,
    proportion_silence_at_end_metric,
    proportion_silent_metric,
    root_mean_square_energy_metric,
    shannon_entropy_amplitude_metric,
    signal_variance_metric,
    spectral_gating_snr_metric,
    zero_crossing_rate_metric,
)
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
# one “positive” sanity test
# ---------------------------------------------------------------------
def test_audio_length_positive_check(stereo_audio_sample: Audio):
    num_samples = stereo_audio_sample.waveform.numel()
    assert audio_length_positive_check(stereo_audio_sample), (
        f"audio_length_positive_check returned False (num_samples={num_samples})"
    )

# ---------------------------------------------------------------------
# headroom
# ---------------------------------------------------------------------
def test_very_low_headroom_check(stereo_audio_sample: Audio):
    m = amplitude_headroom_metric(stereo_audio_sample)
    assert not very_low_headroom_check(stereo_audio_sample), (
        f"very_low_headroom_check flagged sample (headroom={m:.4f})"
    )


def test_very_high_headroom_check(stereo_audio_sample: Audio):
    m = amplitude_headroom_metric(stereo_audio_sample)
    assert not very_high_headroom_check(stereo_audio_sample), (
        f"very_high_headroom_check flagged sample (headroom={m:.4f})"
    )

# ---------------------------------------------------------------------
# IQR
# ---------------------------------------------------------------------
def test_very_low_amplitude_interquartile_range_check(stereo_audio_sample: Audio):
    m = amplitude_interquartile_range_metric(stereo_audio_sample)
    assert not very_low_amplitude_interquartile_range_check(stereo_audio_sample), (
        f"very_low_amplitude_interquartile_range_check flagged (IQR={m:.4f})"
    )


def test_very_high_amplitude_interquartile_range_check(stereo_audio_sample: Audio):
    m = amplitude_interquartile_range_metric(stereo_audio_sample)
    assert not very_high_amplitude_interquartile_range_check(stereo_audio_sample), (
        f"very_high_amplitude_interquartile_range_check flagged (IQR={m:.4f})"
    )

# ---------------------------------------------------------------------
# kurtosis
# ---------------------------------------------------------------------
def test_very_low_amplitude_kurtosis_check(stereo_audio_sample: Audio):
    m = amplitude_kurtosis_metric(stereo_audio_sample)
    assert not very_low_amplitude_kurtosis_check(stereo_audio_sample), (
        f"very_low_amplitude_kurtosis_check flagged (kurtosis={m:.2f})"
    )


def test_very_high_amplitude_kurtosis_check(stereo_audio_sample: Audio):
    m = amplitude_kurtosis_metric(stereo_audio_sample)
    assert not very_high_amplitude_kurtosis_check(stereo_audio_sample), (
        f"very_high_amplitude_kurtosis_check flagged (kurtosis={m:.2f})"
    )

# ---------------------------------------------------------------------
# modulation depth
# ---------------------------------------------------------------------
def test_very_low_amplitude_modulation_depth_check(stereo_audio_sample: Audio):
    m = amplitude_modulation_depth_metric(stereo_audio_sample)
    assert not very_low_amplitude_modulation_depth_check(stereo_audio_sample), (
        f"very_low_amplitude_modulation_depth_check flagged (mod_depth={m:.4f})"
    )


def test_low_amplitude_modulation_depth_check(stereo_audio_sample: Audio):
    m = amplitude_modulation_depth_metric(stereo_audio_sample)
    assert not low_amplitude_modulation_depth_check(stereo_audio_sample), (
        f"low_amplitude_modulation_depth_check flagged (mod_depth={m:.4f})"
    )

# ---------------------------------------------------------------------
# clipping proportion
# ---------------------------------------------------------------------
def test_high_proportion_clipped_check(stereo_audio_sample: Audio):
    m = proportion_clipped_metric(stereo_audio_sample)
    assert not high_proportion_clipped_check(stereo_audio_sample), (
        f"high_proportion_clipped_check flagged (clip_prop={m:.6f})"
    )


def test_clipping_present_check(stereo_audio_sample: Audio):
    m = proportion_clipped_metric(stereo_audio_sample)
    assert not clipping_present_check(stereo_audio_sample), (
        f"clipping_present_check flagged (clip_prop={m:.6f})"
    )

# ---------------------------------------------------------------------
# silence
# ---------------------------------------------------------------------
def test_completely_silent_check(stereo_audio_sample: Audio):
    m = proportion_silent_metric(stereo_audio_sample)
    assert not completely_silent_check(stereo_audio_sample), (
        f"completely_silent_check flagged (silent_prop={m:.4f})"
    )


def test_mostly_silent_check(stereo_audio_sample: Audio):
    m = proportion_silent_metric(stereo_audio_sample)
    assert not mostly_silent_check(stereo_audio_sample), (
        f"mostly_silent_check flagged (silent_prop={m:.4f})"
    )

# ---------------------------------------------------------------------
# skew
# ---------------------------------------------------------------------
def test_high_amplitude_skew_magnitude_check(stereo_audio_sample: Audio):
    m = amplitude_skew_metric(stereo_audio_sample)
    assert not high_amplitude_skew_magnitude_check(stereo_audio_sample), (
        f"high_amplitude_skew_magnitude_check flagged (skew={m:.4f})"
    )

# ---------------------------------------------------------------------
# crest factor
# ---------------------------------------------------------------------
def test_high_crest_factor_check(stereo_audio_sample: Audio):
    m = crest_factor_metric(stereo_audio_sample)
    assert not high_crest_factor_check(stereo_audio_sample), (
        f"high_crest_factor_check flagged (crest={m:.2f})"
    )


def test_low_crest_factor_check(stereo_audio_sample: Audio):
    m = crest_factor_metric(stereo_audio_sample)
    assert not low_crest_factor_check(stereo_audio_sample), (
        f"low_crest_factor_check flagged (crest={m:.2f})"
    )

# ---------------------------------------------------------------------
# dynamic range
# ---------------------------------------------------------------------
def test_very_low_dynamic_range_check(stereo_audio_sample: Audio):
    m = dynamic_range_metric(stereo_audio_sample)
    assert not very_low_dynamic_range_check(stereo_audio_sample), (
        f"very_low_dynamic_range_check flagged (dyn_range={m:.4f})"
    )


def test_very_high_dynamic_range_check(stereo_audio_sample: Audio):
    m = dynamic_range_metric(stereo_audio_sample)
    assert not very_high_dynamic_range_check(stereo_audio_sample), (
        f"very_high_dynamic_range_check flagged (dyn_range={m:.4f})"
    )

# ---------------------------------------------------------------------
# MAD
# ---------------------------------------------------------------------
def test_very_low_mean_absolute_deviation_check(stereo_audio_sample: Audio):
    m = mean_absolute_deviation_metric(stereo_audio_sample)
    assert not very_low_mean_absolute_deviation_check(stereo_audio_sample), (
        f"very_low_mean_absolute_deviation_check flagged (MAD={m:.6f})"
    )


def test_very_high_mean_absolute_deviation_check(stereo_audio_sample: Audio):
    m = mean_absolute_deviation_metric(stereo_audio_sample)
    assert not very_high_mean_absolute_deviation_check(stereo_audio_sample), (
        f"very_high_mean_absolute_deviation_check flagged (MAD={m:.6f})"
    )

# ---------------------------------------------------------------------
# peak-SNR
# ---------------------------------------------------------------------
def test_very_low_peak_snr_from_spectral_check(stereo_audio_sample: Audio):
    m = peak_snr_from_spectral_metric(stereo_audio_sample)
    assert not very_low_peak_snr_from_spectral_check(stereo_audio_sample), (
        f"very_low_peak_snr_from_spectral_check flagged (SNR={m:.2f} dB)"
    )


def test_low_peak_snr_from_spectral_check(stereo_audio_sample: Audio):
    m = peak_snr_from_spectral_metric(stereo_audio_sample)
    assert not low_peak_snr_from_spectral_check(stereo_audio_sample), (
        f"low_peak_snr_from_spectral_check flagged (SNR={m:.2f} dB)"
    )


def test_very_high_peak_snr_from_spectral_check(stereo_audio_sample: Audio):
    m = peak_snr_from_spectral_metric(stereo_audio_sample)
    assert not very_high_peak_snr_from_spectral_check(stereo_audio_sample), (
        f"very_high_peak_snr_from_spectral_check flagged (SNR={m:.2f} dB)"
    )

# ---------------------------------------------------------------------
# phase correlation
# ---------------------------------------------------------------------
def test_low_phase_correlation_check(stereo_audio_sample: Audio):
    m = phase_correlation_metric(stereo_audio_sample)
    assert not low_phase_correlation_check(stereo_audio_sample), (
        f"low_phase_correlation_check flagged (corr={m:.4f})"
    )

# ---------------------------------------------------------------------
# segmental SNR (spectral-gating)
# ---------------------------------------------------------------------
def test_low_spectral_gating_snr_check(stereo_audio_sample: Audio):
    m = spectral_gating_snr_metric(stereo_audio_sample)
    assert not low_spectral_gating_snr_check(stereo_audio_sample), (
        f"low_spectral_gating_snr_check flagged (SNR={m:.2f} dB)"
    )


def test_high_spectral_gating_snr_check(stereo_audio_sample: Audio):
    m = spectral_gating_snr_metric(stereo_audio_sample)
    assert not high_spectral_gating_snr_check(stereo_audio_sample), (
        f"high_spectral_gating_snr_check flagged (SNR={m:.2f} dB)"
    )

# ---------------------------------------------------------------------
# leading / trailing silence
# ---------------------------------------------------------------------
def test_high_proportion_silence_at_beginning_check(stereo_audio_sample: Audio):
    m = proportion_silence_at_beginning_metric(stereo_audio_sample)
    assert not high_proportion_silence_at_beginning_check(stereo_audio_sample), (
        f"high_proportion_silence_at_beginning_check flagged (lead_silence={m:.4f})"
    )


def test_high_proportion_silence_at_end_check(stereo_audio_sample: Audio):
    m = proportion_silence_at_end_metric(stereo_audio_sample)
    assert not high_proportion_silence_at_end_check(stereo_audio_sample), (
        f"high_proportion_silence_at_end_check flagged (trail_silence={m:.4f})"
    )

# ---------------------------------------------------------------------
# RMS energy
# ---------------------------------------------------------------------
def test_very_low_root_mean_square_energy_check(stereo_audio_sample: Audio):
    m = root_mean_square_energy_metric(stereo_audio_sample)
    assert not very_low_root_mean_square_energy_check(stereo_audio_sample), (
        f"very_low_root_mean_square_energy_check flagged (RMS={m:.5f})"
    )


def test_very_high_root_mean_square_energy_check(stereo_audio_sample: Audio):
    m = root_mean_square_energy_metric(stereo_audio_sample)
    assert not very_high_root_mean_square_energy_check(stereo_audio_sample), (
        f"very_high_root_mean_square_energy_check flagged (RMS={m:.5f})"
    )

# ---------------------------------------------------------------------
# entropy
# ---------------------------------------------------------------------
def test_low_shannon_entropy_amplitude_check(stereo_audio_sample: Audio):
    m = shannon_entropy_amplitude_metric(stereo_audio_sample)
    assert not low_shannon_entropy_amplitude_check(stereo_audio_sample), (
        f"low_shannon_entropy_amplitude_check flagged (entropy={m:.2f} bits)"
    )


def test_high_shannon_entropy_amplitude_check(stereo_audio_sample: Audio):
    m = shannon_entropy_amplitude_metric(stereo_audio_sample)
    assert not high_shannon_entropy_amplitude_check(stereo_audio_sample), (
        f"high_shannon_entropy_amplitude_check flagged (entropy={m:.2f} bits)"
    )

# ---------------------------------------------------------------------
# variance
# ---------------------------------------------------------------------
def test_low_signal_variance_check(stereo_audio_sample: Audio):
    m = signal_variance_metric(stereo_audio_sample)
    assert not low_signal_variance_check(stereo_audio_sample), (
        f"low_signal_variance_check flagged (var={m:.6f})"
    )


def test_high_signal_variance_check(stereo_audio_sample: Audio):
    m = signal_variance_metric(stereo_audio_sample)
    assert not high_signal_variance_check(stereo_audio_sample), (
        f"high_signal_variance_check flagged (var={m:.6f})"
    )

# ---------------------------------------------------------------------
# zero-crossing rate
# ---------------------------------------------------------------------
def test_low_zero_crossing_rate_metric_check(stereo_audio_sample: Audio):
    m = zero_crossing_rate_metric(stereo_audio_sample)
    assert not low_zero_crossing_rate_metric_check(stereo_audio_sample), (
        f"low_zero_crossing_rate_metric_check flagged (ZCR={m:.4f})"
    )


def test_high_zero_crossing_rate_metric_check(stereo_audio_sample: Audio):
    m = zero_crossing_rate_metric(stereo_audio_sample)
    assert not high_zero_crossing_rate_metric_check(stereo_audio_sample), (
        f"high_zero_crossing_rate_metric_check flagged (ZCR={m:.4f})"
    )


def test_very_high_zero_crossing_rate_metric_check(stereo_audio_sample: Audio):
    m = zero_crossing_rate_metric(stereo_audio_sample)
    assert not very_high_zero_crossing_rate_metric_check(stereo_audio_sample), (
        f"very_high_zero_crossing_rate_metric_check flagged (ZCR={m:.4f})"
    )
