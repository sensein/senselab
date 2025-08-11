"""Runs all quality control checks on a stereo sample.

If a test fails, the assertion message shows the metric value.
"""

from __future__ import annotations

import os

import pandas as pd

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.bioacoustic_qc.checks import (
    audio_intensity_positive_check,
    audio_length_positive_check,
    clipping_present_check,
    completely_silent_check,
    get_metric,
    high_amplitude_skew_magnitude_check,
    high_crest_factor_check,
    high_proportion_clipped_check,
    high_proportion_silence_at_beginning_check,
    high_proportion_silence_at_end_check,
    high_shannon_entropy_amplitude_check,
    high_signal_variance_check,
    high_spectral_gating_snr_check,
    high_zero_crossing_rate_check,
    low_amplitude_modulation_depth_check,
    low_crest_factor_check,
    low_peak_snr_from_spectral_check,
    low_phase_correlation_check,
    low_shannon_entropy_amplitude_check,
    low_signal_variance_check,
    low_spectral_gating_snr_check,
    low_zero_crossing_rate_check,
    mostly_silent_check,
    very_high_amplitude_interquartile_range_check,
    very_high_amplitude_kurtosis_check,
    very_high_dynamic_range_check,
    very_high_headroom_check,
    very_high_mean_absolute_deviation_check,
    very_high_peak_snr_from_spectral_check,
    very_high_root_mean_square_energy_check,
    very_high_zero_crossing_rate_check,
    very_low_amplitude_kurtosis_check,
    very_low_amplitude_modulation_depth_check,
    very_low_dynamic_range_check,
    very_low_headroom_check,
    very_low_mean_absolute_deviation_check,
    very_low_peak_snr_from_spectral_check,
    very_low_root_mean_square_energy_check,
)
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


def test_get_metric_computes_directly(stereo_audio_sample: Audio) -> None:
    """Should compute the metric when no DataFrame is provided."""
    value = get_metric(stereo_audio_sample, zero_crossing_rate_metric)
    assert isinstance(value, float)
    assert 0 <= value <= 1


def test_get_metric_uses_dataframe_cache(stereo_audio_sample: Audio) -> None:
    """Should return cached value when DataFrame is valid."""
    cached_value = 0.1234
    filepath = stereo_audio_sample.filepath()
    filename = None
    if filepath:
        filename = os.path.basename(filepath)
    df = pd.DataFrame(
        {
            "audio_path_or_id": [filename],
            "zero_crossing_rate_metric": [cached_value],
        }
    )
    value = get_metric(stereo_audio_sample, zero_crossing_rate_metric, df=df)
    assert value == cached_value


def test_get_metric_when_filepath_is_none(stereo_audio_sample: Audio) -> None:
    """Should compute directly when audio has no filepath."""
    cached_value = 0.1234
    df = pd.DataFrame(
        {
            "audio_path_or_id": ["some_file.wav"],
            "zero_crossing_rate_metric": [cached_value],
        }
    )

    # Create an Audio object with no filepath (waveform + sampling_rate only)
    audio_no_filepath = Audio(waveform=stereo_audio_sample.waveform, sampling_rate=stereo_audio_sample.sampling_rate)

    # Verify filepath is None
    assert audio_no_filepath.filepath() is None

    value = get_metric(audio_no_filepath, zero_crossing_rate_metric, df=df)
    # Should compute directly since filepath is None
    assert isinstance(value, float)
    assert 0 <= value <= 1
    # Should not use cached value since filepath is None
    assert value != cached_value


def test_get_metric_caches_with_audio_id_when_no_filepath(
    stereo_audio_sample: Audio,
) -> None:
    """Should cache metric using audio ID when filepath is None."""
    # Create an Audio object with no filepath
    audio_no_filepath = Audio(waveform=stereo_audio_sample.waveform, sampling_rate=stereo_audio_sample.sampling_rate)

    # Verify filepath is None
    assert audio_no_filepath.filepath() is None

    # Create DataFrame that will be modified
    df = pd.DataFrame({"audio_path_or_id": [], "zero_crossing_rate_metric": []})

    # First call should compute and cache the metric
    value1 = get_metric(audio_no_filepath, zero_crossing_rate_metric, df=df)
    assert isinstance(value1, float)
    assert 0 <= value1 <= 1

    # Verify the metric was cached using audio ID
    audio_id = audio_no_filepath.generate_id()
    assert audio_id in df["audio_path_or_id"].values
    assert "zero_crossing_rate_metric" in df.columns

    # Second call should use cached value
    value2 = get_metric(audio_no_filepath, zero_crossing_rate_metric, df=df)
    assert value2 == value1  # Should be same cached value


def test_get_metric_computes_when_not_in_dataframe(
    stereo_audio_sample: Audio,
) -> None:
    """Should compute metric when file not in DataFrame."""
    cached_value = 0.1234
    df = pd.DataFrame(
        {
            "audio_path_or_id": ["not_the_sample.wav"],
            "zero_crossing_rate_metric": [cached_value],
        }
    )
    value = get_metric(stereo_audio_sample, zero_crossing_rate_metric, df=df)
    assert value != cached_value
    assert isinstance(value, float)
    assert 0 <= value <= 1


def test_get_metric_filepath_not_in_dataframe(
    stereo_audio_sample: Audio,
) -> None:
    """Should compute metric when audio filepath not found in DataFrame."""
    # Create a DataFrame with different filenames
    df = pd.DataFrame(
        {
            "audio_path_or_id": ["other_file1.wav", "other_file2.wav"],
            "zero_crossing_rate_metric": [0.1, 0.2],
        }
    )

    # Should compute directly since filepath not in DataFrame
    value = get_metric(stereo_audio_sample, zero_crossing_rate_metric, df=df)
    assert isinstance(value, float)
    assert 0 <= value <= 1

    # Verify it's not one of the cached values
    assert value not in [0.1, 0.2]


def test_get_metric_raises_error_when_filepath_not_in_df() -> None:
    """Should raise FileNotFoundError when filepath doesn't exist."""
    import pytest

    # Create a DataFrame with different filenames
    df = pd.DataFrame(
        {
            "audio_path_or_id": ["other_file1.wav", "other_file2.wav"],
            "zero_crossing_rate_metric": [0.1, 0.2],
        }
    )

    # Pass a filepath string that doesn't exist
    with pytest.raises(FileNotFoundError, match="File nonexistent_file.wav does not exist"):
        get_metric("nonexistent_file.wav", zero_crossing_rate_metric, df=df)


def test_audio_length_positive_check(stereo_audio_sample: Audio) -> None:
    """audio_length_positive_check returns True."""
    n = stereo_audio_sample.waveform.numel()
    assert audio_length_positive_check(
        stereo_audio_sample
    ), f"audio_length_positive_check returned False (num_samples={n})"


def test_very_low_headroom_check(stereo_audio_sample: Audio) -> None:
    """very_low_headroom_check returns False."""
    m = amplitude_headroom_metric(stereo_audio_sample)
    assert not very_low_headroom_check(
        stereo_audio_sample
    ), f"very_low_headroom_check flagged sample (headroom={m:.4f})"


def test_very_high_headroom_check(stereo_audio_sample: Audio) -> None:
    """very_high_headroom_check returns False."""
    m = amplitude_headroom_metric(stereo_audio_sample)
    assert not very_high_headroom_check(
        stereo_audio_sample
    ), f"very_high_headroom_check flagged sample (headroom={m:.4f})"


def test_very_high_amplitude_interquartile_range_check(
    stereo_audio_sample: Audio,
) -> None:
    """very_high_amplitude_interquartile_range_check returns False."""
    m = amplitude_interquartile_range_metric(stereo_audio_sample)
    assert not very_high_amplitude_interquartile_range_check(
        stereo_audio_sample
    ), f"very_high_amplitude_interquartile_range_check flagged (IQR={m:.4f})"


def test_very_low_amplitude_kurtosis_check(stereo_audio_sample: Audio) -> None:
    """very_low_amplitude_kurtosis_check returns False."""
    m = amplitude_kurtosis_metric(stereo_audio_sample)
    assert not very_low_amplitude_kurtosis_check(
        stereo_audio_sample
    ), f"very_low_amplitude_kurtosis_check flagged (kurtosis={m:.2f})"


def test_very_high_amplitude_kurtosis_check(
    stereo_audio_sample: Audio,
) -> None:
    """very_high_amplitude_kurtosis_check returns False."""
    m = amplitude_kurtosis_metric(stereo_audio_sample)
    assert not very_high_amplitude_kurtosis_check(
        stereo_audio_sample
    ), f"very_high_amplitude_kurtosis_check flagged (kurtosis={m:.2f})"


def test_very_low_amplitude_modulation_depth_check(
    stereo_audio_sample: Audio,
) -> None:
    """very_low_amplitude_modulation_depth_check returns False."""
    m = amplitude_modulation_depth_metric(stereo_audio_sample)
    assert not very_low_amplitude_modulation_depth_check(
        stereo_audio_sample
    ), f"very_low_amplitude_modulation_depth_check flagged (mod_depth={m:.4f})"


def test_low_amplitude_modulation_depth_check(
    stereo_audio_sample: Audio,
) -> None:
    """low_amplitude_modulation_depth_check returns False."""
    m = amplitude_modulation_depth_metric(stereo_audio_sample)
    assert not low_amplitude_modulation_depth_check(
        stereo_audio_sample
    ), f"low_amplitude_modulation_depth_check flagged (mod_depth={m:.4f})"


def test_high_proportion_clipped_check(stereo_audio_sample: Audio) -> None:
    """high_proportion_clipped_check returns False."""
    m = proportion_clipped_metric(stereo_audio_sample)
    assert not high_proportion_clipped_check(
        stereo_audio_sample
    ), f"high_proportion_clipped_check flagged (clip_prop={m:.6f})"


def test_clipping_present_check(stereo_audio_sample: Audio) -> None:
    """clipping_present_check returns False."""
    m = proportion_clipped_metric(stereo_audio_sample)
    assert not clipping_present_check(stereo_audio_sample), f"clipping_present_check flagged (clip_prop={m:.6f})"


def test_completely_silent_check(stereo_audio_sample: Audio) -> None:
    """completely_silent_check returns False."""
    m = proportion_silent_metric(stereo_audio_sample)
    assert not completely_silent_check(stereo_audio_sample), f"completely_silent_check flagged (silent_prop={m:.4f})"


def test_mostly_silent_check(stereo_audio_sample: Audio) -> None:
    """mostly_silent_check returns False."""
    m = proportion_silent_metric(stereo_audio_sample)
    assert not mostly_silent_check(stereo_audio_sample), f"mostly_silent_check flagged (silent_prop={m:.4f})"


def test_high_amplitude_skew_magnitude_check(
    stereo_audio_sample: Audio,
) -> None:
    """high_amplitude_skew_magnitude_check returns False."""
    m = amplitude_skew_metric(stereo_audio_sample)
    assert not high_amplitude_skew_magnitude_check(
        stereo_audio_sample
    ), f"high_amplitude_skew_magnitude_check flagged (skew={m:.4f})"


def test_high_crest_factor_check(stereo_audio_sample: Audio) -> None:
    """high_crest_factor_check returns False."""
    m = crest_factor_metric(stereo_audio_sample)
    assert not high_crest_factor_check(stereo_audio_sample), f"high_crest_factor_check flagged (crest={m:.2f})"


def test_low_crest_factor_check(stereo_audio_sample: Audio) -> None:
    """low_crest_factor_check returns False."""
    m = crest_factor_metric(stereo_audio_sample)
    assert not low_crest_factor_check(stereo_audio_sample), f"low_crest_factor_check flagged (crest={m:.2f})"


def test_very_low_dynamic_range_check(stereo_audio_sample: Audio) -> None:
    """very_low_dynamic_range_check returns False."""
    m = dynamic_range_metric(stereo_audio_sample)
    assert not very_low_dynamic_range_check(
        stereo_audio_sample
    ), f"very_low_dynamic_range_check flagged (dyn_range={m:.4f})"


def test_very_high_dynamic_range_check(stereo_audio_sample: Audio) -> None:
    """very_high_dynamic_range_check returns False."""
    m = dynamic_range_metric(stereo_audio_sample)
    assert not very_high_dynamic_range_check(
        stereo_audio_sample
    ), f"very_high_dynamic_range_check flagged (dyn_range={m:.4f})"


def test_very_low_mean_absolute_deviation_check(
    stereo_audio_sample: Audio,
) -> None:
    """very_low_mean_absolute_deviation_check returns False."""
    m = mean_absolute_deviation_metric(stereo_audio_sample)
    assert not very_low_mean_absolute_deviation_check(
        stereo_audio_sample
    ), f"very_low_mean_absolute_deviation_check flagged (MAD={m:.6f})"


def test_very_high_mean_absolute_deviation_check(
    stereo_audio_sample: Audio,
) -> None:
    """very_high_mean_absolute_deviation_check returns False."""
    m = mean_absolute_deviation_metric(stereo_audio_sample)
    assert not very_high_mean_absolute_deviation_check(
        stereo_audio_sample
    ), f"very_high_mean_absolute_deviation_check flagged (MAD={m:.6f})"


def test_very_low_peak_snr_from_spectral_check(
    stereo_audio_sample: Audio,
) -> None:
    """very_low_peak_snr_from_spectral_check returns False."""
    m = peak_snr_from_spectral_metric(stereo_audio_sample)
    assert not very_low_peak_snr_from_spectral_check(
        stereo_audio_sample
    ), f"very_low_peak_snr_from_spectral_check flagged (SNR={m:.2f} dB)"


def test_low_peak_snr_from_spectral_check(stereo_audio_sample: Audio) -> None:
    """low_peak_snr_from_spectral_check returns False."""
    m = peak_snr_from_spectral_metric(stereo_audio_sample)
    assert low_peak_snr_from_spectral_check(
        stereo_audio_sample
    ), f"low_peak_snr_from_spectral_check flagged (SNR={m:.2f} dB)"


def test_very_high_peak_snr_from_spectral_check(
    stereo_audio_sample: Audio,
) -> None:
    """very_high_peak_snr_from_spectral_check returns False."""
    m = peak_snr_from_spectral_metric(stereo_audio_sample)
    assert not very_high_peak_snr_from_spectral_check(
        stereo_audio_sample
    ), f"very_high_peak_snr_from_spectral_check flagged (SNR={m:.2f} dB)"


def test_low_phase_correlation_check(stereo_audio_sample: Audio) -> None:
    """low_phase_correlation_check returns False."""
    m = phase_correlation_metric(stereo_audio_sample)
    assert low_phase_correlation_check(stereo_audio_sample), f"low_phase_correlation_check flagged (corr={m:.4f})"


def test_low_spectral_gating_snr_check(stereo_audio_sample: Audio) -> None:
    """low_spectral_gating_snr_check returns False."""
    m = spectral_gating_snr_metric(stereo_audio_sample)
    assert not low_spectral_gating_snr_check(
        stereo_audio_sample
    ), f"low_spectral_gating_snr_check flagged (SNR={m:.2f} dB)"


def test_high_spectral_gating_snr_check(stereo_audio_sample: Audio) -> None:
    """high_spectral_gating_snr_check returns False."""
    m = spectral_gating_snr_metric(stereo_audio_sample)
    assert not high_spectral_gating_snr_check(
        stereo_audio_sample
    ), f"high_spectral_gating_snr_check flagged (SNR={m:.2f} dB)"


def test_high_proportion_silence_at_beginning_check(
    stereo_audio_sample: Audio,
) -> None:
    """high_proportion_silence_at_beginning_check returns False."""
    m = proportion_silence_at_beginning_metric(stereo_audio_sample)
    msg = "high_proportion_silence_at_beginning_check flagged"
    msg += f" (lead_silence={m:.4f})"
    check_result = high_proportion_silence_at_beginning_check(stereo_audio_sample)
    assert not check_result, msg


def test_high_proportion_silence_at_end_check(
    stereo_audio_sample: Audio,
) -> None:
    """high_proportion_silence_at_end_check returns False."""
    m = proportion_silence_at_end_metric(stereo_audio_sample)
    assert not high_proportion_silence_at_end_check(
        stereo_audio_sample
    ), f"high_proportion_silence_at_end_check flagged (trail_silence={m:.4f})"


def test_very_low_root_mean_square_energy_check(
    stereo_audio_sample: Audio,
) -> None:
    """very_low_root_mean_square_energy_check returns False."""
    m = root_mean_square_energy_metric(stereo_audio_sample)
    assert not very_low_root_mean_square_energy_check(
        stereo_audio_sample
    ), f"very_low_root_mean_square_energy_check flagged (RMS={m:.5f})"


def test_very_high_root_mean_square_energy_check(
    stereo_audio_sample: Audio,
) -> None:
    """very_high_root_mean_square_energy_check returns False."""
    m = root_mean_square_energy_metric(stereo_audio_sample)
    assert not very_high_root_mean_square_energy_check(
        stereo_audio_sample
    ), f"very_high_root_mean_square_energy_check flagged (RMS={m:.5f})"


def test_low_shannon_entropy_amplitude_check(
    stereo_audio_sample: Audio,
) -> None:
    """low_shannon_entropy_amplitude_check returns False."""
    m = shannon_entropy_amplitude_metric(stereo_audio_sample)
    assert not low_shannon_entropy_amplitude_check(
        stereo_audio_sample
    ), f"low_shannon_entropy_amplitude_check flagged (entropy={m:.2f} bits)"


def test_high_shannon_entropy_amplitude_check(
    stereo_audio_sample: Audio,
) -> None:
    """high_shannon_entropy_amplitude_check returns False."""
    m = shannon_entropy_amplitude_metric(stereo_audio_sample)
    assert not high_shannon_entropy_amplitude_check(
        stereo_audio_sample
    ), f"high_shannon_entropy_amplitude_check flagged (entropy={m:.2f} bits)"


def test_low_signal_variance_check(stereo_audio_sample: Audio) -> None:
    """low_signal_variance_check returns False."""
    m = signal_variance_metric(stereo_audio_sample)
    assert not low_signal_variance_check(stereo_audio_sample), f"low_signal_variance_check flagged (var={m:.6f})"


def test_high_signal_variance_check(stereo_audio_sample: Audio) -> None:
    """high_signal_variance_check returns False."""
    m = signal_variance_metric(stereo_audio_sample)
    assert not high_signal_variance_check(stereo_audio_sample), f"high_signal_variance_check flagged (var={m:.6f})"


def test_low_zero_crossing_rate_check(stereo_audio_sample: Audio) -> None:
    """low_zero_crossing_rate_check returns False."""
    m = zero_crossing_rate_metric(stereo_audio_sample)
    assert not low_zero_crossing_rate_check(stereo_audio_sample), f"low_zero_crossing_rate_check flagged (ZCR={m:.4f})"


def test_high_zero_crossing_rate_check(stereo_audio_sample: Audio) -> None:
    """high_zero_crossing_rate_check returns False."""
    m = zero_crossing_rate_metric(stereo_audio_sample)
    assert not high_zero_crossing_rate_check(
        stereo_audio_sample
    ), f"high_zero_crossing_rate_check flagged (ZCR={m:.4f})"


def test_very_high_zero_crossing_rate_check(
    stereo_audio_sample: Audio,
) -> None:
    """very_high_zero_crossing_rate_check returns False."""
    m = zero_crossing_rate_metric(stereo_audio_sample)
    assert not very_high_zero_crossing_rate_check(
        stereo_audio_sample
    ), f"very_high_zero_crossing_rate_check flagged (ZCR={m:.4f})"


def test_audio_intensity_positive_check(stereo_audio_sample: Audio) -> None:
    """audio_intensity_positive_check returns True for non-zero range."""
    m = dynamic_range_metric(stereo_audio_sample)
    assert audio_intensity_positive_check(
        stereo_audio_sample
    ), f"audio_intensity_positive_check returned False (dynamic_range={m:.4f})"
