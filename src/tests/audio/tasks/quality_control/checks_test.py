"""Runs all quality control checks on a stereo sample.

If a test fails, the assertion message shows the metric value.
"""

from __future__ import annotations

from typing import Any, Dict

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.quality_control.checks import (
    audio_intensity_zero_check,
    audio_length_zero_check,
    clipping_present_check,
    completely_silent_check,
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
    primary_speaker_ratio_check,
    very_high_amplitude_interquartile_range_check,
    very_high_amplitude_kurtosis_check,
    very_high_headroom_check,
    very_high_mean_absolute_deviation_check,
    very_high_peak_snr_from_spectral_check,
    very_high_root_mean_square_energy_check,
    very_high_zero_crossing_rate_check,
    very_low_amplitude_kurtosis_check,
    very_low_amplitude_modulation_depth_check,
    very_low_dynamic_range_check,
    very_low_mean_absolute_deviation_check,
    very_low_peak_snr_from_spectral_check,
    very_low_root_mean_square_energy_check,
    voice_activity_detection_check,
)
from senselab.audio.tasks.quality_control.evaluate import get_evaluation
from senselab.audio.tasks.quality_control.metrics import (
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
    primary_speaker_ratio_metric,
    proportion_clipped_metric,
    proportion_silence_at_beginning_metric,
    proportion_silence_at_end_metric,
    proportion_silent_metric,
    root_mean_square_energy_metric,
    shannon_entropy_amplitude_metric,
    signal_variance_metric,
    spectral_gating_snr_metric,
    voice_activity_detection_metric,
    zero_crossing_rate_metric,
)


def test_get_evaluation_computes_directly(stereo_audio_sample: Audio) -> None:
    """Should compute the metric when no existing results are provided."""
    value = get_evaluation(stereo_audio_sample, zero_crossing_rate_metric)
    assert isinstance(value, float)
    assert 0 <= value <= 1


def test_get_evaluation_uses_cache(stereo_audio_sample: Audio) -> None:
    """Should return cached value when existing results are provided."""
    cached_value = 0.1234
    existing_results = {"evaluations": {"zero_crossing_rate_metric": cached_value}}
    value = get_evaluation(stereo_audio_sample, zero_crossing_rate_metric, existing_results)
    assert value == cached_value


def test_get_evaluation_when_no_cache_match(stereo_audio_sample: Audio) -> None:
    """Should compute directly when no cache match is found."""
    cached_value = 0.1234
    existing_results = {
        "evaluations": {
            "some_other_metric": cached_value  # Different metric, no match
        }
    }

    value = get_evaluation(stereo_audio_sample, zero_crossing_rate_metric, existing_results)
    # Should compute directly since no cache match
    assert isinstance(value, float)
    assert 0 <= value <= 1
    # Should NOT be the cached value since metric doesn't match
    assert value != cached_value


def test_get_evaluation_computes_without_cache(
    stereo_audio_sample: Audio,
) -> None:
    """Should compute metric directly when no existing results provided."""
    # Create an Audio object with no filepath
    audio_no_filepath = Audio(waveform=stereo_audio_sample.waveform, sampling_rate=stereo_audio_sample.sampling_rate)

    # Verify filepath is None
    assert audio_no_filepath.filepath() is None

    # First call should compute the metric
    value1 = get_evaluation(audio_no_filepath, zero_crossing_rate_metric)
    assert isinstance(value1, float)
    assert 0 <= value1 <= 1

    # Second call should compute again (no caching without existing_results)
    value2 = get_evaluation(audio_no_filepath, zero_crossing_rate_metric)
    assert value2 == value1  # Should be same since same audio


def test_get_evaluation_computes_when_not_in_cache(
    stereo_audio_sample: Audio,
) -> None:
    """Should compute metric when not found in existing results."""
    cached_value = 0.1234
    existing_results = {
        "evaluations": {
            "some_other_metric": cached_value  # Different metric
        }
    }
    value = get_evaluation(stereo_audio_sample, zero_crossing_rate_metric, existing_results)
    assert value != cached_value
    assert isinstance(value, float)
    assert 0 <= value <= 1


def test_get_evaluation_with_empty_cache(
    stereo_audio_sample: Audio,
) -> None:
    """Should compute metric when existing results are empty."""
    # Create empty existing results
    existing_results: Dict[str, Any] = {
        "evaluations": {}  # Empty cache
    }

    # Should compute directly since no cached values
    value = get_evaluation(stereo_audio_sample, zero_crossing_rate_metric, existing_results)
    assert isinstance(value, float)
    assert 0 <= value <= 1


def test_get_evaluation_returns_none_when_filepath_not_exists() -> None:
    """Should return None when filepath doesn't exist."""
    # Pass a filepath string that doesn't exist
    result = get_evaluation("nonexistent_file.wav", zero_crossing_rate_metric)
    assert result is None


def test_audio_length_zero_check(stereo_audio_sample: Audio) -> None:
    """audio_length_zero_check returns False for non-empty audio."""
    n = stereo_audio_sample.waveform.numel()
    assert not audio_length_zero_check(stereo_audio_sample), f"audio_length_zero_check returned True (num_samples={n})"


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


def test_audio_intensity_zero_check(stereo_audio_sample: Audio) -> None:
    """audio_intensity_zero_check returns False for non-zero range."""
    m = dynamic_range_metric(stereo_audio_sample)
    assert not audio_intensity_zero_check(
        stereo_audio_sample
    ), f"audio_intensity_zero_check returned True (dynamic_range={m:.4f})"


def test_voice_activity_detection_check_with_metadata() -> None:
    """Tests voice_activity_detection_check with precomputed VAD in metadata."""
    from senselab.utils.data_structures import ScriptLine
    import torch

    # Create audio with VAD metadata
    waveform = torch.randn(1, 16000)
    vad_result = [
        ScriptLine(speaker="VOICE", start=0.0, end=0.5),
        ScriptLine(speaker="VOICE", start=0.6, end=1.0),
    ]
    metadata = {"vad": vad_result}
    audio = Audio(waveform=waveform, sampling_rate=16000, metadata=metadata)

    # Test with threshold below voice duration
    result = voice_activity_detection_check(audio, threshold=0.5)
    assert result is True, "Should return True when voice duration > threshold"

    # Test with threshold above voice duration
    result = voice_activity_detection_check(audio, threshold=1.0)
    assert result is False, "Should return False when voice duration <= threshold"


def test_voice_activity_detection_check_no_voice() -> None:
    """Tests voice_activity_detection_check when no voice is detected."""
    import torch

    # Create audio with empty VAD result
    waveform = torch.randn(1, 16000)
    metadata = {"vad": []}
    audio = Audio(waveform=waveform, sampling_rate=16000, metadata=metadata)

    # Test with threshold
    result = voice_activity_detection_check(audio, threshold=0.0)
    assert result is False, "Should return False when no voice detected"


def test_primary_speaker_ratio_check_with_metadata() -> None:
    """Tests primary_speaker_ratio_check with precomputed diarization in metadata."""
    from senselab.utils.data_structures import ScriptLine
    import torch

    # Create audio with diarization metadata (single speaker)
    waveform = torch.randn(1, 16000)
    diarization_result = [
        ScriptLine(speaker="SPEAKER_00", start=0.0, end=1.0),
    ]
    metadata = {"diarization": diarization_result}
    audio = Audio(waveform=waveform, sampling_rate=16000, metadata=metadata)

    # Test with threshold below ratio
    result = primary_speaker_ratio_check(audio, threshold=0.5)
    assert result is True, "Should return True when ratio > threshold"

    # Test with threshold above ratio
    result = primary_speaker_ratio_check(audio, threshold=0.9)
    assert result is True, "Should return True for single speaker (ratio=1.0)"


def test_primary_speaker_ratio_check_multiple_speakers() -> None:
    """Tests primary_speaker_ratio_check with multiple speakers."""
    from senselab.utils.data_structures import ScriptLine
    import torch

    # Create audio with diarization metadata (two speakers)
    waveform = torch.randn(1, 16000)
    diarization_result = [
        ScriptLine(speaker="SPEAKER_00", start=0.0, end=0.7),  # 70% of time
        ScriptLine(speaker="SPEAKER_01", start=0.7, end=1.0),  # 30% of time
    ]
    metadata = {"diarization": diarization_result}
    audio = Audio(waveform=waveform, sampling_rate=16000, metadata=metadata)

    # Test with threshold below ratio
    result = primary_speaker_ratio_check(audio, threshold=0.5)
    assert result is True, "Should return True when ratio (0.7) > threshold (0.5)"

    # Test with threshold above ratio
    result = primary_speaker_ratio_check(audio, threshold=0.8)
    assert result is False, "Should return False when ratio (0.7) <= threshold (0.8)"
