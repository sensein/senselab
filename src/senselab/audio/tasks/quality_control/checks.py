"""Contains tests for bioacoustic quality control checks.

This module defines a set of boolean functions that evaluate individual `Audio` objects
based on specific quality metrics (e.g., clipping, dynamic range, SNR, entropy).

Each check is designed to return `True` when a defined failure condition is met,
enabling easy filtering of problematic recordings.

All checks accept an `Audio` object and optionally a DataFrame of cached metric values.
"""

from typing import Optional, Union

import pandas as pd

from senselab.audio.data_structures import Audio
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
    proportion_clipped_metric,
    proportion_silence_at_beginning_metric,
    proportion_silence_at_end_metric,
    proportion_silent_metric,
    root_mean_square_energy_metric,
    shannon_entropy_amplitude_metric,
    signal_variance_metric,
    spectral_gating_snr_metric,
    zero_crossing_rate_metric,
    percent_clipping_metric,
    primary_speaker_ratio_metric,
    presence_of_voice_metric,
    signal_to_noise_power_ratio_metric

)


def audio_length_zero_check(audio: Audio) -> bool:
    """Check if the waveform is empty.

    Args:
        audio: Audio object to evaluate.

    Returns:
        True if the waveform contains no samples, else False.
    """
    return audio.waveform.numel() == 0


def audio_intensity_zero_check(
    audio_or_path: Union[Audio, str],
    df: Optional[pd.DataFrame] = None,
) -> Optional[bool]:
    """Check that the audio has completely zero intensity.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        df: Optional DataFrame with ``dynamic_range_metric``.

    Returns:
        True if the audio has completely zero dynamic range, None if evaluation fails.
    """
    result = get_evaluation(audio_or_path, dynamic_range_metric, df)
    if result is None:
        return None
    return float(result) == 0


def very_high_headroom_check(
    audio_or_path: Union[Audio, str],
    headroom_threshold: float = 0.95,
    df: Optional[pd.DataFrame] = None,
) -> Optional[bool]:
    """Detect overly quiet signals with excessive headroom.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        headroom_threshold: Minimum headroom fraction considered excessive.
        df: Optional DataFrame with ``amplitude_headroom_metric`` cached.

    Returns:
        True when headroom > ``headroom_threshold``, None if evaluation fails.
    """
    result = get_evaluation(audio_or_path, amplitude_headroom_metric, df)
    if result is None:
        return None
    return float(result) > headroom_threshold


def very_high_amplitude_interquartile_range_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 1.5,
    df: Optional[pd.DataFrame] = None,
) -> Optional[bool]:
    """Detect very wide amplitude spread (noisy or clipped).

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Minimum IQR considered *very high*.
        df: Optional DataFrame containing ``amplitude_interquartile_range_metric``.

    Returns:
        True when IQR > ``threshold``, None if evaluation fails.
    """
    result = get_evaluation(audio_or_path, amplitude_interquartile_range_metric, df)
    if result is None:
        return None
    return float(result) > threshold


def very_low_amplitude_kurtosis_check(
    audio_or_path: Union[Audio, str],
    threshold: float = -100,
    df: Optional[pd.DataFrame] = None,
) -> Optional[bool]:
    """Flag extremely flat/noisy distributions (low kurtosis).

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Upper bound regarded as *very low* kurtosis.
        df: Optional DataFrame with ``amplitude_kurtosis_metric``.

    Returns:
        True when kurtosis < ``threshold``, None if evaluation fails.
    """
    result = get_evaluation(audio_or_path, amplitude_kurtosis_metric, df)
    if result is None:
        return None
    return float(result) < threshold


def very_high_amplitude_kurtosis_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 100,
    df: Optional[pd.DataFrame] = None,
) -> Optional[bool]:
    """Flag spiky or clipped distributions (high kurtosis).

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Lower bound regarded as *very high* kurtosis.
        df: Optional DataFrame with ``amplitude_kurtosis_metric``.

    Returns:
        True when kurtosis > ``threshold``, None if evaluation fails.
    """
    result = get_evaluation(audio_or_path, amplitude_kurtosis_metric, df)
    if result is None:
        return None
    return float(result) > threshold


def very_low_amplitude_modulation_depth_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 0.1,
    df: Optional[pd.DataFrame] = None,
) -> Optional[bool]:
    """Detect audio with almost no loudness variation.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Maximum modulation depth regarded as *very low*.
        df: Optional DataFrame with ``amplitude_modulation_depth_metric``.

    Returns:
        True when modulation depth < ``threshold``, None if evaluation fails.
    """
    result = get_evaluation(audio_or_path, amplitude_modulation_depth_metric, df)
    if result is None:
        return None
    return float(result) < threshold


def low_amplitude_modulation_depth_check(
    audio_or_path: Union[Audio, str],
    min: float = 0.1,
    max: float = 0.3,
    df: Optional[pd.DataFrame] = None,
) -> Optional[bool]:
    """Detect audio with modestly low loudness variation.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        min: Inclusive lower bound for *low* modulation depth.
        max: Exclusive upper bound for *low* modulation depth.
        df: Optional DataFrame with ``amplitude_modulation_depth_metric``.

    Returns:
        True when ``min ≤ depth < max``, None if evaluation fails.
    """
    result = get_evaluation(audio_or_path, amplitude_modulation_depth_metric, df)
    if result is None:
        return None
    depth = float(result)
    return min <= depth < max


def high_proportion_clipped_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 0.0001,
    df: Optional[pd.DataFrame] = None,
) -> Optional[bool]:
    """Detect audio with a high proportion of clipped samples.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Maximum fraction of clipped samples allowed.
        df: Optional DataFrame with ``proportion_clipped_metric``.

    Returns:
        True when clipped proportion > ``threshold``, None if evaluation fails.
    """
    result = get_evaluation(audio_or_path, proportion_clipped_metric, df)
    if result is None:
        return None
    return float(result) > threshold


def clipping_present_check(
    audio_or_path: Union[Audio, str],
    df: Optional[pd.DataFrame] = None,
) -> Optional[bool]:
    """Detect whether *any* clipping is present.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        df: Optional DataFrame with ``proportion_clipped_metric``.

    Returns:
        True when clipped proportion > 0, None if evaluation fails.
    """
    result = get_evaluation(audio_or_path, proportion_clipped_metric, df)
    if result is None:
        return None
    return float(result) > 0


def completely_silent_check(
    audio_or_path: Union[Audio, str],
    silent_proportion: float = 1.0,
    df: Optional[pd.DataFrame] = None,
) -> Optional[bool]:
    """Detect recordings that are entirely silent.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        silent_proportion: Proportion that defines *complete* silence
            (default 1.0 means 100 %).
        df: Optional DataFrame with ``proportion_silent_metric``.

    Returns:
        True when silent proportion ≥ ``silent_proportion``, None if evaluation fails.
    """
    result = get_evaluation(audio_or_path, proportion_silent_metric, df)
    if result is None:
        return None
    return float(result) >= silent_proportion


def mostly_silent_check(
    audio_or_path: Union[Audio, str],
    silent_proportion: float = 0.99,
    df: Optional[pd.DataFrame] = None,
) -> Optional[bool]:
    """Detect recordings that are mostly silent.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        silent_proportion: Threshold above which audio is considered *mostly*
            silent.
        df: Optional DataFrame with ``proportion_silent_metric``.

    Returns:
        True when silent proportion > ``silent_proportion``, None if evaluation fails.
    """
    result = get_evaluation(audio_or_path, proportion_silent_metric, df)
    if result is None:
        return None
    return float(result) > silent_proportion


def high_amplitude_skew_magnitude_check(
    audio_or_path: Union[Audio, str],
    magnitude: float = 5.0,
    df: Optional[pd.DataFrame] = None,
) -> Optional[bool]:
    """Pass audio whose amplitude skew magnitude is ≤ ``magnitude``.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        magnitude: Maximum acceptable |skew|.
        df: Optional DataFrame with ``amplitude_skew_metric``.

    Returns:
        True when |skew| ≤ ``magnitude``, None if evaluation fails.
    """
    result = get_evaluation(audio_or_path, amplitude_skew_metric, df)
    if result is None:
        return None
    return abs(float(result)) > magnitude


def high_crest_factor_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 20.0,
    df: Optional[pd.DataFrame] = None,
) -> Optional[bool]:
    """Detect spiky signals with crest factor ≥ ``threshold``.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Minimum crest factor regarded as too high.
        df: Optional DataFrame with ``crest_factor_metric``.

    Returns:
        True when crest factor ≥ ``threshold``, None if evaluation fails.
    """
    result = get_evaluation(audio_or_path, crest_factor_metric, df)
    if result is None:
        return None
    return float(result) >= threshold


def low_crest_factor_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 1.5,
    df: Optional[pd.DataFrame] = None,
) -> Optional[bool]:
    """Detect over-compressed signals with crest factor ≤ ``threshold``.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Maximum crest factor regarded as too low.
        df: Optional DataFrame with ``crest_factor_metric``.

    Returns:
        True when crest factor ≤ ``threshold``, None if evaluation fails.
    """
    result = get_evaluation(audio_or_path, crest_factor_metric, df)
    if result is None:
        return None
    return float(result) <= threshold


def very_low_dynamic_range_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 0.1,
    df: Optional[pd.DataFrame] = None,
) -> Optional[bool]:
    """Detect signals with insufficient dynamic range.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Maximum dynamic range considered too low.
        df: Optional DataFrame with ``dynamic_range_metric``.

    Returns:
        True when dynamic range < ``threshold``, None if evaluation fails.
    """
    result = get_evaluation(audio_or_path, dynamic_range_metric, df)
    if result is None:
        return None
    return float(result) < threshold


def very_low_mean_absolute_deviation_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 0.001,
    df: Optional[pd.DataFrame] = None,
) -> Optional[bool]:
    """Detect nearly flat signals via very low MAD.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Maximum MAD regarded as too low.
        df: Optional DataFrame with ``mean_absolute_deviation_metric``.

    Returns:
        True when MAD < ``threshold``, None if evaluation fails.
    """
    result = get_evaluation(audio_or_path, mean_absolute_deviation_metric, df)
    if result is None:
        return None
    return float(result) < threshold


def very_high_mean_absolute_deviation_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 0.5,
    df: Optional[pd.DataFrame] = None,
) -> Optional[bool]:
    """Detect overly volatile signals via very high MAD.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Minimum MAD regarded as too high.
        df: Optional DataFrame with ``mean_absolute_deviation_metric``.

    Returns:
        True when MAD > ``threshold``, None if evaluation fails.
    """
    result = get_evaluation(audio_or_path, mean_absolute_deviation_metric, df)
    if result is None:
        return None
    return float(result) > threshold


def very_low_peak_snr_from_spectral_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 10.0,
    df: Optional[pd.DataFrame] = None,
) -> Optional[bool]:
    """Detect audio with very low peak-SNR (< ``threshold``).

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Minimum acceptable peak-SNR (dB).
        df: Optional DataFrame with ``peak_snr_from_spectral_metric``.

    Returns:
        True when peak-SNR < ``threshold``, None if evaluation fails.
    """
    result = get_evaluation(audio_or_path, peak_snr_from_spectral_metric, df)
    if result is None:
        return None
    return float(result) < threshold


def low_peak_snr_from_spectral_check(
    audio_or_path: Union[Audio, str],
    lower: float = 10.0,
    upper: float = 20.0,
    df: Optional[pd.DataFrame] = None,
) -> Optional[bool]:
    """Review audio whose peak-SNR falls in a *low* band.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        lower: Inclusive lower bound for the low SNR band.
        upper: Exclusive upper bound.
        df: Optional DataFrame with ``peak_snr_from_spectral_metric``.

    Returns:
        True when ``lower ≤ SNR < upper``, None if evaluation fails.
    """
    result = get_evaluation(audio_or_path, peak_snr_from_spectral_metric, df)
    if result is None:
        return None
    snr = float(result)
    return lower <= snr < upper


def very_high_peak_snr_from_spectral_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 60.0,
    df: Optional[pd.DataFrame] = None,
) -> Optional[bool]:
    """Flag unrealistically high peak-SNR (> ``threshold``).

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Peak-SNR regarded as suspiciously high (dB).
        df: Optional DataFrame with ``peak_snr_from_spectral_metric``.

    Returns:
        True when peak-SNR > ``threshold``, None if evaluation fails.
    """
    result = get_evaluation(audio_or_path, peak_snr_from_spectral_metric, df)
    if result is None:
        return None
    return float(result) > threshold


def low_phase_correlation_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 0.99,
    df: Optional[pd.DataFrame] = None,
) -> Optional[bool]:
    """Detect stereo signals with weak channel correlation.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Minimum acceptable correlation coefficient.
        df: Optional DataFrame with ``phase_correlation_metric``.

    Returns:
        True when correlation < ``threshold``, None if evaluation fails.
    """
    result = get_evaluation(audio_or_path, phase_correlation_metric, df)
    if result is None:
        return None
    return float(result) < threshold


def high_proportion_silence_at_beginning_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 0.2,
    df: Optional[pd.DataFrame] = None,
) -> Optional[bool]:
    """Flag recordings with > ``threshold`` leading silence.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Maximum acceptable leading silence proportion.
        df: Optional DataFrame with stored metric.

    Returns:
        True when leading silence proportion > ``threshold``, None if evaluation fails.
    """
    result = get_evaluation(audio_or_path, proportion_silence_at_beginning_metric, df)
    if result is None:
        return None
    return float(result) > threshold


def high_proportion_silence_at_end_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 0.2,
    df: Optional[pd.DataFrame] = None,
) -> Optional[bool]:
    """Flag recordings with > ``threshold`` trailing silence.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Maximum acceptable trailing silence proportion.
        df: Optional DataFrame with stored metric.

    Returns:
        True when trailing silence proportion > ``threshold``, None if evaluation fails.
    """
    result = get_evaluation(audio_or_path, proportion_silence_at_end_metric, df)
    if result is None:
        return None
    return float(result) > threshold


def very_low_root_mean_square_energy_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 0.005,
    df: Optional[pd.DataFrame] = None,
) -> Optional[bool]:
    """Detect audio that is too quiet (very low RMS).

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Maximum RMS energy regarded as very low.
        df: Optional DataFrame with ``root_mean_square_energy_metric``.

    Returns:
        True when RMS < ``threshold``, None if evaluation fails.
    """
    result = get_evaluation(audio_or_path, root_mean_square_energy_metric, df)
    if result is None:
        return None
    return float(result) < threshold


def very_high_root_mean_square_energy_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 0.5,
    df: Optional[pd.DataFrame] = None,
) -> Optional[bool]:
    """Detect audio that is too loud (very high RMS).

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Minimum RMS energy regarded as very high.
        df: Optional DataFrame with ``root_mean_square_energy_metric``.

    Returns:
        True when RMS > ``threshold``, None if evaluation fails.
    """
    result = get_evaluation(audio_or_path, root_mean_square_energy_metric, df)
    if result is None:
        return None
    return float(result) > threshold


def low_shannon_entropy_amplitude_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 2.0,
    df: Optional[pd.DataFrame] = None,
) -> Optional[bool]:
    """Detect overly predictable audio via low entropy.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Maximum entropy regarded as too low (bits).
        df: Optional DataFrame with ``shannon_entropy_amplitude_metric``.

    Returns:
        True when entropy < ``threshold``, None if evaluation fails.
    """
    result = get_evaluation(audio_or_path, shannon_entropy_amplitude_metric, df)
    if result is None:
        return None
    return float(result) < threshold


def high_shannon_entropy_amplitude_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 7.5,
    df: Optional[pd.DataFrame] = None,
) -> Optional[bool]:
    """Detect overly noisy audio via high entropy.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Minimum entropy regarded as too high (bits).
        df: Optional DataFrame with ``shannon_entropy_amplitude_metric``.

    Returns:
        True when entropy > ``threshold``, None if evaluation fails.
    """
    result = get_evaluation(audio_or_path, shannon_entropy_amplitude_metric, df)
    if result is None:
        return None
    return float(result) > threshold


def low_signal_variance_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 1e-4,
    df: Optional[pd.DataFrame] = None,
) -> Optional[bool]:
    """Detect signals with extremely small amplitude variance.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Maximum variance regarded as too low.
        df: Optional DataFrame with ``signal_variance_metric``.

    Returns:
        True when variance < ``threshold``, None if evaluation fails.
    """
    result = get_evaluation(audio_or_path, signal_variance_metric, df)
    if result is None:
        return None
    return float(result) < threshold


def high_signal_variance_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 0.3,
    df: Optional[pd.DataFrame] = None,
) -> Optional[bool]:
    """Detect signals with excessively large amplitude variance.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Minimum variance regarded as too high.
        df: Optional DataFrame with ``signal_variance_metric``.

    Returns:
        True when variance > ``threshold``, None if evaluation fails.
    """
    result = get_evaluation(audio_or_path, signal_variance_metric, df)
    if result is None:
        return None
    return float(result) > threshold


def low_spectral_gating_snr_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 10.0,
    df: Optional[pd.DataFrame] = None,
) -> Optional[bool]:
    """Detect recordings whose segmental SNR is < ``threshold`` dB.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Minimum acceptable segmental SNR (dB).
        df: Optional DataFrame with ``spectral_gating_snr_metric``.

    Returns:
        True when SNR < ``threshold``, None if evaluation fails.
    """
    result = get_evaluation(audio_or_path, spectral_gating_snr_metric, df)
    if result is None:
        return None
    return float(result) < threshold


def high_spectral_gating_snr_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 60.0,
    df: Optional[pd.DataFrame] = None,
) -> Optional[bool]:
    """Flag recordings whose segmental SNR is > ``threshold`` dB.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Segmental SNR considered suspiciously high (dB).
        df: Optional DataFrame with ``spectral_gating_snr_metric``.

    Returns:
        True when SNR > ``threshold``, None if evaluation fails.
    """
    result = get_evaluation(audio_or_path, spectral_gating_snr_metric, df)
    if result is None:
        return None
    return float(result) > threshold


def low_zero_crossing_rate_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 0.01,
    df: Optional[pd.DataFrame] = None,
) -> Optional[bool]:
    """Detect signals with ZCR below ``threshold`` (likely silent/DC).

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Minimum acceptable zero-crossing rate.
        df: Optional DataFrame with ``zero_crossing_rate_metric``.

    Returns:
        True when ZCR < ``threshold``, None if evaluation fails.
    """
    result = get_evaluation(audio_or_path, zero_crossing_rate_metric, df)
    if result is None:
        return None
    return float(result) < threshold


def high_zero_crossing_rate_check(
    audio_or_path: Union[Audio, str],
    lower: float = 0.15,
    upper: float = 0.3,
    df: Optional[pd.DataFrame] = None,
) -> Optional[bool]:
    """Review signals with ZCR in a high (but not extreme) range.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        lower: Inclusive lower bound for the *high* ZCR range.
        upper: Exclusive upper bound for the *high* ZCR range.
        df: Optional DataFrame with ``zero_crossing_rate_metric``.

    Returns:
        True when ``lower ≤ ZCR < upper``, None if evaluation fails.
    """
    result = get_evaluation(audio_or_path, zero_crossing_rate_metric, df)
    if result is None:
        return None
    zcr = float(result)
    return lower <= zcr < upper


def very_high_zero_crossing_rate_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 0.3,
    df: Optional[pd.DataFrame] = None,
) -> Optional[bool]:
    """Detect signals with extremely high ZCR (noise / corruption).

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        threshold: Zero-crossing rate regarded as *very high*.
        df: Optional DataFrame with ``zero_crossing_rate_metric``.

    Returns:
        True when ZCR > ``threshold``, None if evaluation fails.
    """
    result = get_evaluation(audio_or_path, zero_crossing_rate_metric, df)
    if result is None:
        return None
    return float(result) > threshold


def audio_intensity_positive_check(
    audio_or_path: Union[Audio, str],
    df: Optional[pd.DataFrame] = None,
) -> Optional[bool]:
    """Check that the audio has non-zero intensity.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        df: Optional DataFrame with ``dynamic_range_metric``.

    Returns:
        True if the audio has non-zero dynamic range, None if evaluation fails.
    """
    result = get_evaluation(audio_or_path, dynamic_range_metric, df)
    if result is None:
        return None
    return float(result) > 0


##### Rahul's Code Below


# calculate clipping
def measure_clipping_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 0.001,
    df: Optional[pd.DataFrame] = None,
) -> Optional[bool]:
    """
    Checks for clipping in an audio file.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.

    Returns:
        True when clipping percent > ``threshold``, None if evaluation fails.
    """

    result = get_evaluation(audio_or_path, percent_clipping_metric, df)
    if result is None:
        return None
    return float(result) > threshold



def primary_speaker_ratio_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 0.8,
    df: Optional[pd.DataFrame] = None,
) -> Optional[bool]:
    """
    Checks that primary speaker ratio in an audio file (the outputs from diarization and then the the ratio of the most common speaker to the total duration) is above threshold

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.

    Returns:
        True when primary speaker > ``threshold``, None if evaluation fails.
    """

    result = get_evaluation(audio_or_path, "../../modeling/diarize/diar_r2.pkl", primary_speaker_ratio_metric, df)
    if result is None:
        return None
    # x.speaker_count > 1 and x.primary_speaker_ratio < 0.8 TODO do I need the first threshold; I don't think so since it should be 1 for 1 speaker
    return float(result) > threshold



def presence_of_voice_check(
    audio_or_path: Union[Audio, str],
    threshold: float = 0,
    df: Optional[pd.DataFrame] = None,
) -> Optional[bool]:
    """
    Check that Voice Activity Detection duration is more than 0.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.

    Returns:
        True when primary speaker > ``threshold``, None if evaluation fails.
    """

    result = get_evaluation(audio_or_path, "../../modeling/diarize/vad_r2.pkl", presence_of_voice_metric, df)
    if result is None:
        return None
    return float(result) > threshold



def signal_to_noise_power_ratio_check(audio_or_path: Union[Audio, str],
    threshold: float = -1,
    df: Optional[pd.DataFrame] = None,
) -> Optional[bool]:
    """
    Check that SNR is below a threshold. Currently not a lot of samples with major background noise, or this algorithm isn't doing a great job

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.

    Returns:
        True when SNR > ``threshold``, None if evaluation fails.
    """
    
    result = get_evaluation(audio_or_path, "../../modeling/diarize/vad_r2.pkl", signal_to_noise_power_ratio_metric, df)
    if result is None:
        return None
    return float(result) > threshold



def find_buzzing_check(audio_or_path: Union[Audio, str],
    threshold: float = #TODO,
    df: Optional[pd.DataFrame] = None,
) -> Optional[bool]:
    """
    Check that SNR is below a threshold. Currently not a lot of samples with major background noise, or this algorithm isn't doing a great job

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.

    Returns:
        True when SNR > ``threshold``, None if evaluation fails.
    """
    
    result = get_evaluation(audio_or_path, #"../../modeling/diarize/vad_r2.pkl", 
                            find_buzzing_metric, df)
    if result is None:
        return None
    return float(result) > threshold
        

