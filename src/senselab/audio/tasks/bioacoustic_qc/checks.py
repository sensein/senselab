"""Audio Quality Control Checks.

Each check function processes a list of Audio objects and returns a dictionary
indicating which files should be excluded or reviewed, along with a list of
audio files that passed the check.

All checks take a list of Audio objects as input and return:
    - A dictionary with the structure:
        {
            "exclude": [Audio],  # List of Audio objects that failed the check
            "review": [Audio]    # List of Audio objects that need manual review
        }
    - A list of Audio objects that passed the check.
"""

import os
from typing import Callable, Dict, List, Optional

import pandas as pd
import torch

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


def get_metric(audio: Audio, metric_func: Callable[[Audio], float], df: Optional[pd.DataFrame] = None) -> float:
    """Returns the metric value from a DataFrame if available; otherwise computes it.

    The metric column is inferred from the name of the function (e.g., 'zero_crossing_rate_metric').

    Args:
        audio (Audio): The SenseLab Audio object.
        metric_func (Callable): Function to compute the metric.
        df (Optional[pd.DataFrame]): DataFrame with precomputed metrics.
            Must contain 'audio_path_or_id' column and metric_func.__name__ column.

    Returns:
        float: The metric value.
    """
    metric_name = metric_func.__name__

    filepath = audio.filepath()
    if df is not None and filepath:
        audio_file_name = os.path.basename(filepath)
        row = df[df["audio_path_or_id"] == audio_file_name]
        if not row.empty and metric_name in row.columns:
            return row[metric_name].iloc[0]

    return metric_func(audio)


def audio_length_positive_check(audio: Audio) -> bool:
    """Checks if an Audio object has a positive length."""
    return audio.waveform.numel() != 0


def very_low_headroom_check(
    audio: Audio,
    headroom_threshold: float = 0.005,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    return get_metric(audio, amplitude_headroom_metric, df) < headroom_threshold


def very_high_headroom_check(
    audio: Audio,
    headroom_threshold: float = 0.95,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    return get_metric(audio, amplitude_headroom_metric, df) > headroom_threshold

def very_low_amplitude_interquartile_range_check(
    audio: Audio,
    threshold: float = 0.01,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    return get_metric(audio, amplitude_interquartile_range_metric, df) < threshold


def very_high_amplitude_interquartile_range_check(
    audio: Audio,
    threshold: float = 1.5,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    return get_metric(audio, amplitude_interquartile_range_metric, df) > threshold


def very_low_amplitude_kurtosis_check(
    audio: Audio,
    threshold: float = -100,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    return get_metric(audio, amplitude_kurtosis_metric, df) < threshold


def very_high_amplitude_kurtosis_check(
    audio: Audio,
    threshold: float = 100,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    return get_metric(audio, amplitude_kurtosis_metric, df) > threshold



def very_low_amplitude_modulation_depth_check(
    audio: Audio,
    threshold: float = 0.1,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    return get_metric(audio, amplitude_modulation_depth_metric, df) < threshold


def low_amplitude_modulation_depth_check(
    audio: Audio,
    min: float = 0.1,
    max: float = 0.3,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    depth = get_metric(audio, amplitude_modulation_depth_metric, df)
    return min <= depth < max




def proportion_clipped_check(
    audio: Audio,
    threshold: float = 0.0001,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    return get_metric(audio, proportion_clipped_metric, df) < threshold


def clipping_present_check(
    audio: Audio,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    return get_metric(audio, proportion_clipped_metric, df) > 0



def completely_silent_check(
    audio: Audio,
    silent_proportion: float = 1.0,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    return get_metric(audio, proportion_silent_metric, df) >= silent_proportion


def mostly_silent_check(
    audio: Audio,
    silent_proportion: float = 0.95,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    return get_metric(audio, proportion_silent_metric, df) > silent_proportion



def high_amplitude_skew_magnitude_check(
    audio: Audio,
    magnitude: float = 5.0,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    return abs(get_metric(audio, amplitude_skew_metric, df)) <= magnitude




def high_crest_factor_check(
    audio: Audio,
    threshold: float = 20.0,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    return get_metric(audio, crest_factor_metric, df) >= threshold


def low_crest_factor_check(
    audio: Audio,
    threshold: float = 1.5,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    return get_metric(audio, crest_factor_metric, df) <= threshold

def very_low_dynamic_range_check(
    audio: Audio,
    threshold: float = 0.1,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    return get_metric(audio, dynamic_range_metric, df) < threshold


def very_high_dynamic_range_check(
    audio: Audio,
    threshold: float = 1.9,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    return get_metric(audio, dynamic_range_metric, df) > threshold




def very_low_mean_absolute_deviation_check(
    audio: Audio,
    threshold: float = 0.001,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    return get_metric(audio, mean_absolute_deviation_metric, df) < threshold


def very_high_mean_absolute_deviation_check(
    audio: Audio,
    threshold: float = 0.5,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    return get_metric(audio, mean_absolute_deviation_metric, df) > threshold


def very_low_peak_snr_from_spectral_check(
    audio: Audio,
    threshold: float = 10.0,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    return get_metric(audio, peak_snr_from_spectral_metric, df) < threshold


def low_peak_snr_from_spectral_check(
    audio: Audio,
    lower: float = 10.0,
    upper: float = 20.0,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    snr = get_metric(audio, peak_snr_from_spectral_metric, df)
    return lower <= snr < upper


def very_high_peak_snr_from_spectral_check(
    audio: Audio,
    threshold: float = 60.0,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    return get_metric(audio, peak_snr_from_spectral_metric, df) > threshold



def low_phase_correlation_check(
    audio: Audio,
    threshold: float = 0.99,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    return get_metric(audio, phase_correlation_metric, df) < threshold




def high_proportion_silence_at_beginning_check(
    audio: Audio,
    threshold: float = 0.2,
) -> bool:
    return proportion_silence_at_beginning_metric(audio) > threshold


def high_proportion_silence_at_end_check(
    audio: Audio,
    threshold: float = 0.2,
) -> bool:
    return proportion_silence_at_end_metric(audio) > threshold



def very_low_root_mean_square_energy_check(
    audio: Audio,
    threshold: float = 0.005,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    return get_metric(audio, root_mean_square_energy_metric, df) < threshold


def very_high_root_mean_square_energy_check(
    audio: Audio,
    threshold: float = 0.5,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    return get_metric(audio, root_mean_square_energy_metric, df) > threshold


def low_shannon_entropy_amplitude_check(audio: Audio, threshold: float = 2.0) -> bool:
    """Returns True if Shannon entropy of amplitude is below the specified threshold.

    Args:
        audio (Audio): The SenseLab Audio object.
        threshold (float): Minimum acceptable entropy (bits).

    Returns:
        bool: True if entropy < threshold, False otherwise.
    """
    return shannon_entropy_amplitude_metric(audio) < threshold


def high_shannon_entropy_amplitude_check(audio: Audio, threshold: float = 7.5) -> bool:
    """Returns True if Shannon entropy of amplitude is above the specified threshold.

    Args:
        audio (Audio): The SenseLab Audio object.
        threshold (float): Maximum acceptable entropy (bits).

    Returns:
        bool: True if entropy > threshold, False otherwise.
    """
    return shannon_entropy_amplitude_metric(audio) > threshold


def low_signal_variance_check(audio: Audio, threshold: float = 1e-4) -> bool:
    """Returns True if signal variance is below the specified threshold (too flat).

    Args:
        audio (Audio): The SenseLab Audio object.
        threshold (float): Minimum acceptable signal variance.

    Returns:
        bool: True if variance < threshold, False otherwise.
    """
    return signal_variance_metric(audio) < threshold


def high_signal_variance_check(audio: Audio, threshold: float = 0.3) -> bool:
    """Returns True if signal variance is above the specified threshold (possibly distorted or noisy).

    Args:
        audio (Audio): The SenseLab Audio object.
        threshold (float): Maximum acceptable signal variance.

    Returns:
        bool: True if variance > threshold, False otherwise.
    """
    return signal_variance_metric(audio) > threshold


def low_spectral_gating_snr_check(audio: Audio, threshold: float = 10.0) -> bool:
    """Returns True if spectral gating SNR is below the specified threshold (too noisy or silent).

    Args:
        audio (Audio): The SenseLab Audio object.
        threshold (float): Minimum acceptable SNR (dB).

    Returns:
        bool: True if SNR < threshold, False otherwise.
    """
    return spectral_gating_snr_metric(audio) < threshold


def high_spectral_gating_snr_check(audio: Audio, threshold: float = 60.0) -> bool:
    """Returns True if spectral gating SNR is above the specified threshold (possibly artificial or clipped).

    Args:
        audio (Audio): The SenseLab Audio object.
        threshold (float): Maximum acceptable SNR (dB).

    Returns:
        bool: True if SNR > threshold, False otherwise.
    """
    return spectral_gating_snr_metric(audio) > threshold


def low_zero_crossing_rate_metric_check(audio: Audio, threshold: float = 0.01) -> bool:
    """Returns True if zero-crossing rate is below the specified threshold (possibly silent or DC offset)."""
    return zero_crossing_rate_metric(audio) < threshold


def high_zero_crossing_rate_metric_check(audio: Audio, lower: float = 0.15, upper: float = 0.3) -> bool:
    """Returns True if zero-crossing rate is between the specified lower and upper thresholds.

    Args:
        audio (Audio): The SenseLab Audio object.
        lower (float): Lower bound of high ZCR range (inclusive).
        upper (float): Upper bound of high ZCR range (exclusive).

    Returns:
        bool: True if lower <= ZCR < upper, False otherwise.
    """
    zcr = zero_crossing_rate_metric(audio)
    return lower <= zcr < upper


def very_high_zero_crossing_rate_metric_check(audio: Audio, threshold: float = 0.3) -> bool:
    """Returns True if zero-crossing rate is far above normal range (likely noise or corrupted)."""
    return zero_crossing_rate_metric(audio) > threshold
