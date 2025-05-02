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

from typing import Callable, Dict, List

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


def audio_length_positive_check(audio: Audio) -> bool:
    """Checks if an Audio object has a positive length."""
    return audio.waveform.numel() != 0


def very_low_headroom_check(audio: Audio, headroom_threshold: float = 0.005) -> bool:
    """Checks if an Audio object has very low headroom."""
    return amplitude_headroom_metric(audio) < headroom_threshold


def very_high_headroom_check(audio: Audio, headroom_threshold: float = 0.95) -> bool:
    """Checks if an Audio object has very high headroom."""
    return amplitude_headroom_metric(audio) > headroom_threshold


def very_low_amplitude_interquartile_range_check(audio: Audio, threshold: float = 0.01) -> bool:
    """Checks if an Audio object has very low IQR."""
    return amplitude_interquartile_range_metric(audio) < threshold


def very_high_amplitude_interquartile_range_check(audio: Audio, threshold: float = 1.5) -> bool:
    """Checks if an Audio object has very high IQR."""
    return amplitude_interquartile_range_metric(audio) > threshold


def very_low_amplitude_kurtosis_check(audio: Audio, threshold: float = -100) -> bool:
    """Checks if an Audio object has very low amplitude kurtosis."""
    return amplitude_kurtosis_metric(audio) < threshold


def very_high_amplitude_kurtosis_check(audio: Audio, threshold: float = 100) -> bool:
    """Checks if an Audio object has very high amplitude kurtosis."""
    return amplitude_kurtosis_metric(audio) > threshold


def very_low_amplitude_modulation_depth_check(audio: Audio, threshold: float = 0.1) -> bool:
    """Checks if an Audio object has very low amplitude modulation depth."""
    return amplitude_modulation_depth_metric(audio) < threshold


def low_amplitude_modulation_depth_check(audio: Audio, min: float = 0.1, max: float = 0.3) -> bool:
    """Checks if an Audio object has very low amplitude modulation depth."""
    modulation_depth = amplitude_modulation_depth_metric(audio)
    return min <= modulation_depth and modulation_depth < max


def proportion_clipped_check(audio: Audio, threshold: float = 0.0001) -> bool:
    """Checks if an Audio object has less than 0.01% clipped samples."""
    return proportion_clipped_metric(audio) < threshold


def clipping_present_check(audio: Audio) -> bool:
    """Checks if an Audio object has clipped samples."""
    return proportion_clipped_metric(audio) > 0


def completely_silent_check(audio: Audio, silence_threshold: float = 0.01) -> bool:
    """Checks if an Audio object is completely silent."""
    return proportion_silent_metric(audio, silence_threshold=silence_threshold) < 1.0


def mostly_silent_check(audio: Audio, silence_threshold: float = 0.01, max_silent_proportion: float = 0.95) -> bool:
    """Checks if an Audio object is completely silent."""
    return proportion_silent_metric(audio, silence_threshold=silence_threshold) < max_silent_proportion


def high_amplitude_skew_magnitude_check(audio: Audio, magnitude: float = 5.0) -> bool:
    """Checks whether the absolute amplitude skew is within a specified magnitude.

    Args:
        audio (Audio): The SenseLab Audio object.
        magnitude (float): Maximum acceptable absolute skew.

    Returns:
        bool: True if abs(skew) <= magnitude, False otherwise.
    """
    return abs(amplitude_skew_metric(audio)) <= magnitude


def high_crest_factor_check(audio: Audio, threshold: float = 20.0) -> bool:
    """Checks whether the crest factor of the audio is less than the specified threshold.

    Args:
        audio (Audio): The SenseLab Audio object.
        threshold (float): Maximum acceptable crest factor.

    Returns:
        bool: True if crest factor <= threshold, False otherwise.
    """
    return crest_factor_metric(audio) <= threshold


def low_crest_factor_check(audio: Audio, threshold: float = 1.5) -> bool:
    """Checks whether the crest factor of the audio is greater than the specified minimum threshold.

    Args:
        audio (Audio): The SenseLab Audio object.
        threshold (float): Minimum acceptable crest factor.

    Returns:
        bool: True if crest factor >= threshold, False otherwise.
    """
    return crest_factor_metric(audio) >= threshold
