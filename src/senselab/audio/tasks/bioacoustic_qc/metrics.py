"""Contains audio quality metrics used in various checks."""

import torch

from senselab.audio.data_structures import Audio


def proportion_silent_metric(audio: Audio, silence_threshold: float = 0.01) -> float:
    """Calculates the proportion of silent samples using normalized amplitude thresholding.

    Args:
        audio (Audio): The SenseLab Audio object.
        silence_threshold (float): The amplitude threshold below which a sample is considered silent.

    Returns:
        float: Proportion of silent samples in the waveform.
    """
    waveform = audio.waveform
    assert waveform.ndim == 2, "Expected waveform to have shape (num_channels, num_samples)"

    # Normalize waveform to [-1, 1] if not already normalized
    if not audio.normalized:
        audio.normalize()

    # Count silent samples (absolute amplitude below threshold)
    silent_samples = (waveform.abs() < silence_threshold).sum().item()
    total_samples = waveform.numel()

    return silent_samples / total_samples
