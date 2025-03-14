"""Contains audio quality metrics used in various checks."""

import torch

from senselab.audio.data_structures import Audio


def proportion_silent_metric(audio: Audio, silence_threshold: float = 0.01) -> float:
    """Calculates the proportion of silent samples.

    Args:
        audio (Audio): The SenseLab Audio object.
        silence_threshold (float): Amplitude below which a sample is silent.

    Returns:
        float: Proportion of silent samples.
    """
    waveform = audio.waveform
    assert waveform.ndim == 2, "Expected waveform shape (num_channels, num_samples)"

    if not audio.normalized:
        audio.normalize()

    silent_samples = (waveform.abs() < silence_threshold).sum().item()
    return silent_samples / waveform.numel()


def proportion_silence_at_beginning_metric(audio: Audio, silence_threshold: float = 0.01) -> float:
    """Calculates the proportion of silence at the start.

    Args:
        audio (Audio): The SenseLab Audio object.
        silence_threshold (float): Amplitude below which a sample is silent.

    Returns:
        float: Proportion of silence at the start.
    """
    waveform = audio.waveform
    assert waveform.ndim == 2, "Expected waveform shape (num_channels, num_samples)"

    if not audio.normalized:
        audio.normalize()

    all_channels_silent = (waveform.abs() < silence_threshold).all(dim=0)
    total_samples = waveform.size(-1)

    non_silent_indices = torch.where(~all_channels_silent)[0]
    if len(non_silent_indices) == 0:
        return 1.0  # Entire audio is silent

    return non_silent_indices[0].item() / total_samples


def proportion_silence_at_end_metric(audio: Audio, silence_threshold: float = 0.01) -> float:
    """Calculates the proportion of silence at the end.

    Args:
        audio (Audio): The SenseLab Audio object.
        silence_threshold (float): Amplitude below which a sample is silent.

    Returns:
        float: Proportion of silence at the end.
    """
    waveform = audio.waveform
    assert waveform.ndim == 2, "Expected waveform shape (num_channels, num_samples)"

    if not audio.normalized:
        audio.normalize()

    all_channels_silent = (waveform.abs() < silence_threshold).all(dim=0)
    total_samples = waveform.size(-1)

    non_silent_indices = torch.where(~all_channels_silent)[0]
    if len(non_silent_indices) == 0:
        return 1.0  # Entire audio is silent

    last_non_silent_idx = non_silent_indices[-1].item()
    return (total_samples - last_non_silent_idx - 1) / total_samples
