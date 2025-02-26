"""All checks take only an audio as an argument."""

import torch

from senselab.audio.data_structures import Audio


def audio_length_positive_check(audio: Audio) -> bool:
    """Checks if an Audio object has a positive length.

    Args:
        audio (Audio): The Audio object to validate.

    Returns:
        bool: True if the audio has samples, False otherwise.

    Raises:
        ValueError: If the waveform is empty or not a 2D tensor.
    """
    if not isinstance(audio.waveform, torch.Tensor) or audio.waveform.ndim != 2:
        raise ValueError("Waveform must be a 2D torch.Tensor with shape (num_channels, num_samples).")

    return audio.waveform.shape[1] > 0


def audio_intensity_positive_check(audio: Audio) -> bool:
    """Checks if an Audio object has nonzero intensity.

    Args:
        audio (Audio): The Audio object to validate.

    Returns:
        bool: True if intensity is greater than zero, False otherwise.

    Raises:
        ValueError: If the waveform is empty or not a 2D tensor.
    """
    if not isinstance(audio.waveform, torch.Tensor) or audio.waveform.ndim != 2:
        raise ValueError("Waveform must be a 2D torch.Tensor with shape (num_channels, num_samples).")

    return torch.mean(torch.abs(audio.waveform)) > 0
