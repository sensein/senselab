"""Contains audio quality metrics used in various checks."""

import torch
import numpy as np
import librosa

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

    all_channels_silent = (waveform.abs() < silence_threshold).all(dim=0)
    total_samples = waveform.size(-1)

    non_silent_indices = torch.where(~all_channels_silent)[0]
    if len(non_silent_indices) == 0:
        return 1.0  # Entire audio is silent

    last_non_silent_idx = non_silent_indices[-1].item()
    return (total_samples - last_non_silent_idx - 1) / total_samples


def amplitude_headroom_metric(audio: Audio) -> float:
    """Calculates the amplitude headroom.

    Amplitude headroom is the difference between the highest amplitude sample
    and the clipping threshold (1.0). If max amplitude > 1.0, an error is raised.

    Args:
        audio (Audio): The SenseLab Audio object.

    Returns:
        float: The amplitude headroom, i.e., `1.0 - max_amplitude`.

    Raises:
        ValueError: If the waveform contains values greater than 1.0.
        TypeError: If the waveform is not of type `torch.float32`.
    """
    if audio.waveform.dtype != torch.float32:
        raise TypeError(f"Expected waveform dtype torch.float32, but got {audio.waveform.dtype}")

    max_amplitude = audio.waveform.max().item()

    if max_amplitude > 1.0:
        raise ValueError(f"Audio contains samples over 1.0. Max amplitude = {max_amplitude:.4f}")

    return 1.0 - max_amplitude


def amplitude_toeroom_metric(audio: Audio) -> float:
    """Calculates the amplitude toeroom.

    Amplitude toeroom represents how far the lowest sample is from -1.0.
    It is calculated as `-1.0 - min_amplitude`, where `min_amplitude` is
    the smallest sample value.

    Args:
        audio (Audio): The SenseLab Audio object.

    Returns:
        float: The amplitude toeroom, i.e., `-1.0 - min_amplitude`.

    Raises:
        TypeError: If the waveform is not of type `torch.float32`.
        ValueError: If the minimum amplitude is less than -1.0.
    """
    if audio.waveform.dtype != torch.float32:
        raise TypeError(f"Expected waveform dtype torch.float32, but got {audio.waveform.dtype}")

    min_amplitude = audio.waveform.min().item()

    if min_amplitude < -1.0:
        raise ValueError(f"Audio contains samples under -1.0. Min amplitude = {min_amplitude:.4f}")

    return -1.0 - min_amplitude


def spectral_gating_snr(audio: Audio, frame_length=2048, hop_length=512, percentile=10) -> float:
    """
    Computes segmental SNR using the spectral gating approach.

    Parameters:
        audio (Audio): Audio object containing waveform and metadata.
        frame_length (int): Frame size for STFT.
        hop_length (int): Hop size for moving window.
        percentile (int): Percentage of lowest-energy frequency bins used for noise estimation.

    Returns:
        float: Estimated segmental SNR in dB.
    """

    waveform = audio.waveform  # Shape: (num_channels, num_samples)
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()

    # Convert stereo/multi-channel to mono by averaging channels
    if waveform.shape[0] > 1:
        waveform = np.mean(waveform, axis=0)

    # Compute STFT (Short-Time Fourier Transform)
    stft = np.abs(librosa.stft(waveform, n_fft=frame_length, hop_length=hop_length))

    # Estimate noise by taking the lowest percentile of STFT energy per frequency bin
    noise_estimate = np.percentile(stft, percentile, axis=1)

    # Compute SNR per frequency band
    snr_per_freq = 10 * np.log10((np.mean(stft**2, axis=1) + 1e-10) / (noise_estimate**2 + 1e-10))

    # Return average SNR over all frequency bands
    return np.mean(snr_per_freq)
