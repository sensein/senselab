"""Contains audio quality metrics used in various checks."""

import librosa
import numpy as np
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
    """Returns the smaller of positive or negative amplitude headroom.

    Args:
        audio (Audio): The SenseLab Audio object.

    Returns:
        float: Minimum headroom to clipping (positive or negative side).

    Raises:
        ValueError: If amplitude exceeds [-1.0, 1.0].
        TypeError: If the waveform is not of type `torch.float32`.
    """
    if audio.waveform.dtype != torch.float32:
        raise TypeError(f"Expected waveform dtype torch.float32, but got {audio.waveform.dtype}")

    max_amp = audio.waveform.max().item()
    min_amp = audio.waveform.min().item()

    if max_amp > 1.0:
        raise ValueError(f"Audio contains samples over 1.0. Max amplitude = {max_amp:.4f}")
    if min_amp < -1.0:
        raise ValueError(f"Audio contains samples under -1.0. Min amplitude = {min_amp:.4f}")

    pos_headroom = 1.0 - max_amp
    neg_headroom = 1.0 + min_amp

    return min(pos_headroom, neg_headroom)


def spectral_gating_snr(audio: Audio, frame_length: int = 2048, hop_length: int = 512, percentile: int = 10) -> float:
    """Computes segmental SNR using the spectral gating approach.

    Parameters:
        audio (Audio): Audio object containing waveform and metadata.
        frame_length (int): Frame size for STFT.
        hop_length (int): Hop size for moving window.
        percentile (int): Percentage of lowest-energy frequency bins used for noise estimation.

    Returns:
        float: Estimated segmental SNR in dB.
    """
    waveform: np.ndarray | torch.Tensor = audio.waveform
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()

    if waveform.shape[0] > 1:
        waveform = np.mean(waveform, axis=0)

    stft: np.ndarray = np.abs(librosa.stft(waveform, n_fft=frame_length, hop_length=hop_length))
    noise_estimate: np.ndarray = np.percentile(stft, percentile, axis=1)
    snr_per_freq: np.ndarray = 10 * np.log10((np.mean(stft**2, axis=1) + 1e-10) / (noise_estimate**2 + 1e-10))

    return float(np.mean(snr_per_freq))
