"""Contains audio quality metrics used in various checks."""

import librosa
import numpy as np
import scipy
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
    max_amp = audio.waveform.max().item()
    min_amp = audio.waveform.min().item()

    if max_amp > 1.0:
        raise ValueError(f"Audio contains samples over 1.0. Max amplitude = {max_amp:.4f}")
    if min_amp < -1.0:
        raise ValueError(f"Audio contains samples under -1.0. Min amplitude = {min_amp:.4f}")

    pos_headroom = 1.0 - max_amp
    neg_headroom = 1.0 + min_amp

    return min(pos_headroom, neg_headroom)


def spectral_gating_snr_metric(
    audio: Audio, frame_length: int = 2048, hop_length: int = 512, percentile: int = 10
) -> float:
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


def proportion_clipped_metric(audio: Audio, clip_threshold: float = 1.0) -> float:
    """Calculates the proportion of clipped samples.

    Args:
        audio (Audio): The SenseLab Audio object.
        clip_threshold (float): Threshold at or above which a sample is considered clipped.

    Returns:
        float: Proportion of samples that are clipped.
    """
    if not clipping_present_metric(audio):
        return 0.0
    else:
        waveform = audio.waveform
        assert waveform.ndim == 2, "Expected waveform shape (num_channels, num_samples)"
        waveform = waveform.abs()

        max_vals = torch.max(waveform, dim=1).values
        clipped_samples = torch.isclose(waveform, max_vals.unsqueeze(1)).sum(dim=1)
        clipped_proportions = clipped_samples / waveform.shape[1]

        return float(np.mean(clipped_proportions))


def clipping_present_metric(audio: Audio, plateau_length: int = 5) -> bool:
    """Detects clipping by looking for flat plateaus of high-amplitude samples.

    Args:
        audio (Audio): The SenseLab Audio object.
        plateau_length (float): Length of maximum-valued samples that determine clipping status.

    Returns:
        bool: True if clipping is present, False otherwise.
    """
    waveform = audio.waveform
    assert waveform.ndim == 2, "Expected waveform shape (num_channels, num_samples)"
    waveform = waveform.abs()
    if (waveform >= 1.0).any().item():
        return True

    for channel in waveform:
        count = 0
        max_val = torch.max(channel)
        for val in channel:
            if torch.isclose(val, max_val):
                count += 1
            else:
                count = 0
            if count >= plateau_length:
                return True
    return False


def amplitude_modulation_depth_metric(audio: Audio) -> float:
    """Calculates the amplitude modulation depth of an audio signal.

    Args:
        audio (Audio): The SenseLab Audio object.

    Returns:
        float: Amplitude modulation depth.
    """
    waveform = audio.waveform
    assert waveform.ndim == 2, "Expected waveform shape (num_channels, num_samples)"

    # Convert to numpy array if it's a torch tensor
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()

    # Compute the analytic signal
    analytic_signal = scipy.signal.hilbert(waveform, axis=-1)

    # Compute the amplitude envelope
    amplitude_envelope = np.abs(analytic_signal)

    # Calculate modulation depth
    max_env = np.max(amplitude_envelope, axis=-1)
    min_env = np.min(amplitude_envelope, axis=-1)
    modulation_depth = (max_env - min_env) / (max_env + min_env + 1e-10)  # Adding epsilon to avoid division by zero

    # Return the mean modulation depth across channels
    return float(np.mean(modulation_depth))


def root_mean_square_energy_metric(audio: Audio) -> float:
    """Calculates the root mean square (RMS) energy of the audio signal.

    Args:
        audio (Audio): The SenseLab Audio object.

    Returns:
        float: RMS energy averaged across channels.
    """
    waveform = audio.waveform
    assert waveform.ndim == 2, "Expected waveform shape (num_channels, num_samples)"

    rms_per_channel = torch.sqrt(torch.mean(waveform**2, dim=1))
    return float(torch.mean(rms_per_channel))


def zero_crossing_rate_metric(audio: Audio) -> float:
    """Estimates the zero-crossing rate of the audio signal.

    Args:
        audio (Audio): The SenseLab Audio object.

    Returns:
        float: Average zero-crossing rate across channels.
    """
    waveform = audio.waveform
    assert waveform.ndim == 2, "Expected waveform shape (num_channels, num_samples)"

    # Compute sign of samples: +1 for positive, -1 for negative, 0 for zero
    signs = torch.sign(waveform)

    # Zero-crossings occur where the sign changes
    crossings = (signs[:, 1:] * signs[:, :-1]) < 0  # shape: (channels, samples - 1)

    # Mean ZCR per channel, then average
    zcr_per_channel = crossings.float().mean(dim=1)
    return float(zcr_per_channel.mean())


def signal_variance_metric(audio: Audio) -> float:
    """Estimates the variance of the audio signal.

    Args:
        audio (Audio): The SenseLab Audio object.

    Returns:
        float: Variance across all samples and channels.
    """
    waveform = audio.waveform
    assert waveform.ndim == 2, "Expected waveform shape (num_channels, num_samples)"
    return float(torch.var(waveform))


def dynamic_range_metric(audio: Audio) -> float:
    """Calculates the dynamic range of the audio signal.

    Dynamic range is defined as the difference between the maximum and minimum
    amplitude values in the signal.

    Args:
        audio (Audio): The SenseLab Audio object.

    Returns:
        float: The dynamic range (max amplitude minus min amplitude).
    """
    waveform = audio.waveform
    assert waveform.ndim == 2, "Expected waveform shape (num_channels, num_samples)"

    # Compute the overall dynamic range across all channels
    max_amp = waveform.max().item()
    min_amp = waveform.min().item()
    return float(max_amp - min_amp)


def mean_absolute_amplitude_metric(audio: Audio) -> float:
    """Calculates the mean absolute amplitude of the audio signal.

    Args:
        audio (Audio): The SenseLab Audio object.

    Returns:
        float: Mean absolute amplitude averaged across channels.
    """
    waveform = audio.waveform
    assert waveform.ndim == 2, "Expected waveform shape (num_channels, num_samples)"

    # Compute the mean absolute amplitude across all samples and channels
    mean_abs = torch.mean(waveform.abs())
    return float(mean_abs)


def mean_absolute_deviation_metric(audio: Audio) -> float:
    """Calculates the mean absolute deviation (MAD) of the audio signal.

    Args:
        audio (Audio): The SenseLab Audio object.

    Returns:
        float: MAD averaged across channels.
    """
    waveform = audio.waveform
    assert waveform.ndim == 2, "Expected waveform shape (num_channels, num_samples)"

    mean_val = torch.mean(waveform, dim=1, keepdim=True)
    mad = torch.mean(torch.abs(waveform - mean_val), dim=1)
    return float(torch.mean(mad))


def shannon_entropy_amplitude_metric(audio: "Audio", num_bins: int = 256) -> float:
    """Calculates the Shannon entropy of the audio signal's amplitude distribution.

    Args:
        audio (Audio): The SenseLab Audio object.
        num_bins (int): Number of bins to discretize the amplitude values.

    Returns:
        float: Shannon entropy.
    """
    waveform = audio.waveform.flatten().numpy()

    # Get histogram counts
    hist, _ = np.histogram(waveform, bins=num_bins)

    # Normalize to get a probability distribution
    prob = hist / np.sum(hist)

    # Remove zero entries to avoid log2(0)
    prob = prob[prob > 0]

    # Compute entropy
    entropy = -np.sum(prob * np.log2(prob))
    return float(entropy)


def crest_factor_metric(audio: Audio) -> float:
    """Calculates the crest factor (peak‑to‑RMS ratio) of the audio signal.

    Args:
        audio (Audio): The SenseLab Audio object.

    Returns:
        float: Crest factor (unitless).
    """
    waveform = audio.waveform
    assert waveform.ndim == 2, "Expected waveform shape (num_channels, num_samples)"

    # Peak absolute amplitude across all channels
    peak = waveform.abs().max().item()

    # RMS across all samples and channels
    rms = torch.sqrt(torch.mean(waveform**2)).item()
    if rms == 0:
        return float("inf")  # silent signal → infinite crest factor

    return peak / rms


def peak_snr_from_spectral_metric(
    audio: Audio, frame_length: int = 2048, hop_length: int = 512, percentile: int = 10
) -> float:
    """Estimates Peak‑SNR (dB) using spectral gating to estimate the noise floor.

    Args:
        audio (Audio): The SenseLab Audio object.
        frame_length (int): STFT window size.
        hop_length (int): STFT hop size.
        percentile (int): Percentile used for noise floor estimation.

    Returns:
        float: Peak‑SNR in decibels.
    """
    waveform = audio.waveform
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()

    # Collapse to mono if multi-channel
    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = np.mean(waveform, axis=0)

    # Get magnitude spectrogram
    stft_mag = np.abs(librosa.stft(waveform, n_fft=frame_length, hop_length=hop_length))

    # Estimate noise floor from quietest bins
    noise_floor = np.percentile(stft_mag, percentile, axis=1)
    noise_rms = np.sqrt(np.mean(noise_floor**2) + 1e-10)

    # Get peak signal amplitude from time-domain
    peak = np.max(np.abs(waveform))

    return 20 * np.log10(peak / noise_rms)


def amplitude_skew_metric(audio: Audio) -> float:
    """Calculates the skew of the audio signal amplitude.

    Args:
        audio (Audio): The SenseLab Audio object.

    Returns:
        float: Skew of the flattened amplitude distribution.
    """
    waveform = audio.waveform
    assert waveform.ndim == 2, "Expected waveform shape (num_channels, num_samples)"
    return float(scipy.stats.skew(waveform.flatten().numpy()))


def amplitude_kurtosis_metric(audio: Audio) -> float:
    """Calculates the kurtosis of the audio signal amplitude.

    Args:
        audio (Audio): The SenseLab Audio object.

    Returns:
        float: Kurtosis of the flattened amplitude distribution.
    """
    waveform = audio.waveform
    assert waveform.ndim == 2, "Expected waveform shape (num_channels, num_samples)"
    return float(scipy.stats.kurtosis(waveform.flatten().numpy()))


def amplitude_interquartile_range_metric(audio: Audio) -> float:
    """Calculates the interquartile range (IQR) of the audio signal amplitude.

    Args:
        audio (Audio): The SenseLab Audio object.

    Returns:
        float: IQR of the flattened amplitude distribution.
    """
    waveform = audio.waveform
    assert waveform.ndim == 2, "Expected waveform shape (num_channels, num_samples)"
    return float(scipy.stats.iqr(waveform.flatten().numpy()))


def phase_correlation_metric(audio: Audio, frame_length: int = 2048, hop_length: int = 512) -> float:
    """Calculates the phase correlation between stereo channels.

    This metric measures the coherence between left and right channels in stereo audio.
    Values close to 1.0 indicate strong positive correlation (in-phase),
    values close to -1.0 indicate strong negative correlation (out-of-phase),
    and values near 0.0 indicate uncorrelated channels.

    Args:
        audio (Audio): The SenseLab Audio object.
        frame_length (int): Frame size for analysis.
        hop_length (int): Hop size for moving window.

    Returns:
        float: Average phase correlation coefficient between channels.

    Raises:
        ValueError: If the audio is not stereo (doesn't have exactly 2 channels).
    """
    waveform = audio.waveform

    # Check if the audio is stereo
    if waveform.ndim != 2 or waveform.shape[0] != 2:
        raise ValueError(f"Expected stereo audio (2 channels), but got {waveform.shape[0]} channels")

    # Convert to numpy if it's a torch tensor
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()

    left_channel = waveform[0]
    right_channel = waveform[1]

    # Calculate the correlation coefficient for each frame and average
    num_frames = 1 + (left_channel.shape[0] - frame_length) // hop_length
    correlation_values = []

    for i in range(num_frames):
        start_idx = i * hop_length
        end_idx = start_idx + frame_length

        if end_idx > left_channel.shape[0]:
            end_idx = left_channel.shape[0]

        left_frame = left_channel[start_idx:end_idx]
        right_frame = right_channel[start_idx:end_idx]

        # Calculate Pearson correlation coefficient, handling edge cases
        if np.std(left_frame) == 0 or np.std(right_frame) == 0:
            corr = 0
        else:
            corr = np.corrcoef(left_frame, right_frame)[0, 1]

            # Handle NaN values (can happen with very low amplitude signals)
            if np.isnan(corr):
                corr = 0

        correlation_values.append(corr)
    mean_correlation = np.mean(correlation_values)
    return float(mean_correlation)
