"""Contains audio quality metrics used in various checks."""

from typing import Dict, List, Optional

import librosa
import numpy as np
import scipy
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speaker_diarization.api import diarize_audios
from senselab.audio.tasks.voice_activity_detection.api import (
    detect_human_voice_activity_in_audios,
)
from senselab.utils.data_structures import ScriptLine
from senselab.utils.data_structures.logging import logger


def proportion_silent_metric(audio: Audio, silence_threshold: float = 0.01) -> float:
    """Calculates the proportion of silent samples.

    Args:
        audio (Audio): The senselab Audio object.
        silence_threshold (float): Amplitude below which a sample is silent.

    Returns:
        float: Proportion of silent samples, or 0.0 if waveform has zero elements.
    """
    waveform = audio.waveform
    if torch.numel(waveform) == 0:
        return 1.0

    silent_samples = (waveform.abs() < silence_threshold).sum().item()
    return silent_samples / waveform.numel()


def proportion_silence_at_beginning_metric(audio: Audio, silence_threshold: float = 0.01) -> float:
    """Calculates the proportion of silence at the start.

    Args:
        audio (Audio): The senselab Audio object.
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
        audio (Audio): The senselab Audio object.
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
        audio (Audio): The senselab Audio object.

    Returns:
        float: Minimum headroom to clipping (positive or negative side).

    Raises:
        ValueError: If amplitude exceeds [-1.0, 1.0].
        TypeError: If the waveform is not of type `torch.float32`.
    """
    if torch.numel(audio.waveform) == 0:
        return 1.0

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
        audio (Audio): The senselab Audio object.
        clip_threshold (float): Threshold at or above which a sample is considered clipped.

    Returns:
        float: Proportion of samples that are clipped.
    """
    waveform = audio.waveform

    if torch.numel(waveform) == 0:
        return 0.0

    assert waveform.ndim == 2, "Expected waveform shape (num_channels, num_samples)"
    waveform = waveform.abs()

    clipped_proportion_by_channel = []

    def is_likely_clipped(channel: torch.Tensor, min_consecutive: int = 3) -> bool:
        """Returns True if there are consecutive samples at the max value, suggesting clipping.

        A plateau pattern (consecutive samples at max) is a strong indicator of clipping,
        even if the max value is below the threshold.

        Args:
            channel: 1D audio tensor.
            min_consecutive: Minimum number of consecutive samples at max to indicate clipping.
                           Default 2 means at least two consecutive samples must be at max.

        Returns:
            bool: True if likely clipped (plateau pattern detected).
        """
        if channel.numel() == 0:
            return False

        max_val = channel.max()
        close_to_max = torch.isclose(channel, max_val)

        # Check for consecutive samples at max value
        # Convert boolean tensor to list for easier consecutive checking
        close_to_max_list = close_to_max.tolist()

        # Find the longest consecutive sequence of True values
        max_consecutive = 0
        current_consecutive = 0

        for is_close in close_to_max_list:
            if is_close:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive > min_consecutive

    for channel in waveform:
        clipped_samples = 0
        max_val = torch.max(channel)
        # Check for clipping: either at/above threshold, or plateau pattern near threshold
        if max_val >= clip_threshold:
            # Samples at or above threshold are definitely clipped
            clipped_samples = (channel >= clip_threshold).sum().item()
        elif is_likely_clipped(channel):
            # Plateau pattern suggests clipping even if below threshold
            # Count samples close to max as potentially clipped
            clipped_samples = torch.isclose(channel, max_val).sum().item()
        clipped_proportion_by_channel.append(clipped_samples / channel.numel())

    return float(np.mean(clipped_proportion_by_channel))


def amplitude_modulation_depth_metric(audio: Audio) -> float:
    """Calculates the amplitude modulation depth of an audio signal.

    Args:
        audio (Audio): The senselab Audio object.

    Returns:
        float: Amplitude modulation depth.
    """
    waveform = audio.waveform
    if torch.numel(waveform) == 0:
        return np.nan
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
        audio (Audio): The senselab Audio object.

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
        audio (Audio): The senselab Audio object.

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
        audio (Audio): The senselab Audio object.

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
        audio (Audio): The senselab Audio object.

    Returns:
        float: The dynamic range (max amplitude minus min amplitude).
    """
    waveform = audio.waveform
    if torch.numel(waveform) == 0:
        return 0.0
    assert waveform.ndim == 2, "Expected waveform shape (num_channels, num_samples)"

    # Compute the overall dynamic range across all channels
    max_amp = waveform.max().item()
    min_amp = waveform.min().item()
    return float(max_amp - min_amp)


def mean_absolute_amplitude_metric(audio: Audio) -> float:
    """Calculates the mean absolute amplitude of the audio signal.

    Args:
        audio (Audio): The senselab Audio object.

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
        audio (Audio): The senselab Audio object.

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
        audio (Audio): The senselab Audio object.
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
        audio (Audio): The senselab Audio object.

    Returns:
        float: Crest factor (unitless).
    """
    waveform = audio.waveform
    if torch.numel(waveform) == 0:
        return np.nan
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
        audio (Audio): The senselab Audio object.
        frame_length (int): STFT window size.
        hop_length (int): STFT hop size.
        percentile (int): Percentile used for noise floor estimation.

    Returns:
        float: Peak‑SNR in decibels.
    """
    waveform = audio.waveform
    if torch.numel(waveform) == 0:
        return np.nan
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

    return float(20 * np.log10(peak / noise_rms))


def amplitude_skew_metric(audio: Audio) -> float:
    """Calculates the skew of the audio signal amplitude.

    Args:
        audio (Audio): The senselab Audio object.

    Returns:
        float: Skew of the flattened amplitude distribution.
    """
    waveform = audio.waveform
    assert waveform.ndim == 2, "Expected waveform shape (num_channels, num_samples)"
    return float(scipy.stats.skew(waveform.flatten().numpy()))


def amplitude_kurtosis_metric(audio: Audio) -> float:
    """Calculates the kurtosis of the audio signal amplitude.

    Args:
        audio (Audio): The senselab Audio object.

    Returns:
        float: Kurtosis of the flattened amplitude distribution.
    """
    waveform = audio.waveform
    assert waveform.ndim == 2, "Expected waveform shape (num_channels, num_samples)"
    return float(scipy.stats.kurtosis(waveform.flatten().numpy()))


def amplitude_interquartile_range_metric(audio: Audio) -> float:
    """Calculates the interquartile range (IQR) of the audio signal amplitude.

    Args:
        audio (Audio): The senselab Audio object.

    Returns:
        float: IQR of the flattened amplitude distribution.
    """
    waveform = audio.waveform
    assert waveform.ndim == 2, "Expected waveform shape (num_channels, num_samples)"
    return float(scipy.stats.iqr(waveform.flatten().numpy()))


def phase_correlation_metric(audio: Audio, frame_length: int = 2048, hop_length: int = 512) -> float:
    """Computes average inter-channel correlation for stereo or multi-channel audio."""
    waveform = audio.waveform

    if waveform.ndim != 2:
        return 1.0  # Treat 1D as mono

    num_channels, num_samples = waveform.shape
    if num_channels < 2:
        return 1.0  # Mono

    if num_samples < frame_length:
        frame_length = num_samples
        hop_length = max(1, num_samples // 2)

    correlation_values = []

    for i in range(0, num_samples - frame_length + 1, hop_length):
        frame = waveform[:, i : i + frame_length]
        if frame.shape[1] < 2:
            continue

        corr_matrix = np.corrcoef(frame.numpy())
        upper_triangle = corr_matrix[np.triu_indices(num_channels, k=1)]

        # Exclude NaNs (e.g. from silent channels)
        valid_corrs = upper_triangle[~np.isnan(upper_triangle)]
        if valid_corrs.size > 0:
            correlation_values.append(np.mean(valid_corrs))

    return float(np.mean(correlation_values)) if correlation_values else 0.0


def primary_speaker_ratio_metric(audio: Audio) -> float:
    """Calculates the ratio of the primary speaker's duration to the total duration.

    The primary speaker ratio is computed from speaker diarization results.
    If diarization is not available in audio.metadata, it will be computed automatically.

    Args:
        audio: The senselab Audio object.

    Returns:
        float: Ratio of primary speaker's duration to total duration.
                If there are no speakers, return nan.
                If only one speaker, return 1.0.
                Else return a number between 0.0 and 1.0.
    """
    # Check Audio metadata for precomputed diarization
    diarization_result: Optional[List[ScriptLine]] = None
    if audio.metadata:
        metadata_diarization = audio.metadata.get("diarization")
        if metadata_diarization is not None:
            # Handle different storage formats: List[List[ScriptLine]] or List[ScriptLine]
            if isinstance(metadata_diarization, list):
                if len(metadata_diarization) == 0:
                    # Empty list means diarization was computed but found no speakers
                    diarization_result = []
                elif isinstance(metadata_diarization[0], list):
                    # Format: List[List[ScriptLine]] - take first audio's diarization
                    diarization_result = metadata_diarization[0]
                elif isinstance(metadata_diarization[0], ScriptLine):
                    # Format: List[ScriptLine]
                    diarization_result = metadata_diarization
            elif isinstance(metadata_diarization, list) and len(metadata_diarization) == 0:
                diarization_result = metadata_diarization

    # Compute diarization if not in metadata
    if diarization_result is None:
        try:
            diarization_results = diarize_audios([audio])
            if not diarization_results or len(diarization_results) == 0:
                return np.nan
            diarization_result = diarization_results[0]
        except Exception as e:
            logger.warning(f"Failed to compute diarization for primary_speaker_ratio_metric: {e}")
            return np.nan

    # Calculate primary speaker ratio from ScriptLine objects
    if not diarization_result or len(diarization_result) == 0:
        return np.nan

    # Calculate duration per speaker
    speaker_durations: Dict[str, float] = {}
    total_duration = 0.0

    for script_line in diarization_result:
        if script_line.speaker is None or script_line.start is None or script_line.end is None:
            continue
        duration = script_line.end - script_line.start
        speaker = script_line.speaker
        speaker_durations[speaker] = speaker_durations.get(speaker, 0.0) + duration
        total_duration += duration

    if total_duration == 0.0 or len(speaker_durations) == 0:
        return np.nan

    # Find primary speaker (speaker with maximum duration)
    primary_speaker_duration = max(speaker_durations.values())
    primary_speaker_ratio = primary_speaker_duration / total_duration

    return float(primary_speaker_ratio)


def voice_activity_detection_metric(audio: Audio) -> float:
    """Calculates the duration of voice activity detected by VAD.

    The voice activity duration is computed from VAD results.
    If VAD is not available in audio.metadata, it will fall back to
    diarization results (treating all speaker segments as voice activity).
    If neither is available, VAD will be computed automatically.

    Args:
        audio: The senselab Audio object.

    Returns:
        float: Duration of voice activity in seconds.
               Returns 0.0 if no voice is detected.
               Returns np.nan if VAD computation fails.
    """
    # Step 1: Check Audio metadata for precomputed VAD
    vad_result: Optional[List[ScriptLine]] = None
    if audio.metadata:
        metadata_vad = audio.metadata.get("vad")
        if metadata_vad is not None:
            # Handle empty list case explicitly
            if isinstance(metadata_vad, list) and len(metadata_vad) == 0:
                vad_result = []
            # Handle different storage formats:
            # List[List[ScriptLine]] or List[ScriptLine]
            elif isinstance(metadata_vad, list) and len(metadata_vad) > 0:
                if isinstance(metadata_vad[0], list):
                    # Format: List[List[ScriptLine]] - take first audio's VAD
                    vad_result = metadata_vad[0]
                elif isinstance(metadata_vad[0], ScriptLine):
                    # Format: List[ScriptLine]
                    vad_result = metadata_vad

    # Step 2: Check for diarization in metadata if VAD not available (do not compute diarization)
    if vad_result is None:
        if audio.metadata:
            metadata_diarization = audio.metadata.get("diarization")
            if metadata_diarization is not None:
                # Handle different storage formats: List[List[ScriptLine]] or List[ScriptLine]
                diarization_result: Optional[List[ScriptLine]] = None
                if isinstance(metadata_diarization, list) and len(metadata_diarization) > 0:
                    if isinstance(metadata_diarization[0], list):
                        # Format: List[List[ScriptLine]] - take first audio's diarization
                        diarization_result = metadata_diarization[0]
                    elif isinstance(metadata_diarization[0], ScriptLine):
                        # Format: List[ScriptLine]
                        diarization_result = metadata_diarization

                # Convert diarization to VAD-like format (all segments with speakers are voice)
                if diarization_result:
                    vad_result = [
                        ScriptLine(speaker="VOICE", start=sl.start, end=sl.end)
                        for sl in diarization_result
                        if sl.speaker is not None and sl.start is not None and sl.end is not None
                    ]

    # Step 3: Compute VAD if neither VAD nor diarization found in metadata
    if vad_result is None:
        try:
            vad_results = detect_human_voice_activity_in_audios([audio])
            if not vad_results or len(vad_results) == 0:
                return np.nan
            vad_result = vad_results[0]
            # Store computed VAD in metadata for reuse
            if audio.metadata is None:
                audio.metadata = {}
            audio.metadata["vad"] = vad_result
        except Exception as e:
            logger.warning(f"Failed to compute VAD for voice_activity_detection_metric: {e}")
            return np.nan

    # Calculate total voice duration from ScriptLine objects
    if not vad_result or len(vad_result) == 0:
        return 0.0

    total_voice_duration = 0.0

    for script_line in vad_result:
        # VAD results have speaker="VOICE" for voice segments
        if script_line.speaker == "VOICE" and script_line.start is not None and script_line.end is not None:
            duration = script_line.end - script_line.start
            total_voice_duration += duration

    return float(total_voice_duration)


def voice_signal_to_noise_power_ratio_metric(audio: Audio) -> float:
    """Calculates the SNR by looking at power during voice versus power when none.

    The signal-to-noise ratio is computed from VAD results, where voice segments
    are considered signal and non-voice segments are considered noise.
    This function calls voice_activity_detection_metric to ensure VAD is available
    in metadata, then uses that VAD result for SNR calculation.

    Args:
        audio: The senselab Audio object.

    Returns:
        float: Signal-to-noise power ratio in dB.
               Returns np.nan if VAD computation fails or if there's no voice/noise.
    """
    # Call voice_activity_detection_metric to ensure VAD is computed and stored in metadata
    voice_duration = voice_activity_detection_metric(audio)
    if np.isnan(voice_duration):
        return np.nan

    # Get VAD result from metadata (now guaranteed to be available)
    vad_result: Optional[List[ScriptLine]] = None
    if audio.metadata:
        metadata_vad = audio.metadata.get("vad")
        if metadata_vad is not None:
            # Handle different storage formats: List[List[ScriptLine]] or List[ScriptLine]
            if isinstance(metadata_vad, list) and len(metadata_vad) == 0:
                vad_result = []
            elif isinstance(metadata_vad, list) and len(metadata_vad) > 0:
                if isinstance(metadata_vad[0], list):
                    # Format: List[List[ScriptLine]] - take first audio's VAD
                    vad_result = metadata_vad[0]
                elif isinstance(metadata_vad[0], ScriptLine):
                    # Format: List[ScriptLine]
                    vad_result = metadata_vad

    # Safety check: VAD should be available after calling voice_activity_detection_metric
    if vad_result is None:
        return np.nan

    # Calculate SNR from ScriptLine objects
    if len(vad_result) == 0:
        return np.nan

    waveform = audio.waveform
    time = np.divide(np.arange(waveform.shape[1]), audio.sampling_rate)

    # Collect voice (signal) and non-voice (noise) sample indices
    voice_indices: List[int] = []
    noise_indices: List[int] = []
    previous_end = 0.0

    for script_line in vad_result:
        if script_line.start is None or script_line.end is None:
            continue

        # Voice segments (signal)
        if script_line.speaker == "VOICE":
            voice_mask = (time >= script_line.start) & (time <= script_line.end)
            voice_indices.extend(np.where(voice_mask)[0].tolist())

        # Noise segments (between voice segments)
        if script_line.start > previous_end:
            noise_mask = (time >= previous_end) & (time < script_line.start)
            noise_indices.extend(np.where(noise_mask)[0].tolist())

        previous_end = max(previous_end, script_line.end)

    # Handle remaining noise after last voice segment
    if previous_end < time[-1]:
        noise_mask = time > previous_end
        noise_indices.extend(np.where(noise_mask)[0].tolist())

    # Calculate power for voice and noise segments
    try:
        if len(voice_indices) == 0 or len(noise_indices) == 0:
            return np.nan

        waveform_np = waveform.squeeze().numpy()
        voice_samples = waveform_np[voice_indices]
        noise_samples = waveform_np[noise_indices]

        signal_power = np.mean(voice_samples**2)
        noise_power = np.mean(noise_samples**2)

        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = np.nan
    except (ValueError, IndexError, ZeroDivisionError) as e:
        logger.warning(f"Error calculating SNR: {e}")
        return np.nan

    return float(snr)
