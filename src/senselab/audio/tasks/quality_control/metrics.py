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
        audio (Audio): The SenseLab Audio object.
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

    def is_likely_clipped(channel: torch.Tensor, min_proportion: float = 0.0001) -> bool:
        """Returns True if a significant proportion of samples are close to the max value, suggesting clipping.

        Args:
            channel: 1D audio tensor.
            min_proportion: Minimum proportion of samples that must be close to max to indicate clipping.

        Returns:
            bool: True if likely clipped.
        """
        if channel.numel() == 0:
            return False

        max_val = channel.max()
        close_to_max = torch.isclose(channel, max_val)
        proportion = close_to_max.sum().item() / channel.numel()
        return proportion >= min_proportion

    for channel in waveform:
        clipped_samples = 0
        max_val = torch.max(channel)
        if torch.isclose(max_val, torch.tensor(1.0)) or is_likely_clipped(channel):
            clipped_samples = torch.isclose(channel, max_val).sum().item()
        clipped_proportion_by_channel.append(clipped_samples / channel.numel())

    return float(np.mean(clipped_proportion_by_channel))


def amplitude_modulation_depth_metric(audio: Audio) -> float:
    """Calculates the amplitude modulation depth of an audio signal.

    Args:
        audio (Audio): The SenseLab Audio object.

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
        audio (Audio): The SenseLab Audio object.
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


#### Rahul's additions below


def percent_clipping_metric(audio: Audio) -> float:
    waveform = audio.waveform

    # Determine the maximum possible amplitude based on the data type
    if waveform.dtype == torch.int16:
        max_amplitude = np.iinfo(np.int16).max
    elif waveform.dtype == torch.float32:
        max_amplitude = np.finfo(np.float32).max
    else:
        raise ValueError("Unsupported audio data type. Only int16 and float32 are supported.")

    # alternate AI generated, this doesn't make sense to met tho
    # audio = audio / np.max(np.abs(audio))
    # clipped_samples = np.sum(np.abs(audio) >= max_amplitude)

    # Count clipped samples
    clipped_samples = torch.sum(torch.abs(waveform) >= max_amplitude)

    # Calculate clipping percentage
    total_samples = waveform.shape[1]
    clipping_percentage = float((clipped_samples / total_samples) * 100 if total_samples > 0 else 0.0)

    # return y_axis_new, y_axis_old
    return clipping_percentage


def primary_speaker_ratio_metric(audio: Audio, **input_source) -> float:
    """Calculates the ratio of the primary speaker's duration to the total duration.

    Args:
        audio (Audio): The SenseLab Audio object.

    Returns:
        float: Ratio of primary speaker's duration to total duration.
                If there are no speakers, return nan
                If only one speaker, return 1.0
                Else return a number between 0.0 and 1.0
    """

    if input_source['precompute'] is None:
        diar #= #TODO diarize the audio
    else:
        diar = input_source['precompute']
        #TODO diar = pd.read_pickle("../../modeling/diarize/diar_r2.pkl")
    
    # Calculate primary speaker ratio;
    try:
        ratio = diar_obj.label_duration(diar_obj.argmax()) / np.sum([c[1] for c in diar_obj.chart()])
    except:
        ratio = np.nan

    return ratio


def presence_of_voice_metric(audio: Audio, **input_source) -> float:
    """Calculates the number of samples where Voice Activity Detection detects voice.

    Args:
        audio (Audio): The SenseLab Audio object.

    Returns:
        float: the duration of voice activity in seconds.
    """

    if input_source['precompute'] is None:
        vad #= #TODO VAD the audio
    else:
        vad = input_source['precompute']
        #TODO diar = pd.read_pickle("../../modeling/diarize/diar_r2.pkl")

    return vad




def signal_to_noise_power_ratio_metric(audio: Audio, **input_source) -> float:

    """Calculates the SNR by looking at power during voice versus power when none.

    Args:
        audio (Audio): The SenseLab Audio object.

    Returns:
        float: Ratio of signal to noise power
                Commented-out return is percent of audio that is noise as a test
    """
    
    
    waveform = audio.waveform
    
    time = np.divide(np.arange(waveform.shape[1]), audio.sampling_rate)

    if input_source['precompute'] is None:
        vad #= #TODO VAD the audio
    else:
        vad = input_source['precompute']
        #TODO diar = pd.read_pickle("../../modeling/diarize/diar_r2.pkl")

    signal = []
    noise = []
    previous_end = 0

    for seg in vad.get_timeline().segments_list_:
        #get the signal segments with VAD time stamps
        #get the noise segments with the time stamps between VAD segments

        signal.append(np.where((time >= seg.start) & (time <= seg.end))[0])
        noise.append(np.where((time < seg.start) & (time >= previous_end))[0])
        previous_end = seg.end

    try:
        signal_wav = waveform.squeeze().numpy()[np.concatenate(signal)]
        noise_wav = waveform.squeeze().numpy()[np.concatenate(noise)]

        signal_power = np.mean(signal_wav**2)
        noise_power = np.mean(noise_wav**2)
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else -30

        percent_noise_of_total = np.divide(noise_wav.shape[0], noise_wav.shape[0] + signal_wav.shape[0])
    except:
        snr = -25
        percent_noise_of_total = np.nan

    return snr #percent_noise_of_total


def find_buzzing_metric(audio: Audio, **input_source) -> float:

    """Calculates buzzing from audio_aes Production Quality metric.

    Args:
        audio (Audio): The SenseLab Audio object.

    Returns:
        float: Ratio of signal to noise power
                Commented-out return is percent of audio that is noise as a test
    """

    def load_json_lines(file_path):
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    json_object = json.loads(line)
                    data.append(json_object)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line.strip()}")
        return data


    ###Data loading

    ##load participant record id and session id
    #subset = 'all'

    #if subset == 'mdd':
    #    diagnosis_df = pd.read_csv('diagnosis_df.csv', index_col=0) #gives us participants/sessions we are analyzing and metadata. this csv points to messy data/recordings that should be removed
    #elif subset == 'all':
    #    diagnosis_df = pd.read_csv(os.path.join(dataset_path, 'phenotype/phq9.tsv'), delimiter='\t')
    #    diagnosis_df.rename(columns={'phq_9_session_id': 'session_id'}, inplace=True)



    #load audio aesthetics metrics
    # CE | Content Enjoyment CU | Content Usefulness PC | Production Complexity PQ | Production Quality

    if input_source['precompute'] is None:
        aes_json #= #TODO aees output the audio
        audio_json #= #TODO aes inout the audio
    else:
        aes_json = input_source['precompute']
        audio_json = input_source['precompute']
        #TODO aes_json = load_json_lines('../audio_aes/output_audio_aes_r2.jsonl')
        #audio_json= load_json_lines('../audio_aes/input_audio_aes_r2.jsonl')
    
    

    aes = pd.DataFrame(aes_json)
    aes[['record_id', 'session_id', 'task']] = [(a['path'].split('sub-')[1].split('/')[0], 
                                                a['path'].split('ses-')[1].split('/')[0], 
                                                a['path'].split('task-')[1].split('.wav')[0]) for a in audio_json]
    aes = aes[['record_id', 'session_id', 'task', 'CE', 'CU', 'PC', 'PQ']]

    return aes.PQ


