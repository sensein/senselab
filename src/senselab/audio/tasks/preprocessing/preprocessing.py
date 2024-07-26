"""This module implements some utilities for the preprocessing task."""

from typing import List, Optional, Tuple

import torch
from scipy import signal
from speechbrain.augment.time_domain import Resample

from senselab.audio.data_structures.audio import Audio


def resample_audios(
    audios: List[Audio],
    resample_rate: int,
    lowcut: Optional[float] = None,
    order: int = 4,
) -> List[Audio]:
    """Resamples a list of audio signals to a given sampling rate.

    Args:
        audios (List[Audio]): List of audio objects to resample.
        resample_rate (int): Target sampling rate.
        lowcut (float, optional): Low cut frequency for IIR filter.
        order (int, optional): Order of the IIR filter. Defaults to 4.

    Returns:
        List[Audio]: Resampled audio objects.
    """
    resampled_audios = []
    for audio in audios:
        if lowcut is None:
            lowcut = resample_rate / 2 - 100
        sos = signal.butter(order, lowcut, btype="low", output="sos", fs=resample_rate)

        channels = []
        for channel in audio.waveform:
            filtered_channel = torch.from_numpy(signal.sosfiltfilt(sos, channel.numpy()).copy()).float()
            resampler = Resample(orig_freq=audio.sampling_rate, new_freq=resample_rate)
            resampled_channel = resampler(filtered_channel.unsqueeze(0)).squeeze(0)
            channels.append(resampled_channel)

        resampled_waveform = torch.stack(channels)
        resampled_audios.append(
            Audio(
                waveform=resampled_waveform,
                sampling_rate=resample_rate,
                metadata=audio.metadata.copy(),
                orig_path_or_id=audio.orig_path_or_id,
            )
        )
    return resampled_audios


def downmix_audios_to_mono(audios: List[Audio]) -> List[Audio]:
    """Downmixes a list of Audio objects to mono by averaging all channels.

    Args:
        audios (List[Audio]): A list of Audio objects with a tensor representing the audio waveform.
                                 Shape: (num_channels, num_samples).

    Returns:
        List[Audio]: The list of audio objects with a mono waveform averaged from all channels. Shape: (num_samples).
    """
    down_mixed_audios = []
    for audio in audios:
        down_mixed_audios.append(
            Audio(
                waveform=audio.waveform.mean(dim=0, keepdim=True),
                sampling_rate=audio.sampling_rate,
                metadata=audio.metadata.copy(),
                orig_path_or_id=audio.orig_path_or_id,
            )
        )

    return down_mixed_audios


def select_channel_from_audios(audios: List[Audio], channel_index: int) -> List[Audio]:
    """Selects a specific channel from a list of Audio objects.

    Args:
        audios (List[Audio]): A list of Audio objects with a tensor representing the audio waveform.
                              Shape: (num_channels, num_samples).
        channel_index (int): The index of the channel to select.

    Returns:
        List[Audio]: The list of audio objects with the selected channel. Shape: (1, num_samples).
    """
    mono_channel_audios = []
    for audio in audios:
        if audio.waveform.size(0) <= channel_index:  # should consider how much sense negative values make
            raise ValueError("channel_index should be valid")

        mono_channel_audios.append(
            Audio(
                waveform=audio.waveform[channel_index, :],
                sampling_rate=audio.sampling_rate,
                metadata=audio.metadata.copy(),
                orig_path_or_id=audio.orig_path_or_id,
            )
        )
    return mono_channel_audios


def chunk_audios(data: List[Tuple[Audio, Tuple[float, float]]]) -> List[Audio]:
    """Chunks the input audios based on the start and end timestamp.

    Args:
        data: List of tuples containing an Audio object and a tuple with start and end (in seconds) for chunking.

    Returns:
        List of Audios that have been chunked based on the provided timestamps
    """
    chunked_audios = []

    for audio, timestamps in data:
        start, end = timestamps
        if start < 0:
            raise ValueError("Start time must be greater than or equal to 0.")
        duration = audio.waveform.shape[1] / audio.sampling_rate
        if end > duration:
            raise ValueError(f"End time must be less than the duration of the audio file ({duration} seconds).")
        start_sample = int(start * audio.sampling_rate)
        end_sample = int(end * audio.sampling_rate)
        chunked_waveform = audio.waveform[:, start_sample:end_sample]

        chunked_audios.append(
            Audio(
                waveform=chunked_waveform,
                sampling_rate=audio.sampling_rate,
                metadata=audio.metadata.copy(),
                orig_path_or_id=audio.orig_path_or_id,
            )
        )
    return chunked_audios


def extract_segments(data: List[Tuple[Audio, List[Tuple[float, float]]]]) -> List[List[Audio]]:
    """Extracts segments from an audio file.

    Args:
        data: List of tuples containing an Audio object and a list of tuples with start
            and end (in seconds) for chunking.

    Returns:
        List of lists of Audios that have been chunked based on the provided timestamps.
    """
    extracted_segments = []
    for audio, timestamps in data:
        segments_data = [(audio, ts) for ts in timestamps]
        single_audio_segments = chunk_audios(segments_data)
        extracted_segments.append(single_audio_segments)
    return extracted_segments


def pad_audios(audios: List[Audio], desired_samples: int) -> List[Audio]:
    """Pads the audio segment to the desired length.

    Args:
        audios: The list of audio objects to be padded.
        desired_samples: The desired length (in samples) for the padded audio.

    Returns:
        A new Audio object with the waveform padded to the desired length.
    """
    padded_audios = []
    for audio in audios:
        current_samples = audio.waveform.shape[1]

        if current_samples >= desired_samples:
            return [audio]

        padding_needed = desired_samples - current_samples
        padded_waveform = torch.nn.functional.pad(audio.waveform, (0, padding_needed))
        padded_audio = Audio(
            waveform=padded_waveform,
            sampling_rate=audio.sampling_rate,
            metadata=audio.metadata.copy(),
            orig_path_or_id=audio.orig_path_or_id,
        )
        padded_audios.append(padded_audio)
    return padded_audios


def evenly_segment_audios(
    audios: List[Audio], segment_length: float, pad_last_segment: bool = True
) -> List[List[Audio]]:
    """Segments multiple audio files into evenly sized segments with optional padding for the last segment.

    Args:
        audios: The list of Audio objects to be segmented.
        segment_length: The desired length of each segment in seconds.
        pad_last_segment: Whether to pad the last segment to the full segment length (default is False).

    Returns:
        List of Audio objects that have been segmented.
    """
    audios_and_segment_timestamps = []
    for i, audio in enumerate(audios):
        total_duration = audio.waveform.shape[1] / audio.sampling_rate
        segment_samples = int(segment_length * audio.sampling_rate)

        # Create a list of tuples with start and end times for each segment
        timestamps = [
            (i * segment_length, (i + 1) * segment_length) for i in range(int(total_duration // segment_length))
        ]
        if total_duration % segment_length != 0:
            timestamps.append((total_duration - (total_duration % segment_length), total_duration))
        audios_and_segment_timestamps.append((audio, timestamps))

    audio_segment_lists = extract_segments([(audio, timestamps)])

    for i, audio_segment_list in enumerate(audio_segment_lists):
        if pad_last_segment and len(audio_segment_list) > 0:
            last_segment = audio_segment_list[-1]
            if last_segment.waveform.shape[1] < segment_samples:
                audio_segment_lists[i][-1] = pad_audios([last_segment], segment_samples)[0]

    return audio_segment_lists


def concatenate_audios(audios: List[Audio]) -> Audio:
    """Concatenates all audios in the list, ensuring they have the same sampling rate and shape.

    Args:
        audios: List of Audio objects to concatenate.

    Returns:
        A single Audio object that is the concatenation of all input audios.

    Raises:
        ValueError: If the audios do not all have the same sampling rate or shape.
    """
    if not audios:
        raise ValueError("The input list is empty. Please provide a list with at least one Audio object.")

    sampling_rate = audios[0].sampling_rate
    num_channels = audios[0].waveform.shape[0]

    for audio in audios:
        if audio.sampling_rate != sampling_rate:
            raise ValueError("All audios must have the same sampling rate to concatenate.")
        if audio.waveform.shape[0] != num_channels:
            raise ValueError("All audios must have the same number of channels (mono or stereo) to concatenate.")

    concatenated_waveform = torch.cat([audio.waveform for audio in audios], dim=1)

    # TODO: do we want to concatenate metadata? TBD

    return Audio(
        waveform=concatenated_waveform,
        sampling_rate=sampling_rate,
    )
