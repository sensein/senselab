"""This module implements some utilities for the preprocessing task."""

from typing import List, Tuple

import pydra
import torchaudio.functional as F

from senselab.audio.data_structures.audio import Audio


def resample_audios(audios: List[Audio], resample_rate: int, rolloff: float = 0.99) -> List[Audio]:
    """Resamples all Audios to a given sampling rate.

    Takes a list of audios and resamples each into the new sampling rate. Notably does not assume any
    specific structure of the audios (can vary in stereo vs. mono as well as their original sampling rate)

    Args:
        audios: List of Audios to resample
        resample_rate: Rate at which to resample the Audio
        rolloff: The roll-off frequency of the filter, as a fraction of the Nyquist.
            Lower values reduce anti-aliasing, but also reduce some of the highest frequencies

    Returns:
        List of Audios that have all been resampled to the given resampling rate
    """
    resampled_audios = []
    for audio in audios:
        resampled = F.resample(audio.waveform, audio.sampling_rate, resample_rate, rolloff=rolloff)
        resampled_audios.append(
            Audio(
                waveform=resampled,
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


resample_audios_pt = pydra.mark.task(resample_audios)
downmix_audios_to_mono_pt = pydra.mark.task(downmix_audios_to_mono)
chunk_audios_pt = pydra.mark.task(chunk_audios)
select_channel_from_audios_pt = pydra.mark.task(select_channel_from_audios)
