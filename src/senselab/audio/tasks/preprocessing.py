"""This module implements some utilities for the preprocessing task."""

from typing import Any, Dict, List, Tuple

import torch
import torchaudio.functional as F
from datasets import Dataset

from senselab.utils.data_structures.audio import Audio
from senselab.utils.tasks.input_output import _from_dict_to_hf_dataset, _from_hf_dataset_to_dict


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
            Audio(waveform=resampled, sampling_rate=resample_rate, metadata=audio.metadata, path_or_id=audio.path_or_id)
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
        if audio.waveform.dim() != 2 or audio.waveform.size(0) < 1:
            raise ValueError("waveform should have shape (num_channels, num_samples)")

        down_mixed_audio = audio.copy()
        down_mixed_audio.waveform = audio.waveform.mean(dim=0)
        down_mixed_audios.append(down_mixed_audio)
    return down_mixed_audios


def select_channel_from_audios(audios: List[Audio], channel_index: int) -> List[Audio]:
    """Selects a specific channel from a list of Audio objects.

    Args:
        audios (List[Audio]): A list of Audio objects with a tensor representing the audio waveform.
                              Shape: (num_channels, num_samples).
        channel_index (int): The index of the channel to select.

    Returns:
        List[Audio]: The list of audio objects with the selected channel. Shape: (num_samples).
    """
    mono_channel_audios = []
    for audio in audios:
        print("audio.waveform.shape")
        print(audio.waveform.shape)
        if audio.waveform.dim() != 2:
            raise ValueError("waveform should have shape (num_channels, num_samples)")
        if audio.waveform.size(0) <= channel_index:
            raise ValueError("channel_index should be valid")

        mono_channel_audio = audio.copy()
        mono_waveform = audio.waveform[channel_index, :]
        if mono_waveform.dim() != 2:
            mono_waveform = mono_waveform.unsqueeze(0)
        mono_channel_audio.waveform = mono_waveform
        mono_channel_audios.append(mono_channel_audio)
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
                metadata=audio.metadata,
                path_or_id=f"{audio.path_or_id}_chunk_{start}_{end}",  # TODO: Fix this
            )
        )
    return chunked_audios


def resample_hf_dataset(dataset: Dict[str, Any], resample_rate: int, rolloff: float = 0.99) -> Dict[str, Any]:
    """Resamples a Hugging Face `Dataset` object."""
    hf_dataset = _from_dict_to_hf_dataset(dataset)

    def _resample_hf_row(row: Dataset, resample_rate: int, rolloff: float = 0.99) -> Dict[str, Any]:
        """Resamples audio data in a hf dataset row.

        A lower rolloff will therefore reduce the amount of aliasing,
        but it will also reduce some of the higher frequencies.
        """
        waveform = row["audio"]["array"]
        # Ensure waveform is a PyTorch tensor
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform)
        sampling_rate = row["audio"]["sampling_rate"]

        resampled_waveform = F.resample(waveform, sampling_rate, resample_rate, rolloff=rolloff)

        return {
            "audio": {
                "array": resampled_waveform,
                "sampling_rate": resample_rate,
            }
        }

    resampled_hf_dataset = hf_dataset.map(lambda x: _resample_hf_row(x, resample_rate, rolloff))
    return _from_hf_dataset_to_dict(resampled_hf_dataset)
