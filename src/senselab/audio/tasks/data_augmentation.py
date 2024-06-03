"""This module implements some utilities for audio data augmentation."""

from typing import Any, Dict, List

import torch
from datasets import Dataset
from torch_audiomentations import Compose

from senselab.utils.data_structures.audio import (
    Audio,
    batch_audios,
    unbatch_audios,
)
from senselab.utils.tasks.input_output import (
    _from_dict_to_hf_dataset,
    _from_hf_dataset_to_dict,
)


def augment_audio_dataset(audios: List[Audio], augmentation: Compose, batched: bool = False) -> List[Audio]:
    """Augments all provided audios with a given augmentation, either individually or all batched together.

    Augment all audios with a user defined augmentation that can be a composition of multiple augmentations. This
    augmentation is either performed on each audio individually or all of the audios provided are batched together
    and run at once. NOTE: if batching, all audios must have the same sampling rate.

    Args:
        audios: List of Audios whose data will be augmented with the given augmentations
        augmentation: A Composition of augmentations to run on each audio (uses torch-audiomentations), should have its
            output_type set to "dict"
        batched: flag for whether the composition should be run on audios individually, typical for a CPU, or
            by batching all of the audios together for a possible speed-up on a GPU

    Returns:
        List of audios that has passed the all of input audios through the provided augmentation. This does
            not necessarily mean that the augmentation has been run on every audio. For more information,
            see the torch-audiomentations documentation.
    """
    augmentation.output_type = "dict"
    new_audios = []
    if not batched:
        for audio in audios:
            audio_to_augment = audio.audio_data.unsqueeze(0)
            augmented_audio = augmentation(audio_to_augment, sample_rate=audio.sampling_rate).samples
            new_audios.append(
                Audio(
                    audio_data=torch.squeeze(augmented_audio),
                    sampling_rate=audio.sampling_rate,
                    metadata=audio.metadata.copy(),
                )
            )
    else:
        batched_audios, sampling_rates, metadatas = batch_audios(audios)
        sampling_rate = sampling_rates[0] if isinstance(sampling_rates, List) else sampling_rates
        augmented_audio = augmentation(batched_audios, sample_rate=sampling_rate).samples

        return unbatch_audios(augmented_audio, sampling_rates, metadatas)

    return new_audios


def augment_hf_dataset(dataset: Dict[str, Any], augmentation: Compose, audio_column: str = "audio") -> Dict[str, Any]:
    """Resamples a Hugging Face `Dataset` object."""
    hf_dataset = _from_dict_to_hf_dataset(dataset)

    def _augment_hf_row(row: Dataset, augmentation: Compose, audio_column: str) -> Dict[str, Any]:
        waveform = row[audio_column]["array"]
        sampling_rate = row[audio_column]["sampling_rate"]

        # Ensure waveform is a PyTorch tensor
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)  # [num_samples] -> [1, 1, num_samples]
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)  # [batch_size, num_samples] -> [batch_size, 1, num_samples]

        augmented_hf_row = augmentation(waveform, sample_rate=sampling_rate).squeeze()

        return {
            "augmented_audio": {
                "array": augmented_hf_row,
                "sampling_rate": sampling_rate,
            }
        }

    augmented_hf_dataset = hf_dataset.map(lambda x: _augment_hf_row(x, augmentation, audio_column))
    augmented_hf_dataset = augmented_hf_dataset.remove_columns([audio_column])
    return _from_hf_dataset_to_dict(augmented_hf_dataset)
