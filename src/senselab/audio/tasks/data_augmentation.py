"""This module implements some utilities for audio data augmentation."""

from typing import Any, Dict

import torch
from datasets import Dataset
from torch_audiomentations import Compose

from senselab.utils.tasks.input_output import (
    _from_dict_to_hf_dataset,
    _from_hf_dataset_to_dict,
)


def augment_hf_dataset(
    dataset: Dict[str, Any], augmentation: Compose, audio_column: str = "audio"
) -> Dict[str, Any]:
    """Resamples a Hugging Face `Dataset` object."""
    hf_dataset = _from_dict_to_hf_dataset(dataset)

    def _augment_hf_row(
        row: Dataset, augmentation: Compose, audio_column: str
    ) -> Dict[str, Any]:
        waveform = row[audio_column]["array"]
        sampling_rate = row[audio_column]["sampling_rate"]

        # Ensure waveform is a PyTorch tensor
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(
                0
            )  # [num_samples] -> [1, 1, num_samples]
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(
                1
            )  # [batch_size, num_samples] -> [batch_size, 1, num_samples]

        augmented_hf_row = augmentation(
            waveform, sample_rate=sampling_rate
        ).squeeze()

        return {
            "augmented_audio": {
                "array": augmented_hf_row,
                "sampling_rate": sampling_rate,
            }
        }

    augmented_hf_dataset = hf_dataset.map(
        lambda x: _augment_hf_row(x, augmentation, audio_column)
    )
    augmented_hf_dataset = augmented_hf_dataset.remove_columns([audio_column])
    return _from_hf_dataset_to_dict(augmented_hf_dataset)
