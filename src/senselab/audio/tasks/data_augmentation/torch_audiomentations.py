"""This module contains functions for applying data augmentation using torch-audiomentations."""

from typing import List

import torch

from senselab.audio.data_structures import Audio, batch_audios, unbatch_audios

try:
    from torch_audiomentations import Compose as _TACompose  # runtime check

    TORCH_AUDIOMENTATIONS_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AUDIOMENTATIONS_AVAILABLE = False
    _TACompose = object  # sentinel to satisfy type-checkers


def augment_audios_with_torch_audiomentations(audios: List[Audio], augmentation: "_TACompose") -> List[Audio]:
    """Augment a list of Audio with torch-audiomentations by processing the entire batch at once.

    Notes:
        - All audios are batched once, augmented in a single call, then unbatched back to `Audio` objects.

    Raises:
        ModuleNotFoundError: If `torch-audiomentations` is not installed.
        TypeError: If `augmentation` is not a `torch_audiomentations.Compose`.
        AssertionError: If batched audios have mixed sampling rates.
    """
    if not TORCH_AUDIOMENTATIONS_AVAILABLE:
        raise ModuleNotFoundError(
            "`torch-audiomentations` is not installed. "
            "Please install senselab audio dependencies using `pip install senselab`"
        )
    if not isinstance(augmentation, _TACompose):
        raise TypeError("`augmentation` must be an instance of torch_audiomentations.Compose")

    # torch-audiomentations Compose needs dict output
    augmentation.output_type = "dict"

    # Batch inputs once
    batched, sampling_rates, metadatas = batch_audios(audios)

    # Ensure a single sampling rate across the batch
    if isinstance(sampling_rates, list):
        assert all(sr == sampling_rates[0] for sr in sampling_rates), (
            "All Audio objects must share the same sampling rate for batched augmentation."
        )
        sampling_rate = sampling_rates[0]
    else:
        sampling_rate = sampling_rates

    # Augment whole batch: expects shape (B, C, T)
    with torch.inference_mode():
        out = augmentation(batched, sample_rate=sampling_rate).samples  # (B, C, T)

    # Unbatch back to List[Audio]
    out = out.detach().cpu()
    return unbatch_audios(out, sampling_rates, metadatas)
