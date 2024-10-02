"""This module provides the API for data augmentation tasks."""

from typing import List, Optional, Union

from audiomentations import Compose as AudiomentationsCompose
from torch_audiomentations import Compose as TorchAudiomentationsCompose

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.data_augmentation.audiomentations import augment_audios_with_audiomentations
from senselab.audio.tasks.data_augmentation.torch_audiomentations import augment_audios_with_torch_audiomentations
from senselab.utils.data_structures import DeviceType


def augment_audios(
    audios: List[Audio],
    augmentation: Union[TorchAudiomentationsCompose, AudiomentationsCompose],
    device: Optional[DeviceType] = None,
) -> List[Audio]:
    """Augments all provided audios.

    Uses either torch-audiomentations or audiomentations, calling the appropriate method based on
    the type of Compose. If runs on CPU, pydra is used for optimization using concurrent futures.

    Args:
        audios: List of Audios whose data will be augmented with the given augmentations.
        augmentation: A composition of augmentations (torch-audiomentations or audiomentations).
        device: The device to use for augmenting (relevant for torch-audiomentations).
            Defaults is None.

    Returns:
        List of augmented audios.
    """
    if isinstance(augmentation, TorchAudiomentationsCompose):
        return augment_audios_with_torch_audiomentations(audios, augmentation, device)
    elif isinstance(augmentation, AudiomentationsCompose):
        return augment_audios_with_audiomentations(audios, augmentation)
    else:
        raise ValueError(
            "Unsupported augmentation type." "Use either torch_audiomentations.Compose or audiomentations.Compose."
        )
