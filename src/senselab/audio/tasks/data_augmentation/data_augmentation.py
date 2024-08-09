"""This module implements some utilities for audio data augmentation."""

from typing import List, Union

import pydra
import torch
from torch_audiomentations import Compose

from senselab.audio.data_structures.audio import (
    Audio,
    batch_audios,
    unbatch_audios,
)
from senselab.utils.data_structures.device import DeviceType, _select_device_and_dtype


def augment_audios(
    audios: List[Audio], augmentation: Compose, device_options: Union[DeviceType, List[DeviceType]] = [DeviceType.CPU]
) -> List[Audio]:
    """Augments all provided audios with a given augmentation, either individually or all batched together.

    Augment all audios with a user defined augmentation that can be a composition of multiple augmentations. This
    augmentation is either performed on each audio individually or all of the audios provided are batched together
    and run at once. NOTE: if batching, all audios must have the same sampling rate.

    Args:
        audios: List of Audios whose data will be augmented with the given augmentations
        augmentation: A Composition of augmentations to run on each audio (uses torch-audiomentations), should have its
            output_type set to "dict"
        device_options: The device, or a List of possible devices, to use for augmenting. If the chosen device
            is MPS or CUDA then the audios are all batched together, so for optimal performance, batching should
            be done by passing a batch_size worth of audios ar a time

    Returns:
        List of audios that has passed the all of input audios through the provided augmentation. This does
            not necessarily mean that the augmentation has been run on every audio. For more information,
            see the torch-audiomentations documentation.
    """
    augmentation.output_type = "dict"
    new_audios = []
    device_type, dtype = _select_device_and_dtype(
        compatible_devices=device_options if isinstance(device_options, List) else [device_options]
    )
    if device_type == DeviceType.CPU:
        for audio in audios:
            audio_to_augment = audio.waveform.unsqueeze(0)
            augmented_audio = augmentation(audio_to_augment, sample_rate=audio.sampling_rate).samples
            new_audios.append(
                Audio(
                    waveform=torch.squeeze(augmented_audio),
                    sampling_rate=audio.sampling_rate,
                    metadata=audio.metadata.copy(),
                    orig_path_or_id=audio.orig_path_or_id,
                )
            )
    else:
        batched_audios, sampling_rates, metadatas = batch_audios(audios)

        batched_audios = batched_audios.to(device=torch.device(str(device_type)), dtype=dtype)
        sampling_rate = sampling_rates[0] if isinstance(sampling_rates, List) else sampling_rates
        augmented_audio = augmentation(batched_audios, sample_rate=sampling_rate).samples

        augmented_audio = augmented_audio.detach().cpu()
        return unbatch_audios(augmented_audio, sampling_rates, metadatas)

    return new_audios


augment_audios_pt = pydra.mark.task(augment_audios)
