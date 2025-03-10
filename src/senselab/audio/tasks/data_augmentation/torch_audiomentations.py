"""This module implements some utilities for audio data augmentation with torch_audiomentations."""

from typing import List, Optional

import torch

from senselab.audio.data_structures import (
    Audio,
    batch_audios,
    unbatch_audios,
)
from senselab.utils.data_structures import DeviceType, _select_device_and_dtype

try:
    from torch_audiomentations import Compose

    TORCH_AUDIOMENTATIONS_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AUDIOMENTATIONS_AVAILABLE = False


def augment_audios_with_torch_audiomentations(
    audios: List[Audio], augmentation: "Compose", device: Optional[DeviceType] = None
) -> List[Audio]:
    """Augments all provided audios with a given augmentation, either individually or all batched together.

    Augment all audios with a user defined augmentation that can be a composition of multiple augmentations.
    This augmentation is either performed on each audio individually (using pydra for optimization)
    or all of the audios provided are batched together and run at once.
    It uses torch_audiomentations. If batching, all audios must have the same sampling rate.

    Args:
        audios: List of Audios whose data will be augmented with the given augmentations
        augmentation: A Composition of augmentations to run on each audio (uses torch-audiomentations), should have its
            output_type set to "dict"
        device: The device to use for augmenting. If the chosen device
            is CUDA then the audios are all batched together, so for optimal performance, batching should
            be done by passing a batch_size worth of audios ar a time.
            Default is None, which will select the device automatically.

    Returns:
        List of audios that has passed the all of input audios through the provided augmentation. This does
            not necessarily mean that the augmentation has been run on every audio. For more information,
            see the torch-audiomentations documentation.
    """
    if not TORCH_AUDIOMENTATIONS_AVAILABLE:
        raise ModuleNotFoundError(
            "`torch-audiomentations` is not installed. "
            "Please install senselab audio dependencies using `pip install senselab['audio']`"
        )

    augmentation.output_type = "dict"
    device_type, _ = _select_device_and_dtype(
        user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
    )
    if device_type == DeviceType.CPU:
        '''
        # The commented code is for parallelizing the augmentation using pydra
        # Due to some issues with pydra, this is disabled for now
        import pydra

        def _augment_single_audio(audio: Audio, augmentation: Compose):  # noqa: ANN202
            """Augments a single audio with torch-audiomentations.

            Args:
                audio: The audio to be augmented.
                augmentation: The torch-audiomentations augmentation to be applied.

            Returns:
                The augmented audio. The returned data type is not explicitly specified
                    in the function signature because it would brake pydra.
            """
            augmented_waveform = augmentation(audio.waveform.unsqueeze(0), sample_rate=audio.sampling_rate).samples
            return Audio(
                waveform=torch.squeeze(augmented_waveform),
                sampling_rate=audio.sampling_rate,
                metadata=audio.metadata.copy(),
                orig_path_or_id=audio.orig_path_or_id,
            )

        _augment_single_audio_pt = pydra.mark.task(_augment_single_audio)

        # Create the workflow
        wf = pydra.Workflow(name="audio_augmentation_wf", input_spec=["y"])
        wf.split("y", y=audios)
        wf.add(_augment_single_audio_pt(name="augment_audio", audio=wf.lzin.y, augmentation=augmentation))
        wf.set_output([("augmented_audio", wf.augment_audio.lzout.out)])

        # Execute the workflow
        with pydra.Submitter(plugin="cf") as submitter:
            submitter(wf)
        outputs = wf.result()
        return [out.output.augmented_audio for out in outputs]
        '''
        new_audios = []
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
        return new_audios
    else:
        batched_audios, sampling_rates, metadatas = batch_audios(audios)

        batched_audios = batched_audios.to(device=torch.device(device_type.value))
        sampling_rate = sampling_rates[0] if isinstance(sampling_rates, List) else sampling_rates
        augmented_audio = augmentation(batched_audios, sample_rate=sampling_rate).samples

        augmented_audio = augmented_audio.detach().cpu()
        return unbatch_audios(augmented_audio, sampling_rates, metadatas)
