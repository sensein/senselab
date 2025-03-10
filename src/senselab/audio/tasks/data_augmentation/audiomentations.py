"""This module implements some utilities for audio data augmentation with audiomentations."""

from typing import List

from senselab.audio.data_structures import Audio

try:
    from audiomentations import Compose

    AUDIOMENTATIONS_AVAILABLE = True
except ModuleNotFoundError:
    AUDIOMENTATIONS_AVAILABLE = False


def augment_audios_with_audiomentations(audios: List[Audio], augmentation: "Compose") -> List[Audio]:
    """Augments all provided audios with audiomentations library.

    Args:
        audios: List of Audios whose data will be augmented with the given augmentations.
        augmentation: A Composition of augmentations to run on each audio (uses audiomentations).

    Returns:
        List of audios that have passed through the provided augmentation.
    """

    def _augment_single_audio(audio: Audio, augmentation: "Compose"):  # noqa: ANN202
        """Augments a single audio with audiomentations.

        Args:
            audio: The audio to be augmented.
            augmentation: The audiomentations augmentation to be applied.

        Returns:
            The augmented audio. The returned data type is not explicitly specified
                in the function signature because it would brake pydra.
        """
        augmented_waveform = augmentation(samples=audio.waveform.numpy(), sample_rate=audio.sampling_rate)
        return Audio(
            waveform=augmented_waveform,
            sampling_rate=audio.sampling_rate,
            metadata=audio.metadata.copy(),
            orig_path_or_id=audio.orig_path_or_id,
        )

    if not AUDIOMENTATIONS_AVAILABLE:
        raise ModuleNotFoundError(
            "`audiomentations` is not installed. "
            "Please install senselab audio dependencies using `pip install senselab['audio']`."
        )

    """
    # The commented code is for parallelizing the augmentation using pydra
    # Due to some issues with pydra, this is disabled for now

    import pydra

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
    """

    augmented_audios = []
    for audio in audios:
        augmented_audios.append(_augment_single_audio(audio, augmentation))

    return augmented_audios
