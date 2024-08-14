"""This module implements some utilities for the voice cloning task."""

from typing import Any, List, Optional

from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.voice_cloning.knnvc import KNNVC
from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.model import SenselabModel, TorchModel


def clone_voices(
    source_audios: List[Audio],
    target_audios: List[Audio],
    model: SenselabModel = TorchModel(path_or_uri="bshall/knn-vc", revision="master"),
    device: Optional[DeviceType] = None,
    **kwargs: Any,  # noqa:ANN401
) -> List[Audio]:
    """Clones voices from source audios to target audios using the given model.

    This function performs a pairwise voice cloning operation, where each audio in the
    `source_audios` list is converted and mapped onto the corresponding audio in the
    `target_audios` list. The resulting list contains target audio samples with their
    voices replaced by the voices from the corresponding source samples.

    Args:
        source_audios (List[Audio]): A list of source audio samples, from which the voices will be cloned.
        target_audios (List[Audio]): A list of target audio samples, to which the voices will be cloned.
        model (SenselabModel, optional): The model to use for voice cloning. As of now, only KNNVC
            (K-Nearest Neighbors Voice Conversion) is supported, which is encapsulated by the `TorchModel`
            class. `TorchModel` is a child class of `SenselabModel` and specifies the model and revision
            for cloning. Defaults to `TorchModel(path_or_uri="bshall/knn-vc", revision="master")`.
        device (Optional[DeviceType], optional): The device to run the model on (e.g., CPU or GPU). Defaults to None.
        **kwargs: Additional keyword arguments for model-specific parameters.
            These will be passed directly to the underlying model's voice cloning method.

    Returns:
        List[Audio]: A list of target audio samples with cloned voices from the corresponding source audios.

    Raises:
        ValueError: If the lengths of `source_audios` and `target_audios` do not match.
        NotImplementedError: If the specified model is not supported. Currently, only KNNVC is supported.

    Examples:
        >>> source_audios = [Audio.from_filepath("source1.wav"), Audio.from_filepath("source2.wav")]
        >>> target_audios = [Audio.from_filepath("target1.wav"), Audio.from_filepath("target2.wav")]
        >>> cloned_audios = clone_voices(source_audios, target_audios)

    Todo:
        Add logging with timestamps.
    """
    if len(source_audios) != len(target_audios):
        raise ValueError("The list of source and target audios must have the same length.")

    if isinstance(model, TorchModel) and model.path_or_uri == "bshall/knn-vc":
        return KNNVC.clone_voices_with_knn_vc(
            source_audios=source_audios, target_audios=target_audios, model=model, device=device, **kwargs
        )
    else:
        raise NotImplementedError("Only KNNVC is supported for now.")
