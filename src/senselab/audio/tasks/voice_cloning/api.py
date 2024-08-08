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

    Args:
        source_audios (List[Audio]): A list of source audio samples.
        target_audios (List[Audio]): A list of target audio samples.
        model (SenselabModel, optional): The model to use for voice cloning. 
            Defaults to TorchModel(path_or_uri="bshall/knn-vc", revision="master").
        device (Optional[DeviceType], optional): The device to run the model on. Defaults to None.
        **kwargs: Additional keyword arguments for model-specific parameters.
            For details, check the documentation of the specific functions 
            (e.g., `KNNVC.clone_voices_with_knn_vc`).

    Returns:
        List[Audio]: A list of audio samples with cloned voices.

    Raises:
        ValueError: If the lengths of source_audios and target_audios do not match.
        NotImplementedError: If the model is not supported.

    Examples:
        >>> source_audios = [Audio.from_filepath("source1.wav"), Audio.from_filepath("source2.wav")]
        >>> target_audios = [Audio.from_filepath("target1.wav"), Audio.from_filepath("target2.wav")]
        >>> cloned_audios = clone_voices(source_audios, target_audios)
    """
    if len(source_audios) != len(target_audios):
        raise ValueError("Source and target audios must have the same length.")

    if isinstance(model, TorchModel) and model.path_or_uri == "bshall/knn-vc":
        return KNNVC.clone_voices_with_knn_vc(
            source_audios=source_audios,
            target_audios=target_audios,
            model=model,
            device=device,
            **kwargs
        )
    else:
        raise NotImplementedError("Only KNNVC is supported for now.")