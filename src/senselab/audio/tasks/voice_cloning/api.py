"""This module implements some utilities for the voice cloning task."""

from typing import Any, List, Optional

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.voice_cloning.knnvc import KNNVC
from senselab.utils.data_structures import DeviceType, SenselabModel, TorchModel


def clone_voices(
    source_audios: List[Audio],
    target_audios: List[Audio],
    model: SenselabModel = TorchModel(path_or_uri="bshall/knn-vc", revision="master"),
    device: Optional[DeviceType] = None,
    **kwargs: Any,  # noqa:ANN401
) -> List[Audio]:
    """Clones voices from source audios to target audios using the given model.

    This function performs pairwise voice cloning, where the voice from each audio sample
    in the `source_audios` list is transformed into the corresponding audio
    sample in the `target_audios` list. The resulting list contains audio samples that
    preserve the content of the original source audio but with the voice replaced by the
    voice from the corresponding target audio.

    Args:
        source_audios (List[Audio]): A list of audio samples whose voices will be "replaced"
            by the voices from the corresponding target audio samples. The content
            (e.g., words) will remain the same, but the voice sounds like the target.
        target_audios (List[Audio]): A list of audio samples whose voices will be extracted
            and used to replace the voices in the corresponding source audio samples.
        model (SenselabModel, optional): The model to use for voice cloning. Currently,
            only KNNVC (K-Nearest Neighbors Voice Conversion) is supported, encapsulated
            by the `TorchModel` class. `TorchModel` is a child class of `SenselabModel`
            and specifies the model and revision for cloning. Defaults to
            `TorchModel(path_or_uri="bshall/knn-vc", revision="master")`.
        device (Optional[DeviceType], optional): The device to run the model on (e.g., CPU or GPU).
            Defaults to None.
        **kwargs: Additional keyword arguments for model-specific parameters that will
            be passed directly to the underlying model's voice cloning method.

    Returns:
        List[Audio]: A list of audio samples with cloned voices from the corresponding source and target audios.

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
