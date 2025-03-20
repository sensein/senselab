"""This module implements some utilities for the voice cloning task."""

from typing import Any, List, Optional

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.voice_cloning.coqui import CoquiVoiceCloner
from senselab.audio.tasks.voice_cloning.sparc import SparcVoiceCloner
from senselab.utils.data_structures import CoquiTTSModel, DeviceType


def clone_voices(
    source_audios: List[Audio],
    target_audios: List[Audio],
    model: Optional[CoquiTTSModel] = None,
    device: Optional[DeviceType] = None,
    **kwargs: Any,  # noqa:ANN401
) -> List[Audio]:
    """Clones voices from source audios to target audios using the given model.

    This function performs pairwise voice cloning, where the voice from each audio sample
    in the `source_audios` list is transformed into the corresponding audio
    sample in the `target_audios` list. The resulting list contains audio samples that
    preserve the content of the original source audio but with the voice replaced by the
    voice from the corresponding target audio. How other audio attributes (e.g., pitch,
    speed, and style) are preserved or transformed depends on the specific model used.

    Args:
        source_audios (List[Audio]): A list of audio samples whose voices will be "replaced"
            by the voices from the corresponding target audio samples. The content
            (e.g., words) will remain the same, but the voice sounds like the target.
        target_audios (List[Audio]): A list of audio samples whose voices will be extracted
            and used to replace the voices in the corresponding source audio samples.
        model (CoquiTTSModel, optional): The `CoquiTTSModel` model to use for voice cloning.
            The `CoquiTTSModel` is a child class of `SenselabModel` and specifies the model for cloning.
            All Coqui TTS models are supported, including "voice_conversion_models/multilingual/multi-dataset/knnvc",
            "voice_conversion_models/multilingual/vctk/freevc24",
            "voice_conversion_models/multilingual/multi-dataset/openvoice_v1",
            "voice_conversion_models/multilingual/multi-dataset/openvoice_v2".
            If None, the default model is SPARC.
        device (Optional[DeviceType], optional): The preferred device to run the model on (e.g., CPU or GPU).
            Defaults to None.
        **kwargs: Additional keyword arguments for model-specific parameters that will
            be passed directly to the underlying model's voice cloning method.
            For instance, the default SPARC model accepts a `lang` parameter to specify the language
            (e.g., `lang=Language("english")`). If `lang` is not None, the model will operate in a
            multi-language mode.

    Returns:
        List[Audio]: A list of audio samples with cloned voices from the corresponding source and target audios.

    Raises:
        ValueError: If the lengths of `source_audios` and `target_audios` do not match.
        NotImplementedError: If the specified model is not supported.
            Currently, only SPARC and Coqui TTS models are supported.

    Examples:
        >>> source_audios = [Audio.from_filepath("source1.wav"), Audio.from_filepath("source2.wav")]
        >>> target_audios = [Audio.from_filepath("target1.wav"), Audio.from_filepath("target2.wav")]
        >>> cloned_audios = clone_voices(source_audios, target_audios)
    """
    if len(source_audios) != len(target_audios):
        raise ValueError("The list of source and target audios must have the same length.")

    if model is None:
        lang = kwargs.pop("lang", None)
        return SparcVoiceCloner.clone_voices(
            source_audios=source_audios, target_audios=target_audios, device=device, lang=lang
        )
    elif isinstance(model, CoquiTTSModel):
        return CoquiVoiceCloner.clone_voices(
            source_audios=source_audios, target_audios=target_audios, model=model, device=device, **kwargs
        )
    else:
        raise NotImplementedError("Only SPARC (default) Coqui-tts models and are supported for now. ")
