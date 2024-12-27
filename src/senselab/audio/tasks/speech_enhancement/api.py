"""This module provides the API for the senselab speech enhancement task."""

from typing import List, Optional

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speech_enhancement.speechbrain import SpeechBrainEnhancer
from senselab.utils.data_structures import DeviceType, SpeechBrainModel


def enhance_audios(
    audios: List[Audio],
    model: Optional[SpeechBrainModel] = None,
    device: Optional[DeviceType] = None,
) -> List[Audio]:
    """Enhances all audios using the given model.

    Args:
        audios (List[Audio]): The list of audio objects to be enhanced.
        model (SenselabModel): The model used for enhancement.
            If None, the default model "speechbrain/sepformer-wham16k-enhancement" is used.
        device (Optional[DeviceType]): The device to run the model on (default is None).

    Returns:
        List[Audio]: The list of enhanced audio objects.
    """
    if model is None:
        model = SpeechBrainModel(path_or_uri="speechbrain/sepformer-wham16k-enhancement", revision="main")

    if isinstance(model, SpeechBrainModel):
        return SpeechBrainEnhancer.enhance_audios_with_speechbrain(audios=audios, model=model, device=device)
    else:
        raise NotImplementedError(
            "Only SpeechBrain models are supported for now. We aim to support more models in the future."
        )
