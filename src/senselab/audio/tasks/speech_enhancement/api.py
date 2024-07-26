"""This module provides the API for the senselab speech enhancement task."""

from typing import List, Optional

import pydra

from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.speech_enhancement.speechbrain import SpeechBrainEnhancer
from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.model import SenselabModel, SpeechBrainModel


def enhance_audios(
    audios: List[Audio],
    model: SenselabModel = SpeechBrainModel(path_or_uri="speechbrain/sepformer-wham16k-enhancement", revision="main"),
    device: Optional[DeviceType] = None,
) -> List[Audio]:
    """Enhances all audios using the given model.

    Args:
        audios (List[Audio]): The list of audio objects to be enhanced.
        model (SenselabModel): The model used for enhancement
            (default is "speechbrain/sepformer-wham16k-enhancement").
        device (Optional[DeviceType]): The device to run the model on (default is None).

    Returns:
        List[Audio]: The list of enhanced audio objects.
    """
    if isinstance(model, SpeechBrainModel):
        return SpeechBrainEnhancer.enhance_audios_with_speechbrain(audios=audios, model=model, device=device)
    else:
        raise NotImplementedError("Only SpeechBrain models are supported for now.")


enhance_audios_pt = pydra.mark.task(enhance_audios)
