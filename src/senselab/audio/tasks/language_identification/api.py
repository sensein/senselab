"""This module provides the API for the senselab language identification task."""

from typing import List, Optional

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.language_identification.speechbrain import SpeechBrainLanguageIdentifier
from senselab.utils.data_structures import DeviceType, SenselabModel, SpeechBrainModel
from senselab.utils.data_structures.language import Language


def identify_languages(
    audios: List[Audio],
    model: SenselabModel = SpeechBrainModel(path_or_uri="speechbrain/lang-id-voxlingua107-ecapa", revision="main"),
    device: Optional[DeviceType] = None,
) -> List[Language | None]:
    """Identifies the language for all provided audio samples.

    Args:
        audios (List[Audio]): The list of audio objects to process.
        model (SenselabModel): The model used for language identification
            (default is "speechbrain/lang-id-voxlingua107-ecapa").
        device (Optional[DeviceType]): The device to run the model on (default is None).

    Returns:
        List[Language]: The list of identified Language objects.
    """
    if isinstance(model, SpeechBrainModel):
        return SpeechBrainLanguageIdentifier.identify_languages(audios=audios, model=model, device=device)
    else:
        raise NotImplementedError("Only SpeechBrain models are supported for now.")
