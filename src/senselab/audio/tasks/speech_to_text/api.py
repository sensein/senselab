"""This module implements some utilities for the speech-to-text task."""

from typing import Any, List, Optional

from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.speech_to_text.huggingface import HuggingFaceASR
from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.language import Language
from senselab.utils.data_structures.model import HFModel, SenselabModel
from senselab.utils.data_structures.script_line import ScriptLine


def transcribe_audios(
    audios: List[Audio],
    model: SenselabModel,
    language: Optional[Language] = None,
    device: Optional[DeviceType] = None,
    **kwargs: Any,  # noqa: ANN401
) -> List[ScriptLine]:
    """Transcribes all audios using the given model.

    Args:
        audios (List[Audio]): The list of audio objects to be transcribed.
        model (SenselabModel): The model used for transcription.
        language (Optional[Language]): The language of the audio (default is None).
        device (Optional[DeviceType]): The device to run the model on (default is None).
        **kwargs: Additional keyword arguments to pass to the transcription function.

    Returns:
        List[ScriptLine]: The list of script lines.

    TODO: add documentation!!!
    TODO: add tutorial
    """
    try:
        if isinstance(model, HFModel):
            return HuggingFaceASR.transcribe_audios_with_transformers(
                audios=audios, model=model, language=language, device=device, **kwargs
            )
        else:
            raise NotImplementedError("Only Hugging Face models are supported for now.")
    except TypeError as e:
        raise TypeError(e)
