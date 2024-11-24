"""This module represents the API for the speech-to-text task within the senselab package.

Currently, it supports only Hugging Face models, with plans to include more in the future.
Users can specify the audio clips to transcribe, the ASR model, the language,
the preferred device, and the model-specific parameters, and senselab handles the rest.
"""

from typing import Any, List, Optional

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speech_to_text.huggingface import HuggingFaceASR
from senselab.utils.data_structures import DeviceType, HFModel, Language, ScriptLine, SenselabModel


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

    Todo:
        - Include more models (e.g., speechbrain, nvidia)
    """
    try:
        if isinstance(model, HFModel):
            return HuggingFaceASR.transcribe_audios_with_transformers(
                audios=audios, model=model, language=language, device=device, **kwargs
            )
        else:
            raise NotImplementedError(
                "Only Hugging Face models are supported for now. We aim to support more models in the future."
            )
    except TypeError as e:
        raise TypeError(e)  # noqa: W0707
