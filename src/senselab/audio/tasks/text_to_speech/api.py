"""This module implements some utilities for the text-to-speech task."""

from typing import Any, List, Optional, Union

from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.text_to_speech.huggingface import HuggingFaceTTS
from senselab.audio.tasks.text_to_speech.style_tts2 import StyleTTS2
from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.language import Language
from senselab.utils.data_structures.model import HFModel, SenselabModel, StyleTTSModel


def synthesize_texts(
    texts: List[str],
    model: SenselabModel = HFModel(path_or_uri="suno/bark", revision="main"),
    reference_audios: Optional[Union[List[Audio], Audio]] = None,
    language: Optional[Language] = None,
    device: Optional[DeviceType] = None,
    **kwargs: Any,  # noqa:ANN401
) -> List[Audio]:
    """Synthesizes speech from all texts using the given model.

    Args:
        texts (List[str]): The list of text strings to be synthesized.
        model (SenselabModel): The model used for synthesis
            (Default is "suno/bark").
        reference_audios: Optional List of Audios or a single audio to use for text-to-speech as a reference
        language (Optional[Language]): The language of the text (default is None).
        device (Optional[DeviceType]): The device to run the model on (default is None).
        kwargs: other keyword arguments used by downstream models (i.e. alpha and beta for StyleTTS)

    Returns:
        List[Audio]: The list of synthesized audio objects.

    Todo:
        - Include more models
        - Include voice cloning for TTS
    """
    if isinstance(model, HFModel):
        return HuggingFaceTTS.synthesize_texts_with_transformers(texts=texts, model=model, device=device)
    elif isinstance(model, StyleTTSModel):
        if not reference_audios:
            raise ValueError("StyleTTS2 requires reference audios for generating the speech.")
        return StyleTTS2.synthesize_texts(
            texts=texts, reference_audios=reference_audios, model=model, language=language, device=device, **kwargs
        )
    else:
        raise NotImplementedError("Only Hugging Face models are supported for now.")
