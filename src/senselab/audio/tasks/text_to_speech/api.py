"""This module implements some utilities for the text-to-speech task."""

from typing import List, Optional

from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.text_to_speech.huggingface import HuggingFaceTTS
from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.language import Language
from senselab.utils.data_structures.model import HFModel, SenselabModel


def synthesize_texts(
    texts: List[str],
    model: SenselabModel = HFModel(path_or_uri="suno/bark", revision="main"),
    language: Optional[Language] = None,
    device: Optional[DeviceType] = None,
) -> List[Audio]:
    """Synthesizes speech from all texts using the given model.

    Args:
        texts (List[str]): The list of text strings to be synthesized.
        model (SenselabModel): The model used for synthesis
            (Default is "suno/bark").
        language (Optional[Language]): The language of the text (default is None).
        device (Optional[DeviceType]): The device to run the model on (default is None).

    Returns:
        List[Audio]: The list of synthesized audio objects.

    Todo:
        - Include more models
        - Include voice cloning for TTS
    """
    if isinstance(model, HFModel):
        return HuggingFaceTTS.synthesize_texts_with_transformers(texts=texts, model=model, device=device)
    else:
        raise NotImplementedError("Only Hugging Face models are supported for now.")
