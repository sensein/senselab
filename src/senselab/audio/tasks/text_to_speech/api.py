"""This module implements some utilities for the text-to-speech task."""

from typing import Any, List, Optional, Union

from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.text_to_speech.huggingface import HuggingFaceTTS
from senselab.audio.tasks.text_to_speech.marstts import Mars5TTS
from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.language import Language
from senselab.utils.data_structures.model import HFModel, SenselabModel, TorchModel


def synthesize_texts(
    texts: List[str],
    target: Optional[List[Union[Audio, Optional[str]]]] = None,
    model: SenselabModel = HFModel(path_or_uri="suno/bark", revision="main"),
    language: Optional[Language] = None,
    device: Optional[DeviceType] = None,
    **kwargs: Any,  # noqa: ANN401
) -> List[Audio]:
    """Synthesizes speech from all texts using the given model.

    Args:
        texts (List[str]): The list of text strings to be synthesized.
        target (Optional[List[Union[Audio, Optional[str]]]]): 
            A list where each element is a tuple of target audio and optional transcript.
        model (SenselabModel): The model used for synthesis (Default is "suno/bark").
        language (Optional[Language]): The language of the text (default is None).
        device (Optional[DeviceType]): The device to run the model on (default is None).
        **kwargs: Additional keyword arguments to pass to the synthesis function.

    Returns:
        List[Audio]: The list of synthesized audio objects.
    """
    target_audios = None
    target_transcripts = None
    if target is not None:
        target_audios = [t[0] for t in target]
        target_transcripts = [t[1] for t in target if t[1] is not None]

        if len(texts) != len(target_audios):
            raise ValueError("The lengths of texts and target audios must be the same.")
        if target_transcripts and len(texts) != len(target_transcripts):
            raise ValueError("The lengths of texts and target transcripts must be the same.")

    if isinstance(model, HFModel):
        return HuggingFaceTTS.synthesize_texts_with_transformers(texts=texts, model=model, device=device, **kwargs)
    elif isinstance(model, TorchModel) and model.path_or_uri == "Camb-ai/mars5-tts" and model.revision == "master":
        target = list(zip(target_audios, target_transcripts))
        return Mars5TTS.synthesize_texts_with_mars5tts(
            texts=texts,
            target=target,
            language=language,
            device=device,
            **kwargs
        )
    else:
        raise NotImplementedError("Only Hugging Face models are supported for now.")
