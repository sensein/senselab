"""This module implements some utilities for the text-to-speech task."""

from typing import Any, List, Optional, Tuple

from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.text_to_speech.huggingface import HuggingFaceTTS
from senselab.audio.tasks.text_to_speech.marstts import Mars5TTS
from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.language import Language
from senselab.utils.data_structures.model import HFModel, SenselabModel, TorchModel


def synthesize_texts(
    texts: List[str],
    target: Optional[List[Tuple[Audio, Optional[str]]]] = None,
    model: SenselabModel = HFModel(path_or_uri="suno/bark", revision="main"),
    language: Optional[Language] = None,
    device: Optional[DeviceType] = None,
    **kwargs: Any,  # noqa: ANN401
) -> List[Audio]:
    """Synthesizes speech from all texts using the given model.

    Args:
        texts (List[str]): The list of text strings to be synthesized.
        target (Optional[List[Tuple[Audio, Optional[str]]]]):
            A list where each element is a tuple of target audio and optional transcript.
        model (HFModel): The model used for synthesis (Default is "suno/bark").
        language (Optional[Language]): The language of the text (default is None).
        device (Optional[DeviceType]): The device to run the model on (default is None).
        **kwargs: Additional keyword arguments to pass to the synthesis function.
            Depending on the model used (e.g., Hugging Face or Mars5-TTS), additional arguments
            may be required. You can find details in the documentation of each function
            (e.g., `synthesize_texts_with_transformers` or `synthesize_texts_with_mars5tts`)
            and in the card of each model.

    Returns:
        List[Audio]: The list of synthesized audio objects.
    """
    target_audios: List[Audio] = []
    target_transcripts: List[Optional[str]] = []
    if target is not None:
        target_audios = []
        target_transcripts = []

        for t in target:
            if isinstance(t, tuple):
                audio, transcript = t
                if isinstance(audio, Audio):
                    target_audios.append(audio)
                else:
                    raise TypeError("Expected the first element of the tuple to be an Audio object.")

                if isinstance(transcript, str):
                    target_transcripts.append(transcript)
                else:
                    raise TypeError("Expected the second element of the tuple to be a string.")
            elif isinstance(t, Audio):
                target_audios.append(t)
                target_transcripts.append(None)
            else:
                raise TypeError("Expected elements of target to be either Audio objects or tuples of (Audio, str).")

        if len(texts) != len(target_audios):
            raise ValueError("The lengths of texts and target audios must be the same.")
        if any(t is not None for t in target_transcripts) and len(texts) != len(target_transcripts):
            raise ValueError("The lengths of texts and target transcripts must be the same.")

    if isinstance(model, HFModel):
        return HuggingFaceTTS.synthesize_texts_with_transformers(texts=texts, model=model, device=device, **kwargs)
    elif isinstance(model, TorchModel) and model.path_or_uri == "Camb-ai/mars5-tts" and model.revision == "master":
        # Converting target to the required format for Mars5TTS
        target_for_mars5tts = []
        for audio, transcript in zip(target_audios, target_transcripts):
            if transcript is None:
                raise ValueError("Transcript is required.")
            target_for_mars5tts.append((audio, transcript))
        return Mars5TTS.synthesize_texts_with_mars5tts(
            texts=texts, target=target_for_mars5tts, language=language, device=device, **kwargs
        )
    else:
        raise NotImplementedError("Only Hugging Face models are supported for now.")
