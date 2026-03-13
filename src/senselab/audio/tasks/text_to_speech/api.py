"""This module implements some utilities for the text-to-speech task."""

from typing import Any, List, Optional, Tuple, TypeGuard

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.text_to_speech.coqui import CoquiTTS
from senselab.audio.tasks.text_to_speech.huggingface import HuggingFaceTTS
from senselab.utils.data_structures import CoquiTTSModel, DeviceType, HFModel, Language, SenselabModel, TorchModel


def synthesize_texts(
    texts: List[str],
    model: Optional[SenselabModel] = None,
    language: Optional[Language] = None,
    device: Optional[DeviceType] = None,
    targets: Optional[List[Audio | Tuple[Audio, str]]] = None,
    **kwargs: Any,  # noqa: ANN401
) -> List[Audio]:
    """Synthesizes speech from all texts using the given model.

    This function synthesizes speech from a list of text strings using the specified text-to-speech (TTS) model.
    It supports models from HuggingFace and coqui-tts.

    Args:
        texts (List[str]): The list of text strings to be synthesized.
        model (SenselabModel): The model used for synthesis.
                If None, the default model "suno/bark" is used.
        language (Optional[Language]): The language of the text
            (default is None).
        device (Optional[DeviceType]): The device to run the model on
            (default is None).
        targets (Optional[List[Audio | Tuple[Audio, str]]]):
            A list where each element is a target audio or a tuple of target audio and transcript.
            Depending on the model being used, the `target` input may need to be provided in a specific format:
            - Hugging Face models do not require a `target` input at all.
        **kwargs: Additional keyword arguments to pass to the synthesis function.
            Depending on the model used (e.g., HFModel), additional arguments
            may be required. Refer to the model-specific documentation for details.

    Returns:
        List[Audio]: The list of synthesized audio objects.
    """
    if model is None:
        model = HFModel(path_or_uri="suno/bark", revision="main")

    if targets is not None:
        assert len(targets) == len(texts), ValueError("Provided targets should be same length as texts")

        for i, target in enumerate(targets):
            if isinstance(target, tuple):
                assert len(target[1]) > 0, ValueError(f"{i}th target was expected to have a transcript, but was empty.")

    if isinstance(model, HFModel):
        return HuggingFaceTTS.synthesize_texts_with_transformers(texts=texts, model=model, device=device, **kwargs)
    elif isinstance(model, CoquiTTSModel):
        coqui_targets: Optional[List[Audio]] = None
        if targets is not None:
            coqui_targets = [
                t if isinstance(t, Audio) else t[0]  # extract Audio from (Audio, str)
                for t in targets
            ]
        return CoquiTTS.synthesize_texts_with_coqui(
            texts=texts, targets=coqui_targets, model=model, device=device, language=language, **kwargs
        )
    else:
        raise NotImplementedError("Only Hugging Face models and select Torch models are supported for now.")


def _check_all_have_transcripts(targets: List[Audio | Tuple[Audio, str]]) -> TypeGuard[List[Tuple[Audio, str]]]:
    for target in targets:
        if isinstance(target, Audio):
            return False
        elif isinstance(target, tuple):
            if len(target) != 2 or not isinstance(target[0], Audio) or not isinstance(target[1], str):
                return False
            elif len(target[1]) == 0:
                return False

    return True
