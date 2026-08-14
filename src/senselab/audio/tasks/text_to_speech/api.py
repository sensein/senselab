"""This module implements some utilities for the text-to-speech task."""

from typing import Any, Dict, List, Optional, Tuple, TypeGuard

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.text_to_speech.coqui import CoquiTTS
from senselab.audio.tasks.text_to_speech.huggingface import HuggingFaceTTS
from senselab.audio.tasks.text_to_speech.qwen_tts import synthesize_texts_with_qwen
from senselab.utils.compatibility import requires_compatibility
from senselab.utils.data_structures import CoquiTTSModel, DeviceType, HFModel, Language, SenselabModel, TorchModel

# Alibaba Qwen3-TTS prefix -- route to a separate subprocess venv (qwen_tts.py) that
# uses Alibaba's qwen-tts PyPI package. Checked before the generic HFModel branch
# below, the same ordering speech_to_text/api.py uses for its own Qwen prefix.
_QWEN_TTS_PREFIXES = ("Qwen/Qwen3-TTS",)


@requires_compatibility("audio.tasks.text_to_speech.synthesize_texts")
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

    if isinstance(model, HFModel) and str(model.path_or_uri).startswith(_QWEN_TTS_PREFIXES):
        qwen_tts_kwargs: Dict[str, Any] = {}
        # `speaker` and `instruct` have no named parameter to bind to, so they do arrive in
        # kwargs. `language` does have one -- testing `if "language" in kwargs` could never
        # match it, so the caller's language was dropped and the worker synthesized on its
        # "Auto" default with no error. Omitted entirely when None so that default still
        # applies; the checkpoint speaks language *names* ("Italian"), not ISO codes.
        if language is not None:
            qwen_tts_kwargs["language"] = language.name
        for key in ("speaker", "instruct"):
            if key in kwargs:
                qwen_tts_kwargs[key] = kwargs.pop(key)
        return synthesize_texts_with_qwen(texts=texts, model=model, device=device, **qwen_tts_kwargs)
    elif isinstance(model, HFModel):
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
