"""This module represents the API for the speech-to-text task within the senselab package.

Supports Hugging Face models (via Transformers) and NeMo models (via isolated subprocess venv).
Users can specify the audio clips to transcribe, the ASR model, the language,
the preferred device, and the model-specific parameters, and senselab handles the rest.
"""

from typing import Any, List, Optional

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speech_to_text.huggingface import HuggingFaceASR
from senselab.audio.tasks.speech_to_text.nemo import NeMoASR
from senselab.utils.compatibility import requires_compatibility
from senselab.utils.data_structures import DeviceType, HFModel, Language, ScriptLine, SenselabModel

# NeMo model ID prefixes — route to NeMo backend when detected
_NEMO_PREFIXES = ("nvidia/stt_", "nvidia/conformer")

# NVIDIA Canary-Qwen prefix — route to a separate NeMo subprocess venv
# (canary_qwen.py) that loads SALM from nemo.collections.speechlm2.models.
# The implementation lives in a follow-up commit; this constant is used by
# the dispatch table below so the routing structure is settled.
_CANARY_PREFIXES = ("nvidia/canary-",)

# Alibaba Qwen3-ASR prefix — route to a separate Qwen subprocess venv
# (qwen.py) that uses Alibaba's qwen-asr Python wrapper plus the optional
# Qwen3-ForcedAligner companion for word-level timestamps.
_QWEN_ASR_PREFIXES = ("Qwen/Qwen3-ASR",)

# HuggingFace ASR models that are known to NOT produce native timestamps —
# their HF pipelines reject return_timestamps. For these we default to
# return_timestamps=False so the pipeline returns text-only ScriptLines;
# downstream code (e.g., scripts/analyze_audio.py) can post-align via
# senselab.audio.tasks.forced_alignment to add per-segment timestamps.
_TIMESTAMP_LESS_HF_MODELS = ("ibm-granite/granite-speech-",)


@requires_compatibility("audio.tasks.speech_to_text.transcribe_audios")
def transcribe_audios(
    audios: List[Audio],
    model: SenselabModel,
    language: Optional[Language] = None,
    device: Optional[DeviceType] = None,
    **kwargs: Any,  # noqa: ANN401
) -> List[ScriptLine]:
    """Transcribe a batch of `Audio` objects using an ASR model.

    Supports **Hugging Face** models (via Transformers) and **NeMo** models
    (via isolated subprocess venv). For best throughput, pass the **entire list**
    of audios in one call so that a single ASR pipeline is initialized and
    reused internally.

    NeMo routing:
        If the ``model`` is an ``HFModel`` whose ``path_or_uri`` starts with
        ``"nvidia/stt_"`` or ``"nvidia/conformer"``, the request is routed to
        the NeMo backend (which runs in an isolated subprocess venv to avoid
        dependency conflicts).

    Args:
        audios (list[Audio]):
            Audio objects to transcribe. Typical ASR models expect **mono, fixed
            sampling rate** (e.g., Whisper expects mono 16 kHz). Validate upstream.
        model (SenselabModel):
            The ASR model to use. Supported: ``HFModel`` (Transformers or NeMo).
        language (Language, optional):
            Spoken language hint for decoding (passed to the HF pipeline when
            supported; not used for NeMo models).
        device (DeviceType, optional):
            Preferred device for inference. Supports ``DeviceType.CPU`` and
            ``DeviceType.CUDA``. If None, the backend will choose ``CUDA`` if
            available, otherwise ``CPU``.
        **kwargs:
            Model-specific options forwarded to the HF ASR helper, e.g.:
              * ``return_timestamps``: ``"word"`` | ``"segment"`` | ``None``
              * ``max_new_tokens``: ``int`` (default 128)
              * ``chunk_length_s``: ``int`` seconds (default 30)
              * ``batch_size``: ``int`` (default 1)

    Returns:
        list[ScriptLine]: One transcript object per input audio, preserving order.

    Raises:
        NotImplementedError: If `model` is not a supported type.
        TypeError: If invalid keyword arguments are provided downstream.

    Example (default settings on CPU):
        >>> from pathlib import Path
        >>> from senselab.audio.data_structures import Audio
        >>> from senselab.utils.data_structures import HFModel, DeviceType, Language
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> a2 = Audio(filepath=Path("sample2.wav").resolve())
        >>> model = HFModel(path_or_uri="openai/whisper-tiny")
        >>> lines = transcribe_audios([a1, a2], model, language=Language.EN, device=DeviceType.CPU)
        >>> lines[0].text

    Example (timestamps, longer chunks, CUDA):
        >>> from pathlib import Path
        >>> from senselab.audio.data_structures import Audio
        >>> from senselab.utils.data_structures import HFModel, DeviceType, Language
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> model = HFModel(path_or_uri="openai/whisper-small", revision="main")
        >>> lines = transcribe_audios(
        ...     [a1],
        ...     model,
        ...     language=Language.EN,
        ...     device=DeviceType.CUDA,
        ...     return_timestamps="word",
        ...     chunk_length_s=30,
        ...     batch_size=1,
        ... )
        >>> lines[0].timestamps
    """
    if not audios:
        return []

    try:
        if isinstance(model, HFModel) and str(model.path_or_uri).startswith(_NEMO_PREFIXES):
            return NeMoASR.transcribe_with_nemo(audios=audios, model=model, device=device)
        elif isinstance(model, HFModel) and str(model.path_or_uri).startswith(_CANARY_PREFIXES):
            raise NotImplementedError(
                "NVIDIA Canary-Qwen routing is registered but not yet wired. The "
                "dedicated subprocess-venv backend lives at "
                "senselab.audio.tasks.speech_to_text.canary_qwen and lands in a "
                "follow-up commit (see specs/20260506-154425-audio-analysis-asr-extensions/tasks.md, T025-T027)."
            )
        elif isinstance(model, HFModel) and str(model.path_or_uri).startswith(_QWEN_ASR_PREFIXES):
            raise NotImplementedError(
                "Alibaba Qwen3-ASR routing is registered but not yet wired. The "
                "dedicated subprocess-venv backend lives at "
                "senselab.audio.tasks.speech_to_text.qwen and lands in a "
                "follow-up commit (see specs/20260506-154425-audio-analysis-asr-extensions/tasks.md, T030-T032)."
            )
        elif isinstance(model, HFModel):
            # Default HF pipeline path. Models known to lack native timestamps
            # default to return_timestamps=False so the pipeline does not raise;
            # callers can override by passing return_timestamps explicitly.
            if "return_timestamps" not in kwargs and str(model.path_or_uri).startswith(_TIMESTAMP_LESS_HF_MODELS):
                kwargs["return_timestamps"] = False
            return HuggingFaceASR.transcribe_audios_with_transformers(
                audios=audios, model=model, language=language, device=device, **kwargs
            )
        else:
            raise NotImplementedError(
                "Only Hugging Face and NeMo models are supported for now. We aim to support more models in the future."
            )
    except TypeError as e:
        raise TypeError(e)
