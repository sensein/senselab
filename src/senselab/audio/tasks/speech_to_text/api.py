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
    """Transcribe a batch of `Audio` objects using an ASR model.

    Currently supports **Hugging Face** models (via Transformers). For best throughput,
    pass the **entire list** of audios in one call so that a single ASR pipeline is
    initialized and reused internally.

    Args:
        audios (list[Audio]):
            Audio objects to transcribe. Typical ASR models expect **mono, fixed
            sampling rate** (e.g., Whisper expects mono 16 kHz). Validate upstream.
        model (SenselabModel):
            The ASR model to use. **Only `HFModel` is supported** at present.
        language (Language, optional):
            Spoken language hint for decoding (passed to the HF pipeline when supported).
        device (DeviceType, optional):
            Preferred device for inference. For HF ASR we currently support
            ``DeviceType.CPU`` and ``DeviceType.CUDA``. If None, the backend will choose
            ``CUDA`` if available, otherwise ``CPU``.
        **kwargs:
            Model-specific options forwarded to the HF ASR helper, e.g.:
              * ``return_timestamps``: ``"word"`` | ``"segment"`` | ``None``
              * ``max_new_tokens``: ``int`` (default 128)
              * ``chunk_length_s``: ``int`` seconds (default 30)
              * ``batch_size``: ``int`` (default 1)

    Returns:
        list[ScriptLine]: One transcript object per input audio, preserving order.

    Raises:
        NotImplementedError: If `model` is not an instance of `HFModel`.
        TypeError: If invalid keyword arguments are provided downstream.

    Todo:
        - Include more models (e.g., speechbrain, nvidia)

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
        raise TypeError(e)
