"""Unified API for SSL embedding extraction across multiple backends.

Routes to the appropriate backend based on model type:
- ``HFModel`` -> HuggingFace Transformers (wav2vec2, HuBERT, WavLM, etc.)
- ``SpeechBrainModel`` -> SpeechBrain EncoderClassifier (ECAPA-TDNN, x-vector, etc.)
- ``str`` -> S3PRL subprocess venv (apc, tera, cpc, etc.)
"""

import os
from typing import List, Optional, Union

import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.ssl_embeddings.self_supervised_features import (
    SpeechBrainSSLEmbeddings,
    SSLEmbeddingsFactory,
)
from senselab.utils.compatibility import requires_compatibility
from senselab.utils.data_structures import DeviceType, HFModel, SenselabModel, SpeechBrainModel


@requires_compatibility("audio.tasks.ssl_embeddings.extract_ssl_embeddings_from_audios")
def extract_ssl_embeddings_from_audios(
    audios: List[Audio],
    model: Union[SenselabModel, str],
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    device: Optional[DeviceType] = None,
) -> List[torch.Tensor]:
    """Extract embedding of audio signals from pre-trained SSL models.

    Dispatches to the correct backend based on the model type:
    - **HFModel**: Uses HuggingFace Transformers (e.g., wav2vec2-base,
      HuBERT-large, WavLM-large). Returns all hidden states concatenated
      with shape ``[num_layers, time_frames, embedding_dim]``.
    - **SpeechBrainModel**: Uses SpeechBrain EncoderClassifier (e.g.,
      ECAPA-TDNN, x-vector, ResNet). Returns fixed-dimensional embeddings
      with shape ``[embedding_dim]``.
    - **str**: Uses S3PRL in an isolated subprocess venv (e.g., "apc",
      "tera", "cpc"). Returns the last hidden state with shape
      ``[time_frames, embedding_dim]``.

    Args:
        audios: A list of Audio objects containing the audio signals.
            All audios must be mono.
        model: The model specification. Can be:
            - ``HFModel(path_or_uri="facebook/wav2vec2-base")``
            - ``SpeechBrainModel(path_or_uri="speechbrain/spkrec-ecapa-voxceleb")``
            - ``"apc"`` or ``"tera"`` (S3PRL model name)
        cache_dir: Path to cache model weights (HFModel only).
        device: Device for inference (default: auto-select).

    Returns:
        List of tensors containing the SSL embeddings for each audio.

    Raises:
        NotImplementedError: If the model type is not supported.
        ValueError: If audio is not mono or has wrong sampling rate.

    Examples:
        >>> # HuggingFace backend
        >>> audios = [Audio(filepath="sample.wav")]
        >>> model = HFModel(path_or_uri="facebook/wav2vec2-base", revision="main")
        >>> embeddings = extract_ssl_embeddings_from_audios(audios, model)
        >>> print(embeddings[0].shape)
        [13, 209, 768] ([# of Layers, Time Frames, Embedding Size])

        >>> # SpeechBrain backend
        >>> model = SpeechBrainModel(path_or_uri="speechbrain/spkrec-ecapa-voxceleb")
        >>> embeddings = extract_ssl_embeddings_from_audios(audios, model)
        >>> print(embeddings[0].shape)
        [192]

        >>> # S3PRL backend (isolated subprocess venv)
        >>> embeddings = extract_ssl_embeddings_from_audios(audios, "apc")
        >>> print(embeddings[0].shape)
        [time_frames, 512]
    """
    if isinstance(model, HFModel) and not isinstance(model, SpeechBrainModel):
        return SSLEmbeddingsFactory.extract_ssl_embeddings(
            audios=audios, model=model, cache_dir=cache_dir, device=device
        )
    elif isinstance(model, SpeechBrainModel):
        return SpeechBrainSSLEmbeddings.extract_speechbrain_ssl_embeddings(audios=audios, model=model, device=device)
    elif isinstance(model, str):
        from senselab.audio.tasks.ssl_embeddings.s3prl import S3PRLEmbeddingExtractor

        return S3PRLEmbeddingExtractor.extract_s3prl_embeddings(audios=audios, model_name=model, device=device)
    else:
        raise NotImplementedError(
            f"Model type {type(model).__name__} is not supported for SSL embedding extraction. "
            "Supported: HFModel (HuggingFace), SpeechBrainModel (SpeechBrain), or str (S3PRL model name)."
        )
