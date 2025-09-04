"""This module implements some utilities to extract speaker embeddings from an audio clip."""

from typing import List, Optional

import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speaker_embeddings.speechbrain import SpeechBrainEmbeddings
from senselab.utils.data_structures import DeviceType, SpeechBrainModel


def extract_speaker_embeddings_from_audios(
    audios: List[Audio],
    model: Optional[SpeechBrainModel] = None,
    device: Optional[DeviceType] = None,
) -> List[torch.Tensor]:
    """Compute fixed-dimensional speaker embeddings for a batch of `Audio` objects.

    By default, this uses a SpeechBrain speaker-recognition model
    (``speechbrain/spkrec-ecapa-voxceleb``, revision ``"main"``) to extract
    one embedding per input audio. The order of outputs matches the order of
    the inputs.

    Notes:
        - Many pretrained speaker models are trained on **mono, 16 kHz** audio.
          For best results, consider preprocessing inputs with
          `downmix_audios_to_mono(...)` and `resample_audios(..., 16000)`.
        - `device` must be a `DeviceType` (`CPU` or `CUDA`). If `None`,
          the backend will choose `CUDA` if available, otherwise `CPU`.
        - The embedding dimensionality depends on the model
          (e.g., ECAPA-TDNN often returns 192-D vectors).

    Args:
        audios (list[Audio]):
            Input audio objects to embed.
        model (SpeechBrainModel, optional):
            SpeechBrain model handle (path or HF Hub URI + revision).
            If ``None``, defaults to
            ``SpeechBrainModel(path_or_uri="speechbrain/spkrec-ecapa-voxceleb", revision="main")``.
        device (DeviceType, optional):
            Target device for inference:
              * ``DeviceType.CPU``
              * ``DeviceType.CUDA`` (GPU, if available)
            If ``None``, a default is chosen by the backend.

    Returns:
        list[torch.Tensor]: One 1-D tensor per input audio (e.g., shape ``[192]`` for ECAPA).

    Raises:
        NotImplementedError:
            If a non-SpeechBrain model is provided (only SpeechBrain is supported currently).

    Example (default model on CPU):
        >>> from pathlib import Path
        >>> from senselab.audio.data_structures import Audio
        >>> from senselab.utils.data_structures import DeviceType
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> a2 = Audio(filepath=Path("sample2.wav").resolve())
        >>> embs = extract_speaker_embeddings_from_audios([a1, a2], device=DeviceType.CPU)
        >>> embs[0].shape
        torch.Size([192])

    Example (explicit model on CUDA):
        >>> from pathlib import Path
        >>> from senselab.audio.data_structures import Audio
        >>> from senselab.utils.data_structures import DeviceType, SpeechBrainModel
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> model = SpeechBrainModel(path_or_uri="speechbrain/spkrec-ecapa-voxceleb", revision="main")
        >>> embs = extract_speaker_embeddings_from_audios([a1], model=model, device=DeviceType.CUDA)
        >>> len(embs), embs[0].ndim
        (1, 1)
    """
    if model is None:
        model = SpeechBrainModel(path_or_uri="speechbrain/spkrec-ecapa-voxceleb", revision="main")

    if isinstance(model, SpeechBrainModel):
        return SpeechBrainEmbeddings.extract_speechbrain_speaker_embeddings_from_audios(
            audios=audios, model=model, device=device
        )
    else:
        raise NotImplementedError(
            "Only SpeechBrain models are supported for now. We aim to support more models in the future."
        )
