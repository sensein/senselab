"""This module implements some utilities to extract speaker embeddings from an audio clip."""

from typing import List, Optional

import torch

from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.speaker_embeddings.speechbrain import SpeechBrainEmbeddings
from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.model import SenselabModel, SpeechBrainModel


def extract_speaker_embeddings_from_audios(
    audios: List[Audio],
    model: SenselabModel = SpeechBrainModel(path_or_uri="speechbrain/spkrec-ecapa-voxceleb", revision="main"),
    device: Optional[DeviceType] = None,
) -> List[torch.Tensor]:
    """Compute the speaker embedding of audio signals.

    Args:
        audios (List[Audio]): A list of Audio objects containing the audio signals and their properties.
        model (SpeechBrainModel): The model used to compute the embeddings
            (default is "speechbrain/spkrec-ecapa-voxceleb").
        device (Optional[DeviceType]): The device to run the model on (default is None).

    Returns:
        List[torch.Tensor]: A list of 1d tensors containing the speaker embeddings for each audio file.

    Raises:
        NotImplementedError: If the model is not a Hugging Face model.

    Examples:
        >>> audios = [Audio.from_filepath("sample.wav")]
        >>> model = SpeechBrainModel(path_or_uri="speechbrain/spkrec-ecapa-voxceleb", revision="main")
        >>> embeddings = extract_speaker_embeddings_from_audios(audios, model, device=DeviceType.CUDA)
        >>> print(embeddings[0].shape)
        torch.Size([192])
    """
    if isinstance(model, SpeechBrainModel):
        return SpeechBrainEmbeddings.extract_speechbrain_speaker_embeddings_from_audios(
            audios=audios, model=model, device=device
        )
    else:
        raise NotImplementedError("The specified model is not supported for now.")
