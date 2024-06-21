"""This module implements some utilities to extract speaker embeddings from a model."""

from typing import List, Optional

import pydra
from torch import Tensor

from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.speaker_embeddings.speechbrain import SpeechBrainEmbeddings
from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.model import HFModel, SenselabModel


def extract_speaker_embeddings_from_audios(
    audios: List[Audio], model: SenselabModel, device: Optional[DeviceType] = None
) -> Tensor:
    """Compute the speaker embedding of audio signals.

    Args:
        audios (List[Audio]): A list of Audio objects containing the audio signals and their properties.
        model (SenselabModel): The model used to compute the embeddings. Must be a Hugging Face model.
        device (Optional[DeviceType]): The device on which to run the model (e.g., 'cpu', 'cuda').
            Defaults to None, which means the default device will be used.

    Returns:
        List[Tensor]: A list of tensors containing the speaker embeddings for each audio file.

    Raises:
        NotImplementedError: If the model is not a Hugging Face model.

    Examples:
        >>> audios = [Audio.from_filepath("sample.wav")]
        >>> model = HFModel(path_or_uri="speechbrain/spkrec-ecapa-voxceleb")
        >>> embeddings = extract_speaker_embeddings_from_audios(audio, model, device=DeviceType.CUDA)
        >>> print(embeddings.shape)
        torch.Size([192])
    """
    if isinstance(model, HFModel):
        return SpeechBrainEmbeddings.extract_speaker_embeddings_from_audios(audios, model, device)
    else:
        raise NotImplementedError("Only Hugging Face models are supported for now.")


extract_speaker_embeddings_from_audios_pt = pydra.mark.task(extract_speaker_embeddings_from_audios)
