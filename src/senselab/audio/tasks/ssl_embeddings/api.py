"""This module implements some utilities to extract embeddings from self-supervised models."""

from typing import List, Optional

import pydra
import torch

from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.ssl_embeddings.self_supervised_features import SSLEmbeddingsFactory
from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.model import HFModel, SenselabModel


def extract_ssl_embeddings_from_audios(
    audios: List[Audio],
    model: SenselabModel,
    cache_dir: str = "~/",
    device: Optional[DeviceType] = None,
) -> List[torch.Tensor]:
    """Extract embedding of audio signals from pre-trained SSL models.

    Args:
        audios (List[Audio]): A list of Audio objects containing the audio signals and their properties.
        model (SenselabModel): The model used to extract their embeddings.
        cache_dir (str): The path to where the model's weights will be saved.
        device (Optional[DeviceType]): The device to run the model on (default is None).

    Returns:
        List[torch.Tensor]: A list of 1d tensors containing the ssl embeddings for each audio file.

    Raises:
        NotImplementedError: If the model is not a Hugging Face model.

    Examples:
        >>> audios = [Audio.from_filepath("sample.wav")]
        >>> model = HFModel(path_or_uri="facebook/wav2vec2-base", revision="main")
        >>> embeddings = extract_ssl_embeddings_from_audios(audios, model, cache_dir="./", device=DeviceType.CUDA)
        >>> print(embeddings[0].shape)
        [13, 209, 768] ([# of Layers, Time Frames, Embedding Size])

    Todo:
        - Make the API compatible with other models than Hugging Face.
    """
    if isinstance(model, HFModel):
        return SSLEmbeddingsFactory.extract_ssl_embeddings(
            audios=audios, model=model, cache_dir=cache_dir, device=device
        )
    else:
        raise NotImplementedError("The specified model is not supported for now.")


extract_ssl_embeddings_from_audios_pt = pydra.mark.task(extract_ssl_embeddings_from_audios)
