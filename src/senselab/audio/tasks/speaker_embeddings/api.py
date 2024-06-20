"""This module implements some utilities to extract speaker embeddings from a model."""

from typing import Optional

import pydra
from speechbrain.inference.speaker import EncoderClassifier
from torch import Tensor

from senselab.audio.data_structures.audio import Audio
from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.model import HFModel, SenselabModel


def extract_embeddings(audio: Audio, model: SenselabModel, device: Optional[DeviceType] = None) -> Tensor:
    """Compute the speaker embedding of the audio signal.

    Args:
        audio (Audio): An Audio object containing the audio signal and its properties.
        model (SenselabModel): The model used to compute the embeddings. Must be a Hugging Face model.
        device (Optional[DeviceType]): The device on which to run the model (e.g., 'cpu', 'cuda').
            Defaults to None, which means the default device will be used.

    Returns:
        Tensor: A tensor containing the speaker embeddings.

    Raises:
        NotImplementedError: If the model is not a Hugging Face model.

    Examples:
        >>> audio = Audio.from_filepath("sample.wav")
        >>> model = HFModel(path_or_uri="speechbrain/spkrec-ecapa-voxceleb")
        >>> embeddings = extract_embeddings(audio, model, device="cuda")
        >>> print(embeddings.shape)
        torch.Size([192])
    """
    if isinstance(model, HFModel):
        classifier = EncoderClassifier.from_hparams(source=model.path_or_uri, run_opts={"device": device})
        embeddings = classifier.encode_batch(audio.waveform)
        return embeddings.squeeze()
    else:
        raise NotImplementedError("Only Hugging Face models are supported for now.")


extract_embeddings_pt = pydra.mark.task(extract_embeddings)
