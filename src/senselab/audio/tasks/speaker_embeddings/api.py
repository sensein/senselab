"""This module implements some utilities to extract speaker embeddings from a model.

Speaker embeddings are fixed-dimensional vector representations that capture the unique characteristics of a speaker's
voice, allowing for tasks such as speaker identification, verification, and diarization.

### Task Overview:
Speaker embedding extraction is a crucial task in speaker recognition systems. It involves transforming variable-length
audio signals into fixed-size vector representations that encapsulate speaker-specific information while being robust
to variations in speech content, background noise, and recording conditions.

### Model Architecture:
The default model used in this module (speechbrain/spkrec-ecapa-voxceleb) is based on the ECAPA-TDNN architecture,
which has shown strong performance across various speaker recognition tasks.
Other supported models include ResNet TDNN (speechbrain/spkrec-resnet-voxceleb) and
xvector (speechbrain/spkrec-xvect-voxceleb).

**Note**: Performance can vary significantly depending on the specific dataset, task, and evaluation protocol used.
Always refer to the most recent literature for up-to-date benchmarks.

### Learn more:
- [SpeechBrain](https://speechbrain.github.io/)
- [ECAPA-TDNN](https://arxiv.org/abs/2005.07143)
- [ResNet TDNN](https://doi.org/10.1016/j.csl.2019.101026)
- [xvector](https://doi.org/10.21437/Odyssey.2018-15)
"""

from typing import List, Optional

import pydra
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


extract_speaker_embeddings_from_audios_pt = pydra.mark.task(extract_speaker_embeddings_from_audios)
