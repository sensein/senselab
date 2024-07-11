"""Audio Processing and Speaker Verification Module.

This module provides functions for resampling audio using an IIR filter and
verifying if two audio samples or files are from the same speaker using a
specified model.
"""

from typing import Tuple

from torch.nn.functional import cosine_similarity

from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.speaker_embeddings.speechbrain import SpeechBrainEmbeddings
from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.model import SpeechBrainModel


def verify_speaker(
    audio1: Audio,
    audio2: Audio,
    model: SpeechBrainModel = SpeechBrainModel(path_or_uri="speechbrain/spkrec-ecapa-voxceleb", revision="main"),
    model_training_sample_rate: int = 16000,  # spkrec-ecapa-voxceleb trained on 16kHz audio
    device: DeviceType = DeviceType.CPU,
    threshold: float = 0.5,
) -> Tuple[float, bool]:
    """Verifies if two audio samples are from the same speaker.

    Args:
        audio1 (Audio): The first audio sample.
        audio2 (Audio): The second audio sample.
        model (SpeechBrainModel, optional): The model for speaker verification.
        model_training_sample_rate (int, optional): The sample rate the model trained on.
        device (DeviceType, optional): The device to run the model on. Defaults to CPU.
        threshold (float, optional): The threshold to determine same speaker.

    Returns:
        Tuple[float, bool]: A tuple containing the verification score and the prediction.
                            The verification score is a float indicating the similarity
                            between the two samples, and the prediction is a boolean
                            indicating if the two samples are from the same speaker.
    """
    if audio1.sampling_rate != model_training_sample_rate:
        raise ValueError(f"{model.path_or_uri} trained on {model_training_sample_rate} \
                            sample audio, but audio1 has sample rate {audio1.sampling_rate}.")
    if audio2.sampling_rate != model_training_sample_rate:
        raise ValueError(f"{model.path_or_uri} trained on {model_training_sample_rate} \
                         sample audio, but audio2 has sample rate {audio2.sampling_rate}.")

    embeddings = SpeechBrainEmbeddings.extract_speechbrain_speaker_embeddings_from_audios(
        audios=[audio1, audio2], model=model, device=device
    )
    embedding1, embedding2 = embeddings
    similarity = cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
    score = similarity.mean().item()
    prediction = score > threshold

    return score, prediction
