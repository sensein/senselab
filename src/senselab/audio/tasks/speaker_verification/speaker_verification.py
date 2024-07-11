"""Audio Processing and Speaker Verification Module.

This module provides functions for resampling audio using an IIR filter and
verifying if two audio samples or files are from the same speaker using a
specified model.
"""

from typing import List, Optional, Tuple

from torch.nn.functional import cosine_similarity

from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.speaker_embeddings.speechbrain import SpeechBrainEmbeddings
from senselab.utils.data_structures.device import DeviceType, _select_device_and_dtype
from senselab.utils.data_structures.model import SpeechBrainModel

TRAINING_SAMPLE_RATE = 16000  # spkrec-ecapa-voxceleb trained on 16kHz audio


def verify_speaker(
    audios: List[Tuple[Audio, Audio]],
    model: SpeechBrainModel = SpeechBrainModel(path_or_uri="speechbrain/spkrec-ecapa-voxceleb", revision="main"),
    device: Optional[DeviceType] = None,
    threshold: float = 0.25,
) -> List[Tuple[float, bool]]:
    """Verifies if two audio samples are from the same speaker.

    Args:
        audios (List[Tuple[Audio, Audio]]): A list of tuples, where each tuple contains
                                            two audio samples to be compared.
        model (SpeechBrainModel, optional): The model for speaker verification.
        device (DeviceType, optional): The device to run the model on. Defaults to CPU.
        threshold (float, optional): The threshold to determine same speaker.

    Returns:
        List[Tuple[float, bool]]: A list of tuples containing the verification score and
                                  the prediction for each pair of audio samples. The
                                  verification score is a float indicating the similarity
                                  between the two samples, and the prediction is a boolean
                                  indicating if the two samples are from the same speaker.
    """
    device = _select_device_and_dtype(compatible_devices=[DeviceType.CPU, DeviceType.CUDA])[0]

    scores_and_predictions = []
    for audio1, audio2 in audios:
        if audio1.sampling_rate != TRAINING_SAMPLE_RATE:
            raise ValueError(f"{model.path_or_uri} trained on {TRAINING_SAMPLE_RATE} \
                                sample audio, but audio1 has sample rate {audio1.sampling_rate}.")
        if audio2.sampling_rate != TRAINING_SAMPLE_RATE:
            raise ValueError(f"{model.path_or_uri} trained on {TRAINING_SAMPLE_RATE} \
                            sample audio, but audio2 has sample rate {audio2.sampling_rate}.")

        embeddings = SpeechBrainEmbeddings.extract_speechbrain_speaker_embeddings_from_audios(
            audios=[audio1, audio2], model=model, device=device
        )
        embedding1, embedding2 = embeddings
        similarity = cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
        score = similarity.mean().item()
        prediction = score > threshold
        scores_and_predictions.append((score, prediction))
    return scores_and_predictions
