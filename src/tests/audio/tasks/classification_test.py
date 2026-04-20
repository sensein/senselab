"""Test audio classification APIs."""

import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.classification.speech_emotion_recognition import (
    classify_emotions_from_speech,
)
from senselab.audio.tasks.preprocessing import resample_audios
from senselab.utils.data_structures import DeviceType, HFModel
from tests.audio.conftest import MONO_AUDIO_PATH


def test_speech_emotion_recognition_continuous(cpu_cuda_device: DeviceType) -> None:
    """Tests continuous speech emotion recognition (Tier 2: ~661MB model)."""
    audio_dataset = [Audio(filepath=MONO_AUDIO_PATH)]
    resampled = resample_audios(audio_dataset, 16000)

    result = classify_emotions_from_speech(
        resampled, HFModel(path_or_uri="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"), device=cpu_cuda_device
    )
    labels = result[0].get_labels()

    for label in labels:
        assert label in ["arousal", "valence", "dominance"], (
            "No emotion here but rather is one of arousal, valence, or dominance"
        )


def test_speech_emotion_recognition_discrete(gpu_device: DeviceType) -> None:
    """Tests discrete speech emotion recognition (Tier 3: ~1.27GB model)."""
    audio_dataset = [Audio(filepath=MONO_AUDIO_PATH)]
    resampled = resample_audios(audio_dataset, 16000)

    result = classify_emotions_from_speech(
        resampled, HFModel(path_or_uri="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"), device=gpu_device
    )
    emotions = result[0].get_labels()
    rav_emotions = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]

    for emotion in emotions:
        assert emotion in rav_emotions
