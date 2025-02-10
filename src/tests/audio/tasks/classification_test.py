"""Test audio classification APIs."""

import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.classification.speech_emotion_recognition import (
    classify_emotions_from_speech,
)
from senselab.audio.tasks.preprocessing import resample_audios
from senselab.utils.data_structures import HFModel
from tests.audio.conftest import MONO_AUDIO_PATH


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
def test_speech_emotion_recognition() -> None:
    """Tests speech emotion recognition."""
    audio_dataset = [Audio.from_filepath(MONO_AUDIO_PATH)]

    resampled_audios = resample_audios(audio_dataset, 16000)  # some pipelines resample for us but can't guarantee

    # Discrete test
    result = classify_emotions_from_speech(
        resampled_audios, HFModel(path_or_uri="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
    )
    emotions = result[0].get_labels()
    rav_emotions = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]

    for emotion in emotions:
        assert emotion in rav_emotions

    # Continuous test
    result = classify_emotions_from_speech(
        resampled_audios, HFModel(path_or_uri="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim")
    )
    labels = result[0].get_labels()

    for label in labels:
        assert label in ["arousal", "valence", "dominance"], "No emotion here but rather is one of \
            arousal, valence, or dominance"
