"""Test audio classification APIs."""

import os

from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.classification.speech_emotion_recognition import speech_emotion_recognition_with_hf_models
from senselab.utils.data_structures.model import HFModel

if os.getenv("GITHUB_ACTIONS") != "true":

    def test_speech_emotion_recognition(resampled_mono_audio_sample: Audio) -> None:
        """Tests speech emotion recognition."""
        # Discrete test
        resampled_mono_audio_samples = [resampled_mono_audio_sample]
        result = speech_emotion_recognition_with_hf_models(
            resampled_mono_audio_samples,
            HFModel(path_or_uri="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"),
        )
        top_emotion, emotion_probs = result[0]
        rav_emotions = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
        assert top_emotion in rav_emotions, "Top emotion should be in RAVDESS Dataset"

        for emotion in emotion_probs:
            assert emotion in rav_emotions

        # Continuous test
        result = speech_emotion_recognition_with_hf_models(
            resampled_mono_audio_samples, HFModel(path_or_uri="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim")
        )
        emotion, continuous_values = result[0]
        assert emotion in ["arousal", "valence", "dominance"], "No emotion here but rather is one of \
            arousal, valence, or dominance"
        assert set(continuous_values.keys()) == set(["arousal", "valence", "dominance"])

# TODO add tests
