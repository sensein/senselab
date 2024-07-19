"""Test audio classification APIs."""

import os

import pytest

from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.classification.speech_emotion_recognition import (
    audio_classification_with_hf_models,
    speech_emotion_recognition_with_hf_models,
)
from senselab.utils.data_structures.model import HFModel

if os.getenv("GITHUB_ACTIONS") != "true":

    @pytest.fixture
    def valid_model() -> HFModel:
        """Fixture for generating a valid HFModel."""
        return HFModel(path_or_uri="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition", revision="main")

    def test_audio_classification_with_hf_models(resampled_mono_audio_sample: Audio) -> None:
        """Tests the audio classification functionality with HuggingFace models.

        This test uses a real HuggingFace model and pipeline to classify a dummy audio sample.
        It verifies that the classification function processes the input correctly and returns
        the expected output.

        Args:
            resampled_mono_audio_sample: A fixture that provides a dummy Audio object.
        """
        # Real model
        model = HFModel(path_or_uri="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition", revision="main")

        # Run the classification function
        result = audio_classification_with_hf_models([resampled_mono_audio_sample], model)

        # Verify the result
        assert len(result) == 1
        assert isinstance(result[0], list)
        assert len(result[0]) > 0  # Ensure there's at least one classification result
        assert isinstance(result[0][0], dict)
        assert "label" in result[0][0]
        assert "score" in result[0][0]

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

    def test_speech_emotion_recognition_stereo_raises_value_error(resampled_stereo_audio_sample: Audio) -> None:
        """Tests that speech emotion recognition raises ValueError with stereo audio samples."""
        resampled_stereo_audio_samples = [resampled_stereo_audio_sample]

        with pytest.raises(ValueError, match="We expect a single channel audio input for AudioClassificationPipeline"):
            speech_emotion_recognition_with_hf_models(
                resampled_stereo_audio_samples,
                HFModel(path_or_uri="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"),
            )

    def test_batch_processing_consistency(resampled_mono_audio_sample: Audio, valid_model: HFModel) -> None:
        """Test batch processing consistency for different batch sizes."""
        audios = [resampled_mono_audio_sample] * 3  # Duplicate the sample to create a list
        result_batch_1 = audio_classification_with_hf_models(audios, valid_model, batch_size=1)
        result_batch_5 = audio_classification_with_hf_models(audios, valid_model, batch_size=5)
        result_batch_10 = audio_classification_with_hf_models(audios, valid_model, batch_size=10)
        assert len(result_batch_1) == len(result_batch_10) == len(result_batch_5)

    def test_speech_emotion_recognition_with_correct_labels(
        resampled_mono_audio_sample: Audio, valid_model: HFModel
    ) -> None:
        """Test that the emotion recognition output contains expected emotion labels."""
        result = speech_emotion_recognition_with_hf_models([resampled_mono_audio_sample], valid_model)
        assert len(result) == 1
        assert isinstance(result[0], tuple)
        assert isinstance(result[0][0], str)
        assert isinstance(result[0][1], dict)

        expected_emotions = ["happy", "sad", "neutral", "positive", "negative", "anger", "disgust", "fear"]
        for emotion in expected_emotions:
            if emotion in result[0][1]:
                break
        else:
            pytest.fail("None of the expected emotion labels found in the output.")
