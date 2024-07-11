"""Test Module for Audio Processing and Speaker Verification.

This module contains minimal tests to ensure the audio processing and speaker verification functions do not fail.

Tests:
    - test_resample_iir: Tests the resample_iir function.
    - test_verify_speaker: Tests the verify_speaker function.
    - test_verify_speaker_from_files: Tests the verify_speaker_from_files function.
"""

import os

import pytest

from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.preprocessing.preprocessing import resample_audios
from senselab.audio.tasks.speaker_verification.speaker_verification import (
    verify_speaker,
)

if os.getenv("GITHUB_ACTIONS") != "true":

    @pytest.mark.large_model
    def test_verify_speaker(mono_audio_sample: Audio) -> None:
        """Tests the verify_speaker function to ensure it does not fail.

        Args:
            mono_audio_sample (Audio): The mono audio sample to use for testing.

        Returns:
            None
        """
        mono_audio_sample = resample_audios([mono_audio_sample], 16000)[0]
        mono_audio_samples = [(mono_audio_sample, mono_audio_sample)] * 3
        scores_and_predictions = verify_speaker(mono_audio_samples)
        assert scores_and_predictions
        assert len(scores_and_predictions[0]) == 2
        assert isinstance(scores_and_predictions[0][0], float)
        assert isinstance(scores_and_predictions[0][1], bool)
