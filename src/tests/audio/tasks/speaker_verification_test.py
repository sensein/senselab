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
from senselab.audio.tasks.speaker_verification.speaker_verification import (
    verify_speaker,
)

MONO_AUDIO_PATH = "src/tests/data_for_testing/audio_48khz_mono_16bits.wav"


@pytest.fixture
def mono_audio_sample() -> Audio:
    """Fixture for sample mono audio."""
    return Audio.from_filepath(MONO_AUDIO_PATH)


if os.getenv("GITHUB_ACTIONS") != "true":

    @pytest.mark.large_model
    def test_verify_speaker(mono_audio_sample: Audio) -> None:
        """Tests the verify_speaker function to ensure it does not fail.

        Args:
            mono_audio_sample (Audio): The mono audio sample to use for testing.

        Returns:
            None
        """
        score, prediction = verify_speaker(mono_audio_sample, mono_audio_sample)
        assert prediction
