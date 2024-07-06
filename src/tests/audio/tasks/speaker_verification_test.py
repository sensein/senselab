"""Test Module for Audio Processing and Speaker Verification.

This module contains minimal tests to ensure the audio processing and speaker verification functions do not fail.

Tests:
    - test_resample_iir: Tests the resample_iir function.
    - test_verify_speaker: Tests the verify_speaker function.
    - test_verify_speaker_from_files: Tests the verify_speaker_from_files function.
"""

import os

import pytest
import torch

from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.speaker_verification.speaker_verification import (
    _resample_iir,
    verify_speaker,
    verify_speaker_from_files,
)

MONO_AUDIO_PATH = "src/tests/data_for_testing/audio_48khz_mono_16bits.wav"


@pytest.fixture
def mono_audio_sample() -> Audio:
    """Fixture for sample mono audio."""
    return Audio.from_filepath(MONO_AUDIO_PATH)


def test_resample_iir() -> None:
    """Tests the resample_iir function to ensure it does not fail.

    Returns:
        None
    """
    audio = Audio(waveform=torch.rand(1, 16000), sampling_rate=16000)
    lowcut = 100.0
    new_sample_rate = 8000
    _resample_iir(audio, lowcut, new_sample_rate)


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

    def test_verify_speaker_from_files() -> None:
        """Tests the verify_speaker_from_files function to ensure it does not fail.

        Returns:
            None
        """
        score, prediction = verify_speaker_from_files(MONO_AUDIO_PATH, MONO_AUDIO_PATH)
        assert prediction
