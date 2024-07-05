"""Test Module for Audio Processing and Speaker Verification.

This module contains minimal tests to ensure the audio processing and speaker verification functions do not fail.

Tests:
    - test_resample_iir: Tests the resample_iir function.
    - test_verify_speaker: Tests the verify_speaker function.
    - test_verify_speaker_from_files: Tests the verify_speaker_from_files function.
"""

from pathlib import Path

import torch

from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.speaker_verification.speaker_verification import (
    _resample_iir,
    verify_speaker,
    verify_speaker_from_files,
)


def test_resample_iir() -> None:
    """Tests the resample_iir function to ensure it does not fail.

    Returns:
        None
    """
    audio = Audio(waveform=torch.rand(1, 16000), sampling_rate=16000)
    lowcut = 100.0
    new_sample_rate = 8000
    _resample_iir(audio, lowcut, new_sample_rate)


def test_verify_speaker() -> None:
    """Tests the verify_speaker function to ensure it does not fail.

    Returns:
        None
    """
    audio1 = Audio(waveform=torch.rand(1, 16000), sampling_rate=16000)
    audio2 = Audio(waveform=torch.rand(1, 16000), sampling_rate=16000)
    model = "some_model_path"
    model_rate = 16000
    verify_speaker(audio1, audio2, model, model_rate)


def test_verify_speaker_from_files() -> None:
    """Tests the verify_speaker_from_files function to ensure it does not fail.

    Returns:
        None
    """
    file1 = Path("path/to/audio1.wav")
    file2 = Path("path/to/audio2.wav")
    model = "some_model_path"
    verify_speaker_from_files(file1, file2, model)
