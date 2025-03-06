"""Module for testing Audio data structures."""

import os
import tempfile
from typing import List

import numpy as np
import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.input_output import read_audios, save_audios
from tests.audio.conftest import MONO_AUDIO_PATH, STEREO_AUDIO_PATH

try:
    import torchaudio  # noqa: F401

    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False


@pytest.mark.skipif(
    TORCHAUDIO_AVAILABLE,
    reason="Torchaudio is available.",
)
def test_read_audios_torchaudio_not_installed() -> None:
    """Tests the read_audios function when torchaudio is not installed."""
    with pytest.raises(ImportError):
        read_audios(file_paths=[MONO_AUDIO_PATH], plugin="serial")


@pytest.mark.skipif(
    not TORCHAUDIO_AVAILABLE,
    reason="Torchaudio is not available.",
)
@pytest.mark.parametrize(
    "audio_paths",
    [
        ([MONO_AUDIO_PATH]),
        ([STEREO_AUDIO_PATH]),
        ([MONO_AUDIO_PATH, STEREO_AUDIO_PATH]),  # Test multiple files
    ],
)
def test_read_audios(audio_paths: List[str | os.PathLike]) -> None:
    """Tests the read_audios function with actual mono and stereo audio files."""
    # Run the function with real audio file paths
    processed_audios = read_audios(file_paths=audio_paths, plugin="serial")

    # Validate results
    assert len(processed_audios) == len(audio_paths), "Incorrect number of processed files."

    for idx, processed_audio in enumerate(processed_audios):
        # Load the same file directly using the Audio class for comparison
        reference_audio = Audio.from_filepath(audio_paths[idx])

        # Verify the processed Audio matches the reference
        assert torch.equal(
            processed_audio.waveform, reference_audio.waveform
        ), f"Waveform for file {audio_paths[idx]} does not match the expected."
        assert (
            processed_audio.sampling_rate == reference_audio.sampling_rate
        ), f"Sampling rate for file {audio_paths[idx]} does not match the expected."


@pytest.mark.skipif(
    not TORCHAUDIO_AVAILABLE,
    reason="Torchaudio is not available.",
)
def test_save_audios() -> None:
    """Test the `save_audios` function."""
    # Create temporary directory for saving audio files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock audio data with correct fields
        audio_mock_1 = Audio(
            waveform=np.array([0.1, 0.2, 0.3]),  # Replace with actual waveform data if needed
            sampling_rate=44100,
        )
        audio_mock_2 = Audio(waveform=np.array([0.4, 0.5, 0.6]), sampling_rate=44100)

        # Prepare tuples of Audio objects and target file paths
        audio_tuples = [
            (audio_mock_1, os.path.join(temp_dir, "audio1.wav")),
            (audio_mock_2, os.path.join(temp_dir, "audio2.wav")),
        ]

        # Call the `save_audios` function
        save_audios(audio_tuples=audio_tuples)

        # Assertions to verify files are saved
        for _, file_path in audio_tuples:
            assert os.path.exists(file_path), f"File {file_path} was not created."
            assert os.path.getsize(file_path) > 0, f"File {file_path} is empty."
