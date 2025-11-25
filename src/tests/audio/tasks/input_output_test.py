"""Module for testing Audio data structures and input/output utilities."""

import os
import tempfile
from pathlib import Path
from typing import Generator, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from pytest import LogCaptureFixture

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.input_output import (
    get_audio_files_from_directory,
    read_audios,
    save_audios,
    validate_audio_paths,
)
from tests.audio.conftest import MONO_AUDIO_PATH, STEREO_AUDIO_PATH

try:
    import torchaudio  # noqa: F401

    TORCHAUDIO_AVAILABLE = True
except ModuleNotFoundError:
    TORCHAUDIO_AVAILABLE = False


@pytest.mark.skipif(
    TORCHAUDIO_AVAILABLE,
    reason="Torchaudio is available.",
)
def test_read_audios_torchaudio_not_installed() -> None:
    """Tests the read_audios function when torchaudio is not installed."""
    with pytest.raises(ModuleNotFoundError):
        audios = read_audios(file_paths=[MONO_AUDIO_PATH])
        audios[0].waveform


@pytest.mark.parametrize(
    "audio_paths",
    [
        ([MONO_AUDIO_PATH]),
        ([STEREO_AUDIO_PATH]),
        ([MONO_AUDIO_PATH, STEREO_AUDIO_PATH]),  # Test multiple files
    ],
)
@patch("torchaudio.load")  # Mock torchaudio.load
@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed.")
def test_read_audio_lazy_loading(mock_torchaudio_load: MagicMock, audio_paths: List[str | os.PathLike]) -> None:
    """Test lazy audio loading by mocking torchaudio.load."""
    # Mock `torchaudio.load` to return a fake waveform tensor and sample rate
    fake_waveform = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    fake_sample_rate = 48000
    mock_torchaudio_load.return_value = (fake_waveform, fake_sample_rate)

    # audio_paths = [MONO_AUDIO_PATH, STEREO_AUDIO_PATH]

    processed_audios = read_audios(audio_paths)
    mock_torchaudio_load.assert_not_called()

    for idx, processed_audio in enumerate(processed_audios):
        _ = processed_audio.waveform
        mock_torchaudio_load.assert_called_with(audio_paths[idx], frame_offset=0, num_frames=-1, backend=None)

        _ = processed_audio.waveform
        mock_torchaudio_load.call_count == (idx + 1)


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
    processed_audios = read_audios(file_paths=audio_paths)

    # Validate results
    assert len(processed_audios) == len(audio_paths), "Incorrect number of processed files."

    for idx, processed_audio in enumerate(processed_audios):
        # Load the same file directly using the Audio class for comparison
        reference_audio = Audio(filepath=audio_paths[idx])

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


# Tests for get_audio_files_from_directory
@pytest.fixture
def sample_audio_directory() -> Generator[Path, None, None]:
    """Create a temporary directory with sample audio files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create various audio files
        (temp_path / "test1.wav").touch()
        (temp_path / "test2.WAV").touch()  # uppercase extension
        (temp_path / "test3.mp3").touch()
        (temp_path / "test4.flac").touch()
        (temp_path / "test5.m4a").touch()
        (temp_path / "not_audio.txt").touch()  # non-audio file

        # Create subdirectory with more files
        sub_dir = temp_path / "subdir"
        sub_dir.mkdir()
        (sub_dir / "nested1.wav").touch()
        (sub_dir / "nested2.ogg").touch()
        (sub_dir / "nested3.aac").touch()

        yield temp_path


def test_get_audio_files_from_directory_default(
    sample_audio_directory: Path,
) -> None:
    """Test default behavior of get_audio_files_from_directory."""
    audio_files = get_audio_files_from_directory(str(sample_audio_directory))

    # Should find all audio files (recursive by default)
    assert len(audio_files) == 8  # 5 in root + 3 in subdir

    # Check that paths are strings
    assert all(isinstance(f, str) for f in audio_files)

    # Check that files are sorted
    assert audio_files == sorted(audio_files)

    # Check that various formats are included
    extensions = {Path(f).suffix.lower() for f in audio_files}
    expected_extensions = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}
    assert extensions == expected_extensions

    # Check that non-audio file is excluded
    assert not any("not_audio.txt" in f for f in audio_files)


def test_get_audio_files_from_directory_non_recursive(
    sample_audio_directory: Path,
) -> None:
    """Test non-recursive search."""
    audio_files = get_audio_files_from_directory(str(sample_audio_directory), recursive=False)

    # Should only find files in root directory
    assert len(audio_files) == 5

    # Should not include files from subdirectory
    assert not any("subdir" in f for f in audio_files)


def test_get_audio_files_from_directory_custom_extensions(
    sample_audio_directory: Path,
) -> None:
    """Test with custom audio extensions."""
    audio_files = get_audio_files_from_directory(str(sample_audio_directory), audio_extensions={".wav", ".mp3"})

    # Should only find .wav and .mp3 files
    assert len(audio_files) == 4  # 3 wav (including uppercase) + 1 mp3

    extensions = {Path(f).suffix.lower() for f in audio_files}
    assert extensions == {".wav", ".mp3"}


def test_get_audio_files_from_directory_case_insensitive(
    sample_audio_directory: Path,
) -> None:
    """Test that extension matching is case-insensitive."""
    audio_files = get_audio_files_from_directory(str(sample_audio_directory), audio_extensions={".wav"})

    # Should find both .wav and .WAV files
    wav_files = [f for f in audio_files if Path(f).suffix.lower() == ".wav"]
    assert len(wav_files) == 3  # test1.wav + test2.WAV + nested1.wav


def test_get_audio_files_from_directory_nonexistent(
    caplog: LogCaptureFixture,
) -> None:
    """Test behavior with nonexistent directory."""
    audio_files = get_audio_files_from_directory("/nonexistent/path")

    assert audio_files == []

    # Check warning message (logger output captured by caplog)
    assert "Directory /nonexistent/path does not exist" in caplog.text


def test_get_audio_files_from_directory_file_not_dir(sample_audio_directory: Path, caplog: LogCaptureFixture) -> None:
    """Test behavior when path points to a file, not directory."""
    file_path = sample_audio_directory / "test1.wav"
    audio_files = get_audio_files_from_directory(str(file_path))

    assert audio_files == []

    # Check warning message (logger output captured by caplog)
    assert f"{file_path} is not a directory" in caplog.text


def test_get_audio_files_from_directory_empty_directory() -> None:
    """Test behavior with empty directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_files = get_audio_files_from_directory(temp_dir)
        assert audio_files == []


def test_get_audio_files_output_format(sample_audio_directory: Path, caplog: LogCaptureFixture) -> None:
    """Test the output format and logging."""
    audio_files = get_audio_files_from_directory(str(sample_audio_directory))

    # Check console output (logger output captured by caplog)
    assert f"Found {len(audio_files)} audio files" in caplog.text
    assert "File types found:" in caplog.text

    # Should list the extensions found
    assert ".wav" in caplog.text
    assert ".mp3" in caplog.text


def test_get_audio_files_from_directory_all_default_extensions() -> None:
    """Test that all default audio extensions can be detected."""
    # All default extensions from the function signature
    default_extensions = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".wma"}

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create one file for each default extension
        for ext in default_extensions:
            (temp_path / f"test{ext}").touch()

        # Also create a non-audio file to ensure it's excluded
        (temp_path / "not_audio.txt").touch()

        # Get audio files using default extensions
        audio_files = get_audio_files_from_directory(str(temp_path))

        # Should find all 7 audio files
        assert len(audio_files) == len(default_extensions)

        # Verify all extensions are found
        extensions_found = {Path(f).suffix.lower() for f in audio_files}
        assert extensions_found == default_extensions

        # Verify non-audio file is excluded
        assert not any("not_audio.txt" in f for f in audio_files)


@pytest.mark.parametrize(
    "extension",
    [".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".wma"],
)
def test_get_audio_files_from_directory_each_default_extension(extension: str) -> None:
    """Test that each default audio extension can be individually detected."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a file with the specific extension
        test_file = temp_path / f"test{extension}"
        test_file.touch()

        # Create a file with uppercase extension to test case-insensitivity
        test_file_upper = temp_path / f"test_upper{extension.upper()}"
        test_file_upper.touch()

        # Create a non-matching audio file to verify exclusion works
        # Use .wav as the "other" extension unless we're testing .wav, then use .mp3
        other_ext = ".wav" if extension != ".wav" else ".mp3"
        (temp_path / f"other{other_ext}").touch()

        # Get audio files with only this extension
        audio_files = get_audio_files_from_directory(str(temp_path), audio_extensions={extension})

        # Should find both lowercase and uppercase versions
        assert len(audio_files) == 2

        # Verify both files are found
        file_names = {Path(f).name for f in audio_files}
        assert f"test{extension}" in file_names
        assert f"test_upper{extension.upper()}" in file_names

        # Verify other extension is excluded
        assert not any(f"other{other_ext}" in f for f in audio_files)

        # Verify all found files have the correct extension (case-insensitive)
        for f in audio_files:
            assert Path(f).suffix.lower() == extension.lower()


# Tests for validate_audio_paths
def test_validate_audio_paths_valid_files(tmp_path: Path) -> None:
    """Test validate_audio_paths with valid files."""
    # Create some test files
    file1 = tmp_path / "test1.wav"
    file2 = tmp_path / "test2.mp3"
    file1.touch()
    file2.touch()

    valid_paths = validate_audio_paths([str(file1), str(file2)])
    assert len(valid_paths) == 2
    assert str(file1) in valid_paths
    assert str(file2) in valid_paths


def test_validate_audio_paths_nonexistent(caplog: LogCaptureFixture, tmp_path: Path) -> None:
    """Test validate_audio_paths with nonexistent files."""
    file1 = tmp_path / "test1.wav"
    file1.touch()
    nonexistent = tmp_path / "nonexistent.wav"

    valid_paths = validate_audio_paths([str(file1), str(nonexistent)], raise_on_empty=False)
    assert len(valid_paths) == 1
    assert str(file1) in valid_paths
    assert "Audio file does not exist" in caplog.text


def test_validate_audio_paths_directory(caplog: LogCaptureFixture, tmp_path: Path) -> None:
    """Test validate_audio_paths with a directory path."""
    file1 = tmp_path / "test1.wav"
    file1.touch()
    subdir = tmp_path / "subdir"
    subdir.mkdir()

    valid_paths = validate_audio_paths([str(file1), str(subdir)], raise_on_empty=False)
    assert len(valid_paths) == 1
    assert str(file1) in valid_paths
    assert "Path is not a file" in caplog.text


def test_validate_audio_paths_empty_raises(tmp_path: Path) -> None:
    """Test validate_audio_paths raises when no valid paths and raise_on_empty=True."""
    nonexistent = tmp_path / "nonexistent.wav"

    with pytest.raises(ValueError, match="No valid audio files found"):
        validate_audio_paths([str(nonexistent)], raise_on_empty=True)


def test_validate_audio_paths_empty_no_raise(tmp_path: Path) -> None:
    """Test validate_audio_paths returns empty list when raise_on_empty=False."""
    nonexistent = tmp_path / "nonexistent.wav"

    valid_paths = validate_audio_paths([str(nonexistent)], raise_on_empty=False)
    assert valid_paths == []
