"""Tests for quality control utility functions."""

import tempfile
from pathlib import Path
from typing import Generator

import pytest
from pytest import LogCaptureFixture

from senselab.audio.tasks.quality_control.utils import (
    get_audio_files_from_directory,
)


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
