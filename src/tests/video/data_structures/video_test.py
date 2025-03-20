"""Tests for Video class."""

import os

import pytest

from senselab.audio.data_structures import Audio
from senselab.video.data_structures import Video

try:
    import av  # noqa: F401

    PYAV_AVAILABLE = True
except ModuleNotFoundError:
    PYAV_AVAILABLE = False


filepath = os.path.abspath("src/tests/data_for_testing/video_48khz_stereo_16bits.mp4")


@pytest.mark.skipif(PYAV_AVAILABLE, reason="PyAV is available.")
def test_video_import_error() -> None:
    """Test Video import error."""
    with pytest.raises(ModuleNotFoundError):
        Video.from_filepath(filepath)


@pytest.mark.skipif(not PYAV_AVAILABLE, reason="PyAV is not available.")
def test_from_filepath() -> None:
    """Test Video.from_filepath by mocking read_video."""
    metadata = {"participant": "test_subject"}

    assert os.path.exists(filepath)

    video = Video.from_filepath(filepath, metadata)

    # Assertions
    assert isinstance(video, Video)
    assert isinstance(video.audio, Audio)
    assert video.orig_path_or_id == filepath
    assert video.metadata == metadata


@pytest.mark.skipif(not PYAV_AVAILABLE, reason="PyAV is not available.")
def test_from_filepath_wrong_filepath() -> None:
    """Test Video.from_filepath with wrong filepath."""
    with pytest.raises(FileNotFoundError):
        Video.from_filepath("wrong_filepath")
