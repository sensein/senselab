"""Tests for Video class."""

import os

import pytest

from senselab.audio.data_structures import Audio
from senselab.video.data_structures import Video
from senselab.video.data_structures.video import TORCHCODEC_AVAILABLE

filepath = os.path.abspath("src/tests/data_for_testing/video_48khz_stereo_16bits.mp4")


@pytest.mark.skip(reason="av is always installed in test environment")
def test_video_import_error() -> None:
    """Test Video import error."""
    with pytest.raises(ModuleNotFoundError):
        Video(filepath=filepath).frames


@pytest.mark.skipif(not TORCHCODEC_AVAILABLE, reason="torchcodec not available.")
def test_constructor() -> None:
    """Test Video constructor by mocking read_video."""
    metadata = {"participant": "test_subject"}

    assert os.path.exists(filepath)

    video = Video(filepath=filepath, metadata=metadata)

    # Assertions
    assert isinstance(video, Video)
    assert isinstance(video.audio, Audio)
    assert video.metadata == metadata


@pytest.mark.skipif(not TORCHCODEC_AVAILABLE, reason="torchcodec not available.")
def test_constructor_wrong_filepath() -> None:
    """Test Video constructor with wrong filepath."""
    with pytest.raises(FileNotFoundError):
        Video(filepath="wrong_filepath")
