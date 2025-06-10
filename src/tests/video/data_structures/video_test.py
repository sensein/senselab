"""Tests for Video class."""

import os

import pytest

from senselab.audio.data_structures import Audio
from senselab.video.data_structures import Video

try:
    import av  # noqa: F401

    AV_AVAILABLE = True
except ModuleNotFoundError:
    AV_AVAILABLE = False


filepath = os.path.abspath("src/tests/data_for_testing/video_48khz_stereo_16bits.mp4")


@pytest.mark.skipif(AV_AVAILABLE, reason="AV is available.")
def test_video_import_error() -> None:
    """Test Video import error."""
    with pytest.raises(ModuleNotFoundError):
        Video(filepath=filepath).frames


@pytest.mark.skipif(not AV_AVAILABLE, reason="AV is not available.")
def test_constructor() -> None:
    """Test Video constructor by mocking read_video."""
    metadata = {"participant": "test_subject"}

    assert os.path.exists(filepath)

    video = Video(filepath=filepath, metadata=metadata)

    # Assertions
    assert isinstance(video, Video)
    assert isinstance(video.audio, Audio)
    assert video.metadata == metadata


@pytest.mark.skipif(not AV_AVAILABLE, reason="AV is not available.")
def test_constructor_wrong_filepath() -> None:
    """Test Video constructor with wrong filepath."""
    with pytest.raises(FileNotFoundError):
        Video(filepath="wrong_filepath")
