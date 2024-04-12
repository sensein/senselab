"""This module tests the video's IOTask class."""

import os
from typing import Any, Dict, List

import pytest

from pipepal.video.tasks import IOTask as VideoIOTask


def test_extract_audios_from_videos_input_errors() -> None:
    """Test the extract_audios_from_videos method.
    
    This test checks if the extract_audios_from_videos method raises appropriate errors for invalid inputs.
    This tests:
    1. Missing 'service' or 'data' keys in the input dictionary.
    2. Invalid file paths in the 'files' list (non-existent files).
    """
    with pytest.raises(ValueError):
        # Missing 'service' key
        VideoIOTask().extract_audios_from_videos({
            "data": {
                "files": ["/path/to/audio/file1.mp4"],
                "output_folder": "/path/to/output/audios",
                "audio_format": "wav",
                "audio_codec": "pcm_s16le"
            }
        })

    with pytest.raises(ValueError):
        # Missing 'data' key
        VideoIOTask().extract_audios_from_videos({
            "service": {
                "service_name": "ffmpeg"
            }
        })

    with pytest.raises(FileNotFoundError):
        # Non-existent file path
        VideoIOTask().extract_audios_from_videos({
            "data": {
                "files": ["/non/existent/path/file1.mp4"],
                "output_folder": "./data_for_testing",
                "audio_format": "wav",
                "audio_codec": "pcm_s16le"
            },
            "service": {
                "service_name": "ffmpeg"
            }
        })

def test_extract_audios_from_videos_output_type() -> None:
    """Test the extract_audios_from_videos method to check if the output is of type list of strings."""
    test_input = {
        "data": {
                "files": ["./data_for_testing/video_48khz_stereo_16bits.mp4"],
                "output_folder": "./data_for_testing",
                "audio_format": "wav",
                "audio_codec": "pcm_s16le"
        },
        "service": {
            "service_name": "ffmpeg"
        }
    }
    response = VideoIOTask().extract_audios_from_videos(test_input)
    assert isinstance(response["output"], list), "Output should be a list"
    assert all(isinstance(item, str) for item in response["output"]), "All items in output list should be strings"

    # Clean up
    for audio_file in response["output"]:
        os.remove(audio_file)

def test_read_audios_from_disk_output_dimensions() -> None:
    """Test the read_audios_from_disk method.
     
    This test checks if the dimensions of the output list of audio files match the input list of video files.
    """
    test_input: Dict[str, Any] = {
        "data": {
            "files": ["./data_for_testing/video_48khz_stereo_16bits.mp4"],
            "output_folder": "./data_for_testing",
            "audio_format": "wav",
            "audio_codec": "pcm_s16le"
        },
        "service": {
            "service_name": "ffmpeg"
        }
    }
    response: Dict[str, List[str]] = VideoIOTask().extract_audios_from_videos(test_input)
    assert len(response["output"]) == len(test_input["data"]["files"]), "The number of items in the output should match the number of input files."

    # Clean up
    for audio_file in response["output"]:
        os.remove(audio_file)
