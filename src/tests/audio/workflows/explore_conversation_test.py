"""Unit tests for the explore_conversation utility.

These tests focus on verifying:
- Output structure
- Argument handling
- Edge cases
"""

import pytest

from senselab.audio.workflows.explore_conversation import explore_conversation


def test_explore_conversation_empty_input() -> None:
    """Test explore_conversation with an empty list of audio files.

    Should return an empty list.
    """
    result = explore_conversation(audio_file_paths=[])
    assert isinstance(result, list)
    assert len(result) == 0


def test_explore_conversation_invalid_file() -> None:
    """Test explore_conversation with invalid file paths."""
    with pytest.raises(FileNotFoundError):
        explore_conversation(audio_file_paths=["nonexistent.wav"])


def test_explore_conversation_default_args() -> None:
    """Test that default arguments are handled correctly."""
    result = explore_conversation(
        audio_file_paths=["src/tests/data_for_testing/english_conversation_higgs_audio_v2.wav"]
    )
    assert isinstance(result, list)
    assert len(result) == 1
