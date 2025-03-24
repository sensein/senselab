"""This script contains unit tests for the filter_by_word_count method."""

from typing import Any

import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.filter_by_word_count import is_num_words_in_range
from senselab.audio.tasks.speech_to_text.api import transcribe_audios
from senselab.utils.data_structures import HFModel


def test_if_can_get_num_words(resampled_mono_audio_sample: Audio) -> None:
    """Tests if the number of words in the audio still evaluates to 19.

    Args:
        resampled_mono_audio_sample (Audio): The mono audio sample to use for testing.

    Returns: None

    """
    model = HFModel(path_or_uri="openai/whisper-tiny")
    transcript = transcribe_audios([resampled_mono_audio_sample], model)[0]
    assert transcript.text
    num_words = len(transcript.text.split())
    assert num_words == 19


def test_if_range_too_high(resampled_mono_audio_sample: Audio) -> None:
    """Tests if is_num_words_in_range returns False when number of words is below all the values in the range.

    Args:
        resampled_mono_audio_sample (Audio): The mono audio sample to use for testing.

    Returns: None

    """
    model = HFModel(path_or_uri="openai/whisper-tiny")
    range = (25, 30)
    assert not is_num_words_in_range(resampled_mono_audio_sample, range, model)


def test_if_range_too_low(resampled_mono_audio_sample: Audio) -> None:
    """Tests if is_num_words_in_range returns False when number of words is above all the values in the range.

    Args:
        resampled_mono_audio_sample (Audio): The mono audio sample to use for testing.

    Returns: None

    """
    model = HFModel(path_or_uri="openai/whisper-tiny")
    range = (5, 12)
    assert not is_num_words_in_range(resampled_mono_audio_sample, range, model)


def test_if_range_just_right(resampled_mono_audio_sample: Audio) -> None:
    """Tests if is_num_words_in_range returns True when the actual number of words is in the range.

    Args:
        resampled_mono_audio_sample (Audio): The mono audio sample to use for testing.

    Returns:
        None

    """
    model = HFModel(path_or_uri="openai/whisper-tiny")
    range = (15, 25)
    assert is_num_words_in_range(resampled_mono_audio_sample, range, model)
