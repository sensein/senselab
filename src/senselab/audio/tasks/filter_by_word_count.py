"""This module counts the number of words in an audio sample and checks if the number is within the range."""

from typing import Any

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speech_to_text.api import transcribe_audios
from senselab.utils.data_structures import HFModel


def is_num_words_in_range(audio: Audio, range: tuple, model: HFModel) -> bool:
    """Counts the number of words in an audio sample. Returns True if it falls within the range, and False otherwise.

    Args:
    audio (Audio): the audio sample
    range (tuple of two ints): the possible range of wordcount, from minimum to maximum
    model: SenselabModel

    Returns:
    True or False (bool): True if the number of words in the audio falls within the range, and False otherwise
    """
    transcript = transcribe_audios([audio], model)[0]
    if transcript and transcript.text:
        num_words = len(transcript.text.split())
        return not (num_words < range[0] or num_words > range[-1])
    return False
