from typing import Any, List, Optional

import os

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speech_to_text.huggingface import HuggingFaceASR
from senselab.audio.tasks.speech_to_text.api import transcribe_audios
from senselab.utils.data_structures import DeviceType, HFModel, Language, ScriptLine, SenselabModel
import torchaudio
from senselab.audio.tasks.preprocessing import resample_audios, downmix_audios_to_mono

def is_num_words_in_range(audio, range, model):
    """"
    Counts the number of words in an audio sample. Returns True if it falls within the range, and False otherwise.

    Args:
    audio (Audio): the audio sample
    range (tuple of two ints): the possible range of wordcount, from minimum to maximum
    model: SenselabModel

    Returns:
    True if the number of words in the audio falls within the range, and false otherwise (Boolean)
    """
    transcript = transcribe_audios([audio], model)[0]
    num_words = len(transcript.text.split())
    if num_words < range[0] or num_words > range[-1]:
        return False
    return True
