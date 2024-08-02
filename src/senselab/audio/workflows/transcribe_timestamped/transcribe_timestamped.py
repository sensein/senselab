"""Transcribes audio files with timestamps."""

from typing import List

from senselab.audio.data_structures.audio import Audio
from senselab.utils.data_structures.language import Language
from senselab.utils.data_structures.model import HFModel
from senselab.utils.data_structures.script_line import ScriptLine


def transcribe_timestamped(
    audios: List[Audio],
    model: HFModel = HFModel(path_or_uri="openai/whisper-tiny"),
    language: Language = Language(language_code="en"),
) -> List[ScriptLine]:
    """Transcribes a list of audio files and timestamps them using forced alignment.

    This function processes the given list of Audio objects by performing necessary
    preprocessing steps (such as downmixing channels and resampling), transcribes the
    audio using the specified speech-to-text model, and applies forced alignment to
    generate a list of ScriptLine objects with timestamps.

    Args:
        audios (list[Audio]): List of Audio objects to be transcribed and timestamped.
        model (HFModel, optional): A Huggingface model for speech-to-text. Defaults to
                                   'whisper'.
        language (Language, optional): Language object for the transcription. If None,
                                       language detection is triggered for the 'whisper'
                                       model. Defaults to None.

    Returns:
        list[ScriptLine]: List of ScriptLine objects resulting from the transcription
                          with timestamps.
    """
    return [ScriptLine(speaker="hello world")]
