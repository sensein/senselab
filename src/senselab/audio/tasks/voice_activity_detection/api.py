"""This module contains functions for computing Voice Activity Detection (VAD)."""

from typing import List, Optional

import pydra

from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.speaker_diarization.pyannote import diarize_audios_with_pyannote
from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.model import PyannoteAudioModel, SenselabModel
from senselab.utils.data_structures.script_line import ScriptLine


def detect_human_voice_activity_in_audios(
    audios: List[Audio],
    model: SenselabModel = PyannoteAudioModel(path_or_uri="pyannote/speaker-diarization-3.1", revision="main"),
    device: Optional[DeviceType] = None,
) -> List[List[ScriptLine]]:
    """Diarizes all audios using the given model.

    Args:
        audios (List[Audio]): The list of audio objects to be processed.
        model (SenselabModel): The model used for voice activity detection.
        device (Optional[DeviceType]): The device to run the model on (default is None).

    Returns:
        List[List[ScriptLine]]: The list of script lines with voice label.
    """
    if isinstance(model, PyannoteAudioModel):
        results = diarize_audios_with_pyannote(audios=audios, model=model, device=device)
        for sample in results:
            for chunk in sample:
                chunk.speaker = "VOICE"
        return results
    else:
        raise NotImplementedError("Only Pyannote models are supported for now.")


detect_human_voice_activity_in_audios_pt = pydra.mark.task(detect_human_voice_activity_in_audios)
