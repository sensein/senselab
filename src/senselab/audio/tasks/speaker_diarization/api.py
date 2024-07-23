"""This module implements some utilities for the speaker diarization task.

# TODO: add computing DER and more evaluation metrics
"""

from typing import List, Optional

import pydra

from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.speaker_diarization.pyannote import diarize_audios_with_pyannote
from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.model import PyannoteAudioModel, SenselabModel
from senselab.utils.data_structures.script_line import ScriptLine


def diarize_audios(
    audios: List[Audio],
    model: SenselabModel,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    device: Optional[DeviceType] = None,
) -> List[List[ScriptLine]]:
    """Diarizes all audios using the given model.

    Args:
        audios (List[Audio]): The list of audio objects to be diarized.
        model (SenselabModel): The model used for diarization.
        device (Optional[DeviceType]): The device to run the model on (default is None).
        num_speakers (Optional[int]): The number of speakers (default is None).
        min_speakers (Optional[int]): The minimum number of speakers (default is None).
        max_speakers (Optional[int]): The maximum number of speakers (default is None).

    Returns:
        List[List[ScriptLine]]: The list of script lines with speaker labels.
    """
    if isinstance(model, PyannoteAudioModel):
        return diarize_audios_with_pyannote(
            audios=audios,
            model=model,
            device=device,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
    else:
        raise NotImplementedError("Only Pyannote models are supported for now.")


diarize_audios_pt = pydra.mark.task(diarize_audios)
