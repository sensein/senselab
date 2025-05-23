"""This module implements some utilities for the speaker diarization task."""

from typing import List, Optional

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speaker_diarization.nvidia import diarize_audios_with_nvidia_sortformer
from senselab.audio.tasks.speaker_diarization.pyannote import diarize_audios_with_pyannote
from senselab.utils.data_structures import DeviceType, HFModel, PyannoteAudioModel, ScriptLine, SenselabModel


def diarize_audios(
    audios: List[Audio],
    model: Optional[SenselabModel] = None,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    device: Optional[DeviceType] = None,
) -> List[List[ScriptLine]]:
    """Diarizes all audios using the given model.

    Args:
        audios (List[Audio]): The list of audio objects to be diarized.
        model (SenselabModel): The model used for diarization.
            If None, the default model "pyannote/speaker-diarization-3.1" is used.
        device (Optional[DeviceType]): The device to run the model on (default is None).
        num_speakers (Optional[int]): The number of speakers (default is None).
            This will be used only if the model supports it (e.g., Pyannote).
        min_speakers (Optional[int]): The minimum number of speakers (default is None).
            This will be used only if the model supports it (e.g., Pyannote).
        max_speakers (Optional[int]): The maximum number of speakers (default is None).
            This will be used only if the model supports it (e.g., Pyannote).

    Returns:
        List[List[ScriptLine]]: The list of script lines with speaker labels.
    """
    if model is None:
        model = PyannoteAudioModel(path_or_uri="pyannote/speaker-diarization-3.1", revision="main")

    if isinstance(model, PyannoteAudioModel):
        return diarize_audios_with_pyannote(
            audios=audios,
            model=model,
            device=device,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
    elif isinstance(model, HFModel):
        return diarize_audios_with_nvidia_sortformer(
            audios=audios,
            model_name=str(model.path_or_uri),
            device=device,
        )
    else:
        raise NotImplementedError(
            "Only Pyannote and NVIDIA Sortformer (from HuggingFace) models are supported for now."
        )
