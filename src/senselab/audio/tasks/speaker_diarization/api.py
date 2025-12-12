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
    """Diarize a batch of `Audio` objects, returning per-speaker time segments.

    Supports **Pyannote** (default) and **NVIDIA Sortformer** (HF) backends:
    - If `model` is a `PyannoteAudioModel`, uses Pyannote (typically expects **mono, 16 kHz**).
      Optional `num_speakers` or (`min_speakers`, `max_speakers`) are honored.
    - If `model` is an `HFModel` and `model.path_or_uri` starts with `"nvidia/diar_sortformer"`,
    uses NVIDIA Sortformer via Docker (nvidia/diar_sortformer_4spk-v1 detects max **4 speakers**).

    Args:
        audios (list[Audio]):
            Audio objects to diarize.
        model (SenselabModel | None):
            Diarization backend:
              * ``PyannoteAudioModel(...)`` → Pyannote (default if ``None``).
              * ``HFModel(...)`` → NVIDIA Sortformer.
        num_speakers (int | None):
            If known, fix the number of speakers (Pyannote only).
        min_speakers (int | None):
            Lower bound when estimating number of speakers (Pyannote only).
        max_speakers (int | None):
            Upper bound when estimating number of speakers (Pyannote only).
            NVIDIA Sortformer is limited to 4 speakers.
        device (DeviceType | None):
            Preferred device (e.g., ``DeviceType.CPU``, ``DeviceType.CUDA``).

    Returns:
        list[list[ScriptLine]]: One list per input audio; each `ScriptLine` carries
        `speaker`, `start`, and `end`.

    Raises:
        NotImplementedError: If an unsupported model type is passed.

    Example (Pyannote, default model, CPU):
        >>> from pathlib import Path
        >>> from senselab.audio.data_structures import Audio
        >>> from senselab.utils.data_structures import DeviceType, PyannoteAudioModel
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> a2 = Audio(filepath=Path("sample2.wav").resolve())
        >>> lines = diarize_audios([a1, a2], device=DeviceType.CPU)
        >>> len(lines) == 2
        True

    Example (NVIDIA Sortformer via HF, CUDA):
        >>> from pathlib import Path
        >>> from senselab.audio.data_structures import Audio
        >>> from senselab.utils.data_structures import DeviceType, HFModel
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> hf = HFModel(path_or_uri="nvidia/diar_sortformer_4spk-v1")
        >>> lines = diarize_audios([a1], model=hf, device=DeviceType.CUDA)
        >>> isinstance(lines[0], list)
        True
    """
    if model is None:
        model = PyannoteAudioModel(path_or_uri="pyannote/speaker-diarization-community-1", revision="main")

    if isinstance(model, PyannoteAudioModel):
        return diarize_audios_with_pyannote(
            audios=audios,
            model=model,
            device=device,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
    elif isinstance(model, HFModel) and str(model.path_or_uri).startswith("nvidia/diar"):
        return diarize_audios_with_nvidia_sortformer(
            audios=audios,
            model=model,
            device=device,
        )
    else:
        raise NotImplementedError(
            "Only Pyannote and NVIDIA Sortformer (from HuggingFace) models are supported for now."
        )
