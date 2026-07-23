"""Thin senselab callers per analysis aspect.

Only diarization is implemented for the first backend; ASR and scene classification will
follow the same pattern. ``prepare_audio`` enforces the mono / 16 kHz that pyannote requires,
and ``resolve_diarization_model`` mirrors ``diarize_audios``'s own model dispatch.
"""

from __future__ import annotations

from typing import Optional, Union

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.preprocessing import downmix_audios_to_mono, resample_audios
from senselab.audio.tasks.speaker_diarization import diarize_audios
from senselab.utils.data_structures import DeviceType, HFModel, PyannoteAudioModel, ScriptLine

TARGET_SAMPLE_RATE = 16000
DEFAULT_PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"
DEFAULT_MODEL_REVISION = "main"
SORTFORMER_PREFIX = "nvidia/diar"


def prepare_audio(audio: Audio) -> Audio:
    """Downmix to mono and resample to 16 kHz (pyannote's hard requirement).

    Args:
        audio: The input audio (any layout / sample rate).

    Returns:
        A mono, 16 kHz copy suitable for diarization.
    """
    mono = downmix_audios_to_mono([audio])[0]
    return resample_audios([mono], TARGET_SAMPLE_RATE)[0]


def resolve_diarization_model(
    model_id: str = DEFAULT_PYANNOTE_MODEL,
    revision: str = DEFAULT_MODEL_REVISION,
) -> Union[PyannoteAudioModel, HFModel]:
    """Resolve a diarization model id to the senselab model wrapper.

    Mirrors ``diarize_audios``'s internal dispatch: ``nvidia/diar*`` ids use the NVIDIA
    Sortformer backend (an ``HFModel``); everything else uses pyannote.

    Args:
        model_id: The model repo id.
        revision: The model revision / branch.

    Returns:
        A ``PyannoteAudioModel`` or ``HFModel``.
    """
    if model_id.startswith(SORTFORMER_PREFIX):
        return HFModel(path_or_uri=model_id, revision=revision)
    return PyannoteAudioModel(path_or_uri=model_id, revision=revision)


def diarize(
    audio: Audio,
    *,
    model_id: str = DEFAULT_PYANNOTE_MODEL,
    revision: str = DEFAULT_MODEL_REVISION,
    device: Optional[DeviceType] = None,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> list[ScriptLine]:
    """Run speaker diarization on one audio and return its segments.

    Args:
        audio: The input audio (prepared to mono / 16 kHz internally).
        model_id: Diarization model repo id.
        revision: Model revision / branch.
        device: Optional device override; ``None`` lets senselab auto-select.
        num_speakers: Exact speaker count, when known (pyannote only).
        min_speakers: Lower bound on speakers (pyannote only).
        max_speakers: Upper bound on speakers (pyannote only).

    Returns:
        The list of diarization segments (``ScriptLine``) for the audio.
    """
    model = resolve_diarization_model(model_id, revision)
    prepared = prepare_audio(audio)
    results = diarize_audios(
        audios=[prepared],
        model=model,
        device=device,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    return results[0] if results else []
