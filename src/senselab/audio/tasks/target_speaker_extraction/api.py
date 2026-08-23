"""Public API for audio-visual target speaker extraction."""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.target_speaker_extraction.clearvoice import (
    CLEARVOICE_TSE_TASK,
    VideoInput,
    extract_target_speakers_with_clearvoice,
)
from senselab.utils.backend_parameters import record_parameters_on, resolve_backend_parameters
from senselab.utils.clearvoice import clearvoice_model_spec, clearvoice_models_for_task, is_clearvoice_model_id
from senselab.utils.compatibility import requires_compatibility
from senselab.utils.data_structures import DeviceType, HFModel, SenselabModel

DEFAULT_TSE_MODEL = clearvoice_models_for_task(CLEARVOICE_TSE_TASK)[0].model_id


@requires_compatibility("audio.tasks.target_speaker_extraction.extract_target_speakers_from_videos")
def extract_target_speakers_from_videos(
    videos: Sequence[VideoInput],
    model: Optional[SenselabModel] = None,
    device: Optional[DeviceType] = None,
    parameters: Optional[Mapping[str, Any]] = None,
) -> List[List[Audio]]:
    """Extract each visible speaker's voice from every video, using their lip motion as the cue.

    One backend today: ClearVoice's ``AV_MossFormer2_TSE_16K``, in an isolated subprocess venv. The
    input must be a video **file** (``.mp4``, ``.avi``, ``.mov``, ``.webm``) and ffmpeg must be on
    PATH.

    Args:
        videos: Video files, as paths or file-backed ``Video`` objects.
        model: ``HFModel`` naming a ClearVoice extraction checkpoint. ``None`` uses
            ``alibabasglab/AV_MossFormer2_TSE_16K``.
        device: CUDA or CPU. ``None`` leaves the choice to the backend.
        parameters: Backend-specific parameters, validated against the selected backend's signature —
            an unknown key raises rather than being ignored. For this backend: ``timeout_s``.

    Returns:
        One list per input video, holding one 16 kHz ``Audio`` per detected face track.

    Raises:
        NotImplementedError: If ``model`` names no known extraction backend.
        ValueError: If a parameter key is not one the backend declares.
    """
    if model is None:
        model = HFModel(path_or_uri=DEFAULT_TSE_MODEL, revision="main")

    if isinstance(model, HFModel) and is_clearvoice_model_id(str(model.path_or_uri)):
        clearvoice_model_spec(str(model.path_or_uri), expected_task=CLEARVOICE_TSE_TASK)
        kwargs, record = resolve_backend_parameters(
            extract_target_speakers_with_clearvoice, parameters, backend_name="clearvoice"
        )
        results = extract_target_speakers_with_clearvoice(videos=videos, model=model, device=device, **kwargs)
        record_parameters_on(results, record)
        return results

    raise NotImplementedError(
        f"No target-speaker-extraction backend for {model.path_or_uri!r}. Supported: HFModel ids "
        f"naming a ClearVoice extraction checkpoint ({DEFAULT_TSE_MODEL})."
    )
