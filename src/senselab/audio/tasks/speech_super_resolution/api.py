"""Public API for speech super-resolution (bandwidth extension)."""

from __future__ import annotations

from typing import Any, List, Mapping, Optional

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speech_super_resolution.clearvoice import (
    CLEARVOICE_SUPER_RESOLUTION_TASK,
    super_resolve_audios_with_clearvoice,
)
from senselab.utils.backend_parameters import record_parameters_on, resolve_backend_parameters
from senselab.utils.clearvoice import clearvoice_model_spec, clearvoice_models_for_task, is_clearvoice_model_id
from senselab.utils.compatibility import requires_compatibility
from senselab.utils.data_structures import DeviceType, HFModel, SenselabModel

DEFAULT_SUPER_RESOLUTION_MODEL = clearvoice_models_for_task(CLEARVOICE_SUPER_RESOLUTION_TASK)[0].model_id


@requires_compatibility("audio.tasks.speech_super_resolution.super_resolve_audios")
def super_resolve_audios(
    audios: List[Audio],
    model: Optional[SenselabModel] = None,
    device: Optional[DeviceType] = None,
    parameters: Optional[Mapping[str, Any]] = None,
) -> List[Audio]:
    """Reconstruct the high band of each audio, returning 48 kHz output.

    One backend today: ClearVoice's ``MossFormer2_SR_48K``, in an isolated subprocess venv.

    Args:
        audios: Inputs. Resampled to the model's rate and downmixed to mono.
        model: ``HFModel`` naming a ClearVoice super-resolution checkpoint. ``None`` uses
            ``alibabasglab/MossFormer2_SR_48K``.
        device: CUDA or CPU. ``None`` leaves the choice to the backend.
        parameters: Backend-specific parameters, validated against the selected backend's signature —
            an unknown key raises rather than being ignored. For this backend: ``timeout_s``.

    Returns:
        One 48 kHz ``Audio`` per input.

    Raises:
        NotImplementedError: If ``model`` names no known super-resolution backend.
        ValueError: If a parameter key is not one the backend declares.
    """
    if model is None:
        model = HFModel(path_or_uri=DEFAULT_SUPER_RESOLUTION_MODEL, revision="main")

    if isinstance(model, HFModel) and is_clearvoice_model_id(str(model.path_or_uri)):
        clearvoice_model_spec(str(model.path_or_uri), expected_task=CLEARVOICE_SUPER_RESOLUTION_TASK)
        kwargs, record = resolve_backend_parameters(
            super_resolve_audios_with_clearvoice, parameters, backend_name="clearvoice"
        )
        results = super_resolve_audios_with_clearvoice(audios=audios, model=model, device=device, **kwargs)
        record_parameters_on(results, record)
        return results

    raise NotImplementedError(
        f"No super-resolution backend for {model.path_or_uri!r}. Supported: HFModel ids naming a "
        f"ClearVoice super-resolution checkpoint ({DEFAULT_SUPER_RESOLUTION_MODEL})."
    )
