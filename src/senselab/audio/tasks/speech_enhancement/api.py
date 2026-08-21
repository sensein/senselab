"""This module provides the API for the senselab speech enhancement task."""

from typing import Any, Callable, List, Mapping, Optional

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speech_enhancement.clearvoice import (
    CLEARVOICE_ENHANCEMENT_TASK,
    enhance_audios_with_clearvoice,
)
from senselab.audio.tasks.speech_enhancement.driftse import enhance_audios_with_driftse
from senselab.audio.tasks.speech_enhancement.speechbrain import SpeechBrainEnhancer
from senselab.utils.backend_parameters import record_parameters_on, resolve_backend_parameters
from senselab.utils.clearvoice import clearvoice_model_spec, is_clearvoice_model_id
from senselab.utils.compatibility import requires_compatibility
from senselab.utils.data_structures import DeviceType, HFModel, SenselabModel, SpeechBrainModel

# Upstream's own MIT-licensed weights mirror. DriftSE is a normal selectable backend: an earlier
# restriction keeping it unreachable except by explicit naming rested on an unanswered licence
# request, which upstream has since answered. It is not the default enhancer, and making it one is a
# separate measured decision -- see specs/20260818-083214-driftse-upstream-mit/design.md.
_DRIFTSE_MODEL_PREFIX = "LIANGXU123/DriftSE"


@requires_compatibility("audio.tasks.speech_enhancement.enhance_audios")
def enhance_audios(
    audios: List[Audio],
    model: Optional[SenselabModel] = None,
    device: Optional[DeviceType] = None,
    parameters: Optional[Mapping[str, Any]] = None,
) -> List[Audio]:
    """Enhance every audio with the given model, returning one signal per input.

    Three backends, selected by the model:

    - a ``SpeechBrainModel`` (or ``None``) uses SpeechBrain, in process;
    - an ``HFModel`` under ``LIANGXU123/DriftSE`` uses DriftSE, one-step diffusion in an isolated venv;
    - an ``HFModel`` naming a ClearVoice enhancement checkpoint uses ClearVoice, in an isolated venv.

    Args:
        audios: The audios to enhance.
        model: The enhancement model. ``None`` uses ``speechbrain/sepformer-wham16k-enhancement``.
        device: The device to run on. ``None`` lets the backend choose.
        parameters: Backend-specific parameters, validated against the **selected** backend's own
            signature: an unknown or misspelled key raises rather than being ignored, and the
            effective set is recorded on each result's ``metadata["backend_parameters"]``. DriftSE
            declares ``variant``, ``seed``, ``sigma``, ``chunk_s``, ``overlap_s`` and ``timeout_s``;
            ClearVoice declares ``timeout_s``; SpeechBrain declares none.

    Returns:
        One enhanced ``Audio`` per input, in order.

    Raises:
        NotImplementedError: If ``model`` names no known enhancement backend.
        ValueError: If a parameter key is not one the selected backend declares.
    """
    if model is None:
        model = SpeechBrainModel(path_or_uri="speechbrain/sepformer-wham16k-enhancement", revision="main")

    backend: Optional[Callable[..., List[Audio]]] = None
    backend_name = ""
    if isinstance(model, SpeechBrainModel):
        backend, backend_name = SpeechBrainEnhancer.enhance_audios_with_speechbrain, "speechbrain"
    elif isinstance(model, HFModel) and str(model.path_or_uri).startswith(_DRIFTSE_MODEL_PREFIX):
        backend, backend_name = enhance_audios_with_driftse, "driftse"
    elif isinstance(model, HFModel) and is_clearvoice_model_id(str(model.path_or_uri)):
        # Rejects a separation or super-resolution checkpoint before any work happens, rather than
        # letting the backend discover it after staging 670 MB of weights.
        clearvoice_model_spec(str(model.path_or_uri), expected_task=CLEARVOICE_ENHANCEMENT_TASK)
        backend, backend_name = enhance_audios_with_clearvoice, "clearvoice"

    if backend is None:
        raise NotImplementedError(
            f"No enhancement backend for {model.path_or_uri!r}. Supported: SpeechBrain models, HFModel "
            f"ids starting with {_DRIFTSE_MODEL_PREFIX!r}, and ClearVoice enhancement checkpoints "
            "under 'alibabasglab/'."
        )

    kwargs, record = resolve_backend_parameters(backend, parameters, backend_name=backend_name)
    enhanced = backend(audios=audios, model=model, device=device, **kwargs)
    record_parameters_on(enhanced, record)
    return enhanced
