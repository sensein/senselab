"""This module provides the API for the senselab speech enhancement task."""

from typing import List, Optional

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speech_enhancement.driftse import enhance_audios_with_driftse
from senselab.audio.tasks.speech_enhancement.speechbrain import SpeechBrainEnhancer
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
) -> List[Audio]:
    """Enhances all audios using the given model.

    Supports **SpeechBrain** (default) and **DriftSE** (HF-identified) backends:
    - If `model` is a `SpeechBrainModel` (or `None`), uses SpeechBrain.
    - If `model` is an `HFModel` and `model.path_or_uri` starts with `"LIANGXU123/DriftSE"`,
      uses DriftSE (one-step diffusion enhancement) via an isolated subprocess venv.

    Args:
        audios (List[Audio]): The list of audio objects to be enhanced.
        model (SenselabModel): The model used for enhancement.
            If None, the default model "speechbrain/sepformer-wham16k-enhancement" is used.
        device (Optional[DeviceType]): The device to run the model on (default is None).

    Returns:
        List[Audio]: The list of enhanced audio objects.

    Raises:
        NotImplementedError: If an unsupported model type is passed.
    """
    if model is None:
        model = SpeechBrainModel(path_or_uri="speechbrain/sepformer-wham16k-enhancement", revision="main")

    if isinstance(model, SpeechBrainModel):
        return SpeechBrainEnhancer.enhance_audios_with_speechbrain(audios=audios, model=model, device=device)
    if isinstance(model, HFModel) and str(model.path_or_uri).startswith(_DRIFTSE_MODEL_PREFIX):
        return enhance_audios_with_driftse(audios=audios, model=model, device=device)
    raise NotImplementedError(
        f"No enhancement backend for {model.path_or_uri!r}. Supported: SpeechBrain models, "
        f"and HFModel ids starting with {_DRIFTSE_MODEL_PREFIX!r}."
    )
