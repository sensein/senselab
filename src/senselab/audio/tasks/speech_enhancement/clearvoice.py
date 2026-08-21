"""ClearVoice speech enhancement: FRCRN, MossFormerGAN and MossFormer2 at 16 or 48 kHz.

Reached through :func:`senselab.audio.tasks.speech_enhancement.enhance_audios` by naming one of the
three checkpoints, or called directly. Runs in an isolated subprocess venv; see
:mod:`senselab.utils.clearvoice` for the venv, the pin and the device contract, and this package's
``doc.md`` for which non-speech elements each checkpoint destroys.
"""

from __future__ import annotations

from typing import List, Optional

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.clearvoice import run_clearvoice_over_audios, single_source_per_input
from senselab.utils.clearvoice import clearvoice_model_spec
from senselab.utils.data_structures import DeviceType, HFModel

CLEARVOICE_ENHANCEMENT_TASK = "speech_enhancement"


def enhance_audios_with_clearvoice(
    audios: List[Audio],
    model: HFModel,
    device: Optional[DeviceType] = None,
    timeout_s: Optional[float] = None,
) -> List[Audio]:
    """Enhance each audio with one of ClearVoice's three enhancement checkpoints.

    Inputs are resampled to the checkpoint's rate (16 or 48 kHz) and downmixed to mono; outputs come
    back at that rate with the input's sample count.

    Args:
        audios: Inputs.
        model: ``HFModel`` naming ``alibabasglab/FRCRN_SE_16K``,
            ``alibabasglab/MossFormerGAN_SE_16K`` or ``alibabasglab/MossFormer2_SE_48K``. Its
            ``commit_sha``, or its ``revision`` when unresolved, selects the weights.
        device: CUDA or CPU. ``None`` leaves the choice to the worker. MPS is not offered.
        timeout_s: Ceiling on the worker, in seconds. ``None`` derives one from the total duration.

    Returns:
        One enhanced ``Audio`` per input, each carrying a ``metadata["clearvoice"]`` record naming the
        model and the resolved commit.

    Raises:
        ValueError: If ``model`` does not name a ClearVoice enhancement checkpoint, or if
            ``timeout_s`` is not positive, or if ``device`` is neither CUDA nor CPU.
        RuntimeError: If the checkpoint returns anything other than one signal per input, or if the
            worker fails or exceeds its ceiling.
    """
    spec = clearvoice_model_spec(str(model.path_or_uri), expected_task=CLEARVOICE_ENHANCEMENT_TASK)
    results = run_clearvoice_over_audios(
        spec,
        audios,
        device=device,
        timeout_s=timeout_s,
        revision=model.commit_sha or model.revision,
    )
    return single_source_per_input(results, spec, "enhance_audios")
