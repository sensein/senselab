"""ClearVoice speech super-resolution: MossFormer2_SR_48K, bandwidth extension to 48 kHz.

Runs in an isolated subprocess venv; see :mod:`senselab.utils.clearvoice` for the venv, the pin and
the device contract, and design.md D-2 for why super-resolution is its own task package rather than a
mode of enhancement.
"""

from __future__ import annotations

from typing import List, Optional

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.clearvoice import run_clearvoice_over_audios, single_source_per_input
from senselab.utils.clearvoice import clearvoice_model_spec
from senselab.utils.data_structures import DeviceType, HFModel

CLEARVOICE_SUPER_RESOLUTION_TASK = "speech_super_resolution"


def super_resolve_audios_with_clearvoice(
    audios: List[Audio],
    model: HFModel,
    device: Optional[DeviceType] = None,
    timeout_s: Optional[float] = None,
) -> List[Audio]:
    """Upsample each audio to 48 kHz and reconstruct the missing high band.

    The input is resampled to 48 kHz on the host first — the model fills in bandwidth, it does not
    resample — so the output is 48 kHz whatever the input rate was. Upstream reports the effective
    input rate should be at least 16 kHz.

    Args:
        audios: Inputs. Resampled to 48 kHz and downmixed to mono.
        model: ``HFModel`` naming ``alibabasglab/MossFormer2_SR_48K``.
        device: CUDA or CPU. ``None`` leaves the choice to the worker. MPS is not offered.
        timeout_s: Ceiling on the worker, in seconds. ``None`` derives one from the total duration.

    Returns:
        One 48 kHz ``Audio`` per input, each carrying a ``metadata["clearvoice"]`` record.

    Raises:
        ValueError: If ``model`` does not name the ClearVoice super-resolution checkpoint, or if
            ``timeout_s`` is not positive, or if ``device`` is neither CUDA nor CPU.
        RuntimeError: If the worker fails or exceeds its ceiling.
    """
    spec = clearvoice_model_spec(str(model.path_or_uri), expected_task=CLEARVOICE_SUPER_RESOLUTION_TASK)
    results = run_clearvoice_over_audios(
        spec,
        audios,
        device=device,
        timeout_s=timeout_s,
        revision=model.commit_sha or model.revision,
    )
    return single_source_per_input(results, spec, "super_resolve_audios")
