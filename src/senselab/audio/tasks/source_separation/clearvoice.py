"""ClearVoice speech separation: MossFormer2_SS_16K, two sources at 16 kHz.

Reached through :func:`senselab.audio.tasks.source_separation.separate_audios` by naming the
checkpoint, or called directly. Runs in an isolated subprocess venv; see
:mod:`senselab.utils.clearvoice` for the venv, the pin and the device contract.

Two properties a caller needs, both in this package's ``doc.md``: this checkpoint separates *speakers*
and is not a class decomposer, and its sources come back RMS-matched to a normalised input rather than
to the caller's level.
"""

from __future__ import annotations

from typing import List, Optional

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.clearvoice import run_clearvoice_over_audios
from senselab.utils.clearvoice import clearvoice_model_spec
from senselab.utils.data_structures import DeviceType, HFModel

CLEARVOICE_SEPARATION_TASK = "speech_separation"


def separate_audios_with_clearvoice(
    audios: List[Audio],
    model: HFModel,
    device: Optional[DeviceType] = None,
    timeout_s: Optional[float] = None,
) -> List[List[Audio]]:
    """Separate each audio into the sources ClearVoice's separator produces.

    Inputs are resampled to 16 kHz and downmixed to mono.

    Args:
        audios: Inputs.
        model: ``HFModel`` naming ``alibabasglab/MossFormer2_SS_16K``.
        device: CUDA or CPU. ``None`` leaves the choice to the worker. MPS is not offered.
        timeout_s: Ceiling on the worker, in seconds. ``None`` derives one from the total duration.

    Returns:
        One list of ``Audio`` per input — as many sources as the checkpoint produced, which for this
        checkpoint is two. Each carries a ``metadata["clearvoice"]`` record naming the model, the
        resolved commit, its source index, and the RMS scalar that was **not** applied to it.

    Raises:
        ValueError: If ``model`` does not name the ClearVoice separation checkpoint, or if
            ``timeout_s`` is not positive, or if ``device`` is neither CUDA nor CPU.
        RuntimeError: If any input yields fewer than two sources, or if the worker fails or exceeds
            its ceiling.
    """
    spec = clearvoice_model_spec(str(model.path_or_uri), expected_task=CLEARVOICE_SEPARATION_TASK)
    results = run_clearvoice_over_audios(
        spec,
        audios,
        device=device,
        timeout_s=timeout_s,
        revision=model.commit_sha or model.revision,
    )
    thin = [(index, len(sources)) for index, sources in enumerate(results) if len(sources) < 2]
    if thin:
        detail = ", ".join(f"input {index} -> {count} source(s)" for index, count in thin[:3])
        raise RuntimeError(
            f"separate_audios expects at least two sources per input, but {spec.model_id} produced "
            f"{detail}. A separator returning one signal has not separated anything, and returning it "
            "as a source list would present an unseparated mixture as a decomposition."
        )
    return results
