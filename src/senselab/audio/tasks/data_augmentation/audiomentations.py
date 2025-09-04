"""This module contains functions for applying data augmentation using audiomentations."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import cloudpickle
from pydra.compose import python, workflow

from senselab.audio.data_structures import Audio

try:
    from audiomentations import Compose

    AUDIOMENTATIONS_AVAILABLE = True
except ModuleNotFoundError:
    AUDIOMENTATIONS_AVAILABLE = False


def augment_audios_with_audiomentations(
    audios: List[Audio],
    augmentation: "Compose",
    plugin: str = "debug",
    plugin_args: Optional[dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
) -> List[Audio]:
    """Apply data augmentation using **audiomentations** via a Pydra workflow.

    The provided `audiomentations.Compose` is serialized once and sent to each
    map task so that multiple `Audio` objects can be augmented in parallel.

    Notes:
        - This function expects a CPU-only pipeline (audiomentations is NumPy-based).
        - For reproducibility, construct your `Compose` with your own random seed
          strategy (e.g., seeding your RNG before creating the pipeline).
        - The returned `Audio` objects preserve sampling rate and copy metadata.

    Args:
        audios (list[Audio]):
            List of `Audio` objects to augment.
        augmentation (Compose):
            An `audiomentations.Compose` pipeline (CPU).
        plugin (str, optional):
            Pydra execution plugin. Common options:
              * ``"serial"`` or ``"debug"``: Run sequentially (default).
              * ``"cf"``: Use concurrent futures for parallel execution.
              * ``"slurm"``: Submit tasks to a SLURM cluster.
        plugin_args (dict, optional):
            Extra keyword arguments passed to the chosen plugin.
            Example (for ``"cf"``): ``{"n_procs": 8}``
            See: https://nipype.github.io/pydra/
        cache_dir (str, optional):
            Directory for Pydra's cache. If ``None``, Pydra uses its default.

    Returns:
        list[Audio]: Augmented `Audio` objects in the same order as input.

    Raises:
        ModuleNotFoundError: If `audiomentations` is not installed.
        Exception: Any error raised during augmentation or workflow execution.

    Example:
        >>> from audiomentations import Compose, AddGaussianNoise, Gain
        >>> from senselab.audio.data_structures import Audio
        >>> from senselab.audio.tasks.data_augmentation.audiomentations import (
        ...     augment_audios_with_audiomentations
        ... )
        >>> aug = Compose([
        ...     AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=1.0),
        ...     Gain(min_gain_in_db=-6, max_gain_in_db=6, p=1.0),
        ... ])
        >>> a1 = Audio(filepath="/abs/path/sample1.wav")
        >>> a2 = Audio(filepath="/abs/path/sample2.wav")
        >>> out = augment_audios_with_audiomentations(
        ...     [a1, a2],
        ...     aug,
        ...     plugin="cf"
        ... )
        >>> len(out)
        2
    """
    if not AUDIOMENTATIONS_AVAILABLE:
        raise ModuleNotFoundError(
            "`audiomentations` is not installed. "
            "Please install senselab audio dependencies using `pip install 'senselab[audio]'`."
        )

    # Serialize augmentation deterministically
    aug_payload = cloudpickle.dumps(augmentation)

    @python.define
    def _augment_single_audio(audio: Audio, aug_payload: Any) -> Audio:  # noqa: ANN401
        aug = cloudpickle.loads(aug_payload)
        augmented = aug(samples=audio.waveform, sample_rate=audio.sampling_rate)
        return Audio(waveform=augmented, sampling_rate=audio.sampling_rate, metadata=audio.metadata.copy())

    @workflow.define
    def _wf(xs: Sequence[Audio], aug_payload: Any) -> List[Audio]:  # noqa: ANN401
        node = workflow.add(
            _augment_single_audio(aug_payload=aug_payload).split(audio=xs),
            name="map_audiomentations",
        )
        return node.out

    worker = "debug" if plugin in ("serial", "debug") else plugin
    res = _wf(xs=audios, aug_payload=aug_payload)(worker=worker, cache_root=cache_dir, **(plugin_args or {}))
    return list(res.out)
