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
    """Apply data augmentation using audiomentations.

    Args:
        audios: List of Audios whose data will be augmented with the given augmentations.
        augmentation: A composition of augmentations (audiomentations).
        plugin: The plugin to use for running the workflow. Default is "debug".
        plugin_args: Additional arguments to pass to the plugin. Default is None.
        cache_dir: The directory to use for caching the workflow. Default is None.

    Returns:
        List of augmented audios.
    """
    if not AUDIOMENTATIONS_AVAILABLE:
        raise ModuleNotFoundError(
            "`audiomentations` is not installed. "
            "Please install senselab audio dependencies using `pip install 'senselab[audio]'`."
        )

    # Serialize augmentation deterministically
    aug_payload = __import__("cloudpickle").dumps(augmentation)

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
    res = _wf(xs=audios, aug_payload=aug_payload)(
        worker=worker, cache_root=cache_dir, **(plugin_args or {})
    )
    return list(res.out)
