"""This module contains functions for applying data augmentation using torch-audiomentations."""

from typing import Any, List, Optional, Sequence

import cloudpickle
import torch
from pydra.compose import python, workflow

from senselab.audio.data_structures import Audio, batch_audios, unbatch_audios
from senselab.utils.data_structures import DeviceType, _select_device_and_dtype

try:
    from torch_audiomentations import Compose as _TACompose  # runtime check

    TORCH_AUDIOMENTATIONS_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AUDIOMENTATIONS_AVAILABLE = False
    _TACompose = object  # sentinel to satisfy type-checkers


def augment_audios_with_torch_audiomentations(
    audios: List[Audio],
    augmentation: "_TACompose",
    device: Optional[DeviceType] = None,
    *,
    plugin: str = "debug",
    plugin_args: Optional[dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
) -> List[Audio]:
    """Augment a list of Audio with torch-audiomentations.

    CPU: parallel map over audios with Pydra compose.
    CUDA: batch once on device for throughput.
    """
    if not TORCH_AUDIOMENTATIONS_AVAILABLE:
        raise ModuleNotFoundError(
            "`torch-audiomentations` is not installed. "
            "Please install senselab audio dependencies using `pip install 'senselab[audio]'`"
        )
    if not isinstance(augmentation, _TACompose):
        raise TypeError("`augmentation` must be an instance of torch_audiomentations.Compose")

    # torch-audiomentations Compose needs dict output
    augmentation.output_type = "dict"

    device_type, _ = _select_device_and_dtype(
        user_preference=device,
        compatible_devices=[DeviceType.CUDA, DeviceType.CPU],
    )

    # --------------------------
    # GPU path: batch once
    # --------------------------
    if device_type == DeviceType.CUDA:
        batched, sampling_rates, metadatas = batch_audios(audios)
        batched = batched.to(device=torch.device(device_type.value))
        sampling_rate = sampling_rates[0] if isinstance(sampling_rates, list) else sampling_rates

        with torch.inference_mode():
            out = augmentation(batched, sample_rate=sampling_rate).samples  # (B, C, T)

        out = out.detach().cpu()
        return unbatch_audios(out, sampling_rates, metadatas)

    # --------------------------
    # CPU path: map with Pydra
    # --------------------------
    aug_payload = cloudpickle.dumps(augmentation)

    @python.define
    def _augment_single_audio(audio: Audio, aug_payload: bytes) -> Audio:
        aug = cloudpickle.loads(aug_payload)

        # torch-audiomentations expects (B, C, T) Tensor
        x = audio.waveform.unsqueeze(0)  # (1, C, T)
        with torch.inference_mode():
            y = aug(x, sample_rate=audio.sampling_rate).samples  # (1, C, T)
        y = torch.squeeze(y, 0)  # (C, T)

        return Audio(
            waveform=y,
            sampling_rate=audio.sampling_rate,
            metadata=audio.metadata.copy(),
        )

    @workflow.define
    def _wf(xs: Sequence[Audio], aug_payload: bytes) -> List[Audio]:
        node = workflow.add(
            _augment_single_audio(aug_payload=aug_payload).split(audio=xs),
            name="map_torch_audiomentations",
        )
        return node.out

    worker = "debug" if plugin in ("serial", "debug") else plugin
    res = _wf(xs=audios, aug_payload=aug_payload)(
        worker=worker,
        cache_root=cache_dir,
        **(plugin_args or {}),
    )
    return list(res.out)
