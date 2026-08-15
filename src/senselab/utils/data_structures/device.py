"""Device selection utilities for senselab.

This module centralizes logic for choosing a **device** and a corresponding
**default dtype** for PyTorch operations. It detects availability of CUDA
(NVIDIA GPUs) and MPS (Apple Silicon) at runtime, honors an optional
user preference, and falls back to a sensible fastest-available choice.

Mappings:
    - ``DTYPE_MAP`` chooses the default dtype per device:
        * ``CPU  → torch.float32``
        * ``CUDA → torch.float16``   (fast, lower memory)
        * ``MPS  → torch.float32``

Notes:
    - The dtype mapping is a **default** heuristic. Some models/operators may
      require ``float32`` (even on CUDA) for numerical stability. You can ignore
      the suggested dtype and cast tensors/model parameters as needed.
"""

from enum import Enum
from typing import Optional

import torch


class DeviceType(Enum):
    """Supported device backends for PyTorch execution.

    Values:
        CPU:  Host CPU (always available).
        CUDA: NVIDIA GPU via CUDA (if available).
        MPS:  Apple Silicon GPU via Metal Performance Shaders (if available).
    """

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


DTYPE_MAP = {DeviceType.CPU: torch.float32, DeviceType.CUDA: torch.float16, DeviceType.MPS: torch.float32}


def _select_device_and_dtype(
    user_preference: Optional[DeviceType] = None,
    compatible_devices: list[DeviceType] = [
        DeviceType.CPU,
        DeviceType.CUDA,
        DeviceType.MPS,
    ],
) -> tuple[DeviceType, torch.dtype]:
    """Select a device and a recommended dtype for PyTorch ops.

    The selector:
      1) Detects available devices (CPU always; optionally CUDA/MPS).
      2) Intersects with ``compatible_devices`` supplied by the caller.
      3) If ``user_preference`` is provided, validates and returns it.
      4) Otherwise, returns the **fastest available** in priority order:
         ``CUDA → MPS → CPU``, with a recommended dtype from ``DTYPE_MAP``.

    Args:
        user_preference (DeviceType | None):
            Optional user-requested device. Must be an instance of ``DeviceType``.
        compatible_devices (list[DeviceType]):
            Devices that the *caller’s algorithm* supports. For example, pass
            ``[DeviceType.CPU, DeviceType.CUDA]`` if MPS is not supported.

    Returns:
        tuple[DeviceType, torch.dtype]:
            The selected device and a suggested default dtype for that device.

    Raises:
        ValueError:
            - If ``user_preference`` is not a ``DeviceType``.
            - If ``user_preference`` is not both **available** and **compatible**.
            - If no devices are both available and compatible (unexpected).

    Notes:
        - CUDA/MPS availability is checked via ``torch.cuda.is_available()`` and
          ``torch.backends.mps.is_available()`` and verified by a tiny tensor
          allocation; errors are caught and ignored with a notice.
        - Suggested dtype for CUDA is ``float16`` by default for performance and
          memory efficiency. If your model needs higher precision, use ``float32``.
        - This helper **does not** mutate global dtype; it only returns a
          recommendation for the caller to use.

    Examples:
        Basic selection (no preference; choose fastest available):
            >>> from senselab.utils.data_structures.device import _select_device_and_dtype, DeviceType
            >>> dev, dtype = _select_device_and_dtype()
            >>> dev in {DeviceType.CPU, DeviceType.CUDA, DeviceType.MPS}
            True

        Respect a user preference (validated against availability/compatibility):
            >>> dev, dtype = _select_device_and_dtype(user_preference=DeviceType.CPU)
            >>> dev is DeviceType.CPU
            True

        Restrict to CPU only (e.g., algorithm not implemented for GPUs):
            >>> dev, dtype = _select_device_and_dtype(compatible_devices=[DeviceType.CPU])
            >>> dev is DeviceType.CPU
            True
    """
    if user_preference:
        if not isinstance(user_preference, DeviceType):
            raise ValueError(f"user_preference should be of type DeviceType, not {type(user_preference)}")

    available_devices = [DeviceType.CPU]

    if torch.cuda.is_available():
        try:
            torch.empty(0, device=DeviceType.CUDA.value)
            available_devices.append(DeviceType.CUDA)
        except Exception as e:
            print(f"CUDA is available but encountered an error: {e}")

    if torch.backends.mps.is_available():
        try:
            torch.empty(0, device=DeviceType.MPS.value)
            available_devices.append(DeviceType.MPS)
        except Exception as e:
            print(f"MPS is available but encountered an error: {e}")

    # Check compatible and available
    useable_devices = []
    for device in available_devices:
        if device in compatible_devices:
            useable_devices.append(device)

    # User preference or fastest option

    if user_preference:
        # user_preference = DeviceType(user_preference) if isinstance(user_preference,str) else user_preference
        if user_preference not in useable_devices:
            raise ValueError(
                "The requested DeviceType is either not available or\
                             compatible with this functionality."
            )
        else:
            return user_preference, DTYPE_MAP[user_preference]
    else:
        if DeviceType.CUDA in useable_devices:
            return DeviceType.CUDA, DTYPE_MAP[DeviceType.CUDA]
        elif DeviceType.MPS in useable_devices:
            return DeviceType.MPS, DTYPE_MAP[DeviceType.MPS]
        elif DeviceType.CPU in useable_devices:
            return DeviceType.CPU, DTYPE_MAP[DeviceType.CPU]
        else:
            raise ValueError("Something went wrong and no devices were available or compatible.")


def device_run_opt(device: DeviceType) -> str:
    """Return a concrete device string suitable for a library's ``run_opts["device"]``.

    ``DeviceType.CUDA.value`` is the bare string ``"cuda"``, which several libraries cannot
    parse. SpeechBrain in particular does ``device.split(":")`` and, when that unpack fails,
    logs "Could not parse CUDA device string" and calls ``torch.cuda.set_device(0)`` itself
    (``speechbrain/inference/interfaces.py``). That silently discards the caller's choice:
    on a node with several visible GPUs it moves the model to card 0 no matter which one was
    selected.

    The index therefore comes from ``torch.cuda.current_device()`` rather than a hardcoded
    ``0``. That is what makes this correct on a cluster: under Slurm, ``CUDA_VISIBLE_DEVICES``
    masks the allocation so the granted GPU is index 0 *inside the process* and this returns
    ``"cuda:0"`` — the allocated card, not merely the first one on the host — while a caller
    who selected a device explicitly gets the card they chose.

    Args:
        device: The selected device.

    Returns:
        ``"cuda:<index>"`` for CUDA when a GPU is actually visible, otherwise the device's
        bare value. ``cpu`` and ``mps`` take no index -- ``"cpu:0"`` is not a valid device
        string -- and a CUDA request with no visible GPU stays bare so the downstream
        library raises its own clearer error rather than this helper failing first.
    """
    if device is not DeviceType.CUDA:
        return str(device.value)
    if not torch.cuda.is_available():
        return str(device.value)
    return f"cuda:{torch.cuda.current_device()}"
