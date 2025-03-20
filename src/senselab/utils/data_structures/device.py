"""Utility functions for utilizing different devices in Senselab."""

from enum import Enum
from typing import Optional

import torch


class DeviceType(Enum):
    """Device types for PyTorch operations."""

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
    """Determines the device and data type for PyTorch operations.

    Allows users to give preferences for DeviceType, but determines based
    on compatible and available devices. Chooses the fastest option if no
    user preference is given.

    Args:
        user_preference: Optional DeviceType that the user wants to use
        compatible_devices: DeviceTypes that work with the functionality of the method calling this
    Returns:
        Tuple of (DeviceType, torch.dtype) where the device is both available and compatible and the
            dtype is the best performing dtype for that DeviceType
    Raises:
        ValueError: if the user specifies a preference that is not available or compatible and a safety
            call if no devices are available or compatible (we believe this to be impossible to trigger).
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
