"""Utility functions for utilizing different devices in Senselab."""

from enum import Enum

import torch


class DeviceType(Enum):
    """Device types for PyTorch operations."""

    CPU: str = "cpu"
    CUDA: str = "cuda"
    MPS: str = "mps"


def _select_device_and_dtype(
    device_options: list[DeviceType] = [
        DeviceType.CPU,
        DeviceType.CUDA,
        DeviceType.MPS,
    ],
) -> tuple[DeviceType, torch.dtype]:
    """Determines the device and data type for PyTorch operations."""
    if torch.cuda.is_available() and DeviceType.CUDA in device_options:
        device = DeviceType.CUDA
        torch_dtype = torch.float16  # Using half precision for CUDA
    elif torch.backends.mps.is_available() and DeviceType.MPS in device_options:
        device = DeviceType.MPS
        torch_dtype = torch.float32
        # Default to float32 on MPS for better precision
    else:
        device = DeviceType.CPU
        torch_dtype = torch.float32
        # Default to float32 on CPU for better precision
    return device, torch_dtype
