"""Utility functions for senselab."""
import os
from enum import Enum
from typing import List

import torch


class DeviceType(Enum):
    """Device types for PyTorch operations."""
    CPU: str = 'cpu'
    CUDA: str = 'cuda'
    MPS: str = 'mps'

def _select_device_and_dtype(device_options: list[DeviceType] = [DeviceType.CPU, DeviceType.CUDA, DeviceType.MPS]) -> tuple[DeviceType, torch.dtype]:
    """Determines the device and data type for PyTorch operations."""
    if torch.cuda.is_available() and DeviceType.CUDA in device_options:
        device = DeviceType.CUDA
        torch_dtype = torch.float16  # Using half precision for CUDA
    elif torch.backends.mps.is_available() and DeviceType.MPS in device_options:
        device = DeviceType.MPS
        torch_dtype = torch.float32  # Default to float32 on MPS for better precision
    else:
        device = DeviceType.CPU
        torch_dtype = torch.float32  # Default to float32 on CPU for better precision
    return device, torch_dtype


def get_common_directory(files: List[str]) -> str:
    """A function to get the common directory from a list of file paths.
    
    Parameters:
    - files: a list of file paths
    
    Returns:
    - the common directory among the file paths
    """
    if len(files) == 1:
        # Ensure the single path's directory ends with a separator
        common_path = os.path.dirname(files[0])
    else:
        # Use commonpath to find the common directory for multiple files
        common_path = os.path.commonpath(files)
    
    # Check if the path ends with the os separator, add if not
    if not common_path.endswith(os.sep):
        common_path += os.sep
    
    return common_path
