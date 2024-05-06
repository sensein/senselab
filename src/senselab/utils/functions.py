"""Utility functions for senselab."""
import torch


def _select_device_and_dtype(device_options: list[str] = ['cuda', 'mps', 'cpu']) -> tuple[str, torch.dtype]:
    """Determines the device and data type for PyTorch operations."""
    # TODO: maybe move this function as a generic utility function?
    if torch.cuda.is_available() and "cuda" in device_options:
        device = "cuda"
        torch_dtype = torch.float16  # Using half precision for CUDA
    elif torch.backends.mps.is_available() and "mps" in device_options:
        device = "mps"
        torch_dtype = torch.float16  # Using half precision for MPS if suitable
    else:
        device = "cpu"
        torch_dtype = torch.float32  # Default to float32 on CPU for better precision
    return device, torch_dtype