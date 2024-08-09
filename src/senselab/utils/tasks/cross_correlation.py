"""This module contains functions for computing the normalized cross-correlation between two signals."""

import numpy as np
import pydra
import torch
from scipy.signal import correlate


def compute_normalized_cross_correlation(signal1: torch.Tensor, signal2: torch.Tensor) -> torch.Tensor:
    """Calculate the normalized cross-correlation between two signals.

    Args:
        signal1 (torch.Tensor): The first input signal as a PyTorch tensor.
        signal2 (torch.Tensor): The second input signal as a PyTorch tensor.

    Returns:
        torch.Tensor: The normalized cross-correlation value between the two input signals.

    Examples:
        >>> signal1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> signal2 = torch.tensor([2.0, 3.0, 4.0])
        >>> normalized_cross_correlation(signal1, signal2)
        Tensor([0.30151134, 0.51298918, 0.77459667, 0.9486833 , 0.90453403, 0.70710678, 0.43643578])

    Note:
        This function assumes the input signals are one-dimensional
        and contain sufficient elements for meaningful cross-correlation.
    """
    # Ensure the inputs are 1D tensors
    if signal1.ndim != 1 or signal2.ndim != 1:
        raise ValueError("Input signals must be one-dimensional")

    # Convert PyTorch tensors to NumPy arrays
    signal1 = signal1.numpy()
    signal2 = signal2.numpy()

    # Calculate the energy of each signal
    energy_signal1 = np.sum(signal1**2)
    energy_signal2 = np.sum(signal2**2)

    # Check for zero energy to avoid division by zero
    if energy_signal1 == 0 or energy_signal2 == 0:
        raise ZeroDivisionError("One of the input signals has zero energy, causing division by zero in normalization")

    # Compute the cross-correlation
    cross_correlation = correlate(signal1, signal2)

    # Calculate the normalized cross-correlation
    normalized_cross_correlation = cross_correlation / np.sqrt(energy_signal1 * energy_signal2)

    print(normalized_cross_correlation)
    return torch.Tensor(normalized_cross_correlation)


compute_normalized_cross_correlation_pt = pydra.mark.task(compute_normalized_cross_correlation)
