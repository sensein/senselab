"""This module implements some utilities for computing the Equal Error Rate (EER)."""

from typing import Tuple

import torch

try:
    from speechbrain.utils.metric_stats import EER

    SPEECHBRAIN_AVAILABLE = True
except ModuleNotFoundError:
    SPEECHBRAIN_AVAILABLE = False


def compute_eer(predictions: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
    """Compute the Equal Error Rate (EER).

    Args:
        predictions (torch.Tensor): A 1D tensor of predictions.
        targets (torch.Tensor): A 1D tensor of targets.

    Returns:
        Tuple[float, float]: The EER and the threshold for the EER.
    """
    if not SPEECHBRAIN_AVAILABLE:
        raise ModuleNotFoundError(
            "`speechbrain` is not installed. "
            "Please install senselab audio dependencies using `pip install senselab['audio']`."
        )

    return EER(predictions, targets)
