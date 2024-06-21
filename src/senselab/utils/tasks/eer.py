"""This module implements some utilities for computing the Equal Error Rate (EER)."""

from typing import Tuple

import pydra
import torch
from speechbrain.utils.metric_stats import EER


def compute_eer(predictions: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
    """Compute the Equal Error Rate (EER).

    Args:
        predictions (torch.Tensor): A 1D tensor of predictions.
        targets (torch.Tensor): A 1D tensor of targets.

    Returns:
        Tuple[float, float]: The EER and the threshold for the EER.
    """
    return EER(predictions, targets)


compute_eer_pt = pydra.mark.task(compute_eer)
