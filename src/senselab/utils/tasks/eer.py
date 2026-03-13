"""Equal Error Rate (EER) utilities.

This module provides a thin wrapper around `speechbrain.utils.metric_stats.EER`
to compute the Equal Error Rate and the operating threshold at which
**false acceptance rate (FAR)** equals **false rejection rate (FRR)**.

Inputs are 1D tensors of **scores** (higher → more likely positive) and
**binary targets** (0 = negative, 1 = positive), with matching length.
"""

from typing import Tuple

import torch

try:
    from speechbrain.utils.metric_stats import EER

    SPEECHBRAIN_AVAILABLE = True
except ModuleNotFoundError:
    SPEECHBRAIN_AVAILABLE = False


def compute_eer(predictions: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
    """Compute the Equal Error Rate (EER) and its threshold.

    Given per-example **scores** and **binary labels**, this function returns:
    - the **EER** (a value in ``[0, 1]``), and
    - the **threshold** on the score such that FAR == FRR.

    Args:
        predictions (torch.Tensor):
            1D tensor of scores with shape ``[N]`` where **higher values indicate
            the positive class** (e.g., similarity, probability, or logit).
            If your model outputs **distances** (lower = more similar), invert or
            negate them before calling this function.
        targets (torch.Tensor):
            1D tensor of **binary** ground-truth labels with shape ``[N]``.
            Values should be 0 (negative) or 1 (positive).

    Returns:
        Tuple[float, float]:
            ``(eer, threshold)`` where ``eer ∈ [0, 1]``.

    Raises:
        ModuleNotFoundError:
            If `speechbrain` is not installed.

    Notes:
        - Both tensors must be 1D and the **same length**.
        - Scores should be oriented so that **larger = more positive**; otherwise
          the computed threshold/EER will be misleading.
        - On datasets with many tied scores, multiple thresholds can achieve the
          same EER; SpeechBrain returns one such threshold.

    Examples:
        Basic usage (well-separated classes):
            >>> import torch
            >>> scores = torch.tensor([0.95, 0.87, 0.12, 0.08])
            >>> labels = torch.tensor([1, 1, 0, 0])
            >>> eer, th = compute_eer(scores, labels)
            >>> 0.0 <= eer <= 1.0
            True

        Using logits (higher = more positive is still satisfied):
            >>> logits = torch.tensor([4.0, 2.0, -1.0, -3.0])
            >>> labels = torch.tensor([1, 1, 0, 0])
            >>> compute_eer(logits, labels)[0]
            0.0...

        If you have distances (lower = more similar), negate first:
            >>> dists = torch.tensor([0.1, 0.2, 1.5, 2.0])  # lower means more positive
            >>> labels = torch.tensor([1, 1, 0, 0])
            >>> scores = -dists  # convert to higher-is-better
            >>> eer, th = compute_eer(scores, labels)
            >>> 0.0 <= eer <= 1.0
            True
    """
    if not SPEECHBRAIN_AVAILABLE:
        raise ModuleNotFoundError(
            "`speechbrain` is not installed. Please install senselab audio dependencies using `pip install senselab`."
        )

    return EER(predictions, targets)
