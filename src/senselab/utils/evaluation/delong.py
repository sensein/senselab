"""DeLong's test for two correlated ROC-AUCs on the same rows.

`delong_test` compares two scorers evaluated on the same labelled rows. The AUROCs, their variances
and their covariance come from the structural components of DeLong et al. (1988), computed in
O(N log N) with midranks as in Sun & Xu (2014): for a scorer, a positive's component is the share of
negatives it outranks (ties counting one half), and a negative's is the share of positives that
outrank it. The variance of the difference is `var_a + var_b - 2 cov_ab`; when it is not positive
(identical scorers, say) the test statistic is undefined and `z` and `p_value` are `None`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.stats import norm, rankdata

Values = Union[np.ndarray, Sequence[Any]]


@dataclass(frozen=True)
class DeLongResult:
    """The outcome of a paired DeLong test.

    Attributes:
        auroc_a: Scorer A's ROC-AUC.
        auroc_b: Scorer B's ROC-AUC.
        var_a: Variance of `auroc_a`.
        var_b: Variance of `auroc_b`.
        cov_ab: Covariance of the two AUROCs.
        z: `(auroc_a - auroc_b) / sqrt(var_a + var_b - 2 cov_ab)`, or `None` when that variance is not positive.
        p_value: Two-sided normal p-value of `z`, or `None` with it.
        n_positive: Positive rows.
        n_negative: Negative rows.
    """

    auroc_a: float
    auroc_b: float
    var_a: float
    var_b: float
    cov_ab: float
    z: Optional[float]
    p_value: Optional[float]
    n_positive: int
    n_negative: int

    def to_dict(self) -> Dict[str, Any]:
        """A plain, JSON-ready copy."""
        return asdict(self)


def _components(scores: np.ndarray, positive: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """AUROC and the structural components of the positives and the negatives, from midranks."""
    x, y = scores[positive], scores[~positive]
    m, n = len(x), len(y)
    combined = rankdata(np.concatenate([x, y]), method="average")
    within_x = rankdata(x, method="average")
    within_y = rankdata(y, method="average")
    auroc = float((combined[:m].sum() / m - (m + 1) / 2.0) / n)
    positive_part = (combined[:m] - within_x) / n
    negative_part = 1.0 - (combined[m:] - within_y) / m
    return auroc, positive_part, negative_part


def delong_test(
    scores_a: Values, scores_b: Values, labels: Values, *, positive_label: Union[str, int, bool] = 1
) -> DeLongResult:
    """Paired DeLong test of two scorers' ROC-AUCs on the same rows.

    Args:
        scores_a: Scorer A's score per row; higher means more likely positive. Finite.
        scores_b: Scorer B's score per row, on the same rows.
        labels: Label per row.
        positive_label: The label that marks a positive row (compared with `==`; default 1).

    Returns:
        The AUROCs, their variances and covariance, and the test statistic with its two-sided p-value.

    Raises:
        ValueError: If the inputs differ in length, a score is not finite, or either class has fewer
            than 2 rows (the covariances need two).
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    truth = np.asarray(labels)
    if not (a.shape == b.shape == truth.shape and a.ndim == 1):
        raise ValueError("scores_a, scores_b and labels must be one-dimensional and the same length")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("scores must be finite (no NaN or infinity)")
    positive = truth == positive_label
    m, n = int(positive.sum()), int((~positive).sum())
    if m < 2 or n < 2:
        raise ValueError("each class needs at least 2 rows")
    auroc_a, pos_a, neg_a = _components(a, positive)
    auroc_b, pos_b, neg_b = _components(b, positive)
    s_pos = np.cov(np.vstack([pos_a, pos_b]), ddof=1)
    s_neg = np.cov(np.vstack([neg_a, neg_b]), ddof=1)
    var_a = float(s_pos[0, 0] / m + s_neg[0, 0] / n)
    var_b = float(s_pos[1, 1] / m + s_neg[1, 1] / n)
    cov_ab = float(s_pos[0, 1] / m + s_neg[0, 1] / n)
    var_diff = var_a + var_b - 2.0 * cov_ab
    z: Optional[float] = None
    p_value: Optional[float] = None
    if var_diff > 0:
        z = float((auroc_a - auroc_b) / math.sqrt(var_diff))
        p_value = float(2.0 * norm.sf(abs(z)))
    return DeLongResult(
        auroc_a=auroc_a,
        auroc_b=auroc_b,
        var_a=var_a,
        var_b=var_b,
        cov_ab=cov_ab,
        z=z,
        p_value=p_value,
        n_positive=m,
        n_negative=n,
    )
