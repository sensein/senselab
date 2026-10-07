"""Cluster bootstrap confidence intervals.

Rows from one group (a subject, a child) are correlated, so intervals resample whole groups with
replacement rather than rows. Each resample draws `n_groups` group indices with one
`Generator.integers(0, n_groups, n_groups)` call; groups are ordered by first appearance and a
group's rows keep their input order, so a seed fixes every resample. Percentile intervals use
`np.percentile` with linear interpolation.

A resample whose truth holds a single class is skipped and counted in `n_skipped`: balanced accuracy
would be trivially defined there, and ranking metrics undefined. Skipping matches the earlier lab
bootstrap, which parity requires. An interval with more than 5 % of its resamples skipped is flagged
`unreliable`: few groups carry the minority class, and the interval under-covers. A resample that drops
one of several classes is kept; balanced accuracy then averages over the classes present in it. A
metric that is undefined on a kept resample (returns `None` or NaN) is left out for that metric only;
one undefined on all rows has `estimate=None`.

`legacy_cluster_bootstrap` reproduces the lab's earlier bootstrap (fixed threshold, F1, balanced
accuracy, ROC-AUC, average precision) for parity checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Union

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from senselab.utils.evaluation._ids import distinct_keys, key_array
from senselab.utils.evaluation.metrics import balanced_accuracy

METHOD = "cluster_bootstrap_percentile"
DEFAULT_RESAMPLES = 2000
# Share of skipped (single-class) resamples above which an interval is flagged unreliable.
SKIP_FRACTION_LIMIT = 0.05

Values = Union[np.ndarray, Sequence[Any]]
RowStatistic = Callable[[np.ndarray], Optional[float]]


@dataclass(frozen=True)
class BootstrapCI:
    """A percentile interval from a cluster bootstrap.

    Attributes:
        estimate: The statistic on all rows, or `None` where it is undefined there.
        low: Lower percentile bound, or `None` if no resample gave a value.
        high: Upper percentile bound, or `None` if no resample gave a value.
        confidence: The interval's nominal coverage.
        n_resamples: Resamples drawn.
        n_used: Resamples that gave a value for this statistic.
        n_skipped: Resamples skipped because their truth held a single class.
        seed: The seed.
        n_groups: Distinct groups resampled.
        unreliable: More than 5 % of resamples were skipped, or none gave a value.
        method: `cluster_bootstrap_percentile`.
    """

    estimate: Optional[float]
    low: Optional[float]
    high: Optional[float]
    confidence: float
    n_resamples: int
    n_used: int
    n_skipped: int
    seed: int
    n_groups: int
    unreliable: bool
    method: str = METHOD


def _make_rng(seed: int) -> np.random.Generator:
    """The random generator for one bootstrap run."""
    return np.random.default_rng(seed)


def _group_members(groups: np.ndarray) -> List[np.ndarray]:
    """Row indices of each group, groups in first-appearance order, rows in input order."""
    keys = [str(g) for g in groups.tolist()]
    members: Dict[str, List[int]] = {}
    for i, key in enumerate(keys):
        members.setdefault(key, []).append(i)
    return [np.array(rows, dtype=int) for rows in members.values()]


def cluster_bootstrap_indices(groups: Values, *, n_resamples: int, seed: int) -> Iterator[np.ndarray]:
    """Row indices of each cluster-bootstrap resample.

    Args:
        groups: Group id per row.
        n_resamples: Resamples to draw.
        seed: The seed.

    Yields:
        The resampled rows: each drawn group's rows, concatenated in draw order.
    """
    members = _group_members(np.asarray(groups))
    rng = _make_rng(seed)
    for _ in range(n_resamples):
        picks = rng.integers(0, len(members), size=len(members))
        yield np.concatenate([members[i] for i in picks])


def validate_groups(n_rows: int, groups: Values, n_resamples: int, confidence: float) -> np.ndarray:
    """The groups as an array after checking lengths, missing ids, group count and settings.

    Groups are told apart as strings, the same way resampling groups them, so 1 and "1" are one group.
    """
    group_arr = key_array(groups, "group")
    if len(group_arr) != n_rows:
        raise ValueError("groups must have one entry per row")
    if distinct_keys(group_arr) < 2:
        raise ValueError("a cluster bootstrap needs at least 2 groups")
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    return group_arr


def _percentiles(confidence: float) -> tuple[float, float]:
    """Lower and upper percentile for a coverage, rounded so 0.95 gives exactly 2.5 and 97.5."""
    low = round(50.0 * (1.0 - confidence), 12)
    return low, 100.0 - low


def _defined(value: Optional[float]) -> Optional[float]:
    """The value as a float, or `None` if it is `None` or NaN."""
    return None if value is None or math.isnan(value) else float(value)


def bootstrap_statistics(
    statistics: Mapping[str, RowStatistic],
    truth: np.ndarray,
    groups: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
    confidence: float,
) -> Dict[str, BootstrapCI]:
    """Cluster-bootstrap intervals for statistics of row indices.

    The engine behind `cluster_bootstrap_ci` and the report's intervals. Validate `groups` with
    `validate_groups` first.

    Args:
        statistics: Name to a function of resampled row indices (`None` or NaN where undefined).
        truth: True label per row (any comparable codes); a resample holding one value is skipped.
        groups: Group id per row.
        n_resamples: Resamples to draw.
        seed: The seed.
        confidence: Nominal coverage.

    Returns:
        Statistic name to its interval.
    """
    all_rows = np.arange(len(truth))
    estimates = {name: _defined(fn(all_rows)) for name, fn in statistics.items()}
    values: Dict[str, List[float]] = {name: [] for name in statistics}
    skipped = 0
    for rows in cluster_bootstrap_indices(groups, n_resamples=n_resamples, seed=seed):
        if np.unique(truth[rows]).size < 2:
            skipped += 1
            continue
        for name, fn in statistics.items():
            value = _defined(fn(rows))
            if value is not None:
                values[name].append(value)
    q_low, q_high = _percentiles(confidence)
    n_groups = distinct_keys(groups)
    unreliable = skipped / n_resamples > SKIP_FRACTION_LIMIT
    result = {}
    for name, collected in values.items():
        result[name] = BootstrapCI(
            estimate=estimates[name],
            low=float(np.percentile(collected, q_low)) if collected else None,
            high=float(np.percentile(collected, q_high)) if collected else None,
            confidence=confidence,
            n_resamples=n_resamples,
            n_used=len(collected),
            n_skipped=skipped,
            seed=seed,
            n_groups=n_groups,
            unreliable=unreliable or not collected,
        )
    return result


def cluster_bootstrap_ci(
    metric_fns: Mapping[str, Callable[[np.ndarray, np.ndarray], Optional[float]]],
    y_true: Values,
    y_values: Values,
    groups: Values,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int,
    confidence: float = 0.95,
) -> Dict[str, BootstrapCI]:
    """Cluster-bootstrap intervals for metrics of `(y_true, y_values)`.

    Args:
        metric_fns: Name to a function of `(y_true, y_values)` restricted to a resample's rows. For
            balanced accuracy pass `metrics.balanced_accuracy` with fixed predictions; nothing
            is re-tuned inside a resample.
        y_true: True label per row.
        y_values: Predictions or scores per row.
        groups: Group id per row; at least two groups, none missing.
        n_resamples: Resamples to draw (default 2000).
        seed: The seed; no default.
        confidence: Nominal coverage (default 0.95).

    Returns:
        Metric name to its interval.

    Raises:
        ValueError: If lengths differ, a group id is missing, fewer than two groups exist, or the
            settings are out of range.
    """
    truth, values = np.asarray(y_true), np.asarray(y_values)
    if len(values) != len(truth):
        raise ValueError("y_true and y_values must have one entry per row")
    group_arr = validate_groups(len(truth), groups, n_resamples, confidence)
    statistics = {name: (lambda rows, fn=fn: fn(truth[rows], values[rows])) for name, fn in metric_fns.items()}
    return bootstrap_statistics(statistics, truth, group_arr, n_resamples=n_resamples, seed=seed, confidence=confidence)


def paired_cluster_bootstrap_delta(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: Values,
    values_a: Values,
    values_b: Values,
    groups: Values,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int,
    confidence: float = 0.95,
) -> BootstrapCI:
    """Interval for `metric(a) - metric(b)`, both computed on the same resampled groups.

    Args:
        metric_fn: A function of `(y_true, values)`.
        y_true: True label per row.
        values_a: Model A's predictions or scores.
        values_b: Model B's predictions or scores.
        groups: Group id per row.
        n_resamples: Resamples to draw (default 2000).
        seed: The seed; no default.
        confidence: Nominal coverage (default 0.95).

    Returns:
        The interval of the difference.
    """
    truth, a, b = np.asarray(y_true), np.asarray(values_a), np.asarray(values_b)
    if not len(truth) == len(a) == len(b):
        raise ValueError("y_true, values_a and values_b must have one entry per row")
    group_arr = validate_groups(len(truth), groups, n_resamples, confidence)

    def delta(rows: np.ndarray) -> float:
        """The paired difference on one resample."""
        return float(metric_fn(truth[rows], a[rows]) - metric_fn(truth[rows], b[rows]))

    return bootstrap_statistics(
        {"delta": delta}, truth, group_arr, n_resamples=n_resamples, seed=seed, confidence=confidence
    )["delta"]


def _ranking(metric: Callable[..., float], truth: np.ndarray, scores: np.ndarray) -> Optional[float]:
    """A ranking metric, or `None` when the truth holds one class."""
    return None if np.unique(truth).size < 2 else float(metric(truth, scores))


def legacy_cluster_bootstrap(
    y_true: Values,
    y_score: Values,
    groups: Values,
    *,
    threshold: float,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int,
    confidence: float = 0.95,
) -> Dict[str, BootstrapCI]:
    """The lab's earlier binary cluster bootstrap, for parity with its published intervals.

    Predictions are `score >= threshold`; the metrics are `f1` (positive class), `balanced_accuracy`
    (through `metrics.balanced_accuracy`), `auroc` and `auprc` (`None` on data holding one class). Resampling is as in
    `cluster_bootstrap_indices`.

    Args:
        y_true: 0/1 label per row.
        y_score: Positive-class score per row.
        groups: Group id per row.
        threshold: The fixed decision threshold.
        n_resamples: Resamples to draw (default 2000).
        seed: The seed; no default.
        confidence: Nominal coverage (default 0.95).

    Returns:
        Metric name to its interval.
    """
    truth = np.asarray(y_true).astype(int)
    scores = np.asarray(y_score, dtype=float)
    if len(scores) != len(truth):
        raise ValueError("y_true and y_score must have one entry per row")
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    group_arr = validate_groups(len(truth), groups, n_resamples, confidence)
    pred = (scores >= threshold).astype(int)
    statistics: Dict[str, RowStatistic] = {
        "f1": lambda rows: float(f1_score(truth[rows], pred[rows], zero_division=0)),
        "balanced_accuracy": lambda rows: balanced_accuracy(truth[rows], pred[rows]),
        "auroc": lambda rows: _ranking(roc_auc_score, truth[rows], scores[rows]),
        "auprc": lambda rows: _ranking(average_precision_score, truth[rows], scores[rows]),
    }
    return bootstrap_statistics(statistics, truth, group_arr, n_resamples=n_resamples, seed=seed, confidence=confidence)
