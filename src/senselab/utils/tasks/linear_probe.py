"""Grouped linear probe: logistic regression on frozen features, regularisation chosen on held-out inner rows.

For each split, a model is fitted per grid value on the `train` rows; the value with the best `objective` on the
inner-validation rows (partition `inner_partition`, `val` by default, the name `grouped_splits` and
`nested_grouped_splits` give it) is selected (the first in grid order on a tie); the model is then refitted on `train`
and the inner-validation rows together and predicts the `test` rows. Test rows are never used to fit or to select.
A split in which any group has rows in two of `train`, the inner-validation partition and `test` raises.

Grid values are scikit-learn's `C` (inverse regularisation strength). The result is deterministic for a fixed seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from senselab.utils.evaluation.metrics import balanced_accuracy
from senselab.utils.evaluation.protocol import ProtocolViolation, check_protocol
from senselab.utils.evaluation.splits import Split

OBJECTIVES: Mapping[str, Callable[[Any, Any], float]] = {"balanced_accuracy": balanced_accuracy}
DEFAULT_INNER_PARTITION = "val"


@dataclass(frozen=True)
class LinearProbeResult:
    """Per-row outputs of a linear probe over several splits.

    Attributes:
        rows: Row indices scored, split by split, each split's test rows in ascending order.
        predictions: Predicted label per scored row, aligned with `rows`.
        scores: Class probabilities per scored row (columns follow `classes`), aligned with `rows`.
        split_index: The split each scored row belongs to, aligned with `rows`.
        selected: The grid value selected on each split's inner validation rows.
        classes: The class labels, in the column order of `scores`.
    """

    rows: np.ndarray
    predictions: np.ndarray
    scores: np.ndarray
    split_index: np.ndarray
    selected: List[float]
    classes: np.ndarray


def _fit(x: np.ndarray, y: np.ndarray, c: float, class_weight: str, seed: int) -> LogisticRegression:
    return LogisticRegression(C=c, class_weight=class_weight, random_state=seed, max_iter=5000).fit(x, y)


def linear_probe(
    features: np.ndarray,
    labels: Sequence[Any],
    groups: Sequence[Any],
    splits: Sequence[Split],
    *,
    grid: Sequence[float],
    class_weight: str,
    objective: str,
    seed: int,
    inner_partition: str = DEFAULT_INNER_PARTITION,
) -> LinearProbeResult:
    """Fit, select and predict a logistic-regression probe on each split.

    Args:
        features: One feature row per item, shape `(n, d)`.
        labels: One label per item.
        groups: One group id per item (a subject, a child).
        splits: Splits whose partitions `train`, `inner_partition` and `test` hold row indices.
        grid: Candidate `C` values.
        class_weight: Passed to scikit-learn; `balanced` weights classes by inverse frequency.
        objective: Name of the selection metric; `balanced_accuracy`.
        seed: Seed of the fitted models.
        inner_partition: Name of the partition the grid value is selected on.

    Returns:
        The per-row predictions and scores of every split's test rows, and the value selected per split.

    Raises:
        ValueError: On an empty grid, an unknown objective, a split without all three partitions, a group with
            rows in two partitions (a `ProtocolViolation`), a training partition missing a class, or inputs of
            different lengths.
    """
    if not grid:
        raise ValueError("grid must hold at least one value")
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; use one of {sorted(OBJECTIVES)}")
    metric = OBJECTIVES[objective]
    x, y, g = np.asarray(features), np.asarray(labels), np.asarray(groups)
    if not len(x) == len(y) == len(g):
        raise ValueError("features, labels and groups must have one entry per row")
    classes = np.unique(y)
    rows: List[np.ndarray] = []
    preds: List[np.ndarray] = []
    scores: List[np.ndarray] = []
    index: List[np.ndarray] = []
    selected: List[float] = []
    for i, split in enumerate(splits):
        names = ("train", inner_partition, "test")
        for part in names:
            if part not in split.partitions:
                raise ValueError(f"split {split.name}: missing partition {part}")
        train, inner, test = (np.asarray(split.partitions[p], dtype=int) for p in names)
        try:
            check_protocol({"train": train, "inner": inner, "test": test}, g, protocol="unseen_subject")
        except ProtocolViolation as error:
            raise ProtocolViolation(f"split {split.name}: {error}") from error
        fitting = np.concatenate([train, inner])
        if set(np.unique(y[train]).tolist()) != set(classes.tolist()):
            raise ValueError(f"split {split.name}: the training rows lack a class")
        best_value, best_score = float(grid[0]), -np.inf
        for c in grid:
            value = metric(y[inner], _fit(x[train], y[train], c, class_weight, seed).predict(x[inner]))
            if value > best_score:
                best_value, best_score = float(c), value
        model = _fit(x[fitting], y[fitting], best_value, class_weight, seed)
        order = np.sort(test)
        rows.append(order)
        preds.append(model.predict(x[order]))
        scores.append(model.predict_proba(x[order]))
        index.append(np.full(len(order), i, dtype=int))
        selected.append(best_value)
    return LinearProbeResult(
        rows=np.concatenate(rows),
        predictions=np.concatenate(preds),
        scores=np.concatenate(scores),
        split_index=np.concatenate(index),
        selected=selected,
        classes=classes,
    )
