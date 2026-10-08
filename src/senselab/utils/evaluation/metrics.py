"""Label metrics shared by reports, threshold tuning and the bootstrap.

`balanced_accuracy` is the evaluation headline: recall averaged over the labels present in `y_true`,
equal to scikit-learn's `balanced_accuracy_score`, with any prediction that is not the true label (an
invalid output, say) counted as a miss. `legacy_macro_recall` is the declared-label formula of the
lab's earlier evaluation code, kept for parity checks only.

Labels are compared as strings. The confusion-matrix functions work on integer codes, so a caller that
scores many resamples encodes the labels once (`encode_labels`) and reuses the codes.
"""

from __future__ import annotations

import math
from typing import Any, List, Sequence, Tuple, Union

import numpy as np
from sklearn.metrics import recall_score

Values = Union[np.ndarray, Sequence[Any]]


def as_strings(values: Values) -> np.ndarray:
    """The values as an object array of strings."""
    return np.array([str(v) for v in np.asarray(values, dtype=object).tolist()], dtype=object)


def encode_labels(*columns: Values, vocabulary: Sequence[str] = ()) -> Tuple[List[np.ndarray], List[str]]:
    """Integer codes for one or more label columns over a shared, string-compared vocabulary.

    Args:
        *columns: Label columns.
        vocabulary: Labels to place first, in this order; others follow sorted.

    Returns:
        The coded columns and the vocabulary (code `i` is `vocabulary[i]`).
    """
    strings = [as_strings(c) for c in columns]
    names = list(vocabulary) + sorted({s for col in strings for s in col.tolist()} - set(vocabulary))
    index = {name: i for i, name in enumerate(names)}
    return [np.array([index[s] for s in col.tolist()], dtype=np.intp) for col in strings], names


def confusion(truth: np.ndarray, pred: np.ndarray, k: int) -> np.ndarray:
    """The `k x k` confusion matrix (rows true, columns predicted) of integer codes."""
    return np.bincount(truth * k + pred, minlength=k * k).reshape(k, k)


def balanced_accuracy_from_confusion(matrix: np.ndarray) -> float:
    """Recall averaged over the rows (true labels) that occur, summed exactly so label order cannot matter."""
    support = matrix.sum(axis=1)
    present = support > 0
    if not present.any():
        raise ValueError("balanced accuracy needs at least one row")
    recalls = (np.diag(matrix)[present] / support[present]).tolist()
    return math.fsum(recalls) / len(recalls)


def balanced_accuracy(y_true: Values, y_pred: Values) -> float:
    """Recall averaged over the labels present in `y_true`: the evaluation headline.

    Equal to scikit-learn's `balanced_accuracy_score` within 1e-15: the recalls are summed with
    `math.fsum`, so the value does not depend on the order of the labels, whereas scikit-learn's plain
    mean over its sorted labels can differ in the last bit. A prediction that is not the true label (for
    example an invalid output) is a miss. Reports, threshold tuning and the bootstrap all use this
    definition (resamples through `balanced_accuracy_from_confusion` on pre-encoded labels).

    Args:
        y_true: True label per row.
        y_pred: Predicted label per row.

    Returns:
        The balanced accuracy.

    Raises:
        ValueError: If the inputs are empty or of different lengths.
    """
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be non-empty and of the same length")
    (truth, pred), names = encode_labels(y_true, y_pred)
    return balanced_accuracy_from_confusion(confusion(truth, pred, len(names)))


def per_class_from_confusion(matrix: np.ndarray, j: int) -> Tuple[float | None, float | None, float | None]:
    """Precision, recall and F1 of label `j`: precision `None` if never predicted, recall and F1 `None` if absent."""
    tp = int(matrix[j, j])
    predicted = int(matrix[:, j].sum())
    support = int(matrix[j, :].sum())
    precision = tp / predicted if predicted else None
    if not support:
        return precision, None, None
    recall = tp / support
    return precision, recall, 2 * tp / (predicted + support)


def macro_f1_from_confusion(matrix: np.ndarray, labels: Sequence[int]) -> float:
    """F1 averaged over the given label codes that occur in the truth."""
    values = [f1 for j in labels if (f1 := per_class_from_confusion(matrix, j)[2]) is not None]
    return float(np.mean(values))


def mcc_from_confusion(matrix: np.ndarray) -> float:
    """Matthews correlation from a confusion matrix (the multi-class form); 0 when undefined."""
    true_sum = matrix.sum(axis=1).astype(float)
    pred_sum = matrix.sum(axis=0).astype(float)
    n = float(matrix.sum())
    correct = float(np.trace(matrix))
    cov_true_pred = correct * n - float(true_sum @ pred_sum)
    cov_pred = n * n - float(pred_sum @ pred_sum)
    cov_true = n * n - float(true_sum @ true_sum)
    if cov_pred * cov_true == 0:
        return 0.0
    return cov_true_pred / float(np.sqrt(cov_true * cov_pred))


def legacy_macro_recall(y_true: Values, y_pred: Values, *, labels: Sequence[Any]) -> float:
    """Macro recall over every declared label with `zero_division=0`, for parity checks only.

    This is the formula of the lab's earlier evaluation code: an absent declared label contributes a
    recall of 0. It is never a report headline; see `balanced_accuracy`.

    Args:
        y_true: True label per row.
        y_pred: Predicted label per row.
        labels: The declared labels.

    Returns:
        The declared-label macro recall.
    """
    label_list = [str(lab) for lab in labels]
    truth, pred = as_strings(y_true).tolist(), as_strings(y_pred).tolist()
    return float(recall_score(truth, pred, labels=label_list, average="macro", zero_division=0))
