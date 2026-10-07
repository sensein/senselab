"""Imbalance-aware classification reports.

`score_classification` scores one split and returns a `ClassificationReport`; `score_splits` scores
several and adds the prevalence of each. Every report names its protocol, leads with balanced
accuracy (the only headline) and gives every metric's chance or trivial floor, its lift over that
floor and the lift normalised by the room above the floor.

| Metric | Floor |
|---|---|
| balanced accuracy | 1/k, k = declared labels present in `y_true` |
| accuracy (secondary) | majority-class prevalence |
| macro-F1 (secondary) | best constant predictor: max over c of 2p_c/(1+p_c), divided by the number of labels |
| MCC (secondary) | 0 |
| positive-class F1 (secondary, binary with `positive_label`) | 2p/(1+p) |
| ROC-AUC | 0.5 |
| average precision | p (binary), 1/k (macro one-vs-rest) |

Labels are compared as strings. A prediction equal to `invalid_label` (a model output that could not be parsed)
counts as a miss: it is a confusion-matrix column, never a row.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Union
from typing import Protocol as TypingProtocol

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)

from senselab.utils.evaluation.protocol import PROTOCOLS, Protocol
from senselab.utils.evaluation.splits import SplitReport

HEADLINE = "balanced_accuracy"
DEFAULT_INVALID_LABEL = "INVALID"


class ThresholdLike(TypingProtocol):
    """A decision threshold chosen on a validation split (see `thresholds.tune_threshold`)."""

    @property
    def threshold(self) -> float:
        """The score at or above which the positive label is predicted."""
        ...

    @property
    def objective(self) -> str:
        """The metric the threshold was tuned to maximise."""
        ...

    @property
    def split_name(self) -> str:
        """The split it was tuned on."""
        ...


@dataclass(frozen=True)
class MetricValue:
    """A metric with its trivial floor.

    Attributes:
        value: The metric.
        floor: What a chance or trivial predictor scores.
        lift: `value - floor`.
        normalized_lift: `(value - floor) / (1 - floor)`, or `None` when the floor is 1.
    """

    value: float
    floor: float
    lift: float
    normalized_lift: Optional[float]


def _metric(value: float, floor: float) -> MetricValue:
    room = 1.0 - floor
    return MetricValue(
        value=float(value),
        floor=float(floor),
        lift=float(value - floor),
        normalized_lift=None if room <= 0 else float((value - floor) / room),
    )


@dataclass(frozen=True)
class ClassificationReport:
    """One split's classification report.

    Attributes:
        protocol: `seen_subject` or `unseen_subject`.
        split_name: The scored split.
        labels: The declared labels, in order.
        n: Scored rows.
        n_dropped: Rows the caller dropped before scoring (for example with no ground truth).
        invalid_rate: Share of predictions equal to the invalid label.
        prevalence: Label to its share of `y_true`.
        predicted_prevalence: Label (and the invalid label) to its share of the predictions.
        reference_prevalence: Caller-supplied reference distribution, shown beside `prevalence`.
        absent_labels: Declared labels with no example in `y_true`.
        headline_value: Balanced accuracy, the only headline.
        balanced_accuracy: Macro recall over the declared labels, invalid outputs counted as misses.
        secondary: `accuracy`, `f1_macro`, `mcc`, and `f1` (positive class) for binary with a positive label.
        per_class: Label to precision, recall, F1, support and prevalence.
        confusion_matrix: `rows` (true labels), `columns` (labels plus the invalid label) and `counts`.
        roc_auc: Binary (positive-label score) or macro one-vs-rest; `None` without scores.
        average_precision: As `roc_auc`.
        positive_label: The positive label of a binary task, if declared.
        threshold: `{value, objective, split_name}` of the decision threshold, if one was applied.
        threshold_tuning: `validation` when a threshold tuned on another split was applied, else `none`.
        split_report: The split report of the splits this split came from, if given.
        headline: Always `balanced_accuracy`.
    """

    protocol: Protocol
    split_name: str
    labels: List[str]
    n: int
    n_dropped: int
    invalid_rate: float
    prevalence: Dict[str, float]
    predicted_prevalence: Dict[str, float]
    reference_prevalence: Optional[Dict[str, float]]
    absent_labels: List[str]
    headline_value: float
    balanced_accuracy: MetricValue
    secondary: Dict[str, MetricValue]
    per_class: Dict[str, Dict[str, Any]]
    confusion_matrix: Dict[str, Any]
    roc_auc: Optional[MetricValue]
    average_precision: Optional[MetricValue]
    positive_label: Optional[str]
    threshold: Optional[Dict[str, Any]]
    threshold_tuning: Literal["validation", "none"]
    split_report: Optional[SplitReport]
    headline: str = field(default=HEADLINE, init=False)

    def _metrics(self) -> Dict[str, MetricValue]:
        metrics = {HEADLINE: self.balanced_accuracy, **self.secondary}
        if self.roc_auc is not None:
            metrics["roc_auc"] = self.roc_auc
        if self.average_precision is not None:
            metrics["average_precision"] = self.average_precision
        return metrics

    @property
    def floors(self) -> Dict[str, float]:
        """Metric name to floor."""
        return {name: m.floor for name, m in self._metrics().items()}

    @property
    def lift(self) -> Dict[str, float]:
        """Metric name to lift over its floor."""
        return {name: m.lift for name, m in self._metrics().items()}

    @property
    def normalized_lift(self) -> Dict[str, Optional[float]]:
        """Metric name to normalised lift."""
        return {name: m.normalized_lift for name, m in self._metrics().items()}


@dataclass(frozen=True)
class EvaluationReport:
    """Reports for several splits of one evaluation.

    Attributes:
        protocol: The protocol shared by every split.
        reports: Split name to its report.
        prevalence_by_split: Split name to its label prevalence.
    """

    protocol: Protocol
    reports: Dict[str, ClassificationReport]
    prevalence_by_split: Dict[str, Dict[str, float]]


def _as_strings(values: Union[np.ndarray, Sequence[Any]]) -> np.ndarray:
    return np.array([str(v) for v in np.asarray(values, dtype=object).tolist()], dtype=object)


def _scores_and_floors(
    y_true: np.ndarray,
    y_score: np.ndarray,
    labels: List[str],
    positive_label: Optional[str],
    prevalence: Dict[str, float],
) -> tuple[Optional[MetricValue], Optional[MetricValue]]:
    """ROC-AUC and average precision with floors; `None` where undefined (a single class present)."""
    if positive_label is not None:
        truth = y_true == positive_label
        if truth.all() or not truth.any():
            return None, None
        return (
            _metric(roc_auc_score(truth, y_score), 0.5),
            _metric(average_precision_score(truth, y_score), prevalence[positive_label]),
        )
    usable = [j for j, lab in enumerate(labels) if 0 < prevalence[lab] < 1]
    if not usable:
        return None, None
    aucs = [roc_auc_score(y_true == labels[j], y_score[:, j]) for j in usable]
    aps = [average_precision_score(y_true == labels[j], y_score[:, j]) for j in usable]
    present = sum(p > 0 for p in prevalence.values())
    return _metric(float(np.mean(aucs)), 0.5), _metric(float(np.mean(aps)), 1.0 / present)


def score_classification(
    y_true: Union[np.ndarray, Sequence[Any]],
    y_pred: Optional[Union[np.ndarray, Sequence[Any]]],
    *,
    labels: Sequence[Any],
    protocol: Protocol,
    split_name: str,
    n_dropped: int,
    split_report: Optional[SplitReport],
    y_score: Optional[Union[np.ndarray, Sequence[Any]]] = None,
    positive_label: Optional[Union[str, int]] = None,
    invalid_label: str = DEFAULT_INVALID_LABEL,
    threshold: Optional[ThresholdLike] = None,
    reference_prevalence: Optional[Mapping[str, float]] = None,
) -> ClassificationReport:
    """Score one split.

    The caller joins predictions to ground truth on `item_ref` before calling, and passes how many rows
    that join dropped.

    Args:
        y_true: True label per row; every value must be a declared label.
        y_pred: Predicted label per row (a declared label or `invalid_label`); `None` when `threshold`
            sets the predictions from `y_score`.
        labels: The declared labels; compared as strings.
        protocol: `seen_subject` or `unseen_subject`; no default.
        split_name: The scored split's name.
        n_dropped: Rows dropped before scoring; required.
        split_report: The `SplitReport` of the splits this split came from, or `None`; its protocol must
            equal `protocol`.
        y_score: Binary: the positive label's score per row. Multi-class: an `(n, k)` array, columns in
            `labels` order.
        positive_label: The positive label of a binary task; required with binary scores, refused for
            multi-class.
        invalid_label: The prediction value that marks an output that could not be parsed; must not be a label.
        threshold: A threshold tuned on another split; predictions become the positive label where the
            score is at or above it. Its objective is recorded; the headline stays balanced accuracy.
        reference_prevalence: Optional reference label distribution (for example the training set's).

    Returns:
        The report.

    Raises:
        ValueError: If the inputs are inconsistent (see the argument descriptions), or the threshold
            was tuned on the split being scored.
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}")
    label_list = [str(lab) for lab in labels]
    if not label_list or len(set(label_list)) != len(label_list):
        raise ValueError("labels must be non-empty and unique")
    if invalid_label in label_list:
        raise ValueError("invalid_label must not be one of the labels")
    if split_report is not None and split_report.protocol != protocol:
        raise ValueError("the split report's protocol differs from protocol")
    truth = _as_strings(y_true)
    n = len(truth)
    if n == 0:
        raise ValueError("y_true is empty")
    if set(truth.tolist()) - set(label_list):
        raise ValueError("y_true holds values outside the declared labels")
    binary = len(label_list) == 2
    positive = None if positive_label is None else str(positive_label)
    if positive is not None and (not binary or positive not in label_list):
        raise ValueError("positive_label must be one of exactly two declared labels")
    scores = None if y_score is None else np.asarray(y_score, dtype=float)
    if scores is not None:
        if binary and positive is None:
            raise ValueError("binary scores need positive_label")
        expected_shape = (n,) if binary else (n, len(label_list))
        if scores.shape != expected_shape:
            raise ValueError(f"y_score must have shape {expected_shape}")
    threshold_record = None
    if threshold is not None:
        if threshold.split_name == split_name:
            raise ValueError("the threshold was tuned on the split being scored")
        if y_pred is not None:
            raise ValueError("pass y_pred=None when a threshold sets the predictions")
        if scores is None or positive is None:
            raise ValueError("a threshold needs binary y_score and positive_label")
        negative = next(lab for lab in label_list if lab != positive)
        pred = np.where(scores >= threshold.threshold, positive, negative).astype(object)
        threshold_record = {
            "value": float(threshold.threshold),
            "objective": threshold.objective,
            "split_name": threshold.split_name,
        }
    else:
        if y_pred is None:
            raise ValueError("y_pred is required unless a threshold sets the predictions")
        pred = _as_strings(y_pred)
    if len(pred) != n:
        raise ValueError("y_true and y_pred must have the same length")
    if set(pred.tolist()) - set(label_list) - {invalid_label}:
        raise ValueError("y_pred holds values outside the declared labels and the invalid label")
    reference = None if reference_prevalence is None else {str(k): float(v) for k, v in reference_prevalence.items()}
    if reference is not None and set(reference) - set(label_list):
        raise ValueError("reference_prevalence names labels that are not declared")

    true_counts = Counter(truth.tolist())
    pred_counts = Counter(pred.tolist())
    prevalence = {lab: true_counts[lab] / n for lab in label_list}
    predicted = {lab: pred_counts[lab] / n for lab in [*label_list, invalid_label]}
    absent = [lab for lab in label_list if true_counts[lab] == 0]
    present = len(label_list) - len(absent)

    y_t, y_p = truth.tolist(), pred.tolist()
    ba = recall_score(y_t, y_p, labels=label_list, average="macro", zero_division=0)
    precision, recall, f1, support = precision_recall_fscore_support(y_t, y_p, labels=label_list, zero_division=0)
    per_class = {
        lab: {
            "precision": float(precision[j]),
            "recall": float(recall[j]),
            "f1": float(f1[j]),
            "support": int(support[j]),
            "prevalence": prevalence[lab],
        }
        for j, lab in enumerate(label_list)
    }
    columns = [*label_list, invalid_label]
    pairs = Counter(zip(y_t, y_p))
    confusion = {
        "rows": label_list,
        "columns": columns,
        "counts": [[pairs[(r, c)] for c in columns] for r in label_list],
    }
    best_constant_f1 = max(2 * p / (1 + p) for p in prevalence.values())
    secondary = {
        "accuracy": _metric(float(np.mean(truth == pred)), max(prevalence.values())),
        "f1_macro": _metric(
            f1_score(y_t, y_p, labels=label_list, average="macro", zero_division=0),
            best_constant_f1 / len(label_list),
        ),
        # One value across truth and predictions leaves MCC undefined; scikit-learn's convention is 0.
        "mcc": _metric(0.0 if len(set(y_t) | set(y_p)) < 2 else float(matthews_corrcoef(y_t, y_p)), 0.0),
    }
    if positive is not None:
        p = prevalence[positive]
        secondary["f1"] = _metric(per_class[positive]["f1"], 2 * p / (1 + p))
    roc, ap = (None, None) if scores is None else _scores_and_floors(truth, scores, label_list, positive, prevalence)
    balanced = _metric(float(ba), 1.0 / present)
    if not math.isfinite(balanced.value):
        raise ValueError("balanced accuracy is not finite")
    return ClassificationReport(
        protocol=protocol,
        split_name=split_name,
        labels=label_list,
        n=n,
        n_dropped=int(n_dropped),
        invalid_rate=predicted[invalid_label],
        prevalence=prevalence,
        predicted_prevalence=predicted,
        reference_prevalence=reference,
        absent_labels=absent,
        headline_value=balanced.value,
        balanced_accuracy=balanced,
        secondary=secondary,
        per_class=per_class,
        confusion_matrix=confusion,
        roc_auc=roc,
        average_precision=ap,
        positive_label=positive,
        threshold=threshold_record,
        threshold_tuning="validation" if threshold_record is not None else "none",
        split_report=split_report,
    )


def score_splits(inputs: Mapping[str, Mapping[str, Any]]) -> EvaluationReport:
    """Score several splits of one evaluation.

    Args:
        inputs: Split name to the keyword arguments of `score_classification` (without `split_name`);
            `y_true` and `y_pred` are passed by name.

    Returns:
        The reports and each split's prevalence.

    Raises:
        ValueError: If no split is given or the splits name different protocols.
    """
    if not inputs:
        raise ValueError("no splits to score")
    reports: Dict[str, ClassificationReport] = {}
    for name, kwargs in inputs.items():
        args = dict(kwargs)
        y_true, y_pred = args.pop("y_true"), args.pop("y_pred", None)
        reports[name] = score_classification(y_true, y_pred, split_name=name, **args)
    protocols = {r.protocol for r in reports.values()}
    if len(protocols) != 1:
        raise ValueError("every split of one evaluation must name the same protocol")
    return EvaluationReport(
        protocol=protocols.pop(),
        reports=reports,
        prevalence_by_split={name: dict(r.prevalence) for name, r in reports.items()},
    )
