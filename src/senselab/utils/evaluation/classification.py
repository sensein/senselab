"""Imbalance-aware classification reports.

`score_classification` scores one split and returns a `ClassificationReport`; `score_splits` scores
several and adds the prevalence of each. Every report names its protocol, leads with balanced
accuracy (the only headline) and gives every metric's chance or trivial floor, its lift over that
floor and the lift normalised by the room above the floor.

Balanced accuracy is recall averaged over the labels present in `y_true` (`metrics.balanced_accuracy`,
equal to scikit-learn's `balanced_accuracy_score`), with invalid outputs counted as misses. The
earlier declared-label formula is `metrics.legacy_macro_recall`, for parity checks only.

| Metric | Floor |
|---|---|
| balanced accuracy | 1/k, k = declared labels present in `y_true` |
| accuracy (secondary) | majority-class prevalence |
| macro-F1 (secondary, over present labels) | 1/k |
| MCC (secondary) | 0 |
| positive-class F1 (secondary, binary with `positive_label`) | 2p/(1+p) |
| ROC-AUC | 0.5 |
| average precision | p (binary), 1/k (macro one-vs-rest) |

Labels are compared as strings. A prediction equal to `invalid_label` (a model output that could not
be parsed) counts as a miss: it is a confusion-matrix column, never a row.

Report mappings are read-only views (`types.MappingProxyType`) and sequences are tuples;
`ClassificationReport.to_dict` gives a plain, JSON-ready copy.
"""

from __future__ import annotations

import dataclasses
import math
from collections import Counter
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union
from typing import Protocol as TypingProtocol

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from senselab.utils.evaluation.bootstrap import DEFAULT_RESAMPLES, BootstrapCI, bootstrap_statistics, validate_groups
from senselab.utils.evaluation.metrics import (
    as_strings,
    balanced_accuracy_from_confusion,
    confusion,
    encode_labels,
    macro_f1_from_confusion,
    mcc_from_confusion,
    native_order,
    per_class_from_confusion,
)
from senselab.utils.evaluation.protocol import PROTOCOLS, Protocol
from senselab.utils.evaluation.splits import SplitReport

HEADLINE = "balanced_accuracy"
DEFAULT_INVALID_LABEL = "INVALID"
PREVALENCE_TOLERANCE = 1e-9
ThresholdTuning = Literal["validation", "none", "test"]


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
    """A `MetricValue` with lift and normalised lift computed from the value and floor."""
    room = 1.0 - floor
    return MetricValue(
        value=float(value),
        floor=float(floor),
        lift=float(value - floor),
        normalized_lift=None if room <= 0 else float((value - floor) / room),
    )


def _plain(value: object) -> object:
    """A JSON-ready copy: mappings to dicts, tuples to lists, dataclasses to dicts."""
    if isinstance(value, MetricValue):
        return dataclasses.asdict(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _plain(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


@dataclass(frozen=True)
class ClassificationReport:
    """One split's classification report; mappings are read-only, sequences are tuples.

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
        balanced_accuracy: Recall averaged over the labels present in `y_true` (`balanced_accuracy`).
        secondary: `accuracy`, `f1_macro` (over present labels), `mcc`, and `f1` (positive class) for
            binary with a positive label that occurs in `y_true` (omitted when it is absent: undefined).
        per_class: Label to precision, recall, F1, support and prevalence. Recall and F1 are `None` for
            an absent label; precision is `None` for a label never predicted.
        confusion_matrix: `rows` (true labels), `columns` (labels plus the invalid label) and `counts`.
        roc_auc: Binary (positive-label score) or macro one-vs-rest; `None` without scores.
        average_precision: As `roc_auc`.
        positive_label: The positive label of a binary task, if declared.
        threshold: `{value, objective, split_name}` of the decision threshold, if one was applied here.
        threshold_tuning: `validation` (a threshold tuned on another split) or `none`.
        split_report: The split report of the splits this split came from, if given.
        ci: With `groups`, cluster-bootstrap intervals for `balanced_accuracy`, `f1_macro`, `mcc`, and
            `roc_auc` and `average_precision` when scores exist; each records its method, resamples,
            seed, groups and skipped resamples. `None` without groups.
        headline: Always `balanced_accuracy`.
    """

    protocol: Protocol
    split_name: str
    labels: Tuple[str, ...]
    n: int
    n_dropped: int
    invalid_rate: float
    prevalence: Mapping[str, float]
    predicted_prevalence: Mapping[str, float]
    reference_prevalence: Optional[Mapping[str, float]]
    absent_labels: Tuple[str, ...]
    headline_value: float
    balanced_accuracy: MetricValue
    secondary: Mapping[str, MetricValue]
    per_class: Mapping[str, Mapping[str, Any]]
    confusion_matrix: Mapping[str, Any]
    roc_auc: Optional[MetricValue]
    average_precision: Optional[MetricValue]
    positive_label: Optional[str]
    threshold: Optional[Mapping[str, Any]]
    threshold_tuning: Literal["validation", "none"]
    split_report: Optional[SplitReport]
    ci: Optional[Mapping[str, BootstrapCI]] = None
    headline: str = field(default=HEADLINE, init=False)

    def _metrics(self) -> Dict[str, MetricValue]:
        """Every metric with a floor, by name."""
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

    def to_dict(self) -> Dict[str, Any]:
        """A plain, JSON-ready copy of the report, with `floors`, `lift` and `normalized_lift` added."""
        data = {f.name: _plain(getattr(self, f.name)) for f in dataclasses.fields(self)}
        data["floors"] = self.floors
        data["lift"] = self.lift
        data["normalized_lift"] = self.normalized_lift
        return data


@dataclass(frozen=True)
class EvaluationReport:
    """Reports for several splits of one evaluation.

    Attributes:
        protocol: The protocol shared by every split.
        reports: Split name to its report.
        prevalence_by_split: Split name to its label prevalence.
    """

    protocol: Protocol
    reports: Mapping[str, ClassificationReport]
    prevalence_by_split: Mapping[str, Mapping[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        """A plain, JSON-ready copy, each report as `ClassificationReport.to_dict` gives it."""
        return {
            "protocol": self.protocol,
            "reports": {name: report.to_dict() for name, report in self.reports.items()},
            "prevalence_by_split": _plain(self.prevalence_by_split),
        }


def _scores_and_floors(
    y_true: np.ndarray,
    y_score: np.ndarray,
    labels: List[str],
    positive_label: Optional[str],
    prevalence: Mapping[str, float],
) -> Tuple[Optional[MetricValue], Optional[MetricValue]]:
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


def _resample_statistics(
    truth: np.ndarray,
    pred: np.ndarray,
    n_codes: int,
    present_codes: List[int],
    label_strings: np.ndarray,
    scores: Optional[np.ndarray],
    labels: List[str],
    positive: Optional[str],
    order: List[int],
) -> Dict[str, Callable[[np.ndarray], Optional[float]]]:
    """Row-index statistics for the bootstrap on pre-encoded labels, computed as the report computes them."""

    def matrix(rows: np.ndarray) -> np.ndarray:
        """The resample's confusion matrix."""
        return confusion(truth[rows], pred[rows], n_codes)

    statistics: Dict[str, Callable[[np.ndarray], Optional[float]]] = {
        HEADLINE: lambda rows: balanced_accuracy_from_confusion(matrix(rows), order),
        "f1_macro": lambda rows: macro_f1_from_confusion(matrix(rows), present_codes),
        "mcc": lambda rows: mcc_from_confusion(matrix(rows)),
    }
    if scores is not None:

        def ranking(rows: np.ndarray, which: int) -> Optional[float]:
            """ROC-AUC (0) or average precision (1) on the resample, `None` where undefined."""
            sample = label_strings[rows]
            prevalence = {lab: float(np.mean(sample == lab)) for lab in labels}
            value = _scores_and_floors(sample, scores[rows], labels, positive, prevalence)[which]
            return None if value is None else value.value

        statistics["roc_auc"] = lambda rows: ranking(rows, 0)
        statistics["average_precision"] = lambda rows: ranking(rows, 1)
    return statistics


def _same_split(a: str, b: str) -> bool:
    """Split names compared without case or surrounding whitespace."""
    return a.strip().casefold() == b.strip().casefold()


def _check_reference(reference: Optional[Mapping[str, float]], labels: List[str]) -> Optional[Dict[str, float]]:
    """A validated copy of the reference prevalence: declared labels, values in [0, 1] summing to 1."""
    if reference is None:
        return None
    values = {str(k): float(v) for k, v in reference.items()}
    if set(values) - set(labels):
        raise ValueError("reference_prevalence names labels that are not declared")
    if any(not 0.0 <= v <= 1.0 for v in values.values()):
        raise ValueError("reference_prevalence values must be in [0, 1]")
    if abs(math.fsum(values.values()) - 1.0) > PREVALENCE_TOLERANCE:
        raise ValueError("reference_prevalence values must sum to 1")
    return values


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
    threshold_tuning: Optional[ThresholdTuning] = None,
    reference_prevalence: Optional[Mapping[str, float]] = None,
    groups: Optional[Union[np.ndarray, Sequence[Any]]] = None,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
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
        n_dropped: Rows dropped before scoring; a non-negative int, required.
        split_report: The `SplitReport` of the splits this split came from, or `None`; its protocol must
            equal `protocol`.
        y_score: Binary: the positive label's score per row. Multi-class: an `(n, k)` array, columns in
            `labels` order. Must be finite.
        positive_label: The positive label of a binary task; required with binary scores, refused for
            multi-class.
        invalid_label: The prediction value that marks an output that could not be parsed; not a label.
        threshold: A threshold tuned on another split (compared without case or surrounding whitespace);
            predictions become the positive label where the score is at or above it. Its objective is
            recorded; the headline stays balanced accuracy.
        threshold_tuning: For predictions made outside this call: `validation` if their threshold was
            tuned on a validation split, `none` (the default) if no threshold was tuned. `test` is
            refused. With `threshold`, it may only be `validation` or omitted.
        reference_prevalence: Optional reference label distribution (for example the training set's):
            declared labels, values in [0, 1] summing to 1.
        groups: Optional group id per row (for example the subject); when given, intervals come from a
            cluster bootstrap over groups, scoring the same fixed predictions in every resample.
        n_resamples: Bootstrap resamples (default 2000).
        seed: Bootstrap seed (default 0).

    Returns:
        The report.

    Raises:
        ValueError: If the inputs are inconsistent (see the argument descriptions), or the threshold
            was tuned on the split being scored.
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}")
    if isinstance(n_dropped, bool) or not isinstance(n_dropped, (int, np.integer)) or n_dropped < 0:
        raise ValueError("n_dropped must be a non-negative int")
    if threshold_tuning == "test":
        raise ValueError("threshold_tuning 'test' is refused: a threshold tuned on the test split inflates the score")
    if threshold_tuning not in (None, "validation", "none"):
        raise ValueError("threshold_tuning must be 'validation' or 'none'")
    label_list = [str(lab) for lab in labels]
    # Balanced accuracy averages the recalls in scikit-learn's sorted label order, for exact equality.
    order = native_order(list(labels))
    if not label_list or len(set(label_list)) != len(label_list):
        raise ValueError("labels must be non-empty and unique")
    if invalid_label in label_list:
        raise ValueError("invalid_label must not be one of the labels")
    if split_report is not None and split_report.protocol != protocol:
        raise ValueError("the split report's protocol differs from protocol")
    truth = as_strings(y_true)
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
        if not np.isfinite(scores).all():
            raise ValueError("y_score must be finite (no NaN or infinity)")
    threshold_record = None
    if threshold is not None:
        if threshold_tuning == "none":
            raise ValueError("a tuned threshold cannot have threshold_tuning 'none'")
        if _same_split(threshold.split_name, split_name):
            raise ValueError("the threshold was tuned on the split being scored")
        if not math.isfinite(threshold.threshold):
            raise ValueError("the threshold must be finite")
        if y_pred is not None:
            raise ValueError("pass y_pred=None when a threshold sets the predictions")
        if scores is None or positive is None:
            raise ValueError("a threshold needs binary y_score and positive_label")
        negative = next(lab for lab in label_list if lab != positive)
        pred = np.where(scores >= threshold.threshold, positive, negative).astype(object)
        threshold_record = MappingProxyType(
            {"value": float(threshold.threshold), "objective": threshold.objective, "split_name": threshold.split_name}
        )
        tuning: Literal["validation", "none"] = "validation"
    else:
        if y_pred is None:
            raise ValueError("y_pred is required unless a threshold sets the predictions")
        pred = as_strings(y_pred)
        tuning = "validation" if threshold_tuning == "validation" else "none"
    if len(pred) != n:
        raise ValueError("y_true and y_pred must have the same length")
    if set(pred.tolist()) - set(label_list) - {invalid_label}:
        raise ValueError("y_pred holds values outside the declared labels and the invalid label")
    reference = _check_reference(reference_prevalence, label_list)

    true_counts = Counter(truth.tolist())
    pred_counts = Counter(pred.tolist())
    prevalence = {lab: true_counts[lab] / n for lab in label_list}
    predicted = {lab: pred_counts[lab] / n for lab in [*label_list, invalid_label]}
    absent = tuple(lab for lab in label_list if true_counts[lab] == 0)
    floor_k = 1.0 / (len(label_list) - len(absent))

    columns = (*label_list, invalid_label)
    (truth_codes, pred_codes), _ = encode_labels(truth, pred, vocabulary=columns)
    n_codes = len(columns)
    matrix = confusion(truth_codes, pred_codes, n_codes)
    per_class: Dict[str, Mapping[str, Any]] = {}
    for j, lab in enumerate(label_list):
        precision, recall, f1 = per_class_from_confusion(matrix, j)
        per_class[lab] = MappingProxyType(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(matrix[j].sum()),
                "prevalence": prevalence[lab],
            }
        )
    present_codes = [j for j, lab in enumerate(label_list) if true_counts[lab] > 0]
    confusion_table = MappingProxyType(
        {
            "rows": tuple(label_list),
            "columns": columns,
            "counts": tuple(tuple(int(c) for c in matrix[j]) for j in range(len(label_list))),
        }
    )
    secondary = {
        "accuracy": _metric(float(np.mean(truth == pred)), max(prevalence.values())),
        "f1_macro": _metric(macro_f1_from_confusion(matrix, present_codes), floor_k),
        "mcc": _metric(mcc_from_confusion(matrix), 0.0),
    }
    f1_positive = None if positive is None else per_class[positive]["f1"]
    if positive is not None and f1_positive is not None:
        p = prevalence[positive]
        secondary["f1"] = _metric(f1_positive, 2 * p / (1 + p))
    roc, ap = (None, None) if scores is None else _scores_and_floors(truth, scores, label_list, positive, prevalence)
    balanced = _metric(balanced_accuracy_from_confusion(matrix, order), floor_k)
    ci = None
    if groups is not None:
        group_arr = validate_groups(n, groups, n_resamples, 0.95)
        statistics = _resample_statistics(
            truth_codes, pred_codes, n_codes, present_codes, truth, scores, label_list, positive, order
        )
        ci = MappingProxyType(
            bootstrap_statistics(
                statistics,
                truth_codes,
                group_arr,
                n_resamples=n_resamples,
                seed=seed,
                confidence=0.95,
            )
        )
    return ClassificationReport(
        protocol=protocol,
        split_name=split_name,
        labels=tuple(label_list),
        n=n,
        n_dropped=int(n_dropped),
        invalid_rate=predicted[invalid_label],
        prevalence=MappingProxyType(prevalence),
        predicted_prevalence=MappingProxyType(predicted),
        reference_prevalence=None if reference is None else MappingProxyType(reference),
        absent_labels=absent,
        headline_value=balanced.value,
        balanced_accuracy=balanced,
        secondary=MappingProxyType(secondary),
        per_class=MappingProxyType(per_class),
        confusion_matrix=confusion_table,
        roc_auc=roc,
        average_precision=ap,
        positive_label=positive,
        threshold=threshold_record,
        threshold_tuning=tuning,
        split_report=split_report,
        ci=ci,
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
        reports=MappingProxyType(reports),
        prevalence_by_split=MappingProxyType({name: r.prevalence for name, r in reports.items()}),
    )
