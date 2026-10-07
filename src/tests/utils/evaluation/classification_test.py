"""Tests for the imbalance-aware classification report."""

from dataclasses import dataclass

import numpy as np
import pytest
from sklearn.metrics import recall_score

from senselab.utils.evaluation.classification import ClassificationReport, score_classification, score_splits
from senselab.utils.evaluation.splits import grouped_splits

BINARY = ["neg", "pos"]


@dataclass(frozen=True)
class _Threshold:
    threshold: float
    objective: str
    split_name: str


def _score(y_true: list, y_pred: list | None, **kwargs: object) -> ClassificationReport:
    fields: dict = {"labels": BINARY, "protocol": "unseen_subject", "split_name": "test", "n_dropped": 0}
    fields["split_report"] = None
    fields.update(kwargs)
    return score_classification(y_true, y_pred, **fields)


def _ninety_ten() -> list[str]:
    return ["neg"] * 900 + ["pos"] * 100


# ── The 90/10 trivial classifier is exposed ───────────────────────────


def test_trivial_classifier_sits_on_every_floor() -> None:
    """All-negative predictions with a constant score: every metric equals its floor."""
    y = _ninety_ten()
    report = _score(y, ["neg"] * 1000, y_score=np.zeros(1000), positive_label="pos")
    acc = report.secondary["accuracy"]
    assert acc.value == pytest.approx(0.9) and acc.floor == pytest.approx(0.9) and acc.lift == pytest.approx(0.0)
    assert report.balanced_accuracy.value == pytest.approx(0.5) == report.balanced_accuracy.floor
    assert report.secondary["mcc"].value == 0.0
    assert report.secondary["f1"].value == 0.0
    assert report.average_precision is not None and report.roc_auc is not None
    assert report.average_precision.value == pytest.approx(0.1) == report.average_precision.floor
    assert report.roc_auc.value == pytest.approx(0.5)


def test_informative_classifier_lifts_off_the_floors() -> None:
    """A classifier with real signal shows normalised lift above 0.3 for BA, MCC and AP."""
    rng = np.random.default_rng(0)
    y = np.array(_ninety_ten())
    scores = np.where(y == "pos", rng.normal(0.75, 0.12, 1000), rng.normal(0.3, 0.12, 1000))
    y_pred = np.where(scores >= 0.5, "pos", "neg")
    report = _score(list(y), list(y_pred), y_score=scores, positive_label="pos")
    assert report.average_precision is not None
    for metric in (report.balanced_accuracy, report.secondary["mcc"], report.average_precision):
        assert metric.normalized_lift is not None and metric.normalized_lift > 0.3


# ── headline and secondary ──────────────────────────────────────────────


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [(["neg", "pos", "pos"], ["neg", "pos", "neg"]), (["pos"] * 5 + ["neg"], ["pos"] * 6)],
)
def test_headline_is_always_balanced_accuracy(y_true: list, y_pred: list) -> None:
    """The headline is balanced accuracy; accuracy, macro-F1 and MCC are secondary with floors."""
    report = _score(y_true, y_pred)
    assert report.headline == "balanced_accuracy"
    assert report.headline_value == report.balanced_accuracy.value
    assert set(report.secondary) >= {"accuracy", "f1_macro", "mcc"}
    majority = max(y_true.count(lab) for lab in BINARY) / len(y_true)
    assert report.secondary["accuracy"].floor == pytest.approx(majority)
    assert not hasattr(report, "accuracy")


def test_no_headline_choice() -> None:
    """There is no parameter for choosing another headline."""
    with pytest.raises(TypeError):
        _score(["neg", "pos"], ["neg", "pos"], headline="accuracy")


def test_invalid_outputs_are_misses() -> None:
    """INVALID predictions lower recall, appear as a confusion column, and never as a row."""
    y_true = ["neg"] * 10 + ["pos"] * 10
    y_pred = ["neg"] * 10 + ["pos"] * 8 + ["INVALID"] * 2
    report = _score(y_true, y_pred)
    assert report.balanced_accuracy.value == pytest.approx((1.0 + 0.8) / 2)
    assert report.invalid_rate == pytest.approx(0.1)
    assert report.confusion_matrix["rows"] == BINARY
    assert report.confusion_matrix["columns"] == BINARY + ["INVALID"]
    assert report.confusion_matrix["counts"] == [[10, 0, 0], [0, 8, 2]]


def test_balanced_accuracy_is_the_macro_recall_formula() -> None:
    """S6 in-repo half: BA equals macro recall over the declared labels, INVALID a miss, on 20 cases."""
    labels = ["a", "b", "c"]
    for seed in range(20):
        rng = np.random.default_rng(seed)
        y_true = list(rng.choice(labels, 200, p=[0.6, 0.3, 0.1]))
        y_pred = list(rng.choice(labels + ["INVALID"], 200))
        report = _score(y_true, y_pred, labels=labels)
        expected = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        assert abs(report.balanced_accuracy.value - expected) <= 1e-12


def test_floors_match_closed_forms() -> None:
    """Floors for prevalences 0.7 / 0.2 / 0.1 match the documented table."""
    labels = ["a", "b", "c"]
    y_true = ["a"] * 70 + ["b"] * 20 + ["c"] * 10
    rng = np.random.default_rng(1)
    scores = rng.random((100, 3))
    report = _score(y_true, list(rng.choice(labels, 100)), labels=labels, y_score=scores)
    assert report.balanced_accuracy.floor == pytest.approx(1 / 3)
    assert report.secondary["accuracy"].floor == pytest.approx(0.7)
    assert report.secondary["mcc"].floor == 0.0
    assert report.secondary["f1_macro"].floor == pytest.approx((2 * 0.7 / 1.7) / 3)
    assert report.roc_auc is not None and report.roc_auc.floor == 0.5
    assert report.average_precision is not None and report.average_precision.floor == pytest.approx(1 / 3)
    assert "f1" not in report.secondary
    assert report.floors["balanced_accuracy"] == report.balanced_accuracy.floor


def test_binary_positive_class_f1_floor() -> None:
    """With a positive class, F1(pos) has the all-positive floor 2p/(1+p)."""
    report = _score(_ninety_ten(), ["pos"] * 1000, positive_label="pos")
    assert report.secondary["f1"].floor == pytest.approx(2 * 0.1 / 1.1)
    assert report.secondary["f1"].value == pytest.approx(2 * 0.1 / 1.1)


def test_absent_label_is_flagged_and_floor_counts_present_classes() -> None:
    """A declared label with no examples is listed, and the BA floor is 1/k over present classes."""
    report = _score(["a", "a", "b", "b"], ["a", "b", "b", "b"], labels=["a", "b", "c"])
    assert report.absent_labels == ["c"]
    assert report.balanced_accuracy.floor == pytest.approx(0.5)


# ── required context and inputs ─────────────────────────────────────────


def test_protocol_and_n_dropped_are_required() -> None:
    """A report without a protocol or a dropped-row count cannot be made."""
    with pytest.raises(TypeError):
        score_classification(["neg"], ["neg"], labels=BINARY, split_name="test", n_dropped=0, split_report=None)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        score_classification(  # type: ignore[call-arg]
            ["neg"], ["neg"], labels=BINARY, protocol="unseen_subject", split_name="test", split_report=None
        )


def test_report_carries_prevalence_and_reference() -> None:
    """Realised, predicted and reference prevalence sit side by side; n_dropped is echoed."""
    report = _score(
        ["neg", "neg", "neg", "pos"],
        ["neg", "pos", "neg", "pos"],
        reference_prevalence={"pos": 0.5, "neg": 0.5},
        n_dropped=3,
    )
    assert report.prevalence == {"neg": 0.75, "pos": 0.25}
    assert report.predicted_prevalence == {"neg": 0.5, "pos": 0.5, "INVALID": 0.0}
    assert report.reference_prevalence == {"pos": 0.5, "neg": 0.5}
    assert report.n == 4 and report.n_dropped == 3 and report.protocol == "unseen_subject"
    with pytest.raises(ValueError, match="reference_prevalence"):
        _score(["neg", "pos"], ["neg", "pos"], reference_prevalence={"other": 1.0})


@pytest.mark.parametrize(
    ("y_true", "y_pred", "message"),
    [(["neg", "maybe"], ["neg", "pos"], "y_true"), (["neg", "pos"], ["neg", "maybe"], "y_pred")],
)
def test_values_outside_the_labels_are_refused(y_true: list, y_pred: list, message: str) -> None:
    """True labels must be declared; predictions must be declared or INVALID."""
    with pytest.raises(ValueError, match=message):
        _score(y_true, y_pred)


def test_positive_label_rules() -> None:
    """Binary scores need a positive label; a positive label is refused for multi-class."""
    with pytest.raises(ValueError, match="positive_label"):
        _score(["neg", "pos"], ["neg", "pos"], y_score=np.array([0.1, 0.9]))
    with pytest.raises(ValueError, match="positive_label"):
        _score(["a", "b", "c"], ["a", "b", "c"], labels=["a", "b", "c"], positive_label="a")


def test_per_class_metrics() -> None:
    """Per-class precision, recall, F1, support and prevalence are reported."""
    report = _score(["neg", "neg", "pos", "pos"], ["neg", "pos", "pos", "pos"])
    assert report.per_class["pos"] == {
        "precision": pytest.approx(2 / 3),
        "recall": 1.0,
        "f1": pytest.approx(0.8),
        "support": 2,
        "prevalence": 0.5,
    }


# ── thresholds and split reports ────────────────────────────────────────


def test_threshold_is_recorded_and_used() -> None:
    """A tuned threshold sets the predictions; its objective and tuning split are recorded."""
    y = ["neg", "neg", "pos", "pos"]
    scores = np.array([0.1, 0.6, 0.55, 0.9])
    report = _score(y, None, y_score=scores, positive_label="pos", threshold=_Threshold(0.58, "f1", "val"))
    assert report.threshold == {"value": 0.58, "objective": "f1", "split_name": "val"}
    assert report.threshold_tuning == "validation"
    assert report.headline == "balanced_accuracy"
    assert report.balanced_accuracy.value == pytest.approx((0.5 + 0.5) / 2)
    assert _score(y, y).threshold_tuning == "none"


def test_threshold_from_the_scored_split_is_refused() -> None:
    """A threshold tuned on the split being scored is refused."""
    with pytest.raises(ValueError, match="tuned on the split"):
        _score(
            ["neg", "pos"],
            None,
            y_score=np.array([0.2, 0.8]),
            positive_label="pos",
            threshold=_Threshold(0.5, "balanced_accuracy", "test"),
        )
    with pytest.raises(ValueError, match="y_pred"):
        _score(
            ["neg", "pos"],
            ["neg", "pos"],
            y_score=np.array([0.2, 0.8]),
            positive_label="pos",
            threshold=_Threshold(0.5, "f1", "val"),
        )


def test_split_report_is_attached_and_must_match_the_protocol() -> None:
    """The split report travels with the metrics, and its protocol must agree."""
    labels = np.array([0, 1] * 30)
    subjects = np.repeat([f"s{i}" for i in range(12)], 5)
    _, split_report = grouped_splits(
        labels, subjects, protocol="unseen_subject", scheme="stratified_group_kfold", seed=0, n_splits=3
    )
    report = _score(["neg", "pos"], ["neg", "pos"], split_report=split_report)
    assert report.split_report is split_report
    with pytest.raises(ValueError, match="protocol"):
        _score(["neg", "pos"], ["neg", "pos"], protocol="seen_subject", split_report=split_report)


def test_score_splits_reports_prevalence_by_split() -> None:
    """Several splits give one report each and a prevalence table; protocols must agree."""
    common = {"labels": BINARY, "protocol": "unseen_subject", "n_dropped": 0, "split_report": None}
    evaluation = score_splits(
        {
            "val": {"y_true": ["neg", "pos", "pos"], "y_pred": ["neg", "pos", "neg"], **common},
            "test": {"y_true": ["neg", "neg", "pos", "pos"], "y_pred": ["neg", "neg", "pos", "neg"], **common},
        }
    )
    assert set(evaluation.reports) == {"val", "test"}
    assert evaluation.prevalence_by_split == {"val": {"neg": 1 / 3, "pos": 2 / 3}, "test": {"neg": 0.5, "pos": 0.5}}
    assert evaluation.protocol == "unseen_subject"
    with pytest.raises(ValueError, match="protocol"):
        score_splits(
            {
                "val": {"y_true": ["neg"], "y_pred": ["neg"], **common},
                "test": {"y_true": ["neg"], "y_pred": ["neg"], **{**common, "protocol": "seen_subject"}},
            }
        )
