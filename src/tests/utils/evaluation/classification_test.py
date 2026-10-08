"""Tests for the imbalance-aware classification report."""

import json
import math
import warnings
from dataclasses import dataclass

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, recall_score

from senselab.utils.evaluation.classification import ClassificationReport, score_classification, score_splits
from senselab.utils.evaluation.metrics import balanced_accuracy, legacy_macro_recall
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


def test_trivial_classifier_sits_on_every_trivial_floor_it_targets() -> None:
    """All-negative predictions with a constant score sit on the floors of the metrics they target.

    Accuracy, balanced accuracy, MCC, ROC-AUC and AP equal their floors. F1(pos) is 0, below its floor
    2p/(1+p): that floor is the all-positive predictor's, which this classifier is not.
    """
    y = _ninety_ten()
    report = _score(y, ["neg"] * 1000, y_score=np.zeros(1000), positive_label="pos")
    acc = report.secondary["accuracy"]
    assert acc.value == pytest.approx(0.9) and acc.floor == pytest.approx(0.9) and acc.lift == pytest.approx(0.0)
    assert report.balanced_accuracy.value == pytest.approx(0.5) == report.balanced_accuracy.floor
    assert report.secondary["mcc"].value == 0.0
    assert report.secondary["f1"].value == 0.0 < report.secondary["f1"].floor
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
    assert report.confusion_matrix["rows"] == tuple(BINARY)
    assert report.confusion_matrix["columns"] == (*BINARY, "INVALID")
    assert report.confusion_matrix["counts"] == ((10, 0, 0), (0, 8, 2))


def _random_cases(n_cases: int, with_absent: bool) -> list[tuple[list, list, list]]:
    """Seeded multi-class cases with INVALID predictions; every other case drops label ``d`` from y_true."""
    labels = ["a", "b", "c", "d"]
    cases = []
    for seed in range(n_cases):
        rng = np.random.default_rng(seed)
        present = labels[:3] if with_absent and seed % 2 == 0 else labels
        y_true = list(rng.choice(present, 200))
        y_pred = list(rng.choice(labels + ["INVALID"], 200))
        cases.append((labels, y_true, y_pred))
    return cases


def test_headline_is_present_label_balanced_accuracy() -> None:
    """The headline equals scikit-learn's balanced_accuracy_score, absent labels and INVALID included."""
    for labels, y_true, y_pred in _random_cases(20, with_absent=True):
        report = _score(y_true, y_pred, labels=labels)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # scikit-learn warns that INVALID never occurs in y_true
            expected = balanced_accuracy_score(y_true, y_pred)
        assert abs(report.headline_value - expected) <= 1e-12
        assert abs(balanced_accuracy(y_true, y_pred) - expected) <= 1e-12


def test_legacy_macro_recall_is_the_declared_label_formula() -> None:
    """Parity helper: macro recall over all declared labels, zero_division=0, absent labels included."""
    for labels, y_true, y_pred in _random_cases(20, with_absent=True):
        expected = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        assert abs(legacy_macro_recall(y_true, y_pred, labels=labels) - expected) <= 1e-12


@pytest.mark.parametrize(
    ("y_true", "y_pred", "present", "legacy"),
    [
        (["a"] * 6 + ["b"] * 4, ["a"] * 10, 0.5, 1 / 3),
        (["a", "a", "b", "b"], ["a", "b", "b", "b"], 0.75, 0.5),
    ],
)
def test_the_two_formulas_differ_on_absent_labels(y_true: list, y_pred: list, present: float, legacy: float) -> None:
    """With label ``c`` declared but absent, the headline averages present labels; the legacy helper all three."""
    report = _score(y_true, y_pred, labels=["a", "b", "c"])
    assert report.headline_value == pytest.approx(present)
    assert legacy_macro_recall(y_true, y_pred, labels=["a", "b", "c"]) == pytest.approx(legacy)


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
    assert report.secondary["f1_macro"].floor == pytest.approx(1 / 3)
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
    assert report.absent_labels == ("c",)
    assert report.balanced_accuracy.floor == pytest.approx(0.5)
    assert report.secondary["f1_macro"].floor == pytest.approx(0.5)


def test_absent_label_per_class_values_are_undefined() -> None:
    """An absent label's recall and F1 are None; its precision is a number only if it was predicted."""
    unpredicted = _score(["a", "b"], ["a", "b"], labels=["a", "b", "c"]).per_class["c"]
    assert (unpredicted["recall"], unpredicted["f1"], unpredicted["precision"]) == (None, None, None)
    predicted = _score(["a", "b"], ["a", "c"], labels=["a", "b", "c"]).per_class["c"]
    assert (predicted["recall"], predicted["f1"], predicted["precision"]) == (None, None, 0.0)


def test_prior_matched_random_predictor_has_no_macro_f1_lift() -> None:
    """A label-blind predictor drawing labels at the true prevalences sits on the 1/k macro-F1 floor."""
    rng = np.random.default_rng(0)
    labels = ["a", "b", "c"]
    p = [0.6, 0.3, 0.1]
    y_true = list(rng.choice(labels, 50_000, p=p))
    y_pred = list(rng.choice(labels, 50_000, p=p))
    report = _score(y_true, y_pred, labels=labels)
    assert abs(report.secondary["f1_macro"].lift) < 0.01
    assert abs(report.balanced_accuracy.lift) < 0.01


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


# ── Review fixes ──────────────────────────────────────────────────────


def test_scored_split_check_ignores_case_and_whitespace() -> None:
    """A threshold tuned on " Test " is still the scored split "test"."""
    with pytest.raises(ValueError, match="tuned on the split"):
        _score(
            ["neg", "pos"],
            None,
            y_score=np.array([0.2, 0.8]),
            positive_label="pos",
            threshold=_Threshold(0.5, "f1", " Test "),
        )


@pytest.mark.parametrize(
    ("threshold", "scores"),
    [(math.nan, np.array([0.2, 0.8])), (0.5, np.array([0.2, math.nan]))],
    ids=["nan-threshold", "nan-score"],
)
def test_nan_threshold_or_scores_are_refused(threshold: float, scores: np.ndarray) -> None:
    """NaN in the threshold or the scores is a ValueError, not a silent negative prediction."""
    with pytest.raises(ValueError, match="finite"):
        _score(["neg", "pos"], None, y_score=scores, positive_label="pos", threshold=_Threshold(threshold, "f1", "val"))


def test_threshold_tuning_for_external_predictions() -> None:
    """External predictions declare how their threshold was tuned; ``test`` is refused."""
    assert _score(["neg", "pos"], ["neg", "pos"]).threshold_tuning == "none"
    assert _score(["neg", "pos"], ["neg", "pos"], threshold_tuning="validation").threshold_tuning == "validation"
    with pytest.raises(ValueError, match="test"):
        _score(["neg", "pos"], ["neg", "pos"], threshold_tuning="test")


@pytest.mark.parametrize("n_dropped", [-1, True, 1.5])
def test_n_dropped_is_a_non_negative_int(n_dropped: object) -> None:
    """The dropped-row count is a non-negative integer."""
    with pytest.raises(ValueError, match="n_dropped"):
        _score(["neg", "pos"], ["neg", "pos"], n_dropped=n_dropped)


@pytest.mark.parametrize("reference", [{"neg": 0.5, "pos": 0.6}, {"neg": 1.2, "pos": -0.2}])
def test_reference_prevalence_is_a_distribution(reference: dict) -> None:
    """Reference prevalences lie in [0, 1] and sum to 1."""
    with pytest.raises(ValueError, match="reference_prevalence"):
        _score(["neg", "pos"], ["neg", "pos"], reference_prevalence=reference)


def test_report_mappings_are_read_only() -> None:
    """The report's mappings cannot be changed after scoring."""
    report = _score(["neg", "pos"], ["neg", "pos"])
    with pytest.raises(TypeError):
        report.prevalence["neg"] = 1.0  # type: ignore[index]
    with pytest.raises(TypeError):
        report.per_class["neg"]["recall"] = 0.0  # type: ignore[index]
    with pytest.raises(TypeError):
        report.secondary["mcc"] = None  # type: ignore[index]


def test_to_dict_is_plain_and_carries_floors() -> None:
    """``to_dict`` is JSON-serialisable and includes the floors, lifts and normalised lifts."""
    report = _score(_ninety_ten(), ["neg"] * 1000, y_score=np.zeros(1000), positive_label="pos")
    data = report.to_dict()
    json.dumps(data)
    assert data["headline"] == "balanced_accuracy" and data["protocol"] == "unseen_subject"
    assert data["floors"]["balanced_accuracy"] == pytest.approx(0.5)
    assert set(data["lift"]) == set(data["floors"]) == set(data["normalized_lift"])
    assert data["secondary"]["accuracy"]["floor"] == pytest.approx(0.9)


# ── Review fixes ──────────────────────────────────────────────────────


def test_macro_f1_averages_over_present_labels() -> None:
    """With c declared but absent and everything predicted a, macro-F1 is (0.75 + 0) / 2, not / 3."""
    report = _score(["a"] * 6 + ["b"] * 4, ["a"] * 10, labels=["a", "b", "c"])
    assert report.secondary["f1_macro"].value == pytest.approx(0.375)


def test_positive_f1_is_omitted_when_the_positive_label_is_absent() -> None:
    """With no positive example, positive-class F1 is undefined: omitted, as per_class reports None."""
    report = _score(["neg"] * 5, ["neg"] * 4 + ["pos"], positive_label="pos")
    assert "f1" not in report.secondary
    assert report.per_class["pos"]["f1"] is None


# ── Order-independent balanced accuracy ───────────────────────────────


def test_headline_does_not_depend_on_label_order() -> None:
    """The reviewer's case: with labels declared c, a, b the headline matches scikit-learn."""
    y_true = list("abbcccacccaacbbabbccbaca")
    y_pred = list("abcbccbaacaacbabaccabaaa")
    expected = balanced_accuracy_score(y_true, y_pred)
    for labels in (["c", "a", "b"], ["a", "b", "c"], ["b", "c", "a"]):
        report = _score(y_true, y_pred, labels=labels)
        assert abs(report.headline_value - expected) <= 1e-15
    values = {_score(y_true, y_pred, labels=order).headline_value for order in (["c", "a", "b"], ["a", "b", "c"])}
    assert len(values) == 1


@pytest.mark.parametrize("labels", [["c", "a", "b"], [2, 3, 10], ["z", "y", "x", "w"]])
def test_headline_matches_scikit_learn_for_unsorted_labels(labels: list) -> None:
    """On random cases with unsorted or integer labels, the headline is within 1e-15 of scikit-learn."""
    rng = np.random.default_rng(0)
    for _ in range(300):
        n = int(rng.integers(6, 40))
        y = rng.choice(np.array(labels, dtype=object), size=n)
        p = np.where(rng.random(n) < 0.5, y, rng.choice(np.array(labels, dtype=object), size=n))
        if len(set(y.tolist())) < 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expected = balanced_accuracy_score(y.tolist(), p.tolist())
        assert abs(balanced_accuracy(y, p) - expected) <= 1e-15
