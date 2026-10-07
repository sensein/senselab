"""Tests for the metrics vocabulary and validator."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from senselab.utils.reproduce.metrics import (
    CODES,
    DRIVER_KEYS,
    WARNINGS,
    MetricsError,
    expected_floor,
    metric_address_error,
    metric_value,
    validate_metrics,
)
from senselab.utils.reproduce.spec import MetricsSchema


def _schema(**over: Any) -> MetricsSchema:  # noqa: ANN401
    data: dict[str, Any] = {
        "kind": "classification",
        "headline": ["balanced_accuracy", "mcc"],
        "headline_labels": ["sys_a", "sys_b"],
        "counts": ["n_systems", "n_flagged"],
        "per_class": ["pos", "neg"],
    }
    data.update(over)
    return MetricsSchema.model_validate(data)


def _split(**over: Any) -> dict[str, Any]:  # noqa: ANN401
    split: dict[str, Any] = {
        "protocol": "seen_subject",
        "unit": "subject",
        "stratify": "class",
        "folds": [
            {"n_groups": 12, "prevalence": {"pos": 0.4, "neg": 0.6}},
            {"n_groups": 3, "prevalence": {"pos": 0.5, "neg": 0.5}},
        ],
        "invariant_ok": True,
    }
    split.update(over)
    return split


def _obj() -> dict[str, Any]:
    return {
        "headline": [
            {"name": "balanced_accuracy", "label": "sys_a", "value": 0.8, "floor": 0.5, "lift": 0.3, "n": 3},
            {
                "name": "balanced_accuracy",
                "label": "sys_b",
                "value": 0.7,
                "ci_low": 0.6,
                "ci_high": 0.8,
                "floor": 0.5,
                "lift": 0.2,
            },
            {
                "name": "mcc",
                "value": 0.4,
                "floor": 0.0,
                "lift": 0.4,
                "n_folds": 3,
                "numdf": 3,
                "dendf": 60.0,
                "unit": "fraction",
            },
        ],
        "per_class": {
            "pos": {"recall": 1.0, "precision": 0.476, "f1": 0.645, "prevalence": 0.4},
            "neg": {"recall": 0.2, "precision": 0.9, "prevalence": 0.6},
        },
        "confusion": {"tp": 10, "fp": 11, "fn": 0, "tn": 6},
        "counts": {"n_systems": 3, "n_flagged": 40},
        "split_report": _split(),
        "threshold_tuning": "validation",
        "n": 120,
        "n_folds": 3,
    }


def _codes(obj: Any, schema: MetricsSchema | None = None, **kw: Any) -> list[str]:  # noqa: ANN401
    clean, codes = validate_metrics(obj, schema or _schema(), **kw)
    assert (clean is None) == any(c.startswith("E_") for c in codes)
    return codes


def _entry(name: str, value: float, floor: float, **extra: Any) -> dict[str, Any]:  # noqa: ANN401
    return {"name": name, "value": value, "floor": floor, "lift": value - floor, **extra}


def test_valid_output_passes_and_is_suppressed() -> None:
    """Valid output passes; small n, counts, confusion and person-unit n_groups are suppressed, df is not."""
    obj = _obj()
    clean, codes = validate_metrics(obj, _schema())
    assert codes == []
    assert clean is not None
    assert clean["confusion"] == {"tp": 10, "fp": 11, "fn": "<5", "tn": 6}
    assert clean["headline"][0]["n"] == "<5"
    assert clean["counts"] == {"n_systems": "<5", "n_flagged": 40}
    assert clean["n"] == 120
    assert clean["per_class"]["pos"]["recall"] == 1.0
    assert clean["headline"][2]["numdf"] == 3
    assert clean["headline"][2]["n_folds"] == 3
    assert clean["n_folds"] == 3
    assert [f["n_groups"] for f in clean["split_report"]["folds"]] == [12, "<5"]
    assert obj["confusion"]["fn"] == 0, "the input is not modified"


@pytest.mark.parametrize(("unit", "kept"), [("subject", False), ("family", False), (None, False), ("video", True)])
def test_n_groups_suppressed_only_for_person_units(unit: str | None, kept: bool) -> None:
    """A fold's n_groups is suppressed when the unit is a person or not given, and kept otherwise."""
    obj = _obj()
    obj["split_report"] = _split(unit=unit) if unit else {k: v for k, v in _split().items() if k != "unit"}
    clean, codes = validate_metrics(obj, _schema())
    assert codes == []
    assert clean is not None
    assert clean["split_report"]["folds"][1]["n_groups"] == (3 if kept else "<5")


def test_min_cell_is_configurable() -> None:
    """The suppression threshold and its token follow ``min_cell``."""
    clean, codes = validate_metrics(_obj(), _schema(), min_cell=11)
    assert codes == []
    assert clean is not None
    assert clean["confusion"] == {"tp": "<11", "fp": 11, "fn": "<11", "tn": "<11"}


def test_already_suppressed_token_is_accepted() -> None:
    """A cell already replaced by the token passes again with the same ``min_cell``."""
    obj = _obj()
    obj["confusion"]["fn"] = "<5"
    assert _codes(obj) == []
    assert "E_METRICS_NOT_NUMBER" in _codes(obj, min_cell=10)


@pytest.mark.parametrize(
    ("path", "value", "code"),
    [
        (("extra",), 1, "E_METRICS_UNKNOWN_KEY"),
        (("per_class", "pos", "auc"), 0.5, "E_METRICS_UNKNOWN_KEY"),
        (("split_report", "seed"), 1, "E_METRICS_UNKNOWN_KEY"),
        (("headline", 0, "value"), "0.8", "E_METRICS_NOT_NUMBER"),
        (("headline", 0, "value"), True, "E_METRICS_NOT_NUMBER"),
        (("headline", 0, "value"), float("nan"), "E_METRICS_NOT_NUMBER"),
        (("headline", 0, "value"), None, "E_METRICS_NOT_NUMBER"),
        (("per_class", "pos", "recall"), "1.0", "E_METRICS_NOT_NUMBER"),
        (("confusion", "fn"), "zero", "E_METRICS_NOT_NUMBER"),
        (("confusion", "fn"), 1.5, "E_METRICS_NOT_INT"),
        (("counts", "n_systems"), -1, "E_METRICS_NEGATIVE"),
        (("per_class", "child_17"), {"prevalence": 0.1}, "E_METRICS_UNDECLARED_LABEL"),
        (("headline", 0, "label"), "sys_z", "E_METRICS_UNDECLARED_LABEL"),
        (("headline", 0, "name"), "top1", "E_METRICS_UNDECLARED_HEADLINE"),
        (("headline", 0, "name"), "anything", "E_METRICS_UNDECLARED_HEADLINE"),
        (("headline", 0, "name"), ["mcc"], "E_METRICS_UNDECLARED_HEADLINE"),
        (("headline", 0, "label"), ["sys_a"], "E_METRICS_UNDECLARED_LABEL"),
        (("counts", "n_rows"), 9, "E_METRICS_UNDECLARED_COUNT"),
        (("headline", 0, "unit"), "percent", "E_METRICS_BAD_TOKEN"),
        (("headline",), {"name": "mcc"}, "E_METRICS_BAD_SHAPE"),
        (("per_class",), [], "E_METRICS_BAD_SHAPE"),
        (("headline", 0, "lift"), 0.2, "E_METRICS_LIFT_MISMATCH"),
        (("headline", 0, "floor"), 0.6, "E_METRICS_BAD_FLOOR"),
        (("headline", 2, "floor"), 0.1, "E_METRICS_BAD_FLOOR"),
        (("threshold_tuning",), "train", "E_METRICS_BAD_TUNING_SPLIT"),
        (("per_class", "pos", "prevalence"), 1.4, "E_METRICS_PREVALENCE_RANGE"),
        (("per_class", "pos", "prevalence"), 0.5, "E_METRICS_PREVALENCE_SUM"),
    ],
)
def test_rejections(path: tuple[Any, ...], value: Any, code: str) -> None:  # noqa: ANN401
    """Each single change outside the vocabulary or schema gives its code."""
    obj = _obj()
    target = obj
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert code in _codes(obj)


def test_top_level_must_be_an_object() -> None:
    """Top level must be an object."""
    assert _codes([1, 2]) == ["E_METRICS_NOT_OBJECT"]


@pytest.mark.parametrize("key", sorted(DRIVER_KEYS))
@pytest.mark.parametrize("where", ["top", "headline", "per_class"])
def test_adapter_may_not_write_driver_keys(key: str, where: str) -> None:
    """Adapter may not write driver keys."""
    obj = _obj()
    value: Any = True if key == "pass" else 0.1
    {"top": obj, "headline": obj["headline"][0], "per_class": obj["per_class"]["pos"]}[where][key] = value
    assert _codes(obj) == ["E_METRICS_DRIVER_KEY"]


def test_driver_may_write_driver_keys() -> None:
    """Driver may write driver keys."""
    obj = _obj()
    obj["headline"][0].update({"target": 0.79, "tolerance": 0.01, "delta": 0.01, "pass": True})
    obj["pass"] = True
    assert _codes(obj, writer="driver") == []
    obj["pass"] = 1
    assert _codes(obj, writer="driver") == ["E_METRICS_NOT_BOOL"]
    obj["pass"] = True
    obj["headline"][0]["target"] = "0.79"
    assert _codes(obj, writer="driver") == ["E_METRICS_NOT_NUMBER"]


def test_classification_without_prevalence_is_rejected() -> None:
    """Classification without prevalence is rejected."""
    obj = _obj()
    del obj["per_class"]["neg"]["prevalence"]
    assert _codes(obj) == ["E_METRICS_NO_PREVALENCE"]
    obj = _obj()
    del obj["per_class"]
    assert _codes(obj) == ["E_METRICS_NO_CLASS_RATES", "E_METRICS_NO_PREVALENCE"]
    obj = _obj()
    del obj["per_class"]["neg"]
    assert _codes(obj) == ["E_METRICS_NO_CLASS_RATES", "E_METRICS_NO_PREVALENCE"]


@pytest.mark.parametrize("field", ["recall", "precision"])
def test_classification_needs_recall_and_precision_per_class(field: str) -> None:
    """Every declared class carries recall and precision as well as prevalence."""
    obj = _obj()
    del obj["per_class"]["neg"][field]
    assert _codes(obj) == ["E_METRICS_NO_CLASS_RATES"]


@pytest.mark.parametrize("field", ["floor", "lift"])
def test_classification_without_floor_or_lift_is_rejected(field: str) -> None:
    """Every headline entry of a classification step carries a floor and a lift."""
    obj = _obj()
    del obj["headline"][1][field]
    assert _codes(obj) == ["E_METRICS_NO_FLOOR" if field == "floor" else "E_METRICS_NO_LIFT"]


def test_classification_without_tuning_split_is_rejected() -> None:
    """Classification without tuning split is rejected."""
    obj = _obj()
    del obj["threshold_tuning"]
    assert _codes(obj) == ["E_METRICS_NO_TUNING_SPLIT"]


def test_threshold_tuned_on_test_is_a_warning() -> None:
    """``test`` is recorded honestly: the object passes and carries ``W_TUNED_ON_TEST``."""
    obj = _obj()
    obj["threshold_tuning"] = "test"
    clean, codes = validate_metrics(obj, _schema())
    assert codes == ["W_TUNED_ON_TEST"]
    assert clean is not None and clean["threshold_tuning"] == "test"
    assert set(codes) <= WARNINGS
    obj["extra"] = 1
    assert _codes(obj) == ["E_METRICS_UNKNOWN_KEY", "W_TUNED_ON_TEST"]


@pytest.mark.parametrize(
    ("name", "k", "positive", "expected"),
    [
        ("balanced_accuracy", 2, None, 0.5),
        ("macro_recall", 4, None, 0.25),
        ("auroc", 3, None, 0.5),
        ("mcc", 2, None, 0.0),
        ("accuracy", 2, None, 0.6),
        ("top1", 2, None, 0.6),
        ("pr_auc", 2, "pos", 0.4),
        ("f1", 2, "pos", 2 * 0.4 / 1.4),
        ("pr_auc", 2, None, None),
        ("f1", 2, None, None),
        ("macro_f1", 2, None, None),
        ("avg_map", 2, None, None),
        ("map_at", 2, None, None),
    ],
)
def test_expected_floor(name: str, k: int, positive: str | None, expected: float | None) -> None:
    """Standard floors follow the class count and prevalences; the rest are presence-only."""
    assert expected_floor(name, {"pos": 0.4, "neg": 0.6}, k=k, positive=positive) == (
        pytest.approx(expected) if expected is not None else None
    )


def test_floors_are_checked_for_every_classification_name() -> None:
    """A binary step with a positive class has every standard floor checked, in pp where declared."""
    schema = _schema(
        headline=["balanced_accuracy", "auroc", "accuracy", "top1", "pr_auc", "f1", "macro_f1", "mcc"],
        positive="pos",
        headline_labels=[],
    )
    obj = _obj()
    obj["headline"] = [
        _entry("balanced_accuracy", 80.0, 50.0, unit="pp"),
        _entry("auroc", 0.7, 0.5),
        _entry("accuracy", 0.7, 0.6),
        _entry("top1", 70.0, 60.0, unit="pp"),
        _entry("pr_auc", 0.6, 0.4),
        _entry("f1", 0.7, 2 * 0.4 / 1.4),
        _entry("macro_f1", 0.7, 0.123),
        _entry("mcc", 0.3, 0.0),
    ]
    assert _codes(copy.deepcopy(obj), schema) == []
    for index, wrong in [(0, 0.4), (3, 0.5), (4, 0.5), (5, 0.4)]:
        bad = copy.deepcopy(obj)
        bad["headline"][index] = _entry(bad["headline"][index]["name"], 0.9, wrong)
        assert _codes(bad, schema) == ["E_METRICS_BAD_FLOOR"], index


def test_detection_kind_needs_prevalence_floor_and_lift() -> None:
    """Detection needs prevalence, floor and lift; its floors are presence-only and rates are optional."""
    schema = _schema(kind="detection", headline=["avg_map", "map_at"], headline_labels=[], counts=[])
    obj: dict[str, Any] = {
        "headline": [{"name": "avg_map", "value": 0.0595}, _entry("map_at", 0.0948, 0.02)],
        "per_class": {"pos": {"prevalence": 0.3}, "neg": {"prevalence": 0.9}},
        "threshold_tuning": "none",
        "split_report": {"protocol": "unseen_subject", "invariant_ok": True, "folds": [{"prevalence": 0.4}]},
    }
    assert _codes(copy.deepcopy(obj), schema) == ["E_METRICS_NO_FLOOR", "E_METRICS_NO_LIFT"]
    obj["headline"][0] = _entry("avg_map", 0.0595, 0.01)
    assert _codes(obj, schema) == []


def test_statistic_kind_needs_no_floor_prevalence_or_split() -> None:
    """Statistic kind needs no floor, prevalence or split report."""
    schema = MetricsSchema.model_validate({"kind": "statistic", "headline": ["f_stat"]})
    obj = {"headline": [{"name": "f_stat", "value": 4.2, "numdf": 3, "dendf": 60}, {"name": "f_stat", "value": 1.1}]}
    clean, codes = validate_metrics(obj, schema)
    assert codes == []
    assert clean == obj


def test_declared_headline_and_counts_must_be_present() -> None:
    """Declared headline and counts must be present."""
    obj = _obj()
    obj["headline"] = [h for h in obj["headline"] if h["name"] != "mcc"]
    assert _codes(obj) == ["E_METRICS_MISSING_HEADLINE"]
    obj = _obj()
    del obj["counts"]["n_flagged"]
    assert _codes(obj) == ["E_METRICS_MISSING_COUNT"]


def test_count_headline_is_an_integer_and_suppressed() -> None:
    """A ``count`` headline value is an integer and is suppressed below the minimum cell."""
    schema = MetricsSchema.model_validate({"kind": "count", "headline": ["count"]})
    assert _codes({"headline": [{"name": "count", "value": 2.5}]}, schema) == ["E_METRICS_NOT_INT"]
    clean, codes = validate_metrics({"headline": [{"name": "count", "value": 2}]}, schema)
    assert codes == [] and clean == {"headline": [{"name": "count", "value": "<5"}]}
    clean, _ = validate_metrics({"headline": [{"name": "count", "value": 7}]}, schema)
    assert clean == {"headline": [{"name": "count", "value": 7}]}


def test_split_report_both_protocols_pass() -> None:
    """Both protocols are valid, whatever the invariant outcome."""
    obj = _obj()
    obj["split_report"] = _split(protocol="seen_subject", invariant_ok=True)
    assert _codes(obj) == []
    obj["split_report"] = _split(protocol="unseen_subject", invariant_ok=False)
    assert _codes(obj) == []


def test_split_report_minimal() -> None:
    """A split report needs only the protocol and the invariant result."""
    obj = _obj()
    obj["split_report"] = {"protocol": "unseen_subject", "invariant_ok": True}
    assert _codes(obj) == []


def test_protocol_none_labels_a_split_without_protocol() -> None:
    """``protocol: none`` is accepted only with ``invariant_ok: false``."""
    obj = _obj()
    obj["split_report"] = _split(protocol="none", unit="row", invariant_ok=False)
    assert _codes(obj) == []
    obj["split_report"] = _split(protocol="none", unit="row", invariant_ok=True)
    assert _codes(obj) == ["E_SPLIT_BAD_INVARIANT"]


def test_split_report_required_unless_opted_out() -> None:
    """A classification step needs a split report unless its schema opts out, and then may not give one."""
    obj = _obj()
    del obj["split_report"]
    assert _codes(copy.deepcopy(obj)) == ["E_SPLIT_MISSING"]
    opted = _schema(split={"none": "whole_dataset"})
    assert _codes(copy.deepcopy(obj), opted) == []
    assert _codes(_obj(), opted) == ["E_SPLIT_UNEXPECTED"]


@pytest.mark.parametrize(
    ("over", "drop", "code"),
    [
        ({}, "protocol", "E_SPLIT_NO_PROTOCOL"),
        ({"protocol": "within_subject_leak"}, None, "E_SPLIT_BAD_PROTOCOL"),
        ({}, "invariant_ok", "E_SPLIT_BAD_INVARIANT"),
        ({"invariant_ok": 1}, None, "E_SPLIT_BAD_INVARIANT"),
        ({"invariant_ok": "true"}, None, "E_SPLIT_BAD_INVARIANT"),
        ({"unit": "sub_0123"}, None, "E_METRICS_BAD_TOKEN"),
        ({"unit": "participant"}, None, "E_METRICS_BAD_TOKEN"),
        ({"stratify": "class_language"}, None, "E_METRICS_BAD_TOKEN"),
        ({"folds": [{"n_groups": 2.5, "prevalence": {"pos": 0.5, "neg": 0.5}}]}, None, "E_METRICS_NOT_INT"),
        ({"folds": [{"prevalence": {"pos": 0.5, "neg": 0.4, "child_9": 0.1}}]}, None, "E_METRICS_UNDECLARED_LABEL"),
        ({"folds": [{"prevalence": {"pos": 1.0}}]}, None, "E_SPLIT_FOLD_PREVALENCE"),
        ({"folds": [{"prevalence": 0.5}]}, None, "E_SPLIT_FOLD_PREVALENCE"),
        ({"folds": [{"n_groups": 10}]}, None, "E_SPLIT_FOLD_PREVALENCE"),
        ({"folds": [{"prevalence": {"pos": 0.5, "neg": 0.6}}]}, None, "E_METRICS_PREVALENCE_SUM"),
        ({"folds": [{"prevalence": {"pos": -0.5, "neg": 1.5}}]}, None, "E_METRICS_PREVALENCE_RANGE"),
        ({"folds": {"n_groups": 2}}, None, "E_METRICS_BAD_SHAPE"),
    ],
)
def test_split_report_rejections(over: dict[str, Any], drop: str | None, code: str) -> None:
    """A missing or unknown protocol, a non-bool invariant, unlisted words and incomplete prevalences are refused."""
    obj = _obj()
    split = _split(**over)
    if drop:
        del split[drop]
    obj["split_report"] = split
    assert code in _codes(obj)


@pytest.mark.parametrize("unit", ["subject", "family", "session", "video", "clip", "recording", "row"])
@pytest.mark.parametrize("stratify", ["none", "class", "class_timepoint", "class_site", "class_age", "timepoint"])
def test_split_words(unit: str, stratify: str) -> None:
    """Every listed unit and stratification is accepted."""
    obj = _obj()
    obj["split_report"] = _split(unit=unit, stratify=stratify)
    assert _codes(obj) == []


def test_rejection_never_echoes_content() -> None:
    """Rejection never echoes content."""
    obj = _obj()
    obj["participant_secret_key"] = "secret-value"
    obj["per_class"]["secret_label"] = {"prevalence": 0.1}
    _, codes = validate_metrics(obj, _schema())
    assert all(c in CODES for c in codes)
    assert not any("secret" in c for c in codes)


def test_codes_are_sorted_and_unique() -> None:
    """Codes are sorted and unique."""
    obj = _obj()
    obj["a"] = 1
    obj["b"] = 2
    obj["headline"][2]["value"] = "x"
    codes = _codes(obj)
    assert codes == sorted(set(codes))
    assert codes == ["E_METRICS_NOT_NUMBER", "E_METRICS_UNKNOWN_KEY"]


@pytest.mark.parametrize(
    ("address", "error"),
    [
        ("mcc", None),
        ("balanced_accuracy@sys_a", None),
        ("counts.n_systems", None),
        ("per_class.pos.recall", None),
        ("confusion.tp", None),
        ("top1", "E_METRICS_UNDECLARED_HEADLINE"),
        ("balanced_accuracy@sys_z", "E_METRICS_UNDECLARED_LABEL"),
        ("counts.n_rows", "E_METRICS_UNDECLARED_COUNT"),
        ("per_class.child.recall", "E_METRICS_UNDECLARED_LABEL"),
        ("per_class.pos.auc", "E_METRICS_BAD_ADDRESS"),
        ("confusion.xx", "E_METRICS_BAD_ADDRESS"),
        ("a.b.c.d", "E_METRICS_BAD_ADDRESS"),
    ],
)
def test_metric_address(address: str, error: str | None) -> None:
    """Addresses are checked against the schema's declared names and labels."""
    assert metric_address_error(address, _schema()) == error


def test_metric_value_resolves_and_refuses_ambiguity() -> None:
    """Metric value resolves and refuses ambiguity."""
    clean, _ = validate_metrics(_obj(), _schema())
    assert clean is not None
    assert metric_value(clean, "mcc") == 0.4
    assert metric_value(clean, "balanced_accuracy@sys_b") == 0.7
    assert metric_value(clean, "per_class.pos.precision") == 0.476
    assert metric_value(clean, "counts.n_flagged") == 40
    with pytest.raises(MetricsError) as info:
        metric_value(clean, "balanced_accuracy")
    assert info.value.code == "E_METRICS_AMBIGUOUS"
    with pytest.raises(MetricsError) as info:
        metric_value(clean, "confusion.fn")
    assert info.value.code == "E_METRICS_SUPPRESSED"
    with pytest.raises(MetricsError) as info:
        metric_value(clean, "per_class.neg.f1")
    assert info.value.code == "E_METRICS_ABSENT"


def _three_class(floor: float) -> tuple[dict[str, Any], MetricsSchema]:
    schema = _schema(headline=["balanced_accuracy"], headline_labels=[], counts=[], per_class=["asd", "td", "other"])
    rates = {"recall": 0.5, "precision": 0.5}
    obj = {
        "headline": [_entry("balanced_accuracy", 0.7, floor)],
        "per_class": {
            "asd": {**rates, "prevalence": 0.3},
            "td": {**rates, "prevalence": 0.7},
            "other": {**rates, "prevalence": 0.0},
        },
        "split_report": {"protocol": "unseen_subject", "invariant_ok": True},
        "threshold_tuning": "validation",
    }
    return obj, schema


def test_absent_class_does_not_lower_the_chance_floor() -> None:
    """With a declared class at prevalence 0, k counts only present classes: 1/3 is refused, 1/2 passes."""
    obj, schema = _three_class(1 / 3)
    assert _codes(obj, schema) == ["E_METRICS_BAD_FLOOR", "W_CLASS_ABSENT"]
    obj, schema = _three_class(0.5)
    clean, codes = validate_metrics(obj, schema)
    assert codes == ["W_CLASS_ABSENT"]
    assert clean is not None


@pytest.mark.parametrize("kind", ["statistic", "count"])
def test_per_class_and_confusion_refused_outside_classification(kind: str) -> None:
    """Statistic and count steps may not report per-class values or a confusion matrix."""
    name = "f_stat" if kind == "statistic" else "count"
    schema = MetricsSchema.model_validate({"kind": kind, "headline": [name]})
    base = {"headline": [{"name": name, "value": 7}]}
    assert _codes(base, schema) == []
    assert _codes({**base, "per_class": {"a": {"recall": 1.0}}}, schema) == ["E_METRICS_KIND_KEY"]
    assert _codes({**base, "confusion": {"tp": 9, "fp": 9, "fn": 9, "tn": 9}}, schema) == ["E_METRICS_KIND_KEY"]


def test_confusion_refused_for_detection() -> None:
    """A confusion matrix belongs to classification only."""
    schema = _schema(kind="detection", headline=["avg_map"], headline_labels=[], counts=[])
    obj = {
        "headline": [_entry("avg_map", 0.06, 0.01)],
        "per_class": {"pos": {"prevalence": 0.3}, "neg": {"prevalence": 0.7}},
        "threshold_tuning": "none",
        "split_report": {"protocol": "unseen_subject", "invariant_ok": True},
    }
    assert _codes(copy.deepcopy(obj), schema) == []
    obj["confusion"] = {"tp": 9}
    assert _codes(obj, schema) == ["E_METRICS_KIND_KEY"]


@pytest.mark.parametrize(
    ("entry", "ok"),
    [
        ({"name": "balanced_accuracy", "value": 60, "floor": 0.5, "lift": 59.5}, False),
        ({"name": "balanced_accuracy", "value": 60, "floor": 50, "lift": 10, "unit": "pp"}, True),
        ({"name": "balanced_accuracy", "value": 101, "floor": 50, "lift": 51, "unit": "pp"}, False),
        ({"name": "balanced_accuracy", "value": -0.1, "floor": 0.5, "lift": -0.6}, False),
        ({"name": "balanced_accuracy", "value": 0.8, "floor": 0.5, "lift": 0.3, "ci_low": 0.85}, False),
        ({"name": "balanced_accuracy", "value": 0.8, "floor": 0.5, "lift": 0.3, "ci_high": 0.75}, False),
        ({"name": "balanced_accuracy", "value": 0.8, "floor": 0.5, "lift": 0.3, "ci_low": 0.7, "ci_high": 0.9}, True),
        ({"name": "mcc", "value": -0.5, "floor": 0.0, "lift": -0.5}, True),
        ({"name": "mcc", "value": -1.5, "floor": 0.0, "lift": -1.5}, False),
        ({"name": "mcc", "value": -50, "floor": 0.0, "lift": -50, "unit": "pp"}, True),
    ],
)
def test_headline_value_ranges(entry: dict[str, Any], ok: bool) -> None:
    """Rates lie in [0, 1] or [0, 100] for pp, mcc in [-1, 1] likewise, and the CI brackets the value."""
    schema = _schema(headline=["balanced_accuracy", "mcc"], headline_labels=[])
    obj = _obj()
    others = [_entry("balanced_accuracy", 0.8, 0.5), _entry("mcc", 0.4, 0.0)]
    obj["headline"] = [entry, *(o for o in others if o["name"] != entry["name"])]
    codes = _codes(obj, schema)
    assert codes == ([] if ok else ["E_METRICS_RANGE"])


def test_f_stat_must_be_non_negative() -> None:
    """An F statistic below 0 is out of range."""
    schema = MetricsSchema.model_validate({"kind": "statistic", "headline": ["f_stat"]})
    assert _codes({"headline": [{"name": "f_stat", "value": -1.0}]}, schema) == ["E_METRICS_RANGE"]
    assert _codes({"headline": [{"name": "f_stat", "value": 0.0}]}, schema) == []


def test_prevalence_below_one_example_is_refused() -> None:
    """With n given, a class prevalence above 0 but below 1/n is not a real proportion."""
    obj, schema = _three_class(1 / 3)
    obj["n"] = 200
    obj["per_class"]["other"]["prevalence"] = 1e-9
    assert _codes(obj, schema) == ["E_METRICS_PREVALENCE_SUBSAMPLE"]
    obj, schema = _three_class(0.5)
    obj["n"] = 200
    assert _codes(obj, schema) == ["W_CLASS_ABSENT"]
    obj["per_class"]["td"]["prevalence"] = 0.695
    obj["per_class"]["other"]["prevalence"] = 0.005
    obj["headline"] = [_entry("balanced_accuracy", 0.7, 1 / 3)]
    assert _codes(obj, schema) == []


@pytest.mark.parametrize(
    ("entry", "ok"),
    [
        ({"name": "macro_f1", "value": 0.5, "floor": -1.0, "lift": 1.5}, False),
        ({"name": "macro_f1", "value": 0.5, "floor": 1.2, "lift": -0.7}, False),
        ({"name": "macro_f1", "value": 0.5, "floor": 0.2, "lift": 0.3}, True),
        ({"name": "macro_f1", "value": 50, "floor": 20, "lift": 30, "unit": "pp"}, True),
        ({"name": "macro_f1", "value": 0.5, "floor": 0.2, "lift": 1.5}, False),
    ],
)
def test_floor_and_lift_ranges(entry: dict[str, Any], ok: bool) -> None:
    """Floors share the value's bounds, presence-only floors included; lift stays within the range width."""
    schema = _schema(headline=["balanced_accuracy", "macro_f1"], headline_labels=[])
    obj = _obj()
    obj["headline"] = [_entry("balanced_accuracy", 0.8, 0.5), entry]
    codes = _codes(obj, schema)
    if ok:
        assert codes == []
    else:
        assert "E_METRICS_RANGE" in codes


@pytest.mark.parametrize("field", ["recall", "precision", "f1"])
def test_per_class_rates_in_unit_interval(field: str) -> None:
    """Per-class recall, precision and f1 lie in [0, 1]."""
    obj, schema = _three_class(0.5)
    obj["per_class"]["asd"][field] = 7
    assert _codes(obj, schema) == ["E_METRICS_RANGE", "W_CLASS_ABSENT"]
