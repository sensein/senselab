"""Tests for protocol-checked grouped splits.

Tolerances were fixed before any splitter existed, from the cohort measurements recorded in the specs
note: a row-level stratified k-fold shares subjects across partitions, plain group k-fold deviates by
more than 0.04 from the overall prevalence, and stratified group k-fold stays within 0.02.
"""

from collections import Counter

import numpy as np
import pytest
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from senselab.utils.evaluation.protocol import ProtocolViolation
from senselab.utils.evaluation.splits import (
    Split,
    SplitReport,
    check_split,
    grouped_splits,
    nested_grouped_splits,
)
from tests.utils.evaluation.conftest import (
    Cohort,
    clip_level_holdout,
    size_greedy_group_folds,
    within_class_subject_split,
)


def _sklearn_splits(folds: list) -> list[Split]:
    return [Split(name=f"fold{i}", partitions={"train": tr, "test": te}) for i, (tr, te) in enumerate(folds)]


def _subjects_disjoint(split: Split, subjects: np.ndarray) -> bool:
    seen: dict[str, str] = {}
    for name, rows in split.partitions.items():
        for s in set(subjects[rows].tolist()):
            if seen.setdefault(s, name) != name:
                return False
    return True


def _test_pos_deviation(report: SplitReport) -> float:
    overall = report.overall_prevalence["1"]
    return max(abs(fold["test"]["prevalence"]["1"] - overall) for fold in report.folds)


# ── The three sklearn baselines ───────────────────────────────────────


def test_s14a_row_stratified_kfold_violates_unseen(imbalanced_cohort: Cohort) -> None:
    """A row-level stratified k-fold puts subjects in train and test, which the unseen invariant rejects."""
    c = imbalanced_cohort
    folds = StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(c.labels)), c.labels)
    with pytest.raises(ProtocolViolation, match="subjects appear in more than one partition"):
        check_split(_sklearn_splits(list(folds)), c.subjects, labels=c.labels, protocol="unseen_subject")


def test_s14b_group_kfold_passes_but_deviates(imbalanced_cohort: Cohort) -> None:
    """Plain group k-fold (size-balanced, label-blind) keeps subjects apart but lets prevalence drift past 0.04."""
    c = imbalanced_cohort
    folds = size_greedy_group_folds(c.subjects, 5)
    report = check_split(_sklearn_splits(folds), c.subjects, labels=c.labels, protocol="unseen_subject")
    assert report.invariant_ok
    assert report.max_abs_prevalence_deviation > 0.04


def test_s14c_stratified_group_kfold_matches_sklearn(imbalanced_cohort: Cohort) -> None:
    """The stratified group k-fold keeps deviation within 0.02 and equals a direct sklearn call."""
    c = imbalanced_cohort
    splits, report = grouped_splits(
        c.labels, c.subjects, protocol="unseen_subject", scheme="stratified_group_kfold", seed=0
    )
    assert report.max_abs_prevalence_deviation <= 0.02
    direct = StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(len(c.labels)), c.labels, c.subjects)
    for split, (train, test) in zip(splits, direct):
        assert np.array_equal(split.partitions["train"], np.sort(train))
        assert np.array_equal(split.partitions["test"], np.sort(test))
    assert all(_subjects_disjoint(s, c.subjects) for s in splits)


# ── other unseen-subject schemes ───────────────────────────────────────


def test_multilabel_stratified_group_kfold(imbalanced_cohort: Cohort) -> None:
    """Subjects stay apart, common labels stay within 0.05, and the rare label reaches every test fold."""
    c = imbalanced_cohort
    splits, report = grouped_splits(
        c.multilabel, c.subjects, protocol="unseen_subject", scheme="multilabel_stratified_group_kfold", seed=0
    )
    assert all(_subjects_disjoint(s, c.subjects) for s in splits)
    rates = c.multilabel.mean(axis=0)
    for label, rate in enumerate(rates):
        key = str(label)
        assert report.overall_prevalence[key] == pytest.approx(rate)
        test_rates = [fold["test"]["prevalence"][key] for fold in report.folds]
        if rate >= 0.2:
            assert max(abs(r - rate) for r in test_rates) <= 0.05
    assert all(c.multilabel[s.partitions["test"], 2].sum() > 0 for s in splits)


def test_label_by_timepoint_strata(imbalanced_cohort: Cohort) -> None:
    """Combined label x timepoint strata stay within 0.05 per cell, still grouped by subject."""
    c = imbalanced_cohort
    splits, report = grouped_splits(
        c.labels,
        c.subjects,
        protocol="unseen_subject",
        scheme="stratified_group_kfold",
        seed=0,
        strata=c.timepoints,
        strata_name="timepoint",
    )
    assert report.stratify == "class_timepoint"
    assert all(_subjects_disjoint(s, c.subjects) for s in splits)
    overall = report.overall_stratum_prevalence
    assert overall is not None
    for fold in report.folds:
        cells = fold["test"]["stratum_prevalence"]
        for stratum, by_label in overall.items():
            for label, share in by_label.items():
                assert abs(cells[stratum][label] - share) <= 0.05


def test_leave_one_subject_out(imbalanced_cohort: Cohort) -> None:
    """Leave-one-subject-out gives one fold per subject, each testing exactly that subject."""
    c = imbalanced_cohort
    splits, report = grouped_splits(
        c.labels, c.subjects, protocol="unseen_subject", scheme="leave_one_subject_out", seed=0
    )
    assert len(splits) == 60 and report.n_splits == 60 and report.stratify == "none"
    assert all(len(set(c.subjects[s.partitions["test"]].tolist())) == 1 for s in splits)
    assert all(_subjects_disjoint(s, c.subjects) for s in splits)


def test_group_shuffle(imbalanced_cohort: Cohort) -> None:
    """Grouped shuffle keeps train, validation and test subjects apart."""
    c = imbalanced_cohort
    splits, _ = grouped_splits(
        c.labels,
        c.subjects,
        protocol="unseen_subject",
        scheme="group_shuffle",
        seed=0,
        n_splits=3,
        fractions=(0.6, 0.2, 0.2),
    )
    assert len(splits) == 3
    assert all(set(s.partitions) == {"train", "val", "test"} and _subjects_disjoint(s, c.subjects) for s in splits)


def test_validation_partition_from_the_train_pool(imbalanced_cohort: Cohort) -> None:
    """``val_inner_splits`` carves a subject-disjoint validation partition out of each training pool."""
    c = imbalanced_cohort
    plain, _ = grouped_splits(c.labels, c.subjects, protocol="unseen_subject", scheme="stratified_group_kfold", seed=0)
    splits, _ = grouped_splits(
        c.labels, c.subjects, protocol="unseen_subject", scheme="stratified_group_kfold", seed=0, val_inner_splits=4
    )
    for p, s in zip(plain, splits):
        assert np.array_equal(s.partitions["test"], p.partitions["test"])
        pool = np.concatenate([s.partitions["train"], s.partitions["val"]])
        assert np.array_equal(np.sort(pool), p.partitions["train"])
        assert _subjects_disjoint(s, c.subjects)


def test_subject_by_timepoint_grouping_violates_unseen(imbalanced_cohort: Cohort) -> None:
    """Grouping by (subject, timepoint) tests one visit while training on another: rejected."""
    c = imbalanced_cohort
    visit = np.char.add(np.char.add(c.subjects, "|"), c.timepoints)
    folds = size_greedy_group_folds(visit, 5)
    with pytest.raises(ProtocolViolation):
        check_split(_sklearn_splits(folds), c.subjects, labels=c.labels, protocol="unseen_subject")


def test_nested_inner_splits_stay_inside_outer_train(imbalanced_cohort: Cohort) -> None:
    """Inner splits use outer-train rows only; no outer-test subject reaches an inner partition."""
    c = imbalanced_cohort
    nested, outer_report, inner_reports = nested_grouped_splits(
        c.labels,
        c.subjects,
        protocol="unseen_subject",
        seed=0,
        outer={"scheme": "stratified_group_kfold", "n_splits": 5},
        inner={"scheme": "stratified_group_kfold", "n_splits": 3},
    )
    assert len(nested) == 5 and len(inner_reports) == 5 and outer_report.n_splits == 5
    for outer, inners in nested:
        train = set(outer.partitions["train"].tolist())
        test_subjects = set(c.subjects[outer.partitions["test"]].tolist())
        assert len(inners) == 3
        for inner in inners:
            for rows in inner.partitions.values():
                assert set(rows.tolist()) <= train
                assert not set(c.subjects[rows].tolist()) & test_subjects


# ── seen-subject schemes ────────────────────────────────────────────────


@pytest.mark.parametrize("n_splits", [2, 3, 5])
def test_seen_subject_kfold_invariant(imbalanced_cohort: Cohort, n_splits: int) -> None:
    """Every test subject is in train and no (subject, recording) is in two partitions."""
    c = imbalanced_cohort
    splits, report = grouped_splits(
        c.labels,
        c.subjects,
        protocol="seen_subject",
        scheme="seen_subject_kfold",
        seed=0,
        n_splits=n_splits,
        units=c.recordings,
        unit_name="recording",
    )
    assert len(splits) == n_splits and report.unit == "recording"
    for split in splits:
        train, test = split.partitions["train"], split.partitions["test"]
        assert set(c.subjects[test].tolist()) <= set(c.subjects[train].tolist())
        train_units = set(zip(c.subjects[train].tolist(), c.recordings[train].tolist()))
        test_units = set(zip(c.subjects[test].tolist(), c.recordings[test].tolist()))
        assert not train_units & test_units
    every_row = np.sort(np.concatenate([s.partitions["test"] for s in splits]))
    assert np.array_equal(every_row, np.arange(len(c.labels)))


def test_seen_subject_with_one_unit_raises(imbalanced_cohort: Cohort) -> None:
    """A subject with a single unit cannot be both trained and tested on."""
    c = imbalanced_cohort
    units = c.recordings.copy()
    units[c.subjects == "s00"] = "only"
    with pytest.raises(ProtocolViolation, match="1 subjects have fewer than 2 units"):
        grouped_splits(
            c.labels,
            c.subjects,
            protocol="seen_subject",
            scheme="seen_subject_kfold",
            seed=0,
            units=units,
            unit_name="recording",
        )


def test_seen_subject_holdout(imbalanced_cohort: Cohort) -> None:
    """Each subject with 3+ units is in all three partitions; its train share is within one unit of 0.6."""
    c = imbalanced_cohort
    splits, report = grouped_splits(
        c.labels,
        c.subjects,
        protocol="seen_subject",
        scheme="seen_subject_holdout",
        seed=0,
        units=c.recordings,
        unit_name="recording",
    )
    (split,) = splits
    assert report.unit == "recording" and set(split.partitions) == {"train", "val", "test"}
    for subject in np.unique(c.subjects):
        mine = c.subjects == subject
        n_units = len(set(c.recordings[mine].tolist()))
        held = {
            name: len(set(c.recordings[rows][c.subjects[rows] == subject].tolist()))
            for name, rows in split.partitions.items()
        }
        assert held["train"] >= 1
        assert abs(held["train"] - 0.6 * n_units) <= 1
        if n_units >= 3:
            assert held["val"] >= 1 and held["test"] >= 1


def test_clip_level_seen_subject_holdout(imbalanced_cohort: Cohort) -> None:
    """A clip-level holdout passes as a clip-unit seen-subject split and fails as a recording-unit one."""
    c = imbalanced_cohort
    split = Split(name="holdout", partitions=clip_level_holdout(c.labels, c.subjects, c.timepoints, seed=42))
    report = check_split(split, c.subjects, labels=c.labels, protocol="seen_subject", units=c.clips, unit_name="clip")
    assert report.invariant_ok and report.unit == "clip"
    with pytest.raises(ProtocolViolation, match="units appear in more than one partition"):
        check_split(
            split, c.subjects, labels=c.labels, protocol="seen_subject", units=c.recordings, unit_name="recording"
        )


def test_s5_mutant(imbalanced_cohort: Cohort) -> None:
    """A within-(label, subject) clip splitter: rejected unseen, rejected per recording, accepted per clip."""
    c = imbalanced_cohort
    split = Split(name="mutant", partitions=within_class_subject_split(c.labels, c.subjects, seed=0))
    with pytest.raises(ProtocolViolation):
        check_split(split, c.subjects, labels=c.labels, protocol="unseen_subject")
    with pytest.raises(ProtocolViolation):
        check_split(
            split, c.subjects, labels=c.labels, protocol="seen_subject", units=c.recordings, unit_name="recording"
        )
    report = check_split(split, c.subjects, labels=c.labels, protocol="seen_subject", units=c.clips, unit_name="clip")
    assert report.unit == "clip" and report.invariant_ok


# ── report fields and the reproduce mapping ─────────────────────────────


def test_report_prevalence_matches_the_indices(imbalanced_cohort: Cohort) -> None:
    """Every partition's counts and prevalence recompute from its row indices."""
    c = imbalanced_cohort
    splits, report = grouped_splits(
        c.labels, c.subjects, protocol="unseen_subject", scheme="stratified_group_kfold", seed=3
    )
    assert report.protocol == "unseen_subject" and report.unit == "subject" and report.stratify == "class"
    assert report.invariant and report.invariant_ok and report.seed == 3
    for split, fold in zip(splits, report.folds):
        for name, rows in split.partitions.items():
            stats = fold[name]
            counts = Counter(c.labels[rows].tolist())
            assert stats["n_rows"] == len(rows)
            assert stats["n_groups"] == len(set(c.subjects[rows].tolist()))
            assert stats["prevalence"] == {"0": counts[0] / len(rows), "1": counts[1] / len(rows)}


def test_protocol_is_required(imbalanced_cohort: Cohort) -> None:
    """Splitting without naming a protocol is a TypeError."""
    c = imbalanced_cohort
    with pytest.raises(TypeError):
        grouped_splits(c.labels, c.subjects, scheme="stratified_group_kfold", seed=0)  # type: ignore[call-arg]


def test_seen_subject_unit_is_the_declared_name(imbalanced_cohort: Cohort) -> None:
    """Under seen_subject the report's unit is the caller's unit name, which has no default."""
    c = imbalanced_cohort
    with pytest.raises(ValueError, match="unit_name"):
        grouped_splits(
            c.labels, c.subjects, protocol="seen_subject", scheme="seen_subject_kfold", seed=0, units=c.recordings
        )
    with pytest.raises(ValueError, match="unit_name"):
        check_split(
            Split(name="x", partitions={"train": np.arange(10), "test": np.arange(10, 20)}),
            c.subjects,
            labels=c.labels,
            protocol="seen_subject",
            units=c.recordings,
        )
    _, default = grouped_splits(
        c.labels,
        c.subjects,
        protocol="seen_subject",
        scheme="seen_subject_kfold",
        seed=0,
        units=c.recordings,
        unit_name="recording",
    )
    _, session = grouped_splits(
        c.labels,
        c.subjects,
        protocol="seen_subject",
        scheme="seen_subject_kfold",
        seed=0,
        units=c.recordings,
        unit_name="session",
    )
    assert (default.unit, session.unit) == ("recording", "session")
    assert default.folds[0]["test"]["n_units"] >= default.folds[0]["test"]["n_groups"]


def test_to_reproduce_mapping(imbalanced_cohort: Cohort) -> None:
    """``to_reproduce`` keeps exactly the flat keys, with each fold's test-partition values."""
    c = imbalanced_cohort
    _, report = grouped_splits(c.labels, c.subjects, protocol="unseen_subject", scheme="stratified_group_kfold", seed=0)
    flat = report.to_reproduce()
    assert set(flat) == {"protocol", "unit", "stratify", "folds", "invariant_ok"}
    assert (flat["protocol"], flat["unit"], flat["stratify"], flat["invariant_ok"]) == (
        "unseen_subject",
        "subject",
        "class",
        True,
    )
    assert len(flat["folds"]) == 5
    for fold, full in zip(flat["folds"], report.folds):
        assert set(fold) == {"n_groups", "prevalence"}
        assert fold == {"n_groups": full["test"]["n_groups"], "prevalence": full["test"]["prevalence"]}


def test_to_reproduce_refuses_tokens_outside_its_vocabulary(imbalanced_cohort: Cohort) -> None:
    """A unit or strata name the reproduce schema does not know is refused rather than emitted."""
    c = imbalanced_cohort
    _, report = grouped_splits(
        c.labels,
        c.subjects,
        protocol="seen_subject",
        scheme="seen_subject_kfold",
        seed=0,
        units=c.recordings,
        unit_name="segment",
    )
    with pytest.raises(ValueError, match="segment"):
        report.to_reproduce()


# ── Input validation ──────────────────────────────────────────────────


def _toy() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subjects a and b, two units each, one row per unit."""
    return np.array([0, 1, 0, 1]), np.array(["a", "a", "b", "b"]), np.array(["u1", "u2", "u1", "u2"])


def test_seen_test_subject_absent_from_train_raises() -> None:
    """Subject a has both its units in test and none in train: rejected."""
    labels, subjects, units = _toy()
    split = Split(name="x", partitions={"train": np.array([2, 3]), "test": np.array([0, 1])})
    with pytest.raises(ProtocolViolation, match="1 test subjects are absent from train"):
        check_split(split, subjects, labels=labels, protocol="seen_subject", units=units, unit_name="recording")


@pytest.mark.parametrize(
    ("partitions", "message"),
    [
        ({"train": np.array([]), "test": np.array([0, 1])}, "empty"),
        ({"train": np.array([2, 3]), "test": np.array([], dtype=int)}, "empty"),
        ({"train": np.array([2, 2]), "test": np.array([0, 1])}, "duplicate"),
        ({"train": np.array([2, 4]), "test": np.array([0, 1])}, "out of range"),
        ({"train": np.array([2, -1]), "test": np.array([0, 1])}, "out of range"),
        ({"train": np.array([False, False, True, True]), "test": np.array([0, 1])}, "boolean"),
        ({"train": np.array([2.0, 3.0]), "test": np.array([0, 1])}, "integer"),
    ],
    ids=["empty-float-train", "empty-test", "duplicate", "too-large", "negative", "mask", "float"],
)
def test_check_split_refuses_malformed_partitions(partitions: dict, message: str) -> None:
    """Malformed index arrays are a ValueError, never an IndexError or a silent pass."""
    labels, subjects, _ = _toy()
    with pytest.raises(ValueError, match=message):
        check_split(Split(name="x", partitions=partitions), subjects, labels=labels, protocol="unseen_subject")


@pytest.mark.parametrize("missing", [None, float("nan"), ""])
def test_missing_subject_ids_are_refused(missing: object) -> None:
    """Every row needs a subject; None, NaN and empty ids are refused by both entry points."""
    labels = np.array([0, 1, 0, 1, 0, 1])
    subjects = np.array(["a", "a", "b", missing, "c", "c"], dtype=object)
    with pytest.raises(ValueError, match="subject"):
        grouped_splits(labels, subjects, protocol="unseen_subject", scheme="stratified_group_kfold", seed=0, n_splits=2)
    split = Split(name="x", partitions={"train": np.array([0, 1]), "test": np.array([4, 5])})
    with pytest.raises(ValueError, match="subject"):
        check_split(split, subjects, labels=labels, protocol="unseen_subject")


def test_lengths_must_match() -> None:
    """Labels, units and strata need one entry per subject entry."""
    labels, subjects, units = _toy()
    split = Split(name="x", partitions={"train": np.array([2, 3]), "test": np.array([0, 1])})
    with pytest.raises(ValueError, match="one entry per row"):
        check_split(split, subjects, labels=labels[:3], protocol="unseen_subject")
    with pytest.raises(ValueError, match="one entry per row"):
        check_split(split, subjects, labels=labels, protocol="seen_subject", units=units[:3], unit_name="recording")
    with pytest.raises(ValueError, match="one entry per row"):
        check_split(split, subjects, labels=labels, protocol="unseen_subject", strata=units[:3])
    with pytest.raises(ValueError, match="one entry per row"):
        grouped_splits(labels[:3], subjects, protocol="unseen_subject", scheme="leave_one_subject_out", seed=0)


@pytest.mark.parametrize("fractions", [(1.2, -0.1, -0.1), (1.0, 0.0, 0.0), (0.5, 0.2, 0.2), (0.6, 0.4)])
def test_bad_fractions_are_refused(imbalanced_cohort: Cohort, fractions: tuple) -> None:
    """Fractions are three non-negative numbers below 1 that sum to 1."""
    c = imbalanced_cohort
    with pytest.raises(ValueError, match="fractions"):
        grouped_splits(
            c.labels, c.subjects, protocol="unseen_subject", scheme="group_shuffle", seed=0, fractions=fractions
        )


@pytest.mark.parametrize(
    ("protocol", "scheme", "extra", "message"),
    [
        ("unseen_subject", "stratified_group_kfold", {"fractions": (0.6, 0.2, 0.2)}, "fractions"),
        ("unseen_subject", "leave_one_subject_out", {"n_splits": 5}, "n_splits"),
        ("unseen_subject", "leave_one_subject_out", {"strata": "timepoints", "strata_name": "timepoint"}, "strata"),
        ("unseen_subject", "group_shuffle", {"fractions": (0.6, 0.2, 0.2), "val_inner_splits": 4}, "val_inner"),
        ("unseen_subject", "stratified_group_kfold", {"units": "recordings"}, "units"),
        ("unseen_subject", "stratified_group_kfold", {"unit_name": "recording"}, "unit_name"),
        ("seen_subject", "seen_subject_kfold", {"fractions": (0.6, 0.2, 0.2)}, "fractions"),
        ("seen_subject", "seen_subject_holdout", {"val_inner_splits": 4}, "val_inner"),
        ("seen_subject", "seen_subject_holdout", {"n_splits": 3}, "n_splits"),
    ],
)
def test_arguments_that_do_not_apply_are_refused(
    imbalanced_cohort: Cohort, protocol: str, scheme: str, extra: dict, message: str
) -> None:
    """An argument the scheme or protocol would ignore is refused instead."""
    c = imbalanced_cohort
    kwargs = {k: getattr(c, v) if isinstance(v, str) and k in ("units", "strata") else v for k, v in extra.items()}
    if protocol == "seen_subject":
        kwargs = {"units": c.recordings, "unit_name": "recording", **kwargs}
    with pytest.raises(ValueError, match=message):
        grouped_splits(c.labels, c.subjects, protocol=protocol, scheme=scheme, seed=0, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("extra", [{"n_splits": 1}, {"val_inner_splits": 1}])
def test_fewer_than_two_folds_is_a_value_error(imbalanced_cohort: Cohort, extra: dict) -> None:
    """One fold, or one inner fold, is a ValueError rather than a protocol violation."""
    c = imbalanced_cohort
    with pytest.raises(ValueError, match="at least 2") as caught:
        grouped_splits(
            c.labels, c.subjects, protocol="unseen_subject", scheme="stratified_group_kfold", seed=0, **extra
        )
    assert not isinstance(caught.value, ProtocolViolation)


def test_nested_seen_thin_inner_pool_names_the_inner_pool(imbalanced_cohort: Cohort) -> None:
    """A subject left with one unit in an outer training pool fails, and the error says it is the inner pool."""
    c = imbalanced_cohort
    with pytest.raises(ProtocolViolation, match="inner pool"):
        nested_grouped_splits(
            c.labels,
            c.subjects,
            protocol="seen_subject",
            seed=0,
            outer={"scheme": "seen_subject_kfold", "n_splits": 5},
            inner={"scheme": "seen_subject_kfold", "n_splits": 2},
            units=c.recordings,
            unit_name="recording",
        )


def test_multilabel_with_strata_is_refused(imbalanced_cohort: Cohort) -> None:
    """Per-stratum prevalence is defined for one label per row only."""
    c = imbalanced_cohort
    split = Split(name="x", partitions={"train": np.arange(100), "test": np.arange(100, 200)})
    with pytest.raises(ValueError, match="strata"):
        check_split(
            split, c.subjects, labels=c.multilabel, protocol="unseen_subject", strata=c.timepoints, stratify="class"
        )


# ── Review fixes ──────────────────────────────────────────────────────


def _two_unit_cohort() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ten subjects, each with exactly two units of three rows; labels alternate by row."""
    subjects = np.repeat([f"s{i}" for i in range(10)], 6)
    units = np.tile(np.repeat(["u1", "u2"], 3), 10)
    labels = np.arange(60) % 2
    return labels, subjects, units


def test_holdout_omits_a_zero_fraction_partition(imbalanced_cohort: Cohort) -> None:
    """A zero validation fraction gives train and test only, as grouped shuffle does."""
    c = imbalanced_cohort
    (split,), _ = grouped_splits(
        c.labels,
        c.subjects,
        protocol="seen_subject",
        scheme="seen_subject_holdout",
        seed=0,
        units=c.recordings,
        unit_name="recording",
        fractions=(0.8, 0.0, 0.2),
    )
    assert set(split.partitions) == {"train", "test"}


def test_holdout_on_two_unit_subjects_names_the_cause() -> None:
    """Three partitions cannot all be filled from two units per subject; the error says so."""
    labels, subjects, units = _two_unit_cohort()
    with pytest.raises(ValueError, match="every subject has 2 units or fewer; seen_subject_holdout needs >= 3"):
        grouped_splits(
            labels, subjects, protocol="seen_subject", scheme="seen_subject_holdout", seed=0, units=units, unit_name="r"
        )


def test_seen_kfold_validation_on_two_unit_subjects_names_the_cause() -> None:
    """With two units per subject nothing is left for validation after the test unit; the error says so."""
    labels, subjects, units = _two_unit_cohort()
    with pytest.raises(ValueError, match="every subject has 2 units or fewer; seen_subject_kfold needs >= 4"):
        grouped_splits(
            labels,
            subjects,
            protocol="seen_subject",
            scheme="seen_subject_kfold",
            seed=0,
            n_splits=2,
            units=units,
            unit_name="r",
            val_inner_splits=2,
        )


def test_pandas_na_is_a_missing_id() -> None:
    """``pandas.NA`` counts as a missing subject or unit id."""
    pd = pytest.importorskip("pandas")
    labels, subjects, units = _two_unit_cohort()
    holed = subjects.astype(object)
    holed[0] = pd.NA
    with pytest.raises(ValueError, match="no subject id"):
        grouped_splits(labels, holed, protocol="unseen_subject", scheme="stratified_group_kfold", seed=0, n_splits=2)
    holed_units = units.astype(object)
    holed_units[0] = pd.NA
    with pytest.raises(ValueError, match="no unit id"):
        grouped_splits(
            labels,
            subjects,
            protocol="seen_subject",
            scheme="seen_subject_kfold",
            seed=0,
            units=holed_units,
            unit_name="r",
        )


# ── Adjacent fixes ────────────────────────────────────────────────────


def test_seen_kfold_validation_needs_follow_val_inner_splits() -> None:
    """The message's unit requirement follows val_inner_splits."""
    labels, subjects, units = _two_unit_cohort()
    with pytest.raises(ValueError, match="needs >= 8"):
        grouped_splits(
            labels,
            subjects,
            protocol="seen_subject",
            scheme="seen_subject_kfold",
            seed=0,
            n_splits=2,
            units=units,
            unit_name="r",
            val_inner_splits=4,
        )


def test_missing_id_message_names_pandas_na() -> None:
    """The missing-id error lists every value it treats as missing."""
    labels, subjects, _ = _two_unit_cohort()
    holed = subjects.astype(object)
    holed[0] = None
    with pytest.raises(ValueError, match="pandas.NA"):
        grouped_splits(labels, holed, protocol="unseen_subject", scheme="stratified_group_kfold", seed=0, n_splits=2)


@pytest.mark.parametrize("missing", [float("nan"), np.float32("nan")], ids=["float-nan", "float32-nan"])
def test_nan_in_a_plain_list_is_refused_before_coercion(missing: object) -> None:
    """A NaN in a plain list must not become the string "nan" and pass as a subject."""
    labels, subjects, _ = _two_unit_cohort()
    as_list = list(subjects)
    as_list[0] = missing
    with pytest.raises(ValueError, match="no subject id"):
        grouped_splits(labels, as_list, protocol="unseen_subject", scheme="stratified_group_kfold", seed=0, n_splits=2)
    split = Split(name="x", partitions={"train": np.arange(30), "test": np.arange(30, 60)})
    with pytest.raises(ValueError, match="no subject id"):
        check_split(split, as_list, labels=labels, protocol="unseen_subject")


def test_pandas_nat_is_a_missing_id() -> None:
    """``pandas.NaT`` counts as missing too."""
    pd = pytest.importorskip("pandas")
    labels, subjects, _ = _two_unit_cohort()
    holed = subjects.astype(object)
    holed[0] = pd.NaT
    with pytest.raises(ValueError, match="no subject id"):
        grouped_splits(labels, holed, protocol="unseen_subject", scheme="stratified_group_kfold", seed=0, n_splits=2)


def _one_subject(n_units: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One subject with ``n_units`` single-row units; labels alternate."""
    return np.arange(n_units) % 2, np.array(["s"] * n_units), np.array([f"u{i}" for i in range(n_units)])


@pytest.mark.parametrize(("m", "n", "v", "needed"), [(7, 2, 4, 8), (5, 2, 4, 8), (3, 2, 2, 4)])
def test_seen_kfold_validation_bound(m: int, n: int, v: int, needed: int) -> None:
    """The bound is max(3, ceil(V*n/(n-1))): the fold that takes ceil(m/n) units must leave V for the inner slots."""
    labels, subjects, units = _one_subject(m)
    with pytest.raises(ValueError, match=f"needs >= {needed} units"):
        grouped_splits(
            labels,
            subjects,
            protocol="seen_subject",
            scheme="seen_subject_kfold",
            seed=0,
            n_splits=n,
            units=units,
            unit_name="r",
            val_inner_splits=v,
        )


def test_seen_kfold_validation_bound_control() -> None:
    """At the bound (m=8, n=2, V=4) every fold gets a validation partition."""
    labels, subjects, units = _one_subject(8)
    splits, _ = grouped_splits(
        labels,
        subjects,
        protocol="seen_subject",
        scheme="seen_subject_kfold",
        seed=0,
        n_splits=2,
        units=units,
        unit_name="r",
        val_inner_splits=4,
    )
    assert all(len(s.partitions["val"]) >= 1 for s in splits)
