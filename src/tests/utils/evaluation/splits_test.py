"""Tests for protocol-checked grouped splits.

Tolerances were fixed before any splitter existed, from the cohort measurements recorded in the specs
note: a row-level stratified k-fold shares subjects across partitions, plain group k-fold deviates by
more than 0.04 from the overall prevalence, and stratified group k-fold stays within 0.02.
"""

from collections import Counter

import numpy as np
import pytest
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold

from senselab.utils.evaluation.protocol import ProtocolViolation
from senselab.utils.evaluation.splits import (
    Split,
    SplitReport,
    check_split,
    grouped_splits,
    nested_grouped_splits,
)
from tests.utils.evaluation.conftest import Cohort, clip_level_holdout, within_class_subject_split


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
    """Plain group k-fold keeps subjects apart but lets fold prevalence drift by more than 0.04."""
    c = imbalanced_cohort
    folds = GroupKFold(5).split(np.zeros(len(c.labels)), c.labels, c.subjects)
    report = check_split(_sklearn_splits(list(folds)), c.subjects, labels=c.labels, protocol="unseen_subject")
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
    folds = GroupKFold(5).split(np.zeros(len(c.labels)), c.labels, visit)
    with pytest.raises(ProtocolViolation):
        check_split(_sklearn_splits(list(folds)), c.subjects, labels=c.labels, protocol="unseen_subject")


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
        grouped_splits(c.labels, c.subjects, protocol="seen_subject", scheme="seen_subject_kfold", seed=0, units=units)


def test_seen_subject_holdout(imbalanced_cohort: Cohort) -> None:
    """Each subject with 3+ units is in all three partitions; its train share is within one unit of 0.6."""
    c = imbalanced_cohort
    splits, report = grouped_splits(
        c.labels, c.subjects, protocol="seen_subject", scheme="seen_subject_holdout", seed=0, units=c.recordings
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
        check_split(split, c.subjects, labels=c.labels, protocol="seen_subject", units=c.recordings)


def test_s5_mutant(imbalanced_cohort: Cohort) -> None:
    """A within-(label, subject) clip splitter: rejected unseen, rejected per recording, accepted per clip."""
    c = imbalanced_cohort
    split = Split(name="mutant", partitions=within_class_subject_split(c.labels, c.subjects, seed=0))
    with pytest.raises(ProtocolViolation):
        check_split(split, c.subjects, labels=c.labels, protocol="unseen_subject")
    with pytest.raises(ProtocolViolation):
        check_split(split, c.subjects, labels=c.labels, protocol="seen_subject", units=c.recordings)
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
    """Under seen_subject the report's unit is the caller's unit name, default ``recording``."""
    c = imbalanced_cohort
    _, default = grouped_splits(
        c.labels, c.subjects, protocol="seen_subject", scheme="seen_subject_kfold", seed=0, units=c.recordings
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
