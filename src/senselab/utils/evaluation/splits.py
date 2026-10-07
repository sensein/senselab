"""Grouped, protocol-checked splits and their reports.

`grouped_splits` is the one entry point for making splits; `check_split` checks a split made
elsewhere; `nested_grouped_splits` draws inner splits from each outer training partition only.
Every call returns a `SplitReport`, which is built only after the protocol invariant has been
checked on every split, so a report that exists is a report that passed.

Unseen-subject schemes (no subject in two partitions):

- `stratified_group_kfold`: scikit-learn's `StratifiedGroupKFold` on the labels, or on label x
  stratum when `strata` is given.
- `multilabel_stratified_group_kfold`: for an `(n, L)` 0/1 label matrix, subjects are placed largest
  first into the fold that keeps every label's share of positives closest to the fold's share of rows.
- `leave_one_subject_out`: scikit-learn's `LeaveOneGroupOut`.
- `group_shuffle`: scikit-learn's `GroupShuffleSplit`, sized by `fractions`.

Seen-subject schemes (every test subject in train, no `(subject, unit)` in two partitions, every
subject with at least two units):

- `seen_subject_kfold`: each subject's units are dealt round-robin over the folds.
- `seen_subject_holdout`: each subject's units are allocated to train, validation and test in
  proportion to `fractions`.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union, get_args

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut, StratifiedGroupKFold

from senselab.utils.evaluation.protocol import PROTOCOLS, Protocol, ProtocolViolation, check_protocol

Scheme = Literal[
    "stratified_group_kfold",
    "multilabel_stratified_group_kfold",
    "leave_one_subject_out",
    "group_shuffle",
    "seen_subject_kfold",
    "seen_subject_holdout",
]
UNSEEN_SCHEMES = (
    "stratified_group_kfold",
    "multilabel_stratified_group_kfold",
    "leave_one_subject_out",
    "group_shuffle",
)
SEEN_SCHEMES = ("seen_subject_kfold", "seen_subject_holdout")

# The tokens a reproduction driver's flat `split_report` accepts for `unit` and `stratify`. These mirror
# the reproduce metrics enums (SPLIT_UNITS, SPLIT_STRATIFY); a cross-check test follows once both meet.
REPRODUCE_UNITS = ("subject", "family", "session", "video", "clip", "recording", "row")
REPRODUCE_STRATIFY = ("none", "class", "class_timepoint", "class_site", "class_age", "timepoint")

DEFAULT_HOLDOUT_FRACTIONS = (0.6, 0.2, 0.2)
# Offset between a fold's seed and the seed of the inner split carved from its training pool.
INNER_SEED_STRIDE = 10_000

PartitionStats = Dict[str, Any]


@dataclass(frozen=True)
class Split:
    """One split: partition name (``train``, ``test``, optionally ``val``) to sorted row indices."""

    name: str
    partitions: Dict[str, np.ndarray]


@dataclass(frozen=True)
class SplitReport:
    """What a set of splits did, checked against its protocol.

    Build it with `SplitReport.from_splits` (which `grouped_splits` and `check_split` call); that
    runs the protocol check on every split before computing anything, so `invariant_ok` is always
    `True`.

    Attributes:
        protocol: `seen_subject` or `unseen_subject`.
        unit: `subject` under `unseen_subject`; the caller-declared unit name under `seen_subject`.
        stratify: What the splits were stratified on: `none`, `class` or `class_<strata name>`.
        scheme: The scheme name, or `external` for a split checked by `check_split`.
        n_splits: Number of splits.
        seed: The seed, or `None` for an external split.
        folds: Per split, partition name to `{n_rows, n_groups, n_units, prevalence,
            stratum_prevalence?}`; prevalence maps each label to its share of the partition's rows
            (for a multi-label matrix, each column's positive rate).
        overall_prevalence: The same over all rows.
        overall_stratum_prevalence: Stratum to label to joint share over all rows, when strata are given.
        max_abs_prevalence_deviation: Largest |partition prevalence - overall prevalence| over all
            splits, partitions and labels.
        invariant: The invariant that was checked, as text.
        invariant_ok: Always `True`.
    """

    protocol: Protocol
    unit: str
    stratify: str
    scheme: str
    n_splits: int
    seed: Optional[int]
    folds: List[Dict[str, PartitionStats]]
    overall_prevalence: Dict[str, float]
    max_abs_prevalence_deviation: float
    invariant: str
    invariant_ok: bool
    overall_stratum_prevalence: Optional[Dict[str, Dict[str, float]]] = field(default=None)

    @classmethod
    def from_splits(
        cls,
        splits: Sequence[Split],
        *,
        labels: np.ndarray,
        subjects: np.ndarray,
        protocol: Protocol,
        units: Optional[np.ndarray],
        unit_name: str,
        stratify: str,
        scheme: str,
        seed: Optional[int],
        strata: Optional[np.ndarray] = None,
    ) -> "SplitReport":
        """Check every split against the protocol, then summarise them.

        Args:
            splits: The splits.
            labels: Label per row, or an `(n, L)` 0/1 matrix.
            subjects: Subject key per row.
            protocol: The protocol to check.
            units: Unit key per row; required under `seen_subject`.
            unit_name: The unit's name, reported under `seen_subject`.
            stratify: What the splits were stratified on.
            scheme: The scheme name.
            seed: The seed used, if any.
            strata: Optional stratum per row, for per-stratum prevalence.

        Returns:
            The report.

        Raises:
            ProtocolViolation: If any split breaks the protocol's invariant.
        """
        invariant = ""
        for split in splits:
            invariant = check_protocol(split.partitions, subjects, protocol=protocol, units=units)
        all_rows = np.arange(len(subjects))
        keys = _label_keys(labels)
        overall = _prevalence(labels, all_rows, keys)
        overall_strata = _stratum_prevalence(labels, strata, all_rows, keys) if strata is not None else None
        folds = []
        deviation = 0.0
        for split in splits:
            fold: Dict[str, PartitionStats] = {}
            for name, rows in split.partitions.items():
                stats = _partition_stats(labels, subjects, units, strata, rows, keys)
                deviation = max(deviation, *(abs(stats["prevalence"][k] - overall[k]) for k in keys))
                fold[name] = stats
            folds.append(fold)
        return cls(
            protocol=protocol,
            unit="subject" if protocol == "unseen_subject" else unit_name,
            stratify=stratify,
            scheme=scheme,
            n_splits=len(splits),
            seed=seed,
            folds=folds,
            overall_prevalence=overall,
            max_abs_prevalence_deviation=deviation,
            invariant=invariant,
            invariant_ok=True,
            overall_stratum_prevalence=overall_strata,
        )

    def to_reproduce(self) -> Dict[str, Any]:
        """The flat `split_report` a reproduction driver records: each fold's test-partition values.

        Returns:
            ``{protocol, unit, stratify, folds: [{n_groups, prevalence}], invariant_ok}``.

        Raises:
            ValueError: If `unit` or `stratify` is not a token the driver's schema accepts.
        """
        if self.unit not in REPRODUCE_UNITS:
            raise ValueError(f"unit {self.unit!r} is not one of {REPRODUCE_UNITS}")
        if self.stratify not in REPRODUCE_STRATIFY:
            raise ValueError(f"stratify {self.stratify!r} is not one of {REPRODUCE_STRATIFY}")
        return {
            "protocol": self.protocol,
            "unit": self.unit,
            "stratify": self.stratify,
            "folds": [
                {"n_groups": fold["test"]["n_groups"], "prevalence": dict(fold["test"]["prevalence"])}
                for fold in self.folds
            ],
            "invariant_ok": self.invariant_ok,
        }


# ── statistics ──────────────────────────────────────────────────────────


def _label_keys(labels: np.ndarray) -> List[str]:
    if labels.ndim == 2:
        return [str(j) for j in range(labels.shape[1])]
    return [str(v) for v in sorted(set(labels.tolist()), key=str)]


def _prevalence(labels: np.ndarray, rows: np.ndarray, keys: List[str]) -> Dict[str, float]:
    if len(rows) == 0:
        return {k: 0.0 for k in keys}
    if labels.ndim == 2:
        rates = labels[rows].mean(axis=0)
        return {k: float(rates[j]) for j, k in enumerate(keys)}
    counts = Counter(str(v) for v in labels[rows].tolist())
    return {k: counts[k] / len(rows) for k in keys}


def _stratum_prevalence(
    labels: np.ndarray, strata: np.ndarray, rows: np.ndarray, keys: List[str]
) -> Dict[str, Dict[str, float]]:
    """Stratum to label to the joint share of `rows` with that stratum and label."""
    names = [str(s) for s in sorted(set(strata.tolist()), key=str)]
    counts = Counter((str(s), str(y)) for s, y in zip(strata[rows].tolist(), labels[rows].tolist()))
    n = max(len(rows), 1)
    return {s: {k: counts[(s, k)] / n for k in keys} for s in names}


def _partition_stats(
    labels: np.ndarray,
    subjects: np.ndarray,
    units: Optional[np.ndarray],
    strata: Optional[np.ndarray],
    rows: np.ndarray,
    keys: List[str],
) -> PartitionStats:
    n_groups = len(set(subjects[rows].tolist()))
    n_units = len(set(zip(subjects[rows].tolist(), units[rows].tolist()))) if units is not None else n_groups
    stats: PartitionStats = {
        "n_rows": int(len(rows)),
        "n_groups": n_groups,
        "n_units": n_units,
        "prevalence": _prevalence(labels, rows, keys),
    }
    if strata is not None:
        stats["stratum_prevalence"] = _stratum_prevalence(labels, strata, rows, keys)
    return stats


# ── unseen-subject schemes ──────────────────────────────────────────────


def _sorted(rows: np.ndarray) -> np.ndarray:
    return np.sort(np.asarray(rows, dtype=int))


def _sgkf(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(_sorted(tr), _sorted(te)) for tr, te in splitter.split(np.zeros(len(y)), y, groups)]


def _multilabel_group_folds(
    y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Greedy group assignment that keeps every label's share of each fold in line with its row share.

    Groups are placed largest first (ties in seeded random order). Each goes to the fold that, after
    adding it, minimises the summed squared difference between each fold's share of every label's
    positives and its share of rows, plus the squared difference between its row share and `1/k`;
    remaining ties are broken at random with the seed.
    """
    rng = np.random.default_rng(seed)
    names, inverse = np.unique(groups, return_inverse=True)
    counts = np.zeros((len(names), y.shape[1]))
    np.add.at(counts, inverse, y)
    sizes = np.bincount(inverse, minlength=len(names)).astype(float)
    totals = np.where(counts.sum(axis=0) > 0, counts.sum(axis=0), 1.0)
    n_rows = sizes.sum()
    assigned = np.zeros((n_splits, y.shape[1]))
    rows = np.zeros(n_splits)
    fold_of = np.full(len(names), -1)
    order = rng.permutation(len(names))
    order = order[np.argsort(-sizes[order], kind="stable")]
    for g in order:
        costs = np.empty(n_splits)
        for f in range(n_splits):
            trial, trial_rows = assigned.copy(), rows.copy()
            trial[f] += counts[g]
            trial_rows[f] += sizes[g]
            label_share, row_share = trial / totals, trial_rows / n_rows
            costs[f] = ((label_share - row_share[:, None]) ** 2).sum() + ((row_share - 1 / n_splits) ** 2).sum()
        f = int(rng.choice(np.flatnonzero(np.isclose(costs, costs.min()))))
        fold_of[g] = f
        assigned[f] += counts[g]
        rows[f] += sizes[g]
    rows_fold = fold_of[inverse]
    all_rows = np.arange(len(groups))
    return [(all_rows[rows_fold != f], all_rows[rows_fold == f]) for f in range(n_splits)]


def _kfold_pairs(
    scheme: str, labels: np.ndarray, subjects: np.ndarray, strata: Optional[np.ndarray], n_splits: int, seed: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if scheme == "multilabel_stratified_group_kfold":
        return _multilabel_group_folds(labels, subjects, n_splits, seed)
    y = labels if strata is None else np.char.add(np.char.add(labels.astype(str), "\x1f"), strata.astype(str))
    return _sgkf(y, subjects, n_splits, seed)


def _unseen_splits(
    scheme: str,
    labels: np.ndarray,
    subjects: np.ndarray,
    *,
    seed: int,
    n_splits: int,
    strata: Optional[np.ndarray],
    val_inner_splits: Optional[int],
    fractions: Optional[Tuple[float, float, float]],
) -> List[Split]:
    if scheme in ("stratified_group_kfold", "multilabel_stratified_group_kfold"):
        splits = []
        for i, (train, test) in enumerate(_kfold_pairs(scheme, labels, subjects, strata, n_splits, seed)):
            parts = {"train": train, "test": test}
            if val_inner_splits is not None:
                inner_strata = None if strata is None else strata[train]
                inner = _kfold_pairs(
                    scheme,
                    labels[train],
                    subjects[train],
                    inner_strata,
                    val_inner_splits,
                    seed + INNER_SEED_STRIDE * (i + 1),
                )
                parts = {"train": _sorted(train[inner[0][0]]), "val": _sorted(train[inner[0][1]]), "test": test}
            splits.append(Split(name=f"fold{i}", partitions=parts))
        return splits
    if val_inner_splits is not None:
        raise ValueError(f"val_inner_splits is not available for {scheme}")
    if scheme == "leave_one_subject_out":
        pairs = LeaveOneGroupOut().split(np.zeros(len(subjects)), labels, subjects)
        return [
            Split(name=f"fold{i}", partitions={"train": _sorted(tr), "test": _sorted(te)})
            for i, (tr, te) in enumerate(pairs)
        ]
    if fractions is None:
        raise ValueError("group_shuffle needs fractions=(train, val, test)")
    f_train, f_val, f_test = fractions
    outer = GroupShuffleSplit(n_splits=n_splits, test_size=f_test, random_state=seed)
    splits = []
    for i, (pool, test) in enumerate(outer.split(np.zeros(len(subjects)), labels, subjects)):
        parts = {"train": _sorted(pool), "test": _sorted(test)}
        if f_val > 0:
            inner = GroupShuffleSplit(
                n_splits=1, test_size=f_val / (f_train + f_val), random_state=seed + INNER_SEED_STRIDE * (i + 1)
            )
            tr, va = next(inner.split(np.zeros(len(pool)), labels[pool], subjects[pool]))
            parts = {"train": _sorted(pool[tr]), "val": _sorted(pool[va]), "test": _sorted(test)}
        splits.append(Split(name=f"split{i}", partitions=parts))
    return splits


# ── seen-subject schemes ────────────────────────────────────────────────


def _majority(values: List[Any]) -> str:
    counts = Counter(str(v) for v in values)
    top = max(counts.values())
    return min(k for k, c in counts.items() if c == top)


def _ordered_units(
    labels: np.ndarray,
    subjects: np.ndarray,
    units: np.ndarray,
    strata: Optional[np.ndarray],
    rng: np.random.Generator,
) -> Dict[str, List[np.ndarray]]:
    """Per subject, its units' row arrays ordered by (majority label, majority stratum), shuffled within ties."""
    rows_by_unit: Dict[Tuple[str, str], List[int]] = {}
    for i, (s, u) in enumerate(zip(subjects.tolist(), units.tolist())):
        rows_by_unit.setdefault((str(s), str(u)), []).append(i)
    by_subject: Dict[str, List[Tuple[Tuple[str, str], np.ndarray]]] = {}
    for (s, u), rows in sorted(rows_by_unit.items()):
        idx = np.array(rows)
        key = (_majority(labels[idx].tolist()), _majority(strata[idx].tolist()) if strata is not None else "")
        by_subject.setdefault(s, []).append((key, idx))
    ordered: Dict[str, List[np.ndarray]] = {}
    for s, entries in by_subject.items():
        result: List[np.ndarray] = []
        for key in sorted({k for k, _ in entries}):
            same = [idx for k, idx in entries if k == key]
            result += [same[j] for j in rng.permutation(len(same))]
        ordered[s] = result
    return ordered


def _seen_kfold(
    labels: np.ndarray,
    subjects: np.ndarray,
    units: np.ndarray,
    *,
    seed: int,
    n_splits: int,
    strata: Optional[np.ndarray],
    val_inner_splits: Optional[int],
) -> List[Split]:
    rng = np.random.default_rng(seed)
    ordered = _ordered_units(labels, subjects, units, strata, rng)
    dealt: Dict[str, List[int]] = {}
    for s, unit_rows in ordered.items():
        offset = int(rng.integers(n_splits))
        dealt[s] = [(offset + j) % n_splits for j in range(len(unit_rows))]
    splits = []
    for f in range(n_splits):
        parts: Dict[str, List[np.ndarray]] = {"train": [], "test": []}
        if val_inner_splits is not None:
            parts["val"] = []
        for s, unit_rows in ordered.items():
            rest = [rows for rows, fold in zip(unit_rows, dealt[s]) if fold != f]
            parts["test"] += [rows for rows, fold in zip(unit_rows, dealt[s]) if fold == f]
            if val_inner_splits is not None and len(rest) >= 2:
                offset = int(rng.integers(val_inner_splits))
                slots = [(offset + j) % val_inner_splits for j in range(len(rest))]
                parts["val"] += [rows for rows, slot in zip(rest, slots) if slot == 0]
                rest = [rows for rows, slot in zip(rest, slots) if slot != 0]
            parts["train"] += rest
        splits.append(Split(name=f"fold{f}", partitions={k: _concat(v) for k, v in parts.items()}))
    return splits


def _concat(arrays: List[np.ndarray]) -> np.ndarray:
    return _sorted(np.concatenate(arrays)) if arrays else np.array([], dtype=int)


def _allocate(n: int, fractions: Tuple[float, ...]) -> List[int]:
    """Largest-remainder counts; every positive fraction gets one when `n` allows, and train keeps one."""
    raw = [n * f for f in fractions]
    counts = [math.floor(r) for r in raw]
    for k in sorted(range(len(raw)), key=lambda k: (-(raw[k] - counts[k]), k))[: n - sum(counts)]:
        counts[k] += 1
    positive = [k for k, f in enumerate(fractions) if f > 0]
    if n >= len(positive):
        for k in positive:
            if counts[k] == 0:
                donor = max(range(len(counts)), key=lambda j: (counts[j], -j))
                counts[donor] -= 1
                counts[k] += 1
    if counts[0] == 0 and n >= 1:
        donor = max(range(1, len(counts)), key=lambda j: (counts[j], -j))
        counts[donor] -= 1
        counts[0] += 1
    return counts


def _interleave(counts: List[int]) -> List[int]:
    """Partition per position, spreading each partition evenly over the ordered units."""
    n = sum(counts)
    assigned = [0] * len(counts)
    sequence = []
    for i in range(n):
        open_parts = [k for k in range(len(counts)) if assigned[k] < counts[k]]
        k = max(open_parts, key=lambda p: (counts[p] * (i + 1) / n - assigned[p], -p))
        assigned[k] += 1
        sequence.append(k)
    return sequence


def _seen_holdout(
    labels: np.ndarray,
    subjects: np.ndarray,
    units: np.ndarray,
    *,
    seed: int,
    strata: Optional[np.ndarray],
    fractions: Tuple[float, float, float],
) -> List[Split]:
    rng = np.random.default_rng(seed)
    names = ("train", "val", "test")
    parts: Dict[str, List[np.ndarray]] = {name: [] for name in names}
    for unit_rows in _ordered_units(labels, subjects, units, strata, rng).values():
        for rows, k in zip(unit_rows, _interleave(_allocate(len(unit_rows), fractions))):
            parts[names[k]].append(rows)
    return [Split(name="holdout", partitions={k: _concat(v) for k, v in parts.items()})]


# ── public API ──────────────────────────────────────────────────────────

KFOLD_SCHEMES = ("stratified_group_kfold", "multilabel_stratified_group_kfold", "seen_subject_kfold")
DEFAULT_N_SPLITS = 5


def _count_missing(values: np.ndarray) -> int:
    """Entries that are None, NaN or the empty string."""
    return sum(
        v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and v == "")
        for v in values.tolist()
    )


def _prepare(
    labels: np.ndarray,
    subjects: np.ndarray,
    protocol: str,
    units: Optional[np.ndarray],
    unit_name: Optional[str],
    strata: Optional[np.ndarray],
) -> None:
    """Checks shared by every entry point; raises ValueError naming what is wrong."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}")
    n = len(subjects)
    if len(labels) != n or (units is not None and len(units) != n) or (strata is not None and len(strata) != n):
        raise ValueError("labels, subjects, units and strata must have one entry per row")
    missing = _count_missing(subjects)
    if missing:
        raise ValueError(f"{missing} rows have no subject id (None, NaN or empty); every row needs one")
    if labels.ndim == 2 and strata is not None:
        raise ValueError("strata need one label per row; they are not supported with a multi-label matrix")
    if protocol == "unseen_subject":
        if units is not None:
            raise ValueError("units do not apply under unseen_subject, where the unit is the subject")
        if unit_name is not None:
            raise ValueError("unit_name does not apply under unseen_subject, where the unit is the subject")
        return
    if units is None:
        raise ValueError("seen_subject needs units")
    if unit_name is None:
        raise ValueError("unit_name is required under seen_subject (for example 'recording' or 'clip')")
    missing = _count_missing(units)
    if missing:
        raise ValueError(f"{missing} rows have no unit id (None, NaN or empty)")


def _check_fractions(fractions: Tuple[float, ...]) -> None:
    if len(fractions) != 3:
        raise ValueError("fractions must be (train, val, test)")
    if any(f < 0 or f >= 1 for f in fractions):
        raise ValueError("fractions must each be >= 0 and < 1")
    if not math.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to 1")


def _scheme_arguments(
    scheme: str,
    n_splits: Optional[int],
    strata: Optional[np.ndarray],
    strata_name: Optional[str],
    val_inner_splits: Optional[int],
    fractions: Optional[Tuple[float, ...]],
) -> int:
    """Refuse arguments the scheme would ignore; return the number of splits to make."""
    if (strata is None) != (strata_name is None):
        raise ValueError("strata and strata_name go together")
    if scheme in ("leave_one_subject_out", "group_shuffle") and strata is not None:
        raise ValueError(f"strata do not apply to {scheme}")
    if scheme in ("leave_one_subject_out", "seen_subject_holdout") and n_splits is not None:
        raise ValueError(f"n_splits does not apply to {scheme}")
    if scheme in KFOLD_SCHEMES or scheme == "leave_one_subject_out":
        if fractions is not None:
            raise ValueError(f"fractions do not apply to {scheme}")
    else:
        if fractions is None and scheme == "group_shuffle":
            raise ValueError("group_shuffle needs fractions=(train, val, test)")
        if fractions is not None:
            _check_fractions(fractions)
    if val_inner_splits is not None:
        if scheme not in KFOLD_SCHEMES:
            raise ValueError(f"val_inner_splits does not apply to {scheme}")
        if val_inner_splits < 2:
            raise ValueError("val_inner_splits must be at least 2")
    if scheme in KFOLD_SCHEMES:
        count = DEFAULT_N_SPLITS if n_splits is None else n_splits
        if count < 2:
            raise ValueError("n_splits must be at least 2")
        return count
    if scheme == "group_shuffle":
        count = 1 if n_splits is None else n_splits
        if count < 1:
            raise ValueError("n_splits must be at least 1")
        return count
    return 0


def _stratify_token(base: str, strata_name: Optional[str]) -> str:
    return base if strata_name is None else f"{base}_{strata_name}"


def grouped_splits(
    labels: Union[np.ndarray, Sequence[Any]],
    subjects: Union[np.ndarray, Sequence[Any]],
    *,
    protocol: Protocol,
    scheme: Scheme,
    seed: int,
    n_splits: Optional[int] = None,
    units: Optional[Union[np.ndarray, Sequence[Any]]] = None,
    unit_name: Optional[str] = None,
    strata: Optional[Union[np.ndarray, Sequence[Any]]] = None,
    strata_name: Optional[str] = None,
    val_inner_splits: Optional[int] = None,
    fractions: Optional[Tuple[float, float, float]] = None,
) -> Tuple[List[Split], SplitReport]:
    """Make grouped, protocol-checked splits.

    An argument that does not apply to the scheme or protocol is refused rather than ignored.

    Args:
        labels: Label per row, or an `(n, L)` 0/1 matrix for `multilabel_stratified_group_kfold`.
        subjects: Subject key per row; none may be missing.
        protocol: `seen_subject` or `unseen_subject`; no default.
        scheme: One of `UNSEEN_SCHEMES` (with `unseen_subject`) or `SEEN_SCHEMES` (with `seen_subject`).
        seed: The random seed; no default.
        n_splits: Number of folds for the k-fold schemes (default 5, at least 2), or of shuffles for
            `group_shuffle` (default 1); refused for `leave_one_subject_out` and `seen_subject_holdout`.
        units: Unit key per row, unique within a subject; required under `seen_subject`, refused under
            `unseen_subject`.
        unit_name: The unit's name, reported as the split's unit; required under `seen_subject`
            (for example `recording` or `clip`), refused under `unseen_subject`.
        strata: Optional stratum per row (for example a timepoint), stratified together with the label;
            refused for `leave_one_subject_out`, `group_shuffle` and a multi-label matrix.
        strata_name: The strata's name; required with `strata`; the report's `stratify` is
            `class_<strata_name>`.
        val_inner_splits: When given (at least 2), a validation partition is carved from each training
            pool as one of this many inner folds; k-fold schemes only.
        fractions: `(train, val, test)`, each in [0, 1) and summing to 1; required for `group_shuffle`,
            optional for `seen_subject_holdout` (default 0.6/0.2/0.2), refused otherwise.

    Returns:
        The splits and their report.

    Raises:
        ProtocolViolation: If a split breaks the protocol's invariant (for example a seen-subject
            subject with fewer than two units).
        ValueError: If the inputs are malformed or the arguments do not fit the scheme or protocol.
    """
    y, groups = np.asarray(labels), np.asarray(subjects)
    unit_arr = None if units is None else np.asarray(units)
    strata_arr = None if strata is None else np.asarray(strata)
    _prepare(y, groups, protocol, unit_arr, unit_name, strata_arr)
    if scheme not in UNSEEN_SCHEMES + SEEN_SCHEMES:
        raise ValueError(f"scheme must be one of {get_args(Scheme)}")
    count = _scheme_arguments(scheme, n_splits, strata_arr, strata_name, val_inner_splits, fractions)
    if scheme in UNSEEN_SCHEMES:
        if protocol != "unseen_subject":
            raise ValueError(f"{scheme} is an unseen_subject scheme")
        if (scheme == "multilabel_stratified_group_kfold") != (y.ndim == 2):
            raise ValueError("an (n, L) label matrix goes with multilabel_stratified_group_kfold only")
        splits = _unseen_splits(
            scheme,
            y,
            groups,
            seed=seed,
            n_splits=count,
            strata=strata_arr,
            val_inner_splits=val_inner_splits,
            fractions=fractions,
        )
        stratify = (
            "none" if scheme in ("leave_one_subject_out", "group_shuffle") else _stratify_token("class", strata_name)
        )
    else:
        if protocol != "seen_subject":
            raise ValueError(f"{scheme} is a seen_subject scheme")
        assert unit_arr is not None
        if y.ndim != 1:
            raise ValueError(f"{scheme} needs one label per row")
        if scheme == "seen_subject_kfold":
            splits = _seen_kfold(
                y, groups, unit_arr, seed=seed, n_splits=count, strata=strata_arr, val_inner_splits=val_inner_splits
            )
        else:
            splits = _seen_holdout(
                y, groups, unit_arr, seed=seed, strata=strata_arr, fractions=fractions or DEFAULT_HOLDOUT_FRACTIONS
            )
        stratify = _stratify_token("class", strata_name)
    report = SplitReport.from_splits(
        splits,
        labels=y,
        subjects=groups,
        protocol=protocol,
        units=unit_arr,
        unit_name=unit_name or "subject",
        stratify=stratify,
        scheme=scheme,
        seed=seed,
        strata=strata_arr,
    )
    return splits, report


def check_split(
    splits: Union[Split, Sequence[Split]],
    subjects: Union[np.ndarray, Sequence[Any]],
    *,
    labels: Union[np.ndarray, Sequence[Any]],
    protocol: Protocol,
    units: Optional[Union[np.ndarray, Sequence[Any]]] = None,
    unit_name: Optional[str] = None,
    stratify: str = "none",
    strata: Optional[Union[np.ndarray, Sequence[Any]]] = None,
) -> SplitReport:
    """Check splits made elsewhere against a protocol and report them.

    Partitions must be integer row-index arrays. Boolean masks are refused rather than converted,
    because a mask and an index array of 0s and 1s are easy to confuse; pass `np.flatnonzero(mask)`.

    Args:
        splits: One split or several (for example the folds of a k-fold).
        subjects: Subject key per row; none may be missing.
        labels: Label per row (or an `(n, L)` 0/1 matrix), for the prevalence figures.
        protocol: `seen_subject` or `unseen_subject`; no default.
        units: Unit key per row; required under `seen_subject`, refused under `unseen_subject`.
        unit_name: The unit's name; required under `seen_subject`, refused under `unseen_subject`.
        stratify: What the caller stratified on, as the report should state it.
        strata: Optional stratum per row, for per-stratum prevalence (one label per row only).

    Returns:
        The report, with scheme `external`.

    Raises:
        ProtocolViolation: If a split breaks the protocol's invariant.
        ValueError: If the inputs are malformed: lengths differ, a subject id is missing, a partition
            is empty, holds duplicate, negative or out-of-range indices, or is not an integer array.
    """
    y, group_arr = np.asarray(labels), np.asarray(subjects)
    unit_arr = None if units is None else np.asarray(units)
    strata_arr = None if strata is None else np.asarray(strata)
    _prepare(y, group_arr, protocol, unit_arr, unit_name, strata_arr)
    as_list = [splits] if isinstance(splits, Split) else list(splits)
    return SplitReport.from_splits(
        as_list,
        labels=y,
        subjects=group_arr,
        protocol=protocol,
        units=unit_arr,
        unit_name=unit_name or "subject",
        stratify=stratify,
        scheme="external",
        seed=None,
        strata=strata_arr,
    )


def nested_grouped_splits(
    labels: Union[np.ndarray, Sequence[Any]],
    subjects: Union[np.ndarray, Sequence[Any]],
    *,
    protocol: Protocol,
    seed: int,
    outer: Mapping[str, Any],
    inner: Mapping[str, Any],
    units: Optional[Union[np.ndarray, Sequence[Any]]] = None,
    unit_name: Optional[str] = None,
    strata: Optional[Union[np.ndarray, Sequence[Any]]] = None,
    strata_name: Optional[str] = None,
) -> Tuple[List[Tuple[Split, List[Split]]], SplitReport, List[SplitReport]]:
    """Nested splits: inner splits drawn from each outer training partition only.

    Inner splits are made on the outer training rows alone, so by construction no inner partition
    holds an outer-test row, subject (unseen) or unit (seen).

    Args:
        labels: Label per row (or an `(n, L)` 0/1 matrix).
        subjects: Subject key per row.
        protocol: The protocol for both levels; no default.
        seed: The outer seed; outer split `i`'s inner seed is `seed + 10000 * (i + 1)`.
        outer: `grouped_splits` keyword arguments for the outer level (at least `scheme`).
        inner: The same for the inner level.
        units: Unit key per row; required under `seen_subject`.
        unit_name: The unit's name; required under `seen_subject`.
        strata: Optional stratum per row.
        strata_name: The strata's name; required with `strata`.

    Returns:
        `(outer split, its inner splits)` pairs with indices into the full data, the outer report, and
        one inner report per outer split, computed over that split's training rows.

    Raises:
        ProtocolViolation: If any split breaks the protocol. An inner failure says which outer split's
            inner pool it came from (for example a seen subject left with one unit in that pool).
        ValueError: As for `grouped_splits`.
    """
    y, groups = np.asarray(labels), np.asarray(subjects)
    unit_arr = None if units is None else np.asarray(units)
    strata_arr = None if strata is None else np.asarray(strata)
    common: Dict[str, Any] = {"protocol": protocol, "unit_name": unit_name, "strata_name": strata_name}
    outer_splits, outer_report = grouped_splits(
        y, groups, seed=seed, units=unit_arr, strata=strata_arr, **common, **dict(outer)
    )
    nested = []
    inner_reports = []
    for i, split in enumerate(outer_splits):
        pool = split.partitions["train"]
        try:
            local, report = grouped_splits(
                y[pool],
                groups[pool],
                seed=seed + INNER_SEED_STRIDE * (i + 1),
                units=None if unit_arr is None else unit_arr[pool],
                strata=None if strata_arr is None else strata_arr[pool],
                **common,
                **dict(inner),
            )
        except ProtocolViolation as error:
            raise ProtocolViolation(f"inner pool of outer split {split.name}: {error}") from error
        mapped = [Split(name=s.name, partitions={k: _sorted(pool[v]) for k, v in s.partitions.items()}) for s in local]
        nested.append((split, mapped))
        inner_reports.append(report)
    return nested, outer_report, inner_reports
