"""Evaluation protocols and their split invariants.

`seen_subject` asks whether a model works on new material from people it has seen; `unseen_subject`
asks whether it works on people it has never seen. Every split and every report names one.

Violations raise `ProtocolViolation`, whose message gives counts only, never key values.
"""

from __future__ import annotations

from typing import Literal, Mapping, Optional, get_args

import numpy as np

Protocol = Literal["seen_subject", "unseen_subject"]
PROTOCOLS: tuple[str, ...] = get_args(Protocol)

UNSEEN_INVARIANT = "no subject in two partitions"
SEEN_INVARIANT = "every test subject in train; no (subject, unit) in two partitions; every subject has >= 2 units"


class ProtocolViolation(ValueError):
    """A split breaks its protocol's invariant."""


def _unit_keys(subjects: np.ndarray, units: np.ndarray) -> np.ndarray:
    """Composite `(subject, unit)` keys, so unit ids need only be unique within a subject."""
    return np.array([f"{s}\x1f{u}" for s, u in zip(subjects.tolist(), units.tolist())], dtype=object)


def _in_several(keys: np.ndarray, partitions: Mapping[str, np.ndarray]) -> int:
    """How many distinct keys occur in more than one partition."""
    owner: dict[object, str] = {}
    shared: set[object] = set()
    for name, rows in partitions.items():
        for key in set(keys[rows].tolist()):
            if owner.setdefault(key, name) != name:
                shared.add(key)
    return len(shared)


def _validate_partitions(partitions: Mapping[str, np.ndarray], n_rows: int) -> None:
    """Partitions are non-empty integer index arrays with unique, in-range entries (masks are refused)."""
    for name, rows in partitions.items():
        rows = np.asarray(rows)
        if rows.size == 0:
            raise ValueError(f"partition {name!r} is empty")
        if rows.dtype == bool:
            raise ValueError(f"partition {name!r} is a boolean mask; pass row indices (np.flatnonzero(mask))")
        if not np.issubdtype(rows.dtype, np.integer):
            raise ValueError(f"partition {name!r} must be an integer index array, not {rows.dtype}")
        if rows.ndim != 1:
            raise ValueError(f"partition {name!r} must be one-dimensional")
        if rows.min() < 0 or rows.max() >= n_rows:
            raise ValueError(f"partition {name!r} holds indices out of range for {n_rows} rows")
        if len(np.unique(rows)) != len(rows):
            raise ValueError(f"partition {name!r} holds duplicate indices")


def check_protocol(
    partitions: Mapping[str, np.ndarray],
    subjects: np.ndarray,
    *,
    protocol: Protocol,
    units: Optional[np.ndarray] = None,
) -> str:
    """Check one split's partitions against a protocol.

    Args:
        partitions: Partition name to row indices; must include ``train`` and ``test``.
        subjects: Subject key per row of the whole dataset.
        protocol: `seen_subject` or `unseen_subject`.
        units: Unit key per row (required for `seen_subject`); keyed together with the subject.

    Returns:
        The invariant that was checked, as text.

    Raises:
        ProtocolViolation: If the invariant does not hold.
        ValueError: If the protocol is unknown, ``train`` or ``test`` is missing, a partition is empty
            or not a unique, in-range integer index array, or `units` is missing under `seen_subject`.
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}")
    if "train" not in partitions or "test" not in partitions:
        raise ValueError("a split needs 'train' and 'test' partitions")
    _validate_partitions(partitions, len(subjects))
    rows_seen: set[int] = set()
    overlap = 0
    for rows in partitions.values():
        as_set = set(np.asarray(rows).tolist())
        overlap += len(rows_seen & as_set)
        rows_seen |= as_set
    if overlap:
        raise ProtocolViolation(f"{overlap} rows appear in more than one partition")
    if protocol == "unseen_subject":
        shared = _in_several(subjects, partitions)
        if shared:
            raise ProtocolViolation(f"{shared} subjects appear in more than one partition")
        return UNSEEN_INVARIANT
    if units is None:
        raise ValueError("seen_subject needs units")
    keys = _unit_keys(subjects, units)
    per_subject: dict[str, set[object]] = {}
    for subject, key in zip(subjects.tolist(), keys.tolist()):
        per_subject.setdefault(subject, set()).add(key)
    thin = sum(len(k) < 2 for k in per_subject.values())
    if thin:
        raise ProtocolViolation(f"{thin} subjects have fewer than 2 units")
    shared = _in_several(keys, partitions)
    if shared:
        raise ProtocolViolation(f"{shared} units appear in more than one partition")
    absent = len(set(subjects[partitions["test"]].tolist()) - set(subjects[partitions["train"]].tolist()))
    if absent:
        raise ProtocolViolation(f"{absent} test subjects are absent from train")
    return SEEN_INVARIANT
