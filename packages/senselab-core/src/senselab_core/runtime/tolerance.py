"""Tolerance profiles (`tolerance@1.0`): where one backend's lives, its model, and outputs judged within it."""

import json
import re
from collections.abc import Mapping, Sequence
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import TypeVar

import numpy
import yaml

from senselab_core.kernel._base import Frozen, NonEmptyStr
from senselab_core.kernel.formats import TOLERANCE_FORMAT, FormatTag, Formatted, readable
from senselab_core.kernel.results import Intervals

Folder = TypeVar("Folder", Path, Traversable)

Value = list[float | None] | str
"""One output leaf: numbers as a float list, a None where a value is absent, or anything else as its JSON."""
Fields = dict[str, Value]
"""An output's leaves by dotted path."""


class NoToleranceProfile(LookupError):
    """A backend has no tolerance profile, so its outputs cannot be judged across contexts."""


class AxisSpread(Frozen):
    """The largest difference (or count of unequal values) over `pairs` pairs of runs differing only along `axis`."""

    axis: NonEmptyStr
    spread: float
    pairs: int


class FieldTolerance(Frozen):
    """One output field's spread per axis and its tolerance, the largest spread."""

    path: NonEmptyStr
    kind: NonEmptyStr
    spreads: tuple[AxisSpread, ...]
    tolerance: float
    shape_mismatches: int = 0


class Derivation(Frozen):
    """What a profile was measured from."""

    shas: tuple[str, ...]
    jobs: tuple[str, ...]
    split: NonEmptyStr
    items: tuple[str, ...]
    contexts: tuple[str, ...]


class ToleranceProfile(Formatted):
    """A backend's cross-context spread on the calibration split."""

    format: FormatTag = TOLERANCE_FORMAT
    task: NonEmptyStr
    backend: NonEmptyStr
    derivation: Derivation
    fields: tuple[FieldTolerance, ...]
    findings: tuple[str, ...]


def tolerance_profile_path(folder: Folder, task: str, backend: str) -> Folder:
    """Return `<folder>/<task>/<backend>.yaml`: a backend name is only unique within its task."""
    return folder.joinpath(task, f"{backend}.yaml")


def read_profile(folder: Path | Traversable, task: str, backend: str) -> ToleranceProfile:
    """Return the profile of `backend` of `task` under `folder`.

    Raises:
        NoToleranceProfile: there is none.
        pydantic.ValidationError: the file is not a `tolerance@1.0` profile.
    """
    path = tolerance_profile_path(folder, task, backend)
    if not path.is_file():
        raise NoToleranceProfile(f"backend {backend!r} of {task!r} has no tolerance profile")
    return ToleranceProfile.model_validate(yaml.safe_load(path.read_text("utf-8")))


def _numeric(value: object) -> bool:
    if isinstance(value, (bool, str)):
        return False
    try:
        return numpy.asarray(value).dtype.kind in "biuf"
    except (TypeError, ValueError):
        return False


def flatten(value: object, prefix: str = "") -> Fields:
    """Return the leaves of `value` by dotted path: numbers and arrays as float lists, anything else as its JSON."""
    if isinstance(value, numpy.ndarray) and not _numeric(value):
        return flatten(value.tolist(), prefix)
    if isinstance(value, Mapping):
        out: Fields = {}
        for key in sorted(value, key=str):
            out |= flatten(value[key], f"{prefix}.{key}" if prefix else str(key))
        return out
    if isinstance(value, (list, tuple)) and not _numeric(value):
        out = {}
        for n, element in enumerate(value):
            out |= flatten(element, f"{prefix}[{n}]")
        return out
    if _numeric(value):
        return {prefix: numpy.asarray(value, dtype=float).ravel().tolist()}
    return {prefix: json.dumps(value, sort_keys=True, default=str)}


INTERVALS_FORMAT = Intervals.model_fields["format"].default
"""The format of the labelled spans whose labels `relabel_turns` maps."""


def best_assignment(weights: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    """Return the one-to-one (row, column) pairs of largest total weight (Hungarian method), one per row or column.

    Pairs are as many as the smaller side; with a tie any optimum is returned.
    """
    n, m = len(weights), len(weights[0]) if weights else 0
    size = max(n, m)
    top = max((w for row in weights for w in row), default=0.0)
    cost = [[top - (weights[i][j] if i < n and j < m else 0.0) for j in range(size)] for i in range(size)]
    u, v, match, way = [0.0] * (size + 1), [0.0] * (size + 1), [0] * (size + 1), [0] * (size + 1)
    for i in range(1, size + 1):
        match[0], column = i, 0
        least, used = [float("inf")] * (size + 1), [False] * (size + 1)
        while match[column]:
            used[column], row, delta, nxt = True, match[column], float("inf"), 0
            for j in range(1, size + 1):
                if not used[j]:
                    reduced = cost[row - 1][j - 1] - u[row] - v[j]
                    if reduced < least[j]:
                        least[j], way[j] = reduced, column
                    if least[j] < delta:
                        delta, nxt = least[j], j
            for j in range(size + 1):
                if used[j]:
                    u[match[j]] += delta
                    v[j] -= delta
                else:
                    least[j] -= delta
            column = nxt
        while column:
            previous = way[column]
            match[column] = match[previous]
            column = previous
    return sorted((match[j] - 1, j - 1) for j in range(1, size + 1) if match[j] - 1 < n and j - 1 < m)


def _intervals(fields: Mapping[str, Value]) -> set[str]:
    """Return the path prefixes under which `fields` hold a flattened `Intervals`."""
    found = set()
    for path, value in fields.items():
        prefix, _, leaf = path.rpartition(".")
        if leaf == "format" and isinstance(value, str) and readable(INTERVALS_FORMAT, json.loads(value)):
            found.add(prefix)
    return found


def _relabel_one(expected: Mapping[str, Value], actual: Mapping[str, Value], prefix: str) -> Fields:
    label = re.compile(re.escape(f"{prefix}.label") + r"\[(\d+)\]")
    times = (f"{prefix}.start_ns", f"{prefix}.end_ns")

    def keys(fields: Mapping[str, object]) -> list[str]:
        found = [(int(m.group(1)), k) for k in fields if (m := label.fullmatch(k))]
        return [k for _, k in sorted(found)]

    ek, ak = keys(expected), keys(actual)
    if not ek or not ak or any(t not in expected or t not in actual for t in times):
        return dict(actual)
    elabel = [json.loads(str(expected[k])) for k in ek]
    alabel = [json.loads(str(actual[k])) for k in ak]
    rows, cols = sorted(set(elabel)), sorted(set(alabel))
    overlap = [[0.0] * len(cols) for _ in rows]
    spans = [[list(f[t]) for t in times] for f in (expected, actual)]
    for i, (es, ee) in enumerate(zip(*spans[0], strict=True)):
        for j, (a_s, a_e) in enumerate(zip(*spans[1], strict=True)):
            if None not in (es, ee, a_s, a_e):
                overlap[rows.index(elabel[i])][cols.index(alabel[j])] += max(0.0, min(ee, a_e) - max(es, a_s))
    mapped = {cols[c]: rows[r] for r, c in best_assignment(overlap) if overlap[r][c] > 0}
    out = dict(actual)
    for k, name in zip(ak, alabel, strict=True):
        out[k] = json.dumps(mapped.get(name, f"<unmatched:{name}>"))
    return out


def relabel_turns(expected: Mapping[str, Value], actual: Mapping[str, Value]) -> Fields:
    """Return `actual` with the labels of each flattened `Intervals` renamed to `expected`'s by most shared time.

    An `Intervals` is found by its `format` leaf on either side, under any path. Labels are arbitrary per run, so
    spans are compared under the one-to-one mapping DER scores under. A label with no counterpart (no shared time)
    is renamed to one no reference label can equal. `actual` is unchanged where either side has no spans.
    """
    out = dict(actual)
    for prefix in sorted(_intervals(expected) | _intervals(actual)):
        out = _relabel_one(expected, out, prefix)
    return out


def numeric_gap(a: Sequence[float | None], b: Sequence[float | None]) -> float | None:
    """Return the largest absolute difference of two equal-length numeric leaves, 0 when empty.

    A None agrees only with a None at the same position; a None against a number returns None, a disagreement
    and never a distance.
    """
    if any((x is None) != (y is None) for x, y in zip(a, b, strict=True)):
        return None
    pairs = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    return max((abs(x - y) for x, y in pairs), default=0.0)


def within(expected: Mapping[str, Value], actual: Mapping[str, Value], profile: ToleranceProfile) -> dict[str, str]:
    """Return, by path of `expected`, why `actual` differs from it beyond `profile`; empty when they agree.

    Speaker turns are compared under `relabel_turns`. A path differs when `actual` lacks it, its shape differs, a
    None stands against a number (`numeric_gap`),
    the profile has no tolerance for it, or the difference exceeds the profile's tolerance (an exact field must be
    equal).
    """
    bound = {f.path: f.tolerance for f in profile.fields}
    actual = relabel_turns(expected, actual)
    out = {}
    for path in sorted(expected):
        a, b = expected[path], actual.get(path)
        if b is None:
            out[path] = "missing from the compared output"
        elif path not in bound:
            out[path] = "the profile has no tolerance for it"
        elif isinstance(a, str) or isinstance(b, str):
            if a != b:
                out[path] = f"{a!r} != {b!r}"
        elif len(a) != len(b):
            out[path] = f"{len(a)} values against {len(b)}"
        elif (gap := numeric_gap(a, b)) is None:
            out[path] = "a None against a number"
        elif gap > bound[path]:
            out[path] = f"difference {gap:g} exceeds tolerance {bound[path]:g}"
    return out
