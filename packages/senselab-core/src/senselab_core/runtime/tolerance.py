"""Tolerance profiles (`tolerance@1.0`): where one backend's lives, its model, and outputs judged within it."""

import json
from collections.abc import Mapping
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import TypeVar

import numpy
import yaml

from senselab_core.kernel._base import Frozen, NonEmptyStr
from senselab_core.kernel.formats import TOLERANCE_FORMAT, FormatTag, Formatted

Folder = TypeVar("Folder", Path, Traversable)

Fields = dict[str, list[float] | str]
"""An output's leaves by dotted path: numbers and arrays as float lists, anything else as its JSON."""


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
        numpy.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return True


def flatten(value: object, prefix: str = "") -> Fields:
    """Return the leaves of `value` by dotted path: numbers and arrays as float lists, anything else as its JSON."""
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


def within(
    expected: Mapping[str, list[float] | str], actual: Mapping[str, list[float] | str], profile: ToleranceProfile
) -> dict[str, str]:
    """Return, by path of `expected`, why `actual` differs from it beyond `profile`; empty when they agree.

    A path differs when `actual` lacks it, its shape differs, the profile has no tolerance for it, or the
    difference exceeds the profile's tolerance (an exact field must be equal).
    """
    bound = {f.path: f.tolerance for f in profile.fields}
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
        elif a and (gap := float(numpy.max(numpy.abs(numpy.asarray(a) - numpy.asarray(b))))) > bound[path]:
            out[path] = f"difference {gap:g} exceeds tolerance {bound[path]:g}"
    return out
