"""Tolerance profiles (`tolerance@2`): where one backend's lives, its model, and outputs judged within it."""

import math
from collections.abc import Mapping, Sequence
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Annotated, Self, TypeVar

import yaml
from pydantic import Field, model_validator

from senselab_core.kernel._base import Frozen, NonEmptyStr
from senselab_core.kernel.formats import TOLERANCE_FORMAT, FormatTag, Formatted, readable
from senselab_core.kernel.results import Intervals
from senselab_core.runtime.compare import (  # noqa: F401 - read here until profiles retire
    FLATTEN_VERSION,
    Fields,
    Value,
    best_assignment,
    flatten,
    index_free,
    stale_flatten,
)
from senselab_core.runtime.compare import relabel as relabel_turns
from senselab_core.runtime.storage import load_policy_values

Folder = TypeVar("Folder", Path, Traversable)


class NoToleranceProfile(LookupError):
    """A backend has no tolerance profile, so its outputs cannot be judged across contexts."""


class ToleranceProfileRetired(ValueError):
    """A tolerance profile in a retired format; the message names the migration."""


class AxisSpread(Frozen):
    """The largest difference (or count of unequal values) over `pairs` pairs of runs differing only along `axis`."""

    axis: NonEmptyStr
    spread: float
    pairs: int


class FieldTolerance(Frozen):
    """One output field's spread per axis and its tolerance, the largest spread, over every index of its path.

    `path` is `index_free`: `segments[*].text` holds every segment's text.
    """

    path: NonEmptyStr
    kind: NonEmptyStr
    spreads: tuple[AxisSpread, ...]
    tolerance: float
    shape_mismatches: int = 0


COVERAGE_FORMAT = "tolerance-coverage@1.0"
Fraction = Annotated[float, Field(gt=0, lt=1)]


def required_items(target: float, confidence: float) -> int:
    """Return the least n whose largest spread covers `target` of items with probability `confidence`."""
    return math.ceil(math.log(1 - confidence) / math.log(target))


def attained_coverage(items: int, confidence: float) -> float:
    """Return the fraction of items the largest of `items` spreads covers with probability `confidence`."""
    return float((1 - confidence) ** (1 / items))


class Coverage(Frozen):
    """The coverage a profile states, from its item count.

    Each tolerance, the largest spread of the calibration items, is a distribution-free one-sided upper tolerance
    bound covering `attained` of items with probability `confidence`. `target` is the coverage
    `data/tolerance_coverage.yaml` asks for, which `required_items` items reach.
    """

    target: Fraction
    confidence: Fraction
    required_items: int
    attained: Fraction

    @classmethod
    def of(cls, items: int) -> Self:
        """Return the coverage `items` calibration items state under the packaged target and confidence."""
        policy = load_policy_values("senselab_core.runtime", "tolerance_coverage.yaml", COVERAGE_FORMAT)
        target, confidence = float(policy["target"]), float(policy["confidence"])  # type: ignore[arg-type]
        return cls(
            target=target,
            confidence=confidence,
            required_items=required_items(target, confidence),
            attained=attained_coverage(items, confidence),
        )

    @model_validator(mode="after")
    def _check_required(self) -> Self:
        if self.required_items != required_items(self.target, self.confidence):
            raise ValueError(f"{self.target}/{self.confidence} needs {required_items(self.target, self.confidence)}")
        return self


class Derivation(Frozen):
    """What a profile was measured from, and the coverage its item count states (absent before `tolerance@2.1`)."""

    shas: tuple[str, ...]
    jobs: tuple[str, ...]
    split: NonEmptyStr
    items: tuple[str, ...]
    contexts: tuple[str, ...]
    coverage: Coverage | None = None

    @model_validator(mode="after")
    def _check_coverage(self) -> Self:
        stated = self.coverage
        if stated is not None and not math.isclose(
            stated.attained, attained_coverage(len(self.items), stated.confidence)
        ):
            raise ValueError(f"{len(self.items)} items do not state coverage {stated.attained}")
        return self


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
        ToleranceProfileRetired: it is in a retired format.
        pydantic.ValidationError: the file is not a `tolerance@2` profile.
    """
    path = tolerance_profile_path(folder, task, backend)
    if not path.is_file():
        raise NoToleranceProfile(f"backend {backend!r} of {task!r} has no tolerance profile")
    return load_profile(path)


def load_profile(path: Path | Traversable) -> ToleranceProfile:
    """Return the profile in the file `path`.

    Raises:
        ToleranceProfileRetired: it is in a retired format.
        pydantic.ValidationError: the file is not a `tolerance@2` profile.
    """
    raw = yaml.safe_load(path.read_text("utf-8"))
    found = raw.get("format") if isinstance(raw, dict) else None
    if isinstance(found, str) and found.startswith("tolerance@") and not readable(TOLERANCE_FORMAT, found):
        raise ToleranceProfileRetired(
            f"the profile {path.name} is {found}, read {TOLERANCE_FORMAT}: regenerate it with"
            " scripts/generate_tolerance.py generate from observations of the current flatten"
        )
    return ToleranceProfile.model_validate(raw)


INTERVALS_FORMAT = Intervals.model_fields["format"].default


def numeric_gap(a: Sequence[float | None], b: Sequence[float | None]) -> float | None:
    """Return the largest absolute difference of two equal-length numeric leaves, 0 when empty.

    A None agrees only with a None at the same position; a None against a number returns None, a disagreement
    and never a distance. Equal values (an infinity against the same infinity) differ by 0; a pair with any
    other non-finite value differs by `inf`.
    """
    if any((x is None) != (y is None) for x, y in zip(a, b, strict=True)):
        return None
    pairs = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    return max((_pair_gap(x, y) for x, y in pairs), default=0.0)


def _pair_gap(x: float, y: float) -> float:
    if x == y:
        return 0.0
    gap = abs(x - y)
    return gap if math.isfinite(gap) else math.inf


def within(expected: Mapping[str, Value], actual: Mapping[str, Value], profile: ToleranceProfile) -> dict[str, str]:
    """Return, by path of `expected`, why `actual` differs from it beyond `profile`; empty when they agree.

    Each path is judged by the profile's field of its `index_free` form. Speaker turns are compared under
    `relabel_turns`. A path differs when `actual` lacks it, its shape differs, a
    None stands against a number (`numeric_gap`),
    the profile has no tolerance for it, or the difference exceeds the profile's tolerance or is infinite (an exact
    field must be equal).
    """
    bound = {f.path: f.tolerance for f in profile.fields}
    actual = relabel_turns(expected, actual)
    out = {}
    for path in sorted(expected):
        a, b = expected[path], actual.get(path)
        if b is None:
            out[path] = "missing from the compared output"
        elif index_free(path) not in bound:
            out[path] = "the profile has no tolerance for it"
        elif isinstance(a, str) or isinstance(b, str):
            if a != b:
                out[path] = f"{a!r} != {b!r}"
        elif len(a) != len(b):
            out[path] = f"{len(a)} values against {len(b)}"
        elif (gap := numeric_gap(a, b)) is None:
            out[path] = "a None against a number"
        elif gap > bound[index_free(path)] or math.isinf(gap):
            out[path] = f"difference {gap:g} exceeds tolerance {bound[index_free(path)]:g}"
    return out
