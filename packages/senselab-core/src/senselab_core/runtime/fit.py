"""Whether a unit fits a resource class, and which class it goes to: the placement rule the plan and the pool share.

A class is too small for a unit when its measured resident memory plus the fixed peak `m0` exceeds
`sizing.memory_headroom` of the class's grant: no batch of any size then fits beside the model. A class is unfit
when a load of the unit ran out of memory in a grant at least as large (`UnfitRecord`). `choose` walks a class and
the classes above it (`larger_class`) and returns the first that fits, with what it found of each class.
"""

import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import senselab_worker
from senselab_worker.digest import file_digest, identity_digest
from senselab_worker.faults import ResourceFault

from senselab_core.kernel.declare import UnknownTask
from senselab_core.runtime.calibration import MemoryFit, UnfitRecord
from senselab_core.runtime.envs import overlay_digest
from senselab_core.runtime.profiles import ExecutionProfile

if TYPE_CHECKING:
    from senselab_core.runtime.measures import MeasureRoots, MeasureUnit

Verdict = Literal["fits", "too small", "unfit", "grant unknown"]


def larger_class(profile: ExecutionProfile, name: str) -> str | None:
    """Return the resource class of `profile` with the least memory above class `name`'s, on its device and GPU type.

    A class whose `cpus` exceed the substrate's `qos_cpu_cap` is skipped: no worker of it could start. None when
    `name` is not a class of the profile, states no memory, or no startable class of its kind has more.
    """
    cap = profile.substrate.qos_cpu_cap
    current = profile.resource_classes.get(name)
    if current is None or current.mem_bytes is None:
        return None
    larger = sorted(
        (c.mem_bytes, n)
        for n, c in profile.resource_classes.items()
        if (c.device, c.gpu_type) == (current.device, current.gpu_type)
        and c.mem_bytes is not None
        and c.mem_bytes > current.mem_bytes
        and (cap is None or c.cpus is None or c.cpus <= cap)
    )
    return larger[0][1] if larger else None


@dataclass(frozen=True)
class ClassTried:
    """One class a unit was judged against: its grant, the verdict, and the figures the verdict rests on.

    `resident` and `m0` are the unit's measured bytes (None unmeasured); `limit` is `headroom × grant`; `unfit_grant`
    is the largest grant a load of the unit ran out of memory in, when one is recorded at or above this grant.
    """

    resource_class: str
    grant_bytes: int | None
    verdict: Verdict
    resident: int | None = None
    m0: float | None = None
    limit: float | None = None
    unfit_grant: int | None = None

    def line(self) -> str:
        """Return the class and what was found of it, with the numbers."""
        if self.verdict == "fits":
            return f"class {self.resource_class} ({self.grant_bytes} bytes) fits"
        if self.verdict == "too small":
            return (
                f"class {self.resource_class} ({self.grant_bytes} bytes) is too small: resident {self.resident} + "
                f"m0 {self.m0:g} bytes exceed {self.limit:.0f} bytes, its headroom of the grant"
            )
        if self.verdict == "unfit":
            return (
                f"class {self.resource_class} ({self.grant_bytes} bytes) is unfit: a load ran out of memory in a "
                f"grant of {self.unfit_grant} bytes"
            )
        return f"class {self.resource_class} states no memory to judge"


class Unplaceable(RuntimeError, ResourceFault):  # noqa: N818 - named for what the plan is, as its callers know it
    """A unit that fits no class the profile has at or above its own: the plan cannot run.

    A plan outcome, not a fault of the environment: its record is a resource fault, never counted against an env.

    Attributes:
        subject: What was placed: a node id or a unit label.
        tried: Each class judged, in order, with its verdict and numbers.
    """

    def __init__(self, subject: str, tried: Sequence[ClassTried]) -> None:
        """Name the subject and each class tried."""
        super().__init__(f"{subject} fits no resource class: " + "; ".join(t.line() for t in tried))
        self.subject = subject
        self.tried = tuple(tried)


def moved_why(why: str, start: str) -> str:
    """Return a placement's reason once its node has moved up from class `start`; no measured figure is in it."""
    return f"{why}; moved from {start}"


@dataclass(frozen=True)
class Code:
    """The code a load runs under: the digest of the senselab source and of its backend's worker overlay."""

    source: str
    overlay: str


@functools.cache
def source_digest() -> str:
    """Return the digest of the senselab-core and senselab-worker source files: the code a load ran under."""
    listed = []
    for package in (Path(__file__).parents[1], Path(senselab_worker.__file__).parent):
        for path in sorted(package.rglob("*.py")):
            listed.append([f"{package.name}/{path.relative_to(package).as_posix()}", file_digest("file@1", path)])
    return identity_digest("core-runtime@1", listed)


def code_of(unit: "MeasureUnit") -> Code:
    """Return the code a load of `unit` runs under: the senselab source and its backend's worker overlay.

    A fix to either gives another `Code`, so an `unfit` record of the code it fixes no longer binds the unit.
    """
    from senselab_core.runtime.call import packaged  # noqa: PLC0415 - call imports the runtime that imports this

    sources: dict[str, Path] = {}
    try:
        declared = packaged().registry.backends(unit.task)
    except UnknownTask:
        declared = []
    for backend in declared:
        if backend.name == unit.backend:
            sources = {module: Path(file) for module, file in backend.worker_sources.items()}
    return Code(source_digest(), overlay_digest(sources))


def current_unfit(roots: "MeasureRoots", unit: "MeasureUnit") -> tuple[UnfitRecord, ...]:
    """Return the `unfit` records of `unit` written under the code it runs under now."""
    code = code_of(unit)
    return tuple(r for r in roots.unfit_records(unit) if (r.source, r.overlay) == (code.source, code.overlay))


@dataclass(frozen=True)
class Choice:
    """The class a unit goes to (None: none fits) and every class tried on the way, the first being its own."""

    to: str | None
    tried: tuple[ClassTried, ...]


def judge(  # noqa: PLR0913 - the class and its grant, what was measured, what was recorded, and the headroom
    name: str,
    grant_bytes: int | None,
    fit: MemoryFit | None,
    resource: Literal["device", "host"],
    unfit: Sequence[UnfitRecord],
    headroom: float,
) -> ClassTried:
    """Return the verdict on class `name` of `grant_bytes`: unfit, too small, fits, or no grant to judge by."""
    if grant_bytes is None:
        return ClassTried(name, None, "grant unknown")
    held = [r.grant_bytes for r in unfit if r.resource == resource and r.grant_bytes >= grant_bytes]
    if held:
        return ClassTried(name, grant_bytes, "unfit", unfit_grant=max(held))
    if fit is not None and fit.resident + fit.m0 > headroom * grant_bytes:
        return ClassTried(name, grant_bytes, "too small", fit.resident, fit.m0, headroom * grant_bytes)
    return ClassTried(name, grant_bytes, "fits", None if fit is None else fit.resident, None if fit is None else fit.m0)


def choose(  # noqa: PLR0913 - the profile and the class, what the unit measured and recorded, and the grants
    profile: ExecutionProfile,
    start: str,
    *,
    fit: MemoryFit | None,
    resource: Literal["device", "host"],
    unfit: Sequence[UnfitRecord],
    grant_of: Callable[[str], int | None],
) -> Choice:
    """Return the first class from `start` up (`larger_class`) the unit fits, with each class tried.

    `grant_of` gives a class's memory grant in bytes, None when it states none.
    """
    tried: list[ClassTried] = []
    name: str | None = start
    while name is not None:
        found = judge(name, grant_of(name), fit, resource, unfit, profile.sizing.memory_headroom)
        tried.append(found)
        if found.verdict in ("fits", "grant unknown"):
            return Choice(name, tuple(tried))
        name = larger_class(profile, name)
    return Choice(None, tuple(tried))


__all__ = [
    "Choice",
    "ClassTried",
    "Unplaceable",
    "choose",
    "Code",
    "code_of",
    "current_unfit",
    "judge",
    "larger_class",
    "moved_why",
    "source_digest",
]
