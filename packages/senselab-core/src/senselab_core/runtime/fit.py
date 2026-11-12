"""Whether a unit fits a resource class, and which class it goes to: the placement rule the plan and the pool share.

A class is too small for a unit when its measured resident memory plus the fixed peak `m0` exceeds
`sizing.memory_headroom` of the class's grant: no batch of any size then fits beside the model. A class is unfit
when a load of the unit ran out of memory in a grant at least as large (`UnfitRecord`). `choose` walks a class and
the classes above it (`larger_class`) and returns the first that fits, with what it found of each class.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from senselab_core.runtime.calibration import MemoryFit, UnfitRecord
from senselab_core.runtime.profiles import ExecutionProfile

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


class Unplaceable(RuntimeError):  # noqa: N818 - named for what the plan is, as its callers know it
    """A unit that fits no class the profile has at or above its own: the plan cannot run.

    Attributes:
        subject: What was placed: a node id or a unit label.
        tried: Each class judged, in order, with its verdict and numbers.
    """

    def __init__(self, subject: str, tried: Sequence[ClassTried]) -> None:
        """Name the subject and each class tried."""
        super().__init__(f"{subject} fits no resource class: " + "; ".join(t.line() for t in tried))
        self.subject = subject
        self.tried = tuple(tried)


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


__all__ = ["Choice", "ClassTried", "Unplaceable", "choose", "judge", "larger_class"]
