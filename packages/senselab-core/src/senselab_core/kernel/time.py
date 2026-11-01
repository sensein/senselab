"""Integer-nanosecond time on named clocks."""

from fractions import Fraction
from typing import Annotated

from pydantic import Field, model_validator

from senselab_core.kernel._base import (
    INT64_MAX,
    INT64_MIN,
    FiniteFloat,
    Frozen,
    Nanoseconds,
    NonEmptyStr,
    NonNegativeNs,
    require_ns,
)
from senselab_core.kernel.formats import FormatTag, Formatted


class TimeAnchor(Frozen):
    """Ties `clock_id` to absolute time: its zero is `absolute_start_ns` since the Unix epoch, if known."""

    clock_id: NonEmptyStr
    absolute_start_ns: Nanoseconds | None


class TimeMap(Formatted):
    """Maps a time on clock `source` to clock `target`: `t + offset_ns + t * drift_ppm / 1e6`.

    `drift_ppm` lies in `(-1e6, 1e6)`, so the map never reverses time. `uncertainty_ns` bounds the
    error of a mapped time.
    """

    format: FormatTag = "timemap@1.0"

    source: NonEmptyStr
    target: NonEmptyStr
    offset_ns: Nanoseconds
    drift_ppm: Annotated[FiniteFloat, Field(gt=-1e6, lt=1e6)] = 0.0
    uncertainty_ns: NonNegativeNs = 0

    def map(self, t_ns: int) -> int:
        """Return `t_ns` on the target clock, with the drift term computed exactly and rounded half to even.

        Raises:
            TypeError: if `t_ns` is not an integer.
            OverflowError: if `t_ns` or the mapped time is outside the int64 nanosecond range.
        """
        t_ns = require_ns(t_ns, "t_ns")
        mapped = t_ns + self.offset_ns + round(Fraction(self.drift_ppm) * t_ns / 1_000_000)
        for name, value in (("t_ns", t_ns), ("the mapped time", mapped)):
            if not INT64_MIN <= value <= INT64_MAX:
                raise OverflowError(f"{name} {value} is outside the int64 nanosecond range")
        return mapped


class Span(Frozen):
    """The half-open interval `[start_ns, end_ns)` on `clock_id`; `end_ns >= start_ns`."""

    start_ns: Nanoseconds
    end_ns: Nanoseconds
    clock_id: NonEmptyStr

    @model_validator(mode="after")
    def _non_negative_duration(self) -> "Span":
        if self.end_ns < self.start_ns:
            raise ValueError(f"span ends before it starts: {self.start_ns} > {self.end_ns}")
        return self
