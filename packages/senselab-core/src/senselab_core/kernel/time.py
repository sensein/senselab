"""Integer-nanosecond time on named clocks."""

from collections.abc import Mapping
from fractions import Fraction
from typing import Annotated, ClassVar, TypeVar, cast

import numpy
from pydantic import Field, model_validator

from senselab_core.kernel._base import (
    INT64_MAX,
    INT64_MIN,
    FiniteFloat,
    Frozen,
    FrozenMapping,
    Nanoseconds,
    NonEmptyStr,
    NonNegativeNs,
    require_ns,
)
from senselab_core.kernel.formats import FormatTag, Formatted


class TimeAnchor(Frozen):
    """Ties `clock_id` to absolute time: its zero is `absolute_start_ns` since the Unix epoch, if known."""

    clock_fields: ClassVar[tuple[str, ...]] = ()

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

    clock_fields: ClassVar[tuple[str, ...]] = ("start_ns", "end_ns")

    start_ns: Nanoseconds
    end_ns: Nanoseconds
    clock_id: NonEmptyStr

    @model_validator(mode="after")
    def _non_negative_duration(self) -> "Span":
        if self.end_ns < self.start_ns:
            raise ValueError(f"span ends before it starts: {self.start_ns} > {self.end_ns}")
        return self


T = TypeVar("T")


def rebind(output: T, stored_anchor: TimeAnchor, reader_anchor: TimeAnchor, timemap: TimeMap) -> T:
    """Return `output` with every clock-relative field on the stored anchor's clock moved to the reader's.

    Walking kernel values, tuples and mappings, a value whose `clock_id` is `stored_anchor.clock_id` gets the
    reader's `clock_id` and each of its `clock_fields` (integers or integer arrays) mapped by `timemap`; a
    `TimeAnchor` naming the stored clock becomes `reader_anchor`. Anything else is returned as it is.

    Raises:
        ValueError: if `timemap` does not map the stored clock to the reader's, or a `TimeAnchor` is to be
            re-bound through a map that moves time.
    """
    source, target = stored_anchor.clock_id, reader_anchor.clock_id
    if (timemap.source, timemap.target) != (source, target):
        raise ValueError(f"timemap maps {timemap.source!r} to {timemap.target!r}, not {source!r} to {target!r}")
    if source == target and timemap.offset_ns == 0 and timemap.drift_ppm == 0:
        return output
    return cast(T, _rebound(output, reader_anchor, timemap))


def _rebound(value: object, anchor: TimeAnchor, timemap: TimeMap) -> object:
    if isinstance(value, TimeAnchor):
        if value.clock_id == timemap.source and (timemap.offset_ns != 0 or timemap.drift_ppm != 0):
            raise ValueError("a recording's anchor re-binds only through a map that keeps time")
        return anchor if value.clock_id == timemap.source else value
    if isinstance(value, Frozen):
        fields = {name: getattr(value, name) for name in type(value).model_fields}
        moved = {name: _rebound(v, anchor, timemap) for name, v in fields.items()}
        if fields.get("clock_id") == timemap.source:
            moved = {
                name: _mapped(v, timemap) if name in (type(value).clock_fields or ()) else v
                for name, v in moved.items()
            }
            moved["clock_id"] = timemap.target
        return value if all(moved[n] is fields[n] for n in fields) else type(value).model_validate(moved)
    if isinstance(value, tuple | Mapping):
        items = dict(value.items() if isinstance(value, Mapping) else enumerate(value))
        moved = {k: _rebound(v, anchor, timemap) for k, v in items.items()}
        if all(moved[k] is items[k] for k in items):
            return value
        return FrozenMapping(moved) if isinstance(value, Mapping) else tuple(moved.values())
    return value


def _mapped(times: object, timemap: TimeMap) -> object:
    if isinstance(times, numpy.ndarray):
        return numpy.array([timemap.map(int(t)) for t in times], dtype=numpy.int64)
    if isinstance(times, int):
        return timemap.map(times)
    return times
