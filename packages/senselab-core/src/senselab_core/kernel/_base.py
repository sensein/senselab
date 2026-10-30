"""Shared base model and field types of the kernel."""

import math
from typing import Annotated

import numpy
from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, Strict, StringConstraints


class Frozen(BaseModel):
    """Immutable pydantic model that may hold numpy arrays."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


def readonly(a: numpy.ndarray) -> numpy.ndarray:
    """Return a non-writeable view of `a`, leaving `a` itself writeable."""
    view = a.view()
    view.flags.writeable = False
    return view


def _numpy_int_to_int(v: object) -> object:
    return int(v) if isinstance(v, numpy.integer) else v


def require_ns(t: object, name: str) -> int:
    """Return `t` as an int if it is an integer (numpy integers included, bools not); raise `TypeError` otherwise."""
    if isinstance(t, bool) or not isinstance(t, (int, numpy.integer)):
        raise TypeError(f"{name} must be integer nanoseconds, got {type(t).__name__}")
    return int(t)


def _finite(v: float) -> float:
    if not math.isfinite(v):
        raise ValueError(f"must be finite, got {v}")
    return v


def as_ns_array(v: object) -> numpy.ndarray:
    """Validate a 1-D signed or unsigned integer array and return it as read-only int64 nanoseconds."""
    a = numpy.asarray(v)
    if a.ndim != 1 or a.dtype.kind not in "iu":
        raise ValueError(f"nanosecond times must be a 1-D integer array, got {a.ndim}-D {a.dtype}")
    return readonly(a.astype(numpy.int64, casting="safe", copy=False))


def as_numeric_array(v: object, ndim: int) -> numpy.ndarray:
    """Validate an `ndim`-D boolean-free numeric array and return a read-only view of it."""
    a = numpy.asarray(v)
    if a.ndim != ndim or a.dtype.kind not in "iufc":
        raise ValueError(f"expected a {ndim}-D numeric array, got {a.ndim}-D {a.dtype}")
    return readonly(a)


def as_label_array(v: object) -> numpy.ndarray:
    """Validate a 1-D array and return a read-only view of it."""
    a = numpy.asarray(v)
    if a.ndim != 1:
        raise ValueError(f"labels must be a 1-D array, got {a.ndim}-D")
    return readonly(a)


Nanoseconds = Annotated[int, BeforeValidator(_numpy_int_to_int), Strict()]
"""Integer nanoseconds; numpy integers are accepted, floats, bools and strings are not."""

NonNegativeNs = Annotated[Nanoseconds, Field(ge=0)]

NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]

FiniteFloat = Annotated[float, AfterValidator(_finite)]

NsArray = Annotated[numpy.ndarray, BeforeValidator(as_ns_array)]
"""1-D read-only int64 nanoseconds."""

LabelArray = Annotated[numpy.ndarray, BeforeValidator(as_label_array)]
"""1-D read-only labels."""
