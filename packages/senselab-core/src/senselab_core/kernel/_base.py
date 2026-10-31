"""Shared base model and field types of the kernel."""

import math
import os
from collections.abc import Callable, Iterator, Mapping
from enum import Enum
from pathlib import PurePath
from types import MappingProxyType
from typing import Annotated, Any, Self, TypeVar

import numpy
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    PrivateAttr,
    Strict,
    StringConstraints,
)
from senselab_worker.canonical import (
    INT_TOKEN,
    MAX_SAFE_INT,
    canonical_npy_bytes,
    escaped_key,
    jcs_dumps,
    tagged_digest,
)
from senselab_worker.hashing import sha256_bytes

K = TypeVar("K")
V = TypeVar("V")
F = TypeVar("F", bound="Frozen")

_DIGEST_TAG = "value@1"


class FrozenMapping(Mapping[K, V]):
    """A read-only mapping that pickles and deep-copies.

    Item assignment and deletion raise `TypeError`; attribute assignment raises `AttributeError`.
    """

    __slots__ = ("_items",)
    _items: MappingProxyType[K, V]

    def __init__(self, items: Mapping[K, V]) -> None:
        object.__setattr__(self, "_items", MappingProxyType(dict(items)))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"FrozenMapping is read-only; cannot set {name!r}")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"FrozenMapping is read-only; cannot delete {name!r}")

    def __getitem__(self, key: K) -> V:
        return self._items[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"FrozenMapping({dict(self._items)!r})"

    def __reduce__(self) -> tuple[type["FrozenMapping[K, V]"], tuple[dict[K, V]]]:
        return type(self), (dict(self._items),)


def frozen_array(a: object) -> numpy.ndarray:
    """Return a C-order little-endian copy of `a` whose buffer is an immutable `bytes` object.

    Neither the returned array nor any array on its `.base` chain can be made writeable.

    Raises:
        ValueError: if `a` is an ndarray subclass (a masked array, for example), whose extra state
            a plain copy would drop, or holds Python objects.
    """
    if isinstance(a, numpy.ndarray) and type(a) is not numpy.ndarray:
        raise ValueError(f"not a plain numpy.ndarray: {type(a).__module__}.{type(a).__qualname__}")
    x = numpy.asarray(a)
    if x.dtype.hasobject:
        raise ValueError(f"an array of Python objects has no bytes to own: {x.dtype}")
    dtype = x.dtype.newbyteorder("<")
    data = numpy.ascontiguousarray(x, dtype=dtype).tobytes(order="C")
    return numpy.frombuffer(data, dtype=dtype).reshape(x.shape)


def _canonical_value(v: object) -> object:
    """Return a dumped field value as the JSON value its digest covers (the `value@1` rules in `canonical_digest`)."""
    if isinstance(v, numpy.integer):
        v = int(v)
    if isinstance(v, numpy.ndarray):
        return sha256_bytes(canonical_npy_bytes(v))
    if isinstance(v, Frozen):
        return _canonical_value(v.model_dump())
    if isinstance(v, Mapping):
        return {_key_text(k): _canonical_value(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_canonical_value(x) for x in v]
    if isinstance(v, (set, frozenset)):
        items = [_canonical_value(x) for x in v]
        return sorted(items, key=jcs_dumps)
    if isinstance(v, Enum):
        return _canonical_value(v.value)
    if isinstance(v, PurePath):
        return os.fspath(v)
    if isinstance(v, type):
        return f"{v.__module__}.{v.__qualname__}"
    if type(v) is int and abs(v) > MAX_SAFE_INT:
        return {INT_TOKEN: str(v)}
    return v


def _key_text(k: object) -> object:
    text = k.value if isinstance(k, Enum) else k
    return escaped_key(text) if isinstance(text, str) else text


def value_digest(type_name: str, fields: Mapping[str, object]) -> str:
    """Return `tagged_digest("value@1", {"type": type_name, "fields": fields})` with arrays as their `.npy` sha256."""
    return tagged_digest(_DIGEST_TAG, {"type": type_name, "fields": _canonical_value(fields)})


def _rebuild(cls: type[F], fields: dict[str, Any]) -> F:
    return cls.model_validate(fields)


class Frozen(BaseModel):
    """Immutable pydantic model that may hold numpy arrays.

    Every array a value holds is a `frozen_array`. Values compare and hash by `canonical_digest`, and
    every copy (pickle, `copy`, `deepcopy`, `model_copy`) is rebuilt through `model_validate`, so it
    runs every validator. `model_construct` is refused.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    _canonical: str | None = PrivateAttr(default=None)

    def canonical_digest(self) -> str:
        """Return `tagged_digest("value@1", {"type": <module.qualname>, "fields": <dump>})`.

        Each array in the dump is replaced by the sha256 of its canonical `.npy` bytes, mappings by
        objects, sets by sorted arrays, enums by their values, paths by their text, types by their
        qualified names, and an integer outside the I-JSON range (int64 nanoseconds, stat fields) by
        `{"$int": "<decimal>"}`. A mapping key that starts with `$` gains one more `$`, so no mapping
        held in a value encodes as that marker. A file-backed value's digest covers its path and
        stamp, not its bytes: two values over the same bytes at two paths are unequal.
        """
        if self._canonical is None:
            t = type(self)
            self._canonical = value_digest(f"{t.__module__}.{t.__qualname__}", self.model_dump())
        return self._canonical

    def __eq__(self, other: object) -> bool:
        """Return whether `other` is of the same type with the same `canonical_digest`."""
        if not isinstance(other, Frozen):
            return NotImplemented
        return type(self) is type(other) and self.canonical_digest() == other.canonical_digest()

    def __hash__(self) -> int:
        """Return `hash(self.canonical_digest())`."""
        return hash(self.canonical_digest())

    def __reduce__(self) -> tuple[Callable[..., Any], tuple[type[Self], dict[str, Any]]]:
        """Pickle as the dump, rebuilt through `model_validate`."""
        return _rebuild, (type(self), self.model_dump())

    def __copy__(self) -> Self:
        """Return a copy rebuilt through `model_validate`."""
        return type(self).model_validate(self.model_dump())

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        """Return a copy rebuilt through `model_validate`; held arrays are copied by validation."""
        copied = type(self).model_validate(self.model_dump())
        if memo is not None:
            memo[id(self)] = copied
        return copied

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        """Return `model_validate` of the dump merged with `update`, so every validator runs; `deep` has no effect.

        Raises:
            ValueError: if `update` names a field the type does not have.
        """
        unknown = sorted(set(update or {}) - set(type(self).model_fields), key=str)
        if unknown:
            raise ValueError(f"{type(self).__name__} has no fields {unknown}; known: {sorted(type(self).model_fields)}")
        return type(self).model_validate({**self.model_dump(), **(update or {})})

    @classmethod  # type: ignore[no-redef]
    def model_construct(cls, _fields_set: set[str] | None = None, **values: object) -> Self:  # type: ignore[override]
        """Refuse: a value is built only through validation.

        Raises:
            TypeError: always.
        """
        raise TypeError("Frozen values are built through validation")


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


def _plain_array(v: object) -> numpy.ndarray:
    if isinstance(v, numpy.ndarray) and type(v) is not numpy.ndarray:
        raise ValueError(f"not a plain numpy.ndarray: {type(v).__module__}.{type(v).__qualname__}")
    return numpy.asarray(v)


NUMERIC_KINDS = "iuf"
"""Dtype kinds a numeric array may hold: signed and unsigned integers and floats."""

LABEL_KINDS = "biufU"
"""Dtype kinds a label array may hold: those `canonical_npy_bytes` stores."""


def as_ns_array(v: object) -> numpy.ndarray:
    """Validate a 1-D signed or unsigned integer array and return it as a frozen int64 copy."""
    a = _plain_array(v)
    if a.ndim != 1 or a.dtype.kind not in "iu":
        raise ValueError(f"nanosecond times must be a 1-D integer array, got {a.ndim}-D {a.dtype}")
    if not numpy.can_cast(a.dtype, numpy.int64, casting="safe"):
        raise ValueError(f"nanosecond times must fit int64, got {a.dtype}")
    return frozen_array(a.astype(numpy.int64, copy=False))


def as_numeric_array(v: object, ndim: int) -> numpy.ndarray:
    """Validate an `ndim`-D integer or float array (no bool, complex or datetime) and return a frozen copy."""
    a = _plain_array(v)
    if a.ndim != ndim or a.dtype.kind not in NUMERIC_KINDS or a.dtype.char in "gG":
        raise ValueError(f"expected a {ndim}-D integer or float array, got {a.ndim}-D {a.dtype}")
    return frozen_array(a)


def as_label_array(v: object) -> numpy.ndarray:
    """Validate a 1-D array of bools, numbers or strings and return a frozen copy."""
    a = _plain_array(v)
    if a.ndim != 1 or a.dtype.kind not in LABEL_KINDS or a.dtype.char in "gG":
        raise ValueError(f"labels must be a 1-D bool, number or str array, got {a.ndim}-D {a.dtype}")
    return frozen_array(a)


INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1

Nanoseconds = Annotated[int, BeforeValidator(_numpy_int_to_int), Strict(), Field(ge=INT64_MIN, le=INT64_MAX)]
"""Integer nanoseconds in the int64 range; numpy integers are accepted, floats, bools and strings are not."""

NonNegativeNs = Annotated[Nanoseconds, Field(ge=0)]

NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]

FiniteFloat = Annotated[float, AfterValidator(_finite)]

Rate = Annotated[float, Strict(), Field(gt=0, allow_inf_nan=False)]
"""A sampling rate in Hz: a positive finite number; bools, strings and non-finite values are refused."""

NsArray = Annotated[numpy.ndarray, BeforeValidator(as_ns_array)]
"""1-D frozen int64 nanoseconds."""

LabelArray = Annotated[numpy.ndarray, BeforeValidator(as_label_array)]
"""1-D frozen labels."""

StrMap = Annotated[Mapping[str, str], AfterValidator(FrozenMapping), PlainSerializer(dict)]
"""A read-only str-to-str mapping; serialised as a dict."""
