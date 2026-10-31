"""Canonical byte forms of JSON values and numpy arrays, and the tagged digest over them (ADR-006).

`jcs_dumps` writes RFC 8785 (JCS) JSON with strings in Unicode NFC; `canonical_npy_bytes` writes
`.npy` version 1.0, C order, little-endian; `tagged_digest` hashes a JSON value under a registered
domain tag. Both encoders accept an exact-type whitelist and refuse everything else.
"""

import hashlib
import json
import math
import struct
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Callable, cast

import numpy

_MAX_SAFE_INT = 2**53 - 1
_PLATFORM_DTYPE_CHARS = "gG"
_CANONICAL_KINDS = "biufU"
_PY_SCALARS = (type(None), bool, int, float, str)
_NPY_MAGIC = b"\x93NUMPY\x01\x00"
_NPY_ALIGN = 64
MAX_DEPTH = 64
"""The most containers on any path from a value's root to a leaf, the root included."""

DIGEST_TAGS: frozenset[str] = frozenset(
    {
        "logical-key@2",
        "output@2",
        "context@2",
        "content-root@2",
        "audio@2",
        "weights@2",
        "value@1",
        "provenance-id@1",
        "data-manifest@2",
    }
)
"""The registered domain tags of `tagged_digest`; no tag is a prefix of another."""

Scalar = None | bool | int | float | str


def canonical_scalar(v: object) -> Scalar:
    """Return `v` as the exact Python JSON scalar it encodes, dispatching on `type(v)` exactly.

    `None`, `bool`, `int`, `float` and `str` pass unchanged. A numpy scalar passes as its `.item()`
    only if its type is defined by numpy itself, its dtype kind is one of `b i u f U`, and its
    `.item()` is one of those Python types (so `longdouble` is refused).

    Raises:
        TypeError: on any other value: subclasses (including `Enum` members with a scalar mix-in),
            `datetime64`, `timedelta64`, `void`, complex, bytes and object scalars.
    """
    t = type(v)
    if t in _PY_SCALARS:
        return v  # type: ignore[return-value]
    if isinstance(v, numpy.generic) and t.__module__ == "numpy" and v.dtype.kind in _CANONICAL_KINDS:
        item = v.item()
        if type(item) in _PY_SCALARS:
            return item  # type: ignore[no-any-return]
    raise TypeError(f"not a canonical JSON scalar: {t.__module__}.{t.__qualname__}")


def jcs_dumps(obj: object) -> bytes:
    """Serialise a JSON value as RFC 8785 canonical UTF-8 bytes.

    Containers are exactly `list`, `tuple` and `dict` with `str` keys; scalars are what
    `canonical_scalar` accepts. Strings and keys are NFC-normalised; keys sort by UTF-16 code units.

    Raises:
        ValueError: on a NaN or infinite float, an integer outside +/-(2**53 - 1), a string that
            is not valid Unicode, two keys of one object that are equal after NFC, a container
            that contains itself, or nesting deeper than `MAX_DEPTH`.
        TypeError: on any other type, including container subclasses, non-`str` keys and arrays.
    """
    parts: list[str] = []
    _write(obj, parts.append, set())
    return "".join(parts).encode("utf-8")


def jcs_loads(data: bytes | str) -> object:
    """Parse JCS bytes back to the value `jcs_dumps` wrote them from.

    An integer literal outside +/-(2**53 - 1) can only come from a float, so it parses as that
    float; `jcs_dumps(jcs_loads(b)) == b` for every `b` that `jcs_dumps` returns.

    Raises:
        ValueError: on text that is not JSON, or on `NaN` and `Infinity` literals.
    """
    return json.loads(data, parse_int=_parse_int, parse_constant=_refuse_constant)


def _parse_int(text: str) -> int | float:
    value = int(text)
    return value if abs(value) <= _MAX_SAFE_INT else float(text)


def _refuse_constant(text: str) -> object:
    raise ValueError(f"non-finite number is not JSON: {text}")


def tagged_digest(tag: str, value: object) -> str:
    """Return the sha256 hex of `tag` in ASCII, a zero byte and the JCS of `value`.

    Raises:
        ValueError: if `tag` is not in `DIGEST_TAGS`, or `jcs_dumps` refuses `value`.
        TypeError: if `jcs_dumps` refuses `value`.
    """
    if tag not in DIGEST_TAGS:
        raise ValueError(f"unregistered digest tag {tag!r}; registered: {sorted(DIGEST_TAGS)}")
    return hashlib.sha256(tag.encode("ascii") + b"\x00" + jcs_dumps(value)).hexdigest()


def require_depth(depth: int) -> None:
    """Refuse a container at `depth`, counted from its line's root (the root is at depth 1).

    Raises:
        ValueError: if `depth` exceeds `MAX_DEPTH`.
    """
    if depth > MAX_DEPTH:
        raise ValueError(f"nesting deeper than {MAX_DEPTH} containers")


@contextmanager
def entered(container: object, active: set[int], base: int = 0) -> Iterator[None]:
    """Mark `container` as being encoded for the duration of the block, on the path held in `active`.

    Every encoder that recurses into JSON containers enters each one through this, so a cycle or
    excess nesting raises the same error wherever values are encoded; a child shared by two parents
    is not a cycle. `base` is the number of containers enclosing the encoded value's root, so the
    container's depth is `base + len(active) + 1`.

    Raises:
        ValueError: if `container` is already on the path, that is, it contains itself, or if its
            depth exceeds `MAX_DEPTH` (`require_depth`).
    """
    if id(container) in active:
        raise ValueError(f"a {type(container).__name__} contains itself; JSON has no cycles")
    require_depth(base + len(active) + 1)
    active.add(id(container))
    try:
        yield
    finally:
        active.discard(id(container))


def _write(obj: object, out: Callable[[str], object], active: set[int]) -> None:
    t = type(obj)
    if t is list or t is tuple or t is dict:
        with entered(obj, active):
            if t is dict:
                _write_object(cast("dict[object, object]", obj), out, active)
            else:
                out("[")
                for i, item in enumerate(cast("list[object] | tuple[object, ...]", obj)):
                    if i:
                        out(",")
                    _write(item, out, active)
                out("]")
        return
    scalar = canonical_scalar(obj)
    if scalar is None:
        out("null")
    elif scalar is True:
        out("true")
    elif scalar is False:
        out("false")
    elif type(scalar) is int:
        out(_int_text(scalar))
    elif type(scalar) is float:
        out(_float_text(scalar))
    else:
        out(_string_text(scalar))  # type: ignore[arg-type]


def _write_object(obj: dict[object, object], out: Callable[[str], object], active: set[int]) -> None:
    members: dict[str, object] = {}
    for key, value in obj.items():
        if type(key) is not str:
            raise TypeError(f"JSON object key is not a str: {type(key).__qualname__}")
        nfc = unicodedata.normalize("NFC", key)
        if nfc in members:
            raise ValueError(f"duplicate JSON object key after NFC: {nfc!r}")
        members[nfc] = value
    out("{")
    for i, key in enumerate(sorted(members, key=utf16_sort_key)):
        if i:
            out(",")
        out(_string_text(key))
        out(":")
        _write(members[key], out, active)
    out("}")


def utf16_sort_key(key: str) -> bytes:
    """Return the sort key that orders strings by UTF-16 code units, as JCS orders object keys."""
    return key.encode("utf-16-be", "surrogatepass")


def _int_text(value: int) -> str:
    if abs(value) > _MAX_SAFE_INT:
        raise ValueError(f"integer outside the I-JSON range of +/-(2**53 - 1): {value}")
    return str(value)


def _string_text(value: str) -> str:
    nfc = unicodedata.normalize("NFC", value)
    nfc.encode("utf-8")
    return json.dumps(nfc, ensure_ascii=False)


def _float_text(value: float) -> str:
    """Format a finite float as ECMAScript Number.prototype.toString does."""
    if not math.isfinite(value):
        raise ValueError(f"non-finite number is not JSON: {value}")
    if value == 0.0:
        return "0"
    sign = "-" if value < 0 else ""
    mantissa, _, exp_text = repr(abs(value)).partition("e")
    int_part, _, frac_part = mantissa.partition(".")
    digits = int_part + frac_part
    point = len(int_part) + int(exp_text or 0)
    stripped = digits.lstrip("0")
    point -= len(digits) - len(stripped)
    digits = stripped.rstrip("0")
    k = len(digits)
    if k <= point <= 21:
        return sign + digits + "0" * (point - k)
    if 0 < point <= 21:
        return sign + digits[:point] + "." + digits[point:]
    if -6 < point <= 0:
        return sign + "0." + "0" * -point + digits
    exponent = point - 1
    exp_sign = "+" if exponent >= 0 else "-"
    head = digits[0] + ("." + digits[1:] if k > 1 else "")
    return f"{sign}{head}e{exp_sign}{abs(exponent)}"


def canonical_npy_bytes(a: numpy.ndarray) -> bytes:
    """Serialise an array as canonical `.npy`: version 1.0, C order, little-endian.

    The header is numpy's version 1.0 dict, space-padded so the data starts on a 64-byte boundary.

    Raises:
        TypeError: if `a` is not exactly a `numpy.ndarray` (masked arrays and other subclasses).
        ValueError: on a dtype whose kind is not one of `b i u f U`, or on `longdouble`, whose
            itemsize and byte layout differ between platforms.
    """
    if type(a) is not numpy.ndarray:
        raise TypeError(f"not a plain numpy.ndarray: {type(a).__module__}.{type(a).__qualname__}")
    if a.dtype.kind not in _CANONICAL_KINDS or a.dtype.fields is not None:
        raise ValueError(f"dtype has no canonical form: {a.dtype}")
    if a.dtype.char in _PLATFORM_DTYPE_CHARS:
        raise ValueError(f"platform-specific dtype has no canonical form: {a.dtype.name}")
    dtype = a.dtype.newbyteorder("<")
    data = a.astype(dtype, order="C", copy=False)
    shape = tuple(int(n) for n in data.shape)
    header = f"{{'descr': {dtype.str!r}, 'fortran_order': False, 'shape': {shape!r}, }}"
    padding = -(len(_NPY_MAGIC) + 2 + len(header) + 1) % _NPY_ALIGN
    header_bytes = (header + " " * padding + "\n").encode("latin1")
    return _NPY_MAGIC + struct.pack("<H", len(header_bytes)) + header_bytes + data.tobytes(order="C")
