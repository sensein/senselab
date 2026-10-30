"""Canonical byte forms of JSON values and numpy arrays (ADR-006).

`jcs_dumps` writes RFC 8785 (JCS) JSON with strings in Unicode NFC; `canonical_npy_bytes` writes
`.npy` version 1.0, C order, little-endian.
"""

import json
import math
import struct
import unicodedata
from typing import Callable

import numpy

_MAX_SAFE_INT = 2**53
_NPY_MAGIC = b"\x93NUMPY\x01\x00"
_NPY_ALIGN = 64


def jcs_dumps(obj: object) -> bytes:
    """Serialise a JSON value as RFC 8785 canonical UTF-8 bytes.

    Accepts None, bool, int, float, str, lists, tuples, dicts with str keys and numpy scalars
    (converted with `.item()`). Strings and keys are NFC-normalised; keys sort by UTF-16 code units.

    Raises:
        ValueError: on a NaN or infinite float, an integer of magnitude above 2**53, a string that
            is not valid Unicode, or two keys of one object that are equal after NFC.
        TypeError: on any other type, including non-string keys and numpy arrays.
    """
    parts: list[str] = []
    _write(obj, parts.append)
    return "".join(parts).encode("utf-8")


def _write(obj: object, out: Callable[[str], object]) -> None:
    if isinstance(obj, numpy.generic):
        obj = obj.item()
    if obj is None:
        out("null")
    elif obj is True:
        out("true")
    elif obj is False:
        out("false")
    elif isinstance(obj, int):
        out(_int_text(obj))
    elif isinstance(obj, float):
        out(_float_text(obj))
    elif isinstance(obj, str):
        out(_string_text(obj))
    elif isinstance(obj, (list, tuple)):
        out("[")
        for i, item in enumerate(obj):
            if i:
                out(",")
            _write(item, out)
        out("]")
    elif isinstance(obj, dict):
        _write_object(obj, out)
    else:
        raise TypeError(f"not a JSON value: {type(obj).__name__}")


def _write_object(obj: dict[object, object], out: Callable[[str], object]) -> None:
    members: dict[str, object] = {}
    for key, value in obj.items():
        if not isinstance(key, str):
            raise TypeError(f"JSON object key is not a string: {type(key).__name__}")
        nfc = unicodedata.normalize("NFC", key)
        if nfc in members:
            raise ValueError(f"duplicate JSON object key after NFC: {nfc!r}")
        members[nfc] = value
    out("{")
    for i, key in enumerate(sorted(members, key=_utf16_sort_key)):
        if i:
            out(",")
        out(_string_text(key))
        out(":")
        _write(members[key], out)
    out("}")


def _utf16_sort_key(key: str) -> bytes:
    return key.encode("utf-16-be", "surrogatepass")


def _int_text(value: int) -> str:
    if abs(value) > _MAX_SAFE_INT:
        raise ValueError(f"integer outside the I-JSON range of +/-2**53: {value}")
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
        ValueError: on an object or structured dtype.
    """
    if a.dtype.hasobject or a.dtype.fields is not None:
        raise ValueError(f"object and structured dtypes have no canonical form: {a.dtype}")
    dtype = a.dtype.newbyteorder("<")
    data = a.astype(dtype, order="C", copy=False)
    shape = tuple(int(n) for n in data.shape)
    header = f"{{'descr': {dtype.str!r}, 'fortran_order': False, 'shape': {shape!r}, }}"
    padding = -(len(_NPY_MAGIC) + 2 + len(header) + 1) % _NPY_ALIGN
    header_bytes = (header + " " * padding + "\n").encode("latin1")
    return _NPY_MAGIC + struct.pack("<H", len(header_bytes)) + header_bytes + data.tobytes(order="C")
