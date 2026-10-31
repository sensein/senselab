"""Canonical byte forms of JSON values and numpy arrays (ADR-006).

`jcs_dumps` writes RFC 8785 (JCS) JSON with strings in their canonical form (`canonical_text`);
`canonical_npy_bytes` writes `.npy` version 1.0, C order, little-endian, one byte form per value;
`Tokeniser` writes a value outside
JSON as JSON with a one-key token object per family. Every encoder accepts an exact-type whitelist, refuses everything
else, and walks containers through one `Walk`.
"""

import json
import math
import re
import struct
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal
from enum import Enum
from importlib import resources
from pathlib import PurePath
from typing import Any, Callable, TypeVar, cast

import numpy

MAX_SAFE_INT = 2**53 - 1
"""The I-JSON integer bound: an integer is JSON only within +/-MAX_SAFE_INT."""
_PLATFORM_DTYPE_CHARS = "gG"
_CANONICAL_KINDS = "biufU"
_PY_SCALARS = (type(None), bool, int, float, str)
_NPY_MAGIC = b"\x93NUMPY\x01\x00"
_NPY_ALIGN = 64
MAX_DEPTH = 64
"""The most containers on any path from a value's root to a leaf, the root included."""
MAX_CANONICAL_CHARS = 64 * 2**20
"""The most characters one `Walk` may write or replay: the work bound of encoding a value that shares children."""

Scalar = None | bool | int | float | str
T = TypeVar("T")

UNICODE_VERSION = "13.0.0"
"""The Unicode version of the database of Python 3.10: text assigned in it is normalised, other text is not."""
UNICODE_TABLE = "data/unicode-assigned.txt"
"""The table of its assigned code points, inside the package; `scripts/unicode_table.py` generates it."""


def _unassigned_pattern() -> "re.Pattern[str]":
    text = resources.files("senselab_worker").joinpath(UNICODE_TABLE).read_text(encoding="ascii")
    header: dict[str, str] = {}
    gaps: list[str] = []
    following = 0
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        first, *rest = line.split(" ")
        if not all(c in "0123456789ABCDEF" for c in first):
            header[first] = " ".join(rest)
            continue
        start, end = int(first, 16), int(rest[0], 16)
        if start > following:
            gaps.append(f"\\U{following:08x}-\\U{start - 1:08x}")
        following = end + 1
    if header.get("unicode_version") != UNICODE_VERSION:
        raise RuntimeError(f"{UNICODE_TABLE} is for Unicode {header.get('unicode_version')}, not {UNICODE_VERSION}")
    if following <= 0x10FFFF:
        gaps.append(f"\\U{following:08x}-\\U0010ffff")
    return re.compile("[" + "".join(gaps) + "]")


_UNASSIGNED = _unassigned_pattern()


_SURROGATE = re.compile("[\\ud800-\\udfff]")


class InvalidText(ValueError):
    """Text holding a lone surrogate, which is not Unicode and has no UTF-8 form."""


def canonical_text(text: str) -> str:
    """Return `text`'s canonical form, the same on every supported Python.

    Text whose code points are all assigned in `UNICODE_VERSION` is put in NFC. Text holding any
    other code point is returned unchanged: it is never normalised, and it is told apart from every
    normalised text by that code point, which no NFC output over assigned code points holds.

    Raises:
        InvalidText: if `text` holds a lone surrogate.
    """
    if text.isascii():
        return text
    found = _SURROGATE.search(text)
    if found is not None:
        raise InvalidText(f"U+{ord(found.group()):04X} at index {found.start()} is a lone surrogate")
    if _UNASSIGNED.search(text) is not None:
        return text
    return unicodedata.normalize("NFC", text)


def is_portable(text: str) -> bool:
    """Whether every code point of `text` is assigned in `UNICODE_VERSION`, so `canonical_text` normalises it."""
    return text.isascii() or _UNASSIGNED.search(text) is None


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


TOKEN_PREFIX = "$"
"""A one-key object whose key starts with it is a token for a value outside JSON; held keys get one more."""
INT_TOKEN = "$int"
"""The one key of the token that stands for an integer outside the I-JSON range."""
FLOAT_TOKEN = "$f"
"""The one key of the token that stands for a non-finite float."""
ARRAY_TOKEN = "$array"
"""The one key of the token that stands for an array by the sha256 of its canonical `.npy`."""
TOKENS = (FLOAT_TOKEN, INT_TOKEN, "$path", "$enum", "$nd", ARRAY_TOKEN, "$dec", "$bytes", "$set", "$type", "$value")
"""The keys of the tokens canonical forms use, one per family of values outside the JSON domain."""


def escaped_key(key: str) -> str:
    """Return `key` with one more `TOKEN_PREFIX` if it starts with one, so no held object reads as a token."""
    return TOKEN_PREFIX + key if key.startswith(TOKEN_PREFIX) else key


def in_json_domain(scalar: Scalar) -> bool:
    """Whether an exact JSON scalar is one `jcs_dumps` writes: a finite float, an integer within +/-MAX_SAFE_INT."""
    if type(scalar) is float:
        return math.isfinite(scalar)
    if type(scalar) is int:
        return abs(scalar) <= MAX_SAFE_INT
    return True


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


class WorkBudgetExceeded(ValueError):
    """An encoding would write or replay more than its `Walk`'s budget of characters."""


class Walk:
    """One encoder's pass over a value: the shared cycle and depth guard, a work budget and a memo.

    `enter` builds a container's result inside `entered` and keeps it by identity and depth, so a
    container shared by many parents is built once per depth. The memo holds each container it
    keys, so no identity is reused while the pass lives.
    """

    def __init__(self, base: int = 0, budget: int = MAX_CANONICAL_CHARS) -> None:
        """Start a pass whose root is enclosed by `base` containers and may spend `budget` characters."""
        self.active: set[int] = set()
        self.base = base
        self.budget = budget
        self.spent = 0
        self._done: dict[tuple[int, int], tuple[object, object]] = {}

    def charge(self, size: int) -> None:
        """Spend `size` characters.

        Raises:
            WorkBudgetExceeded: once the pass has spent more than its budget.
        """
        self.spent += size
        if self.spent > self.budget:
            raise WorkBudgetExceeded(f"canonical form exceeds {self.budget} characters")

    def enter(self, container: object, build: Callable[[], T]) -> T:
        """Return `build()` for `container`, or the result kept from building it at this depth before.

        Raises:
            ValueError: if `container` contains itself or nests deeper than `MAX_DEPTH` (`entered`).
        """
        key = (id(container), len(self.active))
        held = self._done.get(key)
        if held is not None:
            return cast(T, held[1])
        with entered(container, self.active, self.base):
            result = build()
        self._done[key] = (container, result)
        return result

    def remember(self, value: object, build: Callable[[], T]) -> T:
        """Return `build()` for `value`, kept by identity and depth like `enter`, without entering it."""
        key = (id(value), len(self.active))
        held = self._done.get(key)
        if held is not None:
            return cast(T, held[1])
        result = build()
        self._done[key] = (value, result)
        return result


def jcs_dumps(obj: object) -> bytes:
    """Serialise a JSON value as RFC 8785 canonical UTF-8 bytes.

    Containers are exactly `list`, `tuple` and `dict` with `str` keys; scalars are what
    `canonical_scalar` accepts. Strings and keys are in their canonical form (`canonical_text`); keys
    sort by UTF-16 code units.

    Raises:
        ValueError: on a NaN or infinite float, an integer outside +/-(2**53 - 1), a lone surrogate
            (`InvalidText`), two keys of one object with one canonical form, a container
            that contains itself, nesting deeper than `MAX_DEPTH`, or text longer than
            `MAX_CANONICAL_CHARS` (`WorkBudgetExceeded`).
        TypeError: on any other type, including container subclasses, non-`str` keys and arrays.
    """
    writer = _Writer()
    writer.write(obj)
    return "".join(writer.parts).encode("utf-8")


class _Writer:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.walk = Walk()

    def out(self, text: str) -> None:
        self.walk.charge(len(text))
        self.parts.append(text)

    def write(self, obj: object) -> None:
        t = type(obj)
        if t is list or t is tuple or t is dict:
            before = len(self.parts)
            start, end, size = self.walk.enter(obj, lambda: self._container(obj))
            if start < before:
                self.walk.charge(size)
                self.parts.append("".join(self.parts[start:end]))
            return
        scalar = canonical_scalar(obj)
        if scalar is None:
            self.out("null")
        elif scalar is True:
            self.out("true")
        elif scalar is False:
            self.out("false")
        elif type(scalar) is int:
            self.out(_int_text(scalar))
        elif type(scalar) is float:
            self.out(_float_text(scalar))
        else:
            self.out(_json_string(canonical_text(cast(str, scalar))))

    def _container(self, obj: object) -> tuple[int, int, int]:
        start, spent = len(self.parts), self.walk.spent
        if type(obj) is dict:
            self._object(cast("dict[object, object]", obj))
        else:
            self.out("[")
            for i, item in enumerate(cast("list[object] | tuple[object, ...]", obj)):
                if i:
                    self.out(",")
                self.write(item)
            self.out("]")
        return start, len(self.parts), self.walk.spent - spent

    def _object(self, obj: dict[object, object]) -> None:
        members: dict[str, object] = {}
        for key, value in obj.items():
            if type(key) is not str:
                raise TypeError(f"JSON object key is not a str: {type(key).__qualname__}")
            text = canonical_text(key)
            if text in members:
                raise ValueError(f"duplicate JSON object key after NFC: {text!r}")
            members[text] = value
        self.out("{")
        for i, key in enumerate(sorted(members, key=utf16_sort_key)):
            if i:
                self.out(",")
            self.out(_json_string(key))
            self.out(":")
            self.write(members[key])
        self.out("}")


class NotCanonicalJSON(ValueError):
    """Text `jcs_dumps` cannot have written."""


def jcs_loads(data: bytes | str) -> object:
    """Parse JCS text back to the value `jcs_dumps` wrote it from.

    An integer literal outside +/-(2**53 - 1) can only come from a float, so it parses as that
    float. Only text `jcs_dumps` returns is accepted, so `jcs_dumps(jcs_loads(b)) == b`.

    Raises:
        NotCanonicalJSON: on text that is not UTF-8 JSON, repeats a key within one object, holds a
            `NaN` or `Infinity` literal or a number that overflows to infinity, nests deeper than
            `MAX_DEPTH`, or is not the canonical form of the value it parses to.
    """
    raw = data.encode("utf-8", "surrogatepass") if isinstance(data, str) else bytes(data)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_int=_parse_int,
            parse_float=_parse_float,
            parse_constant=_refuse_constant,
            object_pairs_hook=_unique_object,
        )
    except RecursionError as err:
        raise NotCanonicalJSON(f"JSON nests deeper than {MAX_DEPTH} containers") from err
    except NotCanonicalJSON:
        raise
    except ValueError as err:
        raise NotCanonicalJSON(f"not JSON: {err}") from err
    try:
        canonical = jcs_dumps(value)
    except (ValueError, TypeError) as err:
        raise NotCanonicalJSON(f"JSON jcs_dumps cannot write: {err}") from err
    if canonical != raw:
        raise NotCanonicalJSON("JSON is not in its canonical form")
    return value


def _parse_int(text: str) -> int | float:
    value = int(text)
    return value if abs(value) <= MAX_SAFE_INT else _parse_float(text)


def _parse_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise NotCanonicalJSON(f"number {text[:40]} overflows to infinity")
    return value


def _refuse_constant(text: str) -> object:
    raise NotCanonicalJSON(f"non-finite number is not JSON: {text}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise NotCanonicalJSON(f"repeated key {key[:40]!r} in one object")
        out[key] = value
    return out


def utf16_sort_key(key: str) -> bytes:
    """Return the sort key that orders strings by UTF-16 code units, as JCS orders object keys."""
    return key.encode("utf-16-be", "surrogatepass")


def _int_text(value: int) -> str:
    if not in_json_domain(value):
        raise ValueError(f"integer outside the I-JSON range of +/-(2**53 - 1): {value}")
    return str(value)


def _json_string(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def _float_text(value: float) -> str:
    """Format a finite float as ECMAScript Number.prototype.toString does."""
    if not in_json_domain(value):
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


QUIET_NAN_BITS = {2: 0x7E00, 4: 0x7FC00000, 8: 0x7FF8000000000000}
"""Per float itemsize, the bits of the positive quiet NaN with no payload: every NaN's canonical form."""


def npy_bytes(a: numpy.ndarray) -> bytes:
    """Serialise an array as `.npy` version 1.0, C order, little-endian, keeping every value's bits.

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


def canonical_npy_bytes(a: numpy.ndarray) -> bytes:
    """Serialise an array as canonical `.npy`: `npy_bytes` of its values, each in its one canonical form.

    Every NaN becomes the positive quiet NaN with no payload (`QUIET_NAN_BITS`), `-0.0` becomes
    `0.0`, every nonzero bool byte becomes 1, and every string is in its canonical form
    (`canonical_text`), at a `U` width no narrower than the array's.

    Raises:
        TypeError: if `a` is not exactly a `numpy.ndarray` (masked arrays and other subclasses).
        ValueError: on a dtype `npy_bytes` refuses, or a string holding a lone surrogate (`InvalidText`).
    """
    if type(a) is not numpy.ndarray:
        raise TypeError(f"not a plain numpy.ndarray: {type(a).__module__}.{type(a).__qualname__}")
    if a.dtype.fields is None and a.dtype.char not in _PLATFORM_DTYPE_CHARS:
        if a.dtype.kind == "f":
            a = _canonical_floats(a)
        elif a.dtype.kind == "b":
            a = _canonical_bools(a)
        elif a.dtype.kind == "U":
            a = _canonical_strings(a)
    return npy_bytes(a)


def _canonical_floats(a: numpy.ndarray) -> numpy.ndarray:
    nan = numpy.isnan(a)
    negative_zero = (a == 0) & numpy.signbit(a)
    if not (nan.any() or negative_zero.any()):
        return a
    out = numpy.array(a, dtype=a.dtype.newbyteorder("<"), order="C", copy=True)
    bits = out.view(f"<u{out.dtype.itemsize}")
    bits[nan] = QUIET_NAN_BITS[out.dtype.itemsize]
    bits[negative_zero] = 0
    return out


def _canonical_bools(a: numpy.ndarray) -> numpy.ndarray:
    raw = numpy.array(a, order="C", copy=True).view(numpy.uint8)
    if not (raw > 1).any():
        return a
    return numpy.asarray(raw != 0)


def _canonical_strings(a: numpy.ndarray) -> numpy.ndarray:
    held = [str(x) for x in a.reshape(-1).tolist()]
    if all(x.isascii() for x in held):
        return a
    texts = [canonical_text(x) for x in held]
    if texts == held:
        return a
    width = max(a.dtype.itemsize // 4, *(len(t) for t in texts))
    return numpy.array(texts, dtype=f"<U{width}").reshape(a.shape)


class NoCanonicalForm(TypeError, ValueError):
    """A value outside the families a `Tokeniser` writes.

    Attributes:
        path: Where the value sits, from the root, e.g. `value.evidence[2]`.
        type_name: The offending value's type, `module.qualname`.
    """

    refusal = "has no canonical form"
    """What a value of no family is said to be, after its path and type."""

    def __init__(self, path: str, value: object, reason: str) -> None:
        """Name the path, the value's type and why it has no canonical form."""
        self.path = path
        self.type_name = f"{type(value).__module__}.{type(value).__qualname__}"
        super().__init__(f"{path}: {self.type_name} {reason}")


Recurse = Callable[[object, str], object]
"""Writes a value found inside another, at a path: what `Tokeniser` passes to `array`."""
ArrayToken = Callable[[numpy.ndarray, Recurse, str], object]
"""Writes an array, given the recursion and the array's path."""
Extra = Callable[[object], "tuple[str | None, object] | None"]
"""Maps a value outside the shared families to `(token, payload)`, or returns `None` if it has none."""


def array_token(a: numpy.ndarray, recurse: Recurse, path: str) -> object:
    """Return `{"$array": npy_digest("npy@1", a)}`."""
    from senselab_worker.digest import npy_digest

    return {ARRAY_TOKEN: npy_digest("npy@1", a)}


class Tokeniser:
    """Writes a value as JSON `jcs_dumps` accepts, each value outside the JSON domain as a one-key token.

    The families, dispatched on exact type for the JSON ones:

    - `None`, `bool`, `str`, a finite `float`, an `int` within +/-`MAX_SAFE_INT`, or a numpy scalar
      `canonical_scalar` accepts: itself;
    - a non-finite float: `{"$f": "NaN" | "Infinity" | "-Infinity"}`;
    - an integer outside +/-`MAX_SAFE_INT`: `{"$int": "<decimal>"}`;
    - `dict` with `str` keys: an object, a key starting with `$` gaining one more `$`;
    - `list` and `tuple`: an array;
    - `set` and `frozenset`: `{"$set": [<elements, one per JCS form, sorted by their JCS bytes>]}`;
    - `numpy.ndarray`: what `array` returns for it;
    - an `Enum` member: `{"$enum": ["<module>.<qualname>", <its value>]}`;
    - a `PurePath`: `{"$path": <posix string>}`;
    - a `Decimal`: `{"$dec": <str>}`; `bytes` or `bytearray`: `{"$bytes": <hex>}`;
    - a value `extra` maps to `(token, payload)`: `{token: <payload>}`, entered like a container, or
      the payload itself when `token` is `None`;
    - an instance of exactly `verbatim`, a mapping already in this form: written as it is, its keys
      not escaped again, every mapping in it written as an object, every list or tuple as an array,
      and every scalar in the JSON domain.

    Each call is one `Walk`: containers, arrays and `extra` tokens are entered through the shared
    guard, and one shared by many parents is written once per depth.
    """

    def __init__(
        self,
        *,
        array: ArrayToken = array_token,
        extra: Extra | None = None,
        error: type[NoCanonicalForm] = NoCanonicalForm,
        verbatim: type | None = None,
    ) -> None:
        """Write arrays with `array`, values of no shared family with `extra`, and `verbatim` mappings as they are."""
        self.array = array
        self.extra = extra
        self.error = error
        self.verbatim = verbatim

    def __call__(self, value: object, path: str = "value") -> object:
        """Return `value` in tokens.

        Raises:
            NoCanonicalForm: (`error`) naming the path and type, for a value of no family, a non-`str`
                key, a key holding a lone surrogate, two keys with one canonical form, a container that contains itself,
                nesting deeper than `MAX_DEPTH`, set elements over `MAX_CANONICAL_CHARS`, or an integer with more
                digits than the interpreter writes as text.
        """
        return _Tokens(self, Walk()).value(value, path)


class _Tokens:
    def __init__(self, spec: Tokeniser, walk: Walk) -> None:
        self.spec = spec
        self.walk = walk

    def value(self, value: object, path: str) -> object:
        t = type(value)
        if self.spec.verbatim is not None and t is self.spec.verbatim:
            try:
                return self.held(value, path)
            except NoCanonicalForm:
                raise
            except (ValueError, TypeError) as err:
                raise self.spec.error(path, value, str(err)) from err
        if t is dict or t is list or t is tuple or t is set or t is frozenset or t is numpy.ndarray:
            try:
                return self.walk.enter(value, lambda: self.container(value, path))
            except NoCanonicalForm:
                raise
            except (ValueError, TypeError) as err:
                raise self.spec.error(path, value, str(err)) from err
        if isinstance(value, Enum):
            qualname = f"{t.__module__}.{t.__qualname__}"
            return {"$enum": [qualname, self.value(value.value, f"{path}.value")]}
        if isinstance(value, PurePath):
            return {"$path": value.as_posix()}
        if t is bytes or t is bytearray:
            return {"$bytes": cast("bytes | bytearray", value).hex()}
        if t is Decimal:
            return {"$dec": str(value)}
        if self.spec.extra is not None:
            found = self.spec.extra(value)
            if found is not None:
                return self.extra(value, found[0], found[1], path)
        try:
            scalar = canonical_scalar(value)
        except TypeError as err:
            raise self.spec.error(path, value, self.spec.error.refusal) from err
        if in_json_domain(scalar):
            return scalar
        if type(scalar) is int:
            try:
                return {INT_TOKEN: str(scalar)}
            except ValueError as err:
                raise self.spec.error(path, value, str(err)) from err
        number = cast(float, scalar)
        return {FLOAT_TOKEN: "NaN" if math.isnan(number) else ("Infinity" if number > 0 else "-Infinity")}

    def held(self, value: object, path: str) -> object:
        if isinstance(value, (dict, list, tuple)):
            return self.walk.enter(value, lambda: self.held_container(value, path))
        scalar = canonical_scalar(value)
        if not in_json_domain(scalar):
            raise self.spec.error(path, value, "is a number outside the JSON domain inside a held form")
        return scalar

    def held_container(self, value: "dict[str, object] | list[object] | tuple[object, ...]", path: str) -> object:
        if isinstance(value, dict):
            return {key: self.held(item, f"{path}.{key}") for key, item in value.items()}
        return [self.held(item, f"{path}[{i}]") for i, item in enumerate(value)]

    def extra(self, value: object, token: str | None, payload: object, path: str) -> object:
        try:
            if token is None:
                return self.walk.remember(value, lambda: self.value(payload, path))
            return self.walk.enter(value, lambda: {token: self.value(payload, path)})
        except NoCanonicalForm:
            raise
        except ValueError as err:
            raise self.spec.error(path, value, str(err)) from err

    def container(self, value: Any, path: str) -> object:  # noqa: ANN401
        t = type(value)
        if t is dict:
            out: dict[str, object] = {}
            seen: set[str] = set()
            for key, item in value.items():
                if type(key) is not str:
                    raise self.spec.error(f"{path}[{key!r}]", key, "is not a str object key")
                try:
                    text = canonical_text(key)
                except InvalidText as err:
                    raise self.spec.error(f"{path}[{key!r}]", key, str(err)) from err
                if text in seen:
                    raise self.spec.error(f"{path}.{key}", key, "is a key equal after NFC to another key of its object")
                seen.add(text)
                out[escaped_key(key)] = self.value(item, f"{path}.{key}")
            return out
        if t is list or t is tuple:
            return [self.value(item, f"{path}[{i}]") for i, item in enumerate(value)]
        if t is set or t is frozenset:
            forms: dict[bytes, object] = {}
            for item in value:
                token = self.value(item, f"{path}{{}}")
                form = jcs_dumps(token)
                self.walk.charge(len(form))
                forms.setdefault(form, token)
            return {"$set": [forms[form] for form in sorted(forms)]}
        return self.spec.array(value, self.value, path)
