"""Wire framing and array references of the worker protocol.

A message is one JSON object per newline-terminated UTF-8 line. Every numpy array travels as
`{"__array__": {"path", "sha256", "dtype", "shape"}}`, naming an `.npy` file (`npy_bytes`) in the
array directory.
"""

import io
import json
import math
import os
import re
import stat
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Literal

import numpy

from senselab_worker.canonical import MAX_DEPTH, canonical_scalar, entered, npy_bytes, require_depth
from senselab_worker.digest import blob_digest
from senselab_worker.faults import ArrayIntegrityError, FaultClass, ProtocolError, ValueTooLarge, WireValueError
from senselab_worker.ownership import ARRAY_TEMP_NAME, remove_owned

MAX_LINE_BYTES = 16 * 2**20
MAX_ID_CHARS = 1024
Kind = Literal["hello", "init", "ready", "request", "result", "done", "fatal", "stop"]
Slot = Literal["line", "init", "params", "inputs", "outputs"]
ENVELOPE: dict[str, int] = {"line": 0, "init": 1, "params": 1, "inputs": 3, "outputs": 1}
"""Per slot, the containers that enclose a value on its line: `{"init": v}`, `{"params": v}`,
`{"items": [{"inputs": v}]}`, `{"outputs": v}`; `line` is a whole message."""
ARRAY_KEY = "__array__"
ARRAY_REF_HEIGHT = 3
"""Containers in an array reference: the `__array__` object, the reference object, its `shape` list."""
READ_CHUNK_BYTES = 2**20
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


class Deadline:
    """A point in monotonic time by which a sequence of steps must finish; `None` seconds never expires."""

    def __init__(self, seconds: float | None) -> None:
        """Start the clock: the deadline falls `seconds` from now."""
        self._seconds = seconds
        self._end = None if seconds is None else time.monotonic() + max(seconds, 0.0)

    def remaining(self) -> float | None:
        """Return the seconds left, at least 0, or `None` if the deadline never expires."""
        return None if self._end is None else max(self._end - time.monotonic(), 0.0)

    def expired(self) -> bool:
        """Return whether no time is left."""
        return self._end is not None and time.monotonic() >= self._end

    def check(self, what: str) -> None:
        """Raise `TimeoutError` naming `what` if the deadline has passed."""
        if self.expired():
            raise TimeoutError(f"{what} did not finish within {self._seconds} s")


def encode(msg: dict[str, Any], kind: Kind, *, array_dir: Path) -> bytes:
    """Return `msg` as one wire line of `kind`, writing its arrays into `array_dir`.

    Raises:
        WireValueError: on a value with no wire form (see `to_wire`).
        ProtocolError: if the message is not a valid `kind` message (see `validate_message`), or the
            line exceeds `MAX_LINE_BYTES` (`LineTooLarge`).
    """
    return dump_line(to_wire(msg, array_dir=array_dir, at="line"), kind)


def decode(line: bytes, kind: Kind, *, array_dir: Path) -> dict[str, Any]:
    """Return the `kind` message on a wire line, with its array references loaded.

    Raises:
        ProtocolError: if the line is not a valid `kind` message (see `parse_line` and `validate_message`).
        ArrayIntegrityError: if an array reference fails its checks.
    """
    msg: dict[str, Any] = from_wire(validate_message(parse_line(line), kind), array_dir=array_dir)
    return msg


class LineTooLarge(ProtocolError):
    """A wire line, newline included, would exceed `MAX_LINE_BYTES`; `size` is its length in bytes."""

    def __init__(self, size: int) -> None:
        """Name the line's size and the limit."""
        super().__init__(f"wire line of {size} bytes exceeds {MAX_LINE_BYTES}; pass large data as arrays")
        self.size = size


def dump_line(msg: dict[str, Any], kind: Kind) -> bytes:
    """Return a message already in wire form as one line, after checking it is a valid `kind` message.

    Raises:
        ProtocolError: if `validate_message` refuses it, it nests deeper than `MAX_DEPTH`, or it is
            not JSON text.
        LineTooLarge: if the line, newline included, exceeds `MAX_LINE_BYTES`.
    """
    validate_message(msg, kind)
    _check_depth(msg)
    try:
        line = _json_bytes(msg) + b"\n"
    except (ValueError, TypeError, RecursionError) as exc:
        raise ProtocolError(f"{kind} message has no wire form: {type(exc).__name__}: {exc}") from exc
    if len(line) > MAX_LINE_BYTES:
        raise LineTooLarge(len(line))
    return line


def wire_size(value: object) -> int:
    """Return the bytes `value`, already in wire form, takes inside a wire line.

    Raises:
        ValueError, TypeError: if `value` is not JSON data.
    """
    return len(_json_bytes(value))


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _refuse_constant(name: str) -> object:
    raise ValueError(f"{name} is not JSON")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"number {text[:40]} is not finite")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    for key, _ in pairs:
        if key in seen:
            raise ValueError(f"duplicate key {key[:40]!r} in one object")
        seen.add(key)
    return dict(pairs)


def _check_depth(msg: object) -> None:
    try:
        _height(msg, set(), {})
    except ValueError as exc:
        raise ProtocolError(f"wire message: {exc}") from exc


def _height(value: object, active: set[int], heights: dict[int, int]) -> int:
    """Return the containers on the tallest path down from `value`, each distinct container walked once."""
    if isinstance(value, dict):
        children: Any = value.values()
    elif isinstance(value, (list, tuple)):
        children = value
    else:
        return 0
    known = heights.get(id(value))
    if known is not None:
        require_depth(len(active) + known)
        return known
    with entered(value, active):
        height = 1 + max((_height(child, active, heights) for child in children), default=0)
    heights[id(value)] = height
    return height


def parse_line(line: bytes) -> dict[str, Any]:
    """Parse one wire line into a dict, leaving array references as they are.

    Raises:
        ProtocolError: if the line exceeds `MAX_LINE_BYTES`, is not UTF-8 JSON, holds `NaN`,
            `Infinity` or a number that overflows to infinity (`1e999`), repeats a key within one
            object, nests deeper than `MAX_DEPTH` containers, or is not an object.
    """
    if len(line) > MAX_LINE_BYTES:
        raise ProtocolError(f"wire line exceeds {MAX_LINE_BYTES} bytes")
    try:
        msg = json.loads(
            line.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_float=_finite_float,
            parse_constant=_refuse_constant,
        )
    except RecursionError as exc:
        raise ProtocolError(f"wire line nests deeper than {MAX_DEPTH} containers") from exc
    except ValueError as exc:
        raise ProtocolError(f"wire line is not JSON: {exc}") from exc
    if type(msg) is not dict:
        raise ProtocolError(f"wire line is a JSON {type(msg).__name__}, not an object")
    _check_depth(msg)
    return msg


def to_wire(value: object, *, array_dir: Path, at: Slot) -> Any:  # noqa: ANN401
    """Return `value` as plain JSON data, writing each numpy array into `array_dir` as a reference.

    `at` names where the value sits on its line, and so how many containers enclose it
    (`ENVELOPE`): its containers' depths are counted from the line's root, as the reader counts
    them. Containers are exactly `dict` with `str` keys, `list` and `tuple` (which becomes a
    list); scalars are what `canonical_scalar` accepts, so a numpy scalar becomes its `.item()`;
    arrays are exactly `numpy.ndarray`, and a reference is `ARRAY_REF_HEIGHT` containers tall.

    Raises:
        WireValueError: on any other type or a subclass, a non-finite float, a string or key that
            is not valid Unicode, an object or structured array, a dict whose only key is
            `__array__`, a container that contains itself, or a container deeper than `MAX_DEPTH`
            on the line.
        ValueTooLarge: once a lower bound on the value's encoded size exceeds `MAX_LINE_BYTES`.
    """
    encoder = _Encoder(Path(array_dir), ENVELOPE[at])
    try:
        return encoder.encode(value)[0]
    except (WireValueError, ValueTooLarge):
        raise
    except (ValueError, TypeError) as exc:
        raise WireValueError(f"no wire form: {exc}") from exc


class _Encoder:
    """The state of one `to_wire` call."""

    def __init__(self, array_dir: Path, base: int) -> None:
        self.array_dir = array_dir
        self.base = base
        self.active: set[int] = set()
        self.done: dict[int, tuple[Any, int, int]] = {}
        self.spent = 0

    def charge(self, size: int) -> None:
        self.spent += size
        if self.spent > MAX_LINE_BYTES:
            raise ValueTooLarge(f"wire form exceeds {MAX_LINE_BYTES} bytes; pass large data as arrays")

    def encode(self, value: object) -> tuple[Any, int]:
        """Return the wire form of `value` and its height in containers (0 for a scalar)."""
        t = type(value)
        if not (t is dict or t is list or t is tuple or isinstance(value, numpy.ndarray)):
            return self.scalar(value), 0
        found = self.done.get(id(value))
        if found is not None:
            wire, height, size = found
            require_depth(self.base + len(self.active) + height)
            self.charge(size)
            return wire, height
        start = self.spent
        wire, height = self.array(value) if isinstance(value, numpy.ndarray) else self.container(value)
        self.done[id(value)] = (wire, height, self.spent - start)
        return wire, height

    def array(self, a: numpy.ndarray) -> tuple[Any, int]:
        require_depth(self.base + len(self.active) + ARRAY_REF_HEIGHT)
        ref = _write_array(a, self.array_dir)
        self.charge(len(ref["path"]) + len(ref["sha256"]))
        return {ARRAY_KEY: ref}, ARRAY_REF_HEIGHT

    def container(self, value: Any) -> tuple[Any, int]:  # noqa: ANN401
        with entered(value, self.active, self.base):
            self.charge(1 + max(len(value), 1))
            height = 0
            if type(value) is dict:
                if list(value) == [ARRAY_KEY]:
                    raise ValueError(f"a dict whose only key is {ARRAY_KEY!r} is reserved for array references")
                out: Any = {}
                for key, item in value.items():
                    if type(key) is not str:
                        raise TypeError(f"JSON object key is not a str: {type(key).__qualname__}")
                    self.charge(len(key) + 3)
                    key.encode("utf-8")
                    out[key], child = self.encode(item)
                    height = max(height, child)
            else:
                out = []
                for item in value:
                    wire, child = self.encode(item)
                    out.append(wire)
                    height = max(height, child)
            return out, height + 1

    def scalar(self, value: object) -> object:
        scalar = canonical_scalar(value)
        if type(scalar) is str:
            self.charge(len(scalar) + 2)
            scalar.encode("utf-8")
        else:
            self.charge(1)
        if type(scalar) is float and not math.isfinite(scalar):
            raise ValueError(f"non-finite number is not JSON: {scalar}; pass it inside an array")
        return scalar


def _write_array(a: numpy.ndarray, array_dir: Path) -> dict[str, Any]:
    data = npy_bytes(a)
    sha = blob_digest("npy@1", data)
    root = array_dir.absolute()
    path = root / f"{sha}.npy"
    tmp = root / f".{uuid.uuid4().hex}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        remove_owned(root, tmp, pattern=ARRAY_TEMP_NAME)
        raise
    return {"path": str(path), "sha256": sha, "dtype": a.dtype.newbyteorder("<").str, "shape": list(a.shape)}


def _map_refs(value: object, on_ref: Callable[[object], object]) -> Any:  # noqa: ANN401
    if isinstance(value, dict):
        if len(value) == 1 and ARRAY_KEY in value:
            return on_ref(value[ARRAY_KEY])
        return {key: _map_refs(item, on_ref) for key, item in value.items()}
    if isinstance(value, list):
        return [_map_refs(item, on_ref) for item in value]
    return value


def from_wire(value: object, *, array_dir: Path, deadline: Deadline | None = None) -> Any:  # noqa: ANN401
    """Return wire data with each array reference replaced by its array, loaded into memory.

    Each file is opened without following a symlink and without blocking, must be a regular file,
    and is read in chunks of `READ_CHUNK_BYTES`, checking `deadline` between them.

    Raises:
        ArrayIntegrityError: if a reference is malformed, points outside `array_dir`, or its file
            is missing, not a regular file, unreadable, or differs from its sha256, dtype or shape.
        TimeoutError: if `deadline` passes while reading.
    """
    root = Path(array_dir).resolve()
    return _map_refs(value, lambda ref: _read_array(ref, root, deadline or Deadline(None)))


def _read_file(path: Path, deadline: Deadline) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"array file is not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            deadline.check(f"reading array file {path}")
            chunk = os.read(fd, READ_CHUNK_BYTES)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(fd)


def _read_array(ref: object, root: Path, deadline: Deadline) -> numpy.ndarray:
    try:
        if not isinstance(ref, dict):
            raise TypeError(f"array reference is a {type(ref).__name__}, not an object")
        sha = ref["sha256"]
        if not (isinstance(sha, str) and _SHA256_HEX.fullmatch(sha)):
            raise ValueError(f"sha256 is not 64 lowercase hex digits: {sha!r}")
        named = Path(ref["path"])
        if named.name != f"{sha}.npy" or named.parent.resolve() != root:
            raise ValueError(f"array path is not {sha}.npy in {root}: {ref['path']}")
        data = _read_file(root / f"{sha}.npy", deadline)
        if blob_digest("npy@1", data) != sha:
            raise ValueError(f"array file bytes do not match sha256 {sha}")
        a: numpy.ndarray = numpy.load(io.BytesIO(data), allow_pickle=False)
        if a.dtype.str != ref["dtype"] or list(a.shape) != ref["shape"]:
            raise ValueError(f"array is {a.dtype.str} {list(a.shape)}, reference says {ref['dtype']} {ref['shape']}")
    except TimeoutError:
        raise
    except Exception as exc:
        raise ArrayIntegrityError(f"bad array reference: {type(exc).__name__}: {exc}") from exc
    return a


def array_paths(value: object) -> set[Path]:
    """Return the file paths named by the array references in wire data."""
    paths: set[Path] = set()

    def collect(ref: object) -> None:
        if isinstance(ref, dict) and isinstance(ref.get("path"), str):
            paths.add(Path(ref["path"]))

    _map_refs(value, collect)
    return paths


def _fail(kind: str, what: str) -> ProtocolError:
    return ProtocolError(f"not a valid {kind} message: {what}")


def _get(obj: dict[str, Any], key: str, kind: str) -> Any:  # noqa: ANN401
    if key not in obj:
        raise _fail(kind, f"{key!r} is missing")
    return obj[key]


def _object(obj: dict[str, Any], key: str, kind: str) -> dict[str, Any]:
    value = _get(obj, key, kind)
    if type(value) is not dict:
        raise _fail(kind, f"{key!r} is a {type(value).__name__}, not an object")
    return value  # type: ignore[no-any-return]


def _typed(obj: dict[str, Any], key: str, kind: str, types: tuple[type, ...]) -> Any:  # noqa: ANN401
    value = _get(obj, key, kind)
    if type(value) not in types:
        names = " or ".join(t.__name__ for t in types)
        raise _fail(kind, f"{key!r} is a {type(value).__name__}, not {names}")
    return value


def _identifier(obj: dict[str, Any], key: str, kind: str) -> str:
    value: str = _typed(obj, key, kind, (str,))
    if len(value) > MAX_ID_CHARS:
        raise _fail(kind, f"{key!r} is longer than {MAX_ID_CHARS} characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise _fail(kind, f"{key!r} is not valid Unicode") from exc
    return value


def _true(obj: dict[str, Any], key: str, kind: str) -> None:
    if _get(obj, key, kind) is not True:
        raise _fail(kind, f"{key!r} must be exactly true, got {obj[key]!r}")


def _hello(msg: dict[str, Any]) -> None:
    hello = _object(msg, "hello", "hello")
    _typed(hello, "wire", "hello", (int,))
    for key in ("runtime", "python", "numpy"):
        _typed(hello, key, "hello", (str,))
    _typed(hello, "pid", "hello", (int,))


def _init(msg: dict[str, Any]) -> None:
    _typed(_object(msg, "init", "init"), "array_dir", "init", (str,))


def _ready(msg: dict[str, Any]) -> None:
    _typed(_object(msg, "ready", "ready"), "load_s", "ready", (int, float))


def _request(msg: dict[str, Any]) -> None:
    _identifier(msg, "id", "request")
    items = _typed(msg, "items", "request", (list,))
    ids: set[str] = set()
    for item in items:
        if type(item) is not dict:
            raise _fail("request", f"an item is a {type(item).__name__}, not an object")
        item_id = _identifier(item, "id", "request")
        _object(item, "inputs", "request")
        if item_id in ids:
            raise _fail("request", f"item id {item_id!r} is not unique")
        ids.add(item_id)
    _object(msg, "params", "request")


_FAULTS = frozenset(f.value for f in FaultClass)


def _result(msg: dict[str, Any]) -> None:
    _identifier(msg, "id", "result")
    _identifier(msg, "item", "result")
    if ("outputs" in msg) == ("error" in msg):
        raise _fail("result", "it needs exactly one of 'outputs' and 'error'")
    if "outputs" in msg:
        _object(msg, "outputs", "result")
        return
    error = _object(msg, "error", "result")
    for key in ("type", "message", "traceback"):
        _typed(error, key, "result", (str,))
    if _get(error, "fault", "result") not in _FAULTS:
        raise _fail("result", f"unknown fault class {error['fault']!r}")
    _typed(error, "gap", "result", (bool,))


def _done(msg: dict[str, Any]) -> None:
    _identifier(msg, "id", "done")
    if _typed(msg, "done", "done", (int,)) < 0:
        raise _fail("done", "'done' is negative")


def _fatal(msg: dict[str, Any]) -> None:
    _true(msg, "fatal", "fatal")
    _get(msg, "error", "fatal")


def _stop(msg: dict[str, Any]) -> None:
    _true(msg, "stop", "stop")


_VALIDATORS: dict[str, Callable[[dict[str, Any]], None]] = {
    "hello": _hello,
    "init": _init,
    "ready": _ready,
    "request": _request,
    "result": _result,
    "done": _done,
    "fatal": _fatal,
    "stop": _stop,
}


def validate_message(msg: object, kind: Kind) -> dict[str, Any]:
    """Return `msg` if it is a structurally valid `kind` message; the one validator for writer and reader.

    Field types are exact: an integer is `type(v) is int`, so `True` and `1.0` are not integers.
    Ids are strings of valid Unicode of at most `MAX_ID_CHARS` characters. Unknown keys are
    ignored. A `fatal` message needs `"fatal": true` and an `error` of any shape, which the host
    decodes totally; a `result` error must be a complete error record.

    Raises:
        ProtocolError: naming the first part that is malformed.
    """
    if type(msg) is not dict:
        raise _fail(kind, f"it is a {type(msg).__name__}, not an object")
    _VALIDATORS[kind](msg)
    return msg  # type: ignore[return-value]
