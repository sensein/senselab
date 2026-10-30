"""Wire framing and array references of the worker protocol.

A message is one JSON object per newline-terminated UTF-8 line. Every numpy array travels as
`{"__array__": {"path", "sha256", "dtype", "shape"}}`, naming a canonical `.npy` file in the
array directory.
"""

import io
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy

from senselab_worker.canonical import canonical_npy_bytes
from senselab_worker.faults import ArrayIntegrityError, ProtocolError
from senselab_worker.hashing import sha256_bytes

MAX_LINE_BYTES = 16 * 2**20
ARRAY_KEY = "__array__"
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


def encode(msg: dict[str, Any], *, array_dir: Path) -> bytes:
    """Return `msg` as one wire line, writing its arrays into `array_dir`.

    Raises:
        ValueError: on a value with no wire form (see `to_wire`).
        TypeError: on a value of no JSON type.
        ProtocolError: if the line exceeds `MAX_LINE_BYTES`.
    """
    return dump_line(to_wire(msg, array_dir=array_dir))


def decode(line: bytes, *, array_dir: Path) -> dict[str, Any]:
    """Return the message on a wire line, with its array references loaded.

    Raises:
        ProtocolError: if the line is not a JSON object within `MAX_LINE_BYTES`.
        ArrayIntegrityError: if an array reference fails its checks.
    """
    msg: dict[str, Any] = from_wire(parse_line(line), array_dir=array_dir)
    return msg


def dump_line(msg: dict[str, Any]) -> bytes:
    """Return a message already in wire form as one line.

    Raises:
        ProtocolError: if the line, newline included, exceeds `MAX_LINE_BYTES`.
    """
    line = json.dumps(msg, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    if len(line) > MAX_LINE_BYTES:
        raise ProtocolError(f"wire line of {len(line)} bytes exceeds {MAX_LINE_BYTES}; pass large data as arrays")
    return line


def _refuse_constant(name: str) -> object:
    raise ValueError(f"{name} is not JSON")


def parse_line(line: bytes) -> dict[str, Any]:
    """Parse one wire line into a dict, leaving array references as they are.

    Raises:
        ProtocolError: if the line exceeds `MAX_LINE_BYTES`, is not UTF-8 JSON, holds `NaN` or
            `Infinity`, or is not an object.
    """
    if len(line) > MAX_LINE_BYTES:
        raise ProtocolError(f"wire line exceeds {MAX_LINE_BYTES} bytes")
    try:
        msg = json.loads(line.decode("utf-8"), parse_constant=_refuse_constant)
    except ValueError as exc:
        raise ProtocolError(f"wire line is not JSON: {exc}") from exc
    if not isinstance(msg, dict):
        raise ProtocolError(f"wire line is a JSON {type(msg).__name__}, not an object")
    return msg


def to_wire(value: object, *, array_dir: Path) -> Any:  # noqa: ANN401
    """Return `value` as plain JSON data, writing each numpy array into `array_dir` as a reference.

    Numpy scalars become their `.item()`; tuples become lists.

    Raises:
        ValueError: on a non-finite float, an object or structured array, a string that is not
            valid Unicode, or a dict whose only key is `__array__`.
        TypeError: on a value of no JSON type, or a non-string key.
    """
    if isinstance(value, numpy.ndarray):
        return {ARRAY_KEY: _write_array(value, Path(array_dir))}
    if isinstance(value, numpy.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite number is not JSON: {value}; pass it inside an array")
        return value
    if isinstance(value, str):
        value.encode("utf-8")
        return value
    if isinstance(value, (list, tuple)):
        return [to_wire(v, array_dir=array_dir) for v in value]
    if isinstance(value, dict):
        if list(value) == [ARRAY_KEY]:
            raise ValueError(f"a dict whose only key is {ARRAY_KEY!r} is reserved for array references")
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object key is not a string: {type(key).__name__}")
            key.encode("utf-8")
            out[key] = to_wire(item, array_dir=array_dir)
        return out
    raise TypeError(f"not a wire value: {type(value).__name__}")


def _write_array(a: numpy.ndarray, array_dir: Path) -> dict[str, Any]:
    data = canonical_npy_bytes(a)
    sha = sha256_bytes(data)
    path = array_dir.absolute() / f"{sha}.npy"
    fd, tmp = tempfile.mkstemp(dir=array_dir, prefix=".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
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


def from_wire(value: object, *, array_dir: Path) -> Any:  # noqa: ANN401
    """Return wire data with each array reference replaced by its array, loaded into memory.

    Raises:
        ArrayIntegrityError: if a reference is malformed, points outside `array_dir`, or its file
            is missing, unreadable, or differs from its sha256, dtype or shape.
    """
    root = Path(array_dir).resolve()
    return _map_refs(value, lambda ref: _read_array(ref, root))


def _read_array(ref: object, root: Path) -> numpy.ndarray:
    try:
        if not isinstance(ref, dict):
            raise TypeError(f"array reference is a {type(ref).__name__}, not an object")
        sha = ref["sha256"]
        if not (isinstance(sha, str) and _SHA256_HEX.fullmatch(sha)):
            raise ValueError(f"sha256 is not 64 lowercase hex digits: {sha!r}")
        if Path(ref["path"]).resolve() != root / f"{sha}.npy":
            raise ValueError(f"array path is not {sha}.npy in {root}: {ref['path']}")
        data = (root / f"{sha}.npy").read_bytes()
        if sha256_bytes(data) != sha:
            raise ValueError(f"array file bytes do not match sha256 {sha}")
        a: numpy.ndarray = numpy.load(io.BytesIO(data), allow_pickle=False)
        if a.dtype.str != ref["dtype"] or list(a.shape) != ref["shape"]:
            raise ValueError(f"array is {a.dtype.str} {list(a.shape)}, reference says {ref['dtype']} {ref['shape']}")
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise ArrayIntegrityError(f"bad array reference: {exc}") from exc
    return a


def array_paths(value: object) -> set[Path]:
    """Return the file paths named by the array references in wire data."""
    paths: set[Path] = set()

    def collect(ref: object) -> None:
        if isinstance(ref, dict) and isinstance(ref.get("path"), str):
            paths.add(Path(ref["path"]))

    _map_refs(value, collect)
    return paths


def check_request(msg: dict[str, Any]) -> None:
    """Check the shape of a request: a str `id`, `items` with unique str ids and dict inputs, dict[str, Any] `params`.

    Raises:
        ValueError: naming the first part that is malformed.
    """
    items = msg.get("items")
    if not isinstance(msg.get("id"), str):
        raise ValueError("request id must be a string")
    if not isinstance(msg.get("params"), dict):
        raise ValueError("request params must be an object")
    if not isinstance(items, list):
        raise ValueError("request items must be a list")
    ids = []
    for item in items:
        if not (isinstance(item, dict) and isinstance(item.get("id"), str) and isinstance(item.get("inputs"), dict)):
            raise ValueError("each item must be an object with a string id and object inputs")
        ids.append(item["id"])
    if len(set(ids)) != len(ids):
        raise ValueError("item ids must be unique within a request")
