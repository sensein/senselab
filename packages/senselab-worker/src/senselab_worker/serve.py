"""The worker side of the wire protocol: handshake, request loop and per-item fault reporting."""

import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, BinaryIO, Callable

import numpy

from senselab_worker import WIRE_VERSION, runtime_identity
from senselab_worker.codec import (
    MAX_LINE_BYTES,
    check_request,
    dump_line,
    from_wire,
    parse_line,
    to_wire,
)
from senselab_worker.faults import ArrayIntegrityError, ProtocolError, error_record

Load = Callable[[dict[str, Any]], object]
Handle = Callable[[object, list[dict[str, Any]], dict[str, Any]], list[dict[str, Any]]]

EXIT_FATAL = 1


def serve(load: Load, handle: Handle) -> None:
    """Serve the protocol on this process's stdin and stdout until stop or EOF.

    `load(init)` is called once and returns the state. `handle(state, items, params)` gets the
    decoded items (`{"id", "inputs"}`) and returns one `{"outputs": dict[str, Any]}` or
    `{"error": exception}` per item, in order. Before the hello, fd 1 and `sys.stdout` are pointed
    at stderr, so only protocol lines reach the host; import model frameworks inside `load`.

    Raises:
        SystemExit: with a non-zero code after reporting a fatal error.
    """
    sys.stdout.flush()
    protocol = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    code = serve_streams(load, handle, sys.stdin.buffer, protocol)
    if code:
        raise SystemExit(code)


def serve_streams(load: Load, handle: Handle, rx: BinaryIO, tx: BinaryIO) -> int:
    """Serve the protocol reading from `rx` and writing to `tx`; return the process exit code."""
    hello = {
        "wire": WIRE_VERSION,
        "runtime": runtime_identity(),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pid": os.getpid(),
    }
    _send(tx, {"hello": hello})
    try:
        msg = _receive(rx)
        if msg is None or "stop" in msg:
            return 0
        array_dir, init = _init(msg)
        start = time.perf_counter()
        state = load(init)
        _send(tx, {"ready": {"load_s": time.perf_counter() - start}})
        while (msg := _receive(rx)) is not None and "stop" not in msg:
            _send(tx, _respond(handle, state, msg, array_dir))
    except Exception as exc:
        _send(tx, {"error": error_record(exc), "fatal": True})
        return EXIT_FATAL
    return 0


def _send(tx: BinaryIO, msg: dict[str, Any]) -> None:
    tx.write(dump_line(msg))
    tx.flush()


def _receive(rx: BinaryIO) -> dict[str, Any] | None:
    line = rx.readline(MAX_LINE_BYTES + 1)
    return parse_line(line) if line else None


def _init(msg: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    init = msg.get("init")
    if not (isinstance(init, dict) and isinstance(init.get("array_dir"), str)):
        raise ProtocolError("expected an init message with a string array_dir")
    array_dir = Path(init["array_dir"])
    return array_dir, from_wire(init, array_dir=array_dir)


def _respond(handle: Handle, state: object, msg: dict[str, Any], array_dir: Path) -> dict[str, Any]:
    try:
        check_request(msg)
    except ValueError as exc:
        raise ProtocolError(f"not a request: {exc}") from exc
    items: list[dict[str, Any]] = msg["items"]
    entries: list[dict[str, Any]] = [{} for _ in items]
    try:
        params = from_wire(msg["params"], array_dir=array_dir)
    except ArrayIntegrityError as exc:
        entries = [{"error": exc} for _ in items]
    else:
        decoded: list[tuple[int, dict[str, Any]]] = []
        for i, item in enumerate(items):
            try:
                decoded.append((i, {"id": item["id"], "inputs": from_wire(item["inputs"], array_dir=array_dir)}))
            except ArrayIntegrityError as exc:
                entries[i] = {"error": exc}
        if decoded:
            for (i, _), entry in zip(decoded, _call(handle, state, [item for _, item in decoded], params)):
                entries[i] = entry
    return {"id": msg["id"], "results": [_result(item["id"], e, array_dir) for item, e in zip(items, entries)]}


def _call(handle: Handle, state: object, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        entries = handle(state, items, params)
    except Exception as exc:
        return [{"error": exc} for _ in items]
    if not (isinstance(entries, list) and len(entries) == len(items) and all(map(_well_formed, entries))):
        error = ProtocolError("handle must return one {'outputs': dict} or {'error': exception} per item, in order")
        return [{"error": error} for _ in items]
    return entries


def _well_formed(entry: object) -> bool:
    if not isinstance(entry, dict) or len(entry.keys() & {"outputs", "error"}) != 1:
        return False
    return isinstance(entry["outputs"], dict) if "outputs" in entry else isinstance(entry["error"], BaseException)


def _result(item_id: str, entry: dict[str, Any], array_dir: Path) -> dict[str, Any]:
    if "error" in entry:
        return {"id": item_id, "error": error_record(entry["error"])}
    try:
        return {"id": item_id, "outputs": to_wire(entry["outputs"], array_dir=array_dir)}
    except Exception as exc:
        return {"id": item_id, "error": error_record(exc)}
