"""The worker side of the wire protocol: handshake, request loop and per-item result lines."""

import os
import platform
import time
from pathlib import Path
from typing import Any, BinaryIO, Callable

import numpy

from senselab_worker import WIRE_VERSION, runtime_identity
from senselab_worker.codec import (
    MAX_LINE_BYTES,
    Kind,
    LineTooLarge,
    dump_line,
    from_wire,
    parse_line,
    to_wire,
    validate_message,
)
from senselab_worker.faults import (
    PASS_THROUGH,
    ArrayIntegrityError,
    ProtocolError,
    ResponseTooLarge,
    ValueTooLarge,
    error_record,
)

Load = Callable[[dict[str, Any]], object]
Handle = Callable[[object, list[dict[str, Any]], dict[str, Any]], list[dict[str, Any]]]

EXIT_FATAL = 1


def serve(load: Load, handle: Handle, rx: BinaryIO, tx: BinaryIO) -> int:
    """Serve the protocol, reading from `rx` and writing to `tx`; return the process exit code.

    `load(init)` is called once and returns the state. `handle(state, items, params)` gets the
    decoded items (`{"id", "inputs"}`) and returns one `{"outputs": dict}` or `{"error": exception}`
    per item, in order. Each item's result is written as its own line, then `done`. Nothing an item
    or `handle` does ends the loop, except `KeyboardInterrupt` and `SystemExit`. Returns 0 on stop
    or EOF, and `EXIT_FATAL` after reporting a fatal error: a failed load, or a line from the host
    that is not a valid message.
    """
    hello = {
        "wire": WIRE_VERSION,
        "runtime": runtime_identity(),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pid": os.getpid(),
    }
    _send(tx, {"hello": hello}, "hello")
    try:
        msg = _receive(rx)
        if msg is None or _is_stop(msg):
            return 0
        array_dir, init = _init(msg)
        start = time.perf_counter()
        state = load(init)
        _send(tx, {"ready": {"load_s": time.perf_counter() - start}}, "ready")
        while (msg := _receive(rx)) is not None and not _is_stop(msg):
            _respond(handle, state, validate_message(msg, "request"), array_dir, tx)
    except Exception as exc:
        _send(tx, {"error": error_record(exc), "fatal": True}, "fatal")
        return EXIT_FATAL
    return 0


def _send(tx: BinaryIO, msg: dict[str, Any], kind: Kind) -> None:
    _write(tx, dump_line(msg, kind))


def _write(tx: BinaryIO, line: bytes) -> None:
    tx.write(line)
    tx.flush()


def _receive(rx: BinaryIO) -> dict[str, Any] | None:
    line = rx.readline(MAX_LINE_BYTES + 1)
    return parse_line(line) if line else None


def _is_stop(msg: dict[str, Any]) -> bool:
    """Whether `msg` is a stop; a `stop` key with any value but `true` is a `ProtocolError`."""
    if "stop" not in msg:
        return False
    validate_message(msg, "stop")
    return True


def _init(msg: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    init = validate_message(msg, "init")["init"]
    array_dir = Path(init["array_dir"])
    return array_dir, from_wire(init, array_dir=array_dir)


def _respond(handle: Handle, state: object, msg: dict[str, Any], array_dir: Path, tx: BinaryIO) -> None:
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
    for item, entry in zip(items, entries):
        _write(tx, _result(msg["id"], item["id"], entry, array_dir))
    _send(tx, {"id": msg["id"], "done": len(items)}, "done")


def _call(handle: Handle, state: object, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        entries = handle(state, items, params)
    except PASS_THROUGH:
        raise
    except BaseException as exc:  # noqa: BLE001 - whatever handle raises is every item's error
        return [{"error": exc} for _ in items]
    if not (type(entries) is list and len(entries) == len(items)):
        error = ProtocolError("handle must return a list with one entry per item, in order")
        return [{"error": error} for _ in items]
    malformed = ProtocolError("handle's entry for this item is not one {'outputs': dict} or {'error': exception}")
    return [entry if _well_formed(entry) else {"error": malformed} for entry in entries]


def _well_formed(entry: object) -> bool:
    if type(entry) is not dict or len(entry.keys() & {"outputs", "error"}) != 1:
        return False
    return type(entry["outputs"]) is dict if "outputs" in entry else isinstance(entry["error"], BaseException)


def _result(request_id: str, item_id: str, entry: dict[str, Any], array_dir: Path) -> bytes:
    """Return the item's result line; total: a failure to encode or frame it becomes the item's error line."""
    head = {"id": request_id, "item": item_id}
    try:
        if "error" in entry:
            return dump_line({**head, "error": error_record(entry["error"])}, "result")
        return dump_line({**head, "outputs": to_wire(entry["outputs"], array_dir=array_dir, at="outputs")}, "result")
    except PASS_THROUGH:
        raise
    except (LineTooLarge, ValueTooLarge) as exc:
        record = error_record(ResponseTooLarge(f"result for item {item_id!r} exceeds {MAX_LINE_BYTES} bytes: {exc}"))
    except BaseException as exc:  # noqa: BLE001 - an item's encoding failure is that item's error
        record = error_record(exc)
    return dump_line({**head, "error": record}, "result")
