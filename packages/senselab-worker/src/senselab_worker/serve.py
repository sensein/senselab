"""The worker side of the wire protocol: handshake, request loop and per-item result lines."""

import os
import platform
import time
from pathlib import Path
from typing import Any, BinaryIO, Callable, Final

import numpy

from senselab_worker import WIRE_VERSION, runtime_identity
from senselab_worker.codec import (
    MAX_LINE_BYTES,
    LineTooLarge,
    dump_line,
    from_wire,
    parse_line,
    to_wire,
    validate_message,
)
from senselab_worker.device_peak import Meter, torch_meter
from senselab_worker.faults import (
    PASS_THROUGH,
    ArrayIntegrityError,
    ProtocolError,
    ResponseTooLarge,
    ValueTooLarge,
    error_record,
)
from senselab_worker.model_cache import ModelCache, ModelKey

Load = Callable[[dict[str, Any]], object]
Handle = Callable[[object, list[dict[str, Any]], dict[str, Any]], list[dict[str, Any]]]

EXIT_FATAL = 1
UNBOUNDED_BYTES: Final = 2**63 - 1  # invariant: the largest signed 64-bit size, read as no bound
"""The budget of a worker process's model cache: unbounded."""


HOST_KEYS = ("dtype",)
"""Keys of `init` the host sends for the launcher alone; a backend's `load` never sees them."""


def backend_init(init: dict[str, Any]) -> dict[str, Any]:
    """Return `init` without the launcher's keys (`HOST_KEYS`), as a backend's `load` takes it."""
    return {k: v for k, v in init.items() if k not in HOST_KEYS}


def model_key(backend: str, init: dict[str, Any]) -> ModelKey | None:
    """Return the `ModelCache` key `init` names for `backend`: its model's identity, device and dtype; None without.

    Raises:
        ProtocolError: `init` names a model without a string `identity`, `device` and `dtype`.
    """
    if "model" not in init:
        return None
    model, device, dtype = init["model"], init.get("device"), init.get("dtype")
    identity = model.get("identity") if isinstance(model, dict) else None
    if not (isinstance(identity, str) and isinstance(device, str) and isinstance(dtype, str)):
        raise ProtocolError(
            "an init naming a model names its identity, device and dtype as strings; "
            f"got model {type(model).__name__}, device {type(device).__name__}, dtype {type(dtype).__name__}"
        )
    return (backend, identity, device, dtype)


def cached_load(load: Load, backend: str, cache: ModelCache) -> Load:
    """Return `load` given `backend_init(init)`, routed through `cache` under `model_key(backend, init)`.

    An init naming no model loads directly. The returned load raises `ProtocolError` for a partial or malformed
    model, device or dtype, and `ValueError` when the model identity is not a commit or weights digest.
    """

    def through(init: dict[str, Any]) -> object:
        key = model_key(backend, init)
        given = backend_init(init)
        if key is None:
            return load(given)
        return cache.get_or_load(key, lambda: load(given), lambda _: 0).state

    return through


def serve(load: Load, handle: Handle, rx: BinaryIO, tx: BinaryIO, meter: Meter[Any] = torch_meter) -> int:
    """Serve the protocol, reading from `rx` and writing to `tx`; return the process exit code.

    `load(init)` is called once and returns the state. `handle(state, items, params)` gets the
    decoded items (`{"id", "inputs"}`) and returns one `{"outputs": dict}` or `{"error": exception}`
    per item, in order. Each item's result is written as its own line, then `done`, which carries the request's
    device allocation peak and what stays allocated after it (`device_peak_bytes`, `device_allocated_bytes`) when
    `meter` measures them (CUDA in use), the requests the state answered before it (`since_load`) and, on the first
    one, the seconds the load took (`load_s`). Nothing an item
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
    _write(tx, dump_line({"hello": hello}, "hello"))
    try:
        msg = _receive(rx)
        if msg is None or _is_stop(msg):
            return 0
        array_dir, init = _init(msg)
        start = time.perf_counter()
        state = load(init)
        loaded_s = time.perf_counter() - start
        _write(tx, dump_line({"ready": {"load_s": loaded_s}}, "ready"))
        answered = 0
        while (msg := _receive(rx)) is not None and not _is_stop(msg):
            report = {"since_load": answered} | ({"load_s": loaded_s} if not answered else {})
            _respond(handle, state, validate_message(msg, "request"), array_dir, tx, (meter, report))
            answered += 1
    except Exception as exc:
        _write(tx, dump_line({"error": error_record(exc), "fatal": True}, "fatal"))
        return EXIT_FATAL
    return 0


def _write(tx: BinaryIO, line: bytes) -> None:
    tx.write(line)
    tx.flush()


def _receive(rx: BinaryIO) -> dict[str, Any] | None:
    line = rx.readline(MAX_LINE_BYTES + 1)
    return parse_line(line) if line else None


def _is_stop(msg: dict[str, Any]) -> bool:
    """Whether `msg` is a stop; a `stop` key with any value but `true` is a `ProtocolError`."""
    return "stop" in msg and validate_message(msg, "stop") is msg


def _init(msg: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    init = validate_message(msg, "init")["init"]
    array_dir = Path(init["array_dir"])
    return array_dir, from_wire(init, array_dir=array_dir)


def _respond(  # noqa: PLR0913 - the request and where its replies go, then how it is measured and reported
    handle: Handle,
    state: object,
    msg: dict[str, Any],
    array_dir: Path,
    tx: BinaryIO,
    measured: tuple[Meter[Any], dict[str, Any]],
) -> None:
    meter, report = measured
    items: list[dict[str, Any]] = msg["items"]
    entries: list[dict[str, Any]] = [{} for _ in items]
    shared: dict[str, numpy.ndarray] = {}
    device: tuple[int, int] | None = None
    try:
        params = from_wire(msg["params"], array_dir=array_dir, shared=shared)
    except ArrayIntegrityError as exc:
        entries = [{"error": exc} for _ in items]
    else:
        decoded: list[tuple[int, dict[str, Any]]] = []
        for i, item in enumerate(items):
            try:
                inputs = from_wire(item["inputs"], array_dir=array_dir, shared=shared)
                decoded.append((i, {"id": item["id"], "inputs": inputs}))
            except ArrayIntegrityError as exc:
                entries[i] = {"error": exc}
        if decoded:
            results, device = meter(lambda: call_handle(handle, state, [item for _, item in decoded], params))
            for (i, _), entry in zip(decoded, results):
                entries[i] = entry
    for item, entry in zip(items, entries):
        _write(tx, _result(msg["id"], item["id"], entry, array_dir))
    done: dict[str, Any] = {"id": msg["id"], "done": len(items), **report}
    if device is not None:
        done |= {"device_peak_bytes": device[0], "device_allocated_bytes": device[1]}
    _write(tx, dump_line(done, "done"))


def call_handle(
    handle: Handle, state: object, items: list[dict[str, Any]], params: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return `handle(state, items, params)` as one `{"outputs": dict}` or `{"error": exception}` per item; total.

    Whatever `handle` raises becomes every item's error; a wrong-length result or a malformed entry becomes a
    `ProtocolError`. `KeyboardInterrupt` and `SystemExit` propagate. Workers and the host's in-process path both
    call it.
    """
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
