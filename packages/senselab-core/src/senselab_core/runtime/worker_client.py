"""The host side of the worker wire protocol: start a worker, send it requests, map its faults."""

import collections
import logging
import os
import queue
import shutil
import subprocess
import threading
import uuid
from pathlib import Path
from types import TracebackType
from typing import IO, Any

from senselab_worker import WIRE_VERSION, runtime_identity
from senselab_worker.codec import (
    MAX_LINE_BYTES,
    array_paths,
    check_request,
    dump_line,
    from_wire,
    parse_line,
    to_wire,
)
from senselab_worker.faults import (
    ArrayIntegrityError,
    FaultClass,
    InfrastructureFault,
    ProtocolError,
    error_record,
)

HELLO_TIMEOUT_S = 60.0
STOP_GRACE_S = 10.0
STDERR_TAIL_LINES = 100
_STDERR_LINE_BYTES = 64 * 2**10

logger = logging.getLogger(__name__)


class ProtocolMismatch(InfrastructureFault):
    """The worker's wire version or runtime differs from the host's, or it sent no hello."""


class WorkerLost(InfrastructureFault):
    """The worker exited or closed its output while a reply was awaited, or was already discarded."""


class WorkerFatal(InfrastructureFault):
    """The worker reported a fatal error; `error` is its error object."""

    def __init__(self, error: dict[str, Any]) -> None:
        """Wrap the worker's error object."""
        super().__init__(f"worker fatal: {error.get('type')}: {error.get('message')}")
        self.error = error


class WorkerClient:
    """A worker process running `python -m <module>`, spoken to over the wire protocol.

    `array_dir` is created if missing and removed on close. Host faults (`ProtocolMismatch`,
    `ProtocolError`, `WorkerLost`, `WorkerFatal`, `TimeoutError`) discard the worker and carry its
    last stderr lines as a note. `hello` holds the worker's hello.
    """

    def __init__(
        self,
        python: str | os.PathLike[str],
        module: str,
        *,
        init: dict[str, Any],
        array_dir: Path,
        expect_runtime: str | None = None,
        load_timeout: float | None = None,
        request_timeout: float | None = None,
    ) -> None:
        """Start the worker, check its hello and wait until it has loaded `init`.

        Raises:
            ValueError: if `init` uses the reserved key `array_dir` or has no wire form.
            ProtocolMismatch: if the hello is missing or its wire or runtime differ from the host's
                (`expect_runtime` overrides the host's runtime identity).
            WorkerFatal: if loading fails.
            TimeoutError: if loading exceeds `load_timeout`.
        """
        if "array_dir" in init:
            raise ValueError("init key 'array_dir' is reserved")
        self._array_dir = Path(array_dir).absolute()
        self._request_timeout = request_timeout
        self._stderr: collections.deque[str] = collections.deque(maxlen=STDERR_TAIL_LINES)
        self._lines: queue.Queue[bytes | None] = queue.Queue()
        self._proc: subprocess.Popen[bytes] | None = None
        self._threads: list[threading.Thread] = []
        self._closed = False
        self.hello: dict[str, Any] = {}
        self._array_dir.mkdir(parents=True, exist_ok=True)
        try:
            wire_init = to_wire({"array_dir": str(self._array_dir), **init}, array_dir=self._array_dir)
            self._init_paths = self._owned(array_paths(wire_init))
            self._start(python, module)
            self.hello = self._handshake(expect_runtime or runtime_identity())
            self._send(dump_line({"init": wire_init}))
            if "ready" not in self._receive(load_timeout):
                raise ProtocolError("expected ready after init")
        except BaseException as exc:
            self._fail(exc)
            raise

    @property
    def alive(self) -> bool:
        """Whether the worker process is running and not discarded."""
        return not self._closed and self._proc is not None and self._proc.poll() is None

    def run(self, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
        """Run one request and return one result per item, in order.

        Each result is `{"id", "outputs"}` with arrays loaded, or `{"id", "error"}` with the error
        object's `fault` as a `FaultClass`. An item whose inputs have no wire form fails alone with
        a content error and is not sent.

        Raises:
            ValueError: if `items` or `params` are malformed, or `params` has no wire form.
            ProtocolError, WorkerLost, WorkerFatal: infrastructure faults; the worker is discarded.
            TimeoutError: if the reply exceeds `request_timeout`; the worker is killed.
        """
        if self._closed:
            raise WorkerLost("worker was discarded or closed; start a new WorkerClient")
        request_id = str(uuid.uuid4())
        check_request({"id": request_id, "items": items, "params": params})
        wire_params = to_wire(params, array_dir=self._array_dir)
        results: list[dict[str, Any]] = [{} for _ in items]
        sent: list[tuple[int, dict[str, Any]]] = []
        for i, item in enumerate(items):
            try:
                sent.append((i, {"id": item["id"], "inputs": to_wire(item["inputs"], array_dir=self._array_dir)}))
            except Exception as exc:
                results[i] = {"id": item["id"], "error": _mapped(error_record(exc))}
        if not sent:
            return results
        request = {"id": request_id, "items": [wire for _, wire in sent], "params": wire_params}
        line = dump_line(request)
        try:
            self._send(line)
            response = self._receive(self._request_timeout)
            replies = self._decode_response(response, request)
        except BaseException as exc:
            self._fail(exc)
            raise
        for (i, _), reply in zip(sent, replies):
            results[i] = reply
        self._remove_arrays(request, response)
        return results

    def close(self) -> None:
        """Stop the worker (SIGKILL after 10 s) and remove `array_dir`. Safe to call twice."""
        self._shutdown(graceful=True)

    def __enter__(self) -> "WorkerClient":
        """Return the client."""
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        """Close the client."""
        self.close()

    def _start(self, python: str | os.PathLike[str], module: str) -> None:
        pipe = subprocess.PIPE
        self._proc = subprocess.Popen([os.fspath(python), "-m", module], stdin=pipe, stdout=pipe, stderr=pipe)
        assert self._proc.stdout is not None and self._proc.stderr is not None
        self._threads = [
            threading.Thread(target=self._pump_stdout, args=(self._proc.stdout,), daemon=True),
            threading.Thread(target=self._pump_stderr, args=(self._proc.stderr,), daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def _pump_stdout(self, stdout: IO[bytes]) -> None:
        while line := stdout.readline(MAX_LINE_BYTES + 1):
            self._lines.put(line)
        self._lines.put(None)

    def _pump_stderr(self, stderr: IO[bytes]) -> None:
        while line := stderr.readline(_STDERR_LINE_BYTES):
            self._stderr.append(line.decode("utf-8", "replace").rstrip("\n"))

    def _handshake(self, expected_runtime: str) -> dict[str, Any]:
        try:
            line = self._lines.get(timeout=HELLO_TIMEOUT_S)
        except queue.Empty:
            line = None
        hello = _hello(line)
        got = (hello.get("wire"), hello.get("runtime")) if hello is not None else None
        if hello is None or got != (WIRE_VERSION, expected_runtime):
            raise ProtocolMismatch(
                f"worker hello gave (wire, runtime) = {got!r}; host expects ({WIRE_VERSION!r}, {expected_runtime!r})"
            )
        return hello

    def _send(self, line: bytes) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        try:
            self._proc.stdin.write(line)
            self._proc.stdin.flush()
        except OSError as exc:
            raise WorkerLost(f"cannot write to the worker: {exc}") from exc

    def _receive(self, timeout: float | None) -> dict[str, Any]:
        assert self._proc is not None
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"no reply from the worker within {timeout} s") from None
        if line is None:
            try:
                code: int | None = self._proc.wait(timeout=STOP_GRACE_S)
            except subprocess.TimeoutExpired:
                code = None
            raise WorkerLost(f"worker closed its output while a reply was awaited; exit code {code}")
        msg = parse_line(line)
        if msg.get("fatal") is True:
            raise WorkerFatal(_mapped(msg.get("error")))
        return msg

    def _decode_response(self, response: dict[str, Any], request: dict[str, Any]) -> list[dict[str, Any]]:
        results = response.get("results")
        expected = [item["id"] for item in request["items"]]
        if (
            response.get("id") != request["id"]
            or not isinstance(results, list)
            or [r.get("id") if isinstance(r, dict) else None for r in results] != expected
        ):
            raise ProtocolError("response does not answer the request: its id, count, order or item ids differ")
        return [self._decode_result(r) for r in results]

    def _decode_result(self, result: dict[str, Any]) -> dict[str, Any]:
        if ("outputs" in result) == ("error" in result):
            raise ProtocolError(f"result for item {result['id']!r} needs exactly one of outputs and error")
        if "error" in result:
            error = _mapped(result["error"])
            if error["gap"]:
                logger.warning("worker fault classification gap: %s: %s", error.get("type"), error.get("message"))
            return {"id": result["id"], "error": error}
        if not isinstance(result["outputs"], dict):
            raise ProtocolError(f"outputs for item {result['id']!r} are not an object")
        try:
            return {"id": result["id"], "outputs": from_wire(result["outputs"], array_dir=self._array_dir)}
        except ArrayIntegrityError as exc:
            return {"id": result["id"], "error": _mapped(error_record(exc))}

    def _owned(self, paths: set[Path]) -> set[Path]:
        root = self._array_dir.resolve()
        return {p.resolve() for p in paths if p.resolve().parent == root}

    def _remove_arrays(self, *messages: dict[str, Any]) -> None:
        paths = set().union(*(array_paths(m) for m in messages))
        for path in self._owned(paths) - self._init_paths:
            path.unlink(missing_ok=True)

    def _fail(self, exc: BaseException) -> None:
        self._shutdown(graceful=isinstance(exc, ProtocolMismatch))
        if isinstance(exc, (InfrastructureFault, TimeoutError)):
            exc.add_note(f"worker stderr, last {len(self._stderr)} lines:\n" + "\n".join(self._stderr))

    def _shutdown(self, *, graceful: bool) -> None:
        if self._closed:
            return
        self._closed = True
        proc = self._proc
        if proc is not None:
            assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
            if graceful and proc.poll() is None:
                try:
                    proc.stdin.write(dump_line({"stop": True}))
                    proc.stdin.flush()
                except OSError:
                    pass
            try:
                proc.stdin.close()
            except OSError:
                pass
            if not graceful:
                proc.kill()
            try:
                proc.wait(timeout=STOP_GRACE_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            for thread in self._threads:
                thread.join(STOP_GRACE_S)
            proc.stdout.close()
            proc.stderr.close()
        if self._array_dir.exists():
            shutil.rmtree(self._array_dir)


def _hello(line: bytes | None) -> dict[str, Any] | None:
    if line is None:
        return None
    try:
        hello = parse_line(line).get("hello")
    except ProtocolError:
        return None
    return hello if isinstance(hello, dict) else None


def _mapped(error: object) -> dict[str, Any]:
    """Return a wire error object with `fault` as a `FaultClass`, or raise `ProtocolError`."""
    if not (isinstance(error, dict) and isinstance(error.get("gap"), bool)):
        raise ProtocolError(f"malformed error object: {error!r}")
    try:
        fault = FaultClass(error.get("fault"))
    except ValueError as exc:
        raise ProtocolError(f"unknown fault class in error object: {error.get('fault')!r}") from exc
    return {**error, "fault": fault}
