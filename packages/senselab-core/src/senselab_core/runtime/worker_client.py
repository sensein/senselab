"""The host side of the worker wire protocol: start a worker, send it requests, map its faults."""

import collections
import contextlib
import json
import logging
import os
import queue
import select
import selectors
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from types import TracebackType
from typing import Any

from senselab_worker import WIRE_VERSION, runtime_identity
from senselab_worker.codec import (
    MAX_LINE_BYTES,
    Deadline,
    Kind,
    LineTooLarge,
    array_paths,
    dump_line,
    from_wire,
    parse_line,
    to_wire,
    validate_message,
    wire_size,
)
from senselab_worker.faults import (
    ArrayIntegrityError,
    FaultClass,
    InfrastructureFault,
    ProtocolError,
    ValueTooLarge,
    error_record,
    wire_text,
)
from senselab_worker.ownership import ARRAY_NAME, ARRAY_TEMP_NAME, remove_owned

LAUNCHER = "senselab_worker.launch"
HELLO_TIMEOUT_S = 60.0
STOP_GRACE_S = 10.0
TERM_GRACE_S = 3.0
KILL_GRACE_S = 5.0
JOIN_GRACE_S = 2.0
SHUTDOWN_BOUND_S = STOP_GRACE_S + KILL_GRACE_S + JOIN_GRACE_S
EXIT_CODE_WAIT_S = 1.0
DEFERRED_SIGNALS = (signal.SIGINT, signal.SIGTERM)
STDERR_TAIL_LINES = 100
_STDERR_LINE_BYTES = 64 * 2**10
_READ_BYTES = 2**16
_GROUP_POLL_MAX_S = 0.05
_LOGGED_LINE_BYTES = 200

logger = logging.getLogger(__name__)


class ProtocolMismatch(InfrastructureFault):
    """The worker's wire version or runtime differs from the host's, or it sent no hello."""


class WorkerLost(InfrastructureFault):
    """The worker exited or closed its output while a reply was awaited, the client was closed, or it was discarded.

    `returncode` is the worker's exit code once the client has reaped it, else `None`; `str()` names it.
    """

    def __init__(self, message: str, *, returncode: int | None = None) -> None:
        """Carry the message and the worker's exit code."""
        super().__init__(message)
        self.returncode = returncode

    def __str__(self) -> str:
        """The message and the exit code."""
        return f"{super().__str__()}; exit code {_code_text(self.returncode)}"


class WorkerFatal(InfrastructureFault):
    """The worker reported a fatal error; `error` is its error object."""

    def __init__(self, error: dict[str, Any]) -> None:
        """Wrap the worker's error object."""
        super().__init__(f"worker fatal: {error.get('type')}: {error.get('message')}")
        self.error = error


class ConcurrentUse(RuntimeError):
    """`run` was called while another `run` on the same client was in flight."""


class _Pump:
    """A daemon thread splitting one pipe into lines for `sink` until EOF, or until `finish` drains it.

    Lines longer than `limit` are passed on in pieces of `limit` bytes. At the end `sink(None)` is
    called; a trailing piece without a newline is passed on first only if `keep_partial`.
    """

    def __init__(self, fd: int, limit: int, sink: Callable[[bytes | None], None], *, keep_partial: bool) -> None:
        """Start reading `fd`."""
        self._fd = fd
        self._limit = limit
        self._sink = sink
        self._keep_partial = keep_partial
        self._wake_r, self._wake_w = os.pipe()
        self._lock = threading.Lock()
        self._closed = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def finish(self) -> None:
        """Ask the thread to read what is already in the pipe, then stop."""
        with self._lock:
            if not self._closed:
                try:
                    os.write(self._wake_w, b"x")
                except OSError:
                    pass

    def close(self) -> None:
        """Release the wake pipe; call only once the thread has ended."""
        with self._lock:
            if not self._closed:
                self._closed = True
                os.close(self._wake_r)
                os.close(self._wake_w)

    def _run(self) -> None:
        buf = bytearray()
        scanned = 0
        try:
            with selectors.DefaultSelector() as sel:
                sel.register(self._fd, selectors.EVENT_READ)
                sel.register(self._wake_r, selectors.EVENT_READ)
                draining = False
                while True:
                    if not draining and any(key.fd == self._wake_r for key, _ in sel.select()):
                        draining = True
                        os.set_blocking(self._fd, False)
                    try:
                        chunk = os.read(self._fd, _READ_BYTES)
                    except BlockingIOError:
                        if draining:
                            break
                        continue
                    except OSError:
                        break
                    if not chunk:
                        break
                    buf += chunk
                    scanned = self._split(buf, scanned)
        finally:
            if buf and self._keep_partial:
                self._sink(bytes(buf))
            self._sink(None)

    def _split(self, buf: bytearray, start: int) -> int:
        """Pass on every complete line in `buf`, searching from `start`; return where the next search starts."""
        while True:
            newline = buf.find(b"\n", start, self._limit)
            if newline >= 0:
                self._sink(bytes(buf[: newline + 1]))
                del buf[: newline + 1]
            elif len(buf) >= self._limit:
                self._sink(bytes(buf[: self._limit]))
                del buf[: self._limit]
            else:
                return len(buf)
            start = 0


_PS = "/bin/ps"
_HAS_WAITID = sys.platform == "linux"
_PROBE_RUN: Callable[..., "subprocess.CompletedProcess[str]"] = subprocess.run
LEADER_PROBE_TIMEOUT_S = 2.0


def _wait_exit(pid: int) -> None:
    """Block until child `pid` exits, without reaping it.

    Uses `waitid(WNOWAIT)` on Linux and a kqueue `NOTE_EXIT` on macOS, which also returns for a
    leader that exited before the event was registered. The exit code is read when the client reaps.

    Raises:
        ChildProcessError: if `pid` is not an unreaped child (waitid only).
        NotImplementedError: on a platform that is neither Linux nor macOS.
    """
    if sys.platform == "linux":
        os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT)
        return
    if sys.platform == "darwin":
        kq = select.kqueue()
        try:
            flags = select.KQ_EV_ADD | select.KQ_EV_ONESHOT
            kq.control([select.kevent(pid, filter=select.KQ_FILTER_PROC, flags=flags, fflags=select.KQ_NOTE_EXIT)], 1)
        finally:
            kq.close()
        return
    raise NotImplementedError(f"no exit notification for platform {sys.platform!r}")


def _reap(pid: int, exited: threading.Event, stdout: _Pump) -> None:
    """Wait for the leader to exit without reaping it, so its pid and process group id stay reserved."""
    try:
        _wait_exit(pid)
    except ChildProcessError:
        pass
    finally:
        exited.set()
        stdout.finish()


def _leader_held(pid: int, exited: threading.Event, timeout: float) -> bool | None:
    """Whether child `pid` is unreaped, so its pid and group id are still reserved; `None` if that cannot be told.

    A leader whose exit has not been observed is running, hence held. After its exit the answer comes
    from `waitid(WNOHANG|WNOWAIT)` where the platform has it, else from `ps -o ppid=` run by
    `_PROBE_RUN` within `timeout` seconds.
    """
    if not exited.is_set():
        return True
    if sys.platform == "linux" and _HAS_WAITID:
        try:
            os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        except ChildProcessError:
            return False
        return True
    try:
        out = _PROBE_RUN([_PS, "-o", "ppid=", "-p", str(pid)], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("cannot tell whether worker %d is still this process's child: %s", pid, exc)
        return None
    return out.stdout.strip() == str(os.getpid())


def _signal_group(pgid: int, sig: signal.Signals, *, leader_exited: bool) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass
    except OSError as exc:
        # macOS answers EPERM for a group whose members are all zombies or exiting.
        if not (isinstance(exc, PermissionError) and leader_exited and sys.platform == "darwin"):
            logger.warning("signal %s to worker process group %d failed: %s", sig.name, pgid, exc)


def _live_members(pgid: int) -> bool:
    """Whether group `pgid` has a member that is neither a zombie nor exiting: by `/proc`, else by `killpg(pgid, 0)`."""
    if os.path.isdir("/proc/self/task"):
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/stat", "rb") as f:
                    data = f.read()
            except OSError:
                continue
            fields = data[data.rfind(b")") + 2 :].split()
            if len(fields) > 2 and int(fields[2]) == pgid and fields[0] not in (b"Z", b"X"):
                return True
        return False
    try:
        os.killpg(pgid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _members_gone(pgid: int, exited: threading.Event, deadline: Deadline) -> bool:
    """Wait until the leader has exited and group `pgid` has no live member; return whether that happened."""
    delay = 0.001
    while True:
        if exited.is_set() and not _live_members(pgid):
            return True
        remaining = deadline.remaining()
        if remaining is not None and remaining <= 0:
            return False
        pause = delay if remaining is None else min(delay, remaining)
        if not exited.is_set():
            exited.wait(pause)
        else:
            time.sleep(pause)
        delay = min(delay * 2, _GROUP_POLL_MAX_S)


ESCALATE_HINT = "stopping the worker; press Ctrl-C again to kill it now"


@contextlib.contextmanager
def _signals_deferred(*, escalate: bool) -> Iterator[None]:
    """On the main thread, hold `DEFERRED_SIGNALS` for the body, then restore their handlers and raise each held one.

    With `escalate`, the first SIGINT is held and logs `ESCALATE_HINT`, and a second raises `KeyboardInterrupt`
    at once. A signal whose handler was not installed from Python cannot be restored, so it is not held. Held
    signals are not raised again when a wakeup fd is set, since its writer already saw them.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    held: list[int] = []
    previous: dict[int, Any] = {}

    def hold(signum: int, frame: object) -> None:
        if escalate and signum == signal.SIGINT:
            if signal.SIGINT in held:
                held[:] = [s for s in held if s != signal.SIGINT]
                raise KeyboardInterrupt
            logger.warning(ESCALATE_HINT)
        held.append(signum)

    try:
        for sig in DEFERRED_SIGNALS:
            if signal.getsignal(sig) is not None:
                previous[sig] = signal.signal(sig, hold)
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
        wakeup = signal.set_wakeup_fd(-1)
        signal.set_wakeup_fd(wakeup)
        if wakeup == -1:
            for number in dict.fromkeys(held):
                signal.raise_signal(number)


def _code_text(returncode: int | None) -> str:
    return "unknown" if returncode is None else str(returncode)


class WorkerClient:
    """A worker process running `python -m senselab_worker.launch <module>`, spoken to over the wire protocol.

    `module` defines `load` and `handle` (see `senselab_worker.launch`). The client makes its own
    `array_dir` under `array_parent` (the system temporary directory if `None`) and deletes only the
    array files in it, never other names: after each request every array file except those `init`
    references, and on close all of them and then the directory if it is empty. The worker runs in
    its own process group, which `close` and every host fault stop within `SHUTDOWN_BOUND_S`; a
    descendant that calls `setsid` leaves the group and is not reached. One `run` may be in flight at
    a time. Host faults (`ProtocolMismatch`, `ProtocolError`, `WorkerLost`, `WorkerFatal`,
    `TimeoutError`) discard the worker and carry its exit code and last stderr lines as a note.
    `hello` holds the worker's hello.

    Shutdown is two steps, stopping the process group and then removing the files, each recorded
    only once it has completed and each run once. On the main thread SIGINT and SIGTERM are held
    while it runs and raised after it; a second SIGINT, or anything else that interrupts it,
    SIGKILLs the group, reaps it and removes the files before propagating. A copy of the client in
    a forked child is foreign: closing or collecting it closes only the child's descriptors.
    """

    def __init__(
        self,
        python: str | os.PathLike[str],
        module: str,
        *,
        init: dict[str, Any],
        array_parent: Path | None = None,
        expect_runtime: str | None = None,
        load_timeout: float | None = None,
        request_timeout: float | None = None,
    ) -> None:
        """Start the worker, check its hello and wait until it has loaded `init`.

        Raises:
            ValueError: if `init` uses the reserved key `array_dir` or has no wire form.
            ProtocolMismatch: if the worker's first message is not a hello, its wire or runtime differ
                from the host's (`expect_runtime` overrides the host's runtime identity), or no hello
                arrives within `HELLO_TIMEOUT_S`.
            WorkerLost: if the worker exits or closes its output before it is ready.
            WorkerFatal: if loading fails.
            TimeoutError: if sending `init` and loading exceed `load_timeout`.
        """
        self._closed = True
        self._owner_pid = os.getpid()
        if "array_dir" in init:
            raise ValueError("init key 'array_dir' is reserved")
        self._request_timeout = request_timeout
        self._stderr: collections.deque[str] = collections.deque(maxlen=STDERR_TAIL_LINES)
        self._lines: queue.SimpleQueue[bytes | None] = queue.SimpleQueue()
        self._proc: subprocess.Popen[bytes] | None = None
        self._pumps: list[_Pump] = []
        self._reaper: threading.Thread | None = None
        self._exited = threading.Event()
        self._run_lock = threading.Lock()
        self._run_owner: int | None = None
        self._run_idle = threading.Event()
        self._run_idle.set()
        self._channel_lock = threading.Lock()
        self._shutdown_lock = threading.Lock()
        self._shutting_down: set[int] = set()
        self._closed_by_caller = False
        self._stopped = False
        self._released = False
        self._init_names: set[str] = set()
        self.hello: dict[str, Any] = {}
        self._wake_r, self._wake_w = os.pipe()
        try:
            self._array_dir = Path(tempfile.mkdtemp(prefix="senselab-arrays-", dir=array_parent)).absolute()
        except BaseException:
            os.close(self._wake_r)
            os.close(self._wake_w)
            self._stopped = self._released = True
            raise
        self._closed = False
        try:
            wire_init = to_wire({"array_dir": str(self._array_dir), **init}, array_dir=self._array_dir, at="init")
            self._init_names = self._owned(array_paths(wire_init))
            self._start(python, module)
            self.hello = self._handshake(expect_runtime or runtime_identity())
            deadline = Deadline(load_timeout)
            self._send(dump_line({"init": wire_init}, "init"), deadline)
            self._receive(deadline, "loading the backend", "ready")
        except BaseException as exc:
            raised = exc
            if self._closed_by_caller and isinstance(exc, Exception) and not isinstance(exc, WorkerLost):
                raised = WorkerLost("the client was closed while it was starting")
            try:
                self._fail(raised)
            finally:
                if not (self._stopped and self._released):
                    self._shutdown(graceful=False)
            if raised is exc:
                raise
            raise raised from exc

    @property
    def array_dir(self) -> Path:
        """The directory this client created for array files."""
        return self._array_dir

    @property
    def alive(self) -> bool:
        """Whether the worker process is running and not discarded."""
        return not self._closed and self._proc is not None and not self._exited.is_set()

    @property
    def returncode(self) -> int | None:
        """The worker process's exit code once it has been reaped, else `None`."""
        return None if self._proc is None else self._proc.returncode

    def run(self, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
        """Run one request and return one result per item, in order.

        Each result is `{"id", "outputs"}` with arrays loaded, or `{"id", "error"}` with the error
        object's `fault` as a `FaultClass`. An item whose inputs have no wire form fails alone, with
        its content error (`ValueTooLarge`, a resource error, if they cannot fit a line), and is not
        sent. One `request_timeout` covers sending the request, receiving the response and reading
        its arrays.

        Raises:
            ConcurrentUse: if another `run` on this client is in flight; that call is unaffected.
            ValueError: if `items` or `params` are malformed or `params` has no wire form.
            ValueTooLarge: if `params` cannot fit one wire line, or the items that each fit alone do
                not fit one line together; an item that cannot fit alone fails alone instead.
            ProtocolError, WorkerLost, WorkerFatal: infrastructure faults; the worker is discarded.
            TimeoutError: if the request exceeds `request_timeout`; the worker is killed.
        """
        if self._foreign():
            raise WorkerLost(f"this client belongs to process {self._owner_pid}; start a new WorkerClient")
        if not self._run_lock.acquire(blocking=False):
            raise ConcurrentUse("run is already in flight on this client; use one client per thread")
        self._run_owner = threading.get_ident()
        self._run_idle.clear()
        try:
            return self._run(items, params)
        except Exception as exc:
            if self._closed_by_caller and not isinstance(exc, WorkerLost):
                lost = WorkerLost("the client was closed during the run")
                self._fail(lost)
                raise lost from exc
            raise
        finally:
            try:
                if not self._closed:
                    self._remove_arrays(keep=self._init_names)
            finally:
                self._run_owner = None
                try:
                    if self._closed:
                        self._shutdown(graceful=False)
                finally:
                    self._run_idle.set()
                    self._run_lock.release()

    def close(self) -> None:
        """Stop the worker's process group and remove this client's files, within `SHUTDOWN_BOUND_S`.

        Safe to call twice and from another thread; a `run` in flight then raises `WorkerLost`.
        Called again on a thread already closing this client, as from a signal handler, it returns at
        once and the outer call finishes. Called on the thread running `run`, as from a signal
        handler, it marks the client closed and returns; that `run`, if it has not finished, stops
        the worker and raises `WorkerLost`. In a forked child it closes only the child's copies of
        the client's descriptors.
        """
        if self._foreign():
            self._disown()
            return
        if not self._closed:
            self._closed_by_caller = True
        if self._run_owner == threading.get_ident():
            self._closed = True
            self._lines.put(None)
            try:
                os.write(self._wake_w, b"x")
            except OSError:
                pass
            return
        self._shutdown(graceful=self._run_owner is None)

    def __enter__(self) -> "WorkerClient":
        """Return the client."""
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        """Close the client."""
        self.close()

    def __del__(self) -> None:
        """Close a client that was never closed, with a `ResourceWarning`, and finish a shutdown that was cut short."""
        if getattr(self, "_released", True) and getattr(self, "_stopped", True):
            return
        try:
            if self._foreign():
                self._disown()
                return
            if not self._closed:
                warnings.warn(f"WorkerClient for {self._array_dir} was not closed", ResourceWarning, source=self)
            self._shutdown(graceful=not self._closed)
        except Exception:  # noqa: BLE001 - a finaliser must not raise
            logger.exception("closing a leaked WorkerClient failed")

    def _foreign(self) -> bool:
        """Whether this is a copy of the client in a process forked from its owner."""
        return os.getpid() != self._owner_pid

    def _disown(self) -> None:
        """In a forked child, close the child's copies of the client's descriptors; touch nothing the owner has."""
        self._closed = True
        self._stopped = self._released = True
        proc = self._proc
        streams = [] if proc is None else [proc.stdin, proc.stdout, proc.stderr]
        for stream in streams:
            if stream is not None:
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass
        for pump in self._pumps:
            pump.close()
        for name in ("_wake_r", "_wake_w"):
            fd = getattr(self, name)
            setattr(self, name, -1)
            if fd >= 0:
                os.close(fd)

    def _run(self, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
        if self._closed:
            raise WorkerLost("worker was discarded or closed; start a new WorkerClient", returncode=self.returncode)
        request_id = str(uuid.uuid4())
        try:
            validate_message({"id": request_id, "items": items, "params": params}, "request")
        except ProtocolError as exc:
            raise ValueError(f"malformed request: {exc}") from exc
        wire_params = to_wire(params, array_dir=self._array_dir, at="params")
        results: list[dict[str, Any]] = [{} for _ in items]
        sent: list[tuple[int, dict[str, Any]]] = []
        for i, item in enumerate(items):
            try:
                sent.append(
                    (i, {"id": item["id"], "inputs": to_wire(item["inputs"], array_dir=self._array_dir, at="inputs")})
                )
            except Exception as exc:
                results[i] = {"id": item["id"], "error": _mapped(error_record(exc))}
        sent, line = _request_line(request_id, sent, wire_params, results)
        if not sent:
            return results
        request = {"id": request_id, "items": [wire for _, wire in sent]}
        deadline = Deadline(self._request_timeout)
        try:
            self._send(line, deadline)
            replies = self._read_replies(request, deadline)
        except BaseException as exc:
            self._fail(exc)
            raise
        for (i, _), reply in zip(sent, replies):
            results[i] = reply
        return results

    def _start(self, python: str | os.PathLike[str], module: str) -> None:
        pipe = subprocess.PIPE
        argv = [os.fspath(python), "-m", LAUNCHER, module]
        with self._shutdown_lock:
            if self._closed:
                raise WorkerLost("the client was closed before its worker started")
            self._proc = proc = subprocess.Popen(argv, stdin=pipe, stdout=pipe, stderr=pipe, start_new_session=True)
        assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
        os.set_blocking(proc.stdin.fileno(), False)
        tail = self._stderr

        def keep(line: bytes | None) -> None:
            if line is not None:
                tail.append(line.decode("utf-8", "replace").rstrip("\n"))

        stdout = _Pump(proc.stdout.fileno(), MAX_LINE_BYTES + 1, self._lines.put, keep_partial=False)
        self._pumps = [stdout, _Pump(proc.stderr.fileno(), _STDERR_LINE_BYTES, keep, keep_partial=True)]
        self._reaper = threading.Thread(target=_reap, args=(proc.pid, self._exited, stdout), daemon=True)
        self._reaper.start()

    def _handshake(self, expected_runtime: str) -> dict[str, Any]:
        deadline = Deadline(HELLO_TIMEOUT_S)
        hello: dict[str, Any] | None = None
        while True:
            try:
                line = self._lines.get(timeout=deadline.remaining())
            except queue.Empty:
                break
            if line is None:
                self._lines.put(None)
                raise WorkerLost("the worker exited or closed its output before its hello")
            try:
                msg = parse_line(line)
            except ProtocolError:
                logger.warning("worker wrote a non-JSON line before its hello: %r", line[:_LOGGED_LINE_BYTES])
                continue
            try:
                hello = validate_message(msg, "hello")["hello"]
            except ProtocolError as exc:
                raise ProtocolMismatch(
                    f"first message is not a hello: {exc}; host expects wire {WIRE_VERSION!r}"
                ) from exc
            break
        got = (hello.get("wire"), hello.get("runtime")) if hello is not None else None
        if hello is None or got != (WIRE_VERSION, expected_runtime):
            raise ProtocolMismatch(
                f"worker hello gave (wire, runtime) = {got!r}; host expects ({WIRE_VERSION!r}, {expected_runtime!r})"
            )
        return hello

    def _send(self, line: bytes, deadline: Deadline) -> None:
        """Write `line` by non-blocking writes, each under the channel lock and only while the client is open."""
        assert self._proc is not None and self._proc.stdin is not None
        stdin = self._proc.stdin
        view = memoryview(line)
        with selectors.DefaultSelector() as sel:
            sel.register(self._wake_r, selectors.EVENT_READ)
            watched = False
            while view:
                with self._channel_lock:
                    if self._closed or stdin.closed:
                        raise WorkerLost("the client was closed while sending to the worker")
                    fd = stdin.fileno()
                    if not watched:
                        sel.register(fd, selectors.EVENT_WRITE)
                        watched = True
                    try:
                        view = view[os.write(fd, view) :]
                        continue
                    except BlockingIOError:
                        pass
                    except OSError as exc:
                        raise WorkerLost(f"cannot write to the worker: {exc}") from exc
                deadline.check("sending to the worker")
                sel.select(deadline.remaining())

    def _write_stop(self) -> None:
        """Write the stop line if the pipe takes it at once."""
        assert self._proc is not None and self._proc.stdin is not None
        stdin = self._proc.stdin
        with self._channel_lock:
            if stdin.closed:
                return
            try:
                os.write(stdin.fileno(), dump_line({"stop": True}, "stop"))
            except OSError:
                pass

    def _retire_channel(self) -> None:
        """Close the worker's stdin under the channel lock, then wake a `_send` waiting on it."""
        assert self._proc is not None and self._proc.stdin is not None
        with self._channel_lock:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
        if not self._released:
            try:
                os.write(self._wake_w, b"x")
            except OSError:
                pass

    def _receive(self, deadline: Deadline, what: str, *kinds: Kind) -> dict[str, Any]:
        """Return the next line as a message of one of `kinds`, the first that has its key; a `fatal` raises."""
        assert self._proc is not None
        try:
            line = self._lines.get(timeout=deadline.remaining())
        except queue.Empty:
            deadline.check(what)
            raise TimeoutError(f"{what} did not finish in time") from None
        if line is None:
            self._lines.put(None)
            if self._closed_by_caller:
                raise WorkerLost(f"the client was closed while {what}")
            raise WorkerLost(f"worker closed its output while {what}")
        msg = parse_line(line)
        if "fatal" in msg:
            raise WorkerFatal(_fatal_record(validate_message(msg, "fatal")["error"]))
        kind = next((k for k in kinds if k in msg), kinds[-1])
        return validate_message(msg, kind)

    def _read_replies(self, request: dict[str, Any], deadline: Deadline) -> list[dict[str, Any]]:
        """Read one result line per item, in request order, then `done`."""
        replies = []
        for item in request["items"]:
            reply = self._receive(deadline, "running the request", "result")
            if (reply["id"], reply["item"]) != (request["id"], item["id"]):
                raise ProtocolError(
                    f"expected the result of item {item['id']!r} of request {request['id']!r}, "
                    f"got item {reply['item']!r} of request {reply['id']!r}"
                )
            replies.append(self._decode_result(reply, deadline))
        done = self._receive(deadline, "running the request", "done")
        if (done["id"], done["done"]) != (request["id"], len(request["items"])):
            raise ProtocolError(
                f"expected done for {len(request['items'])} items of request {request['id']!r}, got {done!r}"
            )
        return replies

    def _decode_result(self, result: dict[str, Any], deadline: Deadline) -> dict[str, Any]:
        if "error" in result:
            error = _mapped(result["error"])
            if error["gap"]:
                logger.warning("worker fault classification gap: %s: %s", error["type"], error["message"])
            return {"id": result["item"], "error": error}
        try:
            outputs = from_wire(result["outputs"], array_dir=self._array_dir, deadline=deadline)
        except ArrayIntegrityError as exc:
            return {"id": result["item"], "error": _mapped(error_record(exc))}
        return {"id": result["item"], "outputs": outputs}

    def _owned(self, paths: set[Path]) -> set[str]:
        root = self._array_dir.resolve()
        return {p.name for p in paths if p.parent.resolve() == root}

    def _remove_arrays(self, *, keep: set[str]) -> None:
        try:
            names = os.listdir(self._array_dir)
        except Exception as exc:  # noqa: BLE001 - cleanup is total; it never replaces a result
            logger.warning("cannot list array directory %s: %s", self._array_dir, exc)
            return
        for name in names:
            if name in keep:
                continue
            path = self._array_dir / name
            try:
                if not remove_owned(self._array_dir, path, pattern=ARRAY_NAME):
                    remove_owned(self._array_dir, path, pattern=ARRAY_TEMP_NAME)
            except Exception as exc:  # noqa: BLE001 - cleanup is total; it never replaces a result
                logger.warning("cannot remove array file %s: %s", path, exc)

    def _fail(self, exc: BaseException) -> None:
        """Stop the worker for the host fault `exc`, then give `exc` the exit code and the stderr tail."""
        self._closed = True
        if isinstance(exc, WorkerLost) and not self._closed_by_caller:
            self._exited.wait(EXIT_CODE_WAIT_S)
        self._shutdown(graceful=isinstance(exc, ProtocolMismatch))
        if isinstance(exc, WorkerLost) and exc.returncode is None:
            exc.returncode = self.returncode
        if isinstance(exc, (InfrastructureFault, TimeoutError)):
            exc.add_note(
                f"worker exit code {_code_text(self.returncode)}; stderr, last {len(self._stderr)} lines:\n"
                + "\n".join(self._stderr)
            )

    def _shutdown(self, *, graceful: bool) -> None:
        """Run each shutdown step not yet completed: stop the group, then, once no run is in flight, remove files.

        If the steps are interrupted, the group is SIGKILLed and the steps are finished, with signals held,
        before the interruption propagates.
        """
        if self._foreign():
            self._disown()
            return
        me = threading.get_ident()
        if me in self._shutting_down:
            return
        try:
            self._shutting_down.add(me)
            self._closed = True
            try:
                with _signals_deferred(escalate=True):
                    self._steps(graceful=graceful, me=me)
            except BaseException:
                with _signals_deferred(escalate=False):
                    try:
                        self._kill_group_now()
                        self._steps(graceful=False, me=me)
                    except BaseException:  # noqa: BLE001 - the first interruption is the one raised
                        logger.exception("finishing an interrupted shutdown failed")
                raise
        finally:
            self._shutting_down.discard(me)

    def _steps(self, *, graceful: bool, me: int) -> None:
        bound = Deadline(SHUTDOWN_BOUND_S)
        if self._acquire_shutdown(bound):
            try:
                if not self._stopped:
                    self._stop(graceful=graceful)
            finally:
                self._shutdown_lock.release()
        if self._run_owner not in (None, me):
            self._run_idle.wait(bound.remaining())
        if self._acquire_shutdown(bound):
            try:
                if not self._released and self._run_owner in (None, me):
                    self._release()
            finally:
                self._shutdown_lock.release()

    def _acquire_shutdown(self, bound: Deadline) -> bool:
        if self._shutdown_lock.acquire(timeout=max(bound.remaining() or 0.0, 0.0)):
            return True
        logger.error("another thread's shutdown of this client did not finish within %s s", SHUTDOWN_BOUND_S)
        return False

    def _stop(self, *, graceful: bool) -> None:
        try:
            self._terminate(graceful=graceful)
        except Exception:  # noqa: BLE001 - shutdown is total
            logger.exception("stopping the worker failed")
            self._kill_group_now()
            return
        self._stopped = True

    def _kill_group_now(self) -> None:
        """SIGKILL the group if its leader is still this client's unreaped child."""
        proc = self._proc
        if proc is None or proc.returncode is not None:
            return
        exited = self._exited.is_set()
        if not exited or _leader_held(proc.pid, self._exited, LEADER_PROBE_TIMEOUT_S):
            _signal_group(proc.pid, signal.SIGKILL, leader_exited=exited)

    def _release(self) -> None:
        self._remove_arrays(keep=set())
        try:
            os.rmdir(self._array_dir)
        except FileNotFoundError:
            pass
        except OSError:
            try:
                left = sorted(os.listdir(self._array_dir))
            except OSError:
                left = []
            logger.warning("left %s: it holds names this client did not write: %s", self._array_dir, left)
        for name in ("_wake_r", "_wake_w"):
            fd = getattr(self, name)
            setattr(self, name, -1)
            if fd >= 0:
                os.close(fd)
        self._released = True

    def _signal_group_until_gone(self, pgid: int, stop: Deadline) -> None:
        exited = self._exited
        _signal_group(pgid, signal.SIGTERM, leader_exited=exited.is_set())
        _signal_group(pgid, signal.SIGCONT, leader_exited=exited.is_set())
        if not _members_gone(pgid, exited, stop):
            _signal_group(pgid, signal.SIGKILL, leader_exited=exited.is_set())
            if not _members_gone(pgid, exited, Deadline(KILL_GRACE_S)):
                logger.error("worker process group %d still has members after SIGKILL", pgid)

    def _terminate(self, *, graceful: bool) -> None:
        """Stop the group and reap its leader, then stop the pipe readers; each part may be run again."""
        proc = self._proc
        if proc is None:
            return
        assert proc.stdout is not None and proc.stderr is not None
        pgid = proc.pid
        stop = Deadline(STOP_GRACE_S)
        if graceful and not self._exited.is_set():
            self._write_stop()
        self._retire_channel()
        if graceful:
            self._exited.wait(max((stop.remaining() or 0.0) - TERM_GRACE_S, 0.0))
        if proc.returncode is None:
            held = _leader_held(pgid, self._exited, min(LEADER_PROBE_TIMEOUT_S, stop.remaining() or 0.0))
            if held:
                try:
                    self._signal_group_until_gone(pgid, stop)
                except Exception:
                    logger.exception("stopping worker process group %d failed; sending SIGKILL", pgid)
                    _signal_group(pgid, signal.SIGKILL, leader_exited=self._exited.is_set())
                    self._exited.wait(KILL_GRACE_S)
            elif held is None:
                logger.warning(
                    "worker %d: cannot tell whether it is still unreaped; its process group is not signalled", pgid
                )
            else:
                logger.warning("worker %d was reaped outside this client; its process group is not signalled", pgid)
            if self._exited.is_set() and proc.returncode is None:
                proc.wait()
        join = Deadline(JOIN_GRACE_S)
        for pump in self._pumps:
            pump.finish()
        threads = [pump.thread for pump in self._pumps] + ([self._reaper] if self._reaper is not None else [])
        for thread in threads:
            thread.join(join.remaining())
        self._lines.put(None)
        if any(thread.is_alive() for thread in threads):
            logger.warning("worker %d: pipe readers did not stop within %s s; its pipes stay open", pgid, JOIN_GRACE_S)
            return
        for pump in self._pumps:
            pump.close()
        proc.stdout.close()
        proc.stderr.close()


def _request_line(
    request_id: str,
    sent: list[tuple[int, dict[str, Any]]],
    params: object,
    results: list[dict[str, Any]],
) -> tuple[list[tuple[int, dict[str, Any]]], bytes]:
    """Return the items left to send and their request line; no items and an empty line if none is left.

    An item whose request alone, with `params`, cannot fit one line gets its `ValueTooLarge` in
    `results` and is left off.

    Raises:
        ValueTooLarge: if `params` with an empty item cannot fit one line, or the items that each
            fit alone do not fit together.
    """

    def dump(items: list[dict[str, Any]]) -> bytes:
        return dump_line({"id": request_id, "items": items, "params": params}, "request")

    if not sent:
        return sent, b""
    try:
        return sent, dump([wire for _, wire in sent])
    except LineTooLarge as exc:
        whole = exc
    empty = {"id": "", "inputs": {}}
    try:
        frame = len(dump([empty])) - wire_size(empty)
    except LineTooLarge as exc:
        raise ValueTooLarge(f"params do not fit one wire line with any item: {exc}") from exc
    kept: list[tuple[int, dict[str, Any]]] = []
    for i, wire in sent:
        size = frame + wire_size(wire)
        if size > MAX_LINE_BYTES:
            fault = ValueTooLarge(f"item {wire['id']!r} needs a {size}-byte request line; limit {MAX_LINE_BYTES}")
            results[i] = {"id": wire["id"], "error": _mapped(error_record(fault))}
        else:
            kept.append((i, wire))
    if not kept:
        return kept, b""
    try:
        if len(kept) == len(sent):
            raise whole
        return kept, dump([wire for _, wire in kept])
    except LineTooLarge as exc:
        raise ValueTooLarge(f"request of {len(kept)} items does not fit one line; send fewer items: {exc}") from exc


def _mapped(error: dict[str, Any]) -> dict[str, Any]:
    """Return a validated wire error record with `fault` as a `FaultClass`."""
    return {**error, "fault": FaultClass(error["fault"])}


def _fatal_record(error: object) -> dict[str, Any]:
    """Return a `fatal` message's error object as an error record; total over any JSON value.

    `type` and `message` are kept when they are strings, else replaced by placeholders naming what
    was received. A missing or unknown `fault`, or a `gap` that is not a bool, makes the record
    `infrastructure` with `gap: true`.
    """
    fields: dict[str, Any] = dict(error) if type(error) is dict else {}
    record: dict[str, Any] = {**fields}
    record["type"] = fields["type"] if type(fields.get("type")) is str else "<unknown>"
    if type(fields.get("message")) is str:
        record["message"] = fields["message"]
    else:
        record["message"] = wire_text(f"<no message; the worker sent {json.dumps(error)[:_LOGGED_LINE_BYTES]}>")
    record["traceback"] = fields["traceback"] if type(fields.get("traceback")) is str else ""
    fault = fields.get("fault")
    known = type(fault) is str and fault in {f.value for f in FaultClass} and type(fields.get("gap")) is bool
    record["fault"] = FaultClass(fault) if known else FaultClass.INFRASTRUCTURE
    record["gap"] = fields["gap"] if known else True
    return record
