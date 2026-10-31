"""The host side of the worker wire protocol: start a worker, send it requests, map its faults."""

import collections
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
from collections.abc import Callable
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
)
from senselab_worker.faults import (
    ArrayIntegrityError,
    FaultClass,
    InfrastructureFault,
    ProtocolError,
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
STDERR_TAIL_LINES = 100
_STDERR_LINE_BYTES = 64 * 2**10
_READ_BYTES = 2**16
_GROUP_POLL_MAX_S = 0.05
_LOGGED_LINE_BYTES = 200

logger = logging.getLogger(__name__)


class ProtocolMismatch(InfrastructureFault):
    """The worker's wire version or runtime differs from the host's, or it sent no hello."""


class WorkerLost(InfrastructureFault):
    """The worker exited or closed its output while a reply was awaited, or was already discarded.

    `returncode` is the worker's exit code when it is known, else `None`.
    """

    def __init__(self, message: str, *, returncode: int | None = None) -> None:
        """Carry the message and the worker's exit code."""
        super().__init__(message)
        self.returncode = returncode


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


_NOTE_EXITSTATUS = 0x04000000 if sys.platform == "darwin" else 0
_PS = "/bin/ps"
_HAS_WAITID = hasattr(os, "waitid")
_PROBE_RUN: Callable[..., "subprocess.CompletedProcess[str]"] = subprocess.run
LEADER_PROBE_TIMEOUT_S = 2.0


def _wait_exit(pid: int) -> int | None:
    """Block until child `pid` exits, without reaping it; return its exit code when the platform reports one.

    Uses `waitid(WNOWAIT)` where the platform has it (Linux) and a kqueue `NOTE_EXIT` otherwise (macOS).

    Raises:
        ChildProcessError: if `pid` is not an unreaped child (waitid only).
    """
    if hasattr(os, "waitid"):
        info = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT)
        if info is None:
            return None
        return int(info.si_status) if info.si_code == os.CLD_EXITED else -int(info.si_status)
    kq = select.kqueue()
    try:
        flags = select.KQ_EV_ADD | select.KQ_EV_ONESHOT
        event = select.kevent(
            pid, filter=select.KQ_FILTER_PROC, flags=flags, fflags=select.KQ_NOTE_EXIT | _NOTE_EXITSTATUS
        )
        for fired in kq.control([event], 1, None):
            if fired.fflags & _NOTE_EXITSTATUS and not fired.flags & select.KQ_EV_ERROR:
                return os.waitstatus_to_exitcode(fired.data)
        return None
    finally:
        kq.close()


def _reap(pid: int, exited: threading.Event, status: list[int], stdout: _Pump) -> None:
    """Wait for the leader to exit without reaping it, so its pid and process group id stay reserved."""
    try:
        code = _wait_exit(pid)
        if code is not None:
            status.append(code)
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
    if _HAS_WAITID:
        try:
            os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)  # type: ignore[attr-defined,unused-ignore]
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


class WorkerClient:
    """A worker process running `python -m senselab_worker.launch <module>`, spoken to over the wire protocol.

    `module` defines `load` and `handle` (see `senselab_worker.launch`). The client makes its own
    `array_dir` under `array_parent` (the system temporary directory if `None`) and deletes only the
    array files in it, never other names: after each request every array file except those `init`
    references, and on close all of them and then the directory if it is empty. The worker runs in
    its own process group, which `close` and every host fault stop within `SHUTDOWN_BOUND_S`; a
    descendant that calls `setsid` leaves the group and is not reached. One `run` may be in flight at
    a time. Host faults (`ProtocolMismatch`, `ProtocolError`, `WorkerLost`, `WorkerFatal`,
    `TimeoutError`) discard the worker and carry its last stderr lines as a note. `hello` holds the
    worker's hello.
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
            ProtocolMismatch: if the hello is missing or its wire or runtime differ from the host's
                (`expect_runtime` overrides the host's runtime identity).
            WorkerFatal: if loading fails.
            TimeoutError: if sending `init` and loading exceed `load_timeout`.
        """
        self._closed = True
        if "array_dir" in init:
            raise ValueError("init key 'array_dir' is reserved")
        self._request_timeout = request_timeout
        self._stderr: collections.deque[str] = collections.deque(maxlen=STDERR_TAIL_LINES)
        self._lines: queue.Queue[bytes | None] = queue.Queue()
        self._proc: subprocess.Popen[bytes] | None = None
        self._pumps: list[_Pump] = []
        self._reaper: threading.Thread | None = None
        self._exited = threading.Event()
        self._exit_status: list[int] = []
        self._run_lock = threading.Lock()
        self._run_owner: int | None = None
        self._run_idle = threading.Event()
        self._run_idle.set()
        self._shutdown_lock = threading.Lock()
        self._terminated = False
        self._released = False
        self._init_names: set[str] = set()
        self.hello: dict[str, Any] = {}
        self._array_dir = Path(tempfile.mkdtemp(prefix="senselab-arrays-", dir=array_parent)).absolute()
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
            self._fail(exc)
            raise

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
            ValueError: if `items` or `params` are malformed, `params` has no wire form, or the request
                with every encodable item does not fit one wire line.
            ProtocolError, WorkerLost, WorkerFatal: infrastructure faults; the worker is discarded.
            TimeoutError: if the request exceeds `request_timeout`; the worker is killed.
        """
        if not self._run_lock.acquire(blocking=False):
            raise ConcurrentUse("run is already in flight on this client; use one client per thread")
        self._run_owner = threading.get_ident()
        self._run_idle.clear()
        try:
            return self._run(items, params)
        finally:
            if self._closed:
                self._shutdown(graceful=False)
            else:
                self._remove_arrays(keep=self._init_names)
            self._run_owner = None
            self._run_idle.set()
            self._run_lock.release()

    def close(self) -> None:
        """Stop the worker's process group and remove this client's files, within `SHUTDOWN_BOUND_S`.

        Safe to call twice and from another thread; a `run` in flight then raises `WorkerLost`.
        """
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
        """Close a client that was never closed, with a `ResourceWarning`."""
        if getattr(self, "_closed", True):
            return
        warnings.warn(f"WorkerClient for {self._array_dir} was not closed", ResourceWarning, source=self)
        try:
            self.close()
        except Exception:  # noqa: BLE001 - a finaliser must not raise
            logger.exception("closing a leaked WorkerClient failed")

    def _run(self, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
        if self._closed:
            raise WorkerLost("worker was discarded or closed; start a new WorkerClient")
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
        if not sent:
            return results
        request = {"id": request_id, "items": [wire for _, wire in sent], "params": wire_params}
        try:
            line = dump_line(request, "request")
        except LineTooLarge as exc:
            raise ValueError(f"request of {len(sent)} items does not fit one line; send fewer items: {exc}") from exc
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
        self._proc = proc = subprocess.Popen(argv, stdin=pipe, stdout=pipe, stderr=pipe, start_new_session=True)
        assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
        os.set_blocking(proc.stdin.fileno(), False)
        tail = self._stderr

        def keep(line: bytes | None) -> None:
            if line is not None:
                tail.append(line.decode("utf-8", "replace").rstrip("\n"))

        stdout = _Pump(proc.stdout.fileno(), MAX_LINE_BYTES + 1, self._lines.put, keep_partial=False)
        self._pumps = [stdout, _Pump(proc.stderr.fileno(), _STDERR_LINE_BYTES, keep, keep_partial=True)]
        self._reaper = threading.Thread(
            target=_reap, args=(proc.pid, self._exited, self._exit_status, stdout), daemon=True
        )
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
                break
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
        assert self._proc is not None and self._proc.stdin is not None
        fd = self._proc.stdin.fileno()
        view = memoryview(line)
        with selectors.DefaultSelector() as sel:
            sel.register(fd, selectors.EVENT_WRITE)
            while view:
                try:
                    view = view[os.write(fd, view) :]
                    continue
                except BlockingIOError:
                    pass
                except OSError as exc:
                    raise WorkerLost(f"cannot write to the worker: {exc}") from exc
                deadline.check("sending to the worker")
                sel.select(deadline.remaining())

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
            self._exited.wait(EXIT_CODE_WAIT_S)
            code = self._exit_status[0] if self._exit_status else None
            raise WorkerLost(f"worker closed its output while {what}; exit code {code}", returncode=code)
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
        self._shutdown(graceful=isinstance(exc, ProtocolMismatch))
        if isinstance(exc, (InfrastructureFault, TimeoutError)):
            exc.add_note(f"worker stderr, last {len(self._stderr)} lines:\n" + "\n".join(self._stderr))

    def _shutdown(self, *, graceful: bool) -> None:
        bound = Deadline(SHUTDOWN_BOUND_S)
        self._closed = True
        with self._shutdown_lock:
            if not self._terminated:
                self._terminated = True
                try:
                    self._terminate(graceful=graceful)
                except Exception:  # noqa: BLE001 - shutdown is total
                    logger.exception("stopping the worker failed")
        me = threading.get_ident()
        if self._run_owner not in (None, me):
            self._run_idle.wait(bound.remaining())
        with self._shutdown_lock:
            if self._released or self._run_owner not in (None, me):
                return
            self._released = True
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

    def _signal_group_until_gone(self, pgid: int, stop: Deadline) -> None:
        exited = self._exited
        _signal_group(pgid, signal.SIGTERM, leader_exited=exited.is_set())
        _signal_group(pgid, signal.SIGCONT, leader_exited=exited.is_set())
        if not _members_gone(pgid, exited, stop):
            _signal_group(pgid, signal.SIGKILL, leader_exited=exited.is_set())
            if not _members_gone(pgid, exited, Deadline(KILL_GRACE_S)):
                logger.error("worker process group %d still has members after SIGKILL", pgid)

    def _terminate(self, *, graceful: bool) -> None:
        proc = self._proc
        if proc is None:
            return
        assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
        pgid = proc.pid
        stop = Deadline(STOP_GRACE_S)
        if graceful and not self._exited.is_set():
            try:
                self._send(dump_line({"stop": True}, "stop"), Deadline(0))
            except (WorkerLost, TimeoutError):
                pass
        try:
            proc.stdin.close()
        except OSError:
            pass
        if graceful:
            self._exited.wait(max((stop.remaining() or 0.0) - TERM_GRACE_S, 0.0))
        held: bool | None = False
        if proc.returncode is None:
            held = _leader_held(pgid, self._exited, min(LEADER_PROBE_TIMEOUT_S, stop.remaining() or 0.0))
        if held:
            try:
                self._signal_group_until_gone(pgid, stop)
            except Exception:
                logger.exception("stopping worker process group %d failed; sending SIGKILL", pgid)
                _signal_group(pgid, signal.SIGKILL, leader_exited=self._exited.is_set())
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
