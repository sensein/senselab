"""Long-lived subprocess-venv workers: one per model identity per Python process.

A backend supplies a worker script that defines two functions,

    def load(init: dict) -> object: ...
    def handle(state: object, request: dict) -> dict: ...

and calls :func:`serve_in_venv` with an identity tuple. The first call for an identity starts the
interpreter, runs ``load`` once and keeps the process; every call, the first included, sends one
request line and reads one reply line. ``load`` runs with the init payload only, so whatever it
reads is fixed for the worker's lifetime and is part of the identity by construction.

The identity is the caller's tuple, normally ``(venv, model, revision, device, compute_type)``,
plus the interpreter path, a digest of the worker script, a digest of the init payload and a
digest of the environment. Two calls that differ in any of these get two workers.

The process is shut down at interpreter exit (:func:`shutdown_venv_workers`). A worker found dead
before a request is started again and the restart is recorded in :func:`venv_worker_events`; a
worker that dies, times out or fails to load during a request is discarded, the event is recorded,
and the call raises as the one-shot path did.

Wire format: one JSON object per line on stdin and on a private copy of the worker's stdout. The
worker's own file descriptor 1 is pointed at its stderr before the backend script runs, so a
library's prints cannot interleave with replies. Design notes and measurements:
``specs/20261010-persistent-venv-workers/design.md``.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import queue
import subprocess
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from senselab.utils.data_structures.logging import logger

_SERVE_PREAMBLE = r"""
import json as _sv_json
import os as _sv_os
import sys as _sv_sys
import time as _sv_time

_sv_reply = _sv_os.fdopen(_sv_os.dup(1), "w", buffering=1)
_sv_os.dup2(2, 1)
_sv_sys.stdout = _sv_sys.stderr


def _sv_emit(obj):
    _sv_reply.write(_sv_json.dumps(obj) + "\n")
    _sv_reply.flush()


def _sv_error(exc):
    import traceback

    return {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=8)}


def _sv_threads():
    report = {k: _sv_os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    torch = _sv_sys.modules.get("torch")
    if torch is not None:
        report["torch_threads"] = torch.get_num_threads()
        report["torch_interop_threads"] = torch.get_num_interop_threads()
    try:
        report["affinity"] = len(_sv_os.sched_getaffinity(0))
    except AttributeError:
        report["affinity"] = _sv_os.cpu_count()
    return report

"""

_SERVE_LOOP = r"""

def _sv_main():
    started = _sv_time.monotonic()
    try:
        init = _sv_json.loads(_sv_sys.stdin.readline())
        state = load(init)
    except BaseException as exc:
        _sv_emit({"error": _sv_error(exc), "fatal": True})
        _sv_sys.exit(1)
    _sv_emit({"ready": True, "load_s": round(_sv_time.monotonic() - started, 3), "pid": _sv_os.getpid(),
              "threads": _sv_threads()})
    while True:
        line = _sv_sys.stdin.readline()
        if not line:
            return
        if not line.strip():
            continue
        request = _sv_json.loads(line)
        if request.get("__stop__"):
            return
        began = _sv_time.monotonic()
        try:
            result = handle(state, request)
        except Exception as exc:
            _sv_emit({"error": _sv_error(exc), "seconds": round(_sv_time.monotonic() - began, 3)})
            continue
        _sv_emit({"result": result, "seconds": round(_sv_time.monotonic() - began, 3)})


_sv_main()
"""


def _digest(value: object) -> str:
    """Return a short stable digest of a JSON-serialisable value."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _reconstruct(error: Dict[str, Any]) -> Exception:
    """Rebuild a worker-reported exception the way ``parse_subprocess_result`` does.

    Args:
        error: The worker's ``error`` object.

    Returns:
        ``ValueError`` or ``TypeError`` for those two types, ``RuntimeError`` for any other.
    """
    exc_class = {"ValueError": ValueError, "TypeError": TypeError}.get(str(error.get("type", "")), RuntimeError)
    return exc_class(str(error.get("message", "Unknown error")))


class VenvWorker:
    """One venv interpreter with a backend's model loaded, answering requests until stopped.

    Attributes:
        key: The full identity this worker serves.
        label: Name used in messages and events.
        load_s: Seconds from spawn to the worker's ready line.
        pid: The worker's process id.
        threads: The thread settings the worker reported once loaded.
        served: Requests answered.
    """

    def __init__(self, key: Tuple[Any, ...], label: str) -> None:
        """Record the worker's identity. Starting it is :meth:`start`.

        Args:
            key: The full identity this worker serves.
            label: Name used in messages and events.
        """
        self.key = key
        self.label = label
        self.load_s = 0.0
        self.pid: Optional[int] = None
        self.threads: Dict[str, Any] = {}
        self.served = 0
        self._process: Optional["subprocess.Popen[str]"] = None
        self._replies: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._noise: "deque[str]" = deque(maxlen=60)
        self.lock = threading.Lock()

    def start(self, python: str, script: str, init: Dict[str, Any], env: Dict[str, str], timeout_s: float) -> None:
        """Spawn the interpreter, send the init payload and wait for the model.

        Args:
            python: The venv interpreter.
            script: The backend's ``load``/``handle`` source.
            init: The payload ``load`` receives.
            env: The worker's environment.
            timeout_s: Wall-clock ceiling on spawn plus load.

        Raises:
            RuntimeError: The worker died or did not report ready in time.
            ValueError: ``load`` raised ``ValueError``.
            TypeError: ``load`` raised ``TypeError``.
        """
        began = time.monotonic()
        self._process = subprocess.Popen(  # noqa: S603 — the interpreter is this repo's own venv
            [python, "-c", _SERVE_PREAMBLE + script + _SERVE_LOOP],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        self.pid = self._process.pid
        threading.Thread(target=self._pump_replies, daemon=True).start()
        threading.Thread(target=self._pump_noise, daemon=True).start()
        self._send(init)
        reply = self._await(timeout_s, during="load")
        self.load_s = round(time.monotonic() - began, 3)
        self.threads = dict(reply.get("threads") or {})

    @property
    def alive(self) -> bool:
        """Whether the worker process is still running."""
        return self._process is not None and self._process.poll() is None

    def returncode(self) -> Optional[int]:
        """The worker's exit status, or ``None`` while it runs or before it started."""
        return None if self._process is None else self._process.poll()

    def request(self, payload: Dict[str, Any], timeout_s: float) -> Tuple[Dict[str, Any], float]:
        """Send one request and return the handler's result.

        Args:
            payload: The request ``handle`` receives.
            timeout_s: Wall-clock ceiling on this request.

        Returns:
            ``(result, seconds)``: the handler's dict and the worker-measured handling time.

        Raises:
            RuntimeError: The worker died, timed out, or the handler raised a type other than the two below.
            ValueError: The handler raised ``ValueError``.
            TypeError: The handler raised ``TypeError``.
        """
        self._send(payload)
        reply = self._await(timeout_s, during="request")
        self.served += 1
        if "error" in reply:
            raise _reconstruct(reply["error"] or {})
        return dict(reply.get("result") or {}), float(reply.get("seconds") or 0.0)

    def close(self) -> None:
        """End the worker. Safe on a worker that never started or already exited."""
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.write(json.dumps({"__stop__": True}) + "\n")
                process.stdin.flush()
                process.stdin.close()
            process.wait(timeout=10)
        except Exception:  # noqa: BLE001 — a worker that will not stop politely is killed
            process.kill()
            try:
                process.wait(timeout=30)
            except Exception:  # noqa: BLE001, S110 — nothing further is owed to an unreapable child
                pass

    def kill(self) -> None:
        """End the worker without asking it."""
        process, self._process = self._process, None
        if process is None:
            return
        process.kill()
        try:
            process.wait(timeout=30)
        except Exception:  # noqa: BLE001, S110 — nothing further is owed to an unreapable child
            pass

    def tail(self) -> str:
        """The worker's last lines of stderr and stray output."""
        return "\n".join(self._noise)

    def _send(self, payload: Dict[str, Any]) -> None:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError(f"{self.label} worker is not running")
        try:
            self._process.stdin.write(json.dumps(payload, default=str) + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise RuntimeError(f"{self.label} worker closed its input: {exc}\n{self.tail()}") from exc

    def _await(self, timeout_s: float, *, during: str) -> Dict[str, Any]:
        try:
            reply = self._replies.get(timeout=timeout_s)
        except queue.Empty:
            message = f"{self.label} timed out after {timeout_s:.10g}s ({during})\n{self.tail()}"
            raise _WorkerLost(message, cause=VenvWorkerTimeout(message, timeout_s)) from None
        if reply.get("eof"):
            raise _WorkerLost(f"{self.label} worker exited during {during}\n{self.tail()}")
        if reply.get("fatal"):
            error = reply.get("error") or {}
            # Let the stderr pump drain what the dying process wrote.
            time.sleep(0.2)
            raise _WorkerLost(
                f"{self.label} worker failed to load: {error.get('type')}: {error.get('message')}\n"
                f"{error.get('traceback', '')}",
                cause=_reconstruct(error),
            )
        return reply

    def _pump_replies(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            if not line.strip():
                continue
            try:
                self._replies.put(json.loads(line))
            except ValueError:
                self._noise.append(line.rstrip())
        self._replies.put({"eof": True})

    def _pump_noise(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        for line in process.stderr:
            if line.strip():
                self._noise.append(line.rstrip())


class VenvWorkerTimeout(RuntimeError):
    """A worker did not answer within its ceiling and was killed.

    Attributes:
        timeout_s: The ceiling that elapsed.
    """

    def __init__(self, message: str, timeout_s: float) -> None:
        """Record the message and the ceiling.

        Args:
            message: What timed out.
            timeout_s: The ceiling that elapsed.
        """
        super().__init__(message)
        self.timeout_s = timeout_s


class _WorkerLost(RuntimeError):
    """The worker is unusable: it died, timed out, or failed to load. Raised as its ``cause`` when set."""

    def __init__(self, message: str, cause: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.cause = cause


_POOL_LOCK = threading.Lock()
_POOL: Dict[Tuple[Any, ...], VenvWorker] = {}
_EVENTS: List[Dict[str, Any]] = []
_LOST: set = set()


def _event(kind: str, worker: VenvWorker, **fields: object) -> None:
    record = {
        "event": kind,
        "label": worker.label,
        "identity": [str(part) for part in worker.key[:-4]],
        "pid": worker.pid,
        "served": worker.served,
        "time": time.time(),
        **fields,
    }
    _EVENTS.append(record)
    logger.warning("venv worker %s: %s", kind, json.dumps(record, default=str))


def venv_worker_events(clear: bool = False) -> List[Dict[str, Any]]:
    """Return the restart, death and timeout events recorded in this process.

    Args:
        clear: Empty the record after reading it.

    Returns:
        One dict per event, oldest first: ``event``, ``label``, ``identity``, ``pid``, ``served``,
        ``time`` and the event's own fields.
    """
    with _POOL_LOCK:
        events = list(_EVENTS)
        if clear:
            _EVENTS.clear()
    return events


def venv_worker_stats() -> List[Dict[str, Any]]:
    """Return one row per live worker: ``label``, ``identity``, ``pid``, ``load_s``, ``served``, ``threads``."""
    with _POOL_LOCK:
        workers = list(_POOL.values())
    return [
        {
            "label": w.label,
            "identity": [str(part) for part in w.key[:-4]],
            "pid": w.pid,
            "load_s": w.load_s,
            "served": w.served,
            "threads": w.threads,
        }
        for w in workers
    ]


def shutdown_venv_workers() -> None:
    """End every worker this process started. A later call starts new ones."""
    with _POOL_LOCK:
        workers = list(_POOL.values())
        _POOL.clear()
    for worker in workers:
        worker.close()


atexit.register(shutdown_venv_workers)


def serve_in_venv(
    identity: Tuple[Any, ...],
    *,
    python: str,
    script: str,
    init: Dict[str, Any],
    request: Dict[str, Any],
    env: Dict[str, str],
    label: str,
    load_timeout_s: float,
    request_timeout_s: float,
) -> Dict[str, Any]:
    """Answer one request from the long-lived worker for ``identity``, starting it if needed.

    Args:
        identity: The model identity, normally ``(venv, model, revision, device, compute_type)``.
        python: The venv interpreter.
        script: Source defining ``load(init)`` and ``handle(state, request)``.
        init: The payload ``load`` receives; fixed for the worker's lifetime.
        request: The payload ``handle`` receives.
        env: The worker's environment; fixed for its lifetime.
        label: Name used in messages and events.
        load_timeout_s: Wall-clock ceiling on a start.
        request_timeout_s: Wall-clock ceiling on this request.

    Returns:
        The handler's result.

    Raises:
        RuntimeError: The worker could not start, died, timed out, or the handler raised a type
            other than ``ValueError``/``TypeError``.
        ValueError: ``load`` or the handler raised ``ValueError``.
        TypeError: ``load`` or the handler raised ``TypeError``.
    """
    key = (*identity, python, _digest(script), _digest(init), _digest(sorted(env.items())))
    with _POOL_LOCK:
        worker = _POOL.get(key)
        if worker is None:
            worker = VenvWorker(key, label)
            _POOL[key] = worker
    with worker.lock:
        if worker.pid is not None and not worker.alive:
            _event("died_idle", worker, returncode=worker.returncode(), tail=worker.tail()[-2000:])
            with _POOL_LOCK:
                replacement = VenvWorker(key, label)
                _POOL[key] = replacement
                _LOST.add(key)
            worker.close()
            with replacement.lock:
                return _serve(replacement, python, script, init, request, env, load_timeout_s, request_timeout_s)
        return _serve(worker, python, script, init, request, env, load_timeout_s, request_timeout_s)


def _serve(
    worker: VenvWorker,
    python: str,
    script: str,
    init: Dict[str, Any],
    request: Dict[str, Any],
    env: Dict[str, str],
    load_timeout_s: float,
    request_timeout_s: float,
) -> Dict[str, Any]:
    try:
        if worker.pid is None:
            worker.start(python, script, init, env, load_timeout_s)
            if worker.key in _LOST:
                _LOST.discard(worker.key)
                _event("restart", worker, load_s=worker.load_s)
            logger.info(
                "venv worker started: %s pid=%s load_s=%.3f threads=%s",
                worker.label,
                worker.pid,
                worker.load_s,
                worker.threads,
            )
        result, _seconds = worker.request(request, request_timeout_s)
        return result
    except _WorkerLost as lost:
        returncode = worker.returncode()
        worker.kill()
        _event("lost", worker, returncode=returncode, reason=str(lost).splitlines()[0])
        with _POOL_LOCK:
            if _POOL.get(worker.key) is worker:
                del _POOL[worker.key]
            _LOST.add(worker.key)
        if lost.cause is not None:
            raise lost.cause from lost
        raise RuntimeError(str(lost)) from None
