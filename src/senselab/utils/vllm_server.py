"""A ``vllm serve`` process on loopback, held by a context manager.

:class:`VllmServer` starts the server under a start lock named by the host and the devices, so servers that
share a card start one at a time and each sizes its KV cache after its neighbours. It waits for ``/health``,
then checks the served version and the served snapshot commit. The server runs in its own process group; a
watchdog kills that group if the owner dies, and leaving the context stops it (SIGTERM, then SIGKILL).

:func:`server_args` builds the ordered server arguments, :func:`engine_identity` the engine's identity for a
cache key and provenance, and :func:`utilisation` the per-server memory share of a card.

Ported and generalised from the server half of ``sensein/senselab@feat/task-events`` (f2a806af),
``src/senselab/text/tasks/pii_detection/redaction_review_vllm.py``. Standard library only, so the module imports
in a lean environment.
"""

from __future__ import annotations

import fcntl
import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Mapping, Optional, Sequence, Union

HOST = "127.0.0.1"
"""The only address a server listens on."""

START_ERROR_KINDS = ("exited", "timeout", "version", "snapshot")
"""Why a server failed to start.

- ``exited``: it exited before it was healthy. This covers a KV cache that cannot hold ``max_model_len``, and
  also a port taken between :func:`_free_port` releasing it and the server binding it (a process outside the
  start lock can take it in that window).
- ``timeout``: it was not healthy in time.
- ``version``: it reports another version, or its ``/version`` or ``/v1/models`` probe failed (an HTTP error, a
  body that is not JSON, or JSON of the wrong shape).
- ``snapshot``: it serves another snapshot."""

CONTENT_LOGGING_FLAGS = ("--enable-log-requests", "--enable-log-outputs")
"""Server flags (bare or ``flag=value``) that would write request or output content to the log; refused."""

SERVER_FLAGS = (
    ("max_model_len", "--max-model-len"),
    ("max_num_seqs", "--max-num-seqs"),
    ("gpu_memory_utilization", "--gpu-memory-utilization"),
    ("kv_cache_dtype", "--kv-cache-dtype"),
    ("generation_config", "--generation-config"),
    ("limit_mm_per_prompt", "--limit-mm-per-prompt"),
)
"""The settings passed as valued server flags, in order."""

SWITCHES = (("enable_prefix_caching", "--enable-prefix-caching", "--no-enable-prefix-caching"),)
"""The settings passed as on/off server switches, after the valued flags."""

POLL_S = 0.5
"""Seconds between health probes during start-up."""

STOP_TIMEOUT_S = 60.0
"""Seconds a server has to exit after SIGTERM before its group is killed."""

_NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# Runs as ``python -c _WATCHDOG <owner pid> <server group>``, a child of the owner in a session of its own, so a
# signal to the owner's whole group does not reach it. It kills
# the server group once the owner is gone: on the SIGTERM PR_SET_PDEATHSIG sends when the owner dies, or when its
# parent is no longer the owner. A SIGTERM while the owner lives (the owning thread ended) is ignored.
_WATCHDOG = r"""
import ctypes, os, signal, sys, time

owner, group = int(sys.argv[1]), int(sys.argv[2])


def kill_group():
    try:
        os.killpg(group, signal.SIGKILL)
    except OSError:
        pass
    os._exit(0)


def on_term(signum, frame):
    if os.getppid() != owner:
        kill_group()


signal.signal(signal.SIGTERM, on_term)
try:
    ctypes.CDLL(None).prctl(1, signal.SIGTERM)
except Exception:
    pass
while os.getppid() == owner:
    time.sleep(1)
kill_group()
"""


class VllmStartError(RuntimeError):
    """A server that did not start, with the reason in :attr:`kind` (one of :data:`START_ERROR_KINDS`)."""

    def __init__(self, kind: str, message: str) -> None:
        """Name the failure.

        Args:
            kind: One of :data:`START_ERROR_KINDS`.
            message: What happened.

        Raises:
            ValueError: On an unknown kind.
        """
        if kind not in START_ERROR_KINDS:
            raise ValueError(f"unknown start error kind {kind!r}; known: {START_ERROR_KINDS}")
        super().__init__(f"{kind}: {message}")
        self.kind = kind


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind((HOST, 0))
        return int(probe.getsockname()[1])


def _probe(url: str, key: str, timeout_s: float = 60.0) -> Any:  # noqa: ANN401 — a JSON value
    """``GET url`` and return the body's ``key``; any failure is a ``version`` start error."""
    try:
        with _NO_PROXY.open(url, timeout=timeout_s) as response:
            body = json.loads(response.read())
        if not isinstance(body, dict):
            raise ValueError(f"the body is {type(body).__name__}, not an object")
        return body.get(key)
    except (OSError, http.client.HTTPException, ValueError) as exc:
        raise VllmStartError("version", f"probe {url} failed: {exc}") from exc


def _healthy(url: str) -> bool:
    try:
        with _NO_PROXY.open(f"{url}/health", timeout=5) as response:
            return response.status == 200
    except (OSError, http.client.HTTPException):
        return False


def _killpg(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except OSError:
        pass


class VllmServer:
    """One ``vllm serve`` over a model snapshot, listening on a free loopback port while the context is open.

    On entry it takes ``fcntl.flock`` on ``<start_lock_dir>/vllm-start-<host>-<devices>.lock``, starts
    ``<executable> serve <model_dir> --host 127.0.0.1 --port <port> *server_args`` in its own session (with
    ``CUDA_VISIBLE_DEVICES`` set to ``devices`` when they are given), waits for ``/health`` and releases the lock.
    It then checks that ``GET /version`` is ``expected_version`` and that the served model's root ends in
    ``model_dir``'s last component (the ``snapshots/<commit>`` name). On exit, and on any start failure,
    the server's process group gets SIGTERM, then SIGKILL.

    Attributes:
        url: ``http://127.0.0.1:<port>`` while the context is open, else None.
        port: The port while the context is open, else None.
    """

    def __init__(
        self,
        executable: Union[str, Path],
        model_dir: Union[str, Path],
        *,
        server_args: Sequence[str],
        devices: Optional[Sequence[Union[int, str]]] = None,
        start_lock_dir: Union[str, Path],
        expected_version: str,
        log_dir: Optional[Union[str, Path]] = None,
        startup_timeout_s: float = 1800,
    ) -> None:
        """Describe the server; nothing starts until the context is entered.

        Args:
            executable: The ``vllm`` console script to run (for example ``<venv>/bin/vllm``).
            model_dir: The model snapshot directory, named by its commit.
            server_args: The arguments after the port, such as :func:`server_args` builds.
            devices: The server's visible devices. When given, the server's ``CUDA_VISIBLE_DEVICES`` is set to
                them; when None, the inherited environment is left alone and the lock is named by the inherited
                ``CUDA_VISIBLE_DEVICES`` (``all`` if unset).
            start_lock_dir: Where the start lock file lives.
            expected_version: The vLLM version the server must report.
            log_dir: Where the server's output goes, one file per server with mode 600; the owner's stderr when
                None.
            startup_timeout_s: Seconds the server has to become healthy once the start lock is held.

        Raises:
            ValueError: If ``server_args`` asks the server to log request or output content.
        """
        for arg in server_args:
            for flag in CONTENT_LOGGING_FLAGS:
                if arg == flag or str(arg).startswith(flag + "="):
                    raise ValueError(f"{arg} would log request or output content; it is not allowed")
        self.executable = str(executable)
        self.model_dir = Path(model_dir)
        self.server_args = [str(arg) for arg in server_args]
        self.devices = None if devices is None else [str(device) for device in devices]
        self.start_lock_dir = Path(start_lock_dir)
        self.expected_version = expected_version
        self.log_dir = None if log_dir is None else Path(log_dir)
        self.startup_timeout_s = float(startup_timeout_s)
        self.url: Optional[str] = None
        self.port: Optional[int] = None
        self._server: Optional[subprocess.Popen[bytes]] = None
        self._watchdog: Optional[subprocess.Popen[bytes]] = None
        self._log: Optional[IO[bytes]] = None

    @property
    def lock_path(self) -> Path:
        """The start lock shared by every server on this host's ``devices``."""
        if self.devices is None:
            devices = os.environ.get("CUDA_VISIBLE_DEVICES", "all").replace(",", "-") or "none"
        else:
            devices = "-".join(self.devices) or "none"
        return self.start_lock_dir / f"vllm-start-{socket.gethostname()}-{devices}.lock"

    def __enter__(self) -> VllmServer:
        """Start the server and check it.

        Returns:
            This server, with :attr:`url` and :attr:`port` set.

        Raises:
            VllmStartError: If the server exited before it was healthy, was not healthy in time, or serves another
                version or snapshot.
        """
        self.start_lock_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.lock_path, "a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                self._launch()
                self._await_health()
            self._check_identity()
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Stop the server."""
        self.stop()

    def _launch(self) -> None:
        port = _free_port()
        command = [self.executable, "serve", str(self.model_dir), "--host", HOST, "--port", str(port)]
        env = dict(os.environ)
        if self.devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(self.devices)
        sink: Union[int, IO[bytes]] = 2
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            path = self.log_dir / f"vllm-{socket.gethostname()}-{os.getpid()}-{port}.log"
            self._log = os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600), "wb")
            sink = self._log
        self._server = subprocess.Popen(  # noqa: S603 — the caller's vllm executable
            [*command, *self.server_args],
            stdin=subprocess.DEVNULL,
            stdout=sink,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        self._watchdog = subprocess.Popen(  # noqa: S603 — this interpreter
            [sys.executable, "-c", _WATCHDOG, str(os.getpid()), str(self._server.pid)],
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        self.port, self.url = port, f"http://{HOST}:{port}"

    def _await_health(self) -> None:
        assert self._server is not None and self.url is not None
        deadline = time.monotonic() + self.startup_timeout_s
        while not _healthy(self.url):
            code = self._server.poll()
            if code is not None:
                raise VllmStartError("exited", f"vllm serve exited with {code} before it was healthy")
            if time.monotonic() > deadline:
                raise VllmStartError("timeout", f"vllm serve was not healthy within {self.startup_timeout_s:g}s")
            time.sleep(POLL_S)

    def _check_identity(self) -> None:
        assert self.url is not None
        version = _probe(f"{self.url}/version", "version")
        if version != self.expected_version:
            raise VllmStartError("version", f"the server is vllm {version}, not {self.expected_version}")
        data = _probe(f"{self.url}/v1/models", "data")
        if not isinstance(data, list) or not data or not isinstance(data[0], dict):
            raise VllmStartError("version", f"probe {self.url}/v1/models: no model entry")
        model = data[0]
        served = os.path.basename(os.path.normpath(str(model.get("root") or "")))
        if served != self.model_dir.name:
            raise VllmStartError("snapshot", f"the server serves snapshot {served!r}, not {self.model_dir.name!r}")

    def stop(self) -> None:
        """Stop the server's process group (SIGTERM, then SIGKILL) and its watchdog; safe to call twice."""
        server, self._server = self._server, None
        if server is not None:
            _killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                pass
            _killpg(server.pid, signal.SIGKILL)
            server.wait()
        watchdog, self._watchdog = self._watchdog, None
        if watchdog is not None:
            watchdog.kill()
            watchdog.wait()
        log, self._log = self._log, None
        if log is not None:
            log.close()
        self.url = self.port = None


def server_args(settings: Mapping[str, Any]) -> list[str]:
    """The ``vllm serve`` arguments a configuration names, after the port.

    Args:
        settings: A mapping with every key of :data:`SERVER_FLAGS` and :data:`SWITCHES`; ``limit_mm_per_prompt``
            is a mapping of per-modality counts.

    Returns:
        The arguments, in a fixed order; a mapping value is written as compact JSON with sorted keys.

    Raises:
        KeyError: If a key is missing.
    """
    args: list[str] = []
    for key, flag in SERVER_FLAGS:
        value = settings[key]
        if isinstance(value, Mapping):
            value = json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
        args += [flag, str(value)]
    for key, on, off in SWITCHES:
        args.append(on if settings[key] else off)
    return args


def engine_identity(
    version: str, server_args: Sequence[str], *, concurrency: int, batch_size: int, servers_per_card: int
) -> dict[str, Any]:
    """What a vLLM engine is, as a cache key and provenance carry it.

    Args:
        version: The vLLM version.
        server_args: The server arguments.
        concurrency: Requests in flight per server.
        batch_size: Items per request batch.
        servers_per_card: Servers sharing one card.

    Returns:
        ``name``, ``version``, ``server_args`` (a copy), ``concurrency``, ``batch_size`` and ``servers_per_card``.
    """
    return {
        "name": "vllm",
        "version": version,
        "server_args": list(server_args),
        "concurrency": concurrency,
        "batch_size": batch_size,
        "servers_per_card": servers_per_card,
    }


def utilisation(servers_per_card: int) -> float:
    """The ``--gpu-memory-utilization`` of each of ``servers_per_card`` servers on one card.

    Args:
        servers_per_card: Servers sharing the card, at least one.

    Returns:
        ``floor(90 / servers_per_card) / 100``.

    Raises:
        ValueError: If fewer than one server.
    """
    if servers_per_card < 1:
        raise ValueError(f"servers_per_card must be at least 1, not {servers_per_card}")
    return (90 // servers_per_card) / 100
