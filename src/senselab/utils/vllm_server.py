"""A ``vllm serve`` process on loopback, held by a context manager.

:class:`VllmServer` starts the server under a start lock named by the host and the devices, so servers that
share a card start one at a time and each sizes its KV cache after its neighbours. It waits for ``/health``,
then checks the served version and the served snapshot commit. The server runs in its own process group; a
watchdog kills that group if the owner dies, and leaving the context stops it (SIGTERM, then SIGKILL).

:func:`server_args` builds the ordered server arguments, :func:`engine_identity` the engine's identity for a
cache key and provenance, and :func:`utilisation` the per-server memory share of a card.

The clients :func:`chat` (with :func:`content_part` for text, image, video and audio parts), :func:`transcribe`
and :func:`complete` call a running server's OpenAI-compatible endpoints over loopback and return the parsed JSON;
they log nothing. :func:`ensure_vllm_venv` builds the server's venv from its committed lock.

Ported and generalised from the server half of ``sensein/senselab@feat/task-events`` (f2a806af),
``src/senselab/text/tasks/pii_detection/redaction_review_vllm.py``. Standard library only, so the module imports
in a lean environment.
"""

from __future__ import annotations

import base64
import fcntl
import http.client
import json
import mimetypes
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Mapping, Optional, Sequence, Union

HOST = "127.0.0.1"
"""The only address a server listens on."""

VLLM_VENV = "vllm-server"
VLLM_VERSION = "0.31.0"
VLLM_PYTHON = "3.12"
VLLM_REQUIREMENTS = [f"vllm=={VLLM_VERSION}"]
VLLM_MAX_CUDA = (13, 0)
"""The server's venv: its name, the one requirement, its Python and its CUDA index ceiling (vLLM 0.31 needs
CUDA 13). The committed lock ``data/venv_locks/vllm-server.txt`` is compiled from these."""

REQUEST_TIMEOUT_S = 600.0
"""Default seconds a client waits for a reply."""

URL_SCHEMES = ("http", "https", "data", "file")
"""Schemes an image or video part may name directly; anything else is read as a local path."""

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


def ensure_vllm_venv() -> Path:
    """Build or reuse the server's venv from its committed lock and return its ``vllm`` executable.

    The first call installs the venv (it needs the network); later calls reuse it without installing anything.

    Returns:
        ``<venv>/bin/vllm``, to pass to :class:`VllmServer`.

    Raises:
        VenvLockError: If the committed lock is missing or was compiled from other inputs.
    """
    from senselab.utils import subprocess_venv

    venv_dir = subprocess_venv.ensure_venv(
        VLLM_VENV, VLLM_REQUIREMENTS, python_version=VLLM_PYTHON, max_cuda_version=VLLM_MAX_CUDA
    )
    return Path(venv_dir) / "bin" / "vllm"


def _data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0]
    if mime is None:
        raise ValueError(f"cannot tell the media type of {path.name!r} from its suffix")
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def content_part(
    kind: str, source: Union[str, Path], *, local_media_dir: Optional[Union[str, Path]] = None
) -> dict[str, Any]:
    """One content part of a chat message.

    Args:
        kind: ``text``, ``image``, ``video`` or ``audio``.
        source: The text for ``text``. For ``image`` and ``video``, an ``http(s)``, ``data:`` or ``file://`` URL is
            passed as is, and anything else is a local file. ``audio`` is always a local file.
        local_media_dir: The server's ``--allowed-local-media-path``. When given, a local image or video file goes
            in as a ``file://`` URL and must lie under it; otherwise it is read and sent inline as a base64
            ``data:`` URL. Audio is always sent inline.

    Returns:
        A ``text``, ``image_url``, ``video_url`` or ``input_audio`` part. An audio part's ``format`` is the file's
        suffix.

    Raises:
        ValueError: On an unknown kind, an audio URL, a file outside ``local_media_dir``, or a suffix with no
            known media type.
    """
    if kind == "text":
        return {"type": "text", "text": str(source)}
    if kind not in ("image", "video", "audio"):
        raise ValueError(f"unknown content part kind {kind!r}; known: text, image, video, audio")
    if isinstance(source, str) and urllib.parse.urlsplit(source).scheme in URL_SCHEMES:
        if kind == "audio":
            raise ValueError("an audio part is sent inline from a local file, not by URL")
        url = source
    else:
        path = Path(source)
        if kind == "audio":
            data = base64.b64encode(path.read_bytes()).decode()
            return {"type": "input_audio", "input_audio": {"data": data, "format": path.suffix.lstrip(".").lower()}}
        if local_media_dir is None:
            url = _data_url(path)
        else:
            root, resolved = Path(local_media_dir).resolve(), path.resolve()
            if root not in resolved.parents:
                raise ValueError(f"{path} is not under the server's local media directory {local_media_dir}")
            url = resolved.as_uri()
    key = f"{kind}_url"
    return {"type": key, key: {"url": url}}


def _refuse_reserved(sampling: Mapping[str, Any], reserved: Sequence[str]) -> None:
    taken = sorted(set(sampling) & set(reserved))
    if taken:
        raise ValueError(f"sampling may not set {taken}; the client sets them")


def _send(request: urllib.request.Request, timeout_s: float) -> Any:  # noqa: ANN401 — a JSON value
    """Send a request without any proxy and parse the JSON reply; nothing is logged.

    A refused request re-raises its ``HTTPError`` with the server's decoded error body in the message.
    """
    try:
        with _NO_PROXY.open(request, timeout=timeout_s) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise urllib.error.HTTPError(exc.url, exc.code, f"{exc.reason}: {detail}", exc.headers, None) from exc


def _post_json(url: str, body: Mapping[str, Any], timeout_s: float) -> Any:  # noqa: ANN401 — a JSON value
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST"
    )
    return _send(request, timeout_s)


def chat(
    url: str,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    *,
    sampling: Mapping[str, Any],
    timeout_s: float = REQUEST_TIMEOUT_S,
) -> Any:  # noqa: ANN401 — a JSON value
    """``POST /v1/chat/completions``, not streamed.

    Args:
        url: The server's base URL, such as :attr:`VllmServer.url`.
        model: The served model name.
        messages: Chat messages; a message's ``content`` is a string or a list of :func:`content_part` parts.
        sampling: Further request fields (``temperature``, ``max_tokens``, ``seed``, ...).
        timeout_s: Seconds to wait for the reply.

    Returns:
        The parsed reply.

    Raises:
        ValueError: If ``sampling`` sets ``model``, ``messages`` or ``stream``.
        urllib.error.HTTPError: If the server refuses the request; the message carries the server's error body.
    """
    _refuse_reserved(sampling, ("model", "messages", "stream"))
    body = {"model": model, "messages": list(messages), **sampling, "stream": False}
    return _post_json(f"{url}/v1/chat/completions", body, timeout_s)


_UNSAFE_HEADER_CHARS = ('"', "\r", "\n")


def _form_value(name: str, value: Any) -> str:  # noqa: ANN401 — a scalar form value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        return str(value)
    raise TypeError(f"form field {name!r} must be a string, number or bool, not {type(value).__name__}")


def transcribe(
    url: str,
    model: str,
    audio_path: Union[str, Path],
    *,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    sampling: Mapping[str, Any],
    timeout_s: float = REQUEST_TIMEOUT_S,
) -> Any:  # noqa: ANN401 — a JSON value
    """``POST /v1/audio/transcriptions`` with the audio file as multipart form data.

    Args:
        url: The server's base URL.
        model: The served model name.
        audio_path: The audio file. It is sent as ``audio<suffix>``, so its name never leaves the host.
        language: The audio's language, sent only when given.
        prompt: A prompt for the transcription, sent only when given.
        sampling: Further scalar form fields (``temperature``, ``response_format``, ...); a bool is sent as
            ``true``/``false``. The reply is parsed as JSON, so ``response_format`` must be a JSON one.
        timeout_s: Seconds to wait for the reply.

    Returns:
        The parsed reply.

    Raises:
        ValueError: If ``sampling`` sets ``file``, ``model``, ``language`` or ``prompt``, or a field name or the
            file's suffix holds a quote, CR or LF.
        TypeError: If a ``sampling`` value is not a string, number or bool.
        urllib.error.HTTPError: If the server refuses the request; the message carries the server's error body.
    """
    _refuse_reserved(sampling, ("file", "model", "language", "prompt"))
    fields = {"model": model, "language": language, "prompt": prompt}
    form = [(name, value) for name, value in fields.items() if value is not None]
    form += [(name, _form_value(name, value)) for name, value in sampling.items()]
    path = Path(audio_path)
    filename = f"audio{path.suffix}"
    for text in [name for name, _ in form] + [filename]:
        if any(char in text for char in _UNSAFE_HEADER_CHARS):
            raise ValueError(f"{text!r} holds a quote, CR or LF, which a multipart header cannot carry")
    boundary = uuid.uuid4().hex
    chunks: list[bytes] = []
    for name, value in form:
        chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    chunks.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {mime}\r\n\r\n".encode()
    )
    chunks += [path.read_bytes(), f"\r\n--{boundary}--\r\n".encode()]
    request = urllib.request.Request(
        f"{url}/v1/audio/transcriptions",
        data=b"".join(chunks),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    return _send(request, timeout_s)


def complete(
    url: str,
    model: str,
    prompt_token_ids: Sequence[int],
    *,
    sampling: Mapping[str, Any],
    timeout_s: float = REQUEST_TIMEOUT_S,
) -> Any:  # noqa: ANN401 — a JSON value
    """``POST /v1/completions`` with the prompt as token ids, not streamed.

    Args:
        url: The server's base URL.
        model: The served model name.
        prompt_token_ids: The prompt, already tokenised (for example by the model's chat template).
        sampling: Further request fields (``max_tokens``, ``temperature``, ``stop_token_ids``, ...).
        timeout_s: Seconds to wait for the reply.

    Returns:
        The parsed reply.

    Raises:
        ValueError: If the prompt is empty or holds anything but non-negative ints, or ``sampling`` sets
            ``model``, ``prompt`` or ``stream``.
        urllib.error.HTTPError: If the server refuses the request; the message carries the server's error body.
    """
    ids = list(prompt_token_ids)
    if not ids or not all(isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in ids):
        raise ValueError("prompt_token_ids must be a non-empty list of non-negative ints")
    _refuse_reserved(sampling, ("model", "prompt", "stream"))
    body = {"model": model, "prompt": ids, **sampling, "stream": False}
    return _post_json(f"{url}/v1/completions", body, timeout_s)
