"""A ``vllm serve`` process in an `Env.server` env, owned by the calling process, and its HTTP clients.

`VllmServer` starts ``<env>/bin/vllm serve`` on a pinned model under a beating `Lease` named by the host and the
devices, so servers that share a card start one at a time. It waits for ``/health``, reads the served version and model,
and holds an `Attestation` (engine version, model identity, config digest) that every reply it returns carries
and that its `context_fields` put in the execution context. The server is launched through the supervisor in force
(`Supervisor`), in its own process group, which the supervisor's keeper kills if the owner dies; leaving the context
stops it (SIGTERM, then SIGKILL). A `svc` pool worker starts one and registers it for other processes to consume
(`execution.services`); a reply through the registry is checked against the attestation by `attest_reply`, as here.

The clients `chat` (with `content_part` for text, image, video and audio parts) and `complete` call a running
server's OpenAI-compatible endpoints over loopback and return the parsed JSON; they log nothing.
"""

import base64
import http.client
import json
import logging
import mimetypes
import os
import re
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from types import TracebackType
from typing import IO, Any

from senselab_worker.digest import identity_digest

from senselab_core.kernel._base import Frozen, NonEmptyStr
from senselab_core.kernel.identity import ResolvedModel
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.execution.supervisor import current
from senselab_core.runtime.execution_settings import JobLease
from senselab_core.runtime.launch import Launch, launch
from senselab_core.runtime.locks import TIMEOUT_S, Lease, LeaseUnavailable
from senselab_core.runtime.network import Network, network_variables

logger = logging.getLogger("senselab")

HOST = "127.0.0.1"
"""The only address a server listens on."""

ENGINE = "vllm"
CONFIG_TAG = "vllm-config@1"
"""The `identity_digest` tag of a server's configuration: its ordered server arguments, then its engine variables."""

REQUEST_TIMEOUT_S = 600.0
"""Default seconds a client waits for a reply."""

URL_SCHEMES = ("http", "https", "data", "file")
"""Schemes an image or video part may name directly; anything else is read as a local path."""

START_ERROR_KINDS = ("exited", "timeout", "version", "model")
"""Why a server failed to start: it exited before it was healthy, was not healthy in time, its ``/version`` or
``/v1/models`` probe failed, or it serves another model."""

ERROR_DETAIL_CHARS = 300
"""The most characters of a server's error message a raised error keeps."""

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

ENGINE_VARIABLES = (("flashinfer_sampler", "VLLM_USE_FLASHINFER_SAMPLER"),)
"""The settings passed as boolean environment variables of the server process, ``1`` or ``0``."""

POLL_S = 0.5
"""Seconds between health probes during start-up."""

STOP_TIMEOUT_S = 60.0
"""Seconds a server has to exit after SIGTERM before its group is killed."""

_NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class VllmStartError(RuntimeError):
    """A server that did not start, with the reason in `kind` (one of `START_ERROR_KINDS`)."""

    def __init__(self, kind: str, message: str) -> None:
        """Name the failure.

        Raises:
            ValueError: On an unknown kind.
        """
        if kind not in START_ERROR_KINDS:
            raise ValueError(f"unknown start error kind {kind!r}; known: {START_ERROR_KINDS}")
        super().__init__(f"{kind}: {message}")
        self.kind = kind


class AttestationMismatch(RuntimeError):
    """A started server attests something other than what its env lock, model and settings declare."""


class StaleAttestation(RuntimeError):
    """A reply the attested server did not give: the server is no longer running, or another model answered."""


class Attestation(Frozen):
    """What a running server is.

    `engine_version` and `served_model` are observed: what the server reports at `/version` and `/v1/models`.
    `model_identity` and `config_digest` are asserted: the pinned model and the settings the client launched it with.
    """

    engine_version: NonEmptyStr
    served_model: NonEmptyStr
    model_identity: NonEmptyStr
    config_digest: NonEmptyStr

    def driver(self) -> dict[str, str]:
        """Return the attestation as `ExecutionContext.driver` entries; `observed` names the observed ones."""
        return {
            "engine": ENGINE,
            "engine_version": self.engine_version,
            "served_model": self.served_model,
            "model_identity": self.model_identity,
            "config_digest": self.config_digest,
            "observed": "engine_version served_model",
        }


@dataclass(frozen=True)
class Attested:
    """A server's parsed reply and the attestation of the server that gave it."""

    body: Any
    attestation: Attestation


def attest_reply(body: Any, attestation: Attestation, label: str) -> Attested:  # noqa: ANN401 - a JSON value
    """Return `body` as a reply of the server attesting `attestation`; `label` names its model in the error.

    Raises:
        StaleAttestation: the reply names another model than the attested one.
    """
    served = body.get("model") if isinstance(body, dict) else None
    if served != attestation.served_model:
        raise StaleAttestation(f"a reply names another model than {label}")
    return Attested(body, attestation)


def expected_attestation(env: EnvHandle, model: ResolvedModel, settings: Mapping[str, Any]) -> Attestation:
    """Return what a server of `model` with `settings` in `env` must attest, without starting it.

    The engine version is the ``vllm==`` pin of the env's lock.

    Raises:
        VllmStartError: ``version``, the env has no lock or the lock pins no ``vllm``.
        ValueError: as `server_args`.
    """
    text = env.lock.read_text("utf-8") if env.lock is not None else ""
    pin = re.search(r"^vllm==([^\s\\;]+)", text, re.MULTILINE)
    if pin is None:
        raise VllmStartError("version", "the env's lock pins no vllm")
    return Attestation(
        engine_version=pin.group(1),
        served_model=_served(model)[0],
        model_identity=model.identity,
        config_digest=config_digest(settings),
    )


def server_launch(env: EnvHandle, variables: Mapping[str, str], devices: Sequence[str] | None = None) -> Launch:
    """Return how a server of `env` with `variables` (`server_variables`) on `devices` is started."""
    declared = dict(variables) if devices is None else {**variables, "CUDA_VISIBLE_DEVICES": ",".join(devices)}
    return launch(env.python, declared)


def attested_fields(env: EnvHandle, attestation: Attestation, launched: Launch) -> dict[str, Any]:
    """Return the env's `ExecutionContext` fields with `attestation` and the launch's digest added to `driver`."""
    fields = env.context_fields()
    return {**fields, "driver": {**fields["driver"], **attestation.driver(), "launch_env": launched.digest}}


def config_digest(settings: Mapping[str, Any]) -> str:
    """Return the digest a server with `settings` attests: of its `server_args`, then its `server_variables`.

    Raises:
        ValueError: as `server_args`.
    """
    variables = [f"{name}={value}" for name, value in server_variables(settings).items()]
    return identity_digest(CONFIG_TAG, [*server_args(settings), *variables])


def server_variables(settings: Mapping[str, Any]) -> dict[str, str]:
    """Return the environment variables `settings` set on the server process, in `ENGINE_VARIABLES` order.

    Raises:
        ValueError: as `server_args`.
    """
    server_args(settings)
    return {name: "1" if settings[key] else "0" for key, name in ENGINE_VARIABLES}


def server_args(settings: Mapping[str, Any]) -> list[str]:
    """Return the ``vllm serve`` arguments `settings` names, each as one ``--flag=value`` or switch.

    Args:
        settings: Every key of `SERVER_FLAGS`, `SWITCHES` and `ENGINE_VARIABLES` and no other;
            ``limit_mm_per_prompt`` is a mapping of per-modality counts, written as compact JSON with sorted keys.

    Raises:
        ValueError: A key is missing or unknown.
    """
    known = {key for key, _ in SERVER_FLAGS} | {key for key, _, _ in SWITCHES} | {key for key, _ in ENGINE_VARIABLES}
    if set(settings) != known:
        raise ValueError(f"vLLM settings lack {sorted(known - set(settings))} and add {sorted(set(settings) - known)}")
    args: list[str] = []
    for key, flag in SERVER_FLAGS:
        value = settings[key]
        if isinstance(value, Mapping):
            value = json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
        args.append(f"{flag}={value}")
    args += [on if settings[key] else off for key, on, off in SWITCHES]
    return args


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind((HOST, 0))
        return int(probe.getsockname()[1])


def _probe(url: str, key: str, timeout_s: float = 60.0) -> Any:  # noqa: ANN401 - a JSON value
    """``GET url`` and return the body's `key`; any failure is a ``version`` start error."""
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
            return bool(response.status == HTTPStatus.OK)
    except (OSError, http.client.HTTPException):
        return False


def _killpg(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except OSError:
        pass


def _served(model: ResolvedModel) -> tuple[str, list[str]]:
    """Return what ``vllm serve`` is given for `model` and the arguments that bind it to its identity."""
    if not model.pinned:
        raise ValueError(f"an unpinned {model.ref.source} model; a server loads only a pinned model")
    if model.ref.source == "hf":
        return model.ref.id, [f"--revision={model.identity}", f"--tokenizer-revision={model.identity}"]
    if model.ref.source == "local":
        return model.ref.id, []
    raise ValueError(f"a server loads a hub or local model, not a {model.ref.source} one")


class VllmServer:
    """One ``vllm serve`` of a pinned model in a built `Env.server` env, listening on loopback while open.

    On entry it takes a `Lease` on ``<start_lock_dir>/vllm-start-<host>-<devices>``, starts ``<env>/bin/vllm serve``
    in its own session (a hub model with ``--revision`` and ``--tokenizer-revision`` set to its commit;
    ``CUDA_VISIBLE_DEVICES`` set to `devices` when given), waits for ``/health`` and releases the lease; the lease
    beats while it is held. A start whose lease is lost before the launch is refused; one lost after it is logged,
    and the healthy server is kept, since the lease only orders starts. It then
    reads ``/version`` and checks that ``/v1/models`` serves the model. On exit, and on any start failure, the
    server's process group gets SIGTERM, then SIGKILL.

    Attributes:
        label: The model as a message names it: the hub reference, or a local model's identity prefix, never a path.
        url: ``http://127.0.0.1:<port>`` while the context is open, else None.
        attestation: The running server's `Attestation` while the context is open, else None.
    """

    def __init__(  # noqa: PLR0913 - the env, model and settings, then keyword-only placement and bounds
        self,
        env: EnvHandle,
        model: ResolvedModel,
        settings: Mapping[str, Any],
        *,
        start_lock_dir: Path,
        startup_timeout_s: float,
        start_margin_s: float,
        lease: JobLease,
        devices: Sequence[int | str] | None = None,
        log_dir: Path | None = None,
        network: Network = "online",
    ) -> None:
        """Describe the server; nothing starts until the context is entered.

        Args:
            env: The built env (`ensure_env` of an `Env.server` env) whose ``bin/vllm`` runs the server.
            model: The model to serve, pinned to a hub commit or a local weights digest.
            settings: The server settings `server_args` reads.
            start_lock_dir: Where the start lease lives.
            startup_timeout_s: Seconds the server has to become healthy once the lease is held.
            start_margin_s: The profile's `envs.server_start_margin_s`; a waiter for the start lease waits
                `startup_timeout_s` plus this plus the lease's stale bound and `locks.TIMEOUT_S`.
            lease: The profile's `lease`, under which the start lease beats.
            devices: The server's visible devices; when None the inherited environment is left alone and the lease
                is named by the inherited ``CUDA_VISIBLE_DEVICES`` (``all`` if unset).
            log_dir: Where the server's output goes, one mode-600 file per server; the owner's stderr when None.
            network: The profile's `network`; `offline` gives the server `OFFLINE_VARIABLES`.

        Raises:
            ValueError: `model` is unpinned or a url model, or `settings` is not exactly the known keys.
        """
        self.target, self._model_args = _served(model)
        self.label = str(model.ref) if model.ref.source == "hf" else f"local model {model.identity[:12]}"
        self.model = model
        self.args = server_args(settings)
        self.variables = server_variables(settings)
        self._digest = config_digest(settings)
        self.executable = Path(env.python).parent / "vllm"
        self.devices = None if devices is None else [str(device) for device in devices]
        self.start_lock_dir = Path(start_lock_dir)
        self.startup_timeout_s = float(startup_timeout_s)
        self.start_margin_s = float(start_margin_s)
        self.lease = lease
        self.log_dir = log_dir
        self.url: str | None = None
        self.attestation: Attestation | None = None
        self._env = env
        self._network = network
        self._server: subprocess.Popen[bytes] | None = None
        self._supervisor = current()
        self._log: IO[bytes] | None = None

    @property
    def config_digest(self) -> str:
        """The digest of the server arguments and variables, which the attestation carries."""
        return self._digest

    @property
    def lock_path(self) -> Path:
        """The start lease's resource, shared by every server on this host's devices."""
        if self.devices is None:
            devices = os.environ.get("CUDA_VISIBLE_DEVICES", "all").replace(",", "-") or "none"
        else:
            devices = "-".join(self.devices) or "none"
        return self.start_lock_dir / f"vllm-start-{socket.gethostname()}-{devices}"

    def __enter__(self) -> "VllmServer":
        """Start the server and attest it.

        Raises:
            VllmStartError: The server exited before it was healthy, was not healthy in time, or serves another
                model.
            LeaseTimeout, LeaseUnavailable: as `Lease`; `LeaseUnavailable` also when the start lease was lost before
                the launch.
        """
        self.start_lock_dir.mkdir(parents=True, exist_ok=True)
        try:
            stale_after = self.lease.stale_after_s
            wait = self.startup_timeout_s + self.start_margin_s + stale_after + TIMEOUT_S
            held = Lease(self.lock_path, stale_after=stale_after, timeout=wait)
            with held, held.keep_alive(self.lease.renew_fraction) as state:
                if state.lost or not held.prove():
                    raise LeaseUnavailable(held.path, f"the start lease {held.path} was lost before the server started")
                self._launch()
                self._await_health()
                if state.lost:
                    logger.warning(
                        "start lease %s was lost while the server started; the healthy server is kept", held.path
                    )
            self.attestation = self._attest()
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        """Stop the server."""
        self.stop()

    def context_fields(self) -> dict[str, Any]:
        """Return the env's `ExecutionContext` fields with the attestation added to `driver`.

        Raises:
            StaleAttestation: The server is not running.
        """
        return attested_fields(self._env, self._live(), self.launch)

    def chat(self, messages: Sequence[Mapping[str, Any]], *, sampling: Mapping[str, Any]) -> Attested:
        """`chat` with this server's model; the reply carries this server's attestation.

        Raises:
            StaleAttestation: The server is not running, or the reply names another model.
        """
        attestation = self._live()
        return attest_reply(chat(str(self.url), self.target, messages, sampling=sampling), attestation, self.label)

    def complete(self, prompt_token_ids: Sequence[int], *, sampling: Mapping[str, Any]) -> Attested:
        """`complete` with this server's model; the reply carries this server's attestation.

        Raises:
            StaleAttestation: The server is not running, or the reply names another model.
        """
        attestation = self._live()
        reply = complete(str(self.url), self.target, prompt_token_ids, sampling=sampling)
        return attest_reply(reply, attestation, self.label)

    def _live(self) -> Attestation:
        if self.attestation is None or self._server is None or self._server.poll() is not None:
            raise StaleAttestation(f"the vllm server of {self.label} is not running")
        return self.attestation

    @property
    def launch(self) -> Launch:
        """How the server process is started: its declared variables and devices over the allowlisted ambient ones."""
        return server_launch(self._env, {**self.variables, **network_variables(self._network)}, self.devices)

    def _launch(self) -> None:
        port = _free_port()
        started = self.launch
        command = started.script(self.executable, "serve", self.target, f"--host={HOST}", f"--port={port}")
        sink: int | IO[bytes] = 2
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            path = self.log_dir / f"vllm-{socket.gethostname()}-{os.getpid()}-{port}.log"
            self._log = os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600), "wb")
            sink = self._log
        self._supervisor = current()
        self._server = started.start(
            [*command, *self._model_args, *self.args],
            supervisor=self._supervisor,
            stdin=subprocess.DEVNULL,
            stdout=sink,
            stderr=subprocess.STDOUT,
        )
        self.url = f"http://{HOST}:{port}"

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

    def _attest(self) -> Attestation:
        version = _probe(f"{self.url}/version", "version")
        if not isinstance(version, str) or not version:
            raise VllmStartError("version", f"probe {self.url}/version: no version")
        data = _probe(f"{self.url}/v1/models", "data")
        if not isinstance(data, list) or not data or not isinstance(data[0], dict):
            raise VllmStartError("version", f"probe {self.url}/v1/models: no model entry")
        served = data[0].get("root")
        if served != self.target:
            raise VllmStartError("model", f"the server serves another model than {self.label}")
        return Attestation(
            engine_version=version,
            served_model=served,
            model_identity=self.model.identity,
            config_digest=self.config_digest,
        )

    def stop(self) -> None:
        """Stop the server's process group (SIGTERM, then SIGKILL) and release it; safe to call twice."""
        server, self._server = self._server, None
        if server is not None:
            _killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                pass
            _killpg(server.pid, signal.SIGKILL)
            self._supervisor.released(server.pid)
            server.wait()
        log, self._log = self._log, None
        if log is not None:
            log.close()
        self.url = None
        self.attestation = None


def _path_label(path: str | Path) -> str:
    """A media path as an error message may show it: its suffix, never its name."""
    return f"a {Path(path).suffix!r} file"


def _read(path: Path) -> str:
    """Return the file's bytes as base64; an unreadable file raises `ValueError` naming only its suffix."""
    try:
        return base64.b64encode(path.read_bytes()).decode()
    except OSError as exc:
        raise ValueError(f"cannot read {_path_label(path)}: {exc.strerror}") from None


def _data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0]
    if mime is None:
        raise ValueError(f"cannot tell the media type of {_path_label(path)} from its suffix")
    return f"data:{mime};base64,{_read(path)}"


def content_part(kind: str, source: str | Path, *, local_media_dir: str | Path | None = None) -> dict[str, Any]:
    """Return one content part of a chat message.

    Args:
        kind: ``text``, ``image``, ``video`` or ``audio``.
        source: The text for ``text``. For ``image`` and ``video``, an ``http(s)``, ``data:`` or ``file://`` URL is
            passed as is, and anything else is a local file. ``audio`` is always a local file.
        local_media_dir: The server's ``--allowed-local-media-path``. When given, a local image or video file goes
            in as a ``file://`` URL and must lie under it; otherwise it is sent inline as a base64 ``data:`` URL.
            Audio is always sent inline.

    Returns:
        A ``text``, ``image_url``, ``video_url`` or ``input_audio`` part; an audio part's ``format`` is the file's
        suffix.

    Raises:
        ValueError: An unknown kind, an audio URL, a file outside `local_media_dir`, a suffix with no known media
            type, or an unreadable file (named by its suffix only).
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
            return {
                "type": "input_audio",
                "input_audio": {"data": _read(path), "format": path.suffix.lstrip(".").lower()},
            }
        if local_media_dir is None:
            url = _data_url(path)
        else:
            root, resolved = Path(local_media_dir).resolve(), path.resolve()
            if root not in resolved.parents:
                raise ValueError(f"{_path_label(path)} is not under the server's local media directory")
            url = resolved.as_uri()
    key = f"{kind}_url"
    return {"type": key, key: {"url": url}}


def _refuse_reserved(sampling: Mapping[str, Any], reserved: Sequence[str]) -> None:
    taken = sorted(set(sampling) & set(reserved))
    if taken:
        raise ValueError(f"sampling may not set {taken}; the client sets them")


def _error_message(raw: bytes) -> str:
    """Return the message of a vLLM error body, at most `ERROR_DETAIL_CHARS` long.

    Only ``error.message``, a top-level ``message`` or ``detail`` is kept (a validation ``detail`` list as
    ``loc: msg`` items); ``input`` and every other field, which can echo the prompt, are dropped, as is a body
    that is not JSON.
    """
    try:
        body = json.loads(raw)
    except ValueError:
        return ""
    if not isinstance(body, dict):
        return ""
    error = body.get("error")
    message: Any = error.get("message") if isinstance(error, dict) else body.get("message", body.get("detail"))
    if isinstance(message, list):
        items = [item for item in message if isinstance(item, dict)]
        message = "; ".join(f"{'.'.join(map(str, item.get('loc') or ()))}: {item.get('msg', '')}" for item in items)
    return message[:ERROR_DETAIL_CHARS] if isinstance(message, str) else ""


def _post_json(url: str, body: Mapping[str, Any], timeout_s: float) -> Any:  # noqa: ANN401 - a JSON value
    """POST `body` without any proxy and parse the JSON reply; nothing is logged.

    A refused request re-raises its ``HTTPError`` with the server's short error message after the reason.
    """
    request = urllib.request.Request(  # noqa: S310 - a loopback http URL
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with _NO_PROXY.open(request, timeout=timeout_s) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        message = _error_message(exc.read())
        reason = f"{exc.reason}: {message}" if message else str(exc.reason)
        raise urllib.error.HTTPError(exc.url, exc.code, reason, exc.headers, None) from None


def chat(
    url: str,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    *,
    sampling: Mapping[str, Any],
    timeout_s: float = REQUEST_TIMEOUT_S,
) -> Any:  # noqa: ANN401 - a JSON value
    """``POST /v1/chat/completions``, not streamed, and return the parsed reply.

    Args:
        url: The server's base URL, such as `VllmServer.url`.
        model: The served model name.
        messages: Chat messages; a message's ``content`` is a string or a list of `content_part` parts.
        sampling: Further request fields (``temperature``, ``max_tokens``, ``seed``, ...).
        timeout_s: Seconds to wait for the reply.

    Raises:
        ValueError: `sampling` sets ``model``, ``messages`` or ``stream``.
        urllib.error.HTTPError: The server refused the request; the message carries its short error message.
    """
    _refuse_reserved(sampling, ("model", "messages", "stream"))
    body = {"model": model, "messages": list(messages), **sampling, "stream": False}
    return _post_json(f"{url}/v1/chat/completions", body, timeout_s)


def complete(
    url: str,
    model: str,
    prompt_token_ids: Sequence[int],
    *,
    sampling: Mapping[str, Any],
    timeout_s: float = REQUEST_TIMEOUT_S,
) -> Any:  # noqa: ANN401 - a JSON value
    """``POST /v1/completions`` with the prompt as token ids, not streamed, and return the parsed reply.

    Args:
        url: The server's base URL.
        model: The served model name.
        prompt_token_ids: The prompt, already tokenised (for example by the model's chat template).
        sampling: Further request fields (``max_tokens``, ``temperature``, ``stop_token_ids``, ...).
        timeout_s: Seconds to wait for the reply.

    Raises:
        ValueError: The prompt is empty or holds anything but non-negative ints, or `sampling` sets ``model``,
            ``prompt`` or ``stream``.
        urllib.error.HTTPError: The server refused the request; the message carries its short error message.
    """
    ids = list(prompt_token_ids)
    if not ids or not all(isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in ids):
        raise ValueError("prompt_token_ids must be a non-empty list of non-negative ints")
    _refuse_reserved(sampling, ("model", "prompt", "stream"))
    body = {"model": model, "prompt": ids, **sampling, "stream": False}
    return _post_json(f"{url}/v1/completions", body, timeout_s)
