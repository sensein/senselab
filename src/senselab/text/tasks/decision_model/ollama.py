"""A pinned Ollama model, served for the length of one job.

An Ollama tag is mutable: ``nimble:latest`` names whatever the registry last pushed under it. A load
is therefore pinned by digest -- the manifest file, the image config and the model layer's blob --
and :func:`verify_pin` refuses to serve a store whose manifest or any layer does not carry them.
The blob is hashed in full once per host and the result remembered beside the cache, so a job does
not re-read 9.5 GB to learn what an earlier job on the same file already established.

:class:`OllamaServer` starts ``ollama serve`` on a free loopback port with the pinned model store,
offline (it never pulls), and stops it on exit.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Mapping

_REGISTRY = "registry.ollama.ai"
_MODEL_LAYER = "application/vnd.ollama.image.model"
_CHUNK = 1 << 24


class PinMismatchError(RuntimeError):
    """The local model store does not hold the pinned model."""


@dataclass(frozen=True)
class OllamaPin:
    """Which model to serve, and the digests that make it that model.

    Attributes:
        name: The library model name, e.g. ``nimble``.
        tag: The tag the manifest is filed under, e.g. ``9b``.
        blob_digest: ``sha256:<64 hex>`` of the model-weights layer.
        config_digest: ``sha256:<64 hex>`` of the manifest's image config.
        manifest_digest: ``sha256:<64 hex>`` of the manifest file itself, which names every layer --
            the system prompt and the parameters as well as the weights.
    """

    name: str
    tag: str
    blob_digest: str
    config_digest: str
    manifest_digest: str

    @property
    def model_id(self) -> str:
        """The provenance name: ``ollama:<name>:<tag>``."""
        return f"ollama:{self.name}:{self.tag}"


def _manifest_path(models_dir: Path, pin: OllamaPin) -> Path:
    return models_dir / "manifests" / _REGISTRY / "library" / pin.name / pin.tag


def _blob_path(models_dir: Path, digest: str) -> Path:
    return models_dir / "blobs" / digest.replace(":", "-")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def verify_pin(models_dir: Path, pin: OllamaPin, *, verified_dir: Path | None = None) -> Path:
    """Check that the store holds exactly the pinned model, and return its blob.

    Args:
        models_dir: The Ollama model store (``OLLAMA_MODELS``).
        pin: The model and its digests.
        verified_dir: Where a completed full-blob hash is remembered, keyed by digest, size and
            modification time; None hashes the blob every call.

    Returns:
        The path of the verified weights blob.

    Raises:
        PinMismatchError: If the manifest is missing, names other digests, or the blob's size or
            content does not match.
    """
    manifest_path = _manifest_path(models_dir, pin)
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise PinMismatchError(f"no readable manifest for {pin.model_id} at {manifest_path}: {error}") from error
    manifest_digest = f"sha256:{hashlib.sha256(manifest_bytes).hexdigest()}"
    if manifest_digest != pin.manifest_digest:
        raise PinMismatchError(f"{pin.model_id}: manifest is {manifest_digest}, pinned {pin.manifest_digest}")
    config_digest = str((manifest.get("config") or {}).get("digest") or "")
    if config_digest != pin.config_digest:
        raise PinMismatchError(f"{pin.model_id}: manifest config is {config_digest}, pinned {pin.config_digest}")
    layers = [layer for layer in manifest.get("layers") or () if layer.get("mediaType") == _MODEL_LAYER]
    if len(layers) != 1 or layers[0].get("digest") != pin.blob_digest:
        named = [layer.get("digest") for layer in layers]
        raise PinMismatchError(f"{pin.model_id}: model layer is {named}, pinned {pin.blob_digest}")
    for layer in manifest.get("layers") or ():
        if layer.get("mediaType") == _MODEL_LAYER:
            continue
        small = _blob_path(models_dir, str(layer.get("digest")))
        try:
            held = _sha256(small)
        except OSError as error:
            raise PinMismatchError(f"{pin.model_id}: layer {small} is missing: {error}") from error
        if held != layer.get("digest"):
            raise PinMismatchError(f"{pin.model_id}: layer {small.name} hashes to {held}")
    blob = _blob_path(models_dir, pin.blob_digest)
    try:
        stat = blob.stat()
    except OSError as error:
        raise PinMismatchError(f"{pin.model_id}: blob {blob} is missing: {error}") from error
    if stat.st_size != int(layers[0].get("size") or -1):
        raise PinMismatchError(f"{pin.model_id}: blob is {stat.st_size} bytes, manifest says {layers[0].get('size')}")
    marker = None
    if verified_dir is not None:
        marker = verified_dir / f"{pin.blob_digest.replace(':', '-')}.{stat.st_size}.{int(stat.st_mtime)}"
        if marker.exists():
            return blob
    actual = _sha256(blob)
    if actual != pin.blob_digest:
        raise PinMismatchError(f"{pin.model_id}: blob hashes to {actual}, pinned {pin.blob_digest}")
    if marker is not None:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(actual + "\n", encoding="utf-8")
    return blob


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class OllamaServer:
    """``ollama serve`` over a pinned store, for the length of a ``with`` block.

    Args:
        binary: The ``ollama`` executable.
        models_dir: The model store.
        pin: The model to serve; verified before the server starts.
        verified_dir: Passed to :func:`verify_pin`.
        log_path: Where the server's output goes; None discards it.
        startup_timeout_s: How long to wait for the server to answer.
        keep_alive: ``OLLAMA_KEEP_ALIVE``, how long the weights stay loaded between requests.
    """

    def __init__(
        self,
        binary: Path,
        models_dir: Path,
        pin: OllamaPin,
        *,
        verified_dir: Path | None = None,
        log_path: Path | None = None,
        startup_timeout_s: float = 120.0,
        keep_alive: str = "60m",
    ) -> None:
        """Hold the settings; nothing starts until ``__enter__``."""
        self.binary = Path(binary)
        self.models_dir = Path(models_dir)
        self.pin = pin
        self.verified_dir = verified_dir
        self.log_path = log_path
        self.startup_timeout_s = startup_timeout_s
        self.keep_alive = keep_alive
        self.host = ""
        self._process: subprocess.Popen[bytes] | None = None
        self._log: Any = None

    def __enter__(self) -> "OllamaServer":
        """Verify the pin, start the server and wait until it answers."""
        verify_pin(self.models_dir, self.pin, verified_dir=self.verified_dir)
        self.host = f"127.0.0.1:{_free_port()}"
        env = {
            **os.environ,
            "OLLAMA_MODELS": str(self.models_dir),
            "OLLAMA_HOST": self.host,
            "OLLAMA_KEEP_ALIVE": self.keep_alive,
            "OLLAMA_NOPRUNE": "1",
        }
        self._log = self.log_path.open("ab") if self.log_path is not None else subprocess.DEVNULL
        self._process = subprocess.Popen(  # noqa: S603 — a fixed binary with fixed arguments
            [str(self.binary), "serve"], env=env, stdout=self._log, stderr=subprocess.STDOUT
        )
        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(f"ollama serve exited with {self._process.returncode} before answering")
            try:
                with urllib.request.urlopen(f"http://{self.host}/api/version", timeout=5):  # noqa: S310
                    return self
            except (urllib.error.URLError, OSError):
                time.sleep(1.0)
        self.__exit__(None, None, None)
        raise RuntimeError(f"ollama serve did not answer within {self.startup_timeout_s} s")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop the server."""
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None
        if self._log not in (None, subprocess.DEVNULL):
            self._log.close()
        self._log = None


def ask_decisions(
    host: str,
    model: str,
    state: Mapping[str, Any],
    questions: Mapping[str, Mapping[str, Any]],
    *,
    seed: int,
    timeout_s: float = 300.0,
) -> dict[str, Any]:
    """Ask a decision model typed questions about a state, deterministically.

    Args:
        host: ``host:port`` of a running server.
        model: The model name the server knows, ``<name>:<tag>``.
        state: What the questions are about, as named fields.
        questions: Question name to its specification (``type``, ``instructions``, ``criteria``).
        seed: The sampling seed; temperature is 0.
        timeout_s: Per-request timeout.

    Returns:
        The response's ``answers`` mapping, question name to typed answer.

    Raises:
        RuntimeError: If the server answers without ``answers``.
    """
    body = json.dumps(
        {
            "model": model,
            "state": dict(state),
            "questions": {name: dict(spec) for name, spec in questions.items()},
            "options": {"temperature": 0, "seed": int(seed)},
        }
    ).encode("utf-8")
    request = urllib.request.Request(  # noqa: S310 — loopback only
        f"http://{host}/v1/systemone", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
        payload = json.loads(response.read())
    answers = payload.get("answers")
    if not isinstance(answers, Mapping):
        raise RuntimeError(f"decision model answered without answers: {str(payload)[:300]}")
    return dict(answers)
