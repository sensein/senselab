"""A pinned Ollama model, served for the length of one job.

An Ollama tag is mutable: ``nimble:latest`` names whatever the registry last pushed under it. A load
is therefore pinned by digest -- the manifest file, the image config and the model layer's blob --
and :func:`verify_pin` refuses to serve a store whose manifest or any layer does not carry them.
The blob is hashed in full once per host and the result remembered beside the cache, so a job does
not re-read 9.5 GB to learn what an earlier job on the same file already established.

:class:`OllamaServer` starts ``ollama serve`` on a free loopback port with the pinned model store,
offline (it never pulls), and stops it on exit. With ``require_gpu`` it loads the model before the
first question and refuses to serve one that is not wholly resident on the GPU.
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


class ServerUnusableError(RuntimeError):
    """The server answers but cannot serve the pinned model as required."""


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
        num_parallel: ``OLLAMA_NUM_PARALLEL``, how many requests the loaded model answers at once.
        require_gpu: Load the model on entry and raise :class:`ServerUnusableError` unless it is
            wholly resident on the GPU.
        load_timeout_s: How long the load on entry may take.
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
        num_parallel: int = 1,
        require_gpu: bool = True,
        load_timeout_s: float = 300.0,
    ) -> None:
        """Hold the settings; nothing starts until ``__enter__``."""
        self.binary = Path(binary)
        self.models_dir = Path(models_dir)
        self.pin = pin
        self.verified_dir = verified_dir
        self.log_path = log_path
        self.startup_timeout_s = startup_timeout_s
        self.keep_alive = keep_alive
        self.num_parallel = int(num_parallel)
        self.require_gpu = require_gpu
        self.load_timeout_s = load_timeout_s
        self.host = ""
        self._process: subprocess.Popen[bytes] | None = None
        self._log: Any = None

    @property
    def model(self) -> str:
        """The name the server knows the pinned model by, ``<name>:<tag>``."""
        return f"{self.pin.name}:{self.pin.tag}"

    def __enter__(self) -> "OllamaServer":
        """Verify the pin, start the server, wait until it answers and, with ``require_gpu``, load the model."""
        verify_pin(self.models_dir, self.pin, verified_dir=self.verified_dir)
        self.host = f"127.0.0.1:{_free_port()}"
        env = {
            "OLLAMA_VULKAN": "0",
            **os.environ,
            "OLLAMA_MODELS": str(self.models_dir),
            "OLLAMA_HOST": self.host,
            "OLLAMA_KEEP_ALIVE": self.keep_alive,
            "OLLAMA_NUM_PARALLEL": str(self.num_parallel),
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
                    break
            except (urllib.error.URLError, OSError):
                time.sleep(1.0)
        else:
            self.__exit__(None, None, None)
            raise RuntimeError(f"ollama serve did not answer within {self.startup_timeout_s} s")
        if self.require_gpu:
            try:
                self.load()
            except BaseException:
                self.__exit__(None, None, None)
                raise
        return self

    def log_tail(self) -> str:
        """The last non-empty line of the server's log, or an empty string."""
        if self.log_path is None or not self.log_path.exists():
            return ""
        if self._log not in (None, subprocess.DEVNULL):
            self._log.flush()
        with self.log_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 8192))
            lines = [line.strip() for line in handle.read().decode("utf-8", "replace").splitlines() if line.strip()]
        return lines[-1] if lines else ""

    def _unusable(self, reason: str) -> ServerUnusableError:
        tail = self.log_tail()
        return ServerUnusableError(f"{reason}; server log: {tail}" if tail else reason)

    def load(self) -> None:
        """Load the pinned model and require it wholly on the GPU.

        Raises:
            ServerUnusableError: If the load fails or any of the model is held outside GPU memory.
        """
        body = json.dumps({"model": self.model, "prompt": "", "keep_alive": self.keep_alive}).encode("utf-8")
        request = urllib.request.Request(  # noqa: S310 — loopback only
            f"http://{self.host}/api/generate", data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.load_timeout_s):  # noqa: S310
                pass
        except (urllib.error.URLError, OSError) as error:
            raise self._unusable(f"{self.model} did not load: {type(error).__name__}: {error}") from error
        self.require_resident()

    def require_resident(self) -> None:
        """Require the pinned model loaded and wholly in GPU memory, by the server's ``/api/ps``.

        Raises:
            ServerUnusableError: If it is not loaded, or ``size_vram`` is below ``size``.
        """
        try:
            with urllib.request.urlopen(f"http://{self.host}/api/ps", timeout=30) as response:  # noqa: S310
                loaded = json.loads(response.read()).get("models") or []
        except (urllib.error.URLError, OSError, ValueError) as error:
            raise self._unusable(f"/api/ps did not answer: {type(error).__name__}: {error}") from error
        held = [entry for entry in loaded if self.model in (entry.get("name"), entry.get("model"))]
        if not held:
            raise self._unusable(f"{self.model} is not loaded")
        size, vram = int(held[0].get("size") or 0), int(held[0].get("size_vram") or 0)
        if size <= 0 or vram < size:
            raise self._unusable(f"{self.model} is not wholly on the GPU: size_vram {vram} of size {size}")

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
