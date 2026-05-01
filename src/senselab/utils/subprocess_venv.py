"""Runtime subprocess venv manager for isolated backend dependencies.

Uses uv to create and manage isolated virtual environments for backends
that conflict with the core senselab installation. IPC uses a temp
directory with:
- manifest.json: call spec + JSON-serializable args + file metadata
- *.safetensors: tensor data (via safetensors, already a dep)
- *.flac: audio data (lossless compressed via torchaudio)
- *.npy: numpy arrays

File references include optional integrity metadata:
- checksum (SHA-256) for verifying data integrity
- readonly flag to prevent in-place modification
- file locks with heartbeat for concurrent access safety

Safety features are configurable via ``safe_mode`` to minimize
overhead for simple single-process workflows.
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from filelock import FileLock

logger = logging.getLogger("senselab")

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "senselab" / "venvs"


# ── File reference with integrity metadata ────────────────────────────


@dataclass
class FileRef:
    """A file reference with optional integrity and concurrency metadata.

    Use this to wrap file paths passed to ``call_in_venv`` when you need
    checksum verification, read-only enforcement, or file locking.

    For simple workflows, pass raw ``Path`` objects instead — no overhead.

    Args:
        path: Path to the file.
        readonly: If True, the subprocess receives a read-only copy or
            is instructed not to modify the file in-place. Default True.
        checksum: If True, compute SHA-256 before sending and verify
            after receiving. Catches corruption or unintended mutation.
        lock: If True, acquire a file lock (with heartbeat) for the
            duration of the subprocess call. Prevents parallel processes
            from mutating the file.
        lock_timeout: Max seconds to wait for the lock. Default 300.
    """

    path: Path
    readonly: bool = True
    checksum: bool = False
    lock: bool = False
    lock_timeout: int = 300
    _computed_hash: Optional[str] = field(default=None, repr=False)

    def compute_checksum(self) -> str:
        """Compute SHA-256 of the file."""
        h = hashlib.sha256()
        with open(self.path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        self._computed_hash = h.hexdigest()
        return self._computed_hash

    def verify_checksum(self) -> bool:
        """Verify the file matches the previously computed checksum."""
        if self._computed_hash is None:
            raise ValueError("No checksum computed yet — call compute_checksum() first")
        current = hashlib.sha256()
        with open(self.path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                current.update(chunk)
        return current.hexdigest() == self._computed_hash

    def to_manifest(self) -> dict:
        """Serialize metadata for the IPC manifest."""
        entry: dict = {
            "type": "fileref",
            "path": str(self.path),
            "readonly": self.readonly,
        }
        if self.checksum and self._computed_hash:
            entry["checksum"] = self._computed_hash
        return entry


class _FileLockWithHeartbeat:
    """A file lock that touches a heartbeat file while held.

    Other processes can check the heartbeat to distinguish a live lock
    holder from a crashed one.
    """

    def __init__(self, path: Path, timeout: int = 300, heartbeat_interval: int = 15) -> None:
        self._lock = FileLock(str(path.with_suffix(".lock")), timeout=timeout)
        self._heartbeat_path = path.with_suffix(".heartbeat")
        self._interval = heartbeat_interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _beat(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._heartbeat_path.touch()
            except OSError:
                pass

    def __enter__(self) -> "_FileLockWithHeartbeat":
        self._lock.acquire()
        self._stop.clear()
        self._thread = threading.Thread(target=self._beat, daemon=True)
        self._thread.start()
        self._heartbeat_path.touch()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        try:
            self._heartbeat_path.unlink(missing_ok=True)
        except OSError:
            pass
        self._lock.release()


def _cache_dir() -> Path:
    """Return the directory for cached subprocess venvs."""
    cache = Path(os.environ.get("SENSELAB_VENV_CACHE", str(_DEFAULT_CACHE_DIR)))
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _find_uv() -> str:
    """Find the uv binary, auto-installing if not present.

    Checks PATH and common install locations. If uv is not found,
    installs it automatically (needed for environments like Google Colab
    where uv is not pre-installed).
    """
    uv = shutil.which("uv")
    if uv:
        return uv
    for candidate in [
        Path.home() / ".local" / "bin" / "uv",
        Path.home() / ".cargo" / "bin" / "uv",
    ]:
        if candidate.is_file():
            return str(candidate)

    # Auto-install uv (e.g., on Google Colab or fresh environments)
    logger.info("uv not found — installing automatically...")
    result = subprocess.run(
        ["pip", "install", "uv"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode == 0:
        uv = shutil.which("uv")
        if uv:
            return uv
    raise FileNotFoundError("uv not found and auto-install failed. Install with: pip install uv")


# ── Venv management ──────────────────────────────────────────────────


def ensure_venv(
    name: str,
    requirements: list[str],
    python_version: Optional[str] = None,
) -> Path:
    """Create or reuse an isolated virtual environment.

    Args:
        name: Unique identifier for this venv (e.g., "coqui", "ppgs").
        requirements: List of pip install specs (e.g., ["coqui-tts~=0.27"]).
        python_version: Python version (e.g., "3.11"). Defaults to current.

    Returns:
        Path to the venv directory.
    """
    venv_dir = _cache_dir() / name
    lock_path = _cache_dir() / f"{name}.lock"
    marker = venv_dir / ".senselab-installed"

    with FileLock(str(lock_path), timeout=600):
        if marker.is_file():
            stored = json.loads(marker.read_text())
            if stored.get("requirements") == sorted(requirements):
                logger.debug("Reusing existing venv: %s", venv_dir)
                return venv_dir

        uv = _find_uv()
        py_ver = python_version or f"{sys.version_info.major}.{sys.version_info.minor}"
        logger.info("Creating isolated venv '%s' with Python %s", name, py_ver)

        if venv_dir.exists():
            shutil.rmtree(venv_dir)

        try:
            subprocess.run(
                [uv, "venv", "--python", py_ver, str(venv_dir)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            logger.error("Failed to create venv '%s': %s", name, exc.stderr)
            raise

        # Always include IPC serialization deps (safetensors for tensors,
        # numpy for arrays, torchaudio for FLAC audio encoding)
        all_reqs = [*requirements, "safetensors", "numpy", "torchaudio"]
        try:
            subprocess.run(
                [uv, "pip", "install", "--python", venv_python(venv_dir), *all_reqs],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            logger.error("Failed to install in venv '%s': %s", name, exc.stderr)
            raise

        marker.write_text(
            json.dumps(
                {
                    "requirements": sorted(requirements),
                    "python_version": py_ver,
                }
            )
        )
        logger.info("Venv '%s' ready at %s", name, venv_dir)
        return venv_dir


def venv_python(venv_dir: Path) -> str:
    """Return the path to the Python interpreter inside a venv.

    Uses ``Scripts/python.exe`` on Windows, ``bin/python`` elsewhere.
    """
    if sys.platform == "win32":
        return str(venv_dir / "Scripts" / "python.exe")
    return str(venv_dir / "bin" / "python")


def _clean_subprocess_env() -> dict:
    """Return a copy of os.environ without keys that break subprocess venvs.

    Strips MPLBACKEND (matplotlib_inline backend not available in subprocesses)
    and other notebook-specific env vars that cause errors in isolated venvs.
    """
    return {k: v for k, v in os.environ.items() if k not in ("MPLBACKEND",)}


# ── Subprocess result parsing with error propagation ──────────────────


def parse_subprocess_result(result: "subprocess.CompletedProcess[str]", venv_label: str = "subprocess") -> dict:
    """Parse a subprocess result, raising the original exception type if it failed.

    Worker scripts should print JSON to stdout. If the JSON contains an
    ``"error"`` key with ``"type"`` and ``"message"``, the original exception
    is reconstructed and raised.

    Args:
        result: The completed subprocess result.
        venv_label: Label for error messages (e.g., "Coqui", "SPARC").

    Returns:
        Parsed JSON dict from the last line of stdout.

    Raises:
        ValueError, RuntimeError, etc.: Reconstructed from worker error JSON.
        RuntimeError: If the subprocess failed without structured error output.
    """
    if result.returncode != 0:
        # Try to extract structured error from stdout
        stdout_lines = (result.stdout or "").strip().splitlines()
        if stdout_lines:
            try:
                output = json.loads(stdout_lines[-1])
                if "error" in output:
                    err = output["error"]
                    exc_type = err.get("type", "RuntimeError")
                    exc_msg = err.get("message", "Unknown error")
                    # Reconstruct common exception types
                    exc_class = {"ValueError": ValueError, "TypeError": TypeError}.get(exc_type, RuntimeError)
                    raise exc_class(exc_msg)
            except json.JSONDecodeError:
                pass
        raise RuntimeError(f"{venv_label} venv failed:\n{result.stderr}")

    stdout_lines = (result.stdout or "").strip().splitlines()
    if not stdout_lines:
        raise RuntimeError(f"{venv_label} venv produced no output")
    return json.loads(stdout_lines[-1])


# ── Container pack/unpack (host side) ─────────────────────────────────


def _pack_value(key: str, value: object, data_dir: Path) -> dict:
    """Pack a single value into the container, returning its manifest entry.

    Codec selection by type:
    - FileRef → path reference with integrity metadata
    - torch.Tensor → safetensors (fast, safe, HF standard)
    - numpy.ndarray → .npy (native numpy)
    - senselab Audio → .flac (lossless compressed audio)
    - senselab Video / Path to video → path reference (no copy)
    - PIL.Image → .png (lossless)
    - bytes/bytearray → .bin (raw binary)
    - Pydantic BaseModel → .json (via model_dump_json)
    - everything else → JSON
    """
    import numpy as np
    import torch
    from safetensors.torch import save_file

    # FileRef → path reference with integrity metadata
    if isinstance(value, FileRef):
        if value.checksum:
            value.compute_checksum()
        return value.to_manifest()

    # torch.Tensor → safetensors
    if isinstance(value, torch.Tensor):
        path = data_dir / f"{key}.safetensors"
        save_file({"data": value.detach().cpu()}, str(path))
        return {"type": "tensor", "file": f"{key}.safetensors"}

    # numpy.ndarray → .npy
    if isinstance(value, np.ndarray):
        path = data_dir / f"{key}.npy"
        np.save(str(path), value)
        return {"type": "ndarray", "file": f"{key}.npy"}

    # senselab Audio (has waveform + sampling_rate) → FLAC
    if hasattr(value, "waveform") and hasattr(value, "sampling_rate"):
        import torchaudio

        path = data_dir / f"{key}.flac"
        torchaudio.save(str(path), value.waveform.cpu(), value.sampling_rate, format="flac")
        return {"type": "audio", "file": f"{key}.flac", "sr": value.sampling_rate}

    # senselab Video or file path → pass path reference (no copy)
    if hasattr(value, "_file_path") and getattr(value, "_file_path", None) is not None:
        return {"type": "path", "value": str(value._file_path)}
    if isinstance(value, Path):
        return {"type": "path", "value": str(value)}

    # PIL Image → PNG (lossless)
    if type(value).__module__.startswith("PIL") or type(value).__name__ == "Image":
        path = data_dir / f"{key}.png"
        getattr(value, "save")(str(path), format="PNG")
        return {"type": "image", "file": f"{key}.png"}

    # bytes/bytearray → raw binary
    if isinstance(value, (bytes, bytearray)):
        path = data_dir / f"{key}.bin"
        path.write_bytes(value)
        return {"type": "binary", "file": f"{key}.bin"}

    # Pydantic BaseModel → JSON via model_dump
    if hasattr(value, "model_dump_json"):
        return {
            "type": "pydantic",
            "model_class": f"{type(value).__module__}.{type(value).__name__}",
            "value": json.loads(value.model_dump_json()),
        }  # type: ignore[union-attr]

    # JSON-serializable fallback
    return {"type": "json", "value": value}


def _unpack_value(entry: dict, data_dir: Path) -> object:
    """Unpack a single value from its manifest entry."""
    import numpy as np
    import torch
    from safetensors.torch import load_file

    btype = entry["type"]
    if btype == "tensor":
        return load_file(str(data_dir / entry["file"]))["data"]
    if btype == "ndarray":
        return np.load(str(data_dir / entry["file"]), allow_pickle=False)
    if btype == "audio":
        import torchaudio

        waveform, sr = torchaudio.load(str(data_dir / entry["file"]))
        return {"waveform": waveform, "sampling_rate": sr}
    if btype == "path":
        return Path(entry["value"])
    if btype == "fileref":
        ref_path = Path(entry["path"])
        if entry.get("checksum"):
            ref = FileRef(path=ref_path, checksum=True)
            ref._computed_hash = entry["checksum"]
            if not ref.verify_checksum():
                raise ValueError(f"Checksum mismatch for {ref_path} — file was modified during transfer")
        return ref_path
    if btype == "image":
        from PIL import Image

        return Image.open(str(data_dir / entry["file"]))
    if btype == "binary":
        return (data_dir / entry["file"]).read_bytes()
    if btype == "pydantic":
        # Caller is responsible for reconstructing the model
        return entry.get("value")
    return entry.get("value")


# ── Subprocess shim (embedded, runs in the isolated venv) ─────────────

_SHIM = r"""
import json, sys
from pathlib import Path
import numpy as np

container = Path(sys.stdin.read().strip())
manifest = json.loads((container / "manifest.json").read_text())
data_dir = container / "data"

try:
    from safetensors.torch import load_file as _st_load, save_file as _st_save
except ImportError:
    _st_load = _st_save = None

# ── Unpack args ──
args = {}
for key, entry in manifest.get("entries", {}).items():
    t = entry["type"]
    if t == "tensor" and _st_load:
        args[key] = _st_load(str(data_dir / entry["file"]))["data"]
    elif t == "ndarray":
        args[key] = np.load(str(data_dir / entry["file"]), allow_pickle=False)
    elif t == "audio":
        import torchaudio
        wf, sr = torchaudio.load(str(data_dir / entry["file"]))
        args[key] = {"waveform": wf, "sampling_rate": sr}
    elif t == "path":
        args[key] = Path(entry["value"])
    elif t == "fileref":
        args[key] = Path(entry["path"])
    elif t == "image":
        from PIL import Image
        args[key] = Image.open(str(data_dir / entry["file"]))
    elif t == "binary":
        args[key] = (data_dir / entry["file"]).read_bytes()
    elif t == "pydantic":
        args[key] = entry.get("value")  # passed as dict; callee reconstructs if needed
    else:
        args[key] = entry.get("value")

# ── Call function ──
call = manifest["call"]
mod = __import__(call["module"], fromlist=[call["function"]])
result = getattr(mod, call["function"])(**args)

# ── Pack result ──
ret = container / "return"
ret.mkdir(exist_ok=True)
rd = ret / "data"
rd.mkdir(exist_ok=True)
ret_entries = {}

def pack(name, obj):
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            if _st_save:
                _st_save({"data": obj.detach().cpu()}, str(rd / f"{name}.safetensors"))
                return {"type": "tensor", "file": f"{name}.safetensors"}
    except ImportError:
        pass
    if isinstance(obj, np.ndarray):
        np.save(str(rd / f"{name}.npy"), obj)
        return {"type": "ndarray", "file": f"{name}.npy"}
    if hasattr(obj, "waveform") and hasattr(obj, "sampling_rate"):
        import torchaudio
        torchaudio.save(str(rd / f"{name}.flac"), obj.waveform.cpu(), obj.sampling_rate, format="flac")
        return {"type": "audio", "file": f"{name}.flac"}
    if isinstance(obj, Path):
        return {"type": "path", "value": str(obj)}
    if isinstance(obj, (bytes, bytearray)):
        (rd / f"{name}.bin").write_bytes(obj)
        return {"type": "binary", "file": f"{name}.bin"}
    if hasattr(obj, "save") and hasattr(obj, "mode"):  # PIL Image
        (rd / f"{name}.png").parent.mkdir(exist_ok=True)
        obj.save(str(rd / f"{name}.png"), format="PNG")
        return {"type": "image", "file": f"{name}.png"}
    return {"type": "json", "value": obj}

if isinstance(result, dict):
    for k, v in result.items():
        ret_entries[k] = pack(k, v)
elif isinstance(result, (list, tuple)):
    for i, v in enumerate(result):
        ret_entries[f"__item_{i}__"] = pack(f"item_{i}", v)
    ret_entries["__is_sequence__"] = {"type": "json", "value": True}
    ret_entries["__sequence_len__"] = {"type": "json", "value": len(result)}
else:
    ret_entries["__result__"] = pack("result", result)

(ret / "manifest.json").write_text(json.dumps({"entries": ret_entries}, default=str))
print("OK")
"""


# ── Public API ────────────────────────────────────────────────────────


def call_in_venv(
    name: str,
    requirements: list[str],
    module: str,
    function: str,
    args: Optional[dict[str, object]] = None,
    python_version: Optional[str] = None,
    timeout: int = 600,
    safe_mode: bool = False,
) -> object:
    """Call a function in an isolated venv using container-based IPC.

    Data is serialized using efficient codecs:
    - torch.Tensor → safetensors (fast, safe, HF standard)
    - numpy.ndarray → .npy
    - senselab Audio → .flac (lossless compressed)
    - FileRef → path with checksum/lock metadata
    - PIL Image → .png, bytes → .bin, Pydantic → JSON
    - everything else → JSON

    Args:
        name: Venv identifier.
        requirements: Pip install specs.
        module: Python module path (e.g., "TTS.api").
        function: Function name.
        args: Keyword arguments. Tensors, arrays, Audio objects, and
            FileRef objects are handled automatically. Use FileRef to
            wrap paths that need checksum or lock protection.
        python_version: Python version for the venv.
        timeout: Max execution time in seconds.
        safe_mode: If True, automatically wrap all Path args as FileRef
            with checksum=True and readonly=True. Default False for
            minimal overhead in simple workflows.

    Returns:
        The function's return value with blobs loaded back to native types.
    """
    venv_dir = ensure_venv(name, requirements, python_version)
    python = venv_python(venv_dir)

    # In safe_mode, auto-wrap Path args as FileRef with checksum + readonly
    effective_args = dict(args or {})
    if safe_mode:
        for key, value in effective_args.items():
            if isinstance(value, Path) and value.is_file():
                effective_args[key] = FileRef(path=value, readonly=True, checksum=True)

    # Collect FileRef locks to hold during execution
    file_locks: list[_FileLockWithHeartbeat] = []
    for value in effective_args.values():
        if isinstance(value, FileRef) and value.lock:
            fl = _FileLockWithHeartbeat(value.path, timeout=value.lock_timeout)
            fl.__enter__()
            file_locks.append(fl)

    try:
        with tempfile.TemporaryDirectory(prefix="senselab-ipc-") as tmpdir:
            container = Path(tmpdir)
            data_dir = container / "data"
            data_dir.mkdir()

            # Pack args
            entries: dict[str, object] = {}
            for key, value in effective_args.items():
                entries[key] = _pack_value(key, value, data_dir)

            manifest = {
                "call": {"module": module, "function": function},
                "entries": entries,
            }
            (container / "manifest.json").write_text(json.dumps(manifest, default=str))

            # Execute in subprocess
            try:
                result = subprocess.run(
                    [python, "-c", _SHIM],
                    input=str(container),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"Venv '{name}' timed out after {timeout}s") from exc

            if result.returncode != 0:
                raise RuntimeError(f"Venv '{name}' failed:\n{result.stderr}")

            # Verify checksums on FileRef args after subprocess completes
            for value in effective_args.values():
                if isinstance(value, FileRef) and value.checksum and value.readonly:
                    if not value.verify_checksum():
                        raise ValueError(
                            f"File {value.path} was modified during subprocess execution (readonly=True was specified)"
                        )

            # Unpack result
            ret_dir = container / "return"
            if not ret_dir.exists():
                return None

            ret_manifest = json.loads((ret_dir / "manifest.json").read_text())
            ret_data = ret_dir / "data"

            unpacked: dict[str, object] = {}
            for key, entry in ret_manifest.get("entries", {}).items():
                unpacked[key] = _unpack_value(entry, ret_data)

            # Unwrap single result
            if len(unpacked) == 1 and "__result__" in unpacked:
                return unpacked["__result__"]

            # Reconstruct sequences
            if unpacked.get("__is_sequence__"):
                seq_len = int(str(unpacked.get("__sequence_len__", 0)))
                return [unpacked.get(f"__item_{i}__") for i in range(seq_len)]

            # Filter out internal keys
            return {k: v for k, v in unpacked.items() if not k.startswith("__")}
    finally:
        # Release all file locks
        for fl in file_locks:
            fl.__exit__(None, None, None)
