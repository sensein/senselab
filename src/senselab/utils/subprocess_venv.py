"""Runtime subprocess venv manager for isolated backend dependencies.

Uses uv to create and manage isolated virtual environments for backends
that conflict with the core senselab installation. IPC uses a temp
directory with:
- manifest.json: call spec + JSON-serializable args
- *.safetensors: tensor data (via safetensors, already a dep)
- *.flac: audio data (lossless compressed via torchaudio)
- *.npy: numpy arrays

The subprocess runs a shim that unpacks, calls the function, and
packs the result using the same format.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from filelock import FileLock

logger = logging.getLogger("senselab")

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "senselab" / "venvs"


def _cache_dir() -> Path:
    """Return the directory for cached subprocess venvs."""
    cache = Path(os.environ.get("SENSELAB_VENV_CACHE", str(_DEFAULT_CACHE_DIR)))
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _find_uv() -> str:
    """Find the uv binary."""
    uv = shutil.which("uv")
    if uv:
        return uv
    for candidate in [
        Path.home() / ".local" / "bin" / "uv",
        Path.home() / ".cargo" / "bin" / "uv",
    ]:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        "uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    )


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
                check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError as exc:
            logger.error("Failed to create venv '%s': %s", name, exc.stderr)
            raise

        # Always include safetensors + numpy for IPC serialization
        all_reqs = [*requirements, "safetensors", "numpy"]
        try:
            subprocess.run(
                [uv, "pip", "install", "--python", str(venv_dir / "bin" / "python"), *all_reqs],
                check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError as exc:
            logger.error("Failed to install in venv '%s': %s", name, exc.stderr)
            raise

        marker.write_text(json.dumps({
            "requirements": sorted(requirements),
            "python_version": py_ver,
        }))
        logger.info("Venv '%s' ready at %s", name, venv_dir)
        return venv_dir


# ── Container pack/unpack (host side) ─────────────────────────────────


def _pack_value(key: str, value: object, data_dir: Path) -> dict:
    """Pack a single value, returning its manifest entry."""
    import numpy as np
    import torch
    from safetensors.torch import save_file

    if isinstance(value, torch.Tensor):
        path = data_dir / f"{key}.safetensors"
        save_file({"data": value.detach().cpu()}, str(path))
        return {"type": "tensor", "file": f"{key}.safetensors"}

    if isinstance(value, np.ndarray):
        path = data_dir / f"{key}.npy"
        np.save(str(path), value)
        return {"type": "ndarray", "file": f"{key}.npy"}

    # senselab Audio object
    if hasattr(value, "waveform") and hasattr(value, "sampling_rate"):
        import torchaudio

        path = data_dir / f"{key}.flac"
        torchaudio.save(str(path), value.waveform.cpu(), value.sampling_rate, format="flac")
        return {"type": "audio", "file": f"{key}.flac", "sr": value.sampling_rate}

    # JSON-serializable scalar/dict/list
    return {"type": "json", "value": value}


def _unpack_value(entry: dict, data_dir: Path) -> object:
    """Unpack a single value from its manifest entry."""
    import numpy as np
    import torch
    from safetensors.torch import load_file

    btype = entry["type"]
    if btype == "tensor":
        tensors = load_file(str(data_dir / entry["file"]))
        return tensors["data"]
    if btype == "ndarray":
        return np.load(str(data_dir / entry["file"]), allow_pickle=False)
    if btype == "audio":
        import torchaudio

        waveform, sr = torchaudio.load(str(data_dir / entry["file"]))
        return {"waveform": waveform, "sampling_rate": sr}
    return entry.get("value")


# ── Subprocess shim (embedded, runs in the isolated venv) ─────────────

_SHIM = r'''
import json, sys
from pathlib import Path

container = Path(sys.stdin.read().strip())
manifest = json.loads((container / "manifest.json").read_text())
data_dir = container / "data"

# Unpack args
import numpy as np
try:
    from safetensors.torch import load_file as _st_load
except ImportError:
    _st_load = None

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
    elif t == "json":
        args[key] = entry.get("value")

# Call function
call = manifest["call"]
mod = __import__(call["module"], fromlist=[call["function"]])
result = getattr(mod, call["function"])(**args)

# Pack result
ret = container / "return"
ret.mkdir(exist_ok=True)
rd = ret / "data"
rd.mkdir(exist_ok=True)
ret_entries = {}

def pack(name, obj):
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            from safetensors.torch import save_file
            save_file({"data": obj.detach().cpu()}, str(rd / f"{name}.safetensors"))
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
    return {"type": "json", "value": obj}

if isinstance(result, dict):
    for k, v in result.items():
        ret_entries[k] = pack(k, v)
else:
    ret_entries["__result__"] = pack("result", result)

(ret / "manifest.json").write_text(json.dumps({"entries": ret_entries}, default=str))
print("OK")
'''


# ── Public API ────────────────────────────────────────────────────────


def call_in_venv(
    name: str,
    requirements: list[str],
    module: str,
    function: str,
    args: Optional[dict[str, object]] = None,
    python_version: Optional[str] = None,
    timeout: int = 600,
) -> object:
    """Call a function in an isolated venv using container-based IPC.

    Data is serialized using efficient codecs:
    - torch.Tensor -> safetensors (fast, safe, HF standard)
    - numpy.ndarray -> .npy
    - senselab Audio -> .flac (lossless compressed)
    - everything else -> JSON

    Args:
        name: Venv identifier.
        requirements: Pip install specs.
        module: Python module path (e.g., "TTS.api").
        function: Function name.
        args: Keyword arguments. Tensors, arrays, and Audio objects
            are automatically serialized efficiently.
        python_version: Python version for the venv.
        timeout: Max execution time in seconds.

    Returns:
        The function's return value with blobs loaded back to native types.
    """
    venv_dir = ensure_venv(name, requirements, python_version)
    python = str(venv_dir / "bin" / "python")

    with tempfile.TemporaryDirectory(prefix="senselab-ipc-") as tmpdir:
        container = Path(tmpdir)
        data_dir = container / "data"
        data_dir.mkdir()

        # Pack args
        entries: dict[str, object] = {}
        for key, value in (args or {}).items():
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
        return unpacked
