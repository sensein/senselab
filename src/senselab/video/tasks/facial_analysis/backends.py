"""Subprocess-venv backends: MediaPipe Face Landmarker (geometry) and py-feat Detectorv2 (affect).

Each backend runs a standalone worker script from this package directory inside its own venv. The
worker decodes the video itself, so frames never cross the process boundary; only the tile layout
goes in and per-tile arrays come back as one ``.npz``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from senselab.utils.data_structures.logging import logger
from senselab.utils.model_revision import cache_root
from senselab.utils.subprocess_venv import _clean_subprocess_env, ensure_venv, venv_python
from senselab.video.tasks.facial_analysis.results import GalleryLayout

_HERE = Path(__file__).resolve().parent

MEDIAPIPE_VENV = "mediapipe-face"
MEDIAPIPE_REQUIREMENTS = ["mediapipe==1.1.1", "av>=15", "numpy"]
MEDIAPIPE_PYTHON = "3.12"
MEDIAPIPE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MEDIAPIPE_MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"

PYFEAT_VENV = "pyfeat"
PYFEAT_REQUIREMENTS = ["py-feat==2.1.5", "av>=15", "numpy", "safetensors"]
PYFEAT_PYTHON = "3.12"
PYFEAT_MULTITASK_REPO = "py-feat/face_multitask_v2"
PYFEAT_MULTITASK_FILE = "face_multitask_v28.safetensors"
PYFEAT_MULTITASK_SHA = "d710b6f48edfed379ba422953ee2f123e250e698"
PYFEAT_RETINAFACE_REPO = "py-feat/retinaface_r34"
PYFEAT_RETINAFACE_FILE = "model.safetensors"
PYFEAT_RETINAFACE_SHA = "15191e03665fbf49e7610a1aee8d643f8a9119ca"


_THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def mediapipe_model_path() -> Path:
    """Download (once) and verify the Face Landmarker float16 bundle.

    Returns:
        The local ``.task`` file.

    Raises:
        RuntimeError: When the downloaded bytes do not match the pinned SHA-256.
    """
    target = cache_root() / "models" / "mediapipe" / f"face_landmarker_float16_v1_{MEDIAPIPE_MODEL_SHA256[:12]}.task"
    if target.exists() and _sha256(target) == MEDIAPIPE_MODEL_SHA256:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    with urllib.request.urlopen(MEDIAPIPE_MODEL_URL, timeout=120) as response, open(partial, "wb") as handle:  # noqa: S310
        handle.write(response.read())
    digest = _sha256(partial)
    if digest != MEDIAPIPE_MODEL_SHA256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"face_landmarker.task SHA-256 {digest} != pinned {MEDIAPIPE_MODEL_SHA256}")
    partial.replace(target)
    return target


def pyfeat_weight_paths() -> Dict[str, str]:
    """Download (once) the py-feat weights at their pinned commits.

    Returns:
        ``multitask_weights`` and ``retinaface_weights`` local paths.
    """
    from huggingface_hub import hf_hub_download

    from senselab.utils.dependencies import retry_on_transient_error

    def fetch(repo_id: str, filename: str, sha: str) -> str:
        return retry_on_transient_error(lambda: hf_hub_download(repo_id=repo_id, filename=filename, revision=sha))

    return {
        "multitask_weights": fetch(PYFEAT_MULTITASK_REPO, PYFEAT_MULTITASK_FILE, PYFEAT_MULTITASK_SHA),
        "retinaface_weights": fetch(PYFEAT_RETINAFACE_REPO, PYFEAT_RETINAFACE_FILE, PYFEAT_RETINAFACE_SHA),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tile_payload(layout: GalleryLayout) -> List[Dict[str, Any]]:
    return [{"tile_id": t.tile_id, "box": list(t.box), "masks": [list(m) for m in t.masks]} for t in layout.tiles]


def _run_worker(
    venv_dir: Path, script: str, spec: Dict[str, Any], timeout_s: Optional[float]
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Run one worker script on a spec and load its ``.npz``.

    Args:
        venv_dir: The backend venv.
        script: Worker file name in this package directory.
        spec: JSON-serialisable spec; ``out`` is filled in here.
        timeout_s: Wall-clock ceiling, ``None`` for none.

    Returns:
        The worker's status mapping and its arrays.

    Raises:
        RuntimeError: When the worker exits non-zero or prints no status.
    """
    with tempfile.TemporaryDirectory(prefix="senselab-face-") as tmp:
        spec = dict(spec, out=str(Path(tmp) / "out.npz"))
        spec_path = Path(tmp) / "spec.json"
        spec_path.write_text(json.dumps(spec))
        env = _clean_subprocess_env()
        if spec.get("threads"):
            for name in _THREAD_VARIABLES:
                env[name] = str(int(spec["threads"]))
        began = time.monotonic()
        result = subprocess.run(  # noqa: S603 - the interpreter is senselab's own venv
            [venv_python(venv_dir), str(_HERE / script), str(spec_path)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(f"{script} failed (exit {result.returncode}):\n{result.stderr[-4000:]}")
        status_lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
        if not status_lines:
            raise RuntimeError(f"{script} printed no status:\n{result.stderr[-4000:]}")
        status = json.loads(status_lines[-1])
        status["process_wall_s"] = time.monotonic() - began
        with np.load(spec["out"]) as data:
            arrays = {key: data[key] for key in data.files}
    logger.info("%s: %s frames in %.1f s", script, status.get("frames"), status["process_wall_s"])
    return status, arrays


def run_mediapipe(
    video_path: str, layout: GalleryLayout, config: Dict[str, Any], timeout_s: Optional[float] = None
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Run the Face Landmarker worker.

    Args:
        video_path: The video file.
        layout: Tiles to analyse.
        config: The ``geometry`` section of the defaults, with overrides applied.
        timeout_s: Wall-clock ceiling.

    Returns:
        Worker status and arrays keyed ``"<tile_id>/<name>"``.
    """
    venv_dir = ensure_venv(MEDIAPIPE_VENV, MEDIAPIPE_REQUIREMENTS, python_version=MEDIAPIPE_PYTHON)
    spec = {
        "video": str(video_path),
        "model": str(mediapipe_model_path()),
        "tiles": _tile_payload(layout),
        **{k: config[k] for k in config if k != "derivation"},
    }
    return _run_worker(venv_dir, "mediapipe_worker.py", spec, timeout_s)


def run_pyfeat(
    video_path: str, layout: GalleryLayout, config: Dict[str, Any], device: str, timeout_s: Optional[float] = None
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Run the py-feat Detectorv2 worker.

    Args:
        video_path: The video file.
        layout: Tiles to analyse.
        config: The ``affect`` section of the defaults, with overrides applied.
        device: ``"mps"``, ``"cuda"`` or ``"cpu"``.
        timeout_s: Wall-clock ceiling.

    Returns:
        Worker status and arrays keyed ``"<tile_id>/<name>"``.
    """
    venv_dir = ensure_venv(PYFEAT_VENV, PYFEAT_REQUIREMENTS, python_version=PYFEAT_PYTHON)
    spec = {
        "video": str(video_path),
        "tiles": _tile_payload(layout),
        "device": device,
        **pyfeat_weight_paths(),
        **{k: config[k] for k in config if k != "derivation"},
    }
    return _run_worker(venv_dir, "pyfeat_worker.py", spec, timeout_s)
