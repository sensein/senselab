"""Visualization for the Pose Estimation task (cv2-only)."""

import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from senselab.utils.data_structures.docker import docker_is_running
from senselab.video.data_structures.pose import ImagePose
from senselab.video.tasks.pose_estimation.utils import SENSELAB_KEYPOINT_MAPPING

# ---- cv2 is required on the host (as in the original code) ----
try:
    import cv2

    CV2_AVAILABLE = True
except ModuleNotFoundError:
    CV2_AVAILABLE = False

# ---- Docker config ----
DOCKER_IMAGE = "fabiocat93/mediapipe-deps:latest"


# =============================================================================
# Helpers
# =============================================================================


def _default_workdir() -> Path:
    """Folder that should contain mp_visualization_worker.py."""
    return Path(__file__).resolve().parent


def _ensure_worker_exists(workdir: Path) -> Path:
    """Ensure the Docker worker script is present under workdir."""
    worker = workdir / "mp_visualization_worker.py"
    if not worker.exists():
        raise FileNotFoundError(
            f"Worker script not found at {worker}. Place mp_visualization_worker.py in the mounted workdir."
        )
    return worker


def _require_cv2() -> None:
    if not CV2_AVAILABLE:
        raise ModuleNotFoundError(
            "`opencv-python` is required for visualization. Install via `pip install 'senselab[video]'`."
        )


def _write_temp_png_under(workdir: Path, rgb: np.ndarray) -> Path:
    """Write an RGB image to a temp PNG under workdir using cv2 only."""
    _require_cv2()
    if not (isinstance(rgb, np.ndarray) and rgb.ndim == 3 and rgb.shape[2] == 3):
        raise ValueError("`rgb` must be an RGB numpy array of shape (H, W, 3).")
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    tmp = Path(tempfile.mkstemp(prefix="mp_vis_in_", suffix=".png", dir=workdir)[1])
    if not cv2.imwrite(str(tmp), bgr):
        try:
            Path(tmp).unlink()
        except Exception:
            pass
        raise RuntimeError("Failed to write temp image via OpenCV.")
    return tmp


def _poses_json_for_worker(pose_image: ImagePose) -> Dict[str, List[List[Dict[str, float]]]]:
    """Convert poses to the format expected by the Docker worker script.

    Args:
        pose_image: ImagePose object containing detected poses.

    Returns:
        Dictionary with a single key "poses" mapping to a list of poses,
        where each pose is a list of dicts with keys x, y, z, v
        (x, y, z coordinates and visibility).
    """
    names_in_order = list(SENSELAB_KEYPOINT_MAPPING.values())
    poses: List[List[Dict[str, float]]] = []

    for person in pose_image.individuals:
        lm_dict = person.normalized_landmarks
        row: List[Dict[str, float]] = []
        for name in names_in_order:
            lm = lm_dict.get(name)
            conf = float(getattr(lm, "confidence", 1.0)) if lm is not None else 0.0
            if lm is not None and conf > 0.5:
                x = float(getattr(lm, "x", 0.0))
                y = float(getattr(lm, "y", 0.0))
                z = float(getattr(lm, "z", 0.0))
                v = float(getattr(lm, "visibility", conf))
                row.append({"x": x, "y": y, "z": z, "v": v})
            else:
                row.append({"x": 0.0, "y": 0.0, "z": 0.0, "v": 0.0})
        poses.append(row)

    return {"poses": poses}


def _run_docker_visualizer(
    rgb: np.ndarray,
    poses_payload: Dict[str, List[List[Dict[str, float]]]],
    workdir: Optional[Path] = None,
    image_name: str = DOCKER_IMAGE,
    timeout: Optional[int] = None,
) -> np.ndarray:
    """Run the Docker worker to visualize poses on the image.

    Args:
        rgb: Input RGB image as a numpy array.
        poses_payload: Poses in the format expected by the worker.
        workdir: Directory mounted into Docker (must contain the worker).
        image_name: Docker image tag to use for the worker.
        timeout: Optional timeout (seconds) for the Docker process.

    Returns:
        Visualized image as a numpy array.
    """
    if not docker_is_running():
        raise RuntimeError("Docker is not running and is required for visualization.")

    workdir = (workdir or _default_workdir()).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    worker = _ensure_worker_exists(workdir)

    in_img = _write_temp_png_under(workdir, rgb)
    in_json = Path(tempfile.mkstemp(prefix="mp_vis_pose_", suffix=".json", dir=workdir)[1])
    out_img = Path(tempfile.mkstemp(prefix="mp_vis_out_", suffix=".png", dir=workdir)[1])

    try:
        with open(in_json, "w", encoding="utf-8") as f:
            json.dump(poses_payload, f, separators=(",", ":"), ensure_ascii=False)

        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{str(workdir)}:/app",
            "-w",
            "/app",
            "-e",
            "MPLCONFIGDIR=/tmp",
            image_name,
            "python",
            Path(worker).name,  # worker lives in /app
            "--image",
            Path(in_img).name,  # pass filenames (relative to /app)
            "--poses",
            Path(in_json).name,
            "--out",
            Path(out_img).name,
        ]

        try:
            proc = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)
        except subprocess.CalledProcessError as e:
            pretty = " ".join(shlex.quote(c) for c in cmd)
            raise RuntimeError(
                f"Docker visualization failed.\nCommand: {pretty}\nSTDOUT:\n{e.stdout}\n\nSTDERR:\n{e.stderr}"
            ) from e
        except subprocess.TimeoutExpired as e:
            pretty = " ".join(shlex.quote(c) for c in cmd)
            raise TimeoutError(f"Docker visualization timed out.\nCommand: {pretty}") from e

        _require_cv2()
        bgr = cv2.imread(str(out_img), cv2.IMREAD_COLOR)
        if bgr is None:
            stdout = getattr(proc, "stdout", "")
            stderr = getattr(proc, "stderr", "")
            raise RuntimeError(
                f"Docker worker did not produce a readable output image.\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
            )
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    finally:
        # Best-effort cleanup
        for p in (in_img, in_json, out_img):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                # Consider logging this for diagnostics, e.g.,
                # logging.warning(f"Failed to clean up temp file {p}: {e}")
                pass


# =============================================================================
# Public API
# =============================================================================


def visualize(
    pose_image: ImagePose,
    output_path: Optional[str] = None,
    plot: bool = False,
    *,
    workdir: Optional[Path] = None,
    docker_image: str = DOCKER_IMAGE,
    timeout: Optional[int] = None,
) -> np.ndarray:
    """Visualize detected poses by drawing landmarks and connections on the image.

    Requires Docker to be running.

    Args:
        pose_image: Pose estimation result containing detected poses.
        output_path: Optional path to save the visualized image.
        plot: Whether to display the annotated image using matplotlib.
        workdir: Directory mounted into Docker (must contain the worker).
        docker_image: Docker image tag to use for the worker.
        timeout: Optional timeout (seconds) for the Docker process.

    Returns:
        The input image with pose landmarks and connections drawn on it (RGB array).
    """
    _require_cv2()

    poses_payload = _poses_json_for_worker(pose_image)
    annotated = _run_docker_visualizer(
        rgb=pose_image.image,
        poses_payload=poses_payload,
        workdir=workdir or _default_workdir(),
        image_name=docker_image,
        timeout=timeout,
    )

    if output_path:
        outdir = os.path.dirname(output_path) or "."
        os.makedirs(outdir, exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    if plot:
        plt.imshow(annotated)
        plt.axis("off")
        plt.show()
        plt.close()

    return annotated
