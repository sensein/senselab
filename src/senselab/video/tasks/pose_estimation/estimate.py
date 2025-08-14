"""Pose estimation module with MediaPipe (via Docker) and YOLO backends."""

import json
import os
import shlex
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# --- Optional dependencies (cv2, ultralytics) --------------------------------
try:
    import cv2

    CV2_AVAILABLE = True
except ModuleNotFoundError:
    CV2_AVAILABLE = False

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ModuleNotFoundError:
    YOLO_AVAILABLE = False

# --- Senselab imports ---------------------------------------------------------
from senselab.utils.data_structures.docker import docker_is_running
from senselab.video.data_structures.pose import (
    ImagePose,
    IndividualPose,
    MediaPipePoseLandmark,
    PoseModel,
    YOLOPoseLandmark,
)
from senselab.video.tasks.pose_estimation.utils import (
    MEDIAPIPE_KEYPOINT_MAPPING,
    YOLO_KEYPOINT_MAPPING,
    get_model,
)

# =============================================================================
# Base class
# =============================================================================


class PoseEstimator(ABC):
    """Abstract base class for pose estimators."""

    model_path: str

    @abstractmethod
    def __init__(self, model_type: str) -> None:
        """Initialize the estimator with a given model type."""
        ...

    @abstractmethod
    def estimate(self, image: np.ndarray) -> ImagePose:
        """Estimate poses from an RGB uint8 image of shape (H, W, 3)."""
        ...

    @abstractmethod
    def estimate_from_path(self, image_path: str) -> ImagePose:
        """Estimate poses from an image file path."""
        ...


# =============================================================================
# Utilities
# =============================================================================


def _require_cv2() -> None:
    if not CV2_AVAILABLE:
        raise ModuleNotFoundError(
            "`opencv-python` is not installed. "
            "Please install senselab video dependencies using `pip install 'senselab[video]'`."
        )


def _read_rgb_image(path: str | os.PathLike[str]) -> np.ndarray:
    """Read image from disk as RGB uint8."""
    _require_cv2()
    if not Path(path).exists():
        raise FileNotFoundError(f"Image not found at: {path}")
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise ValueError(f"Failed to read image at: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# =============================================================================
# MediaPipe via Docker
# =============================================================================


class MediaPipePoseEstimator(PoseEstimator):
    """MediaPipe implementation that delegates inference to Docker.

    It defaults to using the image: fabiocat93/mediapipe-deps.

    Args:
        model_type: 'lite' | 'full' | 'heavy' (as supported by your MODELS dict).
        workdir: Host directory mounted into the container. The worker script
                 and (a copy/link of) the model must live under this directory.
        image_name: Docker image to use (tag included).
    """

    def __init__(
        self,
        model_type: str = "lite",
        workdir: Optional[str | os.PathLike[str]] = None,
        image_name: str = "fabiocat93/mediapipe-deps:latest",
    ) -> None:
        """Initialize the MediaPipe pose estimator."""
        if not docker_is_running():
            raise RuntimeError("Docker is not running and is required for pose estimation with MediaPipe.")

        # Download/resolve model path on host
        self.model_path = get_model("mediapipe", model_type)
        self.image_name = image_name

        # Default workdir: folder containing this file
        if workdir is None:
            workdir = Path(__file__).resolve().parent
        self.workdir = Path(workdir).resolve()

        # Worker script must be present in the mounted workdir
        self.worker_script = self.workdir / "mp_pose_worker.py"
        if not self.worker_script.exists():
            raise FileNotFoundError(
                f"Worker script not found at {self.worker_script}. " "Place mp_pose_worker.py in the mounted workdir."
            )

        # Ensure model is visible under workdir (hard-link or copy if needed)
        self.model_rel = self._ensure_path_under_workdir(self.model_path)

    # -- Public API ------------------------------------------------------------

    def estimate(self, image: np.ndarray, num_individuals: int = 1) -> ImagePose:
        """Run pose estimation inside Docker and return ImagePose.

        Args:
            image: RGB uint8 numpy array of shape (H, W, 3).
            num_individuals: Number of individuals to estimate (default 1).

        Returns:
            ImagePose containing the original image and estimated individuals.
        """
        if not isinstance(num_individuals, int) or num_individuals < 0:
            raise ValueError("`num_individuals` must be an integer >= 0")

        tmp_img = self._write_temp_image_in_workdir(image)
        try:
            result = self._run_in_docker(
                image_rel=self._rel_to_workdir(tmp_img),
                model_rel=self.model_rel,
                num=num_individuals,
            )
        finally:
            try:
                tmp_img.unlink(missing_ok=True)  # type: ignore[attr-defined]
            except Exception:
                pass

        individuals = self._to_individuals(result.get("poses", []))
        return ImagePose(image=image, individuals=individuals, model=PoseModel.MEDIAPIPE)

    def estimate_from_path(self, image_path: str, num_individuals: int = 1) -> ImagePose:
        """Estimate poses in image from file path using MediaPipe."""
        image = _read_rgb_image(image_path)
        return self.estimate(image, num_individuals)

    # -- Internals -------------------------------------------------------------

    def _run_in_docker(self, image_rel: str, model_rel: str, num: int) -> Dict[str, Any]:
        """Run the MediaPipe worker script inside Docker.

        Args:
            image_rel: Relative path to the image inside the Docker container.
            model_rel: Relative path to the model inside the Docker container.
            num: Number of individuals to estimate.

        Calls:
            docker run --rm -v {workdir}:/app -w /app IMAGE \
                python mp_pose_worker.py --image <rel> --model <rel> --num <n>

        Returns parsed JSON dict from the worker's stdout.
        """
        if not docker_is_running():
            raise RuntimeError("Docker is not running and is required for pose estimation with MediaPipe.")

        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{str(self.workdir)}:/app",
            "-w",
            "/app",
            "-e",
            "MPLCONFIGDIR=/tmp",  # avoid matplotlib permission issues
            self.image_name,
            "python",
            str(self.worker_script.name),
            "--image",
            image_rel,
            "--model",
            model_rel,
            "--num",
            str(num),
        ]

        try:
            proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            # Surface helpful diagnostics including stderr
            raise RuntimeError(
                "MediaPipe Docker worker failed.\n"
                f"Command: {shlex.join(cmd)}\n"
                f"Exit code: {e.returncode}\n"
                f"Stdout: {e.stdout}\n"
                f"Stderr: {e.stderr}"
            ) from e

        stdout = proc.stdout.strip()
        if not stdout:
            raise RuntimeError("MediaPipe Docker worker returned empty output.")

        try:
            return json.loads(stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError("Failed to parse JSON from MediaPipe Docker worker.\n" f"Raw stdout: {stdout}") from e

    def _write_temp_image_in_workdir(self, rgb: np.ndarray) -> Path:
        """Writes a temporary PNG under workdir.

        Args:
            rgb: RGB uint8 numpy array of shape (H, W, 3).

        Returns:
            Path to the temporary image file.

        Raises:
            ValueError: If `rgb` is not a valid RGB uint8 numpy array.
            RuntimeError: If the image cannot be written
                (requires OpenCV on host).
        """
        if not (isinstance(rgb, np.ndarray) and rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.dtype == np.uint8):
            raise ValueError("`image` must be an RGB uint8 numpy array of shape (H, W, 3).")

        tmp = Path(tempfile.mkstemp(prefix="mp_img_", suffix=".png", dir=self.workdir)[1])

        wrote = False
        try:
            if not CV2_AVAILABLE:
                raise ModuleNotFoundError(
                    "`opencv-python` is not installed. "
                    "Please install senselab video dependencies using `pip install 'senselab[video]'`."
                )
            bgr = rgb[:, :, ::-1]
            ok = cv2.imwrite(str(tmp), bgr)
            wrote = bool(ok)
        except Exception:
            wrote = False

        if not wrote:
            try:
                tmp.unlink()
            except Exception:
                pass
            raise RuntimeError("Failed to write temp image (need imageio or PIL or OpenCV on host).")
        return tmp

    def _rel_to_workdir(self, p: Path) -> str:
        return str(Path(p).resolve().relative_to(self.workdir))

    def _ensure_path_under_workdir(self, path_str: str) -> str:
        """Ensures the model file is under workdir.

        If not, creates a hard link or
        copies it into workdir/models/. Returns the *relative* path string.

        Args:
            path_str: Path to the model file.

        Returns:
            Relative path string to the model file under workdir/models/.
        """
        src = Path(path_str).resolve()
        try:
            rel = src.relative_to(self.workdir)
            return str(rel)
        except Exception:
            models_dir = self.workdir / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            dst = models_dir / src.name
            if not dst.exists():
                try:
                    os.link(src, dst)  # hard link if same filesystem
                except Exception:
                    import shutil

                    shutil.copy2(src, dst)
            return str(dst.relative_to(self.workdir))

    def _to_individuals(self, poses: List[List[Dict[str, float]]]) -> List[IndividualPose]:
        """Convert worker JSON -> List[IndividualPose].

        Args:
            poses: List of poses, each pose is a list of dicts with keys x,y,z,v in MediaPipe index order.

        Returns:
            List[IndividualPose]: List of IndividualPose objects.
        """
        individuals: List[IndividualPose] = []
        for idx, person in enumerate(poses):
            normalized = {
                MEDIAPIPE_KEYPOINT_MAPPING.get(i, f"keypoint_{i}"): MediaPipePoseLandmark(
                    x=p.get("x", 0.0),
                    y=p.get("y", 0.0),
                    z=p.get("z", 0.0),
                    visibility=p.get("v", 0.0),
                )
                for i, p in enumerate(person)
            }
            individuals.append(
                IndividualPose(
                    individual_index=idx,
                    normalized_landmarks=normalized,
                    world_landmarks=None,  # not provided by worker
                )
            )
        return individuals


# =============================================================================
# YOLO
# =============================================================================


class YOLOPoseEstimator(PoseEstimator):
    """YOLO implementation of pose estimation."""

    def __init__(self, model_type: str = "8n") -> None:
        """Initialize the YOLO pose estimator.

        Args:
            model_type: Model type to use, e.g. '8n', '8x', etc.

        Raises:
            ModuleNotFoundError: If `ultralytics` is not installed.
        """
        if not YOLO_AVAILABLE:
            raise ModuleNotFoundError(
                "`ultralytics` is not installed. "
                "Please install senselab video dependencies using `pip install 'senselab[video]'`."
            )
        self.model_path = get_model("yolo", model_type)
        self._model = YOLO(self.model_path)

    def estimate(self, image: np.ndarray) -> ImagePose:
        """Estimate poses using YOLO from an RGB uint8 image."""
        results = self._model(image, verbose=False)

        # No detections
        if results[0].keypoints is None or results[0].keypoints.data.numel() == 0:
            return ImagePose(image=image, individuals=[], model=PoseModel.YOLO)

        individuals: List[IndividualPose] = []
        for idx, person_keypoints in enumerate(results[0].keypoints):
            # confidences: shape [1, K]
            confidence_values = person_keypoints.conf.squeeze()

            normalized_dict = {
                YOLO_KEYPOINT_MAPPING.get(i, f"keypoint_{i}"): YOLOPoseLandmark(
                    x=float(kp[0].item()),
                    y=float(kp[1].item()),
                    confidence=float(confidence_values[i].item()) if confidence_values.numel() > i else 0.0,
                )
                for i, kp in enumerate(person_keypoints.xyn[0])
            }

            individuals.append(
                IndividualPose(
                    individual_index=idx,
                    normalized_landmarks=normalized_dict,
                )
            )

        return ImagePose(image=image, individuals=individuals, model=PoseModel.YOLO)

    def estimate_from_path(self, image_path: str) -> ImagePose:
        """Estimate poses in image from file path using YOLO."""
        image = _read_rgb_image(image_path)
        return self.estimate(image)
