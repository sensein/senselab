"""MediaPipe Face Landmarker worker, run inside the ``mediapipe-face`` subprocess venv.

Usage: ``python mediapipe_worker.py SPEC.json``. The spec names the video, the model file, the
sampling rate, the decode scale and the tiles; the worker writes one ``.npz`` with per-tile arrays
and prints a JSON status line to stdout. It imports only ``numpy``, ``av``, ``mediapipe`` and the
sibling ``frame_stream`` module.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

try:
    from senselab.video.tasks.facial_analysis.frame_stream import crop_tile, iter_frames, probe_video, scaled_size
except ImportError:  # inside the backend venv, where senselab is not installed
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from frame_stream import crop_tile, iter_frames, probe_video, scaled_size  # type: ignore[no-redef]

# Iris centres and eye-corner landmarks of the 478-point mesh, ordered image-left to image-right.
_EYES = (
    {"iris": 468, "left": 33, "right": 133},
    {"iris": 473, "left": 362, "right": 263},
)
_MOUTH = {"top": 13, "bottom": 14, "left": 78, "right": 308}


def euler_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Head pitch, yaw, roll in degrees from a 4x4 facial transformation matrix.

    Conventions match py-feat's: pitch positive up, yaw positive toward the subject's right
    (image left), roll positive tilting toward the subject's right.
    """
    r = matrix[:3, :3]
    sy = np.hypot(r[2, 1], r[2, 2])
    pitch = -np.degrees(np.arctan2(r[2, 1], r[2, 2]))
    yaw = -np.degrees(np.arctan2(-r[2, 0], sy))
    roll = np.degrees(np.arctan2(r[1, 0], r[0, 0]))
    return np.array([pitch, yaw, roll], dtype=np.float32)


def eye_offsets(points: np.ndarray) -> np.ndarray:
    """Iris offset inside the eye, ``(horizontal, vertical)`` in eye widths, both eyes averaged.

    Horizontal is the iris centre's position between the corners minus 0.5, positive toward image
    right. Vertical is its distance below the corner-to-corner line over the eye width, positive
    downward; dividing by the width rather than the lid gap keeps it finite through a blink.
    """
    out = []
    for eye in _EYES:
        iris = points[eye["iris"], :2]
        left, right = points[eye["left"], :2], points[eye["right"], :2]
        width = right[0] - left[0]
        if abs(width) <= 1e-6:
            out.append((np.nan, np.nan))
            continue
        h = (iris[0] - left[0]) / width - 0.5
        v = (iris[1] - (left[1] + right[1]) / 2) / width
        out.append((h, v))
    return np.nanmean(np.asarray(out, dtype=np.float32), axis=0)


def mouth_aspect(points: np.ndarray) -> float:
    """Inner-lip opening over mouth width."""
    gap = np.linalg.norm(points[_MOUTH["bottom"], :2] - points[_MOUTH["top"], :2])
    width = np.linalg.norm(points[_MOUTH["right"], :2] - points[_MOUTH["left"], :2])
    return float(gap / width) if width > 1e-6 else float("nan")


def run(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Run the landmarker over every tile of every sampled frame and save the arrays."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    tiles: List[Dict[str, Any]] = spec["tiles"]
    info = probe_video(spec["video"])
    scale = min(1.0, float(spec["tile_width"]) / max(t["box"][2] for t in tiles))
    motion_scale = float(spec["motion_width"]) / max(t["box"][2] for t in tiles)

    def make() -> Any:  # noqa: ANN401
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=spec["model"]),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=int(spec["num_faces"]),
            min_face_detection_confidence=float(spec["min_face_detection_confidence"]),
            min_face_presence_confidence=float(spec["min_face_presence_confidence"]),
            min_tracking_confidence=float(spec["min_tracking_confidence"]),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        return vision.FaceLandmarker.create_from_options(options)

    landmarkers = [make() for _ in tiles]
    columns: Dict[str, Dict[str, List[Any]]] = {t["tile_id"]: {} for t in tiles}
    blendshape_names: List[str] = []
    previous: Dict[str, np.ndarray] = {}
    began = time.monotonic()
    last_ms = -1
    n = 0
    last_t = 0.0
    for t, frame in iter_frames(spec["video"], fps=float(spec["fps"]), scale=scale):
        ms = max(int(round(t * 1000)), last_ms + 1)
        last_ms = ms
        for tile, landmarker in zip(tiles, landmarkers):
            crop = crop_tile(frame, tile["box"], scale=scale, masks=tile.get("masks", []))
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop), ms)
            col = columns[tile["tile_id"]]
            step = max(1, int(round(scale / motion_scale))) if motion_scale < scale else 1
            small = crop[::step, ::step].mean(axis=2, dtype=np.float32)
            prev = previous.get(tile["tile_id"])
            motion = (
                float(np.mean(np.abs(small - prev)) / 255.0)
                if prev is not None and prev.shape == small.shape
                else np.nan
            )
            previous[tile["tile_id"]] = small
            present = bool(result.face_landmarks)
            col.setdefault("time_s", []).append(t)
            col.setdefault("face_present", []).append(present)
            col.setdefault("n_faces", []).append(len(result.face_landmarks))
            col.setdefault("motion_energy", []).append(motion)
            if present:
                points = np.array([(p.x, p.y, p.z) for p in result.face_landmarks[0]], dtype=np.float32)
                if result.facial_transformation_matrixes:
                    matrix = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float32)
                    angles = euler_from_matrix(matrix)
                    translation = matrix[:3, 3]
                else:
                    angles = np.full(3, np.nan, np.float32)
                    translation = np.full(3, np.nan, np.float32)
                scores = [c.score for c in result.face_blendshapes[0]] if result.face_blendshapes else []
                if scores and not blendshape_names:
                    blendshape_names = [c.category_name for c in result.face_blendshapes[0]]
                xs, ys = points[:, 0], points[:, 1]
                box = [float(xs.min()), float(ys.min()), float(xs.max() - xs.min()), float(ys.max() - ys.min())]
                eye = eye_offsets(points) if points.shape[0] >= 478 else np.full(2, np.nan, np.float32)
                mar = mouth_aspect(points)
            else:
                angles = np.full(3, np.nan, np.float32)
                translation = np.full(3, np.nan, np.float32)
                scores = []
                box = [np.nan] * 4
                eye = np.full(2, np.nan, np.float32)
                mar = np.nan
            col.setdefault("pose", []).append(angles)
            col.setdefault("translation", []).append(translation)
            col.setdefault("blendshapes", []).append(scores)
            col.setdefault("face_box", []).append(box)
            col.setdefault("eye_offset", []).append(eye)
            col.setdefault("mouth_aspect", []).append(mar)
        n += 1
        last_t = t
        if n % 2000 == 0:
            elapsed = time.monotonic() - began
            print(
                json.dumps({"progress": {"t": round(t, 2), "frames": n, "wall_s": round(elapsed, 1)}}), file=sys.stderr
            )
    for landmarker in landmarkers:
        landmarker.close()

    width = len(blendshape_names) or 52
    arrays: Dict[str, np.ndarray] = {}
    for tile_id, col in columns.items():
        if not col:
            continue
        blends = np.full((len(col["blendshapes"]), width), np.nan, np.float32)
        for i, row in enumerate(col["blendshapes"]):
            if row:
                blends[i, : len(row)] = row
        arrays[f"{tile_id}/time_s"] = np.asarray(col["time_s"], np.float64)
        arrays[f"{tile_id}/face_present"] = np.asarray(col["face_present"], bool)
        arrays[f"{tile_id}/n_faces"] = np.asarray(col["n_faces"], np.int16)
        arrays[f"{tile_id}/motion_energy"] = np.asarray(col["motion_energy"], np.float32)
        arrays[f"{tile_id}/pose"] = np.stack(col["pose"]).astype(np.float32)
        arrays[f"{tile_id}/translation"] = np.stack(col["translation"]).astype(np.float32)
        arrays[f"{tile_id}/blendshapes"] = blends
        arrays[f"{tile_id}/face_box"] = np.asarray(col["face_box"], np.float32)
        arrays[f"{tile_id}/eye_offset"] = np.stack(col["eye_offset"]).astype(np.float32)
        arrays[f"{tile_id}/mouth_aspect"] = np.asarray(col["mouth_aspect"], np.float32)
    np.savez(spec["out"], **arrays)  # type: ignore[arg-type]
    wall = time.monotonic() - began
    return {
        "ok": True,
        "frames": n,
        "video_s": last_t,
        "wall_s": wall,
        "blendshape_names": blendshape_names,
        "mediapipe_version": getattr(mp, "__version__", "unknown"),
        "decode_size": list(scaled_size(info["width"], info["height"], scale)),
    }


if __name__ == "__main__":
    import faulthandler

    faulthandler.enable()
    with open(sys.argv[1]) as handle:
        status = run(json.load(handle))
    print(json.dumps(status))
