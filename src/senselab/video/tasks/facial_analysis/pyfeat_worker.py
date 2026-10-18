"""py-feat ``Detectorv2`` worker, run inside the ``pyfeat`` subprocess venv.

Usage: ``python pyfeat_worker.py SPEC.json``. The spec names the video, the pinned weight files, the
sampling rate, the decode scale, the batch size, the device and the tiles. Tile crops of each
sampled frame are batched through RetinaFace and the multitask model; per tile and frame the largest
detected face is kept. Writes one ``.npz`` and prints a JSON status line to stdout.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

try:
    from senselab.video.tasks.facial_analysis.frame_stream import crop_tile, iter_frames, tile_placement, widest_box
except ImportError:  # inside the backend venv, where senselab is not installed
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from frame_stream import crop_tile, iter_frames, tile_placement, widest_box  # type: ignore[no-redef]


def pad_stack(crops: List[np.ndarray]) -> np.ndarray:
    """Stack crops of different sizes, zero-padded at the bottom and right to the largest."""
    h = max(c.shape[0] for c in crops)
    w = max(c.shape[1] for c in crops)
    out = np.zeros((len(crops), h, w) + crops[0].shape[2:], crops[0].dtype)
    for i, crop in enumerate(crops):
        out[i, : crop.shape[0], : crop.shape[1]] = crop
    return out


def build_detector(spec: Dict[str, Any]) -> Any:  # noqa: ANN401
    """Assemble a ``Detectorv2`` from pinned weight files, without its unpinned downloads."""
    import torch
    from feat.detector_v2 import Detectorv2
    from feat.face_detectors.Retinaface.Retinaface_test import Retinaface
    from feat.multitask.inference import _DLIB68_IDX, CHIP_SIZE, MultitaskModel
    from safetensors.torch import load_file
    from torch import nn

    device = torch.device(spec["device"])
    detector = Detectorv2.__new__(Detectorv2)
    nn.Module.__init__(detector)
    detector.device = device
    detector.face_size = CHIP_SIZE
    detector.face_detection_threshold = float(spec["face_detection_threshold"])
    detector.face_detector = Retinaface(device=device, pretrained="pinned")
    detector.face_detector.net.load_state_dict(load_file(spec["retinaface_weights"]), strict=True)
    detector.face_detector.net.to(device).eval()
    detector.multitask = MultitaskModel(device=device, weights_path=spec["multitask_weights"], amp=False)
    detector.identity_detector = None
    detector._idx68 = _DLIB68_IDX.to(device)
    detector.info = {"face_model": "retinaface", "multitask_model": "face_multitask_v2", "identity_model": None}
    return detector


def run(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Run the detector over every tile of every sampled frame and save the arrays."""
    import torch  # must load its OpenMP runtime before feat's xgboost does (spec D-6)

    if spec.get("threads"):
        torch.set_num_threads(int(spec["threads"]))
    import feat

    tiles: List[Dict[str, Any]] = spec["tiles"]
    scale = min(1.0, float(spec["tile_width"]) / widest_box(tiles))
    detector = build_detector(spec)
    au_names = list(detector.multitask.au_names)
    emotion_names = list(detector.multitask.emotion_names)
    threshold = float(spec["face_detection_threshold"])
    batch_size = int(spec["batch_size"])

    rows: Dict[str, List[np.ndarray]] = {t["tile_id"]: [] for t in tiles}
    times: Dict[str, List[float]] = {t["tile_id"]: [] for t in tiles}
    pending: List[np.ndarray] = []
    owners: List[tuple] = []
    value_columns: List[str] = []

    def flush() -> None:
        nonlocal value_columns
        if not pending:
            return
        batch = torch.from_numpy(pad_stack(pending)).permute(0, 3, 1, 2).contiguous()
        size = batch.shape[0]
        zeros = torch.zeros(size)
        batch_data = {
            "Image": batch,
            "Scale": torch.ones(size),
            "Padding": {"Left": zeros, "Top": zeros, "Right": zeros, "Bottom": zeros},
        }
        faces = detector.detect_faces(batch, face_detection_threshold=threshold)
        frame = detector.forward(faces, batch_data)
        n_per = [int(f["scores"].shape[0]) for f in faces]
        keep = ["FaceRectX", "FaceRectY", "FaceRectWidth", "FaceRectHeight", "FaceScore"]
        keep += ["Pitch", "Roll", "Yaw"] + emotion_names + ["valence", "arousal"] + au_names
        keep += [c for c in ("gaze_pitch", "gaze_yaw", "gaze_angle") if c in frame.columns]
        keep = [c for c in keep if c in frame.columns]
        value_columns = keep
        values = frame[keep].to_numpy(dtype=np.float32)
        cursor = 0
        for (tile_id, t), n in zip(owners, n_per):
            block = values[cursor : cursor + n]
            cursor += n
            if n > 1:
                area = block[:, 2] * block[:, 3]
                best = int(np.nanargmax(np.where(np.isfinite(area), area, -1)))
            else:
                best = 0
            row = block[best].copy()
            if not np.isfinite(row[0]):
                row[:] = np.nan
                row[keep.index("FaceScore")] = 0.0
            rows[tile_id].append(np.concatenate([row, [float(n if np.isfinite(block[0, 0]) else 0)]]))
            times[tile_id].append(t)
        pending.clear()
        owners.clear()

    began = time.monotonic()
    n_frames = 0
    last_t = 0.0
    for t, image in iter_frames(spec["video"], fps=float(spec["fps"]), scale=scale):
        for tile in tiles:
            placement = tile_placement(tile, t)
            if placement is None:
                continue
            _, box, masks = placement
            pending.append(crop_tile(image, box, scale=scale, masks=masks))
            owners.append((tile["tile_id"], t))
        n_frames += 1
        last_t = t
        if len(pending) >= batch_size:
            flush()
        if n_frames % 500 == 0:
            print(json.dumps({"progress": {"t": round(t, 2), "frames": n_frames}}), file=sys.stderr)
    flush()
    device = str(detector.device)
    if device == "mps":
        torch.mps.empty_cache()

    arrays: Dict[str, np.ndarray] = {}
    for tile_id in rows:
        if rows[tile_id]:
            arrays[f"{tile_id}/time_s"] = np.asarray(times[tile_id], np.float64)
            arrays[f"{tile_id}/values"] = np.stack(rows[tile_id]).astype(np.float32)
    np.savez(spec["out"], **arrays)  # type: ignore[arg-type]
    return {
        "ok": True,
        "frames": n_frames,
        "video_s": last_t,
        "wall_s": time.monotonic() - began,
        "columns": value_columns + ["n_faces"],
        "pyfeat_version": getattr(feat, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "device": device,
    }


if __name__ == "__main__":
    import faulthandler

    faulthandler.enable()
    with open(sys.argv[1]) as handle:
        status = run(json.load(handle))
    print(json.dumps(status))
