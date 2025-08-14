"""MediaPipe Pose Estimation Visualization Worker."""

import argparse
import json
import os
import sys
import traceback
from typing import Any, Dict, List

import numpy as np

MP_LANDMARK_COUNT = 33


def _to_proto(points: List[Dict[str, float]]) -> Any:  # noqa: ANN401
    """Convert a list of {x,y,z,v?} dicts to a MediaPipe NormalizedLandmarkList."""
    from mediapipe.framework.formats import landmark_pb2  # lazy import inside container

    proto = landmark_pb2.NormalizedLandmarkList()
    for p in points:
        lm = landmark_pb2.NormalizedLandmark(
            x=float(p.get("x", 0.0)),
            y=float(p.get("y", 0.0)),
            z=float(p.get("z", 0.0)),
        )
        if "v" in p:
            lm.visibility = float(p["v"])
        proto.landmark.append(lm)
    return proto


def _draw_single_pose(img_rgb: np.ndarray, points: list[dict]) -> None:
    """Draw one pose worth of landmarks (and connections if count==33) onto img_rgb.

    Args:
        img_rgb: Input image in RGB format.
        points: List of dicts with keys x, y, z, v representing the pose landmarks.
    """
    proto = _to_proto(points)
    from mediapipe import solutions

    # Only use connections if we have the canonical 33 landmarks
    connections = solutions.pose.POSE_CONNECTIONS if len(proto.landmark) == MP_LANDMARK_COUNT else None

    # Use uniform specs so drawing_utils doesn't look up a per-connection dict (avoids KeyError)
    landmark_spec = solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=2)
    connection_spec = solutions.drawing_utils.DrawingSpec(thickness=2)

    # Draw
    solutions.drawing_utils.draw_landmarks(
        img_rgb,
        proto,
        connections,
        landmark_drawing_spec=landmark_spec,
        connection_drawing_spec=connection_spec,
    )


def main(image: str, poses_json: str, out: str) -> None:
    """Main function to draw pose landmarks onto an image using MediaPipe drawing utils."""
    try:
        import cv2  # local import to keep module import light

        # Read input image (BGR) -> RGB
        bgr = cv2.imread(image, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Could not read image: {image}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Load poses JSON (expects {"poses": [[{x,y,z,v}, ...], ...]})
        with open(poses_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        poses = data.get("poses", [])
        if not isinstance(poses, list):
            raise ValueError("Invalid 'poses' format: expected a list.")

        # Draw each person
        for pts in poses:
            if not isinstance(pts, list):
                # Skip malformed entries instead of crashing the whole draw
                continue
            _draw_single_pose(rgb, pts)

        # Ensure output dir exists and write result
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        ok = cv2.imwrite(out, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError(f"Failed to write output: {out}")

    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Draw pose landmarks onto an image using MediaPipe drawing utils.")
    ap.add_argument("--image", required=True, help="Input image path (under /app).")
    ap.add_argument(
        "--poses",
        dest="poses_json",
        required=True,
        help="JSON path with {'poses': [[{x,y,z,v}, ...], ...]} (under /app).",
    )
    ap.add_argument("--out", required=True, help="Output image path (under /app).")
    main(**vars(ap.parse_args()))
