"""MediaPipe Pose Estimation Worker Script."""

import argparse
import json


def main(image: str, model: str, num: int) -> None:
    """Run MediaPipe Pose estimation on a single image."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    bgr = cv2.imread(image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"Could not read image: {image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    base = python.BaseOptions(model_asset_path=model)
    opts = vision.PoseLandmarkerOptions(
        base_options=base,
        output_segmentation_masks=True,
        num_poses=num,
    )
    detector = vision.PoseLandmarker.create_from_options(opts)
    res = detector.detect(mp_image)

    poses = []
    if res and res.pose_landmarks:
        for person in res.pose_landmarks:
            poses.append([{"x": lm.x, "y": lm.y, "z": lm.z, "v": lm.visibility} for lm in person])

    print(json.dumps({"num_poses": len(poses), "poses": poses}, separators=(",", ":")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--num", type=int, default=1)
    main(**vars(ap.parse_args()))
