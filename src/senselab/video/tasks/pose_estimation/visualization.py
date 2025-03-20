"""This module implements visualization for the Pose Estimation task."""

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from senselab.video.data_structures.pose import ImagePose
from senselab.video.tasks.pose_estimation.utils import SENSELAB_KEYPOINT_MAPPING

try:
    import cv2

    CV2_AVAILABLE = True
except ModuleNotFoundError:
    CV2_AVAILABLE = False

try:
    from mediapipe import solutions
    from mediapipe.framework.formats import landmark_pb2

    MEDIAPIPE_AVAILABLE = True
except ModuleNotFoundError:
    MEDIAPIPE_AVAILABLE = False


def visualize(pose_image: ImagePose, output_path: Optional[str] = None, plot: bool = False) -> np.ndarray:
    """Visualize detected poses by drawing landmarks and connections on the image.

    Args:
        pose_image (ImagePose): The pose estimation result containing detected poses.
        output_path (str): Optional path to save the visualized image. If provided, saves the
            annotated image to this location.
        plot (bool): Whether to display the annotated image using matplotlib.

    Returns:
        np.ndarray: The input image with pose landmarks and connections drawn on it.
    """
    if not CV2_AVAILABLE:
        raise ModuleNotFoundError(
            "`opencv-python` is not installed. "
            "Please install senselab video dependencies using `pip install senselab['video']`."
        )
    if not MEDIAPIPE_AVAILABLE:
        raise ModuleNotFoundError(
            "`mediapipe` is not installed. "
            "Please install senselab video dependencies using `pip install senselab['video']`."
        )

    annotated_image = pose_image.image.copy()

    for individual in pose_image.individuals:
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        landmarks = []
        pose_lm = individual.normalized_landmarks
        # Filter out landmarks with low confidence
        landmarks = [
            landmark_pb2.NormalizedLandmark(
                x=getattr(pose_lm[lm], "x", 0), y=getattr(pose_lm[lm], "y", 0), z=getattr(pose_lm[lm], "z", 0)
            )  # type: ignore[attr-defined]
            if (lm in pose_lm and getattr(pose_lm[lm], "confidence", 1) > 0.5)
            else landmark_pb2.NormalizedLandmark(x=0, y=0, z=0, visibility=0)
            for lm in SENSELAB_KEYPOINT_MAPPING.values()
        ]
        pose_landmarks_proto.landmark.extend(landmarks)
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style(),
        )

    if output_path:
        print(f"Saving visualization to {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    if plot:
        plt.imshow(annotated_image)
        plt.axis("off")
        plt.show()
        plt.close()

    return annotated_image
