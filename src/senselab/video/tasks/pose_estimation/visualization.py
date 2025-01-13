"""This module implements visualization for the Pose Estimation task."""

import os
from typing import Optional

import cv2
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

from senselab.video.data_structures.pose import ImagePose, PoseModel


def visualize(pose_image: ImagePose, output_path: Optional[str] = None) -> np.ndarray:
    """Visualize detected poses.

    Args:
        pose_image: ImagePose object containing detections.
        output_path: Optional path to save visualization.

    Returns:
        Annotated image.
    """
    annotated_image = pose_image.image.copy()

    if pose_image.model == PoseModel.MEDIAPIPE:
        for individual in pose_image.individuals:
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            landmarks = [
                landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)  # type: ignore[attr-defined]
                for lm in individual.normalized_landmarks.values()
            ]
            pose_landmarks_proto.landmark.extend(landmarks)
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style(),
            )

    elif pose_image.model == PoseModel.YOLO:
        height, width, _ = annotated_image.shape
        for individual in pose_image.individuals:
            for landmark in individual.normalized_landmarks.values():
                x = int(landmark.x * width)  # type: ignore[attr-defined]
                y = int(landmark.y * height)  # type: ignore[attr-defined]
                cv2.circle(annotated_image, (x, y), 3, (0, 255, 0), -1)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    return annotated_image
