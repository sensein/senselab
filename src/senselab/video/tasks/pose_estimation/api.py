"""This module provides the API for pose estimation tasks."""

from typing import Any, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

from senselab.video.data_structures.pose import ImagePose
from senselab.video.tasks.pose_estimation.estimate import (
    MediaPipePoseEstimator,
    YOLOPoseEstimator,
)
from senselab.video.tasks.pose_estimation.visualization import visualize


def estimate_pose(image_path: str, model: str, **kwargs: Any) -> ImagePose:  # noqa ANN401
    """Estimate poses in an image using the specified model.

    Args:
        image_path (str): Path to the input image file.
        model (str): The model to use for pose estimation. Options are 'mediapipe' or 'yolo'.
        **kwargs: Additional keyword arguments for model-specific configurations:
            - For MediaPipe:
                - model_type (str): Type of MediaPipe model ('lite', 'full', 'heavy'). Defaults to 'lite'.
                - num_individuals (int): Maximum number of individuals to detect. Defaults to 1.
            - For YOLO:
                - model_type (str): Type of YOLO model ('8n', '8s', '11l', etc.). Defaults to '8n'.

    Returns:
        ImagePose: An object containing the estimated poses.

    Raises:
        ValueError: If an unsupported model or invalid arguments are provided.

    Examples:
        >>> estimate_pose("image.jpg", model="mediapipe", model_type="full", num_individuals=2)
        >>> estimate_pose("image.jpg", model="yolo", model_type="8n")
    """
    if model == "mediapipe":
        model_type = kwargs.get("model_type", "lite")
        num_individuals = kwargs.get("num_individuals", 1)

        if not isinstance(model_type, str):
            raise ValueError("Invalid 'model_type' for MediaPipe. Must be a string.")
        if not isinstance(num_individuals, int) or num_individuals < 1:
            raise ValueError("'num_individuals' must be a positive integer.")

        estimator = MediaPipePoseEstimator(model_type=model_type)
        return estimator.estimate_from_path(image_path, num_individuals=num_individuals)

    elif model == "yolo":
        model_type = kwargs.get("model_type", "8n")  # type: ignore[no-redef]

        if not isinstance(model_type, str):
            raise ValueError("Invalid 'model_type' for YOLO. Must be a string.")

        estimator = YOLOPoseEstimator(model_type=model_type)  # type: ignore[assignment]
        return estimator.estimate_from_path(image_path)

    else:
        raise ValueError(f"Unsupported model: {model}")


def visualize_pose(pose_image: ImagePose, output_path: Optional[str] = None) -> np.ndarray:
    """Visualize pose estimation results and optionally save the image.

    Args:
        pose_image (ImagePose): The pose estimation result.
        output_path (str): Optional path to save the visualized image.

    Returns:
        np.ndarray: Annotated image.
    """
    annotated_image = visualize(pose_image, output_path=output_path)

    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    return annotated_image


if __name__ == "__main__":
    image_path = "src/tests/data_for_testing/pose_data/single_person.jpg"
    pose_image = estimate_pose(image_path, model="mediapipe", model_type="lite", num_individuals=1)
    visualize_pose(pose_image, "annotate/mp.jpg")
