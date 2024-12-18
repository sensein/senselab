"""This module implements the Pose Estimation task and supporting utilities."""

import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from senselab.video.data_structures.pose import ImagePose, IndividualPose


def visualize_pose(image: ImagePose) -> np.ndarray:
    """Visualizes pose landmarks on the input image.

    Args:
        image: ImagePose object containing image and detected poses

    Returns:
        Annotated image with pose landmarks drawn

    Note:
        Saves the annotated image as 'pose_estimation_output.png'
    """
    # Convert to RGB if needed and create copy
    annotated_image = np.copy(image.image)
    if len(annotated_image.shape) == 2:  # Grayscale
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_GRAY2RGB)

    for individual in image.individuals:
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        landmarks = [
            landmark_pb2.NormalizedLandmark(x=coords[0], y=coords[1], z=coords[2])
            for coords in individual.normalized_landmarks.values()
        ]
        pose_landmarks_proto.landmark.extend(landmarks)

        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style(),
        )

    # Save the annotated image
    annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("pose_estimation_output.png", annotated_image_bgr)
    print("Image saved as pose_estimation_output.png")

    return annotated_image


def estimate_pose_with_mediapipe(
    image_path: str,
    num_of_individuals: int = 1,
    model_path: str = "src/senselab/video/tasks/pose_estimation/models/pose_landmarker.task",
) -> ImagePose:
    """Estimates pose landmarks for individuals in the provided image using MediaPipe.

    Args:
        image_path: Path to the input image file
        num_of_individuals: Maximum number of individuals to detect. Defaults to 1
        model_path: Path to the MediaPipe pose landmarker model file

    Returns:
        ImagePose object containing detected poses

    Raises:
        FileNotFoundError: If image_path or model_path doesn't exist
        RuntimeError: If pose detection fails
    """
    # Validate file paths
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    # Initialize detector
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options, output_segmentation_masks=True, num_poses=num_of_individuals
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    # Load and process image
    image = mp.Image.create_from_file(image_path)
    detection_result = detector.detect(image)

    if not detection_result:
        raise RuntimeError("Pose detection failed")

    # Create IndividualPose objects
    individuals = []
    for idx, (norm_landmarks, world_landmarks) in enumerate(
        zip(detection_result.pose_landmarks, detection_result.pose_world_landmarks)
    ):
        norm_dict = {f"landmark_{i}": [lm.x, lm.y, lm.z, lm.visibility] for i, lm in enumerate(norm_landmarks)}
        world_dict = {f"landmark_{i}": [lm.x, lm.y, lm.z, lm.visibility] for i, lm in enumerate(world_landmarks)}

        individual = IndividualPose(individual_index=idx, normalized_landmarks=norm_dict, world_landmarks=world_dict)
        individuals.append(individual)

    # Create and return ImagePose
    image_array = image.numpy_view().copy()
    return ImagePose(image=image_array, individuals=individuals, detection_result=detection_result)


if __name__ == "__main__":
    import os

    # Example usage
    image_path = "src/tests/data_for_testing/pose_data/three_people.jpg"
    try:
        pose_result = estimate_pose_with_mediapipe(image_path, num_of_individuals=2)
        annotated = visualize_pose(pose_result)
        print(f"Detected {len(pose_result.individuals)} individuals")
        print("Example individual pose data:", pose_result.get_individual(0))
    except Exception as e:
        print(f"Error processing image: {e}")
