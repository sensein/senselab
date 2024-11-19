"""Data structures relevant for pose estimation."""

from typing import Any, Dict, List

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarkerResult
from pydantic import BaseModel, ConfigDict, field_validator


class PoseSkeleton(BaseModel):
    """Data structure for estimated poses of multiple individuals in an image.

    Attributes:
    image: object representing the original image (torch.Tensor)
    normalized_landmarks: list of dictionaries for each person's body landmarks with normalized
        image coordinates (x, y, z).
    world_landmarks: list of dictionaries for each person's body landmarks with real-world
        coordinates (x, y, z).
    detection_result: output of MediaPipe pose estimation
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    image: torch.Tensor
    normalized_landmarks: List[Dict[str, List[float]]]  # List of dictionaries for each person
    world_landmarks: List[Dict[str, List[float]]]  # List of dictionaries for each person
    detection_result: Any

    @field_validator("normalized_landmarks", "world_landmarks", mode="before")
    def validate_landmarks(cls, v: List[Dict[str, List[float]]]) -> List[Dict[str, List[float]]]:
        """Validate that landmarks contain at least 3 coordinates."""
        for person_landmarks in v:
            for coords in person_landmarks.values():
                if len(coords) < 3:
                    raise ValueError("Each landmark must have at least 3 coordinates (x, y, z).")
        return v

    def to_numpy(self) -> np.ndarray:
        """Converts image to numpy array.

        Returns:
            numpy array of image that was initialized in class
        """
        return self.image.cpu().numpy()

    def get_landmark_coordinates(self, landmark: str, person_index: int = 0, world: bool = False) -> List[float]:
        """Returns the coordinates of a specified landmark for a given individual in the image.

        Args:
            person_index (int): Index of the individual in the detection results. Defaults to 0.
            landmark (str): Name of the landmark (e.g., "landmark_0", "landmark_1").
            world (bool): If True, retrieves world coordinates. Otherwise, retrieves normalized coordinates.

        Returns:
            List[float]: Coordinates of the landmark in the form [x, y, z, visibility].

        Raises:
            ValueError: If the landmark does not exist or the person index is out of bounds.
        """
        landmarks = self.world_landmarks if world else self.normalized_landmarks
        if person_index >= len(landmarks):
            raise ValueError(
                f"Person index {person_index} is invalid. Image contains {len(landmarks)} people. "
                f"Valid indices are {f'0 to {len(landmarks)-1}' if len(landmarks) > 0 else 'none'}"
            )

        if landmark not in landmarks[person_index]:
            raise ValueError(
                f"Landmark '{landmark}' not found. Available landmarks: {sorted(landmarks[person_index].keys())}"
            )

        return landmarks[person_index][landmark]

    def visualize_pose(self) -> None:
        """Visualizes pose landmarks on the image and saves the annotated image.

        Saves the annotated image as "pose_estimation_output.png" in the current directory.
        """
        annotated_image = draw_landmarks_on_image(self.to_numpy(), self.detection_result)
        # Save the annotated image
        annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite("pose_estimation_output.png", annotated_image_bgr)
        print("Image saved as pose_estimation_output.png")


def draw_landmarks_on_image(rgb_image: np.ndarray, detection_result: PoseLandmarkerResult) -> np.ndarray:
    """Draws pose landmarks on the input RGB image.

    Args:
        rgb_image: The input image in RGB format
        detection_result: The detection result containing pose landmarks

    Returns:
        Annotated image with pose landmarks drawn
    """
    annotated_image = np.copy(rgb_image)
    for person_landmarks in detection_result.pose_landmarks:
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend(
            [landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in person_landmarks]
        )
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style(),
        )
    return annotated_image


def estimate_pose_with_mediapipe(image_path: str, num_of_individuals: int = 1) -> PoseSkeleton:
    """Estimates pose landmarks for individuals in the provided image using MediaPipe.

    Args:
        image_path (str): Path to the input image file.
        num_of_individuals (int): Maximum number of individuals to detect. Defaults to 1.

    Returns:
        PoseSkeleton object

    Raises:
        FileNotFoundError: If the image file does not exist.
    """
    # MediaPipe Pose Estimation config
    base_options = python.BaseOptions(
        model_asset_path="src/senselab/video/tasks/pose_estimation/models/pose_landmarker.task"
    )
    options = vision.PoseLandmarkerOptions(
        base_options=base_options, output_segmentation_masks=True, num_poses=num_of_individuals
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    # Load the input image
    image = mp.Image.create_from_file(image_path)
    detection_result = detector.detect(image)

    normalized_landmarks_list = []
    world_landmarks_list = []

    for person_landmarks in detection_result.pose_landmarks:
        # Store normalized landmarks (3D) for each person
        person_normalized_landmarks = {}
        for idx, landmark in enumerate(person_landmarks):
            person_normalized_landmarks[f"landmark_{idx}"] = [landmark.x, landmark.y, landmark.z, landmark.visibility]
        normalized_landmarks_list.append(person_normalized_landmarks)

    for person_landmarks in detection_result.pose_world_landmarks:
        # Store world landmarks (3D) for each person
        person_world_landmarks = {}
        for idx, landmark in enumerate(person_landmarks):
            person_world_landmarks[f"landmark_{idx}"] = [landmark.x, landmark.y, landmark.z, landmark.visibility]
        world_landmarks_list.append(person_world_landmarks)

    image_tensor = torch.from_numpy(image.numpy_view().copy())

    # Return PoseSkeleton with all detected individuals' landmarks
    return PoseSkeleton(
        image=image_tensor,
        normalized_landmarks=normalized_landmarks_list,
        world_landmarks=world_landmarks_list,
        detection_result=detection_result,
    )
