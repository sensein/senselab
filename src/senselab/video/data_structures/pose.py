import mediapipe as mp
import torch
import numpy as np
import cv2
from typing import Union, List, Dict
from pydantic import BaseModel, field_validator, ConfigDict
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class PoseSkeleton(BaseModel):
    """
    Data structure for estimated poses of multiple individuals in an image.

    Attributes:
    image: object representing the original image (torch.Tensor)
    normalized_landmarks: list of dictionaries for each person's body landmarks with normalized image coordinates (x, y, z).
    world_landmarks: list of dictionaries for each person's body landmarks with real-world coordinates (x, y, z).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image: torch.Tensor
    normalized_landmarks: List[Dict[str, List[float]]]  # List of dictionaries for each person
    world_landmarks: List[Dict[str, List[float]]]       # List of dictionaries for each person

    @field_validator('normalized_landmarks', 'world_landmarks', mode="before")
    def validate_landmarks(cls, v):
        for person_landmarks in v:
            for coords in person_landmarks.values():
                if len(coords) < 3:
                    raise ValueError("Each landmark must have at least 3 coordinates (x, y, z).")
        return v

    def to_numpy(self) -> np.ndarray:
        return self.image.cpu().numpy()

    def get_landmark_coordinates(self, person_index: int, landmark: str, world: bool = False) -> List[float]:
        landmarks = self.world_landmarks if world else self.normalized_landmarks
        if person_index < len(landmarks):
            return landmarks[person_index].get(landmark, [None, None, None])
        return [None, None, None]

    def visualize_pose(self):
        pass


def draw_landmarks_on_image(rgb_image, detection_result):
    annotated_image = np.copy(rgb_image)
    for person_landmarks in detection_result.pose_landmarks:
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in person_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style()
        )
    return annotated_image


def estimate_pose_with_mediapipe(image_path: str) -> PoseSkeleton:
    base_options = python.BaseOptions(model_asset_path='src/senselab/video/tasks/pose_estimation/models/pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    # Load the input image
    image = mp.Image.create_from_file(image_path)
    detection_result = detector.detect(image)
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

    # Save the annotated image
    annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("pose_estimation_output.png", annotated_image_bgr)
    print("Image saved as pose_estimation_output.png")

    # Initialize lists to hold landmarks for all detected individuals
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

    # Convert the image to torch tensor
    image_tensor = torch.from_numpy(image.numpy_view())

    # Return PoseSkeleton with all detected individuals' landmarks
    return PoseSkeleton(
        image=image_tensor,
        normalized_landmarks=normalized_landmarks_list,
        world_landmarks=world_landmarks_list
    )


# Example usage
if __name__ == "__main__":
    image_path = '/home/brukew/person_sitting.jpg'
    pose_result = estimate_pose_with_mediapipe(image_path)
    print("Normalized Landmarks:", pose_result.normalized_landmarks)
    print("World Landmarks:", pose_result.world_landmarks)
