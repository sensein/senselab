"""Data structures relevant for pose estimation."""

from typing import Dict, List

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

# class Landmark


class IndividualPose(BaseModel):
    """Data structure for estimated pose of single individual in an image.

    Attributes:
    pose_index: index of individual detected.
    normalized_landmarks: Dictionary of body landmarks with normalized image coordinates and visibility (x, y, z, c).
    world_landmarks: Dictionary of body landmarks with real-world coordinates and visibility (x, y, z, c).
    """

    individual_index: int
    normalized_landmarks: Dict[str, List[float]]
    world_landmarks: Dict[str, List[float]]

    @field_validator("normalized_landmarks", "world_landmarks", mode="before")
    def validate_landmarks(cls, v: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """Validate that landmarks contain exactly 4 coordinates (x,y,z,visibility)."""
        for coords in v.values():
            if len(coords) != 4:
                raise ValueError("Each landmark must have exactly 4 coordinates (x, y, z, visibility)")
        return v

    def get_landmark_coordinates(self, landmark: str, world: bool = False) -> List[float]:
        """Returns coordinates for specified landmark.

        Args:
            landmark: Name of the landmark (e.g., "landmark_0")
            world: If True, returns world coordinates instead of normalized
        Returns:
            [x, y, z, visibility] coordinates
        """
        landmarks = self.world_landmarks if world else self.normalized_landmarks
        if landmark not in landmarks:
            raise ValueError(f"Landmark '{landmark}' not found. Available landmarks: {sorted(landmarks.keys())}")
        return landmarks[landmark]


class ImagePose(BaseModel):
    """Data structure for estimated poses of multiple individuals in an image.

    Attributes:
    image: numpy array representing the original image
    individuals: list of IndividualPose objects for each individual with an estimated pose.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: np.ndarray
    individuals: List[IndividualPose]

    def get_individual(self, individual_index: int) -> IndividualPose:
        """Returns IndividualPose object for specified individual."""
        if individual_index >= len(self.individuals) or individual_index < 0:
            raise ValueError(
                f"Individual index {individual_index} is invalid. {len(self.individuals)} poses were estimated. "
                f"Valid indices are {f'0 to {len(self.individuals)-1}' if len(self.individuals) > 0 else 'none'}"
            )
        return self.individuals[individual_index]
