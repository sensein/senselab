"""Data structures relevant for pose estimation."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator


class PoseModel(str, Enum):
    """Enum representing different pose estimation models.

    Attributes:
        MEDIAPIPE (str): Enum value for MediaPipe pose estimation.
        YOLO (str): Enum value for YOLO pose estimation.
    """

    MEDIAPIPE = "mp"
    YOLO = "yolo"


class PoseLandmark(ABC, BaseModel):
    """Abstract base class for representing pose landmarks.

    Methods:
        as_list() -> List[float]:
            Convert the landmark data to a list of coordinates.
    """

    @abstractmethod
    def as_list(self) -> List[float]:
        """Convert the landmark data to a list of floating-point values.

        Returns:
            List[float]: List of landmark data as floats.
        """
        pass


class MediaPipePoseLandmark(PoseLandmark):
    """Represents a pose landmark detected by MediaPipe.

    Attributes:
        x (float): X-coordinate of the landmark.
        y (float): Y-coordinate of the landmark.
        z (float): Z-coordinate of the landmark.
        visibility (float): Probability of the landmark being visible [0, 1].
    """

    x: float
    y: float
    z: float
    visibility: float

    def as_list(self) -> List[float]:
        """Convert the landmark data to a list.

        Returns:
            List[float]: [x, y, z, visibility] values.
        """
        return [self.x, self.y, self.z, self.visibility]


class YOLOPoseLandmark(PoseLandmark):
    """Represents a pose keypoint detected by YOLO.

    Attributes:
        x (float): X-coordinate of the keypoint.
        y (float): Y-coordinate of the keypoint.
        confidence (float): Confidence score of the detected keypoint [0, 1].
    """

    x: float
    y: float
    confidence: float

    def as_list(self) -> List[float]:
        """Convert the keypoint data to a list.

        Returns:
            List[float]: [x, y, confidence] values.
        """
        return [self.x, self.y, self.confidence]


class IndividualPose(BaseModel):
    """Data structure for the estimated pose of a single individual in an image.

    Attributes:
        individual_index (int): Index of the individual in the detection result.
        normalized_landmarks (Dict[str, PoseLandmark]): Dictionary of body landmarks with normalized coordinates.
        world_landmarks (Optional[Dict[str, PoseLandmark]]): Dictionary of body landmarks with real-world coordinates.
    """

    individual_index: int
    normalized_landmarks: Dict[str, PoseLandmark]
    world_landmarks: Optional[Dict[str, PoseLandmark]] = None

    def get_landmark_coordinates(self, landmark: str, world: bool = False) -> PoseLandmark:
        """Retrieve coordinates for a specific landmark.

        Args:
            landmark (str): Name of the landmark (e.g., "right_ankle").
            world (bool): Whether to retrieve world coordinates. Defaults to False.

        Returns:
            PoseLandmark: Object containing information on the specified landmark.

        Raises:
            ValueError: If the specified landmark is not found.
        """
        landmarks = self.world_landmarks if world else self.normalized_landmarks
        if not landmarks:
            raise ValueError("No landmarks available.")
        if landmark not in landmarks:
            raise ValueError(f"Landmark '{landmark}' not found. Available landmarks: {sorted(landmarks.keys())}")
        return landmarks[landmark]


class ImagePose(BaseModel):
    """Data structure for storing estimated poses of multiple individuals in an image.

    Attributes:
        model (PoseModel): The model used for pose estimation.
        image (np.ndarray): Original image as a NumPy array.
        individuals (List[IndividualPose]): List of IndividualPose objects for each detected individual.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: PoseModel
    image: np.ndarray
    individuals: List[IndividualPose]

    @field_validator("image", mode="before")
    def validate_image(cls, value: np.ndarray) -> np.ndarray:
        """Ensures image is a 3D numpy array with three color channels."""
        if not isinstance(value, np.ndarray):
            raise TypeError("Field 'image' must be a NumPy array.")
        if value.ndim != 3 or value.shape[2] != 3:
            raise ValueError("Field 'image' must be a 3D array with three color channels (RGB).")
        return value

    def get_individual(self, individual_index: int) -> IndividualPose:
        """Retrieve a specific individual's pose data.

        Args:
            individual_index (int): Index of the individual to retrieve.

        Returns:
            IndividualPose: Pose data for the specified individual.

        Raises:
            ValueError: If the index is invalid or no individuals are detected.
        """
        if individual_index >= len(self.individuals) or individual_index < 0:
            raise ValueError(
                f"Individual index {individual_index} is invalid. {len(self.individuals)} poses were estimated. "
                f"Valid indices are {f'0 to {len(self.individuals)-1}' if len(self.individuals) > 0 else 'none'}"
            )
        return self.individuals[individual_index]
