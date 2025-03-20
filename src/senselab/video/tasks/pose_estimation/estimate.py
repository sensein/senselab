"""This module implements pose estimation functionality."""

import os
from abc import ABC, abstractmethod
from typing import List

import numpy as np

try:
    import cv2

    CV2_AVAILABLE = True
except ModuleNotFoundError:
    CV2_AVAILABLE = False

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    MEDIAPIPE_AVAILABLE = True
except ModuleNotFoundError:
    MEDIAPIPE_AVAILABLE = False

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ModuleNotFoundError:
    YOLO_AVAILABLE = False

from senselab.video.data_structures.pose import (
    ImagePose,
    IndividualPose,
    MediaPipePoseLandmark,
    PoseModel,
    YOLOPoseLandmark,
)
from senselab.video.tasks.pose_estimation.utils import (
    MEDIAPIPE_KEYPOINT_MAPPING,
    YOLO_KEYPOINT_MAPPING,
    get_model,
)


class PoseEstimator(ABC):
    """Abstract base class for pose estimation models.

    Attributes:
        model_path (str): Path to the PoesEstimator model file.

    Methods:
        estimate(image): Estimate poses in the given image.
        estimate_from_path(image_path): Estimate poses in the image loaded from the specified path.
    """

    model_path: str

    @abstractmethod
    def __init__(self, model_type: str) -> None:
        """Initialize the PoseEstimator.

        Args:
            model_type (str): Type of model to use.
        """
        pass

    @abstractmethod
    def estimate(self, image: np.ndarray) -> ImagePose:
        """Estimate poses in an image.

        Args:
            image (np.ndarray): Input image in RGB format.

        Returns:
            ImagePose: An object containing detected poses and metadata.
        """
        pass

    @abstractmethod
    def estimate_from_path(self, image_path: str) -> ImagePose:
        """Estimate poses in an image loaded from a file path.

        Args:
            image_path (str): Path to the input image file.

        Returns:
            ImagePose: An object containing detected poses and metadata.
        """
        pass


class MediaPipePoseEstimator(PoseEstimator):
    """MediaPipe implementation of pose estimation.

    Attributes:
        model_path (str): Path to the MediaPipe model file.
        num_individuals (int): Number of individuals to detect.
    """

    def __init__(self, model_type: str = "lite") -> None:
        """Initialize the MediaPipePoseEstimator.

        Args:
            model_type (str): Type of MediaPipe model to use ('lite', 'full', 'heavy').
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ModuleNotFoundError(
                "`mediapipe` is not installed. "
                "Please install senselab video dependencies using `pip install senselab['video']`."
            )

        self.model_path = get_model("mediapipe", model_type)
        self.num_individuals = 1
        self._detector = None

    def detector(self, num_individuals: int = 1) -> "vision.PoseLandmarker":
        """Initialization of the MediaPipe detector.

        Args:
            num_individuals (int): Maximum number of individuals to detect. Defaults to 1.

        Returns:
            vision.PoseLandmarker: A MediaPipe pose landmarker object.
        """
        if self._detector is None or self.num_individuals != num_individuals:
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options, output_segmentation_masks=True, num_poses=num_individuals
            )
            self.num_individuals = num_individuals
            self._detector = vision.PoseLandmarker.create_from_options(options)
        return self._detector

    def estimate(self, image: np.ndarray, num_individuals: int = 1) -> ImagePose:
        """Estimate poses using MediaPipe.

        Args:
            image (np.ndarray): Input image in RGB format.
            num_individuals (int): Maximum number of individuals to detect. Defaults to 1.

        Returns:
            ImagePose: An object containing detected poses and metadata.

        Raises:
            ValueError: If `num_individuals` is not a positive integer.
        """
        if not isinstance(num_individuals, int) or num_individuals < 0:
            raise ValueError("Number of individuals must be an integer >=0")

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        detection_result = self.detector(num_individuals).detect(mp_image)

        individuals = self.extract_landmarks(detection_result)
        return ImagePose(
            image=image,
            individuals=individuals,
            model=PoseModel.MEDIAPIPE,
        )

    def estimate_from_path(self, image_path: str, num_individuals: int = 1) -> ImagePose:
        """Estimate poses in image from file path using MediaPipe.

        Args:
            image_path (str): Path to the input image file.
            num_individuals (int): Maximum number of individuals to detect. Defaults to 1.

        Returns:
            ImagePose: An object containing detected poses and metadata.

        Raises:
            FileNotFoundError: If the specified image file does not exist.
            ValueError: If `num_individuals` is not a positive integer.
        """
        if not CV2_AVAILABLE:
            raise ModuleNotFoundError(
                "`opencv-python` is not installed. "
                "Please install senselab video dependencies using `pip install senselab['video']`."
            )

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")

        # Load and process the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.estimate(image, num_individuals)

    def extract_landmarks(self, detection_result: "vision.PoseLandmarkerResult") -> List[IndividualPose]:
        """Extract normalized and world landmarks from the detection result.

        Args:
            detection_result (vision.PoseLandmarkerResult): The result from the MediaPipe pose detector.

        Returns:
            List[IndividualPose]: A list of `IndividualPose` objects representing detected individuals.
        """
        individuals = []
        for idx, (norm_landmarks, world_landmarks) in enumerate(
            zip(detection_result.pose_landmarks, detection_result.pose_world_landmarks)
        ):
            norm_dict = {
                MEDIAPIPE_KEYPOINT_MAPPING.get(i, f"keypoint_{i}"): MediaPipePoseLandmark(
                    x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility
                )
                for i, lm in enumerate(norm_landmarks)
            }

            world_dict = {
                MEDIAPIPE_KEYPOINT_MAPPING.get(i, f"keypoint_{i}"): MediaPipePoseLandmark(
                    x=lm.x, y=lm.y, z=lm.z, visibility=norm_landmarks[i].visibility
                )
                for i, lm in enumerate(world_landmarks)
            }

            individual = IndividualPose(
                individual_index=idx,
                normalized_landmarks=norm_dict,
                world_landmarks=world_dict,
            )
            individuals.append(individual)
        return individuals


class YOLOPoseEstimator(PoseEstimator):
    """YOLO implementation of pose estimation.

    Attributes:
        model_path (str): Path to the YOLO model file.
    """

    def __init__(self, model_type: str = "8n") -> None:
        """Initialize the YOLOPoseEstimator.

        Args:
            model_type (str): Type of YOLO model to use (e.g. '8n', '11p', '11s').
        """
        if not YOLO_AVAILABLE:
            raise ModuleNotFoundError(
                "`ultralytics` is not installed. "
                "Please install senselab video dependencies using `pip install senselab['video']`."
            )

        self.model_path = get_model("yolo", model_type)
        self._model = YOLO(self.model_path)

    def estimate(self, image: np.ndarray) -> ImagePose:
        """Estimate poses using YOLO.

        Args:
            image (np.ndarray): Input image in RGB format.

        Returns:
            ImagePose: An object containing detected poses and metadata.
        """
        results = self._model(image, verbose=False)

        if results[0].keypoints is None or results[0].keypoints.data.numel() == 0:
            # No individuals detected
            return ImagePose(
                image=image,
                individuals=[],
                model=PoseModel.YOLO,
            )

        individuals = []
        for idx, person_keypoints in enumerate(results[0].keypoints):
            confidence_values = person_keypoints.conf.squeeze()  # Extract confidence values

            normalized_dict = {
                YOLO_KEYPOINT_MAPPING.get(i, f"keypoint_{i}"): YOLOPoseLandmark(
                    x=kp[0].item(), y=kp[1].item(), confidence=confidence_values[i].item()
                )
                for i, kp in enumerate(person_keypoints.xyn[0])
            }

            individual = IndividualPose(
                individual_index=idx,
                normalized_landmarks=normalized_dict,
            )
            individuals.append(individual)

        return ImagePose(
            image=image,
            individuals=individuals,
            model=PoseModel.YOLO,
        )

    def estimate_from_path(self, image_path: str) -> ImagePose:
        """Estimate poses in image from file path using YOLO.

        Args:
            image_path (str): Path to the input image file.

        Returns:
            ImagePose: An object containing detected poses and metadata.

        Raises:
            FileNotFoundError: If the specified image file does not exist.
        """
        if not CV2_AVAILABLE:
            raise ModuleNotFoundError(
                "`opencv-python` is not installed. "
                "Please install senselab video dependencies using `pip install senselab['video']`."
            )

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")

        # Load and process the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.estimate(image)
