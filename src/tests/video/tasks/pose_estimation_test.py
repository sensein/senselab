"""Module for testing Pose Estimation tasks."""

import os
from typing import Dict, Optional, Union

import numpy as np
import pytest

from senselab.video.data_structures.pose import (
    ImagePose,
    IndividualPose,
    PoseModel,
    YOLOPoseLandmark,
)
from senselab.video.tasks.pose_estimation import estimate_pose, visualize_pose
from senselab.video.tasks.pose_estimation.estimate import MediaPipePoseEstimator, PoseEstimator, YOLOPoseEstimator

try:
    import cv2

    CV2_AVAILABLE = True
except ModuleNotFoundError:
    CV2_AVAILABLE = False

from senselab.utils.data_structures.docker import docker_is_running

if docker_is_running():
    DOCKER_AVAILABLE = True
else:
    DOCKER_AVAILABLE = False


try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ModuleNotFoundError:
    YOLO_AVAILABLE = False


# Test data
VALID_IMAGE = os.path.abspath("src/tests/data_for_testing/pose_data/single_person.jpg")
MULTIPLE_PEOPLE_IMAGE = os.path.abspath("src/tests/data_for_testing/pose_data/three_people.jpg")
NO_PEOPLE_IMAGE = os.path.abspath("src/tests/data_for_testing/pose_data/no_people.jpeg")
INVALID_IMAGE_PATH = "invalid/path/to/image.jpg"


# Saved file paths (used for cleanup)
MODEL_PATH = os.path.abspath("src/senselab/video/tasks/pose_estimation/models")

# Parameters for testing
MEDIAPIPE_VALID_MODELS = ["full", "heavy"]
YOLO_VALID_MODELS = ["8s", "11l"]
MEDIAPIPE_INVALID_MODELS = ["invalid", "11n", 123]
YOLO_INVALID_MODELS = ["8r", "full", 123]


@pytest.fixture
def sample_pose_mediapipe() -> ImagePose:
    """Create a MediaPipe ImagePose for testing."""
    mp = MediaPipePoseEstimator("lite")
    pose = mp.estimate_from_path(VALID_IMAGE)
    return pose


@pytest.fixture
def sample_pose_yolo() -> ImagePose:
    """Create a sample YOLO ImagePose for testing."""
    image = np.zeros((300, 300, 3), dtype=np.uint8)

    individuals = [
        IndividualPose(
            individual_index=0,
            normalized_landmarks={
                "landmark_0": YOLOPoseLandmark(x=150, y=150, confidence=0.95),
                "landmark_1": YOLOPoseLandmark(x=180, y=180, confidence=0.85),
            },
        ),
        IndividualPose(
            individual_index=1,
            normalized_landmarks={
                "landmark_0": YOLOPoseLandmark(x=200, y=200, confidence=0.95),
                "landmark_1": YOLOPoseLandmark(x=230, y=230, confidence=0.85),
            },
        ),
    ]
    return ImagePose(image=image, individuals=individuals, model=PoseModel.YOLO)


@pytest.mark.skipif(DOCKER_AVAILABLE, reason="Docker is installed and running.")
def test_media_pipe_unavailable() -> None:
    """Test MediaPipePoseEstimator import error."""
    with pytest.raises(RuntimeError):
        MediaPipePoseEstimator("full")


@pytest.mark.skipif(YOLO_AVAILABLE, reason="YOLO is installed.")
def test_yolo_unavailable() -> None:
    """Test YOLOPoseEstimator import error."""
    with pytest.raises(ModuleNotFoundError):
        YOLOPoseEstimator("8n")


@pytest.mark.skipif(not DOCKER_AVAILABLE or not YOLO_AVAILABLE, 
                    reason="Docker is not running or YOLO is not installed.")
@pytest.mark.parametrize(
    "model, model_type, num_individuals",
    [
        ("mediapipe", "lite", 2),
        ("yolo", "8n", None),  # YOLO doesn't use `num_individuals`
    ],
)
class TestPoseEstimators:
    """Test suite for pose estimators."""

    def _run_estimation(
        self, model: str, image_path: str, model_type: str, num_individuals: Optional[int]
    ) -> ImagePose:
        """Helper function to run pose estimation using the API."""
        kwargs: Dict[str, Union[str, int]] = {"model_type": model_type}
        if num_individuals is not None:
            kwargs["num_individuals"] = num_individuals
        return estimate_pose(image_path, model, **{k: v for k, v in kwargs.items() if v is not None})

    @pytest.mark.parametrize(
        "image_path, expected_count",
        [
            (VALID_IMAGE, 1),
            (MULTIPLE_PEOPLE_IMAGE, 3),
            (NO_PEOPLE_IMAGE, 0),
        ],
    )
    def test_pose_estimation(
        self, model: str, model_type: str, image_path: str, expected_count: int, num_individuals: int
    ) -> None:
        """Test pose estimation on various images using the API."""
        result = self._run_estimation(model, image_path, model_type, num_individuals)
        assert isinstance(result, ImagePose)
        expected_count = min(num_individuals, expected_count) if num_individuals else expected_count
        assert len(result.individuals) == expected_count

        if expected_count > 0:
            for individual in result.individuals:
                assert isinstance(individual, IndividualPose)
                assert len(individual.normalized_landmarks) > 0

    @pytest.mark.parametrize(
        "invalid_num_individuals",
        [-1, "3", 1.5],
    )
    def test_invalid_num_individuals(
        self, model: str, model_type: str, 
        invalid_num_individuals: int, 
        num_individuals: int
    ) -> None:
        """Test error handling for invalid number of individuals using the API."""
        if model == "mediapipe":
            with pytest.raises(ValueError):
                self._run_estimation(model, 
                                     MULTIPLE_PEOPLE_IMAGE, 
                                     model_type, 
                                     invalid_num_individuals)

    def test_invalid_image_path(self, model: str, model_type: str, num_individuals: int) -> None:
        """Test error handling for invalid image paths using the API."""
        with pytest.raises(FileNotFoundError):
            self._run_estimation(model, INVALID_IMAGE_PATH, model_type, num_individuals)


@pytest.mark.skipif(not DOCKER_AVAILABLE or not YOLO_AVAILABLE, 
                    reason="Docker is not running or YOLO is not installed.")
@pytest.mark.parametrize(
    "estimator_class, valid_model_types, invalid_model_types",
    [
        (MediaPipePoseEstimator, MEDIAPIPE_VALID_MODELS, MEDIAPIPE_INVALID_MODELS),
        (YOLOPoseEstimator, YOLO_VALID_MODELS, YOLO_INVALID_MODELS),
    ],
)
def test_model_types(
    estimator_class: type[PoseEstimator],
    valid_model_types: list,
    invalid_model_types: list,
) -> None:
    """Test valid and invalid model types for each estimator."""
    # Test valid models
    for model_type in valid_model_types:
        estimator = estimator_class(model_type)
        assert isinstance(estimator, PoseEstimator)
        assert os.path.exists(estimator.model_path)

    # Test invalid models
    for invalid_model_type in invalid_model_types:
        with pytest.raises(ValueError):
            estimator_class(invalid_model_type)


@pytest.mark.skipif(not DOCKER_AVAILABLE or not YOLO_AVAILABLE, 
                    reason="Docker is not running or YOLO is not installed.")
@pytest.mark.parametrize("sample_pose", ["sample_pose_mediapipe", 
                                         "sample_pose_yolo"])
def test_visualize_pose(sample_pose: str, 
                        request: pytest.FixtureRequest, 
                        tmpdir: pytest.TempPathFactory) -> None:
    """Test the visualization of poses for both MediaPipe and YOLO."""
    pose = request.getfixturevalue(sample_pose)
    output_path = os.path.join(str(tmpdir), f"{pose.model.name.lower()}.png")
    annotated_image = visualize_pose(pose, output_path=output_path)

    # Check the annotated image type
    assert isinstance(annotated_image, np.ndarray)
    assert annotated_image.shape == pose.image.shape

    # Check if the output file was created
    assert os.path.exists(output_path)

    # Verify that the output file can be opened
    loaded_image = cv2.imread(output_path)
    assert loaded_image is not None
