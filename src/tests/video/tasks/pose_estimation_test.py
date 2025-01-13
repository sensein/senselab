"""Module for testing Pose Estimation tasks."""

import os
import shutil
from typing import Generator, Optional

import cv2
import numpy as np
import pytest

from senselab.video.data_structures.pose import (
    ImagePose,
    IndividualPose,
    PoseModel,
    YOLOPoseLandmark,
)
from senselab.video.tasks.pose_estimation.estimate import MediaPipePoseEstimator, PoseEstimator, YOLOPoseEstimator
from senselab.video.tasks.pose_estimation.visualization import visualize_pose

# Test data
VALID_IMAGE = "src/tests/data_for_testing/pose_data/single_person.jpg"
MULTIPLE_PEOPLE_IMAGE = "src/tests/data_for_testing/pose_data/three_people.jpg"
NO_PEOPLE_IMAGE = "src/tests/data_for_testing/pose_data/no_people.jpeg"
INVALID_IMAGE_PATH = "invalid/path/to/image.jpg"


# Saved file paths (used for cleanup)
MODEL_PATH = "src/senselab/video/tasks/pose_estimation/models"


MEDIAPIPE_VALID_MODELS = ["full", "heavy"]
YOLO_VALID_MODELS = ["8s", "11l"]
MEDIAPIPE_INVALID_MODELS = ["invalid", "11n", 123]
YOLO_INVALID_MODELS = ["8r", "full", 123]


@pytest.fixture(scope="session", autouse=True)
def cleanup_models() -> Generator[None, None, None]:
    """Cleanup downloaded models the test session."""
    yield
    if os.path.exists(MODEL_PATH):
        shutil.rmtree(MODEL_PATH)


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


@pytest.mark.parametrize(
    "estimator_class, model_type, num_individuals",
    [
        (MediaPipePoseEstimator, "lite", 2),
        (YOLOPoseEstimator, "8n", None),  # YOLO doesn't use `num_individuals`
    ],
)
class TestPoseEstimators:
    """Test suite for pose estimators."""

    @pytest.fixture
    def estimator(self, estimator_class: type[PoseEstimator], model_type: str) -> PoseEstimator:
        """Create a pose estimator for testing."""
        return estimator_class(model_type)

    def _run_estimation(self, estimator: PoseEstimator, image_path: str, num_individuals: Optional[int]) -> ImagePose:
        """Helper function to run pose estimation."""
        if isinstance(estimator, MediaPipePoseEstimator):
            return estimator.estimate_from_path(image_path, num_individuals=num_individuals or 1)
        return estimator.estimate_from_path(image_path)

    @pytest.mark.parametrize(
        "image_path, expected_count",
        [
            (VALID_IMAGE, 1),
            (MULTIPLE_PEOPLE_IMAGE, 3),
            (NO_PEOPLE_IMAGE, 0),
        ],
    )
    def test_pose_estimation(
        self, estimator: PoseEstimator, image_path: str, expected_count: int, num_individuals: int
    ) -> None:
        """Test pose estimation on various images."""
        result = self._run_estimation(estimator, image_path, num_individuals)
        assert isinstance(result, ImagePose)
        expected_count = min(num_individuals, expected_count) if num_individuals else expected_count
        assert len(result.individuals) == expected_count

        if expected_count > 0:
            for individual in result.individuals:
                assert isinstance(individual, IndividualPose)
                assert len(individual.normalized_landmarks) > 0

    @pytest.mark.parametrize(
        "invalid_num_individuals",
        [None, -1, "3", 1.5],
    )
    def test_invalid_num_individuals(
        self, estimator: PoseEstimator, invalid_num_individuals: int, num_individuals: int
    ) -> None:
        """Test error handling for invalid number of individuals."""
        if isinstance(estimator, MediaPipePoseEstimator):
            with pytest.raises(ValueError):
                self._run_estimation(estimator, MULTIPLE_PEOPLE_IMAGE, invalid_num_individuals)  # type: ignore[arg-type]

    def test_invalid_image_path(self, estimator: PoseEstimator, num_individuals: int) -> None:
        """Test error handling for invalid image paths."""
        with pytest.raises(FileNotFoundError):
            self._run_estimation(estimator, INVALID_IMAGE_PATH, num_individuals)


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


@pytest.mark.parametrize("sample_pose", ["sample_pose_mediapipe", "sample_pose_yolo"])
def test_visualize_pose(sample_pose: ImagePose, request: pytest.FixtureRequest, tmpdir: pytest.TempPathFactory) -> None:
    """Test the visualization of poses for both MediaPipe and YOLO."""
    pose = request.getfixturevalue(sample_pose)
    output_path = os.path.join(tmpdir, f"{pose.model.name.lower()}.png")
    annotated_image = visualize_pose(pose, output_path=output_path)

    # Check the annotated image type
    assert isinstance(annotated_image, np.ndarray)
    assert annotated_image.shape == pose.image.shape

    # Check if the output file was created
    assert os.path.exists(output_path)

    # Verify that the output file can be opened
    loaded_image = cv2.imread(output_path)
    assert loaded_image is not None
