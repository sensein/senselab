"""Module for testing Pose Estimation tasks."""

import os
import shutil
from typing import Generator

import pytest

from senselab.video.data_structures.pose import ImagePose, IndividualPose
from senselab.video.tasks.pose_estimation.estimate import MediaPipePoseEstimator, PoseEstimator, YOLOPoseEstimator

# Test data
VALID_IMAGE = "src/tests/data_for_testing/pose_data/single_person.jpg"
MULTIPLE_PEOPLE_IMAGE = "src/tests/data_for_testing/pose_data/three_people.jpg"
NO_PEOPLE_IMAGE = "src/tests/data_for_testing/pose_data/no_people.jpeg"
INVALID_IMAGE_PATH = "invalid/path/to/image.jpg"


# Model folder path (used for cleanup)
MODEL_PATH = "src/senselab/video/tasks/pose_estimation/models"


@pytest.fixture(scope="session", autouse=True)
def cleanup_models() -> Generator[None, None, None]:
    """Cleanup downloaded models the test session."""
    yield
    if os.path.exists(MODEL_PATH):
        shutil.rmtree(MODEL_PATH)


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

    def test_valid_image(self, estimator: PoseEstimator, num_individuals: int) -> None:
        """Test pose estimation on a valid image."""
        if isinstance(estimator, MediaPipePoseEstimator):
            result = estimator.estimate_from_path(VALID_IMAGE, num_individuals=num_individuals)
        else:
            result = estimator.estimate_from_path(VALID_IMAGE)

        assert isinstance(result, ImagePose)
        expected_count = min(num_individuals, 1) if num_individuals else 1
        assert len(result.individuals) == expected_count

        individual = result.get_individual(0)
        assert isinstance(individual, IndividualPose)
        assert len(individual.normalized_landmarks) > 0

    def test_multiple_people_image(self, estimator: PoseEstimator, num_individuals: int) -> None:
        """Test pose estimation on an image with multiple people."""
        if isinstance(estimator, MediaPipePoseEstimator):
            result = estimator.estimate_from_path(MULTIPLE_PEOPLE_IMAGE, num_individuals=num_individuals)
        else:
            result = estimator.estimate_from_path(MULTIPLE_PEOPLE_IMAGE)

        assert isinstance(result, ImagePose)
        expected_count = min(num_individuals, 3) if num_individuals else 3
        assert len(result.individuals) == expected_count

        individual = result.get_individual(0)
        assert isinstance(individual, IndividualPose)
        assert len(individual.normalized_landmarks) > 0

    def test_no_people_image(self, estimator: PoseEstimator, num_individuals: int) -> None:
        """Test pose estimation on an image with no people."""
        if isinstance(estimator, MediaPipePoseEstimator):
            result = estimator.estimate_from_path(NO_PEOPLE_IMAGE, num_individuals=num_individuals)
        else:
            result = estimator.estimate_from_path(NO_PEOPLE_IMAGE)

        assert isinstance(result, ImagePose)
        assert len(result.individuals) == 0

    def test_invalid_num_individuals(self, estimator: PoseEstimator, num_individuals: int) -> None:
        """Test error handling for invalid number of individuals."""
        if isinstance(estimator, MediaPipePoseEstimator):
            for test_num in [None, -1, "3", 1.5]:
                with pytest.raises(ValueError):
                    estimator.estimate_from_path(MULTIPLE_PEOPLE_IMAGE, num_individuals=test_num)  # type: ignore[arg-type]

    def test_invalid_image_path(self, estimator: PoseEstimator, num_individuals: int) -> None:
        """Test error handling for invalid image paths."""
        with pytest.raises(FileNotFoundError):
            if isinstance(estimator, MediaPipePoseEstimator):
                estimator.estimate_from_path(INVALID_IMAGE_PATH, num_individuals=num_individuals)
            else:
                estimator.estimate_from_path(INVALID_IMAGE_PATH)


@pytest.mark.parametrize(
    "estimator_class, valid_model_types, invalid_model_types",
    [
        (MediaPipePoseEstimator, ["full", "heavy"], ["invalid", "11n", 123]),
        (YOLOPoseEstimator, ["8s", "11n"], ["8r", "full", 123]),
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
