"""Module for testing Pose Estimation tasks."""

import pytest

from senselab.video.data_structures.pose import ImagePose, IndividualPose
from senselab.video.tasks.pose_estimation.estimate import MediaPipePoseEstimator, PoseEstimator, YOLOPoseEstimator

# Test data
VALID_IMAGE = "src/tests/data_for_testing/pose_data/single_person.jpg"
MULTIPLE_PEOPLE_IMAGE = "src/tests/data_for_testing/pose_data/three_people.jpg"
NO_PEOPLE_IMAGE = "src/tests/data_for_testing/pose_data/no_people.jpeg"
INVALID_IMAGE_PATH = "invalid/path/to/image.jpg"


@pytest.mark.parametrize(
    "estimator_class, estimator_kwargs, num_individuals",
    [
        (
            MediaPipePoseEstimator,
            {"model_path": "src/senselab/video/tasks/pose_estimation/models/pose_landmarker.task"},
            2,
        ),
        (YOLOPoseEstimator, {"model_path": "yolov8n-pose.pt"}, None),  # YOLO doesn't use `num_individuals`
    ],
)
class TestPoseEstimators:
    """Refactored test suite for pose estimators."""

    @pytest.fixture
    def estimator(self, estimator_class: type[PoseEstimator], estimator_kwargs: dict) -> PoseEstimator:
        """Create a pose estimator for testing."""
        return estimator_class(**estimator_kwargs)

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
