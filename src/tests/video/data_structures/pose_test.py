"""Tests for pose estimation data structures and functionality."""

import os
from typing import Generator

import numpy as np
import pytest

from senselab.video.data_structures.pose import ImagePose, IndividualPose
from senselab.video.tasks.pose_estimation.pose_estimation import estimate_pose_with_mediapipe

TEST_IMAGES_DIR = "src/tests/data_for_testing/pose_data/"
VALID_SINGLE_PERSON_IMAGE = os.path.join(TEST_IMAGES_DIR, "single_person.jpg")
VALID_MULTIPLE_PEOPLE_IMAGE = os.path.join(TEST_IMAGES_DIR, "three_people.jpg")
NO_PEOPLE_IMAGE = os.path.join(TEST_IMAGES_DIR, "no_people.jpeg")
INVALID_IMAGE_PATH = "invalid/path/to/image.jpg"


class TestIndividualPose:
    """Test suite for IndividualPose class."""

    @pytest.fixture
    def sample_individual(self) -> IndividualPose:
        """Create a sample IndividualPose for testing."""
        return IndividualPose(
            individual_index=0,
            normalized_landmarks={"landmark_0": [0.5, 0.5, 0.0, 1.0], "landmark_1": [0.6, 0.6, 0.1, 0.9]},
            world_landmarks={"landmark_0": [0.5, 1.5, 0.0, 1.0], "landmark_1": [0.6, 1.6, 0.1, 0.9]},
        )

    def test_landmark_validation(self) -> None:
        """Test landmark validation during initialization."""
        # Test valid initialization
        valid_pose = IndividualPose(
            individual_index=0,
            normalized_landmarks={"landmark_0": [0.1, 0.2, 0.3, 0.4]},
            world_landmarks={"landmark_0": [0.1, 0.2, 0.3, 0.4]},
        )
        assert valid_pose is not None

        # Test invalid number of coordinates
        with pytest.raises(ValueError) as exc_info:
            IndividualPose(
                individual_index=0,
                normalized_landmarks={"landmark_0": [0.1, 0.2, 0.3]},
                world_landmarks={"landmark_0": [0.1, 0.2, 0.3, 0.4]},
            )
        assert "Each landmark must have exactly 4 coordinates" in str(exc_info.value)

    def test_get_landmark_coordinates(self, sample_individual: IndividualPose) -> None:
        """Test landmark coordinate retrieval."""
        # Test normalized coordinates
        coords = sample_individual.get_landmark_coordinates("landmark_0", world=False)
        assert len(coords) == 4
        assert coords == [0.5, 0.5, 0.0, 1.0]

        # Test world coordinates
        world_coords = sample_individual.get_landmark_coordinates("landmark_0", world=True)
        assert len(world_coords) == 4
        assert world_coords == [0.5, 1.5, 0.0, 1.0]

        # Test invalid landmark
        with pytest.raises(ValueError) as exc_info:
            sample_individual.get_landmark_coordinates("nonexistent_landmark")
        assert "Landmark 'nonexistent_landmark' not found" in str(exc_info.value)
        assert "Available landmarks:" in str(exc_info.value)


class TestImagePose:
    """Test suite for ImagePose class."""

    @pytest.fixture
    def sample_image_pose(self) -> ImagePose:
        """Create a sample ImagePose for testing."""
        individual = IndividualPose(
            individual_index=0,
            normalized_landmarks={"landmark_0": [0.5, 0.5, 0.0, 1.0]},
            world_landmarks={"landmark_0": [0.5, 1.5, 0.0, 1.0]},
        )
        return ImagePose(image=np.zeros((100, 100, 3)), individuals=[individual])

    def test_get_individual(self, sample_image_pose: ImagePose) -> None:
        """Test individual retrieval."""
        # Test valid index
        individual = sample_image_pose.get_individual(0)
        assert isinstance(individual, IndividualPose)
        assert individual.individual_index == 0

        # Test invalid index
        with pytest.raises(ValueError) as exc_info:
            sample_image_pose.get_individual(1)
        assert "Individual index 1 is invalid" in str(exc_info.value)

    def test_empty_image_pose(self) -> None:
        """Test ImagePose with no individuals."""
        empty_pose = ImagePose(image=np.zeros((100, 100, 3)), individuals=[])
        with pytest.raises(ValueError) as exc_info:
            empty_pose.get_individual(0)
        assert "Valid indices are none" in str(exc_info.value)


class TestIntegration:
    """Integration tests with MediaPipe."""

    def test_mediapipe_single_person(self) -> None:
        """Test full pipeline with single person image."""
        result = estimate_pose_with_mediapipe(VALID_SINGLE_PERSON_IMAGE)
        assert isinstance(result, ImagePose)
        assert len(result.individuals) == 1

        individual = result.get_individual(0)
        assert isinstance(individual, IndividualPose)
        assert individual.individual_index == 0
        assert len(individual.normalized_landmarks) > 0
        assert len(individual.world_landmarks) > 0

    def test_mediapipe_multiple_people(self) -> None:
        """Test full pipeline with multiple people image."""
        result = estimate_pose_with_mediapipe(VALID_MULTIPLE_PEOPLE_IMAGE, num_of_individuals=3)
        assert isinstance(result, ImagePose)
        assert len(result.individuals) == 3

    def test_mediapipe_no_people(self) -> None:
        """Test full pipeline with image containing no people."""
        result = estimate_pose_with_mediapipe(NO_PEOPLE_IMAGE)
        assert isinstance(result, ImagePose)
        assert len(result.individuals) == 0

    def test_invalid_image_path(self) -> None:
        """Test error handling for invalid image path."""
        with pytest.raises(FileNotFoundError):
            estimate_pose_with_mediapipe(INVALID_IMAGE_PATH)


@pytest.fixture(autouse=True)
def cleanup() -> Generator[None, None, None]:
    """Clean up any generated files after tests."""
    yield
    if os.path.exists("pose_estimation_output.png"):
        os.remove("pose_estimation_output.png")
