"""Tests for pose estimation data structures and functionality."""

import os
from typing import Generator

import cv2
import pytest
import torch

from senselab.video.data_structures.pose import PoseSkeleton, estimate_pose_with_mediapipe

TEST_IMAGES_DIR = "src/tests/data_for_testing/pose_data/"
VALID_SINGLE_PERSON_IMAGE = os.path.join(TEST_IMAGES_DIR, "single_person.jpg")
VALID_MULTIPLE_PEOPLE_IMAGE = os.path.join(TEST_IMAGES_DIR, "three_people.jpg")
NO_PEOPLE_IMAGE = os.path.join(TEST_IMAGES_DIR, "no_people.jpeg")
INVALID_IMAGE_PATH = "invalid/path/to/image.jpg"


def test_get_landmark_coordinates() -> None:
    """Tests basic landmark retrieval for first person."""
    result = estimate_pose_with_mediapipe(VALID_SINGLE_PERSON_IMAGE)

    # Test valid landmark retrieval
    coords = result.get_landmark_coordinates(landmark="landmark_0", person_index=0)
    assert len(coords) == 4, "Should return [x, y, z, visibility]"
    assert all(isinstance(x, float) for x in coords), "All coordinates should be floats"

    # Test normalized vs world coordinates
    norm_coords = result.get_landmark_coordinates(landmark="landmark_0", person_index=0, world=False)
    world_coords = result.get_landmark_coordinates(landmark="landmark_0", person_index=0, world=True)
    assert norm_coords != world_coords, "World and normalized coordinates should differ"


def test_invalid_person_index() -> None:
    """Tests error handling for invalid person indices."""
    result = estimate_pose_with_mediapipe(VALID_SINGLE_PERSON_IMAGE)

    with pytest.raises(ValueError) as exc_info:
        result.get_landmark_coordinates(person_index=5, landmark="landmark_0")
    assert "Person index 5 is invalid" in str(exc_info.value)
    assert "Image contains 1" in str(exc_info.value)

    # Test with no people
    result_empty = estimate_pose_with_mediapipe(NO_PEOPLE_IMAGE)
    with pytest.raises(ValueError) as exc_info:
        result_empty.get_landmark_coordinates(person_index=0, landmark="landmark_0")
    assert "Image contains 0 people" in str(exc_info.value)
    assert "Valid indices are none" in str(exc_info.value)


def test_invalid_landmark_name() -> None:
    """Tests error handling for invalid landmark names."""
    result = estimate_pose_with_mediapipe(VALID_SINGLE_PERSON_IMAGE)

    with pytest.raises(ValueError) as exc_info:
        result.get_landmark_coordinates(person_index=0, landmark="nonexistent_landmark")
    error_msg = str(exc_info.value)
    assert "Landmark 'nonexistent_landmark' not found" in error_msg
    assert "Available landmarks:" in error_msg


def test_valid_image_single_person() -> None:
    """Tests pose estimation on an image with a single person."""
    result = estimate_pose_with_mediapipe(VALID_SINGLE_PERSON_IMAGE)
    assert isinstance(result, PoseSkeleton), "Result should be an instance of PoseSkeleton"
    assert len(result.normalized_landmarks) == 1, "There should be one detected person"
    assert len(result.world_landmarks) == 1, "There should be one detected person"
    assert (
        result.image.shape == torch.from_numpy(cv2.imread(VALID_SINGLE_PERSON_IMAGE)).shape
    ), "Input and output image shapes should match"


def test_valid_image_multiple_people() -> None:
    """Tests pose estimation on an image with multiple people."""
    result = estimate_pose_with_mediapipe(VALID_MULTIPLE_PEOPLE_IMAGE, 3)
    assert isinstance(result, PoseSkeleton), "Result should be an instance of PoseSkeleton"
    assert len(result.normalized_landmarks) > 1, "There should be multiple detected people"
    assert len(result.world_landmarks) > 1, "There should be multiple detected people"


def test_no_people_in_image() -> None:
    """Tests pose estimation on an image with no people."""
    result = estimate_pose_with_mediapipe(NO_PEOPLE_IMAGE)
    assert isinstance(result, PoseSkeleton), "Result should be an instance of PoseSkeleton"
    assert len(result.normalized_landmarks) == 0, "No landmarks should be detected"
    assert len(result.world_landmarks) == 0, "No landmarks should be detected"


def test_invalid_image_path() -> None:
    """Tests pose estimation on an invalid image path."""
    with pytest.raises(Exception):
        estimate_pose_with_mediapipe(INVALID_IMAGE_PATH)


def test_visualization_single_person() -> None:
    """Tests visualization and saving of annotated images."""
    result = estimate_pose_with_mediapipe(VALID_SINGLE_PERSON_IMAGE)
    result.visualize_pose()
    assert os.path.exists("pose_estimation_output.png"), "Annotated image should be saved"


@pytest.fixture(autouse=True)
def cleanup() -> Generator[None, None, None]:
    """Clean up any generated files after tests.

    Yields:
        None
    """
    yield
    if os.path.exists("pose_estimation_output.png"):
        os.remove("pose_estimation_output.png")
