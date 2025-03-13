"""Module for testing Face Analysis tasks with real data."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from senselab.video.data_structures.video import Video
from senselab.video.tasks.face_analysis.api import (
    analyze_face_attributes,
    extract_face_embeddings,
    recognize_faces,
    verify_faces,
)

# Define constants for test paths
TEST_IMAGES_DIR = Path("src/tests/data_for_testing/face_data")
DB_DIR = TEST_IMAGES_DIR / "db"
IMAGE_PATH = TEST_IMAGES_DIR / "sally_2.jpg"
IMAGE_2_PATH = DB_DIR / "sally_1.jpg"
GROUP_IMAGE_PATH = DB_DIR / "group_of_people.jpg"


@pytest.fixture
def sample_image_array() -> np.array:
    """Convert the test image to numpy array."""
    import cv2

    return cv2.imread(str(IMAGE_PATH))


# @pytest.fixture
# def sample_video(sample_image_array) -> Video:
#     """Create a sample video from the test image."""
#     # Create a simple 2-frame video from the same image (duplicated)
#     frames = [sample_image_array, sample_image_array]
#     return Video(frames=frames, frame_rate=25.0, audio=None)


def test_recognize_faces_str() -> None:
    """Test recognize_faces with a single face image path."""
    # Call the function with the real image path
    results = recognize_faces(str(IMAGE_PATH), db_path=str(DB_DIR))

    # Verify results structure
    assert isinstance(results, list)
    assert len(results) == 1  # One result for the single image

    # The inner result should be a list of DataFrames for found faces
    inner_result = results[0]
    assert isinstance(inner_result, list)

    # If faces are found, check the DataFrame structure
    if inner_result:
        assert isinstance(inner_result[0], pd.DataFrame)


def test_recognize_faces_group() -> None:
    """Test recognize_faces with a group photo that may have multiple faces."""
    results = recognize_faces(str(GROUP_IMAGE_PATH), db_path=str(DB_DIR))

    assert isinstance(results, list)
    assert len(results) == 1  # One result for the single image

    # The inner result should be a list of DataFrames for found faces
    inner_result = results[0]
    assert isinstance(inner_result, list)

    # Group photo may have multiple face results
    if inner_result:
        assert isinstance(inner_result[0], pd.DataFrame)


def test_recognize_faces_ndarray(sample_image_array: np.array) -> None:
    """Test recognize_faces with a numpy array image."""
    results = recognize_faces(sample_image_array, db_path=str(DB_DIR))

    assert isinstance(results, list)
    assert len(results) == 1  # One result for the single image


# def test_recognize_faces_video(sample_video):
#     """
#     Test recognize_faces with a Video object.
#     """
#     results = recognize_faces(sample_video, db_path=str(DB_DIR))

#     # We should get a list with length equal to the number of frames
#     assert isinstance(results, list)
#     assert len(results) == len(sample_video.frames)

#     # Each frame result should be a list (possibly empty if no faces found)
#     for frame_result in results:
#         assert isinstance(frame_result, list)


def test_verify_faces_same() -> None:
    """Test verify_faces function with the same image (should be verified)."""
    # Comparing an image with itself should return verified=True
    result = verify_faces(str(IMAGE_PATH), str(IMAGE_PATH))

    assert isinstance(result, dict)
    assert "verified" in result
    assert result["verified"] is True


def test_verify_faces_different() -> None:
    """Test verify_faces function with two different images of the same person.

    Note: This might pass or fail depending on how similar the images are
    and the threshold settings.
    """
    result = verify_faces(str(IMAGE_PATH), str(IMAGE_2_PATH))

    assert isinstance(result, dict)
    assert "verified" in result


def test_extract_face_embeddings_str() -> None:
    """Test extract_face_embeddings with a single face image."""
    results = extract_face_embeddings(str(IMAGE_PATH))

    assert isinstance(results, list)
    assert len(results) == 1  # One result for the single image

    # Check embedding structure if a face was found
    inner_result = results[0]
    assert isinstance(inner_result, list)

    if inner_result:
        face_embedding = inner_result[0]
        assert isinstance(face_embedding, dict)
        assert "embedding" in face_embedding
        assert isinstance(face_embedding["embedding"], list)
        assert all(isinstance(x, (int, float)) for x in face_embedding["embedding"])


def test_extract_face_embeddings_group() -> None:
    """Test extract_face_embeddings with a group photo."""
    results = extract_face_embeddings(str(GROUP_IMAGE_PATH))

    assert isinstance(results, list)
    assert len(results) == 1  # One result for the single image

    # Check embedding structure
    inner_result = results[0]
    assert isinstance(inner_result, list)

    # Group photo may have multiple face embeddings
    if inner_result:
        assert len(inner_result) >= 1  # Should find at least one face
        for face_embedding in inner_result:
            assert isinstance(face_embedding, dict)
            assert "embedding" in face_embedding


# def test_extract_face_embeddings_video(sample_video):
#     """
#     Test extract_face_embeddings with a Video object.
#     """
#     results = extract_face_embeddings(sample_video)

#     # There should be results for each frame
#     assert isinstance(results, list)
#     assert len(results) == len(sample_video.frames)

#     # Each frame result should be a list of embeddings
#     for frame_result in results:
#         assert isinstance(frame_result, list)


def test_analyze_face_attributes_single() -> None:
    """Test analyze_face_attributes with a single face image."""
    results = analyze_face_attributes(str(IMAGE_PATH), actions=["age", "gender"])

    assert isinstance(results, list)
    assert len(results) == 1  # One result for the single image

    # Check attributes structure
    inner_result = results[0]
    assert isinstance(inner_result, list)

    if inner_result:
        face_attributes = inner_result[0]
        assert isinstance(face_attributes, dict)
        assert "age" in face_attributes
        assert "gender" in face_attributes


def test_analyze_face_attributes_all() -> None:
    """Test analyze_face_attributes with all possible attributes."""
    # Test with all default actions
    results = analyze_face_attributes(str(IMAGE_PATH), actions=["age", "gender", "emotion", "race"])

    assert isinstance(results, list)
    inner_result = results[0]

    if inner_result:
        face_attributes = inner_result[0]
        assert isinstance(face_attributes, dict)
        # Check that all requested attributes are present
        assert "age" in face_attributes
        assert "gender" in face_attributes
        assert "emotion" in face_attributes
        assert "race" in face_attributes


def test_analyze_face_attributes_group() -> None:
    """Test analyze_face_attributes with a group photo."""
    results = analyze_face_attributes(str(GROUP_IMAGE_PATH), actions=["age", "gender"])

    assert isinstance(results, list)
    assert len(results) == 1  # One result for the single image

    # Check attributes structure
    inner_result = results[0]
    assert isinstance(inner_result, list)

    # Group photo should have multiple face results
    if inner_result:
        assert len(inner_result) >= 1  # Should find at least one face
        for face_attributes in inner_result:
            assert isinstance(face_attributes, dict)
            assert "age" in face_attributes
            assert "gender" in face_attributes


# def test_analyze_face_attributes_video(sample_video):
#     """
#     Test analyze_face_attributes with a Video object.
#     """
#     results = analyze_face_attributes(sample_video, actions=["age", "gender"])

#     # There should be results for each frame
#     assert isinstance(results, list)
#     assert len(results) == len(sample_video.frames)

#     # Each frame result should be a list of attribute dictionaries
#     for frame_result in results:
#         assert isinstance(frame_result, list)


# def test_model_variations():
#     """
#     Test different model configurations using the deepface_args parameter.
#     """
#     # Test with different detection and recognition models
#     models_to_test = [
#         {"model_name": "Facenet"},
#         {"model_name": "VGG-Face"},
#         {"model_name": "OpenFace"},
#         {"model_name": "DeepFace"}
#     ]

#     for model_config in models_to_test:
#         try:
#             result = verify_faces(str(IMAGE_PATH), str(IMAGE_PATH), deepface_args=model_config)

#             assert isinstance(result, dict)
#             assert "verified" in result
#             assert result["verified"] is True
#             # Print success message
#             print(f"Successfully tested model: {model_config['model_name']}")

#         except Exception as e:
#             # Some models might not be available, so we'll just log the error
#             print(f"Error with model {model_config['model_name']}: {str(e)}")
#             continue
