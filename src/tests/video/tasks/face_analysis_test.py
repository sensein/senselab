"""Module for testing Face Analysis tasks with real data."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from senselab.video.data_structures.video import Video
from senselab.video.tasks.face_analysis.api import (
    DetectedFace,
    FaceAttributes,
    FaceMatch,
    analyze_face_attributes,
    extract_face_embeddings,
    recognize_faces,
    verify_faces,
)

# Define constants for test paths
TEST_MEDIA_DIR = Path("src/tests/data_for_testing/face_data")
DB_DIR = TEST_MEDIA_DIR / "db"
IMAGE_PATH = TEST_MEDIA_DIR / "sally_1.jpg"
IMAGE_2_PATH = DB_DIR / "sally_2.jpg"
GROUP_IMAGE_PATH = DB_DIR / "group.jpg"
VIDEO_PATH = TEST_MEDIA_DIR / "sally_vid.mp4"


try:
    from deepface import DeepFace

    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@pytest.fixture
def sample_image_array() -> np.array:
    """Convert the test image to numpy array."""
    return cv2.imread(str(IMAGE_PATH))


@pytest.fixture
def sample_video() -> Video:
    """Load the test video into a Video object."""
    return Video.from_filepath(str(VIDEO_PATH))


@pytest.mark.skipif(not DEEPFACE_AVAILABLE or not CV2_AVAILABLE, reason="DeepFace or cv2 not available.")
@pytest.mark.parametrize("model_name", ["Facenet", "VGG-Face", "OpenFace"])
def test_recognize_faces_with_model_variations(model_name: str) -> None:
    """Test recognize_faces with various DeepFace model configurations."""
    results = recognize_faces(str(IMAGE_PATH), db_path=str(DB_DIR), model_name=model_name)

    assert isinstance(results, list)
    assert len(results) == 1
    assert all(isinstance(face, DetectedFace) for face in results[0])


@pytest.mark.skipif(not DEEPFACE_AVAILABLE or not CV2_AVAILABLE, reason="DeepFace or cv2 not available.")
def test_recognize_faces_group() -> None:
    """Test recognize_faces with a group photo that may have multiple faces."""
    results = recognize_faces(str(GROUP_IMAGE_PATH), db_path=str(DB_DIR))

    assert isinstance(results, list)
    assert len(results) == 1
    assert all(isinstance(face, DetectedFace) for face in results[0])


@pytest.mark.skipif(not DEEPFACE_AVAILABLE or not CV2_AVAILABLE, reason="DeepFace or cv2 not available.")
def test_recognize_faces_ndarray(sample_image_array: np.array) -> None:
    """Test recognize_faces with a numpy array image."""
    results = recognize_faces(sample_image_array, db_path=str(DB_DIR))

    assert isinstance(results, list)
    assert len(results) == 1
    assert all(isinstance(face, DetectedFace) for face in results[0])


def test_recognize_faces_video(sample_video: Video) -> None:
    """Test recognize_faces with a Video object."""
    frame_sample_rate = 2.0
    results = recognize_faces(
        sample_video, db_path=str(DB_DIR), frame_sample_rate=frame_sample_rate, enforce_detection=False
    )

    expected_sampled = math.ceil(len(sample_video.frames) / (sample_video.frame_rate / frame_sample_rate))
    assert isinstance(results, list)
    assert len(results) == expected_sampled

    if results and results[0]:
        assert all(isinstance(face, DetectedFace) for face in results[0])


@pytest.mark.skipif(not DEEPFACE_AVAILABLE or not CV2_AVAILABLE, reason="DeepFace or cv2 not available.")
def test_verify_faces_same() -> None:
    """Test verify_faces function with the same image (should be verified)."""
    # Comparing an image with itself should return verified=True
    result = verify_faces(str(IMAGE_PATH), str(IMAGE_PATH))

    assert isinstance(result, DetectedFace)
    assert isinstance(result.face_match, list)
    assert len(result.face_match) == 1
    assert isinstance(result.face_match[0], FaceMatch)
    assert result.face_match[0].verified is True


@pytest.mark.skipif(not DEEPFACE_AVAILABLE or not CV2_AVAILABLE, reason="DeepFace or cv2 not available.")
def test_verify_faces_different() -> None:
    """Test verify_faces function with two different images of the same person."""
    result = verify_faces(str(IMAGE_PATH), str(IMAGE_2_PATH))

    assert isinstance(result, DetectedFace)
    assert isinstance(result.face_match, list)
    assert len(result.face_match) == 1
    assert isinstance(result.face_match[0], FaceMatch)
    assert result.face_match[0].verified is True


@pytest.mark.skipif(not DEEPFACE_AVAILABLE or not CV2_AVAILABLE, reason="DeepFace or cv2 not available.")
def test_extract_face_embeddings_str() -> None:
    """Test extract_face_embeddings with a single face image."""
    results = extract_face_embeddings(str(IMAGE_PATH))

    assert isinstance(results, list)
    assert len(results) == 1
    assert all(isinstance(face, DetectedFace) for face in results[0])
    assert all(isinstance(face.embedding, list) for face in results[0])


@pytest.mark.skipif(not DEEPFACE_AVAILABLE or not CV2_AVAILABLE, reason="DeepFace or cv2 not available.")
def test_extract_face_embeddings_group() -> None:
    """Test extract_face_embeddings with a group photo."""
    results = extract_face_embeddings(str(GROUP_IMAGE_PATH))

    assert isinstance(results, list)
    assert len(results) == 1

    # Group photo should have multiple face embeddings
    assert len(results[0]) >= 1
    assert all(isinstance(face, DetectedFace) for face in results[0])
    assert all(isinstance(face.embedding, list) for face in results[0])


def test_extract_face_embeddings_video(sample_video: Video) -> None:
    """Test extract_face_embeddings with a Video object."""
    frame_sample_rate = 2.0
    results = extract_face_embeddings(sample_video, frame_sample_rate=frame_sample_rate, enforce_detection=False)

    expected_sampled = math.ceil(len(sample_video.frames) / (sample_video.frame_rate / frame_sample_rate))
    assert isinstance(results, list)
    assert len(results) == expected_sampled

    if results and results[0]:
        assert all(isinstance(face, DetectedFace) for face in results[0])
        assert all(isinstance(face.embedding, list) for face in results[0])


@pytest.mark.skipif(not DEEPFACE_AVAILABLE or not CV2_AVAILABLE, reason="DeepFace or cv2 not available.")
def test_analyze_face_attributes_single() -> None:
    """Test analyze_face_attributes with a single face image."""
    results = analyze_face_attributes(str(IMAGE_PATH), actions=["age", "gender"])

    assert isinstance(results, list)
    assert len(results) == 1
    assert all(isinstance(face, DetectedFace) for face in results[0])
    for face in results[0]:
        assert face.attributes is not None
        assert isinstance(face.attributes.age, int)
        assert isinstance(face.attributes.dominant_gender, str)


@pytest.mark.skipif(not DEEPFACE_AVAILABLE or not CV2_AVAILABLE, reason="DeepFace or cv2 not available.")
def test_analyze_face_attributes_all() -> None:
    """Test analyze_face_attributes with all possible attributes."""
    # Test with all default actions
    results = analyze_face_attributes(str(IMAGE_PATH), actions=["age", "gender", "emotion", "race"])

    assert isinstance(results, list)
    assert len(results) == 1
    assert all(isinstance(face, DetectedFace) for face in results[0])
    for face in results[0]:
        assert face.attributes is not None
        assert isinstance(face.attributes.age, int)
        assert isinstance(face.attributes.dominant_gender, str)
        assert isinstance(face.attributes.dominant_emotion, str)
        assert isinstance(face.attributes.dominant_race, str)


def test_analyze_face_attributes_video(sample_video: Video) -> None:
    """Test analyze_face_attributes with a Video object."""
    frame_sample_rate = 2.0
    results = analyze_face_attributes(
        sample_video, actions=["age", "gender"], frame_sample_rate=frame_sample_rate, enforce_detection=False
    )

    expected_sampled = math.ceil(len(sample_video.frames) / (sample_video.frame_rate / frame_sample_rate))
    assert isinstance(results, list)
    assert len(results) == expected_sampled

    if results and results[0]:
        assert all(isinstance(face, DetectedFace) for face in results[0])
        assert all(isinstance(face.attributes, FaceAttributes) for face in results[0])
        for face in results[0]:
            assert face.attributes is not None
            assert isinstance(face.attributes.age, int)
            assert isinstance(face.attributes.dominant_gender, str)
