"""This module provides the API for face analysis tasks."""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from senselab.video.data_structures.video import Video
from senselab.video.tasks.face_analysis.deepface_utils import DeepFaceAnalysis


def recognize_faces(
    input_media: Union[str, np.ndarray, Video],
    db_path: str,
    model_name: Optional[str] = None,
    distance_metric: Optional[str] = None,
    backend: Optional[str] = None,
    align: Optional[bool] = None,
) -> List[List[pd.DataFrame]]:
    """Perform face recognition against a database of faces.

    Args:
        input_media (str | np.ndarray | Video): Media to recognize face(s) from.
        db_path (str): Path to the face database for recognition.
        model_name (Optional[str]): Face recognition model name (e.g., 'Facenet').
        distance_metric (Optional[str]): Distance metric ('cosine', 'euclidean', etc.).
        backend (Optional[str]): Face detection backend ('opencv', 'mtcnn', etc.).
        align (Optional[bool]): Align faces before analysis.

    Returns:
        List[List[pd.DataFrame]]: Nested list of DataFrames mapping detected faces to
            database matches for each image or Video frame.
    """
    face_analyzer = DeepFaceAnalysis(model_name, distance_metric, backend, align)

    if isinstance(input_media, (str, np.ndarray)):
        return [face_analyzer.recognize_faces(img_path=input_media, db_path=db_path)]

    elif isinstance(input_media, Video):
        return [
            face_analyzer.recognize_faces(img_path=frame.numpy() if hasattr(frame, "numpy") else frame, db_path=db_path)
            for frame in input_media.frames
        ]

    raise ValueError(
        "Unsupported input_media type. Must be a file path (str), an image array (np.ndarray), or a Video object."
    )


def verify_faces(
    img1: Union[str, np.ndarray],
    img2: Union[str, np.ndarray],
    model_name: Optional[str] = None,
    distance_metric: Optional[str] = None,
    backend: Optional[str] = None,
    align: Optional[bool] = None,
) -> Dict:
    """Verify if two images contain the same person's face.

    Args:
        img1 (Union[str, np.ndarray]): Path or array for the first image.
        img2 (Union[str, np.ndarray]): Path or array for the second image.
        model_name (Optional[str]): Face recognition model name.
        distance_metric (Optional[str]): Distance metric for verification.
        backend (Optional[str]): Face detection backend.
        align (Optional[bool]): Align faces before verification.

    Returns:
        Dict: Verification result containing similarity and match.
    """
    face_analyzer = DeepFaceAnalysis(model_name, distance_metric, backend, align)
    return face_analyzer.verify_faces(img1_path=img1, img2_path=img2)


def extract_face_embeddings(
    input_media: Union[str, np.ndarray, Video],
    model_name: Optional[str] = None,
    backend: Optional[str] = None,
    align: Optional[bool] = None,
) -> List[List[Dict]]:
    """Extract face embeddings from an image or video.

    Args:
        input_media (str | np.ndarray | Video): Media to extract face embeddings from.
        model_name (Optional[str]): Embedding extraction model name.
        backend (Optional[str]): Face detection backend.
        align (Optional[bool]): Align faces before extraction.

    Returns:
        List[List[Dict]]: Nested list of embeddings per face.
    """
    face_analyzer = DeepFaceAnalysis(model_name, backend=backend, align=align)

    if isinstance(input_media, (str, np.ndarray)):
        return [face_analyzer.extract_face_embeddings(input_media)]

    elif isinstance(input_media, Video):
        return [
            face_analyzer.extract_face_embeddings(frame.numpy() if hasattr(frame, "numpy") else frame)
            for frame in input_media.frames
        ]

    raise ValueError(
        "Unsupported input_media type. Must be a file path (str), an image array (np.ndarray), or a Video object."
    )


def analyze_face_attributes(
    input_media: Union[str, np.ndarray, Video],
    actions: Optional[List[str]] = None,
    backend: Optional[str] = None,
    align: Optional[bool] = None,
) -> List[List[Dict]]:
    """Analyze facial attributes (age, gender, emotion, race).

    Args:
        input_media (str | np.ndarray | Video): Media to analyze face attributes from.
        actions (Optional[List[str]]): Attributes to analyze (default: ['age', 'gender', 'emotion', 'race']).
        backend (Optional[str]): Face detection backend.
        align (Optional[bool]): Align faces before analysis.

    Returns:
        List[List[Dict]]: Nested list of attribute dictionaries per face.
    """
    face_analyzer = DeepFaceAnalysis(backend=backend, align=align)

    if isinstance(input_media, (str, np.ndarray)):
        return [face_analyzer.analyze_face_attributes(input_media, actions)]

    elif isinstance(input_media, Video):
        return [
            face_analyzer.analyze_face_attributes(frame.numpy() if hasattr(frame, "numpy") else frame, actions)
            for frame in input_media.frames
        ]

    raise ValueError(
        "Unsupported input_media type. Must be a file path (str), an image array (np.ndarray), or a Video object."
    )
