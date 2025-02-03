"""This module provides the API for face analysis tasks."""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..data_structures.video import Video
from .deepface_utils import DeepFaceAnalysis


def recognize_faces(
    input_media: Union[str, np.ndarray, Video], db_path: str, deepface_args: Optional[Dict] = None
) -> List[List[pd.DataFrame]]:
    """Perform face recognition on an image, Video, or specific video frames against a database of faces.

    Args:
        input_media (Union[str, np.ndarray, Video]):
            - str: Path to an image file.
            - np.ndarray: Image array.
            - Video: Video object; analyze all frames in the video.
        db_path (str): Path to the face database for recognition.
        deepface_args (Optional[Dict]): Optional dictionary of arguments
            to pass to DeepFaceAnalysis.

    Returns:
        List[List[pd.DataFrame]]:
            A list where each element is a list of pandas DataFrames
            corresponding to a single image or frame. Each list of
            DataFrames maps detected faces to their closest matches
            in the database.

    Raises:
        ValueError: If the input_media type is not supported.
    """
    face_analyzer = DeepFaceAnalysis.from_dict(deepface_args)

    # Handle single image by path or array
    if isinstance(input_media, (str, np.ndarray)):
        dfs_for_faces = face_analyzer.recognize_faces(img_path=input_media, db_path=db_path)
        return [dfs_for_faces]

    # Handle Video: loop through frames
    elif isinstance(input_media, Video):
        results_for_frames = []

        for frame_idx in range(input_media.frames.shape[0]):
            frame = input_media.frames[frame_idx]

            # Convert to NumPy if it's a torch.Tensor
            if hasattr(frame, "numpy"):
                frame = frame.numpy()

            # Each frame call returns a List[pd.DataFrame]
            dfs_for_faces = face_analyzer.recognize_faces(img_path=frame, db_path=db_path)
            results_for_frames.append(dfs_for_faces)

        return results_for_frames

    else:
        raise ValueError(
            "Unsupported input_media type. Must be a file path (str), "
            "an image array (np.ndarray), or a Video object."
        )


def verify_faces(
    img1: Union[str, np.ndarray],
    img2: Union[str, np.ndarray],
    deepface_args: Optional[Dict] = None,
) -> Dict:
    """Perform face verification between two images.

    Args:
        img1 (Union[str, np.ndarray]): Path or array for the first image.
        img2 (Union[str, np.ndarray]): Path or array for the second image.
        deepface_args (Optional[Dict]): Optional dictionary of arguments to pass to DeepFaceAnalysis.

    Returns:
        Dict: Verification result containing similarity and match.
    """
    face_analyzer = DeepFaceAnalysis.from_dict(deepface_args)
    return face_analyzer.verify_faces(img1_path=img1, img2_path=img2)


def extract_face_embeddings(
    input_media: Union[str, np.ndarray, Video], deepface_args: Optional[Dict] = None
) -> List[List[Dict]]:
    """Extract face embeddings from an image or video.

    Args:
        input_media (Union[str, np.ndarray, Video]):
            - str: Path to an image file.
            - np.ndarray: Image array.
            - Video: Video object; analyze all frames in the video.
        deepface_args (Optional[Dict]): Optional dictionary of arguments to pass to DeepFaceAnalysis.

    Returns:
        List[List[Dict]]: A nested list containing face embeddings for each image/frame.
    """
    face_analyzer = DeepFaceAnalysis.from_dict(deepface_args)

    # Single image
    if isinstance(input_media, (str, np.ndarray)):
        embeddings = face_analyzer.extract_face_embeddings(input_media)
        return [embeddings]

    # Video
    elif isinstance(input_media, Video):
        results_for_frames = []
        for frame_idx in range(input_media.frames.shape[0]):
            frame = input_media.frames[frame_idx]
            if hasattr(frame, "numpy"):
                frame = frame.numpy()
            embeddings = face_analyzer.extract_face_embeddings(frame)
            results_for_frames.append(embeddings)
        return results_for_frames

    else:
        raise ValueError(
            "Unsupported input_media type. Must be a file path (str), "
            "an image array (np.ndarray), or a Video object."
        )


def analyze_face_attributes(
    input_media: Union[str, np.ndarray, Video],
    actions: Optional[List[str]] = None,
    deepface_args: Optional[Dict] = None,
) -> List[List[Dict]]:
    """Analyze facial attributes (age, gender, emotion, race) for an image or video.

    Args:
        input_media (Union[str, np.ndarray, Video]):
            - str: Path to an image file.
            - np.ndarray: Image array.
            - Video: Video object; analyze all frames in the video.
        actions (Optional[List[str]]): List of attributes to analyze.
            Defaults to ['age', 'gender', 'emotion', 'race'].
        deepface_args (Optional[Dict]): Optional dictionary of arguments to pass to DeepFaceAnalysis.

    Returns:
        List[List[Dict]]: A list of lists. The top-level list corresponds to each image or frame,
        and each sub-list contains a dictionary of analysis results for each face.
    """
    face_analyzer = DeepFaceAnalysis.from_dict(deepface_args)

    # Single image
    if isinstance(input_media, (str, np.ndarray)):
        analysis = face_analyzer.analyze_face_attributes(img_path=input_media, actions=actions)
        return [analysis]

    # Video
    elif isinstance(input_media, Video):
        results_for_frames = []
        for frame_idx in range(input_media.frames.shape[0]):
            frame = input_media.frames[frame_idx]
            if hasattr(frame, "numpy"):
                frame = frame.numpy()
            analysis = face_analyzer.analyze_face_attributes(img_path=frame, actions=actions)
            results_for_frames.append(analysis)
        return results_for_frames

    else:
        raise ValueError(
            "Unsupported input_media type. Must be a file path (str), "
            "an image array (np.ndarray), or a Video object."
        )
