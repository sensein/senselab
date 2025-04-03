"""This module provides the API for face analysis tasks."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np

from senselab.video.data_structures.video import Video
from senselab.video.tasks.face_analysis.deepface_utils import DeepFaceAnalysis
from senselab.video.tasks.face_analysis.utils import get_sampled_frames


@dataclass
class BoundingBox:
    """Bounding box coordinates and dimensions for a detected face."""

    x: int
    y: int
    w: int
    h: int
    left_eye: Optional[tuple] = None
    right_eye: Optional[tuple] = None


@dataclass
class FaceAttributes:
    """Attributes of a detected face, such as age, gender, emotion, and race."""

    age: Optional[int] = None
    dominant_emotion: Optional[str] = None
    emotion: Optional[Dict[str, float]] = None
    dominant_gender: Optional[str] = None
    gender: Optional[Dict[str, float]] = None
    dominant_race: Optional[str] = None
    race: Optional[Dict[str, float]] = None


@dataclass
class FaceMatch:
    """Information about a matched face."""

    identity: str
    distance: float
    verified: bool
    bbox: BoundingBox
    hash: Optional[str] = None


@dataclass
class DetectedFace:
    """Container for all information related to a detected face."""

    bbox: BoundingBox
    frame_ix: Optional[float] = None
    face_confidence: Optional[float] = None
    attributes: Optional[FaceAttributes] = None
    embedding: Optional[List[float]] = None
    face_match: Optional[List[FaceMatch]] = None


def recognize_faces(
    input_media: Union[str, np.ndarray, Video],
    db_path: str,
    model_name: str = "VGG-Face",
    distance_metric: str = "cosine",
    detector_backend: str = "opencv",
    align: bool = True,
    enforce_detection: bool = True,
    threshold: Optional[float] = None,
    frame_sample_rate: Optional[float] = None,
) -> List[List[DetectedFace]]:
    """Perform face recognition against a database of faces.

    Args:
        input_media (str | np.ndarray | Video): Media to recognize face(s) from.

        db_path (str): Path to the face database for recognition.

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet

        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2'.

        detector_backend (str): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'yolov11n', 'yolov11s',
            'yolov11m', 'centerface' or 'skip'.

        align (bool): Perform alignment based on the eye positions.

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Default is True. Set to False to avoid the exception for low-resolution images.

        threshold (float): Specify a threshold to determine whether a pair represents the same
            person or different individuals. This threshold is used for comparing distances.
            If left unset, default pre-tuned threshold values will be applied based on the specified
            model name and distance metric (default is None).

        frame_sample_rate (float): The desired number of frames per second to sample.
            If None or greater than the video's native frame rate, processes all frames. (default is None)

    Returns:
        List[List[DetectedFace]]: Nested list containing `DetectedFace` objects for each frame.
            Each `DetectedFace` includes matched identities (`FaceMatch`) found in the face database,
            sorted by similarity distance.
    """
    face_analyzer = DeepFaceAnalysis(model_name, distance_metric, detector_backend, align, enforce_detection, threshold)

    if isinstance(input_media, (str, np.ndarray)):
        recognized_frames = [(face_analyzer.recognize_faces(img_path=input_media, db_path=db_path), 0)]

    elif isinstance(input_media, Video):
        sampled_frames = get_sampled_frames(input_media, frame_sample_rate)
        recognized_frames = [
            (
                face_analyzer.recognize_faces(
                    img_path=frame.numpy() if hasattr(frame, "numpy") else frame, db_path=db_path
                ),
                frame_ix,
            )
            for frame_ix, frame in sampled_frames
        ]

    else:
        raise ValueError(
            "Unsupported input_media type. Must be a file path (str), an image array (np.ndarray), or a Video object."
        )

    return [
        [
            DetectedFace(
                bbox=BoundingBox(
                    x=int(face_df.iloc[0]["source_x"]),
                    y=int(face_df.iloc[0]["source_y"]),
                    w=int(face_df.iloc[0]["source_w"]),
                    h=int(face_df.iloc[0]["source_h"]),
                ),
                face_confidence=None,
                attributes=None,
                embedding=None,
                face_match=sorted(
                    [
                        FaceMatch(
                            identity=row["identity"],
                            distance=float(row["distance"]),
                            verified=bool(row["distance"] <= row["threshold"]),
                            bbox=BoundingBox(
                                x=int(row["target_x"]),
                                y=int(row["target_y"]),
                                w=int(row["target_w"]),
                                h=int(row["target_h"]),
                            ),
                            hash=row["hash"],
                        )
                        for _, row in face_df.iterrows()
                    ],
                    key=lambda fm: fm.distance,
                ),
                frame_ix=frame_ix if isinstance(input_media, Video) else None,
            )
            for face_df in frame
            if not face_df.empty
        ]
        for frame, frame_ix in recognized_frames
    ]


def verify_faces(
    img1: Union[str, np.ndarray],
    img2: Union[str, np.ndarray],
    model_name: str = "VGG-Face",
    distance_metric: str = "cosine",
    detector_backend: str = "opencv",
    align: bool = True,
    enforce_detection: bool = True,
    threshold: Optional[float] = None,
) -> DetectedFace:
    """Verify if two images contain the same person's face.

    Args:
        img1 (Union[str, np.ndarray]): Path to the first image.

        img2 (Union[str, np.ndarray]): Path to the second image.

        model_name (Optional[str]): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet

        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2'.

        detector_backend (Optional[str]): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'yolov11n', 'yolov11s',
            'yolov11m', 'centerface' or 'skip'.

        align (Optional[bool]): Perform alignment based on the eye positions.

        enforce_detection (Optional[boolean]): If no face is detected in an image, raise an exception.
            Default is True. Set to False to avoid the exception for low-resolution images.

        threshold (float): Specify a threshold to determine whether a pair represents the same
            person or different individuals. This threshold is used for comparing distances.
            If left unset, default pre-tuned threshold values will be applied based on the specified
            model name and distance metric (default is None).

    Returns:
        DetectedFace: DetectedFace object for `img1`, containing a `FaceMatch`
            representing the verification result against `img2`.
    """
    face_analyzer = DeepFaceAnalysis(model_name, distance_metric, detector_backend, align, enforce_detection, threshold)
    result = face_analyzer.verify_faces(img1_path=img1, img2_path=img2)

    return DetectedFace(
        bbox=BoundingBox(
            x=int(result["facial_areas"]["img1"]["x"]),
            y=int(result["facial_areas"]["img1"]["y"]),
            w=int(result["facial_areas"]["img1"]["w"]),
            h=int(result["facial_areas"]["img1"]["h"]),
            left_eye=result["facial_areas"]["img1"].get("left_eye"),
            right_eye=result["facial_areas"]["img1"].get("right_eye"),
        ),
        face_confidence=None,
        attributes=None,
        embedding=None,
        face_match=[
            FaceMatch(
                identity="img2",
                distance=float(result["distance"]),
                verified=bool(result["verified"]),
                bbox=BoundingBox(
                    x=int(result["facial_areas"]["img2"]["x"]),
                    y=int(result["facial_areas"]["img2"]["y"]),
                    w=int(result["facial_areas"]["img2"]["w"]),
                    h=int(result["facial_areas"]["img2"]["h"]),
                    left_eye=result["facial_areas"]["img2"].get("left_eye"),
                    right_eye=result["facial_areas"]["img2"].get("right_eye"),
                ),
                hash=None,
            )
        ],
    )


def extract_face_embeddings(
    input_media: Union[str, np.ndarray, Video],
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    align: bool = True,
    enforce_detection: bool = True,
    frame_sample_rate: Optional[float] = None,
) -> List[List[DetectedFace]]:
    """Extract face embeddings from an image or video.

    Args:
        input_media (str | np.ndarray | Video): Media to extract face embeddings from.

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'yolov11n', 'yolov11s',
            'yolov11m', 'centerface' or 'skip'.

        align (boolean): Perform alignment based on the eye positions.

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Default is True. Set to False to avoid the exception for low-resolution images.

        frame_sample_rate (float): The desired number of frames per second to sample.
            If None or greater than the video's native frame rate, processes all frames. (default is None)

    Returns:
        List[List[DetectedFace]]: Nested list containing `DetectedFace` objects for each frame.
            Each `DetectedFace` includes its corresponding embedding.
    """
    face_analyzer = DeepFaceAnalysis(
        model_name, detector_backend=detector_backend, align=align, enforce_detection=enforce_detection
    )

    if isinstance(input_media, (str, np.ndarray)):
        embeddings = [(face_analyzer.extract_face_embeddings(input_media), 0)]

    elif isinstance(input_media, Video):
        sampled_frames = get_sampled_frames(input_media, frame_sample_rate)
        embeddings = [
            (face_analyzer.extract_face_embeddings(frame.numpy() if hasattr(frame, "numpy") else frame), frame_ix)
            for frame_ix, frame in sampled_frames
        ]

    else:
        raise ValueError(
            "Unsupported input_media type. Must be a file path (str), an image array (np.ndarray), or a Video object."
        )

    return [
        [
            DetectedFace(
                bbox=BoundingBox(
                    x=int(face["facial_area"]["x"]),
                    y=int(face["facial_area"]["y"]),
                    w=int(face["facial_area"]["w"]),
                    h=int(face["facial_area"]["h"]),
                    left_eye=face["facial_area"].get("left_eye"),
                    right_eye=face["facial_area"].get("right_eye"),
                ),
                face_confidence=float(face["face_confidence"]),
                attributes=None,
                embedding=face["embedding"],
                face_match=None,
                frame_ix=frame_ix if isinstance(input_media, Video) else None,
            )
            for face in frame_data
        ]
        for frame_data, frame_ix in embeddings
    ]


def analyze_face_attributes(
    input_media: Union[str, np.ndarray, Video],
    actions: List[str] = ["age", "gender", "emotion", "race"],
    detector_backend: str = "opencv",
    align: bool = True,
    enforce_detection: bool = True,
    frame_sample_rate: Optional[float] = None,
) -> List[List[DetectedFace]]:
    """Analyze facial attributes (age, gender, emotion, race).

    Args:
        input_media (str | np.ndarray | Video): Media to analyze face attributes from.

        actions (Optional[List[str]]): Attributes to analyze (default: ['age', 'gender', 'emotion', 'race']).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'yolov11n', 'yolov11s', 'yolov11m',
            'centerface' or 'skip' (default is opencv).

        align (boolean): Perform alignment based on the eye positions (default is True).

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Set to False to avoid the exception for low-resolution images (default is True).

        frame_sample_rate (float): The desired number of frames per second to sample.
            If None or greater than the video's native frame rate, processes all frames. (default is None)

    Returns:
        List[List[DetectedFace]]: List of `DetectedFace` objects for each frame.
            Each `DetectedFace` includes the chosen attributes.
    """
    face_analyzer = DeepFaceAnalysis(
        detector_backend=detector_backend, align=align, enforce_detection=enforce_detection
    )

    if isinstance(input_media, (str, np.ndarray)):
        analyzed_frames = [(face_analyzer.analyze_face_attributes(input_media, actions), 0)]

    elif isinstance(input_media, Video):
        sampled_frames = get_sampled_frames(input_media, frame_sample_rate)
        analyzed_frames = [
            (face_analyzer.analyze_face_attributes(frame.numpy() if hasattr(frame, "numpy") else frame), frame_ix)
            for frame_ix, frame in sampled_frames
        ]

    else:
        raise ValueError(
            "Unsupported input_media type. Must be a file path (str), an image array (np.ndarray), or a Video object."
        )

    return [
        [
            DetectedFace(
                bbox=BoundingBox(
                    x=int(face["region"]["x"]),
                    y=int(face["region"]["y"]),
                    w=int(face["region"]["w"]),
                    h=int(face["region"]["h"]),
                    left_eye=face["region"].get("left_eye"),
                    right_eye=face["region"].get("right_eye"),
                ),
                face_confidence=float(face["face_confidence"]),
                attributes=FaceAttributes(
                    age=face.get("age"),
                    dominant_emotion=face.get("dominant_emotion"),
                    emotion={k: float(v) for k, v in face.get("emotion", {}).items()},
                    dominant_gender=face.get("dominant_gender"),
                    gender={k: float(v) for k, v in face.get("gender", {}).items()},
                    dominant_race=face.get("dominant_race"),
                    race={k: float(v) for k, v in face.get("race", {}).items()},
                ),
                embedding=None,
                face_match=None,
                frame_ix=frame_ix if isinstance(input_media, Video) else None,
            )
            for face in frame
        ]
        for frame, frame_ix in analyzed_frames
    ]


def visualize_face_analysis(face: DetectedFace) -> None:
    """Visualize outputs of face analysis methods."""
    raise NotImplementedError
