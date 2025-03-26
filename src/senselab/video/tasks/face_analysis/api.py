"""This module provides the API for face analysis tasks."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np

from senselab.video.data_structures.video import Video
from senselab.video.tasks.face_analysis.deepface_utils import DeepFaceAnalysis


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
    """Information about a recognized face matched from the database."""

    identity: str
    distance: float
    verified: bool
    bbox: BoundingBox
    hash: Optional[str] = None


@dataclass
class DetectedFace:
    """Container for all information related to a detected face."""

    bbox: BoundingBox
    face_confidence: Optional[float] = None
    attributes: Optional[FaceAttributes] = None
    embedding: Optional[List[float]] = None
    face_match: Optional[List[FaceMatch]] = None


def recognize_faces(
    input_media: Union[str, np.ndarray, Video],
    db_path: str,
    model_name: Optional[str] = None,
    distance_metric: Optional[str] = None,
    backend: Optional[str] = None,
    align: Optional[bool] = None,
) -> List[List[DetectedFace]]:
    """Perform face recognition against a database of faces.

    Args:
        input_media (str | np.ndarray | Video): Media to recognize face(s) from.
        db_path (str): Path to the face database for recognition.
        model_name (Optional[str]): Face recognition model name (e.g., 'Facenet').
        distance_metric (Optional[str]): Distance metric ('cosine', 'euclidean', etc.).
        backend (Optional[str]): Face detection backend ('opencv', 'mtcnn', etc.).
        align (Optional[bool]): Align faces before analysis.

    Returns:
        List[List[DetectedFace]]: Nested list containing `DetectedFace` objects for each frame.
            Each `DetectedFace` includes matched identities (`FaceMatch`) found in the face database,
            sorted by similarity distance.
    """
    face_analyzer = DeepFaceAnalysis(model_name, distance_metric, backend, align)

    if isinstance(input_media, (str, np.ndarray)):
        recognized_frames = [face_analyzer.recognize_faces(img_path=input_media, db_path=db_path)]

    elif isinstance(input_media, Video):
        recognized_frames = [
            face_analyzer.recognize_faces(img_path=frame.numpy() if hasattr(frame, "numpy") else frame, db_path=db_path)
            for frame in input_media.frames
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
            )
            for face_df in frame
            if not face_df.empty
        ]
        for frame in recognized_frames
    ]


def verify_faces(
    img1: Union[str, np.ndarray],
    img2: Union[str, np.ndarray],
    model_name: Optional[str] = None,
    distance_metric: Optional[str] = None,
    backend: Optional[str] = None,
    align: Optional[bool] = None,
) -> DetectedFace:
    """Verify if two images contain the same person's face.

    Args:
        img1 (Union[str, np.ndarray]): Path or array for the first image.
        img2 (Union[str, np.ndarray]): Path or array for the second image.
        model_name (Optional[str]): Face recognition model name.
        distance_metric (Optional[str]): Distance metric for verification.
        backend (Optional[str]): Face detection backend.
        align (Optional[bool]): Align faces before verification.

    Returns:
        DetectedFace: DetectedFace object for `img1`, containing a `FaceMatch`
            representing the verification result against `img2`.
    """
    face_analyzer = DeepFaceAnalysis(model_name, distance_metric, backend, align)
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
    model_name: Optional[str] = None,
    backend: Optional[str] = None,
    align: Optional[bool] = None,
) -> List[List[DetectedFace]]:
    """Extract face embeddings from an image or video.

    Args:
        input_media (str | np.ndarray | Video): Media to extract face embeddings from.
        model_name (Optional[str]): Embedding extraction model name.
        backend (Optional[str]): Face detection backend.
        align (Optional[bool]): Align faces before extraction.

    Returns:
        List[List[DetectedFace]]: Nested list containing `DetectedFace` objects for each frame.
            Each `DetectedFace` includes its corresponding embedding.
    """
    face_analyzer = DeepFaceAnalysis(model_name, backend=backend, align=align)

    if isinstance(input_media, (str, np.ndarray)):
        embeddings = [face_analyzer.extract_face_embeddings(input_media)]

    elif isinstance(input_media, Video):
        embeddings = [
            face_analyzer.extract_face_embeddings(frame.numpy() if hasattr(frame, "numpy") else frame)
            for frame in input_media.frames
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
            )
            for face in frame
        ]
        for frame in embeddings
    ]


def analyze_face_attributes(
    input_media: Union[str, np.ndarray, Video],
    actions: Optional[List[str]] = None,
    backend: Optional[str] = None,
    align: Optional[bool] = None,
) -> List[List[DetectedFace]]:
    """Analyze facial attributes (age, gender, emotion, race).

    Args:
        input_media (str | np.ndarray | Video): Media to analyze face attributes from.
        actions (Optional[List[str]]): Attributes to analyze (default: ['age', 'gender', 'emotion', 'race']).
        backend (Optional[str]): Face detection backend.
        align (Optional[bool]): Align faces before analysis.

    Returns:
        List[List[DetectedFace]]: List of `DetectedFace` objects for each frame.
            Each `DetectedFace` includes the chosen attributes.
    """
    face_analyzer = DeepFaceAnalysis(backend=backend, align=align)

    if isinstance(input_media, (str, np.ndarray)):
        analyzed_frames = [face_analyzer.analyze_face_attributes(input_media, actions)]

    elif isinstance(input_media, Video):
        analyzed_frames = [
            face_analyzer.analyze_face_attributes(frame.numpy() if hasattr(frame, "numpy") else frame, actions)
            for frame in input_media.frames
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
            )
            for face in frame
        ]
        for frame in analyzed_frames
    ]
