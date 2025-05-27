"""Provides a utility class for performing face analysis using DeepFace."""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:
    from deepface import DeepFace

    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False


class DeepFaceAnalysis:
    """A utility class for performing face analysis tasks using DeepFace.

    Includes:
    - Face recognition
    - Face verification
    - Face embeddings extraction
    - Facial attribute analysis
    """

    def __init__(
        self,
        model_name: str = "VGG-Face",
        distance_metric: str = "cosine",
        detector_backend: str = "opencv",
        align: bool = True,
        enforce_detection: bool = False,
        threshold: Optional[float] = None,
    ) -> None:
        """Initialize the DeepFaceAnalysis class with configurations.

        Args:
            model_name (Optional[str]): The name of the DeepFace model to use.

             model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet

            distance_metric (string): Metric for measuring similarity. Options: 'cosine',
                'euclidean', 'euclidean_l2'.

            detector_backend (str): face detector backend. Options: 'opencv', 'retinaface',
                'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'yolov11n', 'yolov11s',
                'yolov11m', 'centerface' or 'skip'.

            align (bool): Perform alignment based on the eye positions.

            enforce_detection (boolean): If no face is detected in an image, raise an exception.
                Default is False.

            threshold (float): Specify a threshold to determine whether a pair represents the same
                person or different individuals. This threshold is used for comparing distances.
                If left unset, default pre-tuned threshold values will be applied based on the specified
                model name and distance metric (default is None).
        """
        if not DEEPFACE_AVAILABLE:
            raise ImportError(
                "DeepFace is not available. "
                "Please install senselab video dependencies using `pip install senselab['video']`."
            )
        self.model_name = model_name
        self.distance_metric = distance_metric
        self.detector_backend = detector_backend
        self.align = align
        self.enforce_detection = enforce_detection
        self.threshold = threshold

    def recognize_faces(self, img_path: Union[str, np.ndarray], db_path: str) -> List[pd.DataFrame]:
        """Perform face recognition on an image against a database of faces.

        Args:
            img_path (str or np.ndarray): The path to the image to analyze.
            db_path (str): Path to the face database for recognition.

        Returns:
            List[pd.DataFrame]: A list of pandas DataFrames for each face in the input image.
                Each DataFrame maps detected faces to their closest matches in the database.
        """
        if not DEEPFACE_AVAILABLE:
            raise ImportError(
                "DeepFace is not available. "
                "Please install senselab video dependencies using `pip install senselab['video']`."
            )
        return DeepFace.find(
            img_path=img_path,
            db_path=db_path,
            model_name=self.model_name,
            distance_metric=self.distance_metric,
            detector_backend=self.detector_backend,
            align=self.align,
            enforce_detection=self.enforce_detection,
            threshold=self.threshold,
        )

    def verify_faces(self, img1_path: Union[str, np.ndarray], img2_path: Union[str, np.ndarray]) -> Dict:
        """Perform face verification between two images.

        Args:
            img1_path (str or np.ndarray): The path to the first image.
            img2_path (str or np.ndarray): The path to the second image.

        Returns:
            Dict: Verification result
        """
        if not DEEPFACE_AVAILABLE:
            raise ImportError(
                "DeepFace is not available. "
                "Please install senselab video dependencies using `pip install senselab['video']`."
            )

        return DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name=self.model_name,
            distance_metric=self.distance_metric,
            detector_backend=self.detector_backend,
            align=self.align,
            enforce_detection=self.enforce_detection,
            threshold=self.threshold,
        )

    def extract_face_embeddings(self, img_path: Union[str, np.ndarray]) -> List[Dict]:
        """Extract face embeddings from an image.

        Args:
            img_path (str or np.ndarray): The path to the image to analyze.

        Returns:
            List[Dict]: A list of embeddings for the faces in the image.
        """
        if not DEEPFACE_AVAILABLE:
            raise ImportError(
                "DeepFace is not available. "
                "Please install senselab video dependencies using `pip install senselab['video']`."
            )

        return DeepFace.represent(
            img_path=img_path,
            model_name=self.model_name,
            detector_backend=self.detector_backend,
            align=self.align,
            enforce_detection=self.enforce_detection,
        )

    def analyze_face_attributes(
        self, img_path: Union[str, np.ndarray], actions: List[str] = ["age", "gender", "emotion", "race"]
    ) -> List[Dict]:
        """Analyze facial attributes (age, gender, emotion, race) for faces in an image.

        Args:
            img_path (str or np.ndarray): The path to the image to analyze.
            actions (List[str]): List of attributes to analyze
                (default: ['age', 'gender', 'emotion', 'race']).

        Returns:
            List[Dict]: List of analysis results for each detected face in the image.
        """
        if not DEEPFACE_AVAILABLE:
            raise ImportError(
                "DeepFace is not available. "
                "Please install senselab video dependencies using `pip install senselab['video']`."
            )

        return DeepFace.analyze(
            img_path=img_path,
            actions=actions,
            detector_backend=self.detector_backend,
            align=self.align,
            enforce_detection=self.enforce_detection,
        )
