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
        model_name: Optional[str] = None,
        distance_metric: Optional[str] = None,
        backend: Optional[str] = None,
        align: Optional[bool] = None,
    ) -> None:
        """Initialize the DeepFaceAnalysis class with optional configurations.

        Args:
            model_name (Optional[str]): The name of the DeepFace model to use.
            distance_metric (Optional[str]): The distance metric to use for comparisons.
            backend (Optional[str]): The face detection backend to use.
            align (Optional[bool]): Whether to align faces before analysis.

        If None is passed, the default is used.
        """
        if not DEEPFACE_AVAILABLE:
            raise ImportError(
                "DeepFace is not available. "
                "Please install senselab video dependencies using `pip install senselab['video']`."
            )
        self.model_name = model_name
        self.distance_metric = distance_metric
        self.backend = backend
        self.align = align

    @classmethod
    def from_dict(cls, config: Optional[Dict] = None) -> "DeepFaceAnalysis":
        """Create a DeepFaceAnalysis instance from a dictionary of configuration parameters.

        Args:
            config (Optional[Dict]): A dictionary of optional configuration parameters. Valid keys are:
                - "model_name" (str): The DeepFace model to use.
                - "distance_metric" (str): The distance metric for face verification.
                - "backend" (str): The face detection backend (e.g., "opencv", "mtcnn").
                - "align" (bool): Whether to align faces before analysis.
            If no config or an empty config is provided, default parameters will be used.

        Returns:
            DeepFaceAnalysis: An instance of DeepFaceAnalysis with specified parameters applied.
        """
        if not DEEPFACE_AVAILABLE:
            raise ImportError(
                "DeepFace is not available. "
                "Please install senselab video dependencies using `pip install senselab['video']`."
            )

        if config is None:
            config = {}
        return cls(
            model_name=config.get("model_name"),
            distance_metric=config.get("distance_metric"),
            backend=config.get("backend"),
            align=config.get("align"),
        )

    def get_kwargs(self) -> Dict:
        """Get the keyword arguments for the DeepFace constructor.

        Returns:
            Dict: The keyword arguments for the DeepFace constructor, excluding any that are None.
        """
        # ensures that default DeepFace values are used when no value is passed
        return {
            key: value
            for key, value in {
                "model_name": self.model_name,
                "distance_metric": self.distance_metric,
                "detector_backend": self.backend,
                "align": self.align,
            }.items()
            if value is not None
        }

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
        return DeepFace.find(img_path=img_path, db_path=db_path, **self.get_kwargs())

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

        return DeepFace.verify(img1_path=img1_path, img2_path=img2_path, **self.get_kwargs())

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

        kwargs = {key: value for key, value in self.get_kwargs() if key in ["model_name", "detector_backend", "align"]}
        return DeepFace.represent(img_path=img_path, **kwargs)

    def analyze_face_attributes(
        self, img_path: Union[str, np.ndarray], actions: Optional[List[str]] = None
    ) -> List[Dict]:
        """Analyze facial attributes (age, gender, emotion, race) for faces in an image.

        Args:
            img_path (str or np.ndarray): The path to the image to analyze.
            actions (Optional[List[str]]): List of attributes to analyze
                (default: ['age', 'gender', 'emotion', 'race']).

        Returns:
            List[Dict]: List of analysis results for each detected face in the image.
        """
        if not DEEPFACE_AVAILABLE:
            raise ImportError(
                "DeepFace is not available. "
                "Please install senselab video dependencies using `pip install senselab['video']`."
            )

        if actions is None:
            actions = ["age", "gender", "emotion", "race"]
        kwargs = {key: value for key, value in self.get_kwargs() if key in ["detector_backend", "align"]}
        return DeepFace.analyze(img_path=img_path, actions=actions, **kwargs)
