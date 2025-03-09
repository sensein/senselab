"""This module implements some utilities for using Sentence Transformers for embeddings extraction."""

from typing import Dict, List, Optional

import torch

from senselab.utils.data_structures import DeviceType, SentenceTransformersModel, _select_device_and_dtype

try:
    from sentence_transformers import SentenceTransformer

    SENTENCETRANSFORMERS_AVAILABLE = True
except ModuleNotFoundError:
    SENTENCETRANSFORMERS_AVAILABLE = False


class SentenceTransformerFactory:
    """A factory for managing SentenceTransformer pipelines for text embeddings."""

    _pipelines: Dict[str, "SentenceTransformer"] = {}

    @classmethod
    def _get_sentencetransformer_pipeline(
        cls,
        model: SentenceTransformersModel,
        device: Optional[DeviceType] = None,
    ) -> "SentenceTransformer":
        """Get or create a SentenceTransformer pipeline.

        Args:
            model (SentenceTransformersModel): The Hugging Face model configuration.
            device (Optional[DeviceType]): The device to run the model on.

        Returns:
            SentenceTransformer: The SentenceTransformer pipeline.
        """
        if not SENTENCETRANSFORMERS_AVAILABLE:
            raise ModuleNotFoundError(
                "`sentence-transformers` is not installed. "
                "Please install senselab text dependencies using `pip install senselab['text']`."
            )

        device, _ = _select_device_and_dtype(
            user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )
        key = f"{model.path_or_uri}-{model.revision}-{device.value}"
        if key not in cls._pipelines:
            cls._pipelines[key] = SentenceTransformer(
                model_name_or_path=str(model.path_or_uri),
                revision=model.revision,
                device=device.value,
            )
        return cls._pipelines[key]

    @classmethod
    def extract_text_embeddings(
        cls,
        pieces_of_text: List[str],
        model: Optional[SentenceTransformersModel] = None,
        device: Optional[DeviceType] = None,
    ) -> List[torch.Tensor]:
        """Extracts embeddings from a list of strings using a SentenceTransformer model.

        Args:
            pieces_of_text (List[str]): A list of strings to extract embeddings from.
            model (SentenceTransformersModel, optional): A Hugging Face model configuration.
                If None, the default model "sentence-transformers/all-MiniLM-L6-v2" is used.
            device (Optional[DeviceType], optional): The device to run the model on.
                Defaults to None.

        Returns:
            List[torch.Tensor]: A list of embeddings for the input strings.
        """
        if not SENTENCETRANSFORMERS_AVAILABLE:
            raise ModuleNotFoundError(
                "`sentence-transformers` is not installed. "
                "Please install senselab text dependencies using `pip install senselab['text']`."
            )

        if model is None:
            model = SentenceTransformersModel(path_or_uri="sentence-transformers/all-MiniLM-L6-v2", revision="main")
        pipeline = cls._get_sentencetransformer_pipeline(model, device)
        embeddings = pipeline.encode(pieces_of_text, convert_to_tensor=True)
        return [embedding for embedding in embeddings]
