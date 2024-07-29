"""This module implements some utilities for using Sentence Transformers for embeddings extraction."""

from typing import Dict, List, Optional

import torch
from sentence_transformers import SentenceTransformer

from senselab.utils.data_structures.device import DeviceType, _select_device_and_dtype
from senselab.utils.data_structures.model import SentenceTransformersModel


class SentenceTransformerFactory:
    """A factory for managing SentenceTransformer pipelines for text embeddings."""

    _pipelines: Dict[str, SentenceTransformer] = {}

    @classmethod
    def _get_sentencetransformer_pipeline(
        cls,
        model: SentenceTransformersModel,
        device: Optional[DeviceType] = None,
    ) -> SentenceTransformer:
        """Get or create a SentenceTransformer pipeline.

        Args:
            model (SentenceTransformersModel): The Hugging Face model configuration.
            device (Optional[DeviceType]): The device to run the model on.

        Returns:
            SentenceTransformer: The SentenceTransformer pipeline.
        """
        device, _ = _select_device_and_dtype(
            user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )
        key = f"{model.path_or_uri}-{model.revision}-{device.value}"
        if key not in cls._pipelines:
            cls._pipelines[key] = SentenceTransformer(
                model_name_or_path=model.path_or_uri,
                revision=model.revision,
                device=device.value,
            )
        return cls._pipelines[key]

    @classmethod
    def extract_text_embeddings(
        cls,
        pieces_of_text: List[str],
        model: SentenceTransformersModel = SentenceTransformersModel(
            path_or_uri="sentence-transformers/all-MiniLM-L6-v2", revision="main"
        ),
        device: Optional[DeviceType] = None,
    ) -> List[torch.Tensor]:
        """Extracts embeddings from a list of strings using a SentenceTransformer model.

        Args:
            pieces_of_text (List[str]): A list of strings to extract embeddings from.
            model (SentenceTransformersModel, optional): A Hugging Face model configuration.
                Defaults to SentenceTransformersModel(path_or_uri="sentence-transformers/all-MiniLM-L6-v2").
            device (Optional[DeviceType], optional): The device to run the model on.
                Defaults to None.

        Returns:
            List[torch.Tensor]: A list of embeddings for the input strings.
        """
        pipeline = cls._get_sentencetransformer_pipeline(model, device)
        embeddings = pipeline.encode(pieces_of_text, convert_to_tensor=True)
        return [embedding for embedding in embeddings]
