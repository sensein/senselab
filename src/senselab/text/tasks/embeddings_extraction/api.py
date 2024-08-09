"""This module provides a wrapper method for extracting embeddings from text using different models."""

from typing import List, Optional

import torch

from senselab.text.tasks.embeddings_extraction.huggingface import HFFactory
from senselab.text.tasks.embeddings_extraction.sentence_transformers import SentenceTransformerFactory
from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.model import HFModel, SenselabModel, SentenceTransformersModel


def extract_embeddings_from_text(
    pieces_of_text: List[str], model: SenselabModel, device: Optional[DeviceType] = None
) -> List[torch.Tensor]:
    """Extracts embeddings from a list of strings using the specified model.

    Args:
        pieces_of_text (List[str]): A list of strings to extract embeddings from.
        model (SenselabModel): The model used for extracting embeddings.
        device (Optional[DeviceType]): The device to run the model on. Defaults to None.

    Returns:
        List[torch.Tensor]: A list of embeddings for the input strings.
    """
    if isinstance(model, SentenceTransformersModel):
        return SentenceTransformerFactory.extract_text_embeddings(pieces_of_text, model, device)
    elif isinstance(model, HFModel):
        return HFFactory.extract_text_embeddings(pieces_of_text, model, device)
    raise ValueError("Unsupported model type for text embedding extraction.")
