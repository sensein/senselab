"""This module is for extracting deep learning embeddings from text."""

from typing import List

import pytest
import torch

from senselab.text.tasks.embeddings_extraction import extract_embeddings_from_text
from senselab.utils.data_structures import HFModel, SentenceTransformersModel

try:
    from sentence_transformers import SentenceTransformer

    SENTENCETRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCETRANSFORMERS_AVAILABLE = False


@pytest.fixture
def hf_model() -> HFModel:
    """Fixture for our default embeddings extraction Hugging Face model."""
    return HFModel(path_or_uri="sentence-transformers/all-MiniLM-L6-v2", revision="main")


@pytest.fixture
def sentencetransformers_model() -> SentenceTransformersModel:
    """Fixture for our default embeddings extraction SentenceTransformer model."""
    return SentenceTransformersModel(path_or_uri="sentence-transformers/all-MiniLM-L6-v2", revision="main")


@pytest.mark.skipif(SENTENCETRANSFORMERS_AVAILABLE, reason="SentenceTransformers is installed")
def test_extract_sentencetransformers_embeddings_from_text_import_error() -> None:
    """Test extract_embeddings_from_text import error."""
    with pytest.raises(ImportError):
        extract_embeddings_from_text(
            ["test"], SentenceTransformersModel(path_or_uri="sentence-transformers/all-MiniLM-L6-v2", revision="main")
        )


@pytest.mark.skipif(not SENTENCETRANSFORMERS_AVAILABLE, reason="SentenceTransformers is not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
def test_extract_sentencetransformers_embeddings_from_text(
    sample_texts: List[str], sentencetransformers_model: SentenceTransformersModel
) -> None:
    """Test extract_embeddings_from_text."""
    embeddings = extract_embeddings_from_text(sample_texts, sentencetransformers_model)
    assert isinstance(embeddings, List)
    assert embeddings[0].shape == torch.Size([384])  # shape of "sentence-transformers/all-MiniLM-L6-v2"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
def test_extract_huggingface_embeddings_from_text(sample_texts: List[str], hf_model: HFModel) -> None:
    """Test extract_embeddings_from_text."""
    embeddings = extract_embeddings_from_text(sample_texts, hf_model)
    assert isinstance(embeddings, List)
    print(embeddings[0].shape)
    # 7 layers for "sentence-transformers/all-MiniLM-L6-v2" (6 is the sequence Length in this case)
    assert embeddings[0].shape[0] == 7
    # 384 as Hidden Size for shape of "sentence-transformers/all-MiniLM-L6-v2"
    assert embeddings[0].shape[2] == 384
