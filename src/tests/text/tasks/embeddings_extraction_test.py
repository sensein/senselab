"""This module is for extracting deep learning embeddings from text."""

import os

if os.getenv("GITHUB_ACTIONS") != "true":

    from typing import List

    import pytest
    import torch

    from senselab.text.tasks.embeddings_extraction.api import extract_embeddings_from_text
    from senselab.utils.data_structures.model import HFModel

    @pytest.fixture
    def hf_model() -> HFModel:
        """Fixture for our default embeddings extraction SentenceTransformer model."""
        return HFModel(path_or_uri="sentence-transformers/all-MiniLM-L6-v2", revision="main")

    def test_extract_embeddings_from_text(sample_texts: List[str], hf_model: HFModel) -> None:
        """Test extract_embeddings_from_text."""
        embeddings = extract_embeddings_from_text(sample_texts, hf_model)
        assert isinstance(embeddings, List)
        assert embeddings[0].shape == torch.Size([384])  # shape of "sentence-transformers/all-MiniLM-L6-v2"
