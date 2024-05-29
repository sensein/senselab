"""This module implements some utilities for the Sentence Transformers task."""

from typing import Any, Dict

from datasets import Dataset
from sentence_transformers import SentenceTransformer

from senselab.utils.hf import HFModel
from senselab.utils.tasks.input_output import (
    _from_dict_to_hf_dataset,
    _from_hf_dataset_to_dict,
)


def extract_embeddings_from_hf_dataset(
    dataset: Dict[str, Any],
    model_id: str,
    model_revision: str = "main",
    text_column: str = "text",
) -> Dict[str, Any]:
    """Extracts embeddings from a Hugging Face `Dataset` object."""

    def _extract_embeddings_from_hf_row(
        row: Dataset, text_column: str, model: SentenceTransformer
    ) -> Dataset:
        """Extracts embeddings from a Hugging Face `Dataset` row."""
        text = row[text_column]["text"]
        embeddings = model.encode(text)
        return {"embeddings": embeddings}

    _ = HFModel(hf_model_id=model_id)  # check HF model is valid

    hf_dataset = _from_dict_to_hf_dataset(dataset)
    model = SentenceTransformer(model_id, revision=model_revision)
    embeddings_hf_dataset = hf_dataset.map(
        lambda x: _extract_embeddings_from_hf_row(x, text_column, model)
    )
    embeddings_hf_dataset = embeddings_hf_dataset.remove_columns([text_column])
    return _from_hf_dataset_to_dict(embeddings_hf_dataset)
