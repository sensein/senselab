"""Base class for text analysis pipelines."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.model import HFModel


class BaseAnalysis(ABC):
    """Abstract base class for text analysis pipelines."""

    _tokenizers: Dict[str, AutoTokenizer] = {}
    _models: Dict[str, AutoModelForSequenceClassification] = {}

    @classmethod
    def _get_tokenizer(
        cls,
        model: HFModel,
        task: str,
    ) -> AutoTokenizer:
        """Get or create a tokenizer.

        Args:
            model: The HFModel object.
            task: The task for which the tokenizer is needed.

        Returns:
            The AutoTokenizer object.
        """
        key = f"{model.path_or_uri}-{model.revision}-{task}"
        if key not in cls._tokenizers:
            cls._tokenizers[key] = AutoTokenizer.from_pretrained(model.path_or_uri, revision=model.revision)
        return cls._tokenizers[key]

    @classmethod
    def _load_model(
        cls,
        model: HFModel,
        task: str,
        device: DeviceType,
        torch_dtype: torch.dtype,
    ) -> AutoModelForSequenceClassification:
        """Load or get a model.

        Args:
            model: The HFModel object.
            task: The task for which the model is needed.
            device: The device to load the model on.
            torch_dtype: The torch data type to use.

        Returns:
            The AutoModelForSequenceClassification object.
        """
        key = f"{model.path_or_uri}-{model.revision}-{device.value}-{task}"
        if key not in cls._models:
            cls._models[key] = AutoModelForSequenceClassification.from_pretrained(
                model.path_or_uri, revision=model.revision, torch_dtype=torch_dtype
            ).to(device.value)
        return cls._models[key]

    @classmethod
    @abstractmethod
    def analyze(
        cls,
        pieces_of_text: List[str],
        device: Optional[DeviceType],
        model: Optional[HFModel],
        max_length: int = 512,
        overlap: int = 128,
        **kwargs: Union[str, int, float, bool],
    ) -> List[Dict[str, Union[str, float]]]:
        """Abstract method for text analysis.

        Args:
            pieces_of_text: List of text strings to analyze.
            device: The device to use for computation.
            model: The model to use for analysis.
            max_length: Maximum length of each chunk for long sequences.
            overlap: Overlap between chunks for long sequences.
            **kwargs: Additional keyword arguments.

        Returns:
            A list of dictionaries containing analysis results.
        """
        pass

    @staticmethod
    def validate_input(text: str) -> None:
        """Validate input text.

        Args:
            text: The input text to validate.

        Raises:
            TypeError: If the input is not a string.
            ValueError: If the input string is empty, contains only whitespace, or only punctuation.
        """
        if not isinstance(text, str):
            raise TypeError("Input must be a string.")
        if not text.strip():
            raise ValueError("Input string is empty or contains only whitespace.")
        if all(not c.isalnum() for c in text):
            raise ValueError("Input string contains only punctuation.")

    @staticmethod
    def _chunk_text(text: str, tokenizer: AutoTokenizer, max_length: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks.

        Args:
            text: The input text to chunk.
            tokenizer: The tokenizer to use.
            max_length: Maximum length of each chunk.
            overlap: Overlap between chunks.

        Returns:
            A list of text chunks.
        """
        tokens = tokenizer.tokenize(text)
        chunks = []
        for i in range(0, len(tokens), max_length - overlap):
            chunk = tokens[i : i + max_length - 2]  # -2 for [CLS] and [SEP]
            chunks.append(tokenizer.convert_tokens_to_string(chunk))
        return chunks

    @staticmethod
    def _prepare_input(
        chunks: List[str], tokenizer: AutoTokenizer, max_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare input for the model.

        Args:
            chunks: List of text chunks.
            tokenizer: The tokenizer to use.
            max_length: Maximum length of each chunk.

        Returns:
            A tuple containing input IDs and attention masks.
        """
        encodings = tokenizer(chunks, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
        return encodings["input_ids"], encodings["attention_mask"]

    @classmethod
    def _process_long_text(
        cls,
        text: str,
        tokenizer: AutoTokenizer,
        model: AutoModelForSequenceClassification,
        max_length: int,
        overlap: int,
    ) -> torch.Tensor:
        """Process long text by chunking and aggregating results.

        Args:
            text: The input text to process.
            tokenizer: The tokenizer to use.
            model: The model to use for processing.
            max_length: Maximum length of each chunk.
            overlap: Overlap between chunks.

        Returns:
            A tensor containing the processed results.
        """
        chunks = cls._chunk_text(text, tokenizer, max_length, overlap)
        input_ids, attention_masks = cls._prepare_input(chunks, tokenizer, max_length)

        with torch.no_grad():
            outputs = model(input_ids.to(model.device), attention_mask=attention_masks.to(model.device))
            logits = outputs.logits

        return torch.nn.functional.softmax(logits, dim=1).mean(dim=0)
