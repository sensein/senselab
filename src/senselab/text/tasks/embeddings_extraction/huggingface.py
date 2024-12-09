"""This module contains functions for extracting features from pre-trained self-supervised models."""

from typing import Dict, List, Optional

import torch
from transformers import AutoModel, AutoTokenizer

from senselab.utils.data_structures import DeviceType, HFModel, _select_device_and_dtype


class HFFactory:
    """A factory for managing self-supervised models from Hugging Face."""

    _tokenizers: Dict[str, AutoTokenizer] = {}
    _models: Dict[str, AutoModel] = {}

    @classmethod
    def _get_tokenizer(
        cls,
        model: HFModel,
    ) -> AutoTokenizer:
        """Get or create a tokenizer for SSL model.

        Args:
            model (HFModel): The HuggingFace model.

        Returns:
            AutoTokenizer: The tokenizer for the model.
        """
        key = f"{model.path_or_uri}-{model.revision}"
        if key not in cls._tokenizers:
            cls._tokenizers[key] = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=model.path_or_uri,
                revision=model.revision,
            )
        return cls._tokenizers[key]

    @classmethod
    def _load_model(
        cls,
        model: HFModel,
        device: DeviceType,
    ) -> AutoModel:
        """Load weights of SSL model.

        Args:
            model (HFModel): The Hugging Face model.
            device (DeviceType): The device to run the model on.

        Returns:
            AutoModel: The SSL model.
        """
        key = f"{model.path_or_uri}-{model.revision}-{device.value}"
        if key not in cls._models:
            cls._models[key] = AutoModel.from_pretrained(model.path_or_uri, revision=model.revision).to(device.value)
        return cls._models[key]

    @classmethod
    def extract_text_embeddings(
        cls,
        pieces_of_text: List[str],
        model: Optional[HFModel] = None,
        device: Optional[DeviceType] = None,
    ) -> List[torch.Tensor]:
        """Extracts embeddings from a list of strings using a Hugging Face model.

        Args:
            pieces_of_text (List[str]): A list of strings to extract embeddings from.
            model (HFModel, optional): A Hugging Face model configuration.
                If None, the default model "sentence-transformers/all-MiniLM-L6-v2" is used.
            device (Optional[DeviceType], optional): The device to run the model on.
                Defaults to None.

        Returns:
            List[torch.Tensor]: A list of embeddings for the input strings.
        """
        if model is None:
            model = HFModel(path_or_uri="sentence-transformers/all-MiniLM-L6-v2")
        device, _ = _select_device_and_dtype(
            user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )

        print(f"Using device: {device}")

        # Load tokenizer and model
        tokenizer = cls._get_tokenizer(model=model)
        ssl_model = cls._load_model(model=model, device=device)

        embeddings = []

        # Process each piece of text individually
        for text in pieces_of_text:
            # Tokenize sentence
            encoded_input = tokenizer(text, return_tensors="pt").to(device.value)

            # Compute token embeddings
            with torch.no_grad():
                model_output = ssl_model(**encoded_input, output_hidden_states=True)
                hidden_states = model_output.hidden_states
                concatenated_hidden_states = torch.cat(
                    [state.to(device.value).unsqueeze(0) for state in hidden_states], dim=0
                )
                embeddings.append(concatenated_hidden_states.squeeze())

        return embeddings
