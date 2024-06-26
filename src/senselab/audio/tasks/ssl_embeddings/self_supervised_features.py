"""This module contains functions for extracting features from pre-trained self-supervised models."""

from typing import Dict, List, Optional

from transformers import AutoFeatureExtractor, AutoModel

from senselab.audio.data_structures.audio import Audio
from senselab.utils.data_structures.device import DeviceType, _select_device_and_dtype
from senselab.utils.data_structures.model import HFModel


# TODO: make SSLEmbeddingsFactory work with other APIs, not just HuggingFace
class SSLEmbeddingsFactory:
    """A factory for managing self-supervised models."""

    _feat_extractor: Dict[str, AutoFeatureExtractor] = {}
    _model: Dict[str, AutoModel] = {}

    @classmethod
    def _get_feature_extractor(
        cls,
        model: HFModel,
        cache_dir: str = "~/",
        device: Optional[DeviceType] = None,
    ) -> AutoFeatureExtractor:
        """Get or create a feature extractor for SSL model.

        Args:
            model (HFModel): The HuggingFace model.
            cache_dir (str): The path to where the model's weights will be saved.
            device (Optional[DeviceType]): The device to run the model on.

        Returns:
            AutoFeatureExtractor: The feature extractor for the model.
        """
        device, torch_dtype = _select_device_and_dtype(
            user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )
        key = f"{model.path_or_uri}-{model.revision}-{device.value}"
        if key not in cls._feat_extractor:
            cls._feat_extractor[key] = AutoFeatureExtractor.from_pretrained(model.path_or_uri, cache_dir=cache_dir)
        return cls._feat_extractor[key]

    @classmethod
    def _load_model(
        cls,
        model: HFModel,
        cache_dir: str = "~/",
        device: Optional[DeviceType] = None,
    ) -> AutoModel:
        """Load weights of SSL model.

        Args:
            model (HFModel): The Hugging Face model.
            cache_dir (str): The path to where the model's weights will be saved.
            device (Optional[DeviceType]): The device to run the model on.

        Returns:
            AutoModel: The SSL model.
        """
        device, torch_dtype = _select_device_and_dtype(
            user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )
        key = f"{model.path_or_uri}-{model.revision}-{device.value}"
        if key not in cls._model:
            cls._model[key] = AutoModel.from_pretrained(model.path_or_uri, cache_dir=cache_dir).to(device.value)
        return cls._model[key].to(device.value)

    @classmethod
    def extract_ssl_embeddings(
        cls,
        audios: List[Audio],
        model: HFModel,
        cache_dir: str = "~/",
        device: Optional[DeviceType] = None,
    ) -> List[Dict]:
        """Compute the ssl embeddings of audio signals.

        Args:
            audios (List[Audio]): A list of Audio objects containing the audio signals and their properties.
            model (HFModel): The model used to compute the embeddings
                (default is "facebook/wav2vec2-base").
            cache_dir (str): The path to where the model's weights will be saved.
            device (Optional[DeviceType]): The device to run the model on (default is None).
                Only CPU and CUDA are supported.

        Returns:
            List[torch.Tensor]: A list of tensors containing the ssl embeddings for each audio file.
        """
        device, torch_dtype = _select_device_and_dtype(
            user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )
        feat_extractor = cls._get_feature_extractor(model=model, cache_dir=cache_dir, device=device)
        ssl_model = cls._load_model(model=model, cache_dir=cache_dir, device=device)
        expected_sample_rate = 16000

        # Check that all audio objects have the correct sampling rate
        for audio in audios:
            if audio.waveform.shape[0] != 1:
                raise ValueError(f"Audio waveform must be mono (1 channel), but got {audio.waveform.shape[0]} channels")
            if audio.sampling_rate != expected_sample_rate:
                raise ValueError(
                    "Audio sampling rate "
                    + str(audio.sampling_rate)
                    + " does not match expected "
                    + str(expected_sample_rate)
                )
        # Pre-process audio using the SSL mode feature extractor
        preprocessed_audios = [
            feat_extractor(audio.waveform, sampling_rate=expected_sample_rate, return_tensors="pt") for audio in audios
        ]

        # Extract embeddings (hidden states from all layers) from pre-trained model
        embeddings = [
            ssl_model(audio.input_values.squeeze(0).to(device.value), output_hidden_states=True)
            for audio in preprocessed_audios
        ]
        embeddings = [
            (f"Layer{i}", hidden_state.squeeze(0))
            for embedding in embeddings
            for i, hidden_state in enumerate(list(embedding.hidden_states))
        ]

        # Convert list of tuples into dict with layer name as key and layer hidden state as value
        embeddings = [dict(embeddings) for embedding in embeddings]

        return embeddings
