"""This module provides functionality for performing emotional analysis on text using a specified model."""

from typing import Any, Dict, List, Optional, Union

import torch

from senselab.text.tasks.emotional_analysis.constants import Emotion
from senselab.utils.data_structures.device import DeviceType, _select_device_and_dtype
from senselab.utils.data_structures.model import HFModel
from senselab.utils.interfaces.base_analysis import BaseAnalysis


class EmotionalAnalysis(BaseAnalysis):
    """A class for performing emotional analysis on text."""

    @classmethod
    def analyze(
        cls,
        pieces_of_text: List[str],
        device: Optional[DeviceType],
        model: Optional[HFModel] = None,
        max_length: int = 512,
        overlap: int = 128,
        **kwargs: Union[str, int, float, bool],
    ) -> List[Dict[str, Any]]:
        """Perform emotional analysis on a list of text pieces.

        Args:
            pieces_of_text: List of text strings to analyze.
            device: The device to use for computation.
            model: The model to use for emotional analysis.
            max_length: Maximum length of each chunk for long sequences.
            overlap: Overlap between chunks for long sequences.
            **kwargs: Additional keyword arguments.

        Returns:
            A list of dictionaries, each containing:
                - scores (Dict[str, float]): Probabilities for each emotion.
                - dominant_emotion (str): The most likely emotion in the text.

        Raises:
            ValueError: If the input list is empty or None.
        """
        if not pieces_of_text:
            raise ValueError("Input list is empty or None.")

        if model is None:
            model = HFModel(path_or_uri="j-hartmann/emotion-english-distilroberta-base", revision="main")

        device, torch_dtype = _select_device_and_dtype(
            user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )

        tokenizer = cls._get_tokenizer(model, "emotion-analysis")
        model_instance = cls._load_model(model, "emotion-analysis", device, torch_dtype)

        # Get emotion labels from the model's config
        emotion_labels = model_instance.config.id2label

        results: List[Dict[str, Any]] = []
        for text in pieces_of_text:
            cls.validate_input(text)

            if len(tokenizer.encode(text)) > max_length:
                probs = cls._process_long_text(text, tokenizer, model_instance, max_length, overlap)
            else:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
                inputs = {k: v.to(model_instance.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model_instance(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]

            scores = {Emotion.from_string(emotion_labels[i]): float(prob) for i, prob in enumerate(probs)}
            dominant_emotion = max(scores, key=lambda emotion: scores[emotion])

            results.append({"scores": scores, "dominant_emotion": dominant_emotion})

        return results
