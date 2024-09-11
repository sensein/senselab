"""Sentiment analysis implementation using the BaseAnalysis class."""

from typing import Dict, List, Optional, Union

import torch

from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.model import HFModel
from senselab.utils.interfaces.base_analysis import BaseAnalysis


class SentimentAnalysis(BaseAnalysis):
    """A class for performing sentiment analysis on text."""

    @classmethod
    def analyze(
        cls,
        pieces_of_text: List[str],
        device: Optional[DeviceType],
        model: Optional[HFModel] = None,
        max_length: int = 512,
        overlap: int = 128,
        **kwargs: Union[str, int, float, bool],
    ) -> List[Dict[str, Union[str, float]]]:
        """Perform sentiment analysis on a list of text pieces.

        Args:
            pieces_of_text: List of text strings to analyze.
            device: The device to use for computation.
            model: The model to use for sentiment analysis.
            max_length: Maximum length of each chunk for long sequences.
            overlap: Overlap between chunks for long sequences.
            neutral_threshold: Threshold for classifying sentiment as neutral.
            **kwargs: Additional keyword arguments.

        Returns:
            A list of dictionaries, each containing:
                - score (float): Sentiment score between -1 and 1.
                - label (str): Sentiment label ("negative", "neutral", or "positive").

        Raises:
            ValueError: If the input list is empty or None.
        """
        neutral_threshold = float(kwargs.get("neutral_threshold", "0.05"))

        if not pieces_of_text:
            raise ValueError("Input list is empty or None.")

        if model is None:
            model = HFModel(path_or_uri="distilbert-base-uncased-finetuned-sst-2-english", revision="main")

        tokenizer, model_instance = cls._get_pipeline(model, "sentiment-analysis", device)

        results: List[Dict[str, Union[str, float]]] = []
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

            score = float(probs[1].item() - probs[0].item())  # Range: [-1, 1]

            if abs(score) < neutral_threshold:
                label = "neutral"
            elif score > 0:
                label = "positive"
            else:
                label = "negative"

            results.append({"score": score, "label": label})

        return results
