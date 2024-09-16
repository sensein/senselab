"""Sentiment analysis implementation using the BaseAnalysis class."""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

from senselab.text.tasks.sentiment_analysis.constants import Sentiment
from senselab.utils.data_structures.device import DeviceType, _select_device_and_dtype
from senselab.utils.interfaces.analyses import BaseTextAnalysis
from senselab.utils.model_utils import BaseModelSourceUtils
from senselab.utils.tasks.chunking import chunk_text


class SentimentAnalysis(BaseTextAnalysis):
    """A class for performing sentiment analysis on text."""

    @classmethod
    def analyze(
        cls,
        input_data: List[Any],
        model_utils: BaseModelSourceUtils,
        device: Optional[DeviceType],
        **kwargs: Any,  # noqa: ANN401
    ) -> List[Dict[str, Any]]:
        """Perform sentiment analysis on a list of text pieces.

        Args:
            input_data (List[Any]): List of text strings to analyze.
            model_utils (BaseModelSourceUtils): Utility class for model operations.
            device (Optional[DeviceType]): The device to use for computation (e.g., CPU, CUDA).
            **kwargs (Any): Additional keyword arguments, such as:
                - max_length (int): Maximum length of text chunks (default: 512).
                - overlap (int): Overlap size between text chunks (default: 128).
                - neutral_threshold (float): Threshold for considering sentiment as neutral (default: 0.05).

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing:
                - score (float): Sentiment score between -1 and 1.
                - label (str): The sentiment label ("negative", "neutral", "positive").

        Raises:
            ValueError: If the input list is empty or None.
        """
        if not input_data:
            raise ValueError("Input list is empty or None.")

        max_length = int(kwargs.get("max_length", 512))
        overlap = int(kwargs.get("overlap", 128))
        neutral_threshold = float(kwargs.get("neutral_threshold", 0.05))

        device, torch_dtype = _select_device_and_dtype(
            user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )

        tokenizer = model_utils.get_tokenizer(task="sentiment-analysis")
        pipe = model_utils.get_pipeline(
            task="sentiment-analysis", device=device, torch_dtype=torch_dtype, return_all_scores=True
        )

        results: List[Dict[str, Union[float, str]]] = []

        for text in input_data:
            cls.validate_input(text)

            chunks = chunk_text(text=text, tokenizer=tokenizer, max_length=max_length, overlap=overlap)
            chunks_output = pipe(chunks)

            score_sums: Dict[str, float] = defaultdict(float)

            for chunk_output in chunks_output:
                for result in chunk_output:
                    label = result["label"]
                    score = result["score"]
                    score_sums[label] += score

            total_score = sum(score_sums.values())
            normalized_scores = {label: score / total_score for label, score in score_sums.items()}

            sentiment_score = normalized_scores.get(
                "POSITIVE", normalized_scores.get("positive", 0)
            ) - normalized_scores.get("NEGATIVE", normalized_scores.get("negative", 0))

            if abs(sentiment_score) < neutral_threshold:
                dominant_sentiment = Sentiment.NEUTRAL.value
            elif sentiment_score > 0:
                dominant_sentiment = Sentiment.POSITIVE.value
            else:
                dominant_sentiment = Sentiment.NEGATIVE.value

            results.append({"score": sentiment_score, "label": dominant_sentiment})

        return results
