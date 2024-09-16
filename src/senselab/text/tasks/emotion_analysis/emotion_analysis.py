"""This module provides functionality for performing emotional analysis on text using a specified model."""

from collections import defaultdict
from typing import Any, Dict, List, Optional

from senselab.utils.data_structures.device import DeviceType, _select_device_and_dtype
from senselab.utils.interfaces.analyses import BaseTextAnalysis
from senselab.utils.model_utils import BaseModelSourceUtils
from senselab.utils.tasks.chunking import chunk_text


class EmotionAnalysis(BaseTextAnalysis):
    """A class for performing emotional analysis on text."""

    @classmethod
    def analyze(
        cls,
        input_data: List[Any],
        model_utils: BaseModelSourceUtils,
        device: Optional[DeviceType],
        **kwargs: Any,  # noqa: ANN401
    ) -> List[Dict[str, Any]]:
        """Analyze the emotional content of a list of text pieces.

        Args:
            input_data (List[Any]): A list of text strings to be analyzed.
            model_utils (BaseModelSourceUtils): Utility class for model operations.
            device (Optional[DeviceType]): The device to use for computation (e.g., CPU, CUDA).
            **kwargs (Any): Additional keyword arguments, such as:
                - max_length (int): Maximum length of text chunks (default: 512).
                - overlap (int): Overlap size between text chunks (default: 128).

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing:
                - scores (Dict[str, float]): Probabilities for each emotion.
                - dominant_emotion (str): The most likely emotion in the text.

        Raises:
            ValueError: If the input list is empty or None.
        """
        if not input_data:
            raise ValueError("Input list is empty or None.")

        max_length = int(kwargs.get("max_length", 512))
        overlap = int(kwargs.get("overlap", 128))

        device, torch_dtype = _select_device_and_dtype(
            user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )

        tokenizer = model_utils.get_tokenizer(task="text-classification")
        pipe = model_utils.get_pipeline(
            task="text-classification", device=device, torch_dtype=torch_dtype, return_all_scores=True
        )

        results: List[Dict[str, Any]] = []

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

            dominant_emotion = max(normalized_scores, key=lambda label: normalized_scores[label])

            results.append({"scores": normalized_scores, "dominant_emotion": dominant_emotion})

        return results
