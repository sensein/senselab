"""Base class for text analysis pipelines."""

import string
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from senselab.utils.data_structures.device import DeviceType
from senselab.utils.model_utils import BaseModelSourceUtils


class BaseAnalysis(ABC):
    """Abstract base class for analysis pipelines."""

    @classmethod
    @abstractmethod
    def analyze(
        cls,
        input_data: List[Any],
        model_utils: BaseModelSourceUtils,
        device: Optional[DeviceType],
    ) -> List[Dict[str, Any]]:
        """Abstract method for analysis.

        Args:
            input_data (List[Any]): List of input data to analyze.
            model_utils (BaseModelSourceUtils): Utility class for model operations.
            device (Optional[DeviceType]): The device to use for computation.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing analysis results.
        """
        raise NotImplementedError("Subclasses must implement the analyze method.")

    @staticmethod
    @abstractmethod
    def validate_input(input: object) -> None:
        """Abstract method to validate input data.

        Args:
            input (object): The input data to validate.

        Raises:
            TypeError: If the input is not of the expected type.
            ValueError: If the input data is invalid according to specific criteria.
        """
        raise NotImplementedError("Subclasses must implement the validate_input method.")


class BaseTextAnalysis(BaseAnalysis):
    """Abstract base class for text analysis pipelines."""

    @staticmethod
    def validate_input(input: object) -> None:
        """Validate the input text.

        Args:
            input (object): The input text to validate.

        Raises:
            TypeError: If the input is not a string.
            ValueError: If the input string is empty, contains only whitespace, or only punctuation.
        """
        if not isinstance(input, str):
            raise TypeError("Input text must be a string.")
        if not input.strip():
            raise ValueError("Input text cannot be empty or contain only whitespace.")
        if input.strip(string.punctuation) == "":
            raise ValueError("Input text cannot contain only punctuation.")
