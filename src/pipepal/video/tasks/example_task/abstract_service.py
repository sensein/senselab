"""This module defines an abstract service for the task."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class AbstractService(ABC):
    """Abstract base class for services.

    This class provides a template for services with preprocess, process,
    and postprocess methods.
    """
    def preprocess(self, data: Any) -> Any:  # noqa: ANN401
        """Preprocess input data.

        Args:
            data: The input data to preprocess.

        Returns:
            The preprocessed data.
        """
        return data

    @abstractmethod
    def process(self, data: Any) -> Dict[str, Any]: # noqa: ANN401
        """Process input data.

        Args:
            data: The input data to process.

        Returns:
            A dictionary with the processed output.
        """
    
    def postprocess(self, data: Any) -> Dict[str, Any]: # noqa: ANN401
        """Postprocess the processed data.

        :param data: The data to postprocess.
        :return: The postprocessed data.
        """
        return data