"""This module provides utility classes for handling common utilities based on model type."""

from abc import ABC, abstractmethod
from typing import Callable, Dict

import torch
from transformers import AutoTokenizer, pipeline

from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.model import HFModel


class BaseModelSourceUtils(ABC):
    """Abstract base class for model source utilities.

    This class defines the interface for obtaining tokenizers and pipelines
    for different tasks.
    """

    @abstractmethod
    def get_tokenizer(self, task: str) -> object:
        """Get the tokenizer for a specific task.

        Args:
            task (str): The task for which the tokenizer is needed.

        Returns:
            object: The tokenizer for the specified task.

        Raises:
            NotImplementedError: If the method is not implemented in a derived class.
        """
        raise NotImplementedError("This method must be implemented in a derived class.")

    @abstractmethod
    def get_pipeline(self, task: str, device: DeviceType, torch_dtype: torch.dtype) -> Callable:
        """Get the pipeline for a specific task.

        Args:
            task (str): The task for which the pipeline is needed.
            device (DeviceType): The device to use for computation (e.g., CPU, CUDA).
            torch_dtype (torch.dtype): The data type to use for the pipeline.

        Returns:
            Callable: The pipeline for the specified task.

        Raises:
            NotImplementedError: If the method is not implemented in a derived class.
        """
        raise NotImplementedError("This method must be implemented in a derived class.")


class HFUtils(BaseModelSourceUtils):
    """Utility class for handling Hugging Face models and tokenizers.

    This class provides methods to obtain tokenizers and pipelines for
    different tasks, caching them to avoid redundant loading.
    """

    _instances: Dict[str, "HFUtils"] = {}

    def __init__(self, model: HFModel) -> None:
        """Initialize the HFUtils instance.

        Args:
            model (HFModel): The Hugging Face model configuration.
        """
        self._model: HFModel = model
        self._tokenizers: Dict[str, AutoTokenizer] = {}
        self._pipelines: Dict[str, Callable] = {}

    @staticmethod
    def _get_key(model: HFModel, *args: str) -> str:
        """Generate a consistent string key for caching purposes.

        Args:
            model (HFModel): The Hugging Face model configuration.
            *args (str): Additional arguments to include in the key.

        Returns:
            str: A string key combining the model information and additional arguments.
        """
        return f"{model.path_or_uri}-{model.revision}" + "".join(f"-{arg}" for arg in args if arg is not None)

    @classmethod
    def get_instance(cls, model: HFModel) -> "HFUtils":
        """Get or create an instance of HFUtils for a specific model.

        This method ensures that only one instance is created per unique model configuration.

        Args:
            model (HFModel): The Hugging Face model configuration.

        Returns:
            HFUtils: An instance of HFUtils for the specified model.
        """
        key = cls._get_key(model)
        if key not in cls._instances:
            cls._instances[key] = cls(model)
        return cls._instances[key]

    def get_tokenizer(self, task: str) -> AutoTokenizer:
        """Get the tokenizer for a specific task.

        Args:
            task (str): The task for which the tokenizer is needed.

        Returns:
            AutoTokenizer: The tokenizer for the specified task.
        """
        key = self._get_key(self._model, task)
        if key not in self._tokenizers:
            self._tokenizers[key] = AutoTokenizer.from_pretrained(
                self._model.path_or_uri, revision=self._model.revision
            )
        return self._tokenizers[key]

    def get_pipeline(self, task: str, device: DeviceType, torch_dtype: torch.dtype) -> Callable:
        """Get the pipeline for a specific task.

        Args:
            task (str): The task for which the pipeline is needed.
            device (DeviceType): The device to use for computation (e.g., CPU, CUDA).
            torch_dtype (torch.dtype): The data type to use for the pipeline.

        Returns:
            Callable: The pipeline for the specified task.
        """
        key = self._get_key(self._model, task, device.value, str(torch_dtype))
        if key not in self._pipelines:
            self._pipelines[key] = pipeline(
                task=task,
                model=self._model.path_or_uri,
                revision=self._model.revision,
                device=device.value,
                torch_dtype=torch_dtype,
                return_all_scores=True,
            )
        return self._pipelines[key]
