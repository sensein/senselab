"""This module defines an abstract service for the audio IO task."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class AbstractService(ABC):
    """Abstract base class for audio IO services.

    This class provides a template for services that handle audio input/output operations,
    ensuring that all essential methods such as reading audio from disk, saving datasets to disk,
    and uploading datasets to the Hugging Face Hub are implemented.
    """

    @abstractmethod
    def read_audios_from_disk(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Reads audio files from disk and creates a dataset.

        Args:
            data (Dict[str, Any]): A dictionary with a key 'files' which is a list of file paths to audio files.

        Returns:
            Dict[str, Any]: A dictionary with a single key 'output', containing the dataset created from the audio files.

        Raises:
            ValueError: If the 'files' key is not in the input dictionary.
            FileNotFoundError: If any audio files listed in the 'files' key do not exist.
        """
        pass

    @abstractmethod
    def save_HF_dataset_to_disk(self, input_obj: Dict[str, Any]) -> Dict[str, Any]:
        """Saves a Hugging Face `Dataset` object to disk.

        Args:
            input_obj (Dict[str, Any]): A dictionary containing:
                - 'dataset': A `Dataset` object to save.
                - 'output_path': A string representing the path to save the dataset to.

        Returns:
            Dict[str, Any]: A dictionary confirming the output status.
        """
        pass

    @abstractmethod
    def upload_HF_dataset_to_HF_hub(self, input_obj: Dict[str, Any]) -> Dict[str, Any]:
        """Uploads a Hugging Face `Dataset` object to the Hugging Face Hub.

        Args:
            input_obj (Dict[str, Any]): A dictionary containing:
                - 'dataset': A `Dataset` object to upload.
                - 'output_uri': A string representing the URI to the remote directory where the dataset will be uploaded.

        Returns:
            Dict[str, Any]: A dictionary confirming the upload status.
        """
        pass


    @abstractmethod
    def read_local_HF_dataset(self, input_obj: Dict[str, Any]) -> Dict[str, Any]:
        """Loads a Hugging Face `Dataset` object from a local directory.

        Args:
            input_obj (Dict[str, Any]): A dictionary containing:
                - path (str): The file path to the local directory containing the dataset.

        Returns:
            Dict[str, Any]: A dictionary with a key 'output', containing the loaded `Dataset` object.
        """
        pass

    @abstractmethod
    def read_HF_dataset_from_HF_hub(self, input_obj: Dict[str, Any]) -> Dict[str, Any]:
        """Loads a Hugging Face `Dataset` object from the Hugging Face Hub.

        Args:
            input_obj (Dict[str, Any]): A dictionary containing:
                - uri (str): The URI to the dataset on the Hugging Face Hub.

        Returns:
            Dict[str, Any]: A dictionary with a key 'output', containing the loaded `Dataset` object.
        """
        pass