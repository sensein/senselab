"""This module implements an example service for the task."""

import os
from typing import Any, Dict

from datasets import Audio, Dataset, load_dataset

from ...abstract_service import AbstractService


class Service(AbstractService):
    """Datasets service that extends AbstractService."""

    NAME: str = "Datasets"

    def __init__(self, configs: Dict[str, Any]) -> None: # noqa: ANN401
        """Initialize the service with given configurations.

        Args:
            configs: A dictionary of configurations for the service.
        """
        super().__init__()

    def read_audios_from_disk(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Reads audio files from disk and creates a Hugging Face `Dataset` object.

        This function checks if all audio files listed under the 'files' key in the input dictionary exist on disk.
        If all files are found, it creates a Dataset object where each file is handled as an audio file. The resulting
        dataset is then returned inside a dictionary under the key 'output'.

        Parameters:
        data (Dict[str, Any]): A dictionary with a key 'files' which is a list of strings. Each string should be
                            the file path to an audio file.

        Returns:
        Dict[str, Any]: A dictionary with a single key 'output', which contains the `Dataset` object. The 'audio'
                        column of this dataset is of type `datasets.Audio`.

        Raises:
        ValueError: If the 'files' key is not in the input dictionary.
        FileNotFoundError: If any of the audio files listed in the 'files' key do not exist.

        Example:
        >>> data = {"files": ["path/to/audio1.wav", "path/to/audio2.wav"]}
        >>> output_dataset = self.read_audios_from_disk(data)
        >>> print(type(output_dataset["output"]))
        <class 'datasets.arrow_dataset.Dataset'>
        >>> print(output_dataset["output"].column_names)
        ['audio']
        """
        # Check if 'files' key exists in the data dictionary
        if "files" not in data:
            raise ValueError("Input data must contain 'files' key with a list of audio file paths.")
        
        # Controlling that the input files exist
        missing_files = [file for file in data["files"] if not os.path.exists(file)]
        if missing_files:
            raise FileNotFoundError(f"The following files were not found: {missing_files}")
        
        # Creating the Dataset object
        audio_dataset = Dataset.from_dict({"audio": data["files"]})
        
        # Specifying the column type as Audio
        audio_dataset = audio_dataset.cast_column("audio", Audio(mono=False))
        
        # Wrapping the Dataset object in a dictionary
        return {"output": audio_dataset}

    def save_HF_dataset_to_disk(self, input_obj: Dict[str, Any]) -> Dict[str, Any]:
        """Saves a Hugging Face `Dataset` object to disk.

        Parameters:
        input_obj (Dict[str, Any]): A dictionary with 
            - a key 'dataset' which is a `Dataset` object,
            - a key 'output_path' which is a string representing the path to the output directory.

        Returns:
        None

        Todo:
        - Add error handling
        - Add output format as an optional parameter
        """
        # Use os.makedirs to create the output directory, ignore error if it already exists
        os.makedirs(input_obj['output_path'], exist_ok=True)

        # Saving the Dataset object to disk
        input_obj["dataset"].save_to_disk(input_obj["output_path"])

        return { "output": None }

    def upload_HF_dataset_to_HF_hub(self, input_obj: Dict[str, Any]) -> Dict[str, Any]:
        """Uploads a Hugging Face `Dataset` object to the Hugging Face Hub.

        Parameters:
        input_obj (Dict[str, Any]): A dictionary with 
            - a key 'dataset' which is a `Dataset` object,
            - a key 'output_uri' which is a string representing the URI to the remote directory.

        Returns:
        None

        Todo:
        - Add error handling
        - Add output format as an optional parameter
        - Add token handling for private HF repositories
        """   
        # Uploading the Dataset object to the Hugging Face Hub
        input_obj["dataset"].push_to_hub(input_obj["output_uri"])

        return { "output": None }
    
    def read_local_HF_dataset(self, input_obj: Dict[str, Any]) -> Dict[str, Any]:
        """Loads a Hugging Face `Dataset` object from a local directory.

        Args:
            input_obj (Dict[str, Any]): A dictionary containing:
                - path (str): The file path to the local directory containing the dataset.

        Returns:
            Dict[str, Any]: A dictionary with a key 'output', containing the loaded `Dataset` object.
        """
        dataset = Dataset.load_from_disk(input_obj["path"])
        return {"output": dataset}

    def read_HF_dataset_from_HF_hub(self, input_obj: Dict[str, Any]) -> Dict[str, Any]:
        """Loads a Hugging Face `Dataset` object from the Hugging Face Hub.

        Args:
            input_obj (Dict[str, Any]): A dictionary containing:
                - uri (str): The URI to the dataset on the Hugging Face Hub.

        Returns:
            Dict[str, Any]: A dictionary with a key 'output', containing the loaded `Dataset` object.
        """
        dataset = load_dataset(input_obj["uri"])
        return {"output": dataset}