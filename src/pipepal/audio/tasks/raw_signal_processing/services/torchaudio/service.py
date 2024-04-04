"""This module implements an example service for the task."""

from datasets import Dataset
import numpy as np
import functools
import torch
from typing import Any, Dict, List

from ...abstract_service import AbstractService


class Service(AbstractService):
    """torchaudio service that extends AbstractService."""

    NAME: str = "torchaudio"

    def __init__(self, configs: Dict[str, Any]) -> None:  # noqa: ANN401
        """Initialize the service with given configurations.

        Args:
            configs: A dictionary of configurations for the service.
        """
        super().__init__()

    def preprocess(self, data: Any) -> Any:  # noqa: ANN401
        """Preprocess input data. Implementation can be customized.

        Args:
            data: The input data to preprocess.

        Returns:
            The preprocessed data.
        """
        return super().preprocess(data)

    def process(self, data: Any) -> Dict[str, Any]:  # noqa: ANN401
        """Process input data. Custom implementation for ExampleService.

        Args:
            data: The input data to process.

        Returns:
            A dictionary containing 'output' key with a sample output.
        """
        new_dataset = data["dataset"]
        if "channeling" in data:
            new_dataset = filter_dataset_channels(new_dataset, 
                                                  channels_to_keep=data['channeling']['channels_to_keep'])
        print("SHSHSJSJSJS")
        input("sjsk")

        return {"output": new_dataset}

    def postprocess(self, data: Any) -> Any:  # noqa: ANN401
        """Postprocess processed data. Implementation can be customized.

        Args:
            data: The data to postprocess.

        Returns:
            The postprocessed data.
        """
        return super().postprocess(data)


def filter_dataset_channels(dataset: Dataset, channels_to_keep: List[int]) -> Dataset:
    """Applies channel filtering to all audio objects in a specified column of a Hugging Face Dataset.
    
    Parameters:
    - dataset (Dataset): The Hugging Face Dataset to process. Assumes the dataset has a column
      containing audio objects with 'array' and 'sampling_rate'.
    - channels_to_keep (List[int]): A list of channel indices to keep in the filtered audio objects.

    Returns:
    - Dataset: A new Hugging Face Dataset with filtered audio objects in the specified column.
    """
    
    def filter_audio_channels(batch):
        """Filters specified channels from audio objects in a batch and returns modified audio objects.
        
        Parameters:
        - batch: A batch from the dataset containing 'audio' objects with 'array' and 'sampling_rate'.
        
        Returns:
        - The batch with filtered audio arrays.
        """
        print("channels_to_keep is: ", channels_to_keep)


        # Extract arrays and sampling_rates
        arrays = [item['array'] for item in batch['audio']]
        sampling_rates = [item['sampling_rate'] for item in batch['audio']]

        # Ensure arrays are PyTorch tensors for efficient processing
        processed_arrays = []
        for array in arrays:
            array = torch.tensor(array) if not isinstance(array, torch.Tensor) else array
            if array.ndim == 1:
                array = array.unsqueeze(0)  # Ensure there's a channel dimension
            filtered_array = array[channels_to_keep]  # Filter channels
            processed_arrays.append(filtered_array.squeeze().numpy())  # Convert back to NumPy array for consistency

        # Update the 'audio' objects in the batch
        for i, item in enumerate(batch['audio']):
            item['array'] = processed_arrays[i]
            item['sampling_rate'] = sampling_rates[i]

        return batch

    # Apply the channel filtering using the map function with batched processing
    return dataset.map(
        function=filter_audio_channels,
        batched=True,
        batch_size=None,  # Auto-batch size or specify your own
        with_indices=False,  # Set to True if you need indices within the mapping function
        remove_columns=None  # Specify if you want to remove columns post mapping
    )


'''
def filter_dataset_channels(dataset: Dataset, channels_to_keep: List[int]) -> Dataset:
    """Applies channel filtering to all audio objects in a specified column of a Hugging Face Dataset.

    Parameters:
    - dataset (Dataset): The Hugging Face Dataset to process. Assumes the dataset has a column
      containing audio objects with 'array' and 'sampling_rate'.
    - channels_to_keep (List[int]): A list of channel indices to keep in the filtered audio objects.

    Returns:
    - Dataset: A new Hugging Face Dataset with filtered audio objects in the specified column.
    """

    def filter_audio_channels(
        dataset_row: Dict[str, torch.Tensor], channels_to_keep: List[int]
    ) -> Dict[str, torch.Tensor]:
        """Filters specified channels from an audio object and returns a modified audio object.

        Parameters:
        - dataset_row (Dict[str, torch.Tensor]): An audio object containing 'array' and 'sampling_rate'.
        The 'array' is expected to be a tensor of shape (num_channels, num_frames).
        - channels_to_keep (List[int]): A list of channel indices to keep in the output audio object.

        Returns:
        - Dict[str, torch.Tensor]: A modified audio object with filtered channels. This object will contain
        'array' with only the specified channels and the same 'sampling_rate' as the input.

        Example:
        >>> dataset_row = {"array": torch.randn(2, 16000), "sampling_rate": 16000}
        >>> filtered_audio = filter_audio_channels(dataset_row["audio"], [0])  # Keep only the first channel
        >>> print(filtered_audio["audio"]["array"].shape)
        torch.Size([1, 16000])
        """
        # Extract the array and sampling_rate from the audio object
        array, sampling_rate = dataset_row["audio"]["array"], dataset_row["audio"]["sampling_rate"]

        if isinstance(array, np.ndarray):
            array = torch.from_numpy(array)

        if array.ndim == 1:
            # If array is 1D, treat it as a single channel (mono audio)
            array = array.unsqueeze(0)  # Add a channel dimension

        # Filter the channels
        filtered_array = array[channels_to_keep]

        filtered_array = filtered_array.squeeze(0)
        if isinstance(dataset_row["audio"]["array"], np.ndarray):
            filtered_array = filtered_array.numpy()
        filtered_array = filtered_array.astype(np.float32)

        # Return the filtered audio object
        return {
            "audio": {
                "array": filtered_array,
                "sampling_rate": sampling_rate,
                "path": dataset_row["audio"]["path"],
            }
        }

    def apply_filter(row):
        row = filter_audio_channels(row, channels_to_keep)
        return row

    return dataset.map(apply_filter)
'''

def convert_dataset_to_mono(dataset, conversion_method="average", channels=[0]):
    """Converts all audio files in a Hugging Face dataset from stereo to mono.

    Parameters:
    - dataset: The loaded Hugging Face audio dataset.
    - conversion_method: Method of conversion ('average' or 'single').
    - channels: Channels to use for 'single' conversion method.

    Returns:
    - A new dataset with mono audio.
    """

    def convert_example_to_mono(
        audio_data: dict, conversion_method: str = "average", channels: List[int] = [0]
    ) -> Dict[str, Any]:
        """Converts stereo audio to mono based on the specified conversion method.

        Parameters:
        - audio_data: A dictionary containing 'array' (torch.Tensor) with the audio data and 'sampling_rate'.
        - conversion_method: 'average' for averaging all channels, 'single' to use specific channels.
        - channels: List of channel indices to use for conversion when conversion_method is 'single'.

        Returns:
        - A dictionary containing the mono audio data (torch.Tensor) and the sampling rate (int).
        """
        print(audio_data)
        input("ssjjs")
        if not audio_data or "array" not in audio_data or "sampling_rate" not in audio_data:
            raise ValueError(
                "audio_data must be a dictionary containing 'array' and 'sampling_rate'."
            )

        if conversion_method not in ["average", "single"]:
            raise ValueError("conversion_method must be either 'average' or 'single'.")

        if conversion_method == "average":
            mono_audio = torch.mean(audio_data["array"], dim=0, keepdim=True)
        elif conversion_method == "single":
            if channels is None or not all(isinstance(ch, int) for ch in channels):
                raise ValueError(
                    "When conversion_method is 'single', channels must be a list of integer indices."
                )
            mono_audio = torch.mean(audio_data["array"][channels], dim=0, keepdim=True)
        else:
            raise ValueError("Invalid conversion method.")

        return {"array": mono_audio, "sampling_rate": audio_data["sampling_rate"]}

    # Create a partial function with conversion_method and channels pre-specified
    convert_func = functools.partial(
        convert_example_to_mono, conversion_method=conversion_method, channels=channels
    )

    # Apply the conversion function to each example in the dataset
    return dataset.map(lambda example: convert_func(audio_data=example["audio"]))
