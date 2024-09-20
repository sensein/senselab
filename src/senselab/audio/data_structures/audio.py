"""Data structures relevant for handling audio files and metadata.

The most basic unit is the Audio object which represents the necessary information of a loaded audio
file and its corresponding metadata. Other functionality and abstract data types are provided for
ease of maintaining the codebase and offering consistent public APIs.
"""

import os
import uuid
import warnings
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
from pydantic import BaseModel, Field, ValidationInfo, field_validator

from senselab.utils.constants import SENSELAB_NAMESPACE


class Audio(BaseModel):
    """Pydantic model for audio and its corresponding metadata.

    Pydantic model for audio that holds the necessary attributes, the actual decoded audio data
    and the sampling rate, to work with audio in python. Contains metadata information as needed
    and has a unique identifier for every audio.

    Attributes:
        waveform: The actual audio data read from an audio file, stored as a torch.Tensor
            of shape (num_channels, num_samples)
        sampling_rate: The sampling rate of the audio file
        orig_path_or_id: Optional str for the original path or an ID to track file over time
        metadata: Optional metadata dictionary of information associated with this Audio instance
            (e.g. participant demographics, audio settings, location information)
    """

    waveform: torch.Tensor
    sampling_rate: int
    orig_path_or_id: Optional[str] = None
    metadata: Dict = Field(default={})
    model_config = {"arbitrary_types_allowed": True}

    @field_validator("waveform", mode="before")
    def convert_to_tensor(
        cls, v: Union[List[float], List[List[float]], np.ndarray, torch.Tensor], _: ValidationInfo
    ) -> torch.Tensor:
        """Converts the audio data to torch.Tensor of shape (num_channels, num_samples)."""
        temporary_tensor = None
        if isinstance(v, list):
            temporary_tensor = torch.tensor(v)
        elif isinstance(v, np.ndarray):
            temporary_tensor = torch.tensor(v)
        elif isinstance(v, torch.Tensor):
            temporary_tensor = v
        else:
            raise ValueError("Unsupported data type")

        if len(temporary_tensor.shape) == 1:
            # make the audio data [channels=1, samples]
            temporary_tensor = temporary_tensor.unsqueeze(0)
        return temporary_tensor.to(torch.float32)

    @classmethod
    def from_filepath(cls, filepath: str, metadata: Dict = {}) -> "Audio":
        """Creates an Audio instance from an audio file.

        Args:
            filepath: Filepath of the audio file to read from
            metadata: Additional information associated with the audio file
        """
        array, sampling_rate = torchaudio.load(filepath)

        return cls(waveform=array, sampling_rate=sampling_rate, orig_path_or_id=filepath, metadata=metadata)

    def generate_path(self) -> str:
        """Generate a path like string for this Audio.

        Generates a path like string for the Audio by either utilizing the orig_path_or_id, checking
        if it is a path (has an extension), otherwise using the id if orig_path_or_id is not an ID
        and giving an extension and relative to the current directory.
        """
        if self.orig_path_or_id:
            if os.path.splitext(self.orig_path_or_id)[-1].lower():
                return self.orig_path_or_id
            else:
                return f"{self.orig_path_or_id}"
        else:
            return f"{self.id()}"

    def id(self) -> str:
        """Generate a unique identifier for the Audio.

        Generate a unique identifier for the Audio where equivalent waveforms and sampling
        rates generate the same IDs.

        Returns: String UUID of the Audio generated by an MD5 hash of the waveform and the sampling_rate
        """
        return str(uuid.uuid3(uuid.uuid3(SENSELAB_NAMESPACE, str(self.waveform)), str(self.sampling_rate)))

    def __eq__(self, other: object) -> bool:
        """Overloads the default BaseModel equality to correctly check equivalence, ignoring metadata."""
        if isinstance(other, Audio):
            return self.id() == other.id()
        return False

    def window_iterator(self, window_size: int, step_size: int) -> Generator[torch.Tensor, None, None]:
        """Creates a sliding window iterator for the audio waveform.

        Args:
            window_size: Size of each window (number of samples).
            step_size: Step size for sliding the window (number of samples).

        Raises:
            ValueError: If step_size is greater than window_size.
        """
        if step_size > window_size:
            warnings.warn(
                "Step size is greater than window size. \
                          Some of audio will not be included in the windows."
            )
            step_size = window_size

        num_samples = self.waveform.size(1)
        current_position = 0

        while current_position <= num_samples - window_size:
            yield self.waveform[:, current_position : current_position + window_size]
            current_position += step_size


def batch_audios(audios: List[Audio]) -> Tuple[torch.Tensor, Union[int, List[int]], List[Dict]]:
    """Batches the Audios together into a single Tensor, keeping individual Audio information separate.

    Batch all of the Audios into a single Tensor of shape (len(audios), num_channels, num_samples).
    Keeps the Audio information related to each sampling rate and metadata separate for each Audio to
    allow for unbatching after running relevant functionality.

    Args:
        audios: List of audios to batch together. NOTE: Should all have the same number of channels
            and is generally advised to have the same sampling rates if running functionality
            that relies on the sampling rate.

    Returns:
        Returns a tuple of a Tensor that will have the shape (len(audios), num_channels, num_samples),
        the sampling rate (an integer if all have the same sampling rate), and a list of each individual
        audio's metadata information.

    Raises:
        RuntimeError: if all of the Audios do not have the same number of channels
    """
    sampling_rates = []
    batched_audio = []
    metadatas = []
    for audio in audios:
        sampling_rates.append(audio.sampling_rate)
        batched_audio.append(audio.waveform)
        metadatas.append(audio.metadata)

    return_sampling_rates: List[int] | int = int(sampling_rates[0]) if len(set(sampling_rates)) == 1 else sampling_rates

    return torch.stack(batched_audio), return_sampling_rates, metadatas


def unbatch_audios(batched_audio: torch.Tensor, sampling_rates: int | List[int], metadatas: List[Dict]) -> List[Audio]:
    """Unbatches Audios into a List of Audio objects.

    Uses the batched Audios, their respective sampling rates, and their corresponding metadatas to create
    a list of Audios.

    Args:
        batched_audio: torch.Tensor of shape (batch_size, num_channels, num_samples) to unstack
        sampling_rates: The sampling rate of each batched audio if they differ or a single sampling rate for all of them
        metadatas: The respective metadata for each of the batched audios

    Returns:
        List of Audio objects representing each of the Audios that were previously batched together

    Raises:
        ValueError if the batched_audio is not in the correct shape or if the number of batched_audios does not
            match the amount of metadata and sampling rates (if they were provided as a List) that were provided.
    """
    if len(batched_audio.shape) != 3:
        raise ValueError("Expected batched audios to be of shape (batch_size, num_channels, samples)")
    elif batched_audio.shape[0] != len(metadatas) or (
        isinstance(sampling_rates, List) and batched_audio.shape[0] != len(sampling_rates)
    ):
        raise ValueError(
            "Expected sizes of batched_audio, sampling_rates (if provided as a litst) \
                         and metadata to be equal"
        )

    audios = []
    for i in range(len(metadatas)):
        sampling_rate = sampling_rates[i] if isinstance(sampling_rates, List) else sampling_rates
        metadata = metadatas[i]
        audio = batched_audio[i]
        audios.append(Audio(waveform=audio, sampling_rate=sampling_rate, metadata=metadata))
    return audios
