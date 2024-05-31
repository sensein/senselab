"""Data structures relevant for audio tasks and pipelines

Contains data structures that are useful for audio tasks and pipelines that this package defines.
The most basic unit is an Audio object which represents the necessary information of a loaded audio
file and its corresponding metadata. Other functionality and abstract data types are provided for
ease of maintaining the codebase and offering consistent public APIs.
"""

from typing import Dict, Optional, Union, List
import uuid
import torch
import numpy as np
import math

from pydantic import BaseModel, Field, validator
import torchaudio


class Audio(BaseModel):
    """Pydantic model for audio and its corresponding metadata

    Pydantic model for audio that holds the necessary attributes, the actual decoded audio data
    and the sampling rate, to work with audio in python. Contains metadata information as needed
    and has a unique identifier for every audio.

    Attributes:
        audio_data: The actual audio data read from an audio file, stored as a torch.Tensor
            of shape (num_channels, num_samples)
        sampling_rate: The sampling rate of the audio file
        path_or_id: A unique identifier for the audio, defined either by the path the audio was 
            read from or an UUID
        metadata: Optional metadata dictionary of information associated with this Audio instance
            (e.g. participant demographics, audio settings, location information)
    """

    audio_data: Union[torch.Tensor, np.ndarray, List[List[float]], List[float]]
    sampling_rate: int
    path_or_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Optional[Dict] = Field(default=None)

    @validator('audio_data', pre=True)
    def convert_to_tensor(cls, v):
        """Converts the audio data to torch.Tensor of shape (num_channels, num_samples)
        """
        temporary_tensor = None
        if isinstance(v, list):
            temporary_tensor = torch.tensor(v)
        elif isinstance(v, np.ndarray):
            temporary_tensor = torch.tensor(v)
        elif isinstance(v, torch.Tensor):
            temporary_tensor = v
        else:
            raise ValueError('Unsupported data type')
        
        if len(temporary_tensor.shape) == 1:
            # make the audio data [channels=1, samples]
            temporary_tensor = temporary_tensor.unsqueeze(0)
        return temporary_tensor
    
    @classmethod
    def from_filepath(cls, filepath: str, metadata: Optional[Dict] = None) -> "Audio":
        """Creates an Audio instance from an audio file

        Args:
            filepath: Filepath of the audio file to read from
            metadata: Additional information associated with the audio file
        """
        array, sampling_rate = torchaudio.load(filepath)
        
        return cls(audio_data=array, sampling_rate=sampling_rate, path_or_id=filepath, metadata=metadata)
        

class AudioDataset:
    """Class for maintaining collections of Audios and their corresponding metadata.

    Maintains a collection of Audios and their corresponding metadata for easy integration
    with Pydra Workflows.

    Attributes:
        audios: List of Audios that are either generated based on list of filepaths or 
            already read audio data and their respective sampling rates
        metadata: Metadata related to the dataset overall but not necessarily the metadata of 
            indivudal audios in the dataset
        use_gpu: Flag for determining whether a GPU should be used by default when preparing the 
            dataset for a Pydra workflow
        batch_size: How to batch the audio data for a GPU when preparing the dataset for a Pydra workflow
    """

    def __init__(self, audios: List[Audio], metadata: Optional[Dict] = None, 
                 use_gpu: Optional[bool] = torch.cuda.is_available(), batch_size: Optional[int] = 16):
        """Instantiates the dataset of the given Audios

        Mostly internal instantiator for generating the dataset where external creation will mostly happen
        through the use of the generate_ functions.

        Args:
            audios: List of Audio data that the class maintains and efficiently split-up for Pydra Tasks
            metadata: Optional metadata for the audio dataset, separate from each Audio's individual metadata
            use_gpu: Optional boolean of whether the default Pydra workflow should be split based on running 
                on a CPU vs. a GPU
            batch_size: The number of audios to split into for a Pydra Task that will run on a GPU
        """
        self.metadata = metadata.copy()
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.audios = audios.copy()

    @classmethod
    def generate_dataset_from_filepaths(cls, audio_filepaths: List[str], audio_metadatas: Optional[List[Dict]], 
                                        use_gpu: Optional[bool], batch_size: Optional[int], dataset_metadata: Optional[Dict]) -> "AudioDataset":
        """ Generate an audio dataset from a list of audio filepaths

        Generates an audio dataset by taking in a list of audio filepaths and using torchaudio to decode them.

        Args:
            audio_filepaths: List of filepaths to audios to be decoded
            audio_metadatas: List of corresponding metadata for each audio in audios_data
            use_gpu: Optional boolean of whether the default Pydra workflow should be split based on running 
                on a CPU vs. a GPU
            batch_size: The number of audios to split into for a Pydra Task that will run on a GPU
            dataset_metadata: Metadata relevant to the whole dataset
        """
        audios = []
        for i in range(len(audio_filepaths)):
            audio_metadata =  audio_metadatas[i] if audio_metadatas else None
            audio = Audio.from_filepath(audio_filepaths[i], audio_metadata)
            audios.append(audio)

        return cls(audios, dataset_metadata, use_gpu, batch_size)

    @classmethod
    def generate_dataset_from_audio_data(cls, audios_data: List[List[float]|List[List[float]]|torch.Tensor|np.ndarray], 
                                         sampling_rates: int|List[int], audio_metadatas: Optional[List[Dict]], 
                                        use_gpu: Optional[bool], batch_size: Optional[int], dataset_metadata: Optional[Dict]) -> "AudioDataset":
        """ Generate an audio dataset from already "read" audio files

        Generates an audio dataset by taking in a list of audio data (defined either by a List 
        of floats for mono audio and List of List of floats, NumPy arrays, or torch Tensors for 
        mono or stereo audio). Requires knowing the sampling rate for every audio data passed in and
        passing in each Audio's metadata in at once in a parallel List to audios_data.

        Args:
            audios_data: List of loaded audios (mono or stereo, supporting the 3 most common formats)
            sampling_rates: An integer if all of the Audios were generated with the same sampling rate or a
                list of sampling rates that is parallel to audios_data
            audio_metadatas: List of corresponding metadata for each audio in audios_data
            use_gpu: Optional boolean of whether the default Pydra workflow should be split based on running 
                on a CPU vs. a GPU
            batch_size: The number of audios to split into for a Pydra Task that will run on a GPU
            dataset_metadata: Metadata relevant to the whole dataset
        """
        audios = []
        for i in range(len(audios_data)):
            audio_metadata = audio_metadatas[i] if audio_metadatas else None
            sampling_rate = sampling_rates[i] if isinstance(sampling_rates, List) else sampling_rates
            audio = Audio(audio_data=audios_data[i], sampling_rate=sampling_rate, metadata=audio_metadata)
            audios.append(audio)
        
        return cls(audios, dataset_metadata, use_gpu, batch_size)


    def create_split_for_pydra_task(self, use_gpu: Optional[bool] = None, 
                                    batch_size: Optional[int] = None) -> List[List[Audio]]:
        """ Splits the audio data for Pydra tasks.

        Creates a split of the audio data that can be used for creating individual Pydra tasks using the .split functionality.
        Splits the data such that the inputs for a Pydra workflow are either optimized for the GPU's batch size or a single Audio per CPU thread.

        Args:
            use_gpu: Optional override of the class's use_gpu attribute for whether the split of the audio will be run on a GPU
            batch_size: Optional override of the class's batch_size attribute for how large the batches on the GPU should be

        Returns:
            List of Lists of Audio where each List of Audios will be an input to a Pydra task.
            Each of the sublists are either of size 1 for CPUs or at most batch_size for GPU optimization.

        Raises:
            None
        """
        pt_use_gpu = use_gpu if use_gpu else self.use_gpu
        pt_batch_size = batch_size if batch_size else self.batch_size

        if (pt_use_gpu):
            return [self.audios[pt_batch_size*i:min(pt_batch_size*(i+1), len(self.audios))] for i in range(math.ceil(len(self.audios)/pt_batch_size))]
        else:
            return [[audio] for audio in self.audios]
    