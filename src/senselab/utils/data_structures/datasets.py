"""Module defining datasets useful for the Senselab package."""

import math
from typing import Dict, List, Union, no_type_check

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from senselab.utils.data_structures.audio import Audio
from senselab.utils.data_structures.video import Video


class SenselabDataset(BaseModel):
    """Class for maintaining SenseLab datasets and functionalities.

    Maintains collections of Audios, Videos, and metadata for use of the Senselab tools
    and pipelines.

    Attributes:
        audios: List of Audios that are generated based on list of audio filepaths
        videos: List of Videos generated from a list of video filepaths
        metadata: Metadata related to the dataset overall but not necessarily the metadata of
            indivudal audios in the dataset
    """

    audios: List[Audio] = []
    videos: List[Video] = []
    metadata: Dict = Field(default={})

    @field_validator("audios", mode="before")
    @classmethod
    def generate_audios_from_filepaths(cls, v: Union[List[str], List[Audio]], _: ValidationInfo) -> List[Audio]:
        """Generate the audios in the dataset from a list of audio filepaths.

        Generates the audios in the dataset by taking in a list of audio filepaths
        or a list of Audios

        Args:
            v: Input for audios attribute that we're validating by generating the Audios if filepaths
                are provided or just the list of Audios if pre-generated and passed in

        Returns:
            List of Audios that instantiates the audios attribute in the dataset
        """
        audio_list = []
        if len(v) == 0:
            return []
        else:
            for audio in v:
                if isinstance(audio, Audio):
                    audio_list.append(audio)
                else:
                    audio_list.append(Audio.from_filepath(audio))
        return audio_list

    @field_validator("videos", mode="before")
    @classmethod
    def generate_videos_from_filepaths(cls, v: Union[List[str], List[Video]], _: ValidationInfo) -> List[Video]:
        """Generate the videos in the dataset from a list of video filepaths.

        Generates the videos in the dataset by taking in a list of video filepaths
        or a list of Videos

        Args:
            v: Input for videos attribute that we're validating by generating the Videos if filepaths
                are provided or just the list of Videos if pre-generated and passed in

        Returns:
            List of Videos that instantiates the videos attribute in the dataset
        """
        video_list = []
        if len(v) == 0:
            return []
        else:
            for video in v:
                if isinstance(video, Video):
                    video_list.append(video)
                elif isinstance(video, str):
                    video_list.append(Video.from_filepath(video))

                else:
                    raise ValueError("Unsupported video list")
        return video_list

    @classmethod
    @no_type_check
    def create_bids_dataset(cls, bids_root_filepath: str) -> "SenselabDataset":
        """Create a dataset from a BIDS organized directory.

        Creates a new dataset based off of a BIDS directory structure as defined at
        https://sensein.group/biometrics-book/updated_bids.html
        """
        pass

    def create_audio_split_for_pydra_task(self, batch_size: int = 1) -> List[List[Audio]]:
        """Splits the audio data for Pydra tasks.

        Creates a split of the audio data that can be used for creating individual Pydra tasks using
        the .split functionality. Splits the data such that the inputs for a Pydra workflow are either
        optimized for the GPU's batch size or a single Audio per CPU thread.

        Args:
            batch_size: How to batch Audios for a Pydra task; defaults to 1 since CPU won't batch

        Returns:
            List of Lists of Audio where each List of Audios will be an input to a Pydra task.
            Each of the sublists are either of size 1 for CPUs or at most batch_size for GPU optimization.

        Raises:
            ValueError if the batch size is invalid (less than 1)
        """
        if batch_size > 1:
            # Creates batches of at most size batch_size except the last which contains the remainder of audios
            return [
                self.audios[batch_size * i : min(batch_size * (i + 1), len(self.audios))]
                for i in range(math.ceil(len(self.audios) / batch_size))
            ]
        elif batch_size < 1:
            raise ValueError("Batch size must be greater than or equal to 1")
        else:
            return [[audio] for audio in self.audios]
