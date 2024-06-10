"""Data structures relevant for managing datasets."""

import math
import uuid
from typing import Any, Dict, List, Union, no_type_check

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from senselab.audio.data_structures.audio import Audio
from senselab.video.data_structures.video import Video


class Participant(BaseModel):
    """Data structure for a participant in a dataset."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict = Field(default={})

    @field_validator("id", mode="before")
    def set_id(cls, v: str) -> str:
        """Set the unique id of the participant."""
        return v or str(uuid.uuid4())

    def __eq__(self, other: object) -> bool:
        """Overloads the default BaseModel equality to correctly check that ids are equivalent."""
        if isinstance(other, Participant):
            return self.id == other.id
        return False


class Session(BaseModel):
    """Data structure for a session in a dataset."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict = Field(default={})

    @field_validator("id", mode="before")
    def set_id(cls, v: str) -> str:
        """Set the unique id of the session."""
        return v or str(uuid.uuid4())

    def __eq__(self, other: object) -> bool:
        """Overloads the default BaseModel equality to correctly check that ids are equivalent."""
        if isinstance(other, Session):
            return self.id == other.id
        return False


class SenselabDataset(BaseModel):
    """Class for maintaining SenseLab datasets and functionalities.

    Maintains collections of Audios, Videos, and metadata for use of the Senselab tools
    and pipelines. Includes the ability to manage Sessions and Participants.

    Attributes:
        audios: List of Audios that are generated based on list of audio filepaths
        videos: List of Videos generated from a list of video filepaths
        metadata: Metadata related to the dataset overall but not necessarily the metadata of
            indivudal audios in the dataset
        sessions: Session ID mapping to Session instance
        participants: Mapping of participant ID to a Participant instance
    """

    participants: Dict[str, Participant] = Field(default_factory=dict)
    sessions: Dict[str, Session] = Field(default_factory=dict)
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

    def audio_merge_from_pydra_task(self, audios_to_merge: List[List[Audio]]) -> None:
        """Write later.

        Logic Pydra:
        audios: List of audios that want to give to task
        split: List[List[Audios]] -> task List[Audio]
        pydra task(List[Audio]) -> List[Audio]
        merge(List[List[Audio]]) <- might be a wrapped instead of List of lists
        TODO: Figure out what a merge behavior looks like from Pydra
        """
        self.audios = []
        for audio_task_input in audios_to_merge:
            for audio_output in audio_task_input:
                self.audios.append(audio_output)

    @field_validator("participants", mode="before")
    def check_unique_participant_id(cls, v: Dict[str, Participant], values: Any) -> Dict[str, Participant]:  # noqa: ANN401
        """Check if participant IDs are unique."""
        print("type(values)")
        print(type(values))
        input("Press Enter to continue...")
        participants = values.get("participants", {})
        for participant_id, _ in v.items():
            if participant_id in participants:
                raise ValueError(f"Participant with ID {participant_id} already exists.")
        return v

    @field_validator("sessions", mode="before")
    def check_unique_session_id(cls, v: Dict[str, Session], values: Any) -> Dict[str, Session]:  # noqa: ANN401
        """Check if session IDs are unique."""
        sessions = values.get("sessions", {})
        for session_id, _ in v.items():
            if session_id in sessions:
                raise ValueError(f"Session with ID {session_id} already exists.")
        return v

    def add_participant(self, participant: Participant) -> None:
        """Add a participant to the dataset."""
        if participant.id in self.participants:
            raise ValueError(f"Participant with ID {participant.id} already exists.")
        self.participants[participant.id] = participant

    def add_session(self, session: Session) -> None:
        """Add a session to the dataset."""
        if session.id in self.sessions:
            raise ValueError(f"Session with ID {session.id} already exists.")
        self.sessions[session.id] = session

    def get_participants(self) -> List[Participant]:
        """Get the list of participants in the dataset."""
        return list(self.participants.values())

    def get_sessions(self) -> List[Session]:
        """Get the list of sessions in the dataset."""
        return list(self.sessions.values())
