"""Data structures relevant for managing datasets."""

import math
import uuid
from typing import Dict, List, Union, no_type_check

import torch
from datasets import Audio as HFAudio
from datasets import Dataset, Features, Image, Sequence, Value
from pydantic import BaseModel, Field, ValidationInfo, field_validator
from transformers.image_transforms import to_pil_image

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

    def _get_dict_representation(self) -> Dict:
        audio_data: Dict[str, List] = {}
        video_data: Dict[str, List] = {}
        senselab_dict: Dict[str, Union[Dict[str, List], List]] = {
            "participants": [],
            "sessions": [],
            "audios": audio_data,
            "videos": video_data,
            "metadata": self.metadata.copy(),
        }
        participants_data = []
        sessions_data = []

        video_frames_data = []
        video_fps_data = []
        video_path_data = []
        video_metadata = []
        video_audio_data = []
        video_audio_metadata = []

        audio_waveform_data = []
        audio_metadata = []

        for participant in self.get_participants():
            participants_data.append({"id": participant.id, "metadata": participant.metadata.copy()})
        senselab_dict["participants"] = participants_data

        for session in self.get_sessions():
            sessions_data.append({"id": session.id, "metadata": session.metadata.copy()})
        senselab_dict["sessions"] = sessions_data

        for audio in self.audios:
            audio_waveform_data.append(
                {
                    "array": audio.waveform.T,
                    "sampling_rate": audio.sampling_rate,
                    "path": audio.generate_path(),
                }
            )
            audio_metadata.append(audio.metadata.copy())
        audio_data["audio"] = audio_waveform_data
        audio_data["metadata"] = audio_metadata

        for video in self.videos:
            video_frames_data.append({"image": [to_pil_image(frame.numpy()) for frame in list(video.frames)]})
            video_fps_data.append(video.frame_rate)
            video_path_data.append(video.generate_path())
            video_metadata.append(video.metadata.copy())
            video_audio_data.append(
                None
                if not video.audio
                else {
                    "array": video.audio.waveform.T.to(torch.float32).numpy(),
                    "sampling_rate": video.audio.sampling_rate,
                    "path": video.audio.generate_path(),
                }
            )
            video_audio_metadata.append(None if not video.audio else video.audio.metadata.copy())

        video_data["frames"] = video_frames_data
        video_data["frame_rate"] = video_fps_data
        video_data["path"] = video_path_data
        video_data["metadata"] = video_metadata
        video_data["audio"] = video_audio_data
        video_data["audio_metadata"] = video_audio_metadata
        # raise ValueError('fuck')
        return senselab_dict

    def convert_senselab_dataset_to_hf_datasets(self) -> Dict[str, Dataset]:
        """Converts Senselab datasets into HuggingFace datasets."""
        senselab_dict = self._get_dict_representation()

        # print(senselab_dict['videos']['audio'][0])

        features = Features(
            {
                "frames": {"image": Sequence(feature=Image())},
                "frame_rate": Value("float32"),
                "path": Value("string"),
                "metadata": {},
                "audio": HFAudio(mono=False, sampling_rate=48000),
                "audio_metadata": Value("string"),
            }
        )

        audio_dataset = Dataset.from_dict(senselab_dict["audios"]).cast_column("audio", HFAudio(mono=False))

        video_dataset = Dataset.from_dict(senselab_dict["videos"], features=features)
        # print(video_dataset[0])

        hf_datasets = {}
        hf_datasets["audios"] = audio_dataset
        hf_datasets["videos"] = video_dataset

        # TODO: Create datasets for participants and sessions
        return hf_datasets

    @classmethod
    def convert_hf_dataset_to_senselab_dataset(cls, hf_datasets: Dict[str, Dataset]) -> "SenselabDataset":
        """Converts HuggingFace dataset to a Senselab dataset."""
        audios = []
        videos = []
        sessions: Dict[str, Session] = {}
        participants: Dict[str, Participant] = {}
        if "audios" in hf_datasets:
            audio_dataset = hf_datasets["audios"]
            for audio in audio_dataset:
                audios.append(
                    Audio(
                        waveform=audio["audio"]["array"],
                        sampling_rate=audio["audio"]["sampling_rate"],
                        orig_path_or_id=audio["audio"]["path"],
                        metadata=audio["metadata"] if "metadata" in audio else {},
                    )
                )

        if "videos" in hf_datasets:
            video_dataset = hf_datasets["videos"]
            for video in video_dataset:
                videos.append(
                    Video(
                        frames=video["frames"]["image"],
                        frame_rate=video["frame_rate"],
                        metadata=video["metadata"] if "metadata" in video else {},
                        orig_path_or_id=video["path"],
                        audio=Audio(
                            waveform=video["audio"]["array"],
                            sampling_rate=video["audio"]["sampling_rate"],
                            orig_path_or_id=video["audio"]["path"],
                        )
                        if video["audio"]
                        else None,
                    )
                )
        if "sessions" in hf_datasets:
            pass
        if "participants" in hf_datasets:
            pass

        return SenselabDataset(participants=participants, sessions=sessions, audios=audios, videos=videos)

    def __eq__(self, other: object) -> bool:
        """Overloads the default BaseModel equality to correctly check that datasets are equivalent."""
        if isinstance(other, SenselabDataset):
            return (
                self.audios == other.audios
                and self.videos == other.videos
                and self.participants == other.participants
                and self.sessions == other.sessions
            )
        return False
