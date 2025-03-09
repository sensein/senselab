"""Data structures relevant for video tasks and pipelines."""

import os
import uuid
from typing import Dict, List, Optional, Union

import numpy as np
import PIL
import torch
from pydantic import BaseModel, Field, ValidationInfo, field_validator
from torchvision.io import read_video

from senselab.audio.data_structures import Audio
from senselab.utils.constants import SENSELAB_NAMESPACE

try:
    import av  # noqa: F401

    PYAV_AVAILABLE = True
except ModuleNotFoundError:
    PYAV_AVAILABLE = False


class Video(BaseModel):
    """Pydantic model for video and its corresponding metadata.

    Pydantic model for video that holds the necessary attributes, the actual decoded video data
    and the frame rate, to work with videos in python. Contains metadata information as needed
    and has a unique identifier for every video.

    Attributes:
        frames: Represent the video as a Tensor of all of its frames, each of which is an image
            that we represent through a Tensor of (C, H, W)
        frame_rate: Also known as the frames per second (fps), defines the time component
            of a video (often an integer but some use cases of float approximations)
        audio: the audio associated with the Video (optional)
        orig_path_or_id: Optional str for the original path or an ID to track file over time
        metadata: Optional metadata dictionary of information associated with this Video instance
            (e.g. participant demographics, video settings, location information)
    """

    frames: torch.Tensor
    frame_rate: float
    audio: Optional[Audio]
    orig_path_or_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict = Field(default={})
    model_config = {"arbitrary_types_allowed": True}

    @field_validator("frames", mode="before")
    def check_frames(
        cls, v: Union[torch.Tensor, List[Union[torch.Tensor, np.ndarray, PIL.Image.Image]]], _: ValidationInfo
    ) -> torch.Tensor:
        # print(v)
        """Check that the frames are the correct Tensor shape of (T,H,W,C)."""
        if isinstance(v, torch.Tensor):
            if len(v.shape) != 4:
                raise ValueError(
                    "Expected frames to be of shape (T, H, W, C) where T is the number of frames, \
                                C is the channels, and H and W are the height and width"
                )
            else:
                return v
        elif isinstance(v, List):
            transformed_frames = []
            # print('should get here, is list')
            for frame in v:
                if isinstance(frame, (torch.Tensor, np.ndarray)):
                    if len(frame.shape) != 3:
                        raise ValueError(
                            "Expected frame to be of shape (H, W, C) where \
                                        C is the channels, and H and W are the height and width"
                        )

                    transformed_frames.append(torch.Tensor(frame))
                elif isinstance(frame, PIL.Image.Image):
                    # print('should get here, is image')
                    transformed_frames.append(torch.Tensor(np.array(frame)))
                else:
                    raise ValueError(
                        "Expected each frame in the video to be either a Tensor, numpy array, or PIL.Image"
                    )
            # print(transformed_frames)
            return torch.stack(transformed_frames)
        else:
            raise ValueError("Expected sequence of frames to be either a List of frames or 4d Tensor")

    @classmethod
    def from_filepath(cls, filepath: str, metadata: Dict = {}) -> "Video":
        """Creates a Video instance from a video file.

        Args:
            filepath: Filepath of the video file to read from
            metadata: Additional information associated with the video file
        """
        if not PYAV_AVAILABLE:
            raise ModuleNotFoundError(
                "`pyav` is not installed. "
                "Please install senselab video dependencies using `pip install senselab['video']`."
            )

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} does not exist.")
        v_frames, a_frames, v_metadata = read_video(filename=filepath, pts_unit="sec")
        v_fps = v_metadata["video_fps"]
        a_fps = v_metadata["audio_fps"]
        no_ext_filepath = "random_id"  # os.path.splitext(filepath)[0]
        v_audio = Audio(waveform=a_frames, sampling_rate=a_fps, orig_path_or_id=f"{no_ext_filepath}.wav")

        return cls(frames=v_frames, frame_rate=v_fps, audio=v_audio, orig_path_or_id=filepath, metadata=metadata)

    def generate_path(self) -> str:
        """Generate a path like string for this Video.

        Generates a path like string for the Video by either utilizing the orig_path_or_id, checking
        if it is a path (has an extension), otherwise using the id if orig_path_or_id is not an ID
        and giving an extension and relative to the current directory.
        """
        if self.orig_path_or_id:
            if os.path.splitext(self.orig_path_or_id)[-1].lower():
                return self.orig_path_or_id
            else:
                return f"{self.orig_path_or_id}.mp4"
        else:
            return f"{self.id()}.mp4"

    def id(self) -> str:
        """Generate a unique identifier for the Video.

        Generate a unique identifier for the Video where equivalent video frames and frame rate
        and audio generate the same IDs.

        Returns: String UUID of the Video generated by an MD5 hash of the frames and the frame rate and audio
        """
        temp_hash = uuid.uuid3(uuid.uuid3(SENSELAB_NAMESPACE, str(self.frames)), str(self.frame_rate))
        return str(temp_hash) if not self.audio else str(uuid.uuid3(temp_hash, self.audio.id()))

    def __eq__(self, other: object) -> bool:
        """Overloads the default BaseModel equality to correctly check equivalence, ignoring metadata."""
        if isinstance(other, Audio):
            return self.id() == other.id()
        return False
