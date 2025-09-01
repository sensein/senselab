"""Data structures relevant for video tasks and pipelines."""

import os
import uuid
from typing import Any, Dict, List, Optional, Union

import numpy as np
import PIL
import torch
from pydantic import BaseModel, Field
from torchvision.io import read_video

from senselab.audio.data_structures import Audio
from senselab.utils.constants import SENSELAB_NAMESPACE

try:
    import av  # noqa: F401

    AV_AVAILABLE = True
except ModuleNotFoundError:
    AV_AVAILABLE = False


class Video(BaseModel):
    """Pydantic model for video and its corresponding metadata.

    This class supports both pre-loaded video data (frames and frame_rate) and lazy-loading from a filepath.
    The associated audio is kept as a private attribute.

    Attributes:
        metadata: A dictionary with additional video-related metadata.
    """

    metadata: Dict = Field(default={})
    model_config = {"arbitrary_types_allowed": True}

    # Private attributes for lazy loading and internal state:
    _file_path: Optional[Union[str, os.PathLike]] = None
    _frames: Optional[torch.Tensor] = None  # 4D Tensor: (T, H, W, C)
    _frame_rate: Optional[float] = None  # Frames per second
    _audio: Optional[Audio] = None  # Private Audio object (if available)

    def __init__(self, **data: Any) -> None:  # noqa: ANN401,D417
        """Initialize a Video instance.

        Args:
            frames (optional): Pre-loaded video frames as a 4D tensor (T, H, W, C) or a list of frames.
            frame_rate (optional): The corresponding frame rate (fps). Required if frames is provided.
            filepath (optional): Path to the video file for lazy loading.
            metadata (optional): Metadata dictionary.
            audio (optional): Associated audio data.

        Raises:
            ValueError: If neither frames nor a valid filepath is provided, or if frames is provided without frame_rate.
        """
        frames = data.pop("frames", None)
        provided_frame_rate = data.pop("frame_rate", None)
        filepath = data.pop("filepath", None)
        metadata = data.pop("metadata", {})
        audio = data.pop("audio", None)
        super().__init__(**data)

        if frames is not None:
            if provided_frame_rate is None:
                raise ValueError("When video frames are provided, a frame_rate must also be supplied.")
            self._frames = self.convert_to_tensor(frames)
            self._frame_rate = provided_frame_rate
            if audio is not None:
                self._audio = audio
        else:
            if not filepath:
                raise ValueError(
                    "Either pre-loaded frames or a valid filepath must be provided to construct a Video object."
                )
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File {filepath} does not exist.")
            self._file_path = filepath

        self.metadata = metadata

    @classmethod
    def convert_to_tensor(
        cls, v: Union[List[Union[torch.Tensor, np.ndarray, PIL.Image.Image]], torch.Tensor]
    ) -> torch.Tensor:
        """Converts input video frames to a 4D torch.Tensor of shape (T, H, W, C).

        Args:
            v: Video frames provided as a list of frames or a torch.Tensor.

        Returns:
            A torch.Tensor representation of the video frames.

        Raises:
            ValueError: If the provided input does not conform to the expected types or shapes.
        """
        if isinstance(v, torch.Tensor):
            if v.ndim != 4:
                raise ValueError("Expected a 4D tensor with shape (T, H, W, C) for video frames.")
            return v
        elif isinstance(v, list):
            frames = []
            for frame in v:
                if isinstance(frame, (torch.Tensor, np.ndarray)):
                    frame_tensor = torch.as_tensor(frame)
                    if frame_tensor.ndim != 3:
                        raise ValueError("Each frame must be a 3D tensor with shape (H, W, C).")
                    frames.append(frame_tensor)
                elif isinstance(frame, PIL.Image.Image):
                    frame_tensor = torch.as_tensor(np.array(frame))
                    if frame_tensor.ndim != 3:
                        raise ValueError("Each frame (from PIL.Image) must be a 3D tensor with shape (H, W, C).")
                    frames.append(frame_tensor)
                else:
                    raise ValueError("Frames must be of type torch.Tensor, numpy array, or PIL.Image.")
            return torch.stack(frames)
        else:
            raise ValueError("Unsupported data type for converting video frames.")

    @property
    def frames(self) -> torch.Tensor:
        """Returns the video frames as a 4D torch.Tensor with shape (T, H, W, C).

        Triggers lazy loading from the file path if the frames have not been loaded yet.
        """
        if self._frames is None:
            self._load_from_filepath()
        assert self._frames is not None, "Failed to load frames: _frames is still None."
        return self._frames

    @property
    def frame_rate(self) -> float:
        """Returns the frame rate (fps) of the video.

        Triggers lazy loading from the file path if not already loaded.
        """
        if self._frame_rate is None:
            self._load_from_filepath()
        assert self._frame_rate is not None, "Failed to load frame rate: _frame_rate is still None."
        return self._frame_rate

    @property
    def audio(self) -> Optional[Audio]:
        """Returns the associated audio data as a private Audio instance."""
        if self._audio is None:
            self._load_from_filepath()
        assert self._audio is not None, "Failed to load audio: _audio is still None."
        return self._audio

    def _load_from_filepath(self) -> None:
        """Lazy-loads video (and optionally audio) from the given file path.

        Uses torchvision.io.read_video to load video data and extract the frame rate.
        If audio is present, a private Audio instance is created.

        Raises:
            ValueError: If no file path is available for lazy loading.
            FileNotFoundError: If the file does not exist.
            ModuleNotFoundError: If AV is not available.
        """
        if not self._file_path:
            raise ValueError("No file path available for lazy loading.")
        if not os.path.exists(self._file_path):
            raise FileNotFoundError(f"File {self._file_path} does not exist.")
        if not AV_AVAILABLE:
            raise ModuleNotFoundError(
                "`av` is not installed. "
                "Please install senselab video dependencies using `pip install 'senselab[video]'`."
            )
        # Load video frames, audio frames, and metadata.
        v_frames, a_frames, v_metadata = read_video(filename=self._file_path, pts_unit="sec")
        self._frames = v_frames
        self._frame_rate = v_metadata.get("video_fps")

        # Process audio if available.
        audio_fps = v_metadata.get("audio_fps")
        if a_frames is not None and a_frames.size(0) > 0 and audio_fps:
            self._audio = Audio(waveform=a_frames, sampling_rate=audio_fps)

    def generate_id(self) -> str:
        """Generate a unique identifier for the Video instance.

        The identifier is computed as an MD5-based UUID derived from the video frames and frame rate.
        If a private Audio object exists, its generate_id() is incorporated.

        Returns:
            A string unique identifier.
        """
        temp_hash = uuid.uuid3(uuid.uuid3(SENSELAB_NAMESPACE, str(self.frames)), str(self.frame_rate))
        return str(temp_hash) if self._audio is None else str(uuid.uuid3(temp_hash, self._audio.generate_id()))

    def __eq__(self, other: object) -> bool:
        """Determines equality between Video instances based on their generated unique identifiers."""
        if isinstance(other, Video):
            return self.generate_id() == other.generate_id()
        return False
