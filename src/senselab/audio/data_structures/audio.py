"""Data structures relevant for handling audio files and metadata.

The most basic unit is the Audio object which represents the necessary information of a loaded audio
file and its corresponding metadata. Other functionality and abstract data types are provided for
ease of maintaining the codebase and offering consistent public APIs.
"""

try:
    import torchaudio

    TORCHAUDIO_AVAILABLE = True
except ModuleNotFoundError:
    TORCHAUDIO_AVAILABLE = False

try:
    import soundfile as sf

    SOUNDFILE_AVAILABLE = True
except ModuleNotFoundError:
    SOUNDFILE_AVAILABLE = False

import os
import uuid
import warnings
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch
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
    orig_path_or_id: str | os.PathLike | None = None
    metadata: Dict = Field(default={})
    model_config = {"arbitrary_types_allowed": True}

    @field_validator("waveform", mode="before")
    def convert_to_tensor(
        cls, v: Union[List[float], List[List[float]], np.ndarray, torch.Tensor], _: ValidationInfo
    ) -> torch.Tensor:
        """Converts the audio data to a torch.Tensor of shape (num_channels, num_samples)."""
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
    def from_filepath(
        cls,
        filepath: str | os.PathLike,
        offset_in_sec: float = 0.0,
        duration_in_sec: Optional[float] = None,
        metadata: Optional[Dict] = None,
        backend: Optional[str] = None,
    ) -> "Audio":
        """Creates an Audio instance from an audio file.

        Args:
            filepath: Filepath of the audio file to read from.
            offset_in_sec: Offset in seconds to start reading the audio file.
                Must be a non-negative value. Defaults to 0.0 (start from the beginning).
            duration_in_sec: Duration in seconds to read from the audio file.
                Defaults to None, which is interpreted as -1 (read the entire file).
            metadata: Additional information associated with the audio file.
                Defaults to an empty dictionary.
            backend: I/O backend to use. Valid options include "ffmpeg", "sox", and "soundfile".
                If None, a backend is automatically selected among available options.

        Raises:
            ModuleNotFoundError: If torchaudio is not installed.
            FileNotFoundError: If the specified filepath does not exist.
            ValueError: If offset_in_sec is negative.
            ValueError: If duration_in_sec is provided and is neither -1 nor a positive value.
            ValueError: If the offset exceeds the duration of the audio.
            ValueError: If audio metadata retrieval fails.

        Returns:
            Audio: An instance containing the audio data and its corresponding metadata.
        """
        if not TORCHAUDIO_AVAILABLE:
            raise ModuleNotFoundError(
                "`torchaudio` is not installed. "
                "Please install senselab audio dependencies using `pip install senselab['audio']`."
            )

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} does not exist.")

        if offset_in_sec < 0:
            raise ValueError("Offset must be a non-negative value.")

        # If duration is None, use -1 to indicate the full file
        if duration_in_sec is None:
            duration_in_sec = -1.0

        # Allow duration to be either -1 (full file) or strictly positive.
        if duration_in_sec != -1 and duration_in_sec <= 0:
            raise ValueError("Duration must be -1 (to read full file) or a positive value.")

        # Retrieve audio metadata
        try:
            info = torchaudio.info(filepath)
            sampling_rate = info.sample_rate
            total_frames = info.num_frames
        except Exception as e:
            raise ValueError(f"Failed to retrieve audio metadata for file {filepath}: {str(e)}") from e

        # Convert time-based offset and duration to frame-based values
        frame_offset = int(offset_in_sec * sampling_rate)
        if frame_offset > total_frames:
            raise ValueError(
                f"Offset ({offset_in_sec} seconds) exceeds the duration of the audio file ({filepath}): "
                f"{total_frames / sampling_rate} seconds."
            )

        num_frames = int(duration_in_sec * sampling_rate) if duration_in_sec > 0 else -1

        # Ensure num_frames is within the bounds of the file
        if num_frames > 0:
            num_frames = min(num_frames, total_frames - frame_offset)

        # Load the specified portion of the audio
        array, sampling_rate = torchaudio.load(
            filepath, frame_offset=frame_offset, num_frames=num_frames, backend=backend
        )

        return cls(
            waveform=array,
            sampling_rate=sampling_rate,
            orig_path_or_id=filepath,
            metadata=metadata if metadata else {},
        )

    @classmethod
    def from_stream(
        cls,
        stream_source: Union[str, os.PathLike, bytes],
        chunk_duration: float = 1.0,
        metadata: Optional[Dict] = None,
    ) -> Generator["Audio", None, None]:
        """Yields Audio objects from a live stream (file, stdin, etc.) in fixed-duration chunks.

        Args:
            stream_source: Filepath or input stream.
            chunk_duration: Duration (in seconds) of each audio chunk to read.
            metadata: Additional metadata associated with the audio chunk.

        Raises:
            ModuleNotFoundError: If soundfile is not installed.
            FileNotFoundError: If the specified stream_source does not exist (when provided as a path).
            RuntimeError: If an error occurs while reading from the stream.

        Yields:
            Audio: An instance containing each streamed audio chunk.
        """
        if not SOUNDFILE_AVAILABLE:
            raise ModuleNotFoundError(
                "`soundfile` is not installed. "
                "Please install senselab audio dependencies using `pip install senselab['audio']`."
            )

        if isinstance(stream_source, os.PathLike) and not os.path.exists(stream_source):
            raise FileNotFoundError(f"File {stream_source} does not exist.")

        # Open the audio stream
        with sf.SoundFile(stream_source, "r") as audio_file:
            sampling_rate = audio_file.samplerate
            # Convert chunk duration (in seconds) to frames
            chunk_frames = int(chunk_duration * sampling_rate)

            while True:
                chunk = audio_file.read(frames=chunk_frames, dtype="float32", always_2d=True)
                if chunk.shape[0] == 0:
                    break  # Stop when there are no more frames to read
                yield cls(
                    waveform=chunk.T,
                    sampling_rate=sampling_rate,
                    orig_path_or_id=stream_source,
                    metadata=metadata if metadata else {},
                )

    def generate_path(self) -> str | os.PathLike:
        """Generate a path-like string for this Audio.

        Generates a path-like string for the Audio by either utilizing the orig_path_or_id, checking
        if it is a path (has an extension), otherwise using the id if orig_path_or_id is not an ID,
        and giving an extension relative to the current directory.
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

        Returns:
            str: The UUID of the Audio generated by an MD5 hash of the waveform and the sampling_rate.
        """
        return str(uuid.uuid3(uuid.uuid3(SENSELAB_NAMESPACE, str(self.waveform)), str(self.sampling_rate)))

    def __eq__(self, other: object) -> bool:
        """Overloads the default BaseModel equality to correctly check equivalence, ignoring metadata."""
        if isinstance(other, Audio):
            return self.id() == other.id()
        return False

    def window_generator(self, window_size: int, step_size: int) -> Generator["Audio", None, None]:
        """Creates a sliding window generator for the audio.

        Creates a generator that yields Audio objects corresponding to each window of the waveform
        using a sliding window. The window size and step size are specified in number of samples.
        If the audio waveform doesn't contain an exact number of windows, the remaining samples
        will be included in the last window.

        Args:
            window_size: Size of each window (number of samples).
            step_size: Step size for sliding the window (number of samples).

        Yields:
            Audio: Audio objects corresponding to each window of the waveform.
        """
        if step_size > window_size:
            warnings.warn(
                "Step size is greater than window size. Some of the audio will not be included in the windows."
            )

        num_samples = self.waveform.size(-1)
        current_position = 0

        while current_position < num_samples:
            # Calculate the end position of the window
            end_position = current_position + window_size

            # Adjust end_position if it exceeds total samples
            if end_position > num_samples:
                end_position = num_samples

            # Get the windowed waveform
            window_waveform = self.waveform[:, current_position:end_position]

            # Create a new Audio instance for this window
            window_audio = Audio(
                waveform=window_waveform,
                sampling_rate=self.sampling_rate,
                orig_path_or_id=f"{self.orig_path_or_id}_{current_position}_{end_position}",
                metadata=self.metadata,
            )

            yield window_audio
            current_position += step_size

    def save_to_file(
        self,
        file_path: Union[str, os.PathLike],
        format: Optional[str] = None,
        encoding: Optional[str] = None,
        bits_per_sample: Optional[int] = None,
        buffer_size: int = 4096,
        backend: Optional[str] = None,
        compression: Optional[Union[float, int]] = None,
    ) -> None:
        """Save the `Audio` object to a file using `torchaudio.save`.

        Args:
            file_path (Union[str, os.PathLike]): The path to save the audio file.
            format (Optional[str]): Audio format to use. Valid values include "wav", "ogg", and "flac".
                If None, the format is inferred from the file extension.
            encoding (Optional[str]): Encoding to use. Valid options include "PCM_S", "PCM_U", "PCM_F", "ULAW", "ALAW".
                This is effective for formats like "wav" and "flac".
            bits_per_sample (Optional[int]): Bit depth for the audio file. Valid values are 8, 16, 24, 32, and 64.
            buffer_size (int): Size of the buffer in bytes for processing file-like objects. Default is 4096.
            backend (Optional[str]): I/O backend to use. Valid options include "ffmpeg", "sox", and "soundfile".
                If None, a backend is automatically selected.
            compression (Optional[Union[float, int]]): Compression level for supported formats like "mp3",
            "flac", and "ogg".
                Refer to `torchaudio.save` documentation for specific compression levels.

        Raises:
            ValueError: If the `Audio` waveform is not 2D, or if the sampling rate is invalid.
            RuntimeError: If there is an error saving the audio file.

        Note:
            - https://pytorch.org/audio/master/generated/torchaudio.save.html
        """
        if not TORCHAUDIO_AVAILABLE:
            raise ModuleNotFoundError(
                "`torchaudio` is not installed. "
                "Please install senselab audio dependencies using `pip install senselab['audio']`."
            )

        if self.waveform.ndim != 2:
            raise ValueError("Waveform must be a 2D tensor with shape (num_channels, num_samples).")

        if self.sampling_rate <= 0:
            raise ValueError("Sampling rate must be a positive integer.")

        output_dir = os.path.dirname(file_path)
        if not os.access(output_dir, os.W_OK):
            raise RuntimeError(f"Output directory '{output_dir}' is not writable.")

        try:
            if not os.path.exists(output_dir):
                os.makedirs(os.path.dirname(file_path))
            torchaudio.save(
                uri=file_path,
                src=self.waveform,
                sample_rate=self.sampling_rate,
                channels_first=True,
                format=format,
                encoding=encoding,
                bits_per_sample=bits_per_sample,
                buffer_size=buffer_size,
                backend=backend,
                compression=compression,
            )
        except Exception as e:
            raise RuntimeError(f"Error saving audio to file: {e}") from e


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
    num_channels_list = []
    lengths = []
    batched_audio = []
    metadatas = []

    for audio in audios:
        sampling_rates.append(audio.sampling_rate)
        num_channels_list.append(audio.waveform.shape[0])  # Assuming waveform shape is (num_channels, num_samples)
        lengths.append(audio.waveform.shape[1])
        metadatas.append(audio.metadata)

    # Check if all audios have the same number of channels
    if len(set(num_channels_list)) != 1:
        raise RuntimeError("All audios must have the same number of channels.")

    # Raise a warning if sampling rates are not the same
    if len(set(sampling_rates)) != 1:
        warnings.warn("Not all sampling rates are the same.", UserWarning)

    # Pad waveforms to the same length
    max_length = max(lengths)
    for audio in audios:
        waveform = audio.waveform
        padding = max_length - waveform.shape[1]
        if padding > 0:
            pad = torch.zeros((waveform.shape[0], padding), dtype=waveform.dtype)
            waveform = torch.cat([waveform, pad], dim=1)
        batched_audio.append(waveform)

    return_sampling_rates: Union[int, List[int]] = (
        int(sampling_rates[0]) if len(set(sampling_rates)) == 1 else sampling_rates
    )

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
