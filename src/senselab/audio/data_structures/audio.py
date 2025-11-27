"""Audio data structure module."""

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
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch
from pydantic import BaseModel, Field, PrivateAttr

from senselab.utils.constants import SENSELAB_NAMESPACE


class Audio(BaseModel):
    """Represents an audio file and its associated metadata.

    Users should instantiate Audio via the constructor (from file path or waveform + sampling rate)
    or the 'from_stream' method, which yiels Audio objects from a live audio stream.

    Attributes:
        metadata: A dictionary containing any additional metadata.
    """

    # Private attributes used for lazy loading and internal state.
    _file_path: Union[str, os.PathLike] = PrivateAttr(default="")  # Path to audio file (if not pre-loaded)
    _waveform: Optional[torch.Tensor] = PrivateAttr(default=None)  # Audio data (lazy-loaded from file if not provided)
    _sampling_rate: Optional[int] = PrivateAttr(default=None)  # Actual sampling rate; loaded on demand
    _offset_in_sec: float = PrivateAttr(default=0.0)  # Offset in seconds from which to start loading audio
    _duration_in_sec: Optional[float] = PrivateAttr(default=None)  # Duration in seconds to load; None means full file
    _backend: Optional[str] = PrivateAttr(default=None)  # Backend to use when loading the audio

    # Public fields:
    metadata: Dict = Field(default={})
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data: Any) -> None:  # noqa: ANN401,D417
        """Initialize an Audio instance.

        Args:
            waveform (optional): Pre-loaded audio data as a list, NumPy array, or torch.Tensor.
            sampling_rate (optional): If provided, sets the sampling rate.
                This must be provided if a waveform is supplied.
            filepath (optional): File path for lazy loading the waveform if not provided.
            offset_in_sec (optional): Offset (in seconds) from which to start reading the file. Defaults to 0.0.
            duration_in_sec (optional): Duration (in seconds) to read from the file. If None, the full file is loaded.
            backend (optional): I/O backend to use when loading the audio (e.g. "ffmpeg", "sox", "soundfile").
            metadata (optional): A dictionary of additional metadata.

        Raises:
            ValueError: If neither waveform nor filepath is provided.
        """
        waveform = data.pop("waveform", None)
        provided_sr = data.pop("sampling_rate", None)
        filepath = data.pop("filepath", None)
        offset_in_sec = data.pop("offset_in_sec", 0.0)
        duration_in_sec = data.pop("duration_in_sec", None)
        backend = data.pop("backend", None)
        metadata = data.pop("metadata", {})

        super().__init__(**data)

        if waveform is not None:
            # If a waveform and sampling rate are provided, convert and store them;
            if provided_sr is None:
                raise ValueError("When a waveform is provided, a sampling_rate must also be supplied.")
            self._waveform = self.convert_to_tensor(waveform)
            self._sampling_rate = provided_sr
        else:
            # otherwise, a valid filepath is required for lazy loading.
            if not filepath:
                raise ValueError("Either a waveform or a valid filepath must be provided to construct an Audio object.")
            elif not os.path.exists(filepath):
                raise FileNotFoundError(f"File {filepath} does not exist.")
            else:
                self._file_path = filepath

        # Validate offset
        if offset_in_sec < 0:
            raise ValueError("Offset must be a non-negative value")

        # Validate duration (allowing -1 to indicate full file)
        if duration_in_sec is not None and duration_in_sec < 0 and duration_in_sec != -1:
            raise ValueError("Duration must be -1 (for full file) or a positive value")

        # Validate backend if provided
        allowed_backends = {"ffmpeg", "sox", "soundfile"}
        if backend is not None and backend not in allowed_backends:
            raise ValueError("Unsupported backend")

        self._offset_in_sec = offset_in_sec
        self._duration_in_sec = duration_in_sec
        self._backend = backend

        # Set the metadata
        self.metadata = metadata

    @property
    def waveform(self) -> torch.Tensor:
        """Returns the audio waveform as a torch.Tensor.

        If the waveform has not been loaded yet, it is loaded lazily from the file.
        """
        if self._waveform is None:
            # print("Lazy loading audio data from file...")
            self._waveform = self.convert_to_tensor(self._lazy_load_data_from_filepath(self._file_path))
        assert self._waveform is not None, "Failed to load audio data."
        return self._waveform

    @property
    def sampling_rate(self) -> int:
        """Returns the sampling rate of the audio.

        If the sampling rate is not set and a file is available, it is inferred from the file metadata.
        """
        if self._sampling_rate is None:
            if self._file_path and TORCHAUDIO_AVAILABLE:
                info = torchaudio.info(self._file_path)
                self._sampling_rate = info.sample_rate
            else:
                raise ValueError("Sampling rate is not available.")
        assert self._sampling_rate is not None, "Sampling rate should be set."
        return self._sampling_rate

    @classmethod
    def convert_to_tensor(cls, v: Union[List[float], List[List[float]], np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Converts input audio data to a torch.Tensor with shape (num_channels, num_samples).

        Args:
            v: Audio data in the form of a list, NumPy array, or torch.Tensor.

        Returns:
            A torch.Tensor representation of the audio data.
        """
        if isinstance(v, list):
            temporary_tensor = torch.tensor(v)
        elif isinstance(v, np.ndarray):
            temporary_tensor = torch.tensor(v)
        elif isinstance(v, torch.Tensor):
            temporary_tensor = v.clone()
        else:
            raise ValueError("Unsupported data type for audio conversion.")

        if temporary_tensor.ndim == 1:
            temporary_tensor = temporary_tensor.unsqueeze(0)
        return temporary_tensor.to(torch.float32)

    def _lazy_load_data_from_filepath(self, filepath: Union[str, os.PathLike]) -> torch.Tensor:
        """Lazy-loads audio data from the given filepath.

        Converts the stored offset and duration (in seconds) to the required frame indices for torchaudio.

        Args:
            filepath: The path to the audio file.

        Returns:
            A torch.Tensor containing the loaded audio data.

        Raises:
            ModuleNotFoundError: If torchaudio is not available.
            ValueError: If the offset or duration exceeds the file duration.
        """
        if not TORCHAUDIO_AVAILABLE:
            raise ModuleNotFoundError(
                "`torchaudio` is not installed. "
                "Please install senselab audio dependencies using `pip install senselab`."
            )

        info = torchaudio.info(filepath)
        self._sampling_rate = info.sample_rate
        total_frames = info.num_frames

        # Convert offset_in_sec and duration_in_sec to frame indices.
        frame_offset = int(self._offset_in_sec * self.sampling_rate)
        if frame_offset > total_frames:
            raise ValueError(
                f"Offset ({self._offset_in_sec} s) exceeds the audio file duration "
                f"({total_frames / self.sampling_rate:.2f} s)."
            )
        if self._duration_in_sec is not None and self._duration_in_sec > 0:
            num_frames = int(self._duration_in_sec * self.sampling_rate)
            # Ensure we don't exceed the file length.
            num_frames = min(num_frames, total_frames - frame_offset)
        else:
            num_frames = -1  # Indicates full file reading

        array, _ = torchaudio.load(
            filepath,
            frame_offset=frame_offset,
            num_frames=num_frames,
            backend=self._backend,
        )
        return array

    def filepath(self) -> Union[str, None]:
        """Returns the file path of the audio if available."""
        if self._file_path:
            return str(self._file_path)
        return None

    def generate_id(self) -> str:
        """Generates a unique identifier for the Audio.

        The identifier is computed as an MD5-based UUID derived from the waveform and sampling rate.

        Returns:
            A string representing the generated unique identifier.
        """
        # Use the waveform property so that lazy loading is triggered if needed.
        unique_hash = uuid.uuid3(uuid.uuid3(SENSELAB_NAMESPACE, str(self.waveform)), str(self.sampling_rate))
        return str(unique_hash)

    def __eq__(self, other: object) -> bool:
        """Overrides equality to compare Audio objects based on their generated identifiers.

        Args:
            other: Another object to compare.

        Returns:
            True if both Audio instances have the same generated identifier, False otherwise.
        """
        if isinstance(other, Audio):
            return self.generate_id() == other.generate_id()
        return False

    def window_generator(self, window_size: int, step_size: int) -> Generator["Audio", None, None]:
        """Creates a sliding window generator for the audio waveform.

        Each yielded Audio instance corresponds to a window of the waveform.

        Args:
            window_size: Number of samples in each window.
            step_size: Number of samples to advance for each window.

        Yields:
            Audio: A new Audio instance representing the current window.
        """
        if step_size > window_size:
            warnings.warn("Step size is greater than window size. Some portions of the audio may not be included.")

        num_samples = self.waveform.size(-1)
        current_position = 0

        while current_position < num_samples:
            end_position = min(current_position + window_size, num_samples)
            window_waveform = self.waveform[:, current_position:end_position]

            yield Audio(
                waveform=window_waveform,
                sampling_rate=self.sampling_rate,
                metadata=self.metadata,
            )
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
        """Saves the Audio object to a file using torchaudio.save.

        Args:
            file_path: Destination file path.
            format: Audio format (e.g. "wav", "ogg", "flac"). Inferred from the file extension if None.
            encoding: Encoding to use (e.g. "PCM_S", "PCM_U"). Effective for formats like wav and flac.
            bits_per_sample: Bit depth (e.g. 8, 16, 24, 32, 64).
            buffer_size: Buffer size in bytes for processing.
            backend: I/O backend to use (e.g. "ffmpeg", "sox", "soundfile").
            compression: Compression level for supported formats (e.g. mp3, flac, ogg).

        Raises:
            ModuleNotFoundError: If torchaudio is not available.
            ValueError: If the waveform dimensions or sampling rate are invalid.
            RuntimeError: If saving fails.
        """
        if not TORCHAUDIO_AVAILABLE:
            raise ModuleNotFoundError(
                "`torchaudio` is not installed. "
                "Please install senselab audio dependencies using `pip install senselab`."
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
                os.makedirs(output_dir)
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

    @classmethod
    def from_stream(
        cls,
        stream_source: Union[str, os.PathLike, bytes],
        chunk_duration_in_sec: float = 1.0,
        metadata: Optional[Dict] = None,
    ) -> Generator["Audio", None, None]:
        """Yield Audio objects from a live audio stream in fixed-duration chunks.

        Args:
            stream_source: A file path, stream, or bytes-like object.
            chunk_duration_in_sec: Duration (in seconds) of each audio chunk.
            metadata: Additional metadata for each chunk.

        Yields:
            Audio objects for each chunk read from the stream.
        """
        if not SOUNDFILE_AVAILABLE:
            raise ModuleNotFoundError(
                "`soundfile` is not installed. "
                "Please install senselab audio dependencies using `pip install senselab`."
            )

        if isinstance(stream_source, (os.PathLike, str)) and not os.path.exists(stream_source):
            raise FileNotFoundError(f"File {stream_source} does not exist.")

        with sf.SoundFile(stream_source, "r") as audio_file:
            sampling_rate = audio_file.samplerate
            chunk_frames = int(chunk_duration_in_sec * sampling_rate)

            while True:
                chunk = audio_file.read(frames=chunk_frames, dtype="float32", always_2d=True)
                if chunk.shape[0] == 0:
                    break
                yield cls(
                    waveform=chunk.T,
                    sampling_rate=sampling_rate,
                    metadata=metadata if metadata else {},
                )


def batch_audios(audios: List[Audio]) -> Tuple[torch.Tensor, Union[int, List[int]], List[Dict]]:
    """Batches a list of Audio objects into a single Tensor while preserving individual metadata.

    Args:
        audios: List of Audio objects. They should all have the same number of channels.
                It is advised that they also share the same sampling rate when required by processing.

    Returns:
        A tuple containing:
            - A Tensor of shape (batch_size, num_channels, num_samples),
            - The sampling rate (as an integer if uniform, or a list otherwise),
            - A list of each audio's metadata.

    Raises:
        RuntimeError: If the Audio objects do not share the same number of channels.
    """
    sampling_rates = []
    num_channels_list = []
    lengths = []
    batched_audio = []
    metadatas = []

    for audio in audios:
        sampling_rates.append(audio.sampling_rate)
        num_channels_list.append(audio.waveform.shape[0])
        lengths.append(audio.waveform.shape[1])
        metadatas.append(audio.metadata)

    if len(set(num_channels_list)) != 1:
        raise RuntimeError("All audios must have the same number of channels.")

    if len(set(sampling_rates)) != 1:
        warnings.warn("Not all sampling rates are the same.", UserWarning)

    max_length = max(lengths)
    for audio in audios:
        waveform = audio.waveform
        padding = max_length - waveform.shape[1]
        if padding > 0:
            pad = torch.zeros((waveform.shape[0], padding), dtype=waveform.dtype)
            waveform = torch.cat([waveform, pad], dim=1)
        batched_audio.append(waveform)

    return_sampling_rate: Union[int, List[int]] = (
        int(sampling_rates[0]) if len(set(sampling_rates)) == 1 else sampling_rates
    )

    return torch.stack(batched_audio), return_sampling_rate, metadatas


def unbatch_audios(
    batched_audio: torch.Tensor, sampling_rates: Union[int, List[int]], metadatas: List[Dict]
) -> List[Audio]:
    """Unbatches a Tensor of audio data back into a list of Audio objects.

    Args:
        batched_audio: Tensor of shape (batch_size, num_channels, num_samples).
        sampling_rates: A single sampling rate (if uniform) or a list of sampling rates.
        metadatas: A list of metadata dictionaries for each audio.

    Returns:
        A list of Audio objects reconstituted from the batched data.

    Raises:
        ValueError: If the batched_audio shape is invalid or if the number of items mismatches.
    """
    if len(batched_audio.shape) != 3:
        raise ValueError("Expected batched_audio to have shape (batch_size, num_channels, num_samples).")
    if batched_audio.shape[0] != len(metadatas) or (
        isinstance(sampling_rates, list) and batched_audio.shape[0] != len(sampling_rates)
    ):
        raise ValueError("Batch size, sampling_rates, and metadatas must all have the same number of elements.")

    audios = []
    for i in range(len(metadatas)):
        sr = sampling_rates[i] if isinstance(sampling_rates, list) else sampling_rates
        audios.append(Audio(waveform=batched_audio[i], sampling_rate=sr, metadata=metadatas[i]))
    return audios
