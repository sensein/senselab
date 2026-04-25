"""This module implements the video IOTask."""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Union

import av
import numpy as np
import soundfile as sf

from senselab.utils.data_structures import from_strings_to_files, get_common_directory
from senselab.utils.tasks.input_output import read_files_from_disk


def extract_audios_from_local_videos(
    files: Union[str, List[str]],
    audio_format: str = "wav",
    acodec: str = "pcm_s16le",
) -> Dict[str, Any]:
    """Extract audio tracks from video files and return as a dataset.

    Uses PyAV (av) to decode the audio stream from each video file,
    then writes the raw audio to disk in the requested format.

    Args:
        files: Path(s) to video files.
        audio_format: Output audio format (default: wav).
        acodec: Audio codec hint (default: pcm_s16le). Used to select
            output sample format (16-bit signed int for pcm_s16le).

    Returns:
        A dataset dict of the extracted audio files.
    """

    def _extract_audio_from_local_video(video_path: Path, output_audio_path: str, fmt: str, codec: str) -> None:
        """Extract audio from a video file using PyAV."""
        container = av.open(str(video_path))
        audio_stream = container.streams.audio[0]
        sample_rate = audio_stream.rate or 16000

        frames = []
        for frame in container.decode(audio=0):
            arr = frame.to_ndarray()
            if arr.ndim > 1:
                arr = arr.mean(axis=0)  # downmix to mono
            frames.append(arr)
        container.close()

        if not frames:
            return

        audio_data = np.concatenate(frames)

        # Match codec hint to sample format
        if "s16" in codec:
            audio_data = (
                (audio_data * 32767).clip(-32768, 32767).astype(np.int16)
                if audio_data.dtype != np.int16
                else audio_data
            )

        sf.write(output_audio_path, audio_data, sample_rate, format=fmt.upper())

    if isinstance(files, str):
        files = [files]
    formatted_files = from_strings_to_files(files)
    common_path = get_common_directory(files)

    temp_dir = tempfile.mkdtemp()

    audio_files_paths = []
    for file in formatted_files:
        base_file_name = os.path.splitext(str(file.filepath).replace(common_path, ""))[0]
        output_audio_path = os.path.join(temp_dir, f"{base_file_name}.{audio_format}")
        os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
        _extract_audio_from_local_video(file.filepath, output_audio_path, fmt=audio_format, codec=acodec)
        audio_files_paths.append(output_audio_path)

    return read_files_from_disk(audio_files_paths)
