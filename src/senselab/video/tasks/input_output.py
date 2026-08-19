"""This module implements the video IOTask."""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Union

import av
import numpy as np
import torch

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import from_strings_to_files, get_common_directory, logger
from senselab.utils.portable_audio_io import subtype_preference
from senselab.utils.tasks.input_output import read_files_from_disk

# The codec hints a caller can pass, mapped to what the file is actually written at. An
# unlisted hint resolves to whatever preserves most of what the container can hold.
_SUBTYPE_FOR_ACODEC = {
    "pcm_s16le": "PCM_16",
    "pcm_s24le": "PCM_24",
    "pcm_s32le": "PCM_32",
    "pcm_f32le": "FLOAT",
    "pcm_f64le": "DOUBLE",
}


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
        acodec: Audio codec hint (default: pcm_s16le), mapped to the subtype the file is
            written at -- ``pcm_s16le`` to PCM_16, ``pcm_s24le`` to PCM_24, ``pcm_f32le`` to
            FLOAT, anything else to whatever preserves most of what ``audio_format`` can hold.
            A lossy decode can overshoot ±1; where the requested subtype cannot hold that,
            the excess is clipped and the loss is logged rather than raising, because a bulk
            extraction should not abort on one transient.

    Returns:
        A dataset dict of the extracted audio files.
    """

    def _extract_audio_from_local_video(video_path: Path, output_audio_path: str, fmt: str, codec: str) -> bool:
        """Extract audio from a video file using PyAV.

        Returns:
            True if audio was extracted, False if the video has no audio track.
        """
        with av.open(str(video_path)) as container:
            if not container.streams.audio:
                logger.warning("No audio track found in %s, skipping.", video_path)
                return False

            audio_stream = container.streams.audio[0]
            sample_rate = audio_stream.rate or 16000

            frames = []
            for frame in container.decode(audio=0):
                arr = frame.to_ndarray()
                if arr.ndim > 1:
                    arr = arr.mean(axis=0)  # downmix to mono
                frames.append(arr)

        if not frames:
            return False

        audio_data = np.concatenate(frames).astype(np.float32)

        # Through Audio, so the subtype resolution and the range policy are the ones every
        # other senselab write uses. out_of_range="warn": the caller chose the container, and a
        # bulk extraction should report a clipped transient rather than abort on it.
        Audio(waveform=torch.from_numpy(audio_data).unsqueeze(0), sampling_rate=sample_rate).save_to_file(
            output_audio_path,
            format=fmt,
            subtype=subtype_preference(fmt, _SUBTYPE_FOR_ACODEC.get(codec)),
            out_of_range="warn",
        )
        return True

    if isinstance(files, str):
        files = [files]
    formatted_files = from_strings_to_files(files)
    common_path = get_common_directory(files)

    with tempfile.TemporaryDirectory(prefix="senselab-video-io-") as temp_dir:
        audio_files_paths = []
        for file in formatted_files:
            rel_path = os.path.relpath(file.filepath, common_path)
            base_file_name = os.path.splitext(rel_path)[0]
            output_audio_path = os.path.join(temp_dir, f"{base_file_name}.{audio_format}")
            os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
            if _extract_audio_from_local_video(file.filepath, output_audio_path, fmt=audio_format, codec=acodec):
                audio_files_paths.append(output_audio_path)

        return read_files_from_disk(audio_files_paths) if audio_files_paths else {}
