"""This module implements the video IOTask."""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Union

import ffmpeg

# import shutil
from senselab.utils.data_structures import from_strings_to_files, get_common_directory
from senselab.utils.tasks.input_output import read_files_from_disk


def extract_audios_from_local_videos(
    files: Union[str, List[str]],
    audio_format: str = "wav",
    acodec: str = "pcm_s16le",
) -> Dict[str, Any]:
    """Read files from disk and create a Hugging Face `Dataset` object."""

    def _extract_audio_from_local_video(video_path: Path, output_audio_path: str, format: str, acodec: str) -> None:
        """Extract audio from a video file."""
        try:
            # Input stream configuration
            input_stream = ffmpeg.input(video_path)

            # Audio extraction configuration
            audio_stream = input_stream.audio.output(output_audio_path, format=format, acodec=acodec)

            # Execute ffmpeg command
            ffmpeg.run(audio_stream, overwrite_output=True)

        except ffmpeg.Error as e:
            print("An error occurred while extracting audio:", str(e))

    if isinstance(files, str):
        files = [files]
    formatted_files = from_strings_to_files(files)
    common_path = get_common_directory(files)

    # Create a temporary directory to hold the audio files
    temp_dir = tempfile.mkdtemp()

    audio_files_paths = []
    for file in formatted_files:
        base_file_name = os.path.splitext(str(file.filepath).replace(common_path, ""))[0]
        output_audio_path = os.path.join(temp_dir, f"{base_file_name}.{audio_format}")
        _extract_audio_from_local_video(file.filepath, output_audio_path, format=audio_format, acodec=acodec)
        audio_files_paths.append(output_audio_path)

    audio_dataset = read_files_from_disk(audio_files_paths)

    # Clean up the temporary non empty directory
    # shutil.rmtree(temp_dir)
    return audio_dataset
