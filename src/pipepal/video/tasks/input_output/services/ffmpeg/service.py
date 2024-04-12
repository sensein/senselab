"""This module implements an example service for the task."""

import os
from typing import Any, Dict, List

import ffmpeg

from ...abstract_service import AbstractService


class Service(AbstractService):
    """Example service that extends AbstractService."""

    NAME: str = "ffmpeg"

    def __init__(self, configs: Dict[str, Any]) -> None: # noqa: ANN401
        """Initialize the service with given configurations.

        Args:
            configs: A dictionary of configurations for the service.
        """
        super().__init__()

    def extract_audios_from_videos(self, input_obj: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts audio files from the video files provided in the input dictionary.
        
        This function processes a list of video files, extracting the audio from each file and saving it
        in a specified format and directory. Each audio file's name is derived from its corresponding 
        video file, preserving the original hierarchy in the output directory.

        Parameters:
            input_obj (Dict[str, Any]): A dictionary containing the necessary inputs with the following keys:
                - 'files' (List[str]): A list of paths to the video files.
                - 'output_folder' (str): The directory where the extracted audio files will be saved.
                - 'audio_format' (str): The file extension for the output audio files (e.g., '.mp3', '.wav').
                - 'audio_codec' (str): The codec to use for encoding the audio files (e.g., 'mp3', 'aac').

        Returns:
            Dict[str, Any]: A dictionary containing one key:
                - 'audio_files' (List[str]): A list of paths to the extracted audio files.

        Example:
            >>> input_obj = {
                    'files': ['/path/to/video1.mp4', '/path/to/video2.avi'],
                    'output_folder': '/path/to/output',
                    'audio_format': '.wav',
                    'audio_codec': 'pcm_s16le'
                }
            >>> output = extract_audios_from_videos(input_obj)
            >>> output['audio_files']
            ['/path/to/output/video1.wav', '/path/to/output/video2.wav']

        Note:
            The function assumes that all video files reside under a common root directory.

        Todo:
            - Optimize the code for efficiency.
        """
        audio_files = []

        # Use os.makedirs to create the directory, ignore error if it already exists
        os.makedirs(input_obj['output_folder'], exist_ok=True)

        # Get the common root directory for all files
        common_path = get_common_directory(input_obj['files'])

        # Extract audio from each video
        for file_path in input_obj['files']:
            base_file_name = os.path.splitext(file_path.replace(common_path, ''))[0]
            output_audio_path = os.path.join(input_obj['output_folder'], base_file_name) + f".{input_obj['audio_format']}"            
            extract_audio_from_video(video_path=file_path,
                                    output_audio_path=output_audio_path,
                                    format=input_obj['audio_format'],
                                    acodec=input_obj['audio_codec'])
            audio_files.append(output_audio_path)

        return {
            'output': audio_files
        }


def get_common_directory(files: List[str]) -> str:
    """A function to get the common directory from a list of file paths.
    
    Parameters:
    - files: a list of file paths
    
    Returns:
    - the common directory among the file paths
    """
    if len(files) == 1:
        # Ensure the single path's directory ends with a separator
        common_path = os.path.dirname(files[0])
    else:
        # Use commonpath to find the common directory for multiple files
        common_path = os.path.commonpath(files)
    
    # Check if the path ends with the os separator, add if not
    if not common_path.endswith(os.sep):
        common_path += os.sep
    
    return common_path

def extract_audio_from_video(video_path: str, output_audio_path: str, format: str = 'wav', acodec: str = 'pcm_s16le') -> None:
    """Extracts all audio channels from a video file and saves it in the specified format and codec.
    
    This function utilizes the ffmpeg library to extract audio without re-encoding if the specified
    codec matches the source's codec. The output is in a format that can be specified (e.g., WAV, MP3),
    using the desired codec (e.g., PCM_S16LE for WAV, libmp3lame for MP3).

    Parameters:
        video_path (str): Path to the input video file. This should be the complete path to the video
                        file or a path that the script's context can resolve.
        output_audio_path (str): Path where the output audio file will be saved. This should include the
                                filename and the appropriate file extension based on the format.
        format (str): The format of the output audio file (default is 'wav'). Common formats include
                    'wav', 'mp3', 'aac', etc.
        acodec (str): The audio codec to use for the output file (default is 'pcm_s16le'). Common codecs
                    include 'pcm_s16le' (for WAV), 'libmp3lame' (for MP3), 'aac' (for AAC), etc.

    Returns:
        None: This function does not return any values but will raise an error if the extraction fails.

    Raises:
        ValueError: If the video path does not exist, or the specified format/codec is not supported.
        ffmpeg.Error: An error occurred during the ffmpeg processing, such as an issue with the input file
                    or codec compatibility.

    Examples:
        # Example 1: Extract audio in WAV format with default codec (PCM_S16LE)
        extract_audio_from_video("example.mp4", "output_audio.wav")

        # Example 2: Extract audio in MP3 format using the libmp3lame codec
        extract_audio_from_video("example.mp4", "output_audio.mp3", format="mp3", acodec="libmp3lame")

        # Example 3: Extract audio in AAC format with native AAC codec
        extract_audio_from_video("example.mp4", "output_audio.aac", format="aac", acodec="aac")

    Todo:
        - Dinamically check the supported formats and codecs
    """
    # Check if the video file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"The video file {video_path} does not exist.")

    # Validate format and codec
    valid_formats = ['wav', 'mp3', 'aac', 'flac', 'ogg', 'mka']
    valid_codecs = ['pcm_s16le', 'libmp3lame', 'aac', 'flac', 'libvorbis', 'copy']
    
    if format not in valid_formats:
        raise ValueError(f"Unsupported format: {format}. Supported formats: {valid_formats}")
    if acodec not in valid_codecs:
        raise ValueError(f"Unsupported codec: {acodec}. Supported codecs: {valid_codecs}")

    try:
        # Input stream configuration
        input_stream = ffmpeg.input(video_path)
        
        # Audio extraction configuration
        audio_stream = input_stream.audio.output(
            output_audio_path,
            format=format,
            acodec=acodec
        )
        
        # Execute ffmpeg command
        ffmpeg.run(audio_stream, overwrite_output=True)
        
    except ffmpeg.Error as e:
        print("An error occurred while extracting audio:", str(e))