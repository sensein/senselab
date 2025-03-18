"""This module contains implementations related to audio processing."""

from senselab.utils.data_structures.ffmpeg import check_ffmpeg_version

check_ffmpeg_version(min_version=4.3, max_version=7)  # Check if ffmpeg is installed and is compatible with torchaudio
