"""Utility functions for quality control."""

import os
from pathlib import Path
from typing import List, Set, Union

from senselab.utils.data_structures.logging import logger


def validate_audio_paths(
    audio_paths: List[Union[str, os.PathLike]], raise_on_empty: bool = True
) -> List[Union[str, os.PathLike]]:
    """Validate that audio paths exist and are files.

    Args:
        audio_paths: List of paths to audio files to validate
        raise_on_empty: If True, raise ValueError when no valid paths are found.
                       If False, return empty list instead.

    Returns:
        List of valid audio file paths

    Raises:
        ValueError: If raise_on_empty is True and no valid audio files are found
    """
    valid_audio_paths = []
    for path in audio_paths:
        path_obj = Path(path)
        if not path_obj.exists():
            logger.warning(f"Audio file does not exist, skipping: {path}")
            continue
        if not path_obj.is_file():
            logger.warning(f"Path is not a file, skipping: {path}")
            continue
        valid_audio_paths.append(path)

    if not valid_audio_paths and raise_on_empty:
        raise ValueError("No valid audio files found in the provided paths")

    if len(valid_audio_paths) < len(audio_paths):
        logger.warning(f"Skipped {len(audio_paths) - len(valid_audio_paths)} invalid audio paths")

    return valid_audio_paths


def get_audio_files_from_directory(
    directory_path: str,
    audio_extensions: Set[str] = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".wma"},
    recursive: bool = True,
) -> List[str]:
    """Get a list of all audio files in a directory.

    Args:
        directory_path: Path to the directory to search for audio files
        audio_extensions: Set of audio file extensions to search for
                         (default includes common formats)
        recursive: If True, search recursively through subdirectories

    Returns:
        List of paths to audio files found in the directory
    """
    audio_files: List[Path] = []
    directory = Path(directory_path)

    if not directory.exists():
        logger.warning(f"Directory {directory_path} does not exist")
        return []

    if not directory.is_dir():
        logger.warning(f"{directory_path} is not a directory")
        return []

    # Convert extensions to lowercase for case-insensitive matching
    extensions_lower = {ext.lower() for ext in audio_extensions}

    # Search for audio files
    if recursive:
        # Find all files recursively
        for ext in extensions_lower:
            audio_files.extend(directory.rglob(f"*{ext}"))
            # Also search for uppercase extensions
            audio_files.extend(directory.rglob(f"*{ext.upper()}"))
    else:
        # Find files in current directory only
        for ext in extensions_lower:
            audio_files.extend(directory.glob(f"*{ext}"))
            # Also search for uppercase extensions
            audio_files.extend(directory.glob(f"*{ext.upper()}"))

    # Convert to strings, remove duplicates, and sort for consistent ordering
    audio_files_str = list(set(str(path) for path in audio_files))
    audio_files_str.sort()

    logger.info(f"Found {len(audio_files_str)} audio files in {directory_path}")
    if audio_files_str:
        # Show summary of file types found
        extensions_found = set(Path(f).suffix.lower() for f in audio_files_str)
        logger.info(f"File types found: {', '.join(sorted(extensions_found))}")

    return audio_files_str
