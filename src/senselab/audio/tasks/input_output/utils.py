"""Utilities for reading and saving audio files.

This module provides high-level functions for audio I/O
(reading and saving multiple files).
It integrates with the `Audio` data structure defined in `senselab.audio`.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures.logging import logger


def read_audios(file_paths: List[str | os.PathLike]) -> List[Audio]:
    """Read multiple audio files into `Audio` objects.

    Each file path is wrapped into an `Audio` object, allowing for convenient
    handling of audio metadata and waveform data in downstream tasks.

    Args:
        file_paths (list[str | os.PathLike]):
            A list of **paths** to audio files.

    Returns:
        list[Audio]: A list of `Audio` objects, one for each input path.

    Example:
        >>> from pathlib import Path
        >>> from senselab.audio.tasks.input_output import read_audios
        >>> files = [Path("sample1.wav").resolve(), Path("sample2.wav").resolve()]
        >>> audios = read_audios(files)
        >>> len(audios)
        2
        >>> audios[0].filepath
        '/absolute/path/to/sample1.wav'
    """
    # Sequential loading
    return [Audio(filepath=path) for path in file_paths]


def save_audios(
    audio_tuples: Sequence[Tuple[Audio, Union[str, os.PathLike]]], save_params: Optional[Dict[str, Any]] = None
) -> None:
    """Save multiple `(Audio, output_path)` pairs.

    Args:
        audio_tuples (Sequence[tuple[Audio, str | os.PathLike]]):
            A sequence of pairs ``(audio, output_path)``, where:
              * ``audio`` is an `Audio` object.
              * ``output_path`` is the target **path** to write to.
        save_params (dict, optional):
            Keyword arguments forwarded to `Audio.save_to_file`, e.g.:
            ``{"format": "wav"}``.

    Returns:
        None: Files are written to disk.

    Example:
        >>> from pathlib import Path
        >>> from senselab.audio.tasks.input_output import save_audios
        >>> from senselab.audio.data_structures import Audio
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> a2 = Audio(filepath=Path("sample2.wav").resolve())
        >>> outs = [
        ...     (a1, Path("out1.wav").resolve()),
        ...     (a2, Path("out2.wav").resolve()),
        ... ]
        >>> save_audios(outs)
    """
    params = save_params or {}

    # Sequential save
    for audio, out_path in audio_tuples:
        audio.save_to_file(os.fspath(out_path), **params)
    return None


def get_valid_audio_paths(
    audio_paths: List[Union[str, os.PathLike]], raise_on_empty: bool = True
) -> List[Union[str, os.PathLike]]:
    """Gets paths from the input that exist and are files.

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
