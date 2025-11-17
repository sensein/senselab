"""Utilities for reading and saving audio files.

This module provides high-level functions for audio I/O
(reading and saving multiple files).
It integrates with the `Audio` data structure defined in `senselab.audio`.
"""

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from senselab.audio.data_structures import Audio


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
