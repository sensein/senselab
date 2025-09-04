"""Utilities for reading and saving audio files using Pydra workflows.

This module provides high-level functions for parallelized audio I/O
(reading and saving multiple files) by leveraging the Pydra workflow engine.
It integrates with the `Audio` data structure defined in `senselab.audio`.

The functions are particularly useful when dealing with large batches of files,
as they enable parallel execution using Pydra's execution plugins.
"""

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from pydra.compose import python, workflow

from senselab.audio.data_structures import Audio


def read_audios(
    file_paths: List[str | os.PathLike],
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    plugin: str = "debug",
    plugin_args: Optional[Dict[str, Any]] = None,
) -> List[Audio]:
    """Read multiple audio files into `Audio` objects using a Pydra workflow.

    Each file path is wrapped into an `Audio` object, allowing for convenient
    handling of audio metadata and waveform data in downstream tasks.

    Args:
        file_paths (list[str | os.PathLike]):
            A list of **absolute paths** to audio files.
            ⚠️ Required: must be absolute paths for Pydra to behave correctly.
        cache_dir (str | os.PathLike, optional):
            Directory for caching intermediate results of the workflow.
            If ``None``, Pydra will use its default cache directory.
        plugin (str, optional):
            The Pydra execution plugin to use for running the workflow.
            Common options:
              * ``"serial"`` or ``"debug"``: Run tasks sequentially (default).
              * ``"cf"``: Use concurrent futures for parallel execution.
              * ``"slurm"``: Submit tasks to a SLURM cluster.
        plugin_args (dict, optional):
            Extra keyword arguments passed to the chosen Pydra plugin.

    Examples:
              * For ``cf``: ``{"n_procs": 8}``
            See:https://nipype.github.io/pydra/

    Returns:
        list[Audio]: A list of `Audio` objects, one for each input path.

    Example:
        >>> from pathlib import Path
        >>> from senselab.audio.tasks.input_output import read_audios
        >>> files = [Path("sample1.wav").resolve(), Path("sample2.wav").resolve()]
        >>> audios = read_audios(files, plugin="cf")
        >>> len(audios)
        2
        >>> audios[0].filepath
        '/absolute/path/to/sample1.wav'
    """
    plugin_args = plugin_args or {}

    @python.define
    def _load_audio_file(path: str | os.PathLike) -> Audio:
        return Audio(filepath=path)

    @workflow.define
    def _wf(xs: Sequence[str | os.PathLike]) -> List[Audio]:
        node = workflow.add(
            _load_audio_file().split(path=xs),
            name="map_read_audio",
        )
        return node.out

    worker = "debug" if plugin in ("serial", "debug") else plugin
    res = _wf(xs=file_paths)(worker=worker, cache_root=cache_dir, **plugin_args)
    return list(res.out)


def save_audios(
    audio_tuples: Sequence[Tuple[Audio, Union[str, os.PathLike]]],
    save_params: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    plugin: str = "debug",
    plugin_args: Optional[Dict[str, Any]] = None,
) -> None:
    """Save multiple `(Audio, output_path)` pairs using a Pydra workflow.

    This function executes saves in parallel (depending on the plugin), enabling
    efficient batch export of audio files.

    Args:
        audio_tuples (Sequence[tuple[Audio, str | os.PathLike]]):
            A sequence of pairs ``(audio, output_path)``, where:
              * ``audio`` is an `Audio` object.
              * ``output_path`` is the target **absolute path** to write to.
            ⚠️ Required: all output paths must be absolute for Pydra to behave correctly.
        save_params (dict, optional):
            Keyword arguments forwarded to `Audio.save_to_file`, e.g.:
            ``{"format": "wav"}``.
        cache_dir (str | os.PathLike, optional):
            Directory for caching intermediate results of the workflow.
            If ``None``, Pydra will use its default cache directory.
        plugin (str, optional):
            The Pydra execution plugin to run the workflow.
            Common options:
              * ``"serial"`` or ``"debug"``: Run tasks sequentially (default).
              * ``"cf"``: Use concurrent futures for parallel execution.
              * ``"slurm"``: Submit tasks to a SLURM cluster.
        plugin_args (dict, optional):
            Extra keyword arguments passed to the chosen Pydra plugin.

    Examples:
              * For ``cf``: ``{"n_procs": 8}``
            See: https://nipype.github.io/pydra/

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
        >>> save_audios(outs, plugin="cf")
    """
    save_params = save_params or {}
    plugin_args = plugin_args or {}

    @python.define
    def _save_pair(pair: Tuple[Audio, Union[str, os.PathLike]], params: Dict[str, Any]) -> str:
        audio, out_path = pair
        audio.save_to_file(os.fspath(out_path), **params)
        return os.fspath(out_path)

    @workflow.define
    def _wf(pairs: Sequence[Tuple[Audio, Union[str, os.PathLike]]], params: Dict[str, Any]) -> List[str]:
        node = workflow.add(
            _save_pair(params=params).split(pair=pairs),
            name="map_save_audio",
        )
        return node.out  # list of saved file paths

    worker = "debug" if plugin in ("serial", "debug") else plugin
    # Execute and ignore returned paths to keep the original None-returning API
    _ = _wf(pairs=audio_tuples, params=save_params)(worker=worker, cache_root=cache_dir, **plugin_args)
