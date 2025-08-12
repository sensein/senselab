"""This module provides utilities for reading and saving (multiple) audio files using Pydra."""

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
    """Read and wrap a list of audio files as `Audio` objects via a Pydra workflow.

    Args:
        file_paths: A list of audio file paths.
        cache_dir: The directory to use for caching the workflow. Default is None.
        plugin: The Pydra plugin to use for workflow submission. Default is "debug".
        plugin_args: Additional arguments to pass to the plugin. Default is None.

    Returns:
        A list of `Audio` objects.
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
    """Save a sequence of `(Audio, output_path)` pairs via a Pydra workflow.

    Args:
        audio_tuples: A sequence of `(Audio, output_path)` pairs.
        save_params: A dictionary of parameters to pass to `Audio.save_to_file`.
        cache_dir: The directory to use for caching the workflow. Default is None.
        plugin: The Pydra plugin to use for workflow submission. Default is "serial".
        plugin_args: Additional arguments to pass to the plugin. Default is None.

    Returns:
        None
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
