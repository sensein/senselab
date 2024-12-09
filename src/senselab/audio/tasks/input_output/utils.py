"""This module provides utilities for reading and saving (multiple) audio files using Pydra."""

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pydra

from senselab.audio.data_structures import Audio


def read_audios(
    file_paths: List[str | os.PathLike],
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    plugin: str = "serial",
    plugin_args: Dict[str, Any] = {},
) -> List[Audio]:
    """Read and process a list of audio files using Pydra workflow.

    Args:
        file_paths (List[str]): List of paths to audio files.
        cache_dir (str, optional): Directory for caching intermediate results. Defaults to None.
        plugin (str, optional): Pydra plugin to use for workflow submission. Defaults to "serial".
        plugin_args (dict, optional): Additional arguments for the Pydra plugin. Defaults to {}.

    Returns:
        List[Audio]: A list of Audio objects containing the waveform and sample rate for each processed file.
    """

    @pydra.mark.task
    def load_audio_file(file_path: str | os.PathLike) -> Any:  # noqa: ANN401
        """Load an audio file and return an Audio object.

        Args:
            file_path (str): Path to the audio file.

        Returns:
            Audio: An instance of the Audio class containing the waveform and sample rate.
        """
        return Audio.from_filepath(file_path)

    # Create the workflow
    wf = pydra.Workflow(name="read_audio_files_workflow", input_spec=["x"], cache_dir=cache_dir)
    wf.split("x", x=file_paths)
    wf.add(load_audio_file(name="load_audio_file", file_path=wf.lzin.x))
    wf.set_output([("processed_files", wf.load_audio_file.lzout.out)])

    # Run the workflow
    with pydra.Submitter(plugin=plugin, **plugin_args) as sub:
        sub(wf)

    # Collect and return the results
    outputs = wf.result()
    return [output.output.processed_files for output in outputs]


def save_audios(
    audio_tuples: Sequence[Tuple[Audio, Union[str, os.PathLike]]],
    save_params: Dict[str, Any] = {},
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    plugin: str = "serial",
    plugin_args: Dict[str, Any] = {},
) -> None:
    """Save a list of Audio objects to specified files using Pydra workflow.

    Args:
        audio_tuples (Sequence[Tuple[Audio, Union[str, os.PathLike]]): Sequence of tuples where each tuple contains
            an Audio object, its output path (str or os.PathLike).
        save_params (dict, optional): Additional parameters for saving audio files.
            Defaults to {}
        cache_dir (str, optional): Directory for caching intermediate results.
            Defaults to None.
        plugin (str, optional): Pydra plugin to use for workflow submission.
            Defaults to "serial".
        plugin_args (dict, optional): Additional arguments for the Pydra plugin.
            Defaults to {}.

    Raises:
        RuntimeError: If any output directory in the provided paths does not exist or is not writable.
    """

    @pydra.mark.task
    def _extract_audio(audio_tuple: Tuple[Audio, Union[str, os.PathLike]]) -> Any:  # noqa: ANN401
        """Extract the Audio object from the tuple."""
        return audio_tuple[0]

    @pydra.mark.task
    def _extract_output_path(audio_tuple: Tuple[Audio, Union[str, os.PathLike]]) -> Union[str, os.PathLike]:
        """Extract the output path from the tuple."""
        return audio_tuple[1]

    @pydra.mark.task
    def _save_audio(audio: Audio, file_path: str, save_params: Dict[str, Any]) -> None:
        """Save an Audio object to a file.

        Args:
            audio (Audio): The Audio object to save.
            file_path (str): Path to save the audio file.
            save_params (dict): Additional parameters for saving audio files.
        """
        audio.save_to_file(file_path, **save_params)

    # Create the workflow
    wf = pydra.Workflow(name="save_audio_files_workflow", input_spec=["audio_tuples"], cache_dir=cache_dir)
    wf.split("audio_tuples", audio_tuples=audio_tuples)
    wf.add(_extract_audio(name="_extract_audio", audio_tuple=wf.lzin.audio_tuples))
    wf.add(_extract_output_path(name="_extract_output_path", audio_tuple=wf.lzin.audio_tuples))
    wf.add(
        _save_audio(
            name="_save_audio",
            audio=wf._extract_audio.lzout.out,
            file_path=wf._extract_output_path.lzout.out,
            save_params=save_params,
        )
    )
    wf.set_output([])

    # Run the workflow
    with pydra.Submitter(plugin=plugin, **plugin_args) as sub:
        sub(wf)
