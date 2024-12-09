"""This module implements the Pyannote Diarization task."""

import time
from typing import Dict, List, Optional, Union

import torch
from pyannote.audio import Pipeline
from pyannote.core import Annotation

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, PyannoteAudioModel, ScriptLine, _select_device_and_dtype
from senselab.utils.data_structures.logging import logger


class PyannoteDiarization:
    """A factory for managing Pyannote Diarization pipelines."""

    _pipelines: Dict[str, Pipeline] = {}

    @classmethod
    def _get_pyannote_diarization_pipeline(
        cls,
        model: PyannoteAudioModel,
        device: Union[DeviceType, None],
    ) -> Pipeline:
        """Get or create a Pyannote Diarization pipeline.

        Args:
            model (PyannoteAudioModel): The Pyannote model.
            device (DeviceType): The device to run the model on.

        Returns:
            Pipeline: The diarization pipeline.
        """
        device, _ = _select_device_and_dtype(
            user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )
        key = f"{model.path_or_uri}-{model.revision}-{device}"
        if key not in cls._pipelines:
            pipeline = Pipeline.from_pretrained(checkpoint_path=f"{model.path_or_uri}@{model.revision}").to(
                torch.device(device.value)
            )
            cls._pipelines[key] = pipeline
        return cls._pipelines[key]


def diarize_audios_with_pyannote(
    audios: List[Audio],
    model: Optional[PyannoteAudioModel] = None,
    device: Optional[DeviceType] = None,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> List[List[ScriptLine]]:
    """Diarizes a list of audio files using the Pyannote speaker diarization model.

    Args:
        audios (List[Audio]): A list of audio files.
        model (PyannoteAudioModel): The model to use for diarization.
            If None, the default model "pyannote/speaker-diarization-3.1" is used.
        device (Optional[DeviceType]): The device to use for diarization.
        num_speakers (Optional[int]): Number of speakers, when known.
        min_speakers (Optional[int]): Minimum number of speakers. Has no effect when `num_speakers` is provided.
        max_speakers (Optional[int]): Maximum number of speakers. Has no effect when `num_speakers` is provided.

    Returns:
        List[ScriptLine]: A list of ScriptLine objects containing the diarization results.
    """
    if model is None:
        model = PyannoteAudioModel(path_or_uri="pyannote/speaker-diarization-3.1", revision="main")

    def _annotation_to_script_lines(annotation: Annotation) -> List[ScriptLine]:
        """Convert a Pyannote annotation to a list of script lines.

        Args:
            annotation (Annotation): The Pyannote annotation object.

        Returns:
            List[ScriptLine]: A list of script lines.
        """
        diarization_list: List[ScriptLine] = []
        for segment, _, label in annotation.itertracks(yield_label=True):
            diarization_list.append(ScriptLine(speaker=label, start=segment.start, end=segment.end))
        return diarization_list

    # 16khz comes from the model cards of pyannote/speaker-diarization-3.1
    expected_sample_rate = 16000

    # Check that all audio objects have the correct sampling rate
    for audio in audios:
        if audio.waveform.shape[0] != 1:
            raise ValueError(f"Audio waveform must be mono (1 channel), but got {audio.waveform.shape[0]} channels")
        if audio.sampling_rate != expected_sample_rate:
            raise ValueError(
                "Audio sampling rate "
                + str(audio.sampling_rate)
                + " does not match expected "
                + str(expected_sample_rate)
            )

    # Take the start time of the model initialization
    start_time_model = time.time()
    pipeline = PyannoteDiarization._get_pyannote_diarization_pipeline(model=model, device=device)
    end_time_model = time.time()
    elapsed_time_model = end_time_model - start_time_model
    logger.info(f"Time taken to initialize the pyannote model: {elapsed_time_model:.2f} seconds")

    # Perform diarization
    start_time_diarization = time.time()
    results: List[List[ScriptLine]] = []
    for audio in audios:
        diarization = pipeline(
            {"waveform": audio.waveform, "sample_rate": audio.sampling_rate},
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        results.append(_annotation_to_script_lines(diarization))
    end_time_diarization = time.time()
    elapsed_time_diarization = end_time_diarization - start_time_diarization
    logger.info(f"Time taken to perform diarization: {elapsed_time_diarization:.2f} seconds")

    return results
