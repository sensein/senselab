"""This module implements the Pyannote Diarization task."""

import time
from typing import Dict, List, Optional, Union

import torch

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, PyannoteAudioModel, ScriptLine, _select_device_and_dtype
from senselab.utils.data_structures.logging import logger

try:
    from pyannote.audio import Pipeline
    from pyannote.core import Annotation

    PYANNOTEAUDIO_AVAILABLE = True
except ModuleNotFoundError:
    PYANNOTEAUDIO_AVAILABLE = False


class PyannoteDiarization:
    """Factory for creating and caching **Pyannote** diarization pipelines.

    Pipelines are cached per *(model.path_or_uri, revision, device)*, so repeated calls
    with the same configuration reuse the initialized pipeline.

    Guidance:
        - Pyannote models typically expect **mono, 16 kHz** audio.
        - If you know the number of speakers, set `num_speakers`; otherwise use
          `min_speakers`/`max_speakers` bounds to help estimation.
        - Supported devices: ``DeviceType.CPU`` and ``DeviceType.CUDA``.
    """

    _pipelines: Dict[str, "Pipeline"] = {}

    @classmethod
    def _get_pyannote_diarization_pipeline(
        cls,
        model: PyannoteAudioModel,
        device: Union[DeviceType, None],
    ) -> "Pipeline":
        """Get or create a Pyannote Diarization pipeline.

        Args:
            model (PyannoteAudioModel): The Pyannote model.
            device (DeviceType): The device to run the model on.

        Returns:
            Pipeline: The diarization pipeline.
        """
        if not PYANNOTEAUDIO_AVAILABLE:
            raise ModuleNotFoundError(
                "`pyannote-audio` is not installed. "
                "Please install senselab audio dependencies using `pip install senselab`."
            )

        device, _ = _select_device_and_dtype(
            user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )
        key = f"{model.path_or_uri}-{model.revision}-{device}"
        if key not in cls._pipelines:
            pipeline = Pipeline.from_pretrained(checkpoint=f"{model.path_or_uri}", revision=f"{model.revision}")
            if not pipeline:
                raise ValueError(f"Pyannote model {model.path_or_uri} not found.")
            pipeline = pipeline.to(torch.device(device.value))
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
    """Diarize audios with **Pyannote**; returns per-speaker segments per audio.

    Requirements:
        - Input must be **mono** (`[1, T]`); stereo/multi-channel is rejected.
        - Sampling rate must be **16 kHz** (per model card for `3.1`).

    Args:
        audios (list[Audio]):
            Audio clips to diarize (mono, 16 kHz).
        model (PyannoteAudioModel | None):
            Pyannote model. Defaults to ``pyannote/speaker-diarization-community-1@main``.
        device (DeviceType | None):
            Inference device (``CPU`` or ``CUDA``).
        num_speakers (int | None):
            If known, fix the number of speakers.
        min_speakers (int | None):
            Minimum speakers when estimating (ignored if `num_speakers` is set).
        max_speakers (int | None):
            Maximum speakers when estimating (ignored if `num_speakers` is set).

    Returns:
        list[list[ScriptLine]]: One list per input audio with `(speaker, start, end)`.

    Raises:
        ModuleNotFoundError:
            If `pyannote-audio` is not installed.
        ValueError:
            If audio is not mono or sampling rate ≠ 16 kHz.

    Example (estimate speakers within bounds):
        >>> from pathlib import Path
        >>> from senselab.audio.data_structures import Audio
        >>> from senselab.utils.data_structures import DeviceType, PyannoteAudioModel
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> mdl = PyannoteAudioModel(path_or_uri="pyannote/speaker-diarization-community-1", revision="main")
        >>> diar = diarize_audios_with_pyannote(
        ...     [a1],
        ...     model=mdl,
        ...     device=DeviceType.CPU,
        ...     min_speakers=1,
        ...     max_speakers=3,
        ... )
        >>> len(diar[0]) >= 0
        True

    Example (known number of speakers):
        >>> from pathlib import Path
        >>> from senselab.audio.data_structures import Audio
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> diar = diarize_audios_with_pyannote([a1], num_speakers=2)
        >>> len(diar[0]) >= 0
        True
    """

    def _annotation_to_script_lines(annotation: "Annotation") -> List[ScriptLine]:
        """Convert a Pyannote annotation to a list of script lines.

        Args:
            annotation (Annotation): The Pyannote annotation object.

        Returns:
            List[ScriptLine]: A list of script lines.
        """
        diarization_list: List[ScriptLine] = []
        for segment, label in annotation:
            diarization_list.append(ScriptLine(speaker=label, start=segment.start, end=segment.end))
        return diarization_list

    if not PYANNOTEAUDIO_AVAILABLE:
        raise ModuleNotFoundError(
            "`pyannote-audio` is not installed. "
            "Please install senselab audio dependencies using `pip install senselab`."
        )

    if model is None:
        model = PyannoteAudioModel(path_or_uri="pyannote/speaker-diarization-community-1", revision="main")

    # 16khz comes from the model cards of pyannote/speaker-diarization-community-1
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
        results.append(_annotation_to_script_lines(diarization.exclusive_speaker_diarization))
    end_time_diarization = time.time()
    elapsed_time_diarization = end_time_diarization - start_time_diarization
    logger.info(f"Time taken to perform diarization: {elapsed_time_diarization:.2f} seconds")

    return results
