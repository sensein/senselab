"""Dedicated Pyannote Voice Activity Detection pipeline.

Uses ``pyannote/voice-activity-detection`` (a dedicated segmentation model)
instead of repurposing a full diarization pipeline. This is lighter-weight
and produces cleaner speech/non-speech boundaries.

The dedicated VAD model outputs speech segments directly, without speaker
labels. Each segment is labeled ``"VOICE"``.
"""

import time
from typing import Dict, List, Optional, Union

import torch

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, PyannoteAudioModel, ScriptLine, _select_device_and_dtype
from senselab.utils.data_structures.logging import logger
from senselab.utils.data_structures.model import get_huggingface_token
from senselab.utils.dependencies import retry_on_transient_error

try:
    from pyannote.audio import Pipeline
    from pyannote.core import Annotation

    PYANNOTEAUDIO_AVAILABLE = True
except ModuleNotFoundError:
    PYANNOTEAUDIO_AVAILABLE = False


class PyannoteVAD:
    """Factory for creating and caching Pyannote VAD pipelines.

    Pipelines are cached per *(model.path_or_uri, revision, device)* so
    repeated calls with the same configuration reuse the initialized pipeline.

    Guidance:
        - The default model ``pyannote/voice-activity-detection`` expects
          **mono, 16 kHz** audio.
        - This is a dedicated segmentation model, not a diarization model.
          It produces speech vs. non-speech boundaries directly.
        - Supported devices: ``DeviceType.CPU`` and ``DeviceType.CUDA``.
    """

    _pipelines: Dict[str, "Pipeline"] = {}

    @classmethod
    def _get_pyannote_vad_pipeline(
        cls,
        model: PyannoteAudioModel,
        device: Union[DeviceType, None],
    ) -> "Pipeline":
        """Get or create a Pyannote VAD pipeline.

        Args:
            model: The Pyannote VAD model.
            device: The device to run the model on.

        Returns:
            The VAD pipeline.
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
            pipeline = retry_on_transient_error(
                Pipeline.from_pretrained,
                checkpoint=f"{model.path_or_uri}",
                revision=f"{model.revision}",
                token=get_huggingface_token(),
            )
            if not pipeline:
                raise ValueError(f"Pyannote VAD model {model.path_or_uri} not found.")
            pipeline = pipeline.to(torch.device(device.value))
            cls._pipelines[key] = pipeline
        return cls._pipelines[key]

    @classmethod
    def detect_voice_activity(
        cls,
        audios: List[Audio],
        model: Optional[PyannoteAudioModel] = None,
        device: Optional[DeviceType] = None,
    ) -> List[List[ScriptLine]]:
        """Detect voice activity using a dedicated Pyannote VAD pipeline.

        Requirements:
            - Input must be **mono** (``[1, T]``); stereo/multi-channel is rejected.
            - Sampling rate must be **16 kHz** (per model card).

        Args:
            audios: Audio clips to analyze (mono, 16 kHz).
            model: Pyannote VAD model. Defaults to
                ``pyannote/voice-activity-detection@main``.
            device: Inference device (``CPU`` or ``CUDA``).

        Returns:
            One list per input audio; each inner list contains ``ScriptLine``
            entries with ``(start, end)`` and ``speaker="VOICE"``.

        Raises:
            ModuleNotFoundError: If ``pyannote-audio`` is not installed.
            ValueError: If audio is not mono or sampling rate is not 16 kHz.

        Example:
            >>> from pathlib import Path
            >>> from senselab.audio.data_structures import Audio
            >>> from senselab.utils.data_structures import DeviceType, PyannoteAudioModel
            >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
            >>> mdl = PyannoteAudioModel(
            ...     path_or_uri="pyannote/voice-activity-detection", revision="main"
            ... )
            >>> vad = PyannoteVAD.detect_voice_activity([a1], model=mdl, device=DeviceType.CPU)
            >>> all(chunk.speaker == "VOICE" for chunk in vad[0])
            True
        """
        if not PYANNOTEAUDIO_AVAILABLE:
            raise ModuleNotFoundError(
                "`pyannote-audio` is not installed. "
                "Please install senselab audio dependencies using `pip install senselab`."
            )

        if model is None:
            model = PyannoteAudioModel(path_or_uri="pyannote/voice-activity-detection", revision="main")

        expected_sample_rate = 16000

        for audio in audios:
            if audio.waveform.shape[0] != 1:
                raise ValueError(f"Audio waveform must be mono (1 channel), but got {audio.waveform.shape[0]} channels")
            if audio.sampling_rate != expected_sample_rate:
                raise ValueError(
                    f"Audio sampling rate {audio.sampling_rate} does not match expected {expected_sample_rate}"
                )

        start_time_model = time.time()
        pipeline = cls._get_pyannote_vad_pipeline(model=model, device=device)
        end_time_model = time.time()
        logger.info(f"Time taken to initialize the pyannote VAD model: {end_time_model - start_time_model:.2f} seconds")

        start_time_vad = time.time()
        results: List[List[ScriptLine]] = []
        for audio in audios:
            output = pipeline({"waveform": audio.waveform, "sample_rate": audio.sampling_rate})

            # The VAD pipeline returns a pyannote.core.Annotation
            # Iterate over speech segments
            segments: List[ScriptLine] = []
            for segment, _, label in output.itertracks(yield_label=True):
                segments.append(
                    ScriptLine(
                        speaker="VOICE",
                        start=segment.start,
                        end=segment.end,
                    )
                )
            results.append(sorted(segments, key=lambda x: x.start or 0.0))

        end_time_vad = time.time()
        logger.info(f"Time taken to perform VAD: {end_time_vad - start_time_vad:.2f} seconds")

        return results
