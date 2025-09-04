"""This module implements the NVIDIA Sortformer Diarization task."""

import os
import tempfile
import time
from typing import Dict, List, Optional, Union

import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.features_extraction.praat_parselmouth import extract_audio_duration
from senselab.utils.data_structures import DeviceType, ScriptLine, _select_device_and_dtype
from senselab.utils.data_structures.logging import logger

try:
    from nemo.collections.asr.models import SortformerEncLabelModel

    NEMO_SORTFORMER_AVAILABLE = True
except ModuleNotFoundError:
    NEMO_SORTFORMER_AVAILABLE = False


class NvidiaSortformerDiarization:
    """Factory for creating and caching **NVIDIA Sortformer** diarization models.

    Models are cached per *(model_name, device)* pair, so reusing the same configuration
    avoids repeated downloads/initialization.

    Notes:
        - Sortformer supports up to **4 speakers** (model dependent).
        - The model’s **expected sampling rate** is taken from `model.cfg.sample_rate`.
        - Supported devices here: ``DeviceType.CPU`` and ``DeviceType.CUDA``.
    """

    _models: Dict[str, "SortformerEncLabelModel"] = {}

    @classmethod
    def _get_sortformer_model(
        cls,
        model_name: str = "nvidia/diar_sortformer_4spk-v1",
        device: Optional[DeviceType] = None,
    ) -> "SortformerEncLabelModel":
        """Get or create a Sortformer diarization model.

        Args:
            model_name (str): The Hugging Face model card name.
            device (DeviceType): The device to run the model on.

        Returns:
            SortformerEncLabelModel: The diarization model.
        """
        if not NEMO_SORTFORMER_AVAILABLE:
            raise ModuleNotFoundError(
                "`nemo_toolkit` is not installed. "
                "Please install senselab audio dependencies using `pip install 'senselab[audio]'`."
            )

        device, _ = _select_device_and_dtype(
            user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )
        key = f"{model_name}-{device}"
        if key not in cls._models:
            model = SortformerEncLabelModel.from_pretrained(model_name, map_location=device.value)
            model.eval()
            cls._models[key] = model
        return cls._models[key]


def diarize_audios_with_nvidia_sortformer(
    audios: List[Audio],
    model_name: str = "nvidia/diar_sortformer_4spk-v1",
    device: Optional[DeviceType] = None,
) -> List[List[ScriptLine]]:
    """Diarize audios with **NVIDIA Sortformer** (NeMo), returning per-speaker segments.

    The current model supports a maximum of **4 speakers**. Inputs must be **mono** and
    match the model’s expected sampling rate (`model.cfg.sample_rate`). Internally,
    each clip is serialized to a temporary WAV file for NeMo inference and then cleaned up.

    Args:
        audios (list[Audio]):
            Audio clips to diarize (mono, correct sampling rate).
        model_name (str, optional):
            HF model card name (e.g., ``"nvidia/diar_sortformer_4spk-v1"``).
        device (DeviceType | None):
            Inference device (``CPU`` or ``CUDA``).

    Returns:
        list[list[ScriptLine]]: One list per input audio with `(speaker, start, end)`, sorted by start time.

    Raises:
        ModuleNotFoundError:
            If `nemo_toolkit` is not installed.
        ValueError:
            If input is not mono or sampling rate mismatches the model.

    Example (CPU):
        >>> from pathlib import Path
        >>> from senselab.audio.data_structures import Audio
        >>> from senselab.utils.data_structures import DeviceType
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> diar = diarize_audios_with_nvidia_sortformer([a1], device=DeviceType.CPU)
        >>> len(diar[0]) >= 0
        True

    Example (CUDA, explicit model):
        >>> from pathlib import Path
        >>> from senselab.audio.data_structures import Audio
        >>> from senselab.utils.data_structures import DeviceType
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> diar = diarize_audios_with_nvidia_sortformer(
        ...     [a1],
        ...     model_name="nvidia/diar_sortformer_4spk-v1",
        ...     device=DeviceType.CUDA,
        ... )
        >>> isinstance(diar[0], list)
        True
    """
    if not NEMO_SORTFORMER_AVAILABLE:
        raise ModuleNotFoundError(
            "`nemo_toolkit` is not installed. "
            "Please install senselab audio dependencies using `pip install 'senselab[audio]'`."
        )

    # Initialize model
    start_time_model = time.time()
    model = NvidiaSortformerDiarization._get_sortformer_model(model_name=model_name, device=device)
    end_time_model = time.time()
    elapsed_time_model = end_time_model - start_time_model
    logger.info(f"Time taken to initialize the NVIDIA Sortformer model: {elapsed_time_model:.2f} seconds")

    # The model expected sample rate
    expected_sample_rate = model.cfg.sample_rate

    # Check that all audio objects have the correct sampling rate and are mono
    for audio in audios:
        if audio.waveform.shape[0] != 1:
            raise ValueError(f"Audio waveform must be mono (1 channel), but got {audio.waveform.shape[0]} channels")
        if audio.sampling_rate != expected_sample_rate:
            raise ValueError(
                f"Audio sampling rate {audio.sampling_rate} does not match expected {expected_sample_rate}"
            )

    # Perform diarization
    start_time_diarization = time.time()
    results: List[List[ScriptLine]] = []

    for audio in audios:
        # Save the waveform to a temporary WAV file using Audio.save_to_file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            tmpfile_path = tmpfile.name
        try:
            audio.save_to_file(tmpfile_path)
            audio_duration = extract_audio_duration(audio)["duration"]
            diarization_segments = model.diarize(audio=tmpfile_path)[0]

            # diarization_segments: List[List[str]], e.g. [['0.080 4.950 speaker_0']]
            script_lines: List[ScriptLine] = []
            for seg in diarization_segments:
                # seg: List[str], e.g. '[0.080 4.950 speaker_0]'
                parts = seg.strip().split()
                if len(parts) == 3:
                    start, end, speaker = parts
                    script_lines.append(
                        ScriptLine(
                            speaker=speaker,
                            start=float(start),
                            end=float(end) if float(end) < audio_duration else audio_duration,
                        )
                    )
            results.append(sorted(script_lines, key=lambda x: x.start if x.start is not None else 0.0))
        finally:
            # Clean up the temporary file
            if os.path.exists(tmpfile_path):
                # print("Deleting temporary file:", tmpfile_path)
                os.remove(tmpfile_path)
    end_time_diarization = time.time()
    elapsed_time_diarization = end_time_diarization - start_time_diarization
    logger.info(f"Time taken to perform diarization: {elapsed_time_diarization:.2f} seconds")

    return results
