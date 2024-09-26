"""This module provides the Speechbrain interface for speech enhancement."""

import time
from typing import Dict, List, Optional, Tuple

import torch
from speechbrain.inference.separation import SepformerSeparation as separator

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, SpeechBrainModel, _select_device_and_dtype
from senselab.utils.data_structures.logging import logger


class SpeechBrainEnhancer:
    """A factory for managing SpeechBrain enhancement pipelines."""

    _models: Dict[str, separator] = {}

    @classmethod
    def _get_speechbrain_model(
        cls,
        model: SpeechBrainModel,
        device: Optional[DeviceType] = None,
    ) -> Tuple[separator, DeviceType, torch.dtype]:
        """Get or create a SpeechBrain enhancement model.

        Args:
            model (SpeechBrainModel): The SpeechBrain model.
            device (Optional[DeviceType]): The device to run the model on.
                Only CPU and CUDA are supported.

        Returns:
            separator: The SpeechBrain enhancement model.
            device: The device used for the model.
            dtype: The dtype used for the model.
        """
        device, dtype = _select_device_and_dtype(
            user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )
        key = f"{model.path_or_uri}-{model.revision}-{device.value}"
        if key not in cls._models:
            cls._models[key] = separator.from_hparams(source=model.path_or_uri, run_opts={"device": device.value})
        return cls._models[key], device, dtype

    @classmethod
    def enhance_audios_with_speechbrain(
        cls,
        audios: List[Audio],
        model: SpeechBrainModel = SpeechBrainModel(
            path_or_uri="speechbrain/sepformer-wham16k-enhancement", revision="main"
        ),
        device: Optional[DeviceType] = None,
        batch_size: int = 16,
    ) -> List[Audio]:
        """Enhances all audio samples using the given speechbrain model.

        Args:
            audios (List[Audio]): The list of audio objects to be enhanced.
            model (SpeechBrainModel): The SpeechBrain model used for enhancement.
            device (Optional[DeviceType]): The device to run the model on (default is None).
            batch_size (int): The size of batches to use when processing on a GPU.

        Returns:
            List[Audio]: The list of enhanced audio objects.
        """
        # Take the start time of the model initialization
        start_time_model = time.time()
        enhancer, device, _ = cls._get_speechbrain_model(model=model, device=device)
        end_time_model = time.time()
        elapsed_time_model = end_time_model - start_time_model
        logger.info(f"Time taken to initialize the speechbrain model: {elapsed_time_model:.2f} seconds")

        expected_sample_rate = enhancer.hparams.sample_rate

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

        # Take the start time of the enhancement
        start_time_enhancement = time.time()
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

            # Enhance waveforms in a batch
            enhanced_waveform = enhancer.separate_batch(audio.waveform)

            audio.waveform = enhanced_waveform.reshape(1, -1)

        end_time_enhancement = time.time()
        elapsed_time_enhancement = end_time_enhancement - start_time_enhancement
        logger.info(f"Time taken for enhancing the audios: {elapsed_time_enhancement:.2f} seconds")

        return audios
