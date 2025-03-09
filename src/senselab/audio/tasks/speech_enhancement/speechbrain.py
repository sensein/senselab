"""This module provides the Speechbrain interface for speech enhancement."""

import time
from typing import Dict, List, Optional, Tuple, Union

import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.preprocessing import concatenate_audios, evenly_segment_audios
from senselab.utils.data_structures import DeviceType, SpeechBrainModel, _select_device_and_dtype
from senselab.utils.data_structures.logging import logger

try:
    from speechbrain.inference.enhancement import SpectralMaskEnhancement as enhance_model
    from speechbrain.inference.separation import SepformerSeparation as separator

    SPEECHBRAIN_AVAILABLE = True
except ModuleNotFoundError:
    SPEECHBRAIN_AVAILABLE = False


class SpeechBrainEnhancer:
    """A factory for managing SpeechBrain enhancement pipelines."""

    MAX_DURATION_SECONDS = 60  # Maximum duration per segment in seconds
    MIN_LENGTH = 16  # kernel size for speechbrain/sepformer-wham16k-enhancement
    _models: Dict[str, Union["separator", "enhance_model"]] = {}

    @classmethod
    def _get_speechbrain_model(
        cls,
        model: SpeechBrainModel,
        device: Optional[DeviceType] = None,
    ) -> Tuple["separator", DeviceType, torch.dtype]:
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
        if not SPEECHBRAIN_AVAILABLE:
            raise ModuleNotFoundError(
                "`speechbrain` is not installed. "
                "Please install senselab audio dependencies using `pip install senselab['audio']`."
            )

        device, dtype = _select_device_and_dtype(
            user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )
        key = f"{model.path_or_uri}-{model.revision}-{device.value}"
        if key not in cls._models:
            try:
                cls._models[key] = enhance_model.from_hparams(
                    source=model.path_or_uri, run_opts={"device": device.value}
                )
            except Exception as e:
                print("Failed to load SpeechBrain model as a SpectralMaskEnhancement model:", e)
                print("Trying to load as a SepformerSeparation model...")
                cls._models[key] = separator.from_hparams(source=model.path_or_uri, run_opts={"device": device.value})

        return cls._models[key], device, dtype

    @classmethod
    def enhance_audios_with_speechbrain(
        cls, audios: List[Audio], model: Optional[SpeechBrainModel] = None, device: Optional[DeviceType] = None
    ) -> List[Audio]:
        """Enhances all audio samples using the given speechbrain model.

        Audio clips longer than MAX_DURATION_SECONDS (= 60s) will be split into segments,
        and each segment will be enhanced separately and then concatenated.
        This is because the speechbrain model is not able to handle long clips.

        Args:
            audios (List[Audio]): The list of audio objects to be enhanced.
            model (SpeechBrainModel): The SpeechBrain model used for enhancement.
                If None, the default model "speechbrain/sepformer-wham16k-enhancement" is used.
            device (Optional[DeviceType]): The device to run the model on (default is None).
            batch_size (int): The size of batches to use when processing on a GPU.

        Returns:
            List[Audio]: The list of enhanced audio objects.
        """
        if not SPEECHBRAIN_AVAILABLE:
            raise ModuleNotFoundError(
                "`speechbrain` is not installed. "
                "Please install senselab audio dependencies using `pip install senselab['audio']`."
            )

        if model is None:
            model = SpeechBrainModel(path_or_uri="speechbrain/sepformer-wham16k-enhancement", revision="main")

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
        enhanced_audios = []

        for audio in audios:
            segments = evenly_segment_audios([audio], cls.MAX_DURATION_SECONDS, pad_last_segment=False)[0]
            enhanced_segments = []

            for segment in segments:
                if segment.waveform.shape[-1] < cls.MIN_LENGTH:
                    print(f"Skipping segment with length {segment.waveform.shape[-1]}")
                    # Append it as it is
                    enhanced_segments.append(segment)
                else:
                    if isinstance(enhancer, enhance_model):
                        enhanced_waveform = enhancer.enhance_batch(segment.waveform, lengths=torch.tensor([1.0]))
                    else:
                        enhanced_waveform = enhancer.separate_batch(segment.waveform)

                    enhanced_segments.append(
                        Audio(waveform=enhanced_waveform.reshape(1, -1), sampling_rate=segment.sampling_rate)
                    )

            # TODO: decide what to do with metadata
            enhanced_audio = concatenate_audios(enhanced_segments)
            enhanced_audio.metadata = audio.metadata
            enhanced_audios.append(enhanced_audio)

        end_time_enhancement = time.time()
        logger.info(f"Time taken for enhancing the audios: {end_time_enhancement - start_time_enhancement:.2f} seconds")

        return enhanced_audios
