"""This module provides the implementation of Mars5-TTS-based text-to-speech pipelines."""

import time
from typing import Dict, List, Optional, Tuple

import torch

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, Language, TorchModel, _select_device_and_dtype
from senselab.utils.data_structures.logging import logger


class Mars5TTS:
    """A class for managing Torch-based Mars5TTS models."""

    _models: Dict[str, Tuple[torch.nn.Module, type]] = {}

    @classmethod
    def _get_torch_tts_model(
        cls,
        model: Optional[TorchModel] = None,
        language: Optional[Language] = Language(language_code="en"),
        device: Optional[DeviceType] = None,
    ) -> Tuple[torch.nn.Module, type]:
        """Get or create a Torch-based Mars5TTS model.

        Args:
            model (TorchModel): The Torch model (currently only supports "Camb-ai/mars5-tts").
            language (Optional[Language]): The language of the text (default is Language(language_code="en")).
                The only supported language is "en" for now.
            device (DeviceType): The device to run the model on (default is None). Supported devices are CPU and CUDA.

        Returns:
            model: The Torch-based Mars5TTS model.
            config_class: The configuration class used by the model.
        """
        if model is None:
            model = TorchModel(path_or_uri="Camb-ai/mars5-tts", revision="master")

        if model.path_or_uri != "Camb-ai/mars5-tts" or model.revision != "master":
            raise NotImplementedError("Only the 'Camb-ai/mars5-tts' model is supported for now.")
        if language == Language(language_code="en"):
            model_name: str = "mars5_english"  # This is the default model they have for English.
        else:
            raise NotImplementedError("Only English is supported for now.")
        device, _ = _select_device_and_dtype(
            user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )

        key = f"{model.path_or_uri}-{model.revision}-{language.name}-{device.value}"
        if key not in cls._models:
            my_model, config_class = torch.hub.load(
                f"{model.path_or_uri}:{model.revision}", model_name, trust_repo=True
            )
            cls._models[key] = (my_model.to(device.value), config_class)
        return cls._models[key]

    @classmethod
    def synthesize_texts_with_mars5tts(
        cls,
        texts: List[str],
        targets: List[Tuple[Audio, str]],
        model: Optional[TorchModel] = None,
        language: Optional[Language] = None,
        device: Optional[DeviceType] = None,
        deep_clone: bool = True,
        rep_penalty_window: int = 100,
        top_k: int = 100,
        temperature: float = 0.7,
        freq_penalty: float = 3,
    ) -> List[Audio]:
        """Synthesizes speech from all the provided text samples and target voices+transcripts using Mars5-TTS.

        Args:
            texts (List[str]): The list of text strings to be synthesized.
            targets (List[Tuple[Audio, str]]):
                The list of tuples containing audio objects and transcripts.
            model (TorchModel): The Torch model.
                If None, the default model "Camb-ai/mars5-tts" is used.
            language (Optional[Language]): The language of the text (default is None).
                The only supported language is "en" for now.
            device (DeviceType): The device to run the model on (default is None). Supported devices are CPU and CUDA.
            deep_clone (bool): Whether to deep clone the target audio (default is True).
            rep_penalty_window (int): The window size for repetition penalty (default is 100).
            top_k (int): The number of top-k candidates to consider (default is 100).
            temperature (float): The temperature for sampling (default is 0.7).
            freq_penalty (float): The frequency penalty for sampling (default is 3).

        Returns:
            List[Audio]: The list of synthesized audio objects.

        Some tips for best quality:
            - Make sure reference audio is clean and between 1 second and 12 seconds.
            - Use deep clone and provide an accurate transcript for the reference.
              Mars5-TTS can potentially accept audio target voices without transcripts. After some testing,
              we found that the best quality is achieved with deep cloning and providing an accurate transcript.
            - Use proper punctuation -- the model can be guided and made better or worse with proper use of punctuation
                and capitalization.

        The original repo of the model is: https://github.com/Camb-ai/MARS5-TTS.
        """
        if model is None:
            model = TorchModel(path_or_uri="Camb-ai/mars5-tts", revision="master")

        # Take the start time of the model initialization
        start_time_model = time.time()
        my_model, config_class = cls._get_torch_tts_model(model, language, device)
        cfg = config_class(
            deep_clone=deep_clone,
            rep_penalty_window=rep_penalty_window,
            top_k=top_k,
            temperature=temperature,
            freq_penalty=freq_penalty,
        )

        # Take the end time of the model initialization
        end_time_model = time.time()
        # Print the time taken for initialize the Mars5-TTS model
        elapsed_time_pipeline = end_time_model - start_time_model
        logger.info(f"Time taken to initialize the Mars5-TTS model: {elapsed_time_pipeline:.2f} seconds")

        # Check that the target audios are mono and have the correct sampling rate
        expected_sampling_rate = my_model.sr
        for idx, item in enumerate(targets):
            target_audio, _ = item
            if target_audio.waveform.shape[0] != 1:
                raise ValueError(
                    f"Stereo audio is not supported for audio at index {idx}. "
                    f"Got {target_audio.waveform.shape[0]} channels"
                )
            if target_audio.sampling_rate != expected_sampling_rate:
                raise ValueError(
                    f"Incorrect sampling rate for audio at index {idx}. "
                    f"Expected {expected_sampling_rate}, got {target_audio.sampling_rate}"
                )

        # Take the start time of text-to-speech synthesis
        start_time_tts = time.time()
        audios = []
        for i, text in enumerate(texts):
            item = targets[i]
            target_audio, target_transcript = item
            duration = target_audio.waveform.shape[1] / target_audio.sampling_rate
            if duration < 1 or duration > 12:
                logger.warning(
                    f"Warning: Reference audio at index {i} has a duration of {duration} seconds. "
                    "It is recommended to be between 1 second and 12 seconds."
                )

            _, wav_out = my_model.tts(text, target_audio.waveform, target_transcript, cfg=cfg)  # type: ignore
            audios.append(Audio(waveform=wav_out, sampling_rate=my_model.sr))

        # Take the end time of the text-to-speech synthesis
        end_time_tts = time.time()
        # Print the time taken for text-to-speech synthesis
        elapsed_time_tts = end_time_tts - start_time_tts
        logger.info(f"Time taken for synthesizing audios: {elapsed_time_tts:.2f} seconds")

        return audios
