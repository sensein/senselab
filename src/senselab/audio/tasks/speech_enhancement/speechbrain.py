"""This module provides the Speechbrain interface for speech enhancement."""

from typing import Dict, List, Optional

import torch
from speechbrain.inference.separation import SepformerSeparation as separator

from senselab.audio.data_structures.audio import Audio, batch_audios, unbatch_audios
from senselab.utils.data_structures.device import DeviceType, _select_device_and_dtype
from senselab.utils.data_structures.model import SpeechBrainModel


class SpeechBrainEnhancer:
    """A factory for managing SpeechBrain enhancement pipelines."""

    _models: Dict[str, separator] = {}

    @classmethod
    def _get_speechbrain_model(
        cls,
        model: SpeechBrainModel,
        device: Optional[DeviceType] = None,
    ) -> separator:
        """Get or create a SpeechBrain enhancement model.

        Args:
            model (SpeechBrainModel): The SpeechBrain model.
            device (Optional[DeviceType]): The device to run the model on.
                Only CPU and CUDA are supported.

        Returns:
            separator: The SpeechBrain enhancement model.
        """
        device, _ = _select_device_and_dtype(
            user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )
        key = f"{model.path_or_uri}-{model.revision}-{device.value}"
        if key not in cls._models:
            cls._models[key] = separator.from_hparams(source=model.path_or_uri, run_opts={"device": device.value})
        return cls._models[key]

    @classmethod
    def enhance_audios_with_speechbrain(
        cls,
        audios: List[Audio],
        model: SpeechBrainModel = SpeechBrainModel(
            path_or_uri="speechbrain/sepformer-wham16k-enhancement", revision="main"
        ),
        device: Optional[DeviceType] = None,
        batch_size: int = 8,
    ) -> List[Audio]:
        """Enhances all audio samples in the dataset.

        Args:
            audios (List[Audio]): The list of audio objects to be enhanced.
            model (SpeechBrainModel): The SpeechBrain model used for enhancement.
            device (Optional[DeviceType]): The device to run the model on (default is None).
            batch_size (int): The size of batches to use when processing on a GPU.

        Returns:
            List[Audio]: The list of enhanced audio objects.

        Todo:
            - Optimizing the computation by working in batches
            - Double-checking the input size of enhancer.encode_batch
        """
        enhancer = cls._get_speechbrain_model(model=model, device=device)
        expected_sample_rate = enhancer.hparams.sample_rate
        device, _ = _select_device_and_dtype(
            user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )

        if device == DeviceType.CPU:
            # Check that all audio objects have the correct sampling rate
            for audio in audios:
                if audio.waveform.shape[0] != 1:
                    raise ValueError(
                        f"Audio waveform must be mono (1 channel), but got {audio.waveform.shape[0]} channels"
                    )
                if audio.sampling_rate != expected_sample_rate:
                    raise ValueError(
                        "Audio sampling rate "
                        + str(audio.sampling_rate)
                        + " does not match expected "
                        + str(expected_sample_rate)
                    )

            # Stack audio waveforms for batch processing
            waveforms = torch.stack([audio.waveform.squeeze() for audio in audios])

            # Enhance waveforms in a batch
            enhanced_waveforms = enhancer.separate_batch(waveforms)

            # Update the original audio objects with the enhanced waveforms
            for audio, enhanced_waveform in zip(audios, enhanced_waveforms):
                audio.waveform = enhanced_waveform.reshape(1, -1)
        else:
            enhanced_audios = []
            for i in range(0, len(audios), batch_size):
                batch = audios[i : i + batch_size]
                batched_audios, sampling_rates, metadatas = batch_audios(batch)

                # Ensure batched_audios has the correct shape
                if batched_audios.dim() != 3 or batched_audios.size(1) != 1:
                    raise ValueError("Batched audio tensor must have shape [batch_size, 1, num_samples]")

                batched_audios = batched_audios.to(device=torch.device(str(device)), dtype=torch.float32)
                enhanced_audio = enhancer.separate_batch(batched_audios)

                enhanced_audio = enhanced_audio.detach().cpu()
                enhanced_batch = unbatch_audios(enhanced_audio, sampling_rates, metadatas)
                enhanced_audios.extend(enhanced_batch)

            audios = enhanced_audios  # Replace the original audios with the enhanced ones
        return audios
