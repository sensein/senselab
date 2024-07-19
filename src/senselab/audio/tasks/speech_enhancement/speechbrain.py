"""This module provides the Speechbrain interface for speech enhancement."""

from typing import Dict, List, Optional

from speechbrain.inference.separation import SepformerSeparation as separator

from senselab.audio.data_structures.audio import Audio
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
    ) -> List[Audio]:
        """Enhances all audio samples in the dataset.

        Args:
            audios (List[Audio]): The list of audio objects to be enhanced.
            model (SpeechBrainModel): The SpeechBrain model used for enhancement.
            device (Optional[DeviceType]): The device to run the model on (default is None).

        Returns:
            List[Audio]: The list of enhanced audio objects.

        Todo:
            - Optimizing the computation by working in batches
            - Double-checking the input size of enhancer.encode_batch
        """
        enhancer = cls._get_speechbrain_model(model=model, device=device)
        expected_sample_rate = enhancer.hparams.sample_rate

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

        return audios
