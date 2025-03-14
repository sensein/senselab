"""This module implements some utilities for the text-to-speech task."""

from typing import Any, Dict, List, Optional

from transformers import pipeline

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, HFModel, _select_device_and_dtype


class HuggingFaceTTS:
    """A factory for managing Hugging Face TTS pipelines."""

    _pipelines: Dict[str, pipeline] = {}

    @classmethod
    def _get_hf_tts_pipeline(
        cls,
        model: HFModel,
        device: Optional[DeviceType] = None,
    ) -> pipeline:
        """Get or create a Hugging Face TTS pipeline.

        Args:
            model (HFModel): The Hugging Face model.
            device (Optional[DeviceType]): The device to run the model on.

        Returns:
            pipeline: The TTS pipeline.
        """
        device, _ = _select_device_and_dtype(
            user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )
        key = f"{model.path_or_uri}-{model.revision}-{device.value}"
        if key not in cls._pipelines:
            cls._pipelines[key] = pipeline(
                "text-to-speech",
                model=model.path_or_uri,
                revision=model.revision,
                device=device.value,
            )
        return cls._pipelines[key]

    @classmethod
    def synthesize_texts_with_transformers(
        cls,
        texts: List[str],
        model: Optional[HFModel] = None,
        device: Optional[DeviceType] = None,
        forward_params: Optional[Dict[str, Any]] = None,
    ) -> List[Audio]:
        """Synthesizes speech from all the provided text samples.

        Several text-to-speech models are currently available in Transformers,
        such as Bark, MMS, VITS and SpeechT5.

        Args:
            texts (List[str]): The list of text strings to be synthesized.
            model (HFModel): The Hugging Face model used for synthesis.
                If None, the default model "suno/bark" is used.
            device (Optional[DeviceType]): The device to run the model on (default is None).
            forward_params (Optional[Dict[str, Any]]): Additional parameters to pass to the forward function.

        Returns:
            List[Audio]: The list of synthesized audio objects.

        Todo:
            - Add speaker embeddings as they do in here:
            https://huggingface.co/docs/transformers/tasks/text-to-speech
        """
        if model is None:
            model = HFModel(path_or_uri="suno/bark", revision="main")
        pipe = HuggingFaceTTS._get_hf_tts_pipeline(model=model, device=device)

        synthesized_audios = pipe(texts, forward_params=forward_params)

        audios = []
        for synth in synthesized_audios:
            audios.append(Audio(waveform=synth["audio"], sampling_rate=synth["sampling_rate"]))
        return audios
