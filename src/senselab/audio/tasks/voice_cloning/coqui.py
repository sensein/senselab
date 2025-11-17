"""This module contains functions for voice cloning using Coqui TTS."""

from typing import Any, Dict, List, Optional, Union

try:
    from TTS.api import TTS

    TTS_AVAILABLE = True
except ModuleNotFoundError:
    TTS_AVAILABLE = False

from pathlib import Path

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import CoquiTTSModel, DeviceType, _select_device_and_dtype


class CoquiVoiceCloner:
    """A factory for managing voice cloning pipelines using Coqui TTS."""

    _models: Dict[str, "TTS"] = {}

    @classmethod
    def _get_tts_model(cls, model_id: Union[str, Path], user_preference: Optional[DeviceType] = None) -> "TTS":
        """Get or create a TTS voice conversion pipeline."""
        if not TTS_AVAILABLE:
            raise ModuleNotFoundError(
                "`coqui-tts` is not available. "
                "Please install senselab audio dependencies using `pip install senselab`."
            )
        device, _ = _select_device_and_dtype(
            user_preference=user_preference, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )
        key = f"{model_id}-{device.value}"
        if key not in cls._models:
            tts = TTS(model_id).to(device=device.value)
            cls._models[key] = tts
        return cls._models[key]

    @classmethod
    def clone_voices(
        cls,
        source_audios: List[Audio],
        target_audios: List[Audio],
        model: Optional[CoquiTTSModel] = None,
        device: Optional[DeviceType] = None,
    ) -> List[Audio]:
        """Clone voices from source audios to target audios using Coqui TTS.

        Args:
            source_audios (List[Audio]): List of source audio objects.
            target_audios (List[Audio]): List of target audio objects.
            model (CoquiTTSModel, optional): The Coqui TTS model to use for voice conversion.
                All Coqui TTS models are supported, including
                "voice_conversion_models/multilingual/multi-dataset/knnvc",
                "voice_conversion_models/multilingual/vctk/freevc24",
                "voice_conversion_models/multilingual/multi-dataset/openvoice_v1",
                "voice_conversion_models/multilingual/multi-dataset/openvoice_v2".
                If None, the default model "voice_conversion_models/multilingual/multi-dataset/knnvc" is used.
            device (Optional[DeviceType], optional): The device to run the model on.
                Defaults to None.
        """
        # Validate model
        if model is None:
            model = CoquiTTSModel(path_or_uri="voice_conversion_models/multilingual/multi-dataset/knnvc")
        if model._scope != "voice_conversion_models":
            all_models = TTS.list_models()
            voice_conversion_models = [m for m in all_models if m.startswith("voice_conversion_models")]
            raise ValueError(
                f"Model {model.path_or_uri} is not a voice conversion model. "
                f"Available voice conversion models: {voice_conversion_models}"
            )

        # Get TTS pipeline
        tts = cls._get_tts_model(model_id=model.path_or_uri, user_preference=device)

        # Validate sampling rates
        audio_config = tts.voice_converter.vc_config.audio

        expected_sample_rate = (
            audio_config.input_sample_rate
            if hasattr(audio_config, "input_sample_rate")
            else getattr(audio_config, "sample_rate", None)
        )

        output_sample_rate = (
            audio_config.output_sample_rate
            if hasattr(audio_config, "output_sample_rate")
            else getattr(audio_config, "sample_rate", None)
        )

        if len(source_audios) != len(target_audios):
            raise ValueError("Number of source and target audios must be the same.")

        for i, (source_audio, target_audio) in enumerate(zip(source_audios, target_audios)):
            source_path = (
                source_audio.filepath() if hasattr(source_audio, "filepath") and source_audio.filepath() else ""
            )
            target_path = (
                target_audio.filepath() if hasattr(target_audio, "filepath") and target_audio.filepath() else ""
            )

            source_desc = f"{source_audio.generate_id()}{f' ({source_path})' if source_path else ''}"
            target_desc = f"{target_audio.generate_id()}{f' ({target_path})' if target_path else ''}"

            if source_audio.waveform.squeeze().dim() != 1 or target_audio.waveform.squeeze().dim() != 1:
                raise ValueError(
                    f"[Pair index {i}] Only mono audio files are supported. "
                    f"Source: {source_desc}, Target: {target_desc}."
                )

            if source_audio.sampling_rate != expected_sample_rate or target_audio.sampling_rate != expected_sample_rate:
                raise ValueError(
                    f"[Pair index {i}] Expected sample rate {expected_sample_rate}, but got "
                    f"{source_audio.sampling_rate} (source) and {target_audio.sampling_rate} (target). "
                    f"Source: {source_desc}, Target: {target_desc}."
                )

            # Ensure mono audio
            source_waveform = source_audio.waveform.squeeze()
            target_waveform = target_audio.waveform.squeeze()
            if source_waveform.dim() > 1 or target_waveform.dim() > 1:
                raise ValueError("Both source and target audios must be mono.")

        cloned_audios = []
        for source_audio, target_audio in zip(source_audios, target_audios):
            # Perform voice conversion
            converted_waveform = tts.voice_conversion(source_wav=source_waveform, target_wav=target_waveform)

            cloned_audios.append(Audio(waveform=converted_waveform, sampling_rate=output_sample_rate))

        return cloned_audios
