"""This module contains functions for voice cloning using SPARC."""

from typing import Dict, List, Optional

import numpy as np

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, Language, _select_device_and_dtype

try:
    from sparc import SPARC, load_model

    SPARC_AVAILABLE = True
except ModuleNotFoundError:
    SPARC_AVAILABLE = False


class SparcVoiceCloner:
    """A factory for managing voice cloning pipelines using SPARC."""

    _models: Dict[str, "SPARC"] = {}

    @classmethod
    def _get_sparc_model(cls, lang: Optional[Language] = None, device: Optional[DeviceType] = None) -> "SPARC":
        """Get or create a SPARC codec model."""
        if not SPARC_AVAILABLE:
            raise ModuleNotFoundError(
                "`sparc` is not available. "
                "Please install senselab audio dependencies using `pip install senselab['audio']`."
            )
        device, _ = _select_device_and_dtype(
            user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )
        if lang is None:
            used_language = "multi"
        elif lang.name == "english":
            used_language = "en+"
            # we could use also "en" here (training dataset LibriTTS-R),
            # but it seems better to use "en+" (training datasets LibriTTS-R, LibriTTS, EXPRESSO)
        else:
            raise ValueError(
                f"Language {lang.name} not supported for now. "
                "Supported languages are: english or None, which means multi-language."
            )

        key = f"{used_language}-{device.value}"
        if key not in cls._models:
            cls._models[key] = load_model(used_language, device=device.value)
        return cls._models[key]

    @classmethod
    def clone_voices(
        cls,
        source_audios: List[Audio],
        target_audios: List[Audio],
        lang: Optional[Language] = None,
        device: Optional[DeviceType] = None,
    ) -> List[Audio]:
        """Clone voices from source audios to target audios using SPARC.

        Args:
            source_audios (List[Audio]): List of source audio objects.
            target_audios (List[Audio]): List of target audio objects.
            lang (Optional[Language], optional): Language for the SPARC model.
                Defaults to None, which means multi-language.
            device (Optional[DeviceType], optional): The device to run the model on.
                Defaults to None.

        Returns:
            List[Audio]: List of cloned audio objects.
        """
        if not SPARC_AVAILABLE:
            raise ModuleNotFoundError(
                "`sparc` is not available. "
                "Please install senselab audio dependencies using `pip install senselab['audio']`."
            )

        if len(source_audios) != len(target_audios):
            raise ValueError("Number of source and target audios must be the same.")

        # Get SPARC model
        coder = cls._get_sparc_model(lang=lang, device=device)
        expected_sample_rate = coder.sr

        for source_audio, target_audio in zip(source_audios, target_audios):
            if source_audio.waveform.squeeze().dim() != 1 or target_audio.waveform.squeeze().dim() != 1:
                raise ValueError(
                    "Error with the pair of source and target audios: "
                    f"{source_audio.orig_path_or_id} and {target_audio.orig_path_or_id}. "
                    "Only mono audio files are supported."
                )

            if source_audio.sampling_rate != expected_sample_rate or target_audio.sampling_rate != expected_sample_rate:
                raise ValueError(
                    "Error with the pair of source and target audios: "
                    f"{source_audio.orig_path_or_id} and {target_audio.orig_path_or_id}. "
                    f"Expected sample rate {expected_sample_rate}, but got "
                    f"{source_audio.sampling_rate} (source) and {target_audio.sampling_rate} (target)."
                )

        cloned_audios = []
        for source_audio, target_audio in zip(source_audios, target_audios):
            # Extract and flatten waveforms
            src_wav = source_audio.waveform.numpy().squeeze().astype(np.float32)
            trg_wav = target_audio.waveform.numpy().squeeze().astype(np.float32)

            # Perform voice conversion
            converted_waveform = coder.convert(src_wav=src_wav, trg_wav=trg_wav)

            cloned_audios.append(Audio(waveform=converted_waveform, sampling_rate=expected_sample_rate))

        return cloned_audios
