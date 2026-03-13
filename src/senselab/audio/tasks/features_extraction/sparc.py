"""This module provides the implementation of Speech Articulatory Coding utilities for audio features extraction."""

import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.preprocessing import resample_audios
from senselab.utils.data_structures import DeviceType, Language, _select_device_and_dtype, logger

try:
    from sparc import SPARC, load_model

    SPARC_AVAILABLE = True
except ModuleNotFoundError:
    SPARC_AVAILABLE = False


class SparcFeatureExtractor:
    """A factory for managing feature extraction pipelines using SPARC."""

    _models: Dict[str, "SPARC"] = {}

    @classmethod
    def _get_sparc_model(cls, lang: Optional[Language] = None, device: Optional[DeviceType] = None) -> "SPARC":
        """Get or create a SPARC codec model."""
        if not SPARC_AVAILABLE:
            raise ModuleNotFoundError(
                "`sparc` is not available. "
                "Please install senselab audio dependencies using `pip install 'senselab[articulatory]'`."
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
    def extract_sparc_features(
        cls,
        audios: List[Audio],
        lang: Optional[Language] = None,
        device: Optional[DeviceType] = None,
        resample: Optional[bool] = False,
    ) -> List[Dict[str, torch.Tensor]]:
        """Clone voices from source audios to target audios using SPARC.

        Args:
            audios (List[Audio]): List of audio objects.
            lang (Optional[Language], optional): Language for the SPARC model.
                Defaults to None, which means multi-language.
            device (Optional[DeviceType], optional): The device to run the model on.
                Defaults to None.
            resample (Optional[bool]): Whether to resample the audios if not at proper sampling rate.
                An error will be raised if the audios are not at the correct sampling rate and not resampled.

        Returns:
            # All features are in 50 Hz except speaker encoding
            List of:
            {
                "ema": (L, 12) array, #'TDX','TDY','TBX','TBY','TTX','TTY','LIX','LIY','ULX','ULY','LLX','LLY'
                "loudness": (L, 1) array,
                "pitch": (L, 1) array,
                "periodicity": (L, 1) array, # auxiliary output of pitch tracker
                "pitch_stats": (pitch mean, pitch std),
                "spk_emb": (spk_emb_dim,) array, # all shared models use spk_emb_dim=64
                "ft_len": Length of features, # useful when batched processing with padding
            }
        """
        if not SPARC_AVAILABLE:
            raise ModuleNotFoundError(
                "`sparc` is not available. "
                "Please install senselab audio dependencies using `pip install 'senselab[articulatory]'`."
            )

        # Get SPARC model
        coder = cls._get_sparc_model(lang=lang, device=device)
        expected_sample_rate = coder.sr

        codes = []
        if resample:
            audios = resample_audios(audios, resample_rate=expected_sample_rate)

        for audio in audios:
            if audio.waveform.squeeze().dim() != 1:
                raise ValueError(f"Only mono audio files are supported. Source: {audio.generate_id()}")

            if audio.sampling_rate != expected_sample_rate:
                raise ValueError(
                    f"Expected sample rate {expected_sample_rate}, but got "
                    f"{audio.sampling_rate} (source)"
                    f"Source: {audio.generate_id()}."
                )

            try:
                codes.append(coder.encode(audio.waveform.squeeze().numpy()))
            except Exception as e:
                logger.error(f"Exception encountered: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                codes.append(
                    {
                        "ema": torch.tensor(
                            torch.nan
                        ),  #'TDX','TDY','TBX','TBY','TTX','TTY','LIX','LIY','ULX','ULY','LLX','LLY'
                        "loudness": torch.tensor(torch.nan),
                        "pitch": torch.tensor(torch.nan),
                        "pperiodicity": torch.tensor(torch.nan),  # auxiliary output of pitch tracker
                        "pitch_stats": torch.tensor(torch.nan),
                        "spk_emb": torch.tensor(torch.nan),  # all shared models use spk_emb_dim=64
                        "ft_len": torch.nan,
                    }
                )

        return codes
