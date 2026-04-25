"""This module provides the implementation of torchaudio squim utilities for audio features extraction."""

import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import torch.nn

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, _select_device_and_dtype, logger
from senselab.utils.dependencies import torchaudio_available

TORCHAUDIO_AVAILABLE = torchaudio_available()
if TORCHAUDIO_AVAILABLE:
    from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE

_objective_model = None
_subjective_model = None


def _get_objective_model() -> torch.nn.Module:
    """Lazily load the SQUIM objective quality model."""
    global _objective_model  # noqa: PLW0603
    if _objective_model is None:
        _objective_model = SQUIM_OBJECTIVE.get_model()
    return _objective_model


def _get_subjective_model() -> torch.nn.Module:
    """Lazily load the SQUIM subjective quality model."""
    global _subjective_model  # noqa: PLW0603
    if _subjective_model is None:
        _subjective_model = SQUIM_SUBJECTIVE.get_model()
    return _subjective_model


def extract_objective_quality_features_from_audios(
    audios: List["Audio"], device: Optional[DeviceType] = None
) -> List[Dict[str, Any]]:
    """Extracts objective audio features from a list of Audio objects.

    Features include:
    - Wideband Perceptual Estimation of Speech Quality (PESQ)
    - Short-Time Objective Intelligibility (STOI)
    - Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)

    Currently, Torchaudio-Squim model only supports mono audio at 16000 Hz sampling rate.

    Args:
        audios (List[Audio]): List of Audio objects.
        device (DeviceType, optional): device to run feature extraction on

    Returns:
        List[Dict[str, Any]]: List of dictionaries, each containing extracted features for an audio input.
    """
    if not TORCHAUDIO_AVAILABLE:
        raise ModuleNotFoundError(
            "`torchaudio` is not installed. Please install senselab audio dependencies using `pip install senselab`."
        )

    device, _ = _select_device_and_dtype(user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU])
    if any(audio.waveform.shape[0] != 1 for audio in audios):
        raise ValueError("Only mono audio is supported by Torchaudio-Squim model.")

    if any(audio.sampling_rate != 16000 for audio in audios):
        raise ValueError("Only 16000 Hz sampling rate is supported by Torchaudio-Squim model.")

    features: List[Dict[str, Any]] = []
    model = _get_objective_model().to(device.value)

    for audio in audios:
        audio_features = {}
        try:
            stoi, pesq, si_sdr = model(audio.waveform.to(device.value))
            audio_features["stoi"] = stoi.cpu().item()
            audio_features["pesq"] = pesq.cpu().item()
            audio_features["si_sdr"] = si_sdr.cpu().item()
        except RuntimeError as e:
            audio_features["stoi"] = np.nan
            audio_features["pesq"] = np.nan
            audio_features["si_sdr"] = np.nan
            raise (e)

        features.append(audio_features)

    return features


def extract_subjective_quality_features_from_audios(
    audios: List["Audio"], non_matching_references: List["Audio"]
) -> List[Dict[str, Any]]:
    """Extracts subjective audio features from a list of Audio objects.

    Currently, Torchaudio-Squim model only supports mono audio at 16000 Hz sampling rate.

    Args:
        audios (List[Audio]): List of Audio objects.
        non_matching_references (List[Audio]): Reference Audio objects for the subjective model.

    Returns:
        List[Dict[str, Any]]: List of dictionaries, each containing extracted features for an audio input.
    """
    if not TORCHAUDIO_AVAILABLE:
        raise ModuleNotFoundError(
            "`torchaudio` is not installed. Please install senselab audio dependencies using `pip install senselab`."
        )

    # Check if any audio is not mono
    if any(audio.waveform.shape[0] != 1 for audio in audios) or any(
        ref.waveform.shape[0] != 1 for ref in non_matching_references
    ):
        raise ValueError("Only mono audio is supported by Torchaudio-Squim model.")

    # Check if any audio has a sampling rate other than 16000 Hz
    if any(audio.sampling_rate != 16000 for audio in audios) or any(
        ref.sampling_rate != 16000 for ref in non_matching_references
    ):
        raise ValueError("Only 16000 Hz sampling rate is supported by Torchaudio-Squim model.")

    features: List[Dict[str, Any]] = []

    for i, audio in enumerate(audios):
        audio_features = {}
        try:
            mos = _get_subjective_model()(audio.waveform, non_matching_references[i].waveform)
            audio_features["mos"] = mos.item()
        except RuntimeError as e:
            audio_features["mos"] = np.nan
            logger.error(f"RuntimeException encountered: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

        features.append(audio_features)

    return features
