"""This module provides the implementation of Phonetic Posteriorgrams (PPGs) for audio features extraction."""

import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, _select_device_and_dtype, logger

try:
    import ppgs

    PPGS_AVAILABLE = True
except ModuleNotFoundError:
    PPGS_AVAILABLE = False


def extract_ppgs_from_audios(audios: List["Audio"], device: Optional[DeviceType] = None) -> List[torch.Tensor]:
    """Extracts phonetic posteriorgrams (PPGs) from every audio.

    Args:
        audios (List[Audio]): The audios to extract PPGs from
        device (Optional[DeviceType]): Device to use for extracting PPGs

    Returns:
        List[Tensor]: The PPG for each input audio
    """
    if not PPGS_AVAILABLE:
        raise ModuleNotFoundError(
            "`ppgs` is not installed. Please install senselab audio dependencies using `pip install senselab`."
        )

    device, _ = _select_device_and_dtype(user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU])
    if any(audio.waveform.shape[0] != 1 for audio in audios):
        raise ValueError("Only mono audio is supported by ppgs model.")

    posteriorgrams = []
    for audio in audios:
        try:
            posteriorgrams.append(
                ppgs.from_audio(
                    torch.unsqueeze(audio.waveform, dim=0),
                    ppgs.SAMPLE_RATE,
                    gpu=0 if device == DeviceType.CUDA else None,
                ).cpu()
            )

        except RuntimeError as e:
            logger.error(f"Encountered RuntimeError when extracting ppgs: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            posteriorgrams.append(torch.tensor(torch.nan))

    return posteriorgrams
