"""This module provides the implementation of torchaudio squim utilities for audio features extraction."""

from typing import Any, Dict, List

import pydra
from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE

from senselab.audio.data_structures.audio import Audio

objective_model = SQUIM_OBJECTIVE.get_model()
subjective_model = SQUIM_SUBJECTIVE.get_model()


def extract_objective_quality_features_from_audios(audio_list: List[Audio]) -> Dict[str, Any]:
    """Extracts objective audio features from a list of Audio objects.

    Currently, Torchaudio-Squim model only supports mono audio at 16000 Hz sampling rate.

    Args:
        audio_list (List[Audio]): List of Audio objects.

    Returns:
        Dict[str, Any]: Dictionary containing extracted features.
    """
    if any(audio.waveform.shape[0] != 1 for audio in audio_list):
        raise ValueError("Only mono audio is supported by Torchaudio-Squim model.")

    if any(audio.sampling_rate != 16000 for audio in audio_list):
        raise ValueError("Only 16000 Hz sampling rate is supported by Torchaudio-Squim model.")

    features: Dict[str, Any] = {"stoi": [], "pesq": [], "si_sdr": []}

    for audio in audio_list:
        stoi, pesq, si_sdr = objective_model(audio.waveform)
        features["stoi"].append(stoi.item())
        features["pesq"].append(pesq.item())
        features["si_sdr"].append(si_sdr.item())

    return features


def extract_subjective_quality_features_from_audios(
    audio_list: List[Audio], non_matching_references: List[Audio]
) -> Dict[str, Any]:
    """Extracts subjective audio features from a list of Audio objects.

    Currently, Torchaudio-Squim model only supports mono audio at 16000 Hz sampling rate.

    Args:
        audio_list (List[Audio]): List of Audio objects.
        non_matching_references (List[Audio]): Reference Audio objects for the subjective model.

    Returns:
        Dict[str, Any]: Dictionary containing extracted features.
    """
    # Check if any audio is not mono
    if any(audio.waveform.shape[0] != 1 for audio in audio_list) or any(
        ref.waveform.shape[0] != 1 for ref in non_matching_references
    ):
        raise ValueError("Only mono audio is supported by Torchaudio-Squim model.")

    # Check if any audio has a sampling rate other than 16000 Hz
    if any(audio.sampling_rate != 16000 for audio in audio_list) or any(
        ref.sampling_rate != 16000 for ref in non_matching_references
    ):
        raise ValueError("Only 16000 Hz sampling rate is supported by Torchaudio-Squim model.")

    features: Dict[str, Any] = {"mos": []}

    for i, audio in enumerate(audio_list):
        mos = subjective_model(audio.waveform, non_matching_references[i].waveform)
        features["mos"].append(mos.item())

    return features


extract_objective_quality_features_from_audios_pt = pydra.mark.task(extract_objective_quality_features_from_audios)
extract_subjective_quality_features_from_audios_pt = pydra.mark.task(extract_subjective_quality_features_from_audios)
