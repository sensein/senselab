"""This module contains functions for extracting openSMILE features."""

from typing import Any, Dict, List

import opensmile
import pydra

from senselab.audio.data_structures.audio import Audio


class OpenSmileFeatureExtractorFactory:
    """A factory for managing openSMILE feature extractors."""

    _extractors: Dict[str, opensmile.Smile] = {}

    @classmethod
    def get_opensmile_extractor(cls, feature_set: str, feature_level: str) -> opensmile.Smile:
        """Get or create an openSMILE feature extractor.

        Args:
            feature_set (str): The openSMILE feature set.
            feature_level (str): The openSMILE feature level.

        Returns:
            opensmile.Smile: The openSMILE feature extractor.
        """
        key = f"{feature_set}-{feature_level}"
        if key not in cls._extractors:
            cls._extractors[key] = opensmile.Smile(
                feature_set=opensmile.FeatureSet[feature_set],
                feature_level=opensmile.FeatureLevel[feature_level],
            )
        return cls._extractors[key]


def extract_opensmile_features_from_audios(
    audios: List[Audio],
    feature_set: str = "eGeMAPSv02",
    feature_level: str = "Functionals",
) -> List[Dict[str, Any]]:
    """Apply feature extraction across a list of audio files.

    Args:
        audios (List[Audio]): The list of audio objects to extract features from.
        feature_set (str): The openSMILE feature set (default is "eGeMAPSv02").
        feature_level (str): The openSMILE feature level (default is "Functionals").

    Returns:
        List[Dict[str, Any]]: The list of feature dictionaries for each audio.
    """

    def _extract_feats_from_audio(sample: Audio, smile: opensmile.Smile) -> Dict[str, Any]:
        """Extract features from a single audio sample using openSMILE.

        Args:
            sample (Audio): The audio object.
            smile (opensmile.Smile): The openSMILE feature extractor.

        Returns:
            Dict[str, Any]: The extracted features as a dictionary.
        """
        audio_array = sample.waveform.squeeze().numpy()
        sampling_rate = sample.sampling_rate
        sample_features = smile.process_signal(audio_array, sampling_rate)
        # Convert to a dictionary with float values and return it
        return {
            k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in sample_features.to_dict("list").items()
        }

    smile = OpenSmileFeatureExtractorFactory.get_opensmile_extractor(feature_set, feature_level)
    features = [_extract_feats_from_audio(audio, smile) for audio in audios]
    return features


extract_opensmile_features_from_audios_pt = pydra.mark.task(extract_opensmile_features_from_audios)
