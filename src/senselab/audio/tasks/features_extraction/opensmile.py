"""This module contains functions for extracting openSMILE features.

It includes a factory class for managing openSMILE feature extractors, ensuring
each extractor is created only once per feature set and feature level. The main
function, `extract_opensmile_features_from_audios`, applies feature extraction
across a list of audio samples using openSMILE in a simple for-loop.
"""

from __future__ import annotations

from typing import Any, Dict, List

from senselab.audio.data_structures import Audio

try:
    import opensmile

    OPENSMILE_AVAILABLE = True
except ModuleNotFoundError:
    OPENSMILE_AVAILABLE = False

    class DummyOpenSmile:
        """Dummy class to represent openSMILE when it's not available."""

        def __init__(self) -> None:
            """Dummy constructor for when openSMILE is not available."""
            self.__dict__ = {}

        class Smile:
            """Dummy class for when openSMILE is not available."""

            def __init__(self, *args: object, **kwargs: object) -> None:
                """Dummy constructor for when openSMILE is not available."""
                pass

    opensmile = DummyOpenSmile()  # type: ignore[assignment]


class OpenSmileFeatureExtractorFactory:
    """A factory for managing openSMILE feature extractors."""

    _extractors: Dict[str, opensmile.Smile] = {}

    @classmethod
    def get_opensmile_extractor(cls, feature_set: str, feature_level: str) -> opensmile.Smile:
        """Get an openSMILE feature extractor for a given feature set and feature level.

        Args:
            feature_set (str): The feature set to use.
            feature_level (str): The feature level to use.

        Returns:
            opensmile.Smile: The openSMILE feature extractor.

        Raises:
            ModuleNotFoundError: If `opensmile` is not installed.
        """
        if not OPENSMILE_AVAILABLE:
            raise ModuleNotFoundError(
                "`opensmile` is not installed. "
                "Please install senselab audio dependencies using `pip install senselab`"
            )

        key = f"{feature_set}-{feature_level}"
        if key not in cls._extractors:
            cls._extractors[key] = opensmile.Smile(
                feature_set=opensmile.FeatureSet[feature_set],
                feature_level=opensmile.FeatureLevel[feature_level],
            )
        return cls._extractors[key]


def extract_opensmile_features_from_audios(
    audios: List[Audio], feature_set: str = "eGeMAPSv02", feature_level: str = "Functionals"
) -> List[Dict[str, Any]]:
    """Extract openSMILE features from a list of audio files.

    Args:
        audios (List[Audio]): A list of Audio objects.
        feature_set (str, optional): The feature set to use. Defaults to "eGeMAPSv02".
        feature_level (str, optional): The feature level to use. Defaults to "Functionals".

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the extracted features.

    Raises:
        ModuleNotFoundError: If `opensmile` is not installed.
    """
    if not OPENSMILE_AVAILABLE:
        raise ModuleNotFoundError(
            "`opensmile` is not installed. Please install the necessary dependencies using:\n" "`pip install senselab`"
        )

    # Initialize extractor once
    smile = OpenSmileFeatureExtractorFactory.get_opensmile_extractor(feature_set, feature_level)

    results: List[Dict[str, Any]] = []
    for sample in audios:
        audio_array = sample.waveform.squeeze().numpy()
        sampling_rate = sample.sampling_rate
        try:
            df = smile.process_signal(audio_array, sampling_rate)
            # Convert DataFrame (1 row) to a flat dict of {feature_name: value}
            row_dict = {k: (v[0] if isinstance(v, list) and len(v) == 1 else v) for k, v in df.to_dict("list").items()}
            results.append(row_dict)
        except Exception as e:
            filepath = sample.filepath() if hasattr(sample, "filepath") and sample.filepath() else ""
            desc = f"{sample.generate_id()}{f' ({filepath})' if filepath else ''}"
            print(f"Error processing sample {desc}: {e}")
            names = getattr(smile, "feature_names", [])
            results.append({name: float("nan") for name in names} if names else {})

    return results
