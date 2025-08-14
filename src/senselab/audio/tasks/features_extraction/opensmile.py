"""This module contains functions for extracting openSMILE features.

It includes a factory class for managing openSMILE feature extractors, ensuring
each extractor is created only once per feature set and feature level. The main
function, `extract_opensmile_features_from_audios`, applies feature extraction
across a list of audio samples using openSMILE, managed as a Pydra workflow
for parallel processing. This approach supports efficient and scalable feature
extraction across multiple audio files.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

from pydra.compose import python, workflow

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
                "Please install senselab audio dependencies using `pip install 'senselab[audio]'`"
            )

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
    plugin: str = "debug",
    plugin_args: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str | os.PathLike] = None,
) -> List[Dict[str, Any]]:
    """Extract openSMILE features from a list of audio files using a Pydra compose workflow.

    Args:
        audios (List[Audio]): A list of Audio objects.
        feature_set (str, optional): The feature set to use. Defaults to "eGeMAPSv02".
        feature_level (str, optional): The feature level to use. Defaults to "Functionals".
        plugin (str, optional): The Pydra plugin to use for workflow submission. Defaults to "debug".
        plugin_args (Optional[Dict[str, Any]], optional): Additional arguments for the Pydra plugin.
            Defaults to None.
        cache_dir (Optional[str | os.PathLike], optional): The directory for caching intermediate results.
            Defaults to None.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the extracted features.

    Raises:
        ModuleNotFoundError: If `opensmile` is not installed.
    """
    if not OPENSMILE_AVAILABLE:
        raise ModuleNotFoundError(
            "`opensmile` is not installed. Please install the necessary dependencies using:\n"
            "`pip install 'senselab[audio]'`"
        )

    @python.define
    def _extract_feats_from_audio(sample: Audio, feature_set: str, feature_level: str) -> Dict[str, Any]:
        smile = OpenSmileFeatureExtractorFactory.get_opensmile_extractor(feature_set, feature_level)
        audio_array = sample.waveform.squeeze().numpy()
        sampling_rate = sample.sampling_rate
        try:
            df = smile.process_signal(audio_array, sampling_rate)
            return {k: (v[0] if isinstance(v, list) and len(v) == 1 else v) for k, v in df.to_dict("list").items()}
        except Exception as e:
            filepath = sample.filepath() if hasattr(sample, "filepath") and sample.filepath() else ""
            desc = f"{sample.generate_id()}{f' ({filepath})' if filepath else ''}"
            print(f"Error processing sample {desc}: {e}")
            names = getattr(smile, "feature_names", [])
            return {name: float("nan") for name in names} if names else {}

    @workflow.define
    def _wf(xs: Sequence[Audio], feature_set: str, feature_level: str) -> List[Dict[str, Any]]:
        t = _extract_feats_from_audio(
            feature_set=feature_set,
            feature_level=feature_level,
        ).split(sample=xs)

        node = workflow.add(t, name="map_extract_feats")
        return node.out

    worker = "debug" if plugin in ("serial", "debug") else plugin
    worker_kwargs = plugin_args or {}

    wf = _wf(xs=audios, feature_set=feature_set, feature_level=feature_level)
    res: Any = wf(worker=worker, cache_root=cache_dir, **worker_kwargs)
    return list(res.out)
