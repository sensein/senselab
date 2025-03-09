"""This module contains functions for extracting openSMILE features.

It includes a factory class for managing openSMILE feature extractors, ensuring
each extractor is created only once per feature set and feature level. The main
function, `extract_opensmile_features_from_audios`, applies feature extraction
across a list of audio samples using openSMILE, managed as a Pydra workflow
for parallel processing. This approach supports efficient and scalable feature
extraction across multiple audio files.
"""

try:
    import opensmile

    OPENSMILE_AVAILABLE = True
except ModuleNotFoundError:
    OPENSMILE_AVAILABLE = False

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pydra

from senselab.audio.data_structures import Audio


class OpenSmileFeatureExtractorFactory:
    """A factory for managing openSMILE feature extractors.

    This class creates and caches openSMILE feature extractors, allowing for
    efficient reuse. It ensures only one instance of each feature extractor
    exists per unique combination of `feature_set` and `feature_level`.
    """

    _extractors: Dict[str, "opensmile.Smile"] = {}  # Cache for feature extractors

    @classmethod
    def get_opensmile_extractor(cls, feature_set: str, feature_level: str) -> "opensmile.Smile":
        """Get or create an openSMILE feature extractor.

        Args:
            feature_set (str): The openSMILE feature set.
            feature_level (str): The openSMILE feature level.

        Returns:
            opensmile.Smile: The openSMILE feature extractor.
        """
        if not OPENSMILE_AVAILABLE:
            raise ModuleNotFoundError(
                "`opensmile` is not installed. "
                "Please install senselab audio dependencies using `pip install senselab['audio']`"
            )

        key = f"{feature_set}-{feature_level}"  # Unique key for each feature extractor
        if key not in cls._extractors:  # Check if extractor exists in cache
            # Create and store a new extractor if not found in cache
            cls._extractors[key] = opensmile.Smile(
                feature_set=opensmile.FeatureSet[feature_set],
                feature_level=opensmile.FeatureLevel[feature_level],
            )
        return cls._extractors[key]  # Return cached or newly created extractor


def extract_opensmile_features_from_audios(
    audios: List[Audio],
    feature_set: str = "eGeMAPSv02",
    feature_level: str = "Functionals",
    plugin: str = "serial",
    plugin_args: Optional[Dict[str, Any]] = {},
    cache_dir: Optional[str | os.PathLike] = None,
) -> List[Dict[str, Any]]:
    """Extract openSMILE features from a list of audio files using Pydra workflow.

    This function sets up a Pydra workflow for parallel processing of openSMILE
    feature extraction on a list of audio samples. Each sample's features are
    extracted and formatted as dictionaries.

    Args:
        audios (List[Audio]): The list of audio objects to extract features from.
        feature_set (str): The openSMILE feature set (default is "eGeMAPSv02").
        feature_level (str): The openSMILE feature level (default is "Functionals").
        plugin (str): The Pydra plugin to use (default is "serial").
        plugin_args (Optional[Dict[str, Any]]): Additional arguments for the Pydra plugin.
        cache_dir (Optional[str | os.PathLike]): The path to the Pydra cache directory.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing extracted features.
    """
    if not OPENSMILE_AVAILABLE:
        raise ModuleNotFoundError(
            "`opensmile` is not installed. Please install the necessary dependencies using:\n"
            "`pip install senselab['audio']`"
        )

    def _extract_feats_from_audio(sample: Audio, smile: opensmile.Smile) -> Dict[str, Any]:
        """Extract features from a single audio sample using openSMILE.

        Args:
            sample (Audio): The audio object.
            smile (opensmile.Smile): The openSMILE feature extractor.

        Returns:
            Dict[str, Any]: The extracted features as a dictionary.
        """
        # Convert audio tensor to a NumPy array for processing
        audio_array = sample.waveform.squeeze().numpy()
        sampling_rate = sample.sampling_rate  # Get sampling rate from Audio object
        try:
            # Process the audio and extract features
            sample_features = smile.process_signal(audio_array, sampling_rate)
            # Convert features to a dictionary and handle single-item lists
            return {
                k: v[0] if isinstance(v, list) and len(v) == 1 else v
                for k, v in sample_features.to_dict("list").items()
            }
        except Exception as e:
            # Log error and return NaNs if feature extraction fails
            print(f"Error processing sample {sample.orig_path_or_id}: {e}")
            return {feature: np.nan for feature in smile.feature_names}

    # Decorate the feature extraction function for Pydra
    _extract_feats_from_audio_pt = pydra.mark.task(_extract_feats_from_audio)

    # Obtain the feature extractor using the factory
    smile = OpenSmileFeatureExtractorFactory.get_opensmile_extractor(feature_set, feature_level)

    # Create a Pydra workflow, split it over the list of audio samples
    wf = pydra.Workflow(name="wf", input_spec=["x"], cache_dir=cache_dir)
    wf.split("x", x=audios)  # Each audio is treated as a separate task
    # Add feature extraction task to the workflow
    wf.add(_extract_feats_from_audio_pt(name="_extract_feats_from_audio_pt", sample=wf.lzin.x, smile=smile))

    # Set workflow output to the results of each audio feature extraction
    wf.set_output([("opensmile", wf._extract_feats_from_audio_pt.lzout.out)])

    # Run the workflow using the specified Pydra plugin and arguments
    with pydra.Submitter(plugin=plugin, **plugin_args) as sub:
        sub(wf)

    # Retrieve results from the completed workflow
    outputs = wf.result()

    # Format the outputs into a list of dictionaries
    formatted_output: List[Dict[str, Any]] = []
    for output in outputs:
        # Extract features and organize into a dictionary
        formatted_output_item = {
            f"{feature}": output.output.opensmile[f"{feature}"] for feature in output.output.opensmile
        }
        formatted_output.append(formatted_output_item)  # Append to final output list
    return formatted_output  # Return the list of formatted feature dictionaries
