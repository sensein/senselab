"""This module contains functions for extracting features from audios using openSMILE."""

from typing import Any, Dict

import opensmile

from senselab.utils.tasks.input_output import _from_dict_to_hf_dataset, _from_hf_dataset_to_dict


def extract_feats_from_dataset(dataset: Dict[str, Any], audio_column: str = 'audio', feature_set: str = "eGeMAPSv02", feature_level: str = "Functionals") -> Dict[str, Any]:
    """Apply feature extraction across a dataset of audio files.
    
    Low-level descriptors are extracted on 20ms windows with a hop of 10ms.
    Functional descriptors are extracted on the entire audio signal.
    """
    def _load_opensmile_model(feature_set: str, feature_level: str) -> opensmile.Smile:
        """Load an openSMILE configuration to extract audio features."""
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet[feature_set],
            feature_level=opensmile.FeatureLevel[feature_level],
        )
        return smile

    def _extract_feats_from_row(sample: Dict[str, Any], smile: opensmile.Smile, audio_column: str) -> Dict[str, Any]:
        """Extract features from a single audio sample using the specified openSMILE model."""
        # Extracting audio data
        audio_array = sample[audio_column]['array']
        sampling_rate = sample[audio_column]['sampling_rate']

        # Processing the audio sample to compute features
        sample_features = smile.process_signal(audio_array, sampling_rate)
        return sample_features.to_dict("list")

    hf_dataset = _from_dict_to_hf_dataset(dataset)
    unnecessary_columns = [col for col in hf_dataset.column_names if col != audio_column]
    hf_dataset = hf_dataset.remove_columns(unnecessary_columns)

    smile = _load_opensmile_model(feature_set, feature_level)
    features_dataset = hf_dataset.map(
        _extract_feats_from_row,
        fn_kwargs={"smile": smile, "audio_column": audio_column},
    )
    features_dataset = features_dataset.remove_columns([audio_column])
    return _from_hf_dataset_to_dict(features_dataset)
