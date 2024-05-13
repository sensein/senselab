"""This module contains functions for extracting features from audios using openSMILE.

TODO: fix installation conflicts with openSMILE
"""

'''
from datasets import load_dataset, load_from_disk
import opensmile
from tqdm.auto import tqdm
import numpy as np

def load_opensmile_model(feature_set, feature_level):
    """Load an openSMILE configuration to extract audio features."""
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet[feature_set],
        feature_level=opensmile.FeatureLevel[feature_level],
    )
    return smile

def extract_feats_from_batch(batch, smile):
    """Extract features from a batch of audio samples using the specified openSMILE model."""
    # Extracting audio data
    audio_arrays = [audio['array'] for audio in batch['audio']]
    sampling_rates = [audio['sampling_rate'] for audio in batch['audio']]

    # Processing each audio sample in the batch to compute features
    batch_features = [smile.process_signal(array, rate) for array, rate in zip(audio_arrays, sampling_rates)]

    # Returning a dictionary with only 'features' field
    return {'features': batch_features}

def extract_feats_from_dataset(dataset, feature_set, feature_level, batch_size=16):
    """Apply feature extraction across a dataset of audio files in batches."""
    smile = load_opensmile_model(feature_set, feature_level)
    updated_dataset = dataset.map(
        extract_feats_from_batch,
        batched=True,
        batch_size=batch_size,
        fn_kwargs={"smile": smile}
    )
    return updated_dataset
'''