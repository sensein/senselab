"""This module defines a pydra API for praat_parselmouth features extraction."""

import pydra

from senselab.audio.tasks.features_extraction.opensmile import (
    extract_feats_from_dataset,
)

extract_feats_from_dataset_pt = pydra.mark.task(extract_feats_from_dataset)
