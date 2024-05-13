"""This module defines a pydra API for the praat_parselmouth features extraction task."""
import pydra

from senselab.audio.tasks.features_extraction.praat_parselmouth import (
    get_hf_dataset_durations,
    get_hf_dataset_f0_descriptors,
    get_hf_dataset_harmonicity_descriptors,
    get_hf_dataset_jitter_descriptors,
    get_hf_dataset_shimmer_descriptors,
)

get_hf_dataset_durations_pt = pydra.mark.task(get_hf_dataset_durations)
get_hf_dataset_f0_descriptors_pt = pydra.mark.task(get_hf_dataset_f0_descriptors)
get_hf_dataset_harmonicity_descriptors_pt = pydra.mark.task(get_hf_dataset_harmonicity_descriptors)
get_hf_dataset_jitter_descriptors_pt = pydra.mark.task(get_hf_dataset_jitter_descriptors)
get_hf_dataset_shimmer_descriptors_pt = pydra.mark.task(get_hf_dataset_shimmer_descriptors)
