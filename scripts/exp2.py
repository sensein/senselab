"""This script is used to test the audio tasks."""
from senselab.audio.tasks.features_extraction.opensmile import extract_feats_from_dataset
from senselab.audio.tasks.features_extraction.praat_parselmouth import (
    get_hf_dataset_durations,
    get_hf_dataset_f0_descriptors,
    get_hf_dataset_harmonicity_descriptors,
    get_hf_dataset_jitter_descriptors,
    get_hf_dataset_shimmer_descriptors,
)
from senselab.utils.tasks.input_output import read_files_from_disk

dataset = read_files_from_disk(["/Users/fabiocat/Documents/git/pp/senselab/src/tests/data_for_testing/audio_48khz_mono_16bits.wav"])

print(dataset)

duration_dataset = get_hf_dataset_durations(dataset)
f0_dataset = get_hf_dataset_f0_descriptors(dataset, f0min=100, f0max=500)
harmonicity_dataset = get_hf_dataset_harmonicity_descriptors(dataset, f0min=100)
jitter_dataset = get_hf_dataset_jitter_descriptors(dataset, f0min=100, f0max=500)
shimmer_dataset = get_hf_dataset_shimmer_descriptors(dataset, f0min=100, f0max=500)

print(duration_dataset)
print(f0_dataset)
print(harmonicity_dataset)
print(jitter_dataset)
print(shimmer_dataset)

opensmile_feats = extract_feats_from_dataset(dataset, audio_column="audio", feature_set="eGeMAPSv02", feature_level="Functionals")

print(opensmile_feats)

