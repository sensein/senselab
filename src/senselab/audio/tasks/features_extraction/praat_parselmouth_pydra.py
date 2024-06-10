"""This module defines a pydra API for praat_parselmouth features extraction."""

import pydra

from senselab.audio.tasks.features_extraction.praat_parselmouth import (
    get_audios_durations,
    get_audios_f0_descriptors,
    get_audios_harmonicity_descriptors,
    get_audios_jitter_descriptors,
    get_audios_shimmer_descriptors,
)

get_audios_durations_pt = pydra.mark.task(get_audios_durations)
get_audios_f0_descriptors_pt = pydra.mark.task(get_audios_f0_descriptors)
get_audios_harmonicity_descriptors_pt = pydra.mark.task(get_audios_harmonicity_descriptors)
get_audios_jitter_descriptors_pt = pydra.mark.task(get_audios_jitter_descriptors)
get_audios_shimmer_descriptors_pt = pydra.mark.task(get_audios_shimmer_descriptors)
