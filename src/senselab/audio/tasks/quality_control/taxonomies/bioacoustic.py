"""Bioacoustic activity taxonomy tree implementation."""

from senselab.audio.tasks.quality_control.checks import (
    completely_silent_check,
    high_proportion_clipped_check,
    low_spectral_gating_snr_check,
    very_low_amplitude_modulation_depth_check,
    very_low_peak_snr_from_spectral_check,
)
from senselab.audio.tasks.quality_control.metrics import (
    amplitude_modulation_depth_metric,
    peak_snr_from_spectral_metric,
    proportion_clipped_metric,
    proportion_silent_metric,
    spectral_gating_snr_metric,
)
from senselab.audio.tasks.quality_control.taxonomy import TaxonomyNode


def build_bioacoustic_activity_taxonomy() -> TaxonomyNode:
    """Build the complete bioacoustic activity taxonomy tree.

    Returns:
        The root node of the bioacoustic activity taxonomy tree
    """
    # Create root node
    root = TaxonomyNode(
        name="bioacoustic",
        checks=[
            high_proportion_clipped_check,
            completely_silent_check,
            very_low_peak_snr_from_spectral_check,
            low_spectral_gating_snr_check,
            very_low_amplitude_modulation_depth_check,
        ],
        metrics=[
            proportion_clipped_metric,
            proportion_silent_metric,
            peak_snr_from_spectral_metric,
            spectral_gating_snr_metric,
            amplitude_modulation_depth_metric,
        ],
    )

    # Create human node
    human = TaxonomyNode(name="human", checks=[], metrics=[])
    root.add_child("human", human)

    # Create breathing branch
    breathing = TaxonomyNode(name="breathing", checks=[], metrics=[])
    human.add_child("breathing", breathing)

    # Create exhalation branch under breathing
    exhalation = TaxonomyNode(name="exhalation", checks=[], metrics=[])
    breathing.add_child("exhalation", exhalation)

    # Cough sub-branch under exhalation
    cough = TaxonomyNode(name="cough", checks=[], metrics=[])
    exhalation.add_child("cough", cough)

    # Cough types for Bridge2AI - specific file mappings
    audio_check_3 = TaxonomyNode(
        name="audio_check_3", checks=[], metrics=[]
    )
    respiration_and_cough_cough_1 = TaxonomyNode(
        name="respiration_and_cough_cough_1", checks=[], metrics=[]
    )
    respiration_and_cough_cough_2 = TaxonomyNode(
        name="respiration_and_cough_cough_2", checks=[], metrics=[]
    )
    voluntary_cough = TaxonomyNode(
        name="voluntary_cough", checks=[], metrics=[]
    )

    cough.add_child("audio_check_3", audio_check_3)
    cough.add_child(
        "respiration_and_cough_cough_1", respiration_and_cough_cough_1
    )
    cough.add_child(
        "respiration_and_cough_cough_2", respiration_and_cough_cough_2
    )
    cough.add_child("voluntary_cough", voluntary_cough)

    # Create vocalization branch
    vocalization = TaxonomyNode(name="vocalization", checks=[], metrics=[])
    human.add_child("vocalization", vocalization)

    # Speech sub-branch
    speech = TaxonomyNode(name="speech", checks=[], metrics=[])
    vocalization.add_child("speech", speech)

    # Speech types (simplified)
    unscripted = TaxonomyNode(name="unscripted", checks=[], metrics=[])
    reading = TaxonomyNode(name="reading", checks=[], metrics=[])

    speech.add_child("unscripted", unscripted)
    speech.add_child("reading", reading)

    # Non-speech sub-branch
    non_speech = TaxonomyNode(name="non_speech", checks=[], metrics=[])
    vocalization.add_child("non_speech", non_speech)

    # Non-speech types
    laughter = TaxonomyNode(name="laughter", checks=[], metrics=[])
    crying = TaxonomyNode(name="crying", checks=[], metrics=[])
    humming = TaxonomyNode(name="humming", checks=[], metrics=[])
    throat_clearing = TaxonomyNode(
        name="throat_clearing", checks=[], metrics=[]
    )

    non_speech.add_child("laughter", laughter)
    non_speech.add_child("crying", crying)
    non_speech.add_child("humming", humming)
    non_speech.add_child("throat_clearing", throat_clearing)

    # Productive vocabulary under unscripted
    productive_vocabulary = TaxonomyNode(
        name="productive_vocabulary", checks=[], metrics=[]
    )
    unscripted.add_child("productive_vocabulary", productive_vocabulary)

    # Specific productive vocabulary files
    productive_vocabulary_1 = TaxonomyNode(
        name="productive_vocabulary_1", checks=[], metrics=[]
    )
    productive_vocabulary_2 = TaxonomyNode(
        name="productive_vocabulary_2", checks=[], metrics=[]
    )
    productive_vocabulary_3 = TaxonomyNode(
        name="productive_vocabulary_3", checks=[], metrics=[]
    )
    productive_vocabulary_4 = TaxonomyNode(
        name="productive_vocabulary_4", checks=[], metrics=[]
    )
    productive_vocabulary_5 = TaxonomyNode(
        name="productive_vocabulary_5", checks=[], metrics=[]
    )
    productive_vocabulary_6 = TaxonomyNode(
        name="productive_vocabulary_6", checks=[], metrics=[]
    )

    productive_vocabulary.add_child(
        "productive_vocabulary_1", productive_vocabulary_1
    )
    productive_vocabulary.add_child(
        "productive_vocabulary_2", productive_vocabulary_2
    )
    productive_vocabulary.add_child(
        "productive_vocabulary_3", productive_vocabulary_3
    )
    productive_vocabulary.add_child(
        "productive_vocabulary_4", productive_vocabulary_4
    )
    productive_vocabulary.add_child(
        "productive_vocabulary_5", productive_vocabulary_5
    )
    productive_vocabulary.add_child(
        "productive_vocabulary_6", productive_vocabulary_6
    )

    return root


# Create the global taxonomy tree instance
BIOACOUSTIC_ACTIVITY_TAXONOMY = build_bioacoustic_activity_taxonomy()
