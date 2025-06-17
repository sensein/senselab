"""Bioacoustic activity taxonomy tree implementations."""

from senselab.audio.tasks.quality_control.checks import (
    audio_intensity_positive_check,
    audio_length_positive_check,
)
from senselab.audio.tasks.quality_control.metrics import (
    amplitude_headroom_metric,
    amplitude_interquartile_range_metric,
    amplitude_kurtosis_metric,
    amplitude_modulation_depth_metric,
    amplitude_skew_metric,
    crest_factor_metric,
    dynamic_range_metric,
    mean_absolute_amplitude_metric,
    mean_absolute_deviation_metric,
    peak_snr_from_spectral_metric,
    phase_correlation_metric,
    proportion_clipped_metric,
    proportion_silence_at_beginning_metric,
    proportion_silence_at_end_metric,
    proportion_silent_metric,
    root_mean_square_energy_metric,
    shannon_entropy_amplitude_metric,
    signal_variance_metric,
    spectral_gating_snr_metric,
    zero_crossing_rate_metric,
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
            audio_length_positive_check,
            audio_intensity_positive_check,
        ],
        metrics=[
            proportion_silent_metric,
            proportion_silence_at_beginning_metric,
            proportion_silence_at_end_metric,
            amplitude_headroom_metric,
            spectral_gating_snr_metric,
            proportion_clipped_metric,
            amplitude_modulation_depth_metric,
            root_mean_square_energy_metric,
            zero_crossing_rate_metric,
            signal_variance_metric,
            dynamic_range_metric,
            mean_absolute_amplitude_metric,
            mean_absolute_deviation_metric,
            shannon_entropy_amplitude_metric,
            crest_factor_metric,
            peak_snr_from_spectral_metric,
            amplitude_skew_metric,
            amplitude_kurtosis_metric,
            amplitude_interquartile_range_metric,
            phase_correlation_metric,
        ],
    )

    # Create human node
    human = TaxonomyNode(name="human", checks=[], metrics=[])
    root.add_child("human", human)

    # Create breathing branch
    breathing = TaxonomyNode(name="breathing", checks=[], metrics=[])
    human.add_child("breathing", breathing)

    # Breathing types
    deep = TaxonomyNode(name="deep", checks=[], metrics=[])
    shallow = TaxonomyNode(name="shallow", checks=[], metrics=[])

    breathing.add_child("deep", deep)
    breathing.add_child("shallow", shallow)

    # Create exhalation branch under breathing
    exhalation = TaxonomyNode(name="exhalation", checks=[], metrics=[])
    breathing.add_child("exhalation", exhalation)

    # Cough sub-branch under exhalation
    cough = TaxonomyNode(name="cough", checks=[], metrics=[])
    exhalation.add_child("cough", cough)

    # Sigh under exhalation
    sigh = TaxonomyNode(name="sigh", checks=[], metrics=[])
    exhalation.add_child("sigh", sigh)

    # Create inhalation branch under breathing
    inhalation = TaxonomyNode(name="inhalation", checks=[], metrics=[])
    breathing.add_child("inhalation", inhalation)

    # Inhalation types
    sniff = TaxonomyNode(name="sniff", checks=[], metrics=[])
    gasp = TaxonomyNode(name="gasp", checks=[], metrics=[])

    inhalation.add_child("sniff", sniff)
    inhalation.add_child("gasp", gasp)

    # Create vocalization branch
    vocalization = TaxonomyNode(name="vocalization", checks=[], metrics=[])
    human.add_child("vocalization", vocalization)

    # Speech sub-branch
    speech = TaxonomyNode(name="speech", checks=[], metrics=[])
    vocalization.add_child("speech", speech)

    # Speech types (simplified)
    spontaneous = TaxonomyNode(name="spontaneous", checks=[], metrics=[])
    reading = TaxonomyNode(name="reading", checks=[], metrics=[])

    speech.add_child("spontaneous", spontaneous)
    speech.add_child("reading", reading)

    # Non-speech sub-branch
    non_speech = TaxonomyNode(name="non_speech", checks=[], metrics=[])
    vocalization.add_child("non_speech", non_speech)

    # Non-speech types
    laughter = TaxonomyNode(name="laughter", checks=[], metrics=[])
    crying = TaxonomyNode(name="crying", checks=[], metrics=[])
    humming = TaxonomyNode(name="humming", checks=[], metrics=[])
    throat_clearing = TaxonomyNode(name="throat_clearing", checks=[], metrics=[])

    non_speech.add_child("laughter", laughter)
    non_speech.add_child("crying", crying)
    non_speech.add_child("humming", humming)
    non_speech.add_child("throat_clearing", throat_clearing)

    return root


# Create the global taxonomy tree instance
BIOACOUSTIC_ACTIVITY_TAXONOMY = build_bioacoustic_activity_taxonomy()
