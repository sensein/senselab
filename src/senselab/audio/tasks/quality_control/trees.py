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

    # Create respiration branch
    respiration = TaxonomyNode(name="respiration", checks=[], metrics=[])
    human.add_child("respiration", respiration)

    # Breathing sub-branch
    breathing = TaxonomyNode(name="breathing", checks=[], metrics=[])
    respiration.add_child("breathing", breathing)

    # Breathing types
    quiet = TaxonomyNode(name="quiet", checks=[], metrics=[])
    deep = TaxonomyNode(name="deep", checks=[], metrics=[])
    rapid = TaxonomyNode(name="rapid", checks=[], metrics=[])
    sigh = TaxonomyNode(name="sigh", checks=[], metrics=[])

    breathing.add_child("quiet", quiet)
    breathing.add_child("deep", deep)
    breathing.add_child("rapid", rapid)
    breathing.add_child("sigh", sigh)

    # Exhalation sub-branch
    exhalation = TaxonomyNode(name="exhalation", checks=[], metrics=[])
    respiration.add_child("exhalation", exhalation)

    # Cough sub-branch
    cough = TaxonomyNode(name="cough", checks=[], metrics=[])
    exhalation.add_child("cough", cough)

    # Cough types
    voluntary = TaxonomyNode(name="voluntary", checks=[], metrics=[])
    reflexive = TaxonomyNode(name="reflexive", checks=[], metrics=[])

    cough.add_child("voluntary", voluntary)
    cough.add_child("reflexive", reflexive)

    # Inhalation sub-branch
    inhalation = TaxonomyNode(name="inhalation", checks=[], metrics=[])
    respiration.add_child("inhalation", inhalation)

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

    # Speech types
    spontaneous_speech = TaxonomyNode(name="spontaneous_speech", checks=[], metrics=[])
    read_speech = TaxonomyNode(name="read_speech", checks=[], metrics=[])
    repetitive_speech = TaxonomyNode(name="repetitive_speech", checks=[], metrics=[])
    sustained_phonation = TaxonomyNode(name="sustained_phonation", checks=[], metrics=[])

    speech.add_child("spontaneous_speech", spontaneous_speech)
    speech.add_child("read_speech", read_speech)
    speech.add_child("repetitive_speech", repetitive_speech)
    speech.add_child("sustained_phonation", sustained_phonation)

    # Repetitive speech types
    diadochokinesis = TaxonomyNode(name="diadochokinesis", checks=[], metrics=[])
    counting = TaxonomyNode(name="counting", checks=[], metrics=[])

    repetitive_speech.add_child("diadochokinesis", diadochokinesis)
    repetitive_speech.add_child("counting", counting)

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
