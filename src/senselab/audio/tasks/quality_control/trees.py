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


def build_bridge2ai_voice_taxonomy() -> TaxonomyNode:
    """Build the Bridge2AI voice assessment taxonomy tree.

    Returns:
        The root node of the Bridge2AI voice taxonomy tree
    """
    # Create root node
    root = TaxonomyNode(
        name="bridge2ai_voice",
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

    # Cough types for Bridge2AI
    voluntary_cough = TaxonomyNode(name="voluntary_cough", checks=[], metrics=[])
    mixed_cough = TaxonomyNode(name="mixed_cough", checks=[], metrics=[])

    cough.add_child("voluntary_cough", voluntary_cough)
    cough.add_child("mixed_cough", mixed_cough)

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

    # Additional breathing patterns for Bridge2AI
    breath_sequences = TaxonomyNode(name="breath_sequences", checks=[], metrics=[])
    breathing.add_child("breath_sequences", breath_sequences)

    # Breath sequence types
    single_breath = TaxonomyNode(name="single_breath", checks=[], metrics=[])
    multiple_breaths = TaxonomyNode(name="multiple_breaths", checks=[], metrics=[])
    quick_breaths = TaxonomyNode(name="quick_breaths", checks=[], metrics=[])
    breath_sounds = TaxonomyNode(name="breath_sounds", checks=[], metrics=[])

    breath_sequences.add_child("single_breath", single_breath)
    breath_sequences.add_child("multiple_breaths", multiple_breaths)
    breath_sequences.add_child("quick_breaths", quick_breaths)
    breath_sequences.add_child("breath_sounds", breath_sounds)

    # Create vocalization branch
    vocalization = TaxonomyNode(name="vocalization", checks=[], metrics=[])
    human.add_child("vocalization", vocalization)

    # Speech sub-branch
    speech = TaxonomyNode(name="speech", checks=[], metrics=[])
    vocalization.add_child("speech", speech)

    # Speech types (maintaining base structure)
    spontaneous = TaxonomyNode(name="spontaneous", checks=[], metrics=[])
    reading = TaxonomyNode(name="reading", checks=[], metrics=[])

    speech.add_child("spontaneous", spontaneous)
    speech.add_child("reading", reading)

    # Extended spontaneous speech types for Bridge2AI
    free_speech = TaxonomyNode(name="free_speech", checks=[], metrics=[])
    picture_description = TaxonomyNode(name="picture_description", checks=[], metrics=[])
    story_recall = TaxonomyNode(name="story_recall", checks=[], metrics=[])
    open_response = TaxonomyNode(name="open_response", checks=[], metrics=[])

    spontaneous.add_child("free_speech", free_speech)
    spontaneous.add_child("picture_description", picture_description)
    spontaneous.add_child("story_recall", story_recall)
    spontaneous.add_child("open_response", open_response)

    # Extended reading types for Bridge2AI
    passage_reading = TaxonomyNode(name="passage_reading", checks=[], metrics=[])
    sentence_reading = TaxonomyNode(name="sentence_reading", checks=[], metrics=[])
    story_reading = TaxonomyNode(name="story_reading", checks=[], metrics=[])

    reading.add_child("passage_reading", passage_reading)
    reading.add_child("sentence_reading", sentence_reading)
    reading.add_child("story_reading", story_reading)

    # Additional speech tasks for Bridge2AI
    motor_speech = TaxonomyNode(name="motor_speech", checks=[], metrics=[])
    speech.add_child("motor_speech", motor_speech)

    # Diadochokinesis under motor speech
    diadochokinesis = TaxonomyNode(name="diadochokinesis", checks=[], metrics=[])
    motor_speech.add_child("diadochokinesis", diadochokinesis)

    # DDK types
    single_syllable = TaxonomyNode(name="single_syllable", checks=[], metrics=[])
    alternating_motion = TaxonomyNode(name="alternating_motion", checks=[], metrics=[])
    sequential_motion = TaxonomyNode(name="sequential_motion", checks=[], metrics=[])
    word_repetition = TaxonomyNode(name="word_repetition", checks=[], metrics=[])

    diadochokinesis.add_child("single_syllable", single_syllable)
    diadochokinesis.add_child("alternating_motion", alternating_motion)
    diadochokinesis.add_child("sequential_motion", sequential_motion)
    diadochokinesis.add_child("word_repetition", word_repetition)

    # Cognitive tasks
    cognitive_tasks = TaxonomyNode(name="cognitive_tasks", checks=[], metrics=[])
    speech.add_child("cognitive_tasks", cognitive_tasks)

    # Cognitive types
    fluency_tasks = TaxonomyNode(name="fluency_tasks", checks=[], metrics=[])
    vocabulary_tasks = TaxonomyNode(name="vocabulary_tasks", checks=[], metrics=[])
    stroop_task = TaxonomyNode(name="stroop_task", checks=[], metrics=[])
    random_generation = TaxonomyNode(name="random_generation", checks=[], metrics=[])

    cognitive_tasks.add_child("fluency_tasks", fluency_tasks)
    cognitive_tasks.add_child("vocabulary_tasks", vocabulary_tasks)
    cognitive_tasks.add_child("stroop_task", stroop_task)
    cognitive_tasks.add_child("random_generation", random_generation)

    # Voice assessment
    voice_assessment = TaxonomyNode(name="voice_assessment", checks=[], metrics=[])
    speech.add_child("voice_assessment", voice_assessment)

    # Voice types
    loudness_tasks = TaxonomyNode(name="loudness_tasks", checks=[], metrics=[])
    cape_v_sentences = TaxonomyNode(name="cape_v_sentences", checks=[], metrics=[])

    voice_assessment.add_child("loudness_tasks", loudness_tasks)
    voice_assessment.add_child("cape_v_sentences", cape_v_sentences)

    # Audio checks
    audio_checks = TaxonomyNode(name="audio_checks", checks=[], metrics=[])
    speech.add_child("audio_checks", audio_checks)

    # Non-speech sub-branch
    non_speech = TaxonomyNode(name="non_speech", checks=[], metrics=[])
    vocalization.add_child("non_speech", non_speech)

    # Non-speech types (maintaining base structure)
    laughter = TaxonomyNode(name="laughter", checks=[], metrics=[])
    crying = TaxonomyNode(name="crying", checks=[], metrics=[])
    humming = TaxonomyNode(name="humming", checks=[], metrics=[])
    throat_clearing = TaxonomyNode(name="throat_clearing", checks=[], metrics=[])

    non_speech.add_child("laughter", laughter)
    non_speech.add_child("crying", crying)
    non_speech.add_child("humming", humming)
    non_speech.add_child("throat_clearing", throat_clearing)

    # Additional non-speech for Bridge2AI
    phonation = TaxonomyNode(name="phonation", checks=[], metrics=[])
    non_speech.add_child("phonation", phonation)

    # Phonation types
    maximum_phonation_time = TaxonomyNode(name="maximum_phonation_time", checks=[], metrics=[])
    prolonged_vowels = TaxonomyNode(name="prolonged_vowels", checks=[], metrics=[])

    phonation.add_child("maximum_phonation_time", maximum_phonation_time)
    phonation.add_child("prolonged_vowels", prolonged_vowels)

    # Glides
    glides = TaxonomyNode(name="glides", checks=[], metrics=[])
    non_speech.add_child("glides", glides)

    # Glide directions
    high_to_low = TaxonomyNode(name="high_to_low", checks=[], metrics=[])
    low_to_high = TaxonomyNode(name="low_to_high", checks=[], metrics=[])

    glides.add_child("high_to_low", high_to_low)
    glides.add_child("low_to_high", low_to_high)

    return root


# Create the global Bridge2AI voice taxonomy tree instance
BRIDGE2AI_VOICE_TAXONOMY = build_bridge2ai_voice_taxonomy()
BRIDGE2AI_VOICE_TAXONOMY.print_tree()
