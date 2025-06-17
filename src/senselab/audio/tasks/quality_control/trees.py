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

    # Create exhalation branch under breathing
    exhalation = TaxonomyNode(name="exhalation", checks=[], metrics=[])
    breathing.add_child("exhalation", exhalation)

    # Cough sub-branch under exhalation
    cough = TaxonomyNode(name="cough", checks=[], metrics=[])
    exhalation.add_child("cough", cough)

    # Cough types for Bridge2AI - specific file mappings
    audio_check_3 = TaxonomyNode(name="audio_check_3", checks=[], metrics=[])
    respiration_and_cough_cough_1 = TaxonomyNode(name="respiration_and_cough_cough_1", checks=[], metrics=[])
    respiration_and_cough_cough_2 = TaxonomyNode(name="respiration_and_cough_cough_2", checks=[], metrics=[])
    voluntary_cough = TaxonomyNode(name="voluntary_cough", checks=[], metrics=[])

    cough.add_child("audio_check_3", audio_check_3)
    cough.add_child("respiration_and_cough_cough_1", respiration_and_cough_cough_1)
    cough.add_child("respiration_and_cough_cough_2", respiration_and_cough_cough_2)
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

    # Create exhalation branch under breathing
    exhalation = TaxonomyNode(name="exhalation", checks=[], metrics=[])
    breathing.add_child("exhalation", exhalation)

    # Cough sub-branch under exhalation
    cough = TaxonomyNode(name="cough", checks=[], metrics=[])
    exhalation.add_child("cough", cough)

    # Cough types for Bridge2AI - specific file mappings
    audio_check_3 = TaxonomyNode(name="audio_check_3", checks=[], metrics=[])
    respiration_and_cough_cough_1 = TaxonomyNode(name="respiration_and_cough_cough_1", checks=[], metrics=[])
    respiration_and_cough_cough_2 = TaxonomyNode(name="respiration_and_cough_cough_2", checks=[], metrics=[])
    voluntary_cough = TaxonomyNode(name="voluntary_cough", checks=[], metrics=[])

    cough.add_child("audio_check_3", audio_check_3)
    cough.add_child("respiration_and_cough_cough_1", respiration_and_cough_cough_1)
    cough.add_child("respiration_and_cough_cough_2", respiration_and_cough_cough_2)
    cough.add_child("voluntary_cough", voluntary_cough)

    # Additional breathing patterns for Bridge2AI
    breath_sequences = TaxonomyNode(name="breath_sequences", checks=[], metrics=[])
    breathing.add_child("breath_sequences", breath_sequences)

    # Breath sequence types - specific file mappings
    audio_check_4 = TaxonomyNode(name="audio_check_4", checks=[], metrics=[])
    breath_sounds = TaxonomyNode(name="breath_sounds", checks=[], metrics=[])
    respiration_and_cough_breath_1 = TaxonomyNode(name="respiration_and_cough_breath_1", checks=[], metrics=[])
    respiration_and_cough_breath_2 = TaxonomyNode(name="respiration_and_cough_breath_2", checks=[], metrics=[])
    respiration_and_cough_five_breaths_1 = TaxonomyNode(
        name="respiration_and_cough_five_breaths_1", checks=[], metrics=[]
    )
    respiration_and_cough_five_breaths_2 = TaxonomyNode(
        name="respiration_and_cough_five_breaths_2", checks=[], metrics=[]
    )
    respiration_and_cough_five_breaths_3 = TaxonomyNode(
        name="respiration_and_cough_five_breaths_3", checks=[], metrics=[]
    )
    respiration_and_cough_five_breaths_4 = TaxonomyNode(
        name="respiration_and_cough_five_breaths_4", checks=[], metrics=[]
    )
    respiration_and_cough_three_quick_breaths_1 = TaxonomyNode(
        name="respiration_and_cough_three_quick_breaths_1", checks=[], metrics=[]
    )
    respiration_and_cough_three_quick_breaths_2 = TaxonomyNode(
        name="respiration_and_cough_three_quick_breaths_2", checks=[], metrics=[]
    )

    breath_sequences.add_child("audio_check_4", audio_check_4)
    breath_sequences.add_child("breath_sounds", breath_sounds)
    breath_sequences.add_child("respiration_and_cough_breath_1", respiration_and_cough_breath_1)
    breath_sequences.add_child("respiration_and_cough_breath_2", respiration_and_cough_breath_2)
    breath_sequences.add_child("respiration_and_cough_five_breaths_1", respiration_and_cough_five_breaths_1)
    breath_sequences.add_child("respiration_and_cough_five_breaths_2", respiration_and_cough_five_breaths_2)
    breath_sequences.add_child("respiration_and_cough_five_breaths_3", respiration_and_cough_five_breaths_3)
    breath_sequences.add_child("respiration_and_cough_five_breaths_4", respiration_and_cough_five_breaths_4)
    breath_sequences.add_child(
        "respiration_and_cough_three_quick_breaths_1", respiration_and_cough_three_quick_breaths_1
    )
    breath_sequences.add_child(
        "respiration_and_cough_three_quick_breaths_2", respiration_and_cough_three_quick_breaths_2
    )

    # Create vocalization branch
    vocalization = TaxonomyNode(name="vocalization", checks=[], metrics=[])
    human.add_child("vocalization", vocalization)

    # Speech sub-branch
    speech = TaxonomyNode(name="speech", checks=[], metrics=[])
    vocalization.add_child("speech", speech)

    # Speech types (maintaining base structure)
    unscripted = TaxonomyNode(name="unscripted", checks=[], metrics=[])
    reading = TaxonomyNode(name="reading", checks=[], metrics=[])

    speech.add_child("unscripted", unscripted)
    speech.add_child("reading", reading)

    # Extended unscripted speech types for Bridge2AI
    free_speech = TaxonomyNode(name="free_speech", checks=[], metrics=[])
    picture_description = TaxonomyNode(name="picture_description", checks=[], metrics=[])
    story_recall = TaxonomyNode(name="story_recall", checks=[], metrics=[])
    open_response = TaxonomyNode(name="open_response", checks=[], metrics=[])

    unscripted.add_child("free_speech", free_speech)
    unscripted.add_child("picture_description", picture_description)
    unscripted.add_child("story_recall", story_recall)
    unscripted.add_child("open_response", open_response)

    # Specific free speech files
    free_speech_1 = TaxonomyNode(name="free_speech_1", checks=[], metrics=[])
    free_speech_2 = TaxonomyNode(name="free_speech_2", checks=[], metrics=[])
    free_speech_3 = TaxonomyNode(name="free_speech_3", checks=[], metrics=[])

    free_speech.add_child("free_speech_1", free_speech_1)
    free_speech.add_child("free_speech_2", free_speech_2)
    free_speech.add_child("free_speech_3", free_speech_3)

    # Extended reading types for Bridge2AI
    passage_reading = TaxonomyNode(name="passage_reading", checks=[], metrics=[])
    cape_v_sentences = TaxonomyNode(name="cape_v_sentences", checks=[], metrics=[])

    reading.add_child("passage_reading", passage_reading)
    reading.add_child("cape_v_sentences", cape_v_sentences)

    # Specific passage reading files
    caterpillar_passage = TaxonomyNode(name="caterpillar_passage", checks=[], metrics=[])
    rainbow_passage = TaxonomyNode(name="rainbow_passage", checks=[], metrics=[])

    passage_reading.add_child("caterpillar_passage", caterpillar_passage)
    passage_reading.add_child("rainbow_passage", rainbow_passage)

    # Specific sentence reading files (Cape-V)
    cape_v_sentences_1 = TaxonomyNode(name="cape_v_sentences_1", checks=[], metrics=[])
    cape_v_sentences_2 = TaxonomyNode(name="cape_v_sentences_2", checks=[], metrics=[])
    cape_v_sentences_3 = TaxonomyNode(name="cape_v_sentences_3", checks=[], metrics=[])
    cape_v_sentences_4 = TaxonomyNode(name="cape_v_sentences_4", checks=[], metrics=[])
    cape_v_sentences_5 = TaxonomyNode(name="cape_v_sentences_5", checks=[], metrics=[])
    cape_v_sentences_6 = TaxonomyNode(name="cape_v_sentences_6", checks=[], metrics=[])

    cape_v_sentences.add_child("cape_v_sentences_1", cape_v_sentences_1)
    cape_v_sentences.add_child("cape_v_sentences_2", cape_v_sentences_2)
    cape_v_sentences.add_child("cape_v_sentences_3", cape_v_sentences_3)
    cape_v_sentences.add_child("cape_v_sentences_4", cape_v_sentences_4)
    cape_v_sentences.add_child("cape_v_sentences_5", cape_v_sentences_5)
    cape_v_sentences.add_child("cape_v_sentences_6", cape_v_sentences_6)

    # Specific story reading files
    cinderella_story = TaxonomyNode(name="cinderella_story", checks=[], metrics=[])

    reading.add_child("cinderella_story", cinderella_story)

    # Audio check files under reading
    audio_check_1 = TaxonomyNode(name="audio_check_1", checks=[], metrics=[])
    audio_check_2 = TaxonomyNode(name="audio_check_2", checks=[], metrics=[])

    reading.add_child("audio_check_1", audio_check_1)
    reading.add_child("audio_check_2", audio_check_2)

    # Diadochokinesis directly under speech
    diadochokinesis = TaxonomyNode(name="diadochokinesis", checks=[], metrics=[])
    speech.add_child("diadochokinesis", diadochokinesis)

    # Specific DDK files directly under diadochokinesis
    diadochokinesis_ka = TaxonomyNode(name="diadochokinesis_ka", checks=[], metrics=[])
    diadochokinesis_pa = TaxonomyNode(name="diadochokinesis_pa", checks=[], metrics=[])
    diadochokinesis_ta = TaxonomyNode(name="diadochokinesis_ta", checks=[], metrics=[])
    diadochokinesis_pataka = TaxonomyNode(name="diadochokinesis_pataka", checks=[], metrics=[])
    diadochokinesis_buttercup = TaxonomyNode(name="diadochokinesis_buttercup", checks=[], metrics=[])

    diadochokinesis.add_child("diadochokinesis_ka", diadochokinesis_ka)
    diadochokinesis.add_child("diadochokinesis_pa", diadochokinesis_pa)
    diadochokinesis.add_child("diadochokinesis_ta", diadochokinesis_ta)
    diadochokinesis.add_child("diadochokinesis_pataka", diadochokinesis_pataka)
    diadochokinesis.add_child("diadochokinesis_buttercup", diadochokinesis_buttercup)

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

    # Specific cognitive task files
    animal_fluency = TaxonomyNode(name="animal_fluency", checks=[], metrics=[])
    fluency_tasks.add_child("animal_fluency", animal_fluency)

    productive_vocabulary_1 = TaxonomyNode(name="productive_vocabulary_1", checks=[], metrics=[])
    productive_vocabulary_2 = TaxonomyNode(name="productive_vocabulary_2", checks=[], metrics=[])
    productive_vocabulary_3 = TaxonomyNode(name="productive_vocabulary_3", checks=[], metrics=[])
    productive_vocabulary_4 = TaxonomyNode(name="productive_vocabulary_4", checks=[], metrics=[])
    productive_vocabulary_5 = TaxonomyNode(name="productive_vocabulary_5", checks=[], metrics=[])
    productive_vocabulary_6 = TaxonomyNode(name="productive_vocabulary_6", checks=[], metrics=[])

    vocabulary_tasks.add_child("productive_vocabulary_1", productive_vocabulary_1)
    vocabulary_tasks.add_child("productive_vocabulary_2", productive_vocabulary_2)
    vocabulary_tasks.add_child("productive_vocabulary_3", productive_vocabulary_3)
    vocabulary_tasks.add_child("productive_vocabulary_4", productive_vocabulary_4)
    vocabulary_tasks.add_child("productive_vocabulary_5", productive_vocabulary_5)
    vocabulary_tasks.add_child("productive_vocabulary_6", productive_vocabulary_6)

    word_color_stroop = TaxonomyNode(name="word_color_stroop", checks=[], metrics=[])
    stroop_task.add_child("word_color_stroop", word_color_stroop)

    random_item_generation = TaxonomyNode(name="random_item_generation", checks=[], metrics=[])
    random_generation.add_child("random_item_generation", random_item_generation)

    # Voice assessment
    voice_assessment = TaxonomyNode(name="voice_assessment", checks=[], metrics=[])
    speech.add_child("voice_assessment", voice_assessment)

    # Voice types
    loudness_tasks = TaxonomyNode(name="loudness_tasks", checks=[], metrics=[])
    cape_v_sentences = TaxonomyNode(name="cape_v_sentences", checks=[], metrics=[])

    voice_assessment.add_child("loudness_tasks", loudness_tasks)
    voice_assessment.add_child("cape_v_sentences", cape_v_sentences)

    # Specific voice assessment files
    loudness = TaxonomyNode(name="loudness", checks=[], metrics=[])
    loudness_tasks.add_child("loudness", loudness)

    # Non-speech sub-branch
    non_speech = TaxonomyNode(name="non_speech", checks=[], metrics=[])
    vocalization.add_child("non_speech", non_speech)

    # Additional non-speech for Bridge2AI
    phonation = TaxonomyNode(name="phonation", checks=[], metrics=[])
    non_speech.add_child("phonation", phonation)

    # Phonation types
    maximum_phonation_time = TaxonomyNode(name="maximum_phonation_time", checks=[], metrics=[])
    prolonged_vowel = TaxonomyNode(name="prolonged_vowel", checks=[], metrics=[])

    phonation.add_child("maximum_phonation_time", maximum_phonation_time)
    phonation.add_child("prolonged_vowel", prolonged_vowel)

    # Specific phonation files
    maximum_phonation_time_1 = TaxonomyNode(name="maximum_phonation_time_1", checks=[], metrics=[])
    maximum_phonation_time_2 = TaxonomyNode(name="maximum_phonation_time_2", checks=[], metrics=[])
    maximum_phonation_time_3 = TaxonomyNode(name="maximum_phonation_time_3", checks=[], metrics=[])

    maximum_phonation_time.add_child("maximum_phonation_time_1", maximum_phonation_time_1)
    maximum_phonation_time.add_child("maximum_phonation_time_2", maximum_phonation_time_2)
    maximum_phonation_time.add_child("maximum_phonation_time_3", maximum_phonation_time_3)

    # Glides
    glides = TaxonomyNode(name="glides", checks=[], metrics=[])
    non_speech.add_child("glides", glides)

    # Specific glide files
    glides_high_to_low = TaxonomyNode(name="glides_high_to_low", checks=[], metrics=[])
    glides_low_to_high = TaxonomyNode(name="glides_low_to_high", checks=[], metrics=[])

    glides.add_child("glides_high_to_low", glides_high_to_low)
    glides.add_child("glides_low_to_high", glides_low_to_high)

    return root


# Create the global Bridge2AI voice taxonomy tree instance
BRIDGE2AI_VOICE_TAXONOMY = build_bridge2ai_voice_taxonomy()
BRIDGE2AI_VOICE_TAXONOMY.print_tree()
