# What each task asks for, and what every task shares

Generated from `EXPECTATIONS` and the packaged config by `specs/20260817-triage-workflow-dag/settings-table.py`. Regenerate it rather than editing it.

## The operating points are global

Every gate below applies identically to every task. `branch:` carries no per-family override, and VERDICT's only per-family hook, `conformance_flags_by_family`, ships empty. So a 0.5 s `production_min_s` governs a glide, a sustained vowel, a cough and a DDK train alike.

| operating point | value |
| --- | --- |
| `burst_window_ms` | `20.0` |
| `continuity_min` | `0.5` |
| `coverage_min` | `0.5` |
| `dominant_segment_min_fraction` | `0.5` |
| `echo_ngram_n` | `3` |
| `echo_overlap_max` | `0.5` |
| `effort_split_hz` | `1000.0` |
| `event_min_s` | `0.05` |
| `f0_spread_max_semitones` | `2.0` |
| `f0_spread_window_s` | `0.5` |
| `gap_off_task_min_s` | `1.0` |
| `interval_max_s` | `2.0` |
| `monotone_tolerance_semitones` | `1.0` |
| `pause_min_s` | `0.25` |
| `peak_prominence_db` | `6.0` |
| `production_min_s` | `0.5` |
| `rate_prominence_min` | `2.0` |
| `repeat_overlap_min` | `0.5` |
| `response_min_s` | `0.5` |
| `run_gap_max_s` | `0.5` |
| `score_min` | `0.2` |
| `smoothing_window_s` | `0.05` |
| `train_min_s` | `1.0` |
| `trough_return_db` | `3.0` |
| `verbatim_overlap_max` | `0.5` |
| `voiced_fraction_min` | `0.5` |
| `voiced_strength_min` | `0.45` |

## What differs is the expectation row

**39 distinct settings groups over 48 families.**


### AIRWAY

| pattern | settings | families |
| --- | --- | --- |
| `EVENT_SERIES` | `emit_filler`=True; `expected_event_count`=3; `label_set`=breath; `timed_intervals`=True | respiration-and-cough-threequickbreaths, respiration-and-cough-v2-threebreaths |
| `EVENT_SERIES` | `emit_filler`=True; `expected_event_count`=5; `label_set`=cough | respiration-and-cough-cough |
| `EVENT_SERIES` | `emit_filler`=True; `label_set`=cough; `unviable`=(('effort_absolute', 'no within-recording contrast and no SPL reference; `hard` is not measurable'),) | respiration-and-cough-v2-hardcough |
| `EVENT_ALTERNATION` | `emit_filler`=True; `expected_event_count`=3; `label_set`=cough | voluntary-cough |
| `EVENT_SERIES` | `emit_filler`=True; `expected_event_count`=5; `label_set`=breath; `route_from_index`=True; `unviable`=(('route', 'the discriminating band sits above the 8 kHz c | respiration-and-cough-fivebreaths |
| `EVENT_SERIES` | `declared_route`=nose; `emit_filler`=True; `expected_event_count`=3; `label_set`=breath; `unviable`=(('route', 'as `fivebreaths`'),) | respiration-and-cough-v2-threebreathsnose |
| `EVENT_SERIES` | `declared_route`=mouth; `emit_filler`=True; `expected_event_count`=3; `label_set`=breath; `unviable`=(('route', 'as `fivebreaths`'),) | respiration-and-cough-v2-threebreathsmouth |
| `SOUND_COVERAGE` | `declared_duration_s`=30.0; `emit_filler`=True; `label_set`=breath | respiration-and-cough-breath |
| `SOUND_COVERAGE` | `declared_duration_s`=20.0; `declared_route`=mouth; `emit_filler`=True; `label_set`=breath; `unviable`=(('route', 'as `fivebreaths`'),) | respiration-and-cough-v2-breath |
| `EVENT_SERIES` | `declared_duration_s`=73.0; `declared_route`=mouth; `emit_filler`=True; `expected_event_count`=3; `label_set`=breath; `relax_s`=60.0; `unviable`=(('route', 'as  | breath-sounds |

### SPEECH

| pattern | settings | families |
| --- | --- | --- |
| `ORDERED_TOKENS` | `emit_filler`=True; `token_source`=stimulus_text | cape-v-sentences, cape-v-sentences-v2, harvard-sentences-list |
| `FREE_RESPONSE` | `connected`=True; `emit_filler`=True | picture-description, picture-description-option1, picture-description-option2 |
| `ORDERED_TOKENS` | `connected`=True; `emit_filler`=True; `token_source`=stimulus_text | caterpillar-passage, rainbow-passage |
| `FREE_RESPONSE` | `anti_pattern`=verbatim_source; `emit_filler`=True; `token_source`=stimulus_text | story-recall, story-recall-v2 |
| `ITEM_LIST` | `emit_filler`=True; `repetition_from_category`=True; `unviable`=(('category_membership', 'a lexicon or a text embedding, one consumer, no waveform'),) | random-item-generation, random-item-generation-v2 |
| `ORDERED_TOKENS` | `declared_duration_s`=75.0; `token_source`=stimulus_text | word-color-stroop |
| `ORDERED_TOKENS` | `emit_filler`=True; `expected_event_count`=3; `tokens`=('hey', 'hey', 'hey') | loudness |
| `ORDERED_TOKENS` | `emit_filler`=True; `expected_event_count`=2; `tokens`=('hey', 'hey') | loudness-v2 |
| `FREE_RESPONSE` | `anti_pattern`=verbatim_prompt; `emit_filler`=True; `token_source`=stimulus_text | free-speech |
| `FREE_RESPONSE` | `declared_duration_s`=30.0; `emit_filler`=True | free-speech-v2 |
| `FREE_RESPONSE` | `emit_filler`=True; `unviable`=(('source_overlap', '`stimulus_text` is empty on all 258; the source is a physical storybook'),) | cinderella-story |
| `FREE_RESPONSE` | `emit_filler`=True; `token_source`=stimulus_text; `unviable`=(('defines_its_cue', 'a lexicon or a text model, branch-local, and no waveform'),) | productive-vocabulary |
| `FREE_RESPONSE` | `connected`=True; `declared_duration_s`=30.0; `emit_filler`=True; `token_source`=stimulus_text | open-response-questions |
| `ITEM_LIST` | `declared_duration_s`=60.0; `emit_filler`=True; `unviable`=(('category_membership', 'a lexicon or a text embedding, one consumer, no waveform'),) | animal-fluency |
| `SYLLABLE_TRAIN` | `emit_filler`=True; `expected_event_count`=10; `sequence`=('p', 'aa') | diadochokinesis-pa |
| `SYLLABLE_TRAIN` | `emit_filler`=True; `expected_event_count`=10; `sequence`=('t', 'aa') | diadochokinesis-ta |
| `SYLLABLE_TRAIN` | `emit_filler`=True; `expected_event_count`=10; `sequence`=('k', 'aa') | diadochokinesis-ka |
| `SYLLABLE_TRAIN` | `declared_duration_s`=5.0; `emit_filler`=True; `sequence`=('p', 'ah') | diadochokinesis-v2-puh |
| `SYLLABLE_TRAIN` | `declared_duration_s`=5.0; `emit_filler`=True; `sequence`=('t', 'ah') | diadochokinesis-v2-tuh |
| `SYLLABLE_TRAIN` | `declared_duration_s`=5.0; `emit_filler`=True; `sequence`=('k', 'ah') | diadochokinesis-v2-kuh |
| `SYLLABLE_SEQUENCE` | `emit_filler`=True; `expected_event_count`=30; `sequence`=('p', 'aa', 't', 'aa', 'k', 'aa') | diadochokinesis-pataka |
| `SYLLABLE_SEQUENCE` | `declared_duration_s`=5.0; `emit_filler`=True; `sequence`=('p', 'ah', 't', 'ah', 'k', 'ah') | diadochokinesis-v2-puhtuhkuh |
| `SYLLABLE_SEQUENCE` | `emit_filler`=True; `expected_event_count`=30; `sequence`=('b', 'ah', 't', 'er', 'k', 'ah', 'p') | diadochokinesis-buttercup |
| `SYLLABLE_SEQUENCE` | `declared_duration_s`=5.0; `emit_filler`=True; `sequence`=('b', 'ah', 't', 'er', 'k', 'ah', 'p') | diadochokinesis-v2-buttercup |

### VOICE

| pattern | settings | families |
| --- | --- | --- |
| `GLIDE` | `declared_direction`=down; `emit_filler`=True | glides-high-to-low, high-to-low |
| `SUSTAINED` | `declared_duration_s`=12.0; `emit_filler`=True; `lexical_separator`=True; `token_source`=instructions; `tokens`=('one', 'two', 'three') | prolonged-vowel |
| `SUSTAINED` | `emit_filler`=True; `expect_inhale`=True; `forbid_lexical`=True | maximum-phonation-time |
| `SUSTAINED` | `emit_filler`=True; `forbid_lexical`=True | maximum-phonation-time-v2 |
| `GLIDE` | `declared_direction`=up; `emit_filler`=True | glides-low-to-high |
