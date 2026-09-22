# What each task asks for, and what every task shares

Generated from `EXPECTATIONS` and the packaged config by `specs/20260817-triage-workflow-dag/settings-table.py`. Regenerate it rather than editing it.

## The instrument settings are global

Every setting below says how a reading is taken and applies identically to every task. `branch:` carries no per-family override and holds no gate.

| instrument setting | value |
| --- | --- |
| `burst_window_ms` | `20.0` |
| `echo_ngram_n` | `3` |
| `effort_split_hz` | `1000.0` |
| `event_min_s` | `0.05` |
| `f0_spread_window_s` | `0.5` |
| `pause_min_s` | `0.25` |
| `peak_prominence_db` | `6.0` |
| `run_gap_max_s` | `0.5` |
| `smoothing_window_s` | `0.05` |
| `trough_return_db` | `3.0` |
| `voiced_strength_min` | `0.45` |

## The gates resolve family, then group, then default, in `verdict.gates`

A gate says what reading is good enough. A layer that names no value for a gate does not apply it, which is why `GLIDE` carries no `f0_spread_max_semitones`. A family overrides its group key by key.

`default` carries nothing — no gate reaches every group. `by_family` carries nothing: no per-family difference has been derived yet.

| gate | `ORDERED_TOKENS` | `FREE_RESPONSE` | `ITEM_LIST` | `SUSTAINED` | `GLIDE` | `EFFORT` | `PER_SENTENCE` | `EVENT_SERIES` | `EVENT_ALTERNATION` | `SOUND_COVERAGE` | `SYLLABLE_TRAIN` | `SYLLABLE_SEQUENCE` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `continuity_min` | — | — | — | `0.5` | — | — | — | — | — | — | — | — |
| `coverage_min` | — | `0.5` | — | — | — | — | — | — | — | — | — | — |
| `dominant_segment_min_fraction` | — | — | — | — | `0.5` | — | — | — | — | — | — | — |
| `echo_overlap_max` | — | `0.5` | — | — | — | — | — | — | — | — | — | — |
| `events_min` | — | — | — | — | — | — | — | `1` | `1` | — | — | — |
| `expected_tokens_matched_min` | `1` | — | — | — | — | — | — | — | — | — | — | — |
| `f0_spread_max_semitones` | — | — | — | `2.0` | — | — | — | — | — | — | — | — |
| `gap_off_task_min_s` | `1.0` | `1.0` | `1.0` | — | — | — | — | `1.0` | `1.0` | `1.0` | — | — |
| `interval_max_s` | — | — | — | — | — | — | — | `2.0` | — | — | — | — |
| `items_min` | — | — | `1` | — | — | — | — | — | — | — | — | — |
| `monotone_tolerance_semitones` | — | — | — | — | `1.0` | — | — | — | — | — | — | — |
| `omissions_max` | `0` | — | — | — | — | — | — | — | — | — | — | — |
| `production_min_s` | — | — | — | `0.5` | `0.5` | — | — | — | — | — | — | — |
| `rate_prominence_min` | — | — | — | — | — | — | — | — | — | — | `2.0` | `2.0` |
| `repeat_overlap_min` | `0.5` | — | — | — | — | — | — | — | — | — | — | — |
| `repetitions_min` | — | — | — | — | — | — | — | — | — | — | `1` | `1` |
| `response_min_s` | — | `0.5` | — | — | — | — | — | — | — | — | — | — |
| `score_min` | — | — | — | — | — | — | — | `0.2` | `0.2` | `0.2` | — | — |
| `train_min_s` | — | — | — | — | — | — | — | — | — | — | `1.0` | `1.0` |
| `verbatim_overlap_max` | — | `0.5` | — | — | — | — | — | — | — | — | — | — |
| `voiced_fraction_min` | — | — | — | `0.5` | `0.5` | — | — | — | — | — | — | — |

## What else differs is the expectation row

**39 distinct settings groups over 48 families.**


### AIRWAY

| pattern | settings | families |
| --- | --- | --- |
| `EVENT_SERIES` | `emit_filler`=True; `label_set`=breath; `required_count`=3 events, from the instruction; `timed_intervals`=True | respiration-and-cough-threequickbreaths, respiration-and-cough-v2-threebreaths |
| `EVENT_SERIES` | `emit_filler`=True; `label_set`=cough; `required_count`=5 events, from the instruction | respiration-and-cough-cough |
| `EVENT_SERIES` | `emit_filler`=True; `label_set`=cough; `unviable`=[['effort_absolute', 'no within-recording contrast and no SPL reference; `hard` is not measurable']] | respiration-and-cough-v2-hardcough |
| `EVENT_ALTERNATION` | `emit_filler`=True; `label_set`=cough; `required_count`=3 events, from the instruction | voluntary-cough |
| `EVENT_SERIES` | `emit_filler`=True; `label_set`=breath; `required_count`=5 events, from the instruction; `route_from_index`=True; `unviable`=[['route', 'the discriminating band | respiration-and-cough-fivebreaths |
| `EVENT_SERIES` | `declared_route`=nose; `emit_filler`=True; `label_set`=breath; `required_count`=3 events, from the instruction; `unviable`=[['route', 'as `fivebreaths`']] | respiration-and-cough-v2-threebreathsnose |
| `EVENT_SERIES` | `declared_route`=mouth; `emit_filler`=True; `label_set`=breath; `required_count`=3 events, from the instruction; `unviable`=[['route', 'as `fivebreaths`']] | respiration-and-cough-v2-threebreathsmouth |
| `SOUND_COVERAGE` | `declared_duration_s`=30.0; `emit_filler`=True; `label_set`=breath | respiration-and-cough-breath |
| `SOUND_COVERAGE` | `declared_duration_s`=20.0; `declared_route`=mouth; `emit_filler`=True; `label_set`=breath; `unviable`=[['route', 'as `fivebreaths`']] | respiration-and-cough-v2-breath |
| `EVENT_SERIES` | `declared_duration_s`=73.0; `declared_route`=mouth; `emit_filler`=True; `label_set`=breath; `relax_s`=60.0; `required_count`=3 events, from the instruction; `un | breath-sounds |

### SPEECH

| pattern | settings | families |
| --- | --- | --- |
| `ORDERED_TOKENS` | `emit_filler`=True; `token_source`=stimulus_text | cape-v-sentences, cape-v-sentences-v2, harvard-sentences-list |
| `FREE_RESPONSE` | `connected`=True; `emit_filler`=True | picture-description, picture-description-option1, picture-description-option2 |
| `ORDERED_TOKENS` | `connected`=True; `emit_filler`=True; `token_source`=stimulus_text | caterpillar-passage, rainbow-passage |
| `FREE_RESPONSE` | `anti_pattern`=verbatim_source; `emit_filler`=True; `token_source`=stimulus_text | story-recall, story-recall-v2 |
| `ITEM_LIST` | `emit_filler`=True; `repetition_from_category`=True; `unviable`=[['category_membership', 'a lexicon or a text embedding, one consumer, no waveform']] | random-item-generation, random-item-generation-v2 |
| `ORDERED_TOKENS` | `declared_duration_s`=75.0; `token_source`=stimulus_text | word-color-stroop |
| `ORDERED_TOKENS` | `emit_filler`=True; `required_count`=3 tokens, from the instruction; `tokens`=['hey', 'hey', 'hey'] | loudness |
| `ORDERED_TOKENS` | `emit_filler`=True; `required_count`=2 tokens, from the instruction; `tokens`=['hey', 'hey'] | loudness-v2 |
| `FREE_RESPONSE` | `anti_pattern`=verbatim_prompt; `emit_filler`=True; `token_source`=stimulus_text | free-speech |
| `FREE_RESPONSE` | `declared_duration_s`=30.0; `emit_filler`=True | free-speech-v2 |
| `FREE_RESPONSE` | `emit_filler`=True; `unviable`=[['source_overlap', '`stimulus_text` is empty on all 258; the source is a physical storybook']] | cinderella-story |
| `FREE_RESPONSE` | `emit_filler`=True; `token_source`=stimulus_text; `unviable`=[['defines_its_cue', 'a lexicon or a text model, branch-local, and no waveform']] | productive-vocabulary |
| `FREE_RESPONSE` | `connected`=True; `declared_duration_s`=30.0; `emit_filler`=True; `token_source`=stimulus_text | open-response-questions |
| `ITEM_LIST` | `declared_duration_s`=60.0; `emit_filler`=True; `unviable`=[['category_membership', 'a lexicon or a text embedding, one consumer, no waveform']] | animal-fluency |
| `SYLLABLE_TRAIN` | `emit_filler`=True; `sequence`=['p', 'aa']; `typical_count`=11 repetitions, measured median | diadochokinesis-pa |
| `SYLLABLE_TRAIN` | `emit_filler`=True; `sequence`=['t', 'aa']; `typical_count`=11 repetitions, measured median | diadochokinesis-ta |
| `SYLLABLE_TRAIN` | `emit_filler`=True; `sequence`=['k', 'aa']; `typical_count`=10 repetitions, measured median | diadochokinesis-ka |
| `SYLLABLE_TRAIN` | `declared_duration_s`=5.0; `emit_filler`=True; `sequence`=['p', 'ah'] | diadochokinesis-v2-puh |
| `SYLLABLE_TRAIN` | `declared_duration_s`=5.0; `emit_filler`=True; `sequence`=['t', 'ah'] | diadochokinesis-v2-tuh |
| `SYLLABLE_TRAIN` | `declared_duration_s`=5.0; `emit_filler`=True; `sequence`=['k', 'ah'] | diadochokinesis-v2-kuh |
| `SYLLABLE_SEQUENCE` | `emit_filler`=True; `sequence`=['p', 'aa', 't', 'aa', 'k', 'aa']; `typical_count`=10 repetitions, measured median | diadochokinesis-pataka |
| `SYLLABLE_SEQUENCE` | `declared_duration_s`=5.0; `emit_filler`=True; `sequence`=['p', 'ah', 't', 'ah', 'k', 'ah'] | diadochokinesis-v2-puhtuhkuh |
| `SYLLABLE_SEQUENCE` | `emit_filler`=True; `sequence`=['b', 'ah', 't', 'er', 'k', 'ah', 'p']; `typical_count`=10 repetitions, measured median | diadochokinesis-buttercup |
| `SYLLABLE_SEQUENCE` | `declared_duration_s`=5.0; `emit_filler`=True; `sequence`=['b', 'ah', 't', 'er', 'k', 'ah', 'p'] | diadochokinesis-v2-buttercup |

### VOICE

| pattern | settings | families |
| --- | --- | --- |
| `GLIDE` | `declared_direction`=down; `emit_filler`=True | glides-high-to-low, high-to-low |
| `SUSTAINED` | `declared_duration_s`=12.0; `emit_filler`=True; `lexical_separator`=True; `token_source`=instructions; `tokens`=['one', 'two', 'three'] | prolonged-vowel |
| `SUSTAINED` | `emit_filler`=True; `expect_inhale`=True; `forbid_lexical`=True | maximum-phonation-time |
| `SUSTAINED` | `emit_filler`=True; `forbid_lexical`=True | maximum-phonation-time-v2 |
| `GLIDE` | `declared_direction`=up; `emit_filler`=True | glides-low-to-high |
