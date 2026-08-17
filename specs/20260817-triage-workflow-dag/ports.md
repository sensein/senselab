# Port tables

The normative interface definition for the triage workflow. Where this file and
[`workflow.nf`](workflow.nf) disagree, this file is the specification. The diagrams in
[`flowchart.md`](flowchart.md) draw exactly these ports; [`design.md`](design.md) argues for them.

## 1. The rules

1. **A task is a thing with named input ports and named output ports.** Whether its body is a
   function call or another workflow is invisible from outside.
2. **A task may read nothing that is not one of its input ports.** No ambient config object, no
   shared mutable context dictionary, no untyped summary bag. This is the rule that the current
   `pass_summary: dict[str, Any]` breaks: eight top-level keys read from nine modules at 33 sites
   (measured by grep over `src/senselab/audio/workflows/audio_analysis/*.py`), none of them declared
   by any signature.
3. **Two kinds of input port.** A *data port* is wired to another task's output port. A *parameter
   port* is wired to one key of the single versioned config,
   `src/senselab/audio/workflows/audio_analysis/data/run_config/default.yaml`. Both are declared;
   neither is optional-by-omission.
4. **Every port is typed**, and where the type is a table its columns are named here or in a
   contract this file points at.
5. **One producer per product name.** Two tasks declaring the same output name is a build error, not
   a merge.
6. **A port with no producer stops the graph.** It does not read as an empty list. §6 lists the ones
   that exist in the code today and read as empty lists instead.
7. **Absence is not a value.** A task whose input port has no product does not run and produces
   nothing. There is no skip flag anywhere in this design.

## 2. Product registry

Every product, its type, and its single producer. `Estimate` is
`senselab.utils.data_structures.estimate.Estimate` — `estimate.py:28`, which already exists and
which nothing in the workflow currently uses.

| product | type | produced by |
| --- | --- | --- |
| `audio_file` | `Path` | workflow input |
| `hints` | `AudioHints` or absent | workflow input |
| `audio_raw` | `Audio` | `decode_audio` |
| `audio_enhanced` | `Audio` | `enhance_audio` |
| `audio_signature` | `str`, sha256 of PCM + rate | `decode_audio` |
| `level_track` | `Series`, LUFS and true-peak dBTP per frame | `level_and_floor` |
| `band_floor` | table: `band_hz_low, band_hz_high, floor_db, status` | `level_and_floor` |
| `clip_track` | `Series`, clipped sample fraction per frame | `level_and_floor` |
| `signal_present` | `Estimate` over {absent, present} | `signal_gate` |
| `speech_posterior` | `Series`, P of voice per model frame, plus `frame_hop_s` | `speech_frame_posterior` |
| `event_posterior` | table: `start, end, label, score`, one row per label per frame | `sound_event_posterior` |
| `voicing_track` | `Series` per measure: F0, HNR, voiced fraction | `voicing_track` |
| `taxonomy_track` | table on the reporting grid: `start, end, p_lexical, p_nonlexical, p_nonvocal, p_silence, n_voters, dispersion` | `taxonomy_fold` |
| `vocal_spans` | `Spans` with `class` in {lexical, nonlexical} | `taxonomy_fold` |
| `content` | `Estimate` per taxonomy class | `content_gate` |
| `hypotheses` | `{model_id: ScriptLine}` | `transcribe` |
| `word_times` | `{model_id: [{word, start, end, conf}]}` | `align_words` |
| `transcript` | `[{text, start, end, confidence, onset_confidence, offset_confidence, member_agreement, n_recognizers}]` | `fuse_words` |
| `transcript_confidence` | `Estimate` | `transcript_gate` |
| `task_match` | `Estimate`, or absent | `task_match_gate` |
| `pii_candidates` | `{detector: [PiiSpan]}` | `pii_scan` |
| `pii` | `Estimate` over {absent, present} | `pii_gate` |
| `pii_spans` | `Spans` with `entity_type, detectors, n_detectors` | `pii_gate` |
| `window_embeddings` | `{embedder: [WindowEmbedding]}` | `window_embeddings` |
| `cluster_structure` | `{k_candidates: [{k, separation}], assignments, dominant_cluster_id, dominant_share}` | `cluster_windows` |
| `diarization` | `{model_id: [ScriptLine]}` | `diarize` |
| `harmonized_speakers` | `{cluster_id: {model_id: label}}` plus per-cluster spans | `harmonize_labels` |
| `overlap_track` | `Series`, P of two or more concurrent voices | `harmonize_labels` |
| `speaker_count` | `Estimate` over integers | `speaker_count_gate` |
| `novelty_track` | `Series`, 1 minus cosine to the dominant centroid, per embedder | `novelty_track` |
| `off_target_track` | `Series` in [0,1] per window, plus the per-voter contributions | `off_target_fold` |
| `off_target` | `Estimate` over {absent, present} | `off_target_gate` |
| `off_target_spans` | `Spans` with `evidence` naming which voters fired | `off_target_gate` |
| `snr_track`, `c50_track`, `bandwidth_track` | `Series` in dB / dB / Hz | `scene_quality_frames` |
| `quality_measures` | table on the reporting grid, one column per measure, in native units | `quality_measures` |
| `quality` | `Estimate` per degradation axis | `degradation_gate` |
| `defect_spans` | `Spans` with `defect` in {clipping, dropout, level} | `defect_spans` |
| `trim_candidates` | `Spans` with `reason` and the product that proposed it | `trim_proposal` |
| `trim_regions` | `Spans` with `reason` and an `Estimate` each | `trim_gate` |
| `ledger` | `{answer_name: Estimate}` plus every span set | `evidence_ledger` |
| `review_flag` | `bool` | `review_flag_gate` |
| `reasons` | `[{answer, arm, estimate}]`, ranked | `review_flag_gate` |
| `candidate_actions` | `[{answer, action, region, predicted_new_sources}]` | `rank_undecided` |
| `stop_reason` | one of DECIDED, IRREDUCIBLE, OSCILLATING, EXHAUSTED | `stop_or_continue` |
| `audio_regions` | `[Audio]`, each with its offset in the parent | `narrow_input` |

## 3. Parameter ports

Each row is a config key, the tasks whose parameter ports read it, and its status **today**. Status
`live` means some production call site reads it now; `dead` means the config declares it and nothing
reads it; `absent` means the design needs the key and no such key exists.

| config key | read by | status today | evidence |
| --- | --- | --- | --- |
| `device` | every inference task | live | `run_config.py:112` |
| `models.enhancement` | `enhance_audio` | live | `default.yaml:155` |
| `models.diarization` | `diarize` | live | `default.yaml:141-143` |
| `models.asr` | `transcribe` | live | `default.yaml:144-147` |
| `models.embeddings` | `window_embeddings` | live | `default.yaml:148-150` |
| `models.ast`, `models.yamnet` | `sound_event_posterior` | live | `default.yaml:151-152` |
| `alignment.aligner`, `.qwen_model`, `.mms_model`, `.language`, `.qwen_native_timestamps` | `align_words` | live | `default.yaml:157-164` |
| `grid.win_length`, `grid.hop_length` | `taxonomy_fold`, `quality_measures` | live | `default.yaml:109-111` |
| `scene.top_k`, `.ast_win_length`, `.ast_hop_length`, `.yamnet_win_length`, `.yamnet_hop_length` | `sound_event_posterior` | live | `default.yaml:166-173` |
| `features.win_length`, `.hop_length` | `voicing_track` | live | `default.yaml:175-177` |
| `embeddings.window_s`, `.hop_s` | `window_embeddings` | live via config, **but the function default disagrees** | `default.yaml:179-181` is 0.5/0.25; `compute.py:101-102` and `:535-536` default to 2.0/1.0; wired at `run_config.py:478-479` and `scripts/analyze_audio.py:747-748` |
| `speaker.same_floor`, `.diff_floor`, `.cluster_cosine_threshold`, `.clustering_algorithm` | `cluster_windows`, `novelty_track` | live | `default.yaml:183-195` |
| `speaker.centroid_min_similarity` | `harmonize_labels` | live | `default.yaml:191` |
| `speaker.supported_threshold` | `speaker_count_gate` | live | `default.yaml:194` |
| `task.type` | `degradation_gate`, `task_match_gate` | live | `default.yaml:197-201` |
| `enhancement.mode` | `enhance_audio` | live | `default.yaml:206-210` |
| `triage.speech_threshold`, `.min_speech_s` | `signal_gate` | live | `default.yaml:212-216` |
| `triage.snr_floor_db`, `.low_snr_fraction` | `enhance_audio` | live, **and declared underived in the file itself** | `default.yaml:218-240` |
| `uncertainty.aggregator` | `taxonomy_fold`, `off_target_fold` | live | `default.yaml:243` |
| `uncertainty.speech_presence_labels` | `taxonomy_fold` | live | `default.yaml:246-253` |
| `linking.frame_speech_threshold`, `.label_mass_threshold`, `.speech_excess_db`, `.lufs_silence`, `.lufs_speech`, `.no_speech_threshold`, `.asr_confidence_pooling`, `.coarse_voter_weight`, `.coarse_window_ratio` | `taxonomy_fold` | live | `default.yaml:268-303` |
| `linking.asr_slot_overlap`, `.asr_slot_mid_tol_s` | `fuse_words` | live only since the F-162 fix | `default.yaml:316-317` |
| `profiles.calibration`, `.detection_margin` | `degradation_gate` | live | `default.yaml:355-360` |
| `rounds.max_rounds` | `stop_or_continue` | live | `default.yaml:365` |
| `rounds.epistemic_tolerance`, `.cycle_window` | `rank_undecided` | **dead** | `RunConfig.rounds_policy` is in `KNOWN_UNREAD`, `run_config_liveness_test.py:66` |
| `quality.analysis_win_length`, `.analysis_hop_length`, `.floor_percentile`, `.max_gain_cap_db` | `scene_quality_frames`, `level_and_floor` | **dead** | `RunConfig.quality_policy` in `KNOWN_UNREAD`, `run_config_liveness_test.py:68`; `run_config.py:196-203` says so in prose |
| `support.min_evidence_spread`, `.evidence_low_threshold`, `.min_low_fraction` | `evidence_ledger` | **dead** | `RunConfig.support_policy` in `KNOWN_UNREAD`, `run_config_liveness_test.py:70` |
| `labelstudio.low_threshold`, `.high_threshold` | `review_flag_gate` | **dead** | `RunConfig.labelstudio_policy` in `KNOWN_UNREAD`, `run_config_liveness_test.py:69` |
| `taxonomy.classes`, `.voter_weights`, `.evidence_floor` | `taxonomy_fold`, `content_gate` | **absent** | no `taxonomy:` section exists |
| `pii.presidio_score_threshold`, `.gliner_threshold`, `.corroboration_min` | `pii_scan`, `pii_gate` | **absent** | the three values are Python defaults at `pii.py:82`, `pii.py:85`, `pii.py:241`; `grep -n pii default.yaml` returns nothing |
| `evidence.<answer>.prior`, `.pseudo_count`, `.floor`, `.population` | every gate | **absent** | no such section; `Estimate` at `estimate.py:28` has the fields and no configured source |
| `off_target.novelty_threshold`, `.min_span_s`, `.dominant_share_floor` | `off_target_gate` | **absent** | no off-target implementation exists at all |
| `mask.nontarget_active_confidence` | not used by this design | **code literal** | `background_mask.py:297` reads it with a `0.5` fallback, and the shipped profile `data/detection_margin/2026-07-29.json` does not contain the key, so 0.5 is the operative value and lives in Python |

**This design adds no new configuration mechanism.** One versioned YAML remains the only surface:
no node registry, no per-task override file, no CLI flags beyond the existing two arguments. The
five `absent` rows are keys in that same file, each of which must ship with a `derivation` block, and
`derivation: unfitted` is an acceptable value where nothing has been fitted. The four `dead` rows are
the reason the design insists that parameter ports be declared: a key nothing reads is
indistinguishable, from the outside, from a key that is working.

## 4. Task port tables

### ADMIT

**`decode_audio`** — inference-adjacent, cached.

| port | dir | kind | type | wired |
| --- | --- | --- | --- | --- |
| `audio_file` | in | data | `Path` | workflow input |
| `device` | in | param | `str` | `cfg.device` |
| `audio_raw` | out | data | `Audio`, mono, resampled | → `level_and_floor`, `enhance_audio`, and every raw-variant task |
| `audio_signature` | out | data | `str` | → every cache key |

**`level_and_floor`** — pure.

| port | dir | kind | type | wired |
| --- | --- | --- | --- | --- |
| `audio_raw` | in | data | `Audio` | `decode_audio.audio_raw` |
| `floor_percentile` | in | param | `float` | `cfg.quality.floor_percentile` — dead today |
| `gain_cap_db` | in | param | `float` | `cfg.quality.max_gain_cap_db` — dead today |
| `level_track` | out | data | `Series` | → `taxonomy_fold`, `quality_measures` |
| `band_floor` | out | data | table | → `taxonomy_fold`, `quality_measures` |
| `clip_track` | out | data | `Series` | → `quality_measures` |

**`signal_gate`** — gate.

| port | dir | kind | type | wired |
| --- | --- | --- | --- | --- |
| `level_track`, `band_floor` | in | data | as above | `level_and_floor` |
| `speech_threshold`, `min_speech_s` | in | param | `float` | `cfg.triage.speech_threshold`, `cfg.triage.min_speech_s` |
| `signal_present` | out | data | `Estimate` | → `evidence_ledger` |
| `audio_usable` | out | gate | `Audio` or empty | → everything downstream; empty ends the run |

**`enhance_audio`** — inference, cached.

| port | dir | kind | type | wired |
| --- | --- | --- | --- | --- |
| `audio_raw` | in | data | `Audio` | `signal_gate.audio_usable` |
| `enhancement_model` | in | param | `str` | `cfg.models.enhancement` |
| `mode` | in | param | `str` | `cfg.enhancement.mode` |
| `snr_floor_db`, `low_snr_fraction` | in | param | `float` | `cfg.triage.*` |
| `audio_enhanced` | out | data | `Audio` | → `transcribe`, `scene_quality_frames` only. **Deliberately not wired to VOICE IDENTITY.** |

### TAXONOMY

**`speech_frame_posterior`** — inference, cached.

| port | dir | kind | type | wired |
| --- | --- | --- | --- | --- |
| `audio_raw` | in | data | `Audio` | `signal_gate.audio_usable` |
| `device` | in | param | `str` | `cfg.device` |
| `speech_posterior` | out | data | `Series` plus `frame_hop_s` | → `taxonomy_fold` |

**`sound_event_posterior`** — inference, cached.

| port | dir | kind | type | wired |
| --- | --- | --- | --- | --- |
| `audio_raw` | in | data | `Audio` | `signal_gate.audio_usable` |
| `ast_model`, `yamnet_model` | in | param | `str` | `cfg.models.ast`, `cfg.models.yamnet` |
| `win_length`, `hop_length`, `top_k` | in | param | `float`, `int` | `cfg.scene.*` |
| `event_posterior` | out | data | table | → `taxonomy_fold`, `off_target_fold` |

**`voicing_track`** — inference or DSP, cached.

| port | dir | kind | type | wired |
| --- | --- | --- | --- | --- |
| `audio_raw` | in | data | `Audio` | `signal_gate.audio_usable` |
| `win_length`, `hop_length` | in | param | `float` | `cfg.features.*` |
| `voicing_track` | out | data | `Series` per measure | → `taxonomy_fold` |

**`taxonomy_fold`** — pure. The one task whose output the whole graph hangs on.

| port | dir | kind | type | wired |
| --- | --- | --- | --- | --- |
| `speech_posterior` | in | data | `Series` | `speech_frame_posterior` |
| `event_posterior` | in | data | table | `sound_event_posterior` |
| `voicing_track` | in | data | `Series` | `voicing_track` |
| `level_track`, `band_floor` | in | data | as above | `level_and_floor` |
| `grid_win_length`, `grid_hop_length` | in | param | `float` | `cfg.grid.*` |
| `speech_presence_labels` | in | param | list | `cfg.uncertainty.speech_presence_labels` |
| `frame_speech_threshold`, `label_mass_threshold`, `speech_excess_db`, `lufs_silence`, `lufs_speech`, `coarse_voter_weight`, `coarse_window_ratio` | in | param | `float` | `cfg.linking.*` |
| `voter_weights`, `classes` | in | param | mapping | `cfg.taxonomy.*` — **absent today** |
| `aggregator` | in | param | `str` | `cfg.uncertainty.aggregator` |
| `taxonomy_track` | out | data | table | → `content_gate`, `task_match_gate`, `off_target_fold`, `trim_proposal` |
| `vocal_spans` | out | data | `Spans` | → `window_embeddings`, `transcript_gate` |

**`content_gate`** — gate.

| port | dir | kind | type | wired |
| --- | --- | --- | --- | --- |
| `taxonomy_track` | in | data | table | `taxonomy_fold` |
| `evidence_floor`, `prior`, `pseudo_count`, `population` | in | param | per class | `cfg.evidence.content.*` — **absent today** |
| `content` | out | data | `Estimate` per class | → `degradation_gate`, `evidence_ledger` |

### SPEECH CONTENT

**`transcribe`** — inference, cached, one call per model in `cfg.models.asr`.

| port | dir | kind | type | wired |
| --- | --- | --- | --- | --- |
| `audio` | in | data | `Audio` | `signal_gate.audio_usable` and `enhance_audio.audio_enhanced` |
| `asr_models` | in | param | list | `cfg.models.asr` |
| `hints.expected_speech` | in | data | text, optional | `hints`, used as a biasing vocabulary |
| `hypotheses` | out | data | `{model: ScriptLine}` | → `align_words` |

The `hints.expected_speech` port is the contextual-biasing path the SOTA review ranks sixth by
impact. There is no equivalent today: nothing in `src/senselab/audio/workflows/` reads `.hints` at
all.

**`align_words`** — inference, cached separately from ASR.

| port | dir | kind | type | wired |
| --- | --- | --- | --- | --- |
| `audio` | in | data | `Audio` | same variant as the hypotheses |
| `hypotheses` | in | data | `{model: ScriptLine}` | `transcribe` |
| `aligner`, `qwen_model`, `mms_model`, `language`, `qwen_native_timestamps` | in | param | — | `cfg.alignment.*` |
| `word_times` | out | data | `{model: [word]}` | → `fuse_words`, `pii_gate` |

**`fuse_words`** — pure.

| port | dir | kind | type | wired |
| --- | --- | --- | --- | --- |
| `word_times` | in | data | `{model: [word]}` | `align_words` |
| `slot_overlap`, `slot_mid_tol_s` | in | param | `float` | `cfg.linking.asr_slot_overlap`, `.asr_slot_mid_tol_s` |
| `transcript` | out | data | word list with confidences | → `transcript_gate`, `task_match_gate`, `pii_scan`, `off_target_fold` |

**`transcript_gate`** — gate.

| port | dir | kind | type | wired |
| --- | --- | --- | --- | --- |
| `transcript` | in | data | word list | `fuse_words` |
| `vocal_spans` | in | data | `Spans` | `taxonomy_fold` |
| `prior`, `pseudo_count`, `floor`, `population` | in | param | — | `cfg.evidence.transcript.*` — absent |
| `transcript_confidence` | out | data | `Estimate` | → `evidence_ledger` |

**`task_match_gate`** — gate. Does not run without hints.

| port | dir | kind | type | wired |
| --- | --- | --- | --- | --- |
| `expected_speech` | in | data | `ExpectedSpeech` | `hints.expected_speech`; **no product, no run** |
| `transcript` | in | data | word list | `fuse_words` |
| `taxonomy_track` | in | data | table | `taxonomy_fold` |
| `task_type` | in | param | `str` or null | `cfg.task.type` |
| `task_match` | out | data | `Estimate` | → `evidence_ledger`, and Arm 3 of the flag |

**`pii_scan`** — inference, cached, one call per detector.

| port | dir | kind | type | wired |
| --- | --- | --- | --- | --- |
| `transcript` | in | data | word list | `fuse_words` |
| `presidio_score_threshold`, `gliner_threshold` | in | param | `float` | `cfg.pii.*` — **absent today; the values are Python defaults at `pii.py:82,85`** |
| `pii_candidates` | out | data | `{detector: [PiiSpan]}` | → `pii_gate` |

**`pii_gate`** — gate.

| port | dir | kind | type | wired |
| --- | --- | --- | --- | --- |
| `pii_candidates` | in | data | as above | `pii_scan` |
| `word_times` | in | data | `{model: [word]}` | `align_words` |
| `corroboration_min` | in | param | `int` | `cfg.pii.corroboration_min` — absent; the `count >= 2` literal is at `pii.py:241` |
| `pii` | out | data | `Estimate` | → `evidence_ledger` |
| `pii_spans` | out | data | `Spans` | → `trim_proposal` |

The `count >= 2` cross-witness rule is exactly an evidence count. It becomes
`Estimate.n_evidence`, and the corroboration minimum becomes that answer's evidence floor rather
than a boolean gate that publishes a `False` indistinguishable from "no detector ran".

### VOICE IDENTITY

**`window_embeddings`** — inference, cached, one call per embedder.

| port | dir | kind | type | wired |
| --- | --- | --- | --- | --- |
| `audio_raw` | in | data | `Audio`, **raw only** | `signal_gate.audio_usable` |
| `vocal_spans` | in | data | `Spans` | `taxonomy_fold` |
| `embeddings_models` | in | param | list | `cfg.models.embeddings` |
| `window_s`, `hop_s` | in | param | `float` | `cfg.embeddings.window_s`, `.hop_s` |
| `window_embeddings` | out | data | `{embedder: [WindowEmbedding]}` | → `cluster_windows`, `harmonize_labels`, `novelty_track` |

Windows are drawn on `vocal_spans`, which includes non-lexical voice. Today the equivalent mask is
`compute._speech_window_mask` (`compute.py:893`), a YAMNet top-1 veto whose own docstring names the
child-voice-as-Music failure mode — F-170.

**`cluster_windows`** — pure.

| port | dir | kind | type | wired |
| --- | --- | --- | --- | --- |
| `window_embeddings` | in | data | as above | `window_embeddings` |
| `cluster_cosine_threshold`, `same_floor`, `diff_floor`, `clustering_algorithm` | in | param | — | `cfg.speaker.*` |
| `cluster_structure` | out | data | mapping | → `speaker_count_gate`, `novelty_track`, published as evidence |

**`diarize`** — inference, cached, one call per diarizer.

| port | dir | kind | type | wired |
| --- | --- | --- | --- | --- |
| `audio_raw` | in | data | `Audio` | `signal_gate.audio_usable` |
| `diarization_models` | in | param | list | `cfg.models.diarization` |
| `diarization` | out | data | `{model: [ScriptLine]}` | → `harmonize_labels` |

**`harmonize_labels`** — pure.

| port | dir | kind | type | wired |
| --- | --- | --- | --- | --- |
| `diarization` | in | data | as above | `diarize` |
| `window_embeddings` | in | data | as above | `window_embeddings` |
| `centroid_min_similarity` | in | param | `float` | `cfg.speaker.centroid_min_similarity` |
| `harmonized_speakers` | out | data | mapping | → `speaker_count_gate` |
| `overlap_track` | out | data | `Series` | → `off_target_fold` |

**`speaker_count_gate`** — gate.

| port | dir | kind | type | wired |
| --- | --- | --- | --- | --- |
| `cluster_structure` | in | data | mapping | `cluster_windows` |
| `harmonized_speakers` | in | data | mapping | `harmonize_labels` |
| `supported_threshold` | in | param | `float` | `cfg.speaker.supported_threshold` |
| `prior`, `pseudo_count`, `floor`, `population` | in | param | — | `cfg.evidence.speaker_count.*` — absent |
| `speaker_count` | out | data | `Estimate` over integers | → `evidence_ledger` |

**`novelty_track`** — pure.

| port | dir | kind | type | wired |
| --- | --- | --- | --- | --- |
| `window_embeddings` | in | data | as above | `window_embeddings` |
| `cluster_structure` | in | data | mapping | `cluster_windows` |
| `novelty_track` | out | data | `Series` per embedder | → `off_target_fold` |

**`off_target_fold`** — pure. Five voters, each declared.

| port | dir | kind | type | wired |
| --- | --- | --- | --- | --- |
| `novelty_track` | in | data | `Series` | `novelty_track` |
| `overlap_track` | in | data | `Series` | `harmonize_labels` |
| `event_posterior` | in | data | table | `sound_event_posterior`, the babble/crowd/chatter subtree |
| `taxonomy_track` | in | data | table | `taxonomy_fold` |
| `transcript` | in | data | word list, optional | `fuse_words`, for read-passage deviation |
| `aggregator` | in | param | `str` | `cfg.uncertainty.aggregator` |
| `off_target_track` | out | data | `Series` plus per-voter columns | → `off_target_gate` |

**`off_target_gate`** — gate.

| port | dir | kind | type | wired |
| --- | --- | --- | --- | --- |
| `off_target_track` | in | data | `Series` | `off_target_fold` |
| `cluster_structure` | in | data | mapping | `cluster_windows`, for `dominant_share` |
| `target_voice` | in | data | `TargetSpeakerEmbedding`, optional | `hints.target_speaker` |
| `novelty_threshold`, `min_span_s`, `dominant_share_floor` | in | param | — | `cfg.off_target.*` — absent |
| `off_target` | out | data | `Estimate` | → `evidence_ledger` |
| `off_target_spans` | out | data | `Spans` | → `trim_proposal` |

### QUALITY and TRIM

**`scene_quality_frames`** — inference, cached.

| port | dir | kind | type | wired |
| --- | --- | --- | --- | --- |
| `audio` | in | data | `Audio`, both variants | `signal_gate`, `enhance_audio` |
| `analysis_win_length`, `analysis_hop_length` | in | param | `float` | `cfg.quality.*` — dead today |
| `snr_track`, `c50_track`, `bandwidth_track` | out | data | `Series` | → `quality_measures` |

**`quality_measures`** — pure. Emits native units only; nothing here is a score.

| port | dir | kind | type | wired |
| --- | --- | --- | --- | --- |
| `snr_track`, `c50_track`, `bandwidth_track` | in | data | `Series` | `scene_quality_frames` |
| `level_track`, `band_floor`, `clip_track` | in | data | — | `level_and_floor` |
| `grid_win_length`, `grid_hop_length` | in | param | `float` | `cfg.grid.*` |
| `quality_measures` | out | data | table | → `degradation_gate`, `defect_spans`, published as evidence |

**`degradation_gate`** — gate.

| port | dir | kind | type | wired |
| --- | --- | --- | --- | --- |
| `quality_measures` | in | data | table | `quality_measures` |
| `content` | in | data | `Estimate` per class | `content_gate` — **selects the anchors** |
| `calibration_profile`, `detection_margin_profile` | in | param | path or name | `cfg.profiles.*` |
| `quality` | out | data | `Estimate` per axis | → `evidence_ledger` |

Today `degradation.scene_degradation` (`degradation.py:129`) takes no task-type or content input at
all, and its anchors are fixed at 25 dB SNR / 30 dB C50 (`degradation.py:33-44`) — F-169. The
`content` port is that finding expressed as a wire.

**`defect_spans`** — pure.

| port | dir | kind | type | wired |
| --- | --- | --- | --- | --- |
| `quality_measures` | in | data | table | `quality_measures` |
| `defect_spans` | out | data | `Spans` | → `trim_proposal`, `evidence_ledger` |

**`trim_proposal`** — pure.

| port | dir | kind | type | wired |
| --- | --- | --- | --- | --- |
| `taxonomy_track` | in | data | table | `taxonomy_fold` |
| `off_target_spans` | in | data | `Spans` | `off_target_gate` |
| `pii_spans` | in | data | `Spans` | `pii_gate` |
| `defect_spans` | in | data | `Spans` | `defect_spans` |
| `trim_candidates` | out | data | `Spans` with `reason` | → `trim_gate` |

**`trim_gate`** — gate.

| port | dir | kind | type | wired |
| --- | --- | --- | --- | --- |
| `trim_candidates` | in | data | `Spans` | `trim_proposal` |
| `prior`, `pseudo_count`, `floor` | in | param | per reason | `cfg.evidence.trim.*` — absent |
| `trim_regions` | out | data | `Spans` with an `Estimate` each | → `evidence_ledger` |

### DECIDE

**`evidence_ledger`** — pure.

| port | dir | kind | type | wired |
| --- | --- | --- | --- | --- |
| `signal_present`, `content`, `transcript_confidence`, `task_match`, `pii`, `speaker_count`, `off_target`, `quality`, `trim_regions` | in | data | `Estimate` each | their gates |
| `pii_spans`, `off_target_spans`, `defect_spans` | in | data | `Spans` | their gates |
| `prior`, `pseudo_count`, `floor`, `population` per answer | in | param | — | `cfg.evidence.*` — absent |
| `ledger` | out | data | mapping | → `review_flag_gate`, `rank_undecided` |

**`review_flag_gate`** — gate.

| port | dir | kind | type | wired |
| --- | --- | --- | --- | --- |
| `ledger` | in | data | mapping | `evidence_ledger` |
| `low_threshold`, `high_threshold` | in | param | `float` | `cfg.labelstudio.*` — dead today |
| `review_flag` | out | data | `bool` | workflow output |
| `reasons` | out | data | ranked list | workflow output |

### REFINE

**`rank_undecided`** — pure.

| port | dir | kind | type | wired |
| --- | --- | --- | --- | --- |
| `ledger` | in | data | mapping | `evidence_ledger` of round *k* |
| `action_history` | in | data | list of executed action sets | previous iteration |
| `epistemic_tolerance`, `cycle_window` | in | param | — | `cfg.rounds.*` — dead today |
| `candidate_actions` | out | data | list | → `stop_or_continue` |

**`stop_or_continue`** — gate. The only task that decides whether there is another round.

| port | dir | kind | type | wired |
| --- | --- | --- | --- | --- |
| `candidate_actions` | in | data | list | `rank_undecided` |
| `budget_remaining` | in | data | scalar | previous iteration |
| `max_rounds` | in | param | `int` | `cfg.rounds.max_rounds` |
| `stop_reason` | out | data | enum or empty | workflow output when non-empty |
| `actions` | out | gate | list or empty | → `narrow_input`; empty ends the loop |

**`narrow_input`** — pure.

| port | dir | kind | type | wired |
| --- | --- | --- | --- | --- |
| `actions` | in | data | list | `stop_or_continue` |
| `audio_raw` | in | data | `Audio` | `signal_gate.audio_usable` |
| `audio_regions` | out | data | `[Audio]` with offsets | → the re-entered tasks |

## 5. Sub-workflow port tables

A sub-workflow's ports are the union of its members' unwired ports. Nothing else is visible.

| sub-workflow | input ports | output ports |
| --- | --- | --- |
| `ADMIT` | `audio_file`; params `device`, `models.enhancement`, `enhancement.mode`, `triage.*`, `quality.floor_percentile`, `quality.max_gain_cap_db` | `audio_raw`, `audio_enhanced`, `audio_signature`, `level_track`, `band_floor`, `clip_track`, `signal_present` |
| `TAXONOMY` | `audio_raw`, `level_track`, `band_floor`; params `models.ast`, `models.yamnet`, `scene.*`, `features.*`, `grid.*`, `linking.*`, `uncertainty.speech_presence_labels`, `uncertainty.aggregator`, `taxonomy.*`, `evidence.content.*` | `taxonomy_track`, `vocal_spans`, `event_posterior`, `content` |
| `SPEECH_CONTENT` | `audio_raw`, `audio_enhanced`, `vocal_spans`, `taxonomy_track`, `hints.expected_speech`; params `models.asr`, `alignment.*`, `linking.asr_slot_*`, `task.type`, `pii.*`, `evidence.transcript.*` | `transcript`, `word_times`, `transcript_confidence`, `task_match`, `pii`, `pii_spans` |
| `VOICE_IDENTITY` | `audio_raw`, `vocal_spans`, `taxonomy_track`, `event_posterior`, `transcript`, `hints.target_speaker`; params `models.diarization`, `models.embeddings`, `embeddings.*`, `speaker.*`, `uncertainty.aggregator`, `off_target.*`, `evidence.speaker_count.*` | `speaker_count`, `off_target`, `off_target_spans`, `cluster_structure`, `overlap_track` |
| `QUALITY` | `audio_raw`, `audio_enhanced`, `level_track`, `band_floor`, `clip_track`, `content`; params `quality.*`, `grid.*`, `profiles.*` | `quality`, `quality_measures`, `defect_spans` |
| `TRIM` | `taxonomy_track`, `off_target_spans`, `pii_spans`, `defect_spans`; params `evidence.trim.*` | `trim_regions` |
| `DECIDE` | every `Estimate` and every span set; params `labelstudio.*`, `evidence.*` | `review_flag`, `reasons`, `ledger` |
| `REFINE` | `ledger`, `budget_remaining`, `action_history`, `audio_raw`; params `rounds.*` | `stop_reason`, `audio_regions`, and the next iteration's `ledger` |

`VOICE_IDENTITY` has no `audio_enhanced` input port. That is the enforcement mechanism for the rule
that off-target detection runs on raw audio: there is no wire, so there is no way to violate it by
accident.

## 6. Ports with no producer, and config keys with no consumer

Both directions of the same defect class. Every row is measured.

### Input ports nothing writes

| consumer | port read | what the producer actually emits | consequence |
| --- | --- | --- | --- |
| `speaker.py:549` | `pass_summary.background_mask.result.regions` | `BackgroundMask.to_json()` (`background_mask.py:152-168`) emits **13 counter keys and no `regions`** | `mask_regions == []` always; `attribution.target_activity_doubt` returns `(None, None)` for every bucket; the `target_free` clear (`speaker.py:557`), the `_VOCAL_ACTIVITY` exemption (`speaker.py:562`) and the `target_activity` voter (`speaker.py:606`) have never fired. F-187. Corroborated in artifacts: `target_activity.parquet` is absent from all three completed runs while the sibling `speaker_assignment.parquet` is present in all three; every `L2/background_mask.json` has exactly 13 keys. |
| `specs/20260728-221507-per-speaker-identity-scene/contracts/background-mask.md:67-80` | the `mask_introspection.json` artifact, which declares a `regions` key | nothing — `grep -rn mask_introspection src/` returns nothing | The contract `speaker.py:549` was written against was never built. This is the mechanism behind F-187, and it is not in the register. |
| `compute.harvest_pass` | `derive_window_clusters` | nothing calls it (`speech_presence_link.py:377`) | `cluster_pass_speakers` (`embeddings.py:97`) is reachable only from inside it, so **the whole window-embed-and-cluster chain is dead in production**, and the `embedding_silhouette` signal its docstring promises is never produced. The in-file comment at `speech_presence_link.py:444` — "`derive_window_clusters` below stays — it is what `compute.harvest_pass` calls" — is false. |
| `Audio.hints` (`audio.py:60`) | `AudioHints` | callers may set it; **no workflow reads it** — `grep` for `.hints` over `src/senselab/audio/workflows/` and `scripts/analyze_audio.py` returns zero hits | `ExpectedSpeech`, `TargetSpeakerEmbedding` and the whole hint mechanism are declared and unconsumed. `audio_hints.py:5` says so itself. |
| `attribution.speaker_assignment_doubt(..., target=)` (`attribution.py:57`) | a target speaker id | `speaker.py` calls it with no `target=` | the "future targeted mode" hook has no caller. |

### Parameter ports whose config key nothing reads

Twelve `RunConfig` fields, listed as `KNOWN_UNREAD` at `run_config_liveness_test.py:65-88`: the
five whole-section mappings `rounds_policy`, `speaker_policy`, `quality_policy`,
`labelstudio_policy`, `support_policy`, and the seven stage booleans `run_diarization`, `run_ast`,
`run_yamnet`, `run_features`, `run_asr`, `run_alignment`, `run_comparisons`. The stage booleans are
a dead predecessor of `skipped_stages`, which is what production actually branches on. The five
mappings mean the YAML keys under `quality:`, `support:`, `labelstudio:` and the two non-`max_rounds`
keys under `rounds:` reach no production call site — they advertise control they do not have.

### Decision constants that are code literals rather than config

| literal | site | why it matters here |
| --- | --- | --- |
| `presidio_score_threshold=0.4` | `pii.py:82` | gates the PII verdict; no `pii:` config section exists |
| `gliner_threshold=0.5` | `pii.py:85` | same |
| `count >= 2` | `pii.py:241` | same, and it is an evidence count wearing a boolean |
| `nontarget_active_confidence` default `0.5` | `background_mask.py:297` | the shipped detection-margin profile does not contain the key, so the literal is the operative value — against CLAUDE.md's rule that thresholds live in `data/` with a derivation |
| `embedding_window_s=2.0`, `embedding_hop_s=1.0` | `compute.py:101-102`, `:535-536` | the config sets 0.5/0.25, so any caller that reaches `compute` without the config silently gets a 4× wider window. A lifted chain is exactly such a caller. |
