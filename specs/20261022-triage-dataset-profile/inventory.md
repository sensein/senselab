# What moves out of senselab: triage data and config — 2026-10-22

Every packaged file under `src/senselab/audio/workflows/triage/data/` and every B2AI-specific key of
`data/config/default.yaml`, at `d633b989f` (alpha `98c8048b6` + step 1), with where it goes.

**Rule (owner of this plan, 2026-10-22):**
- When in doubt, a value is B2AI-specific and moves to the B2AI profile.
- Where the same value, or a nearby one, is defensible for any dataset, senselab keeps it as a default.
- The B2AI profile still records the value it uses, so B2AI results do not move if senselab's default changes.

| Column | Values |
|---|---|
| Home | **stays**: engine, in senselab. **moves**: the B2AI profile only; senselab has no value. **moves + default**: the B2AI profile records it, and senselab keeps a default. **split**: some keys each way. |
| Absent → | What the loader does when no file is supplied: **raises**, or **()** for an empty result. |
| At import | The file is read when the module is imported, so the read must become lazy before it can move. |

**Research basis (2026-10-22).** A research agent checked the "moves + default" candidates against the
literature and model cards. Its citations are listed in the step-2 PR; they are reported by the agent and not yet
checked. The pattern:
- **Same or nearby default:** values that describe signal physics, physiology or a model's own output.
- **No default:** values computed as a quantile, fence or median of the B2AI corpus, and anything keyed by B2AI
  task names.
- **Speaker-embedding thresholds** carry over only for the same model, revision and window. They degrade on
  children's voices, so a pediatric profile supplies its own.

**Consequence.** The classifier-score routing gates and a few corpus fences have no default, so triage cannot run
without a profile. `dataset_profile.require_dataset_values` checks this at the start of every run (`run_triage`,
`replay_decisions` and `refold_verdict`). It lists every missing value in one error. Acceptance: no profile means
a clear error, not detect mode.

## Data files

| File | Home | Why | Absent → | At import |
|---|---|---|---|---|
| `airway_event_requirements.yaml` | moves | names B2AI tasks; owner listening | raises | |
| `airway_event_tokens.yaml` | moves | r12 consensus transcripts, 62,550 recordings; per-family | raises | |
| `asr_degenerate.yaml` | moves + same default (decoder-loop shape; recogniser behaviour) | measured on 6,000 r18 recordings; describes recogniser behaviour | raises | |
| `b2ai_task_registry/` | moves (step 7, with `task_registry.py`) | vendored from b2aiprep | raises | |
| `background_model.yaml` | stays | task-independent; unfitted or spec-derived | | |
| `background_speech.yaml` | stays | owner choice, mostly unfitted, general | | |
| `breath_pattern.yaml` | split | `measure`, `modulation`, `review` and the `train` physiology (event, period, interval, burst and distance windows) stay (breath anatomy). The `train` prominences and template move + same default, marked heuristic (fitted on 56 owner-listened recordings). `veto.active_fraction_min` and `train.quick_families` move with no default. `veto.speech_words_min` stays. The period window fails for young children (faster than 30/min) | raises | |
| `classifier_ontology/` | stays | HeAR→AudioSet map built from the AudioSet ontology | raises | yes |
| `cohort_conditions/` | moves | conditions the B2AI adult study recruits for | raises | |
| `cohort_stage.yaml` | split | window, hop, quantile and check settings stay. `comparison.cosine_min` and `non_target_voice.cosine_min` (0.398) move + same default, tied to the speaker model and revision. `participant.consistency_cosine_min` (a Tukey fence over r18d), `condition.*`, `shared_nonmatch.fraction_min`, `enrollment.families`, `comparison.target.families` and `non_task.task_syllables` move with no default | raises | |
| `config/default.yaml` | split | see below | raises | yes |
| `cough_pattern.yaml` | moves + same default (`measure`: cough onset physiology) | fitted on 15 owner-listened cough recordings | raises | |
| `decision_evidence.yaml` | stays | the decision table's vocabulary | | |
| `decision_reasons.yaml` | split | the vocabulary keys stay. `reviewed_releases` and `second_opinion_not_free_withholds` move (owner release policy) | raises | |
| `detector_profile/` | moves, no default | the corpus distribution of 175 detectors over 62,547 B2AI recordings. Only the routing-analysis tools read it, not the triage graph; the catalogue is now built on first use instead of at import | raises | yes |
| `disagreement_profile/` | stays | "a declared reporting convention, not a fitted cut" | | |
| `discard_contested.yaml` | moves | owner-label agreement on B2AI | raises | |
| `discard_replaced_task.yaml` | split | kinds and fraction bounds stay. `enabled`, family lists, `free_speech_words_per_s`, `family_median_s` move | raises | |
| `evaluation/` | moves (next to `scripts/evaluate_task_events.py`) | owner listening labels and subject splits; not read by the package | | |
| `function_words_es.yaml` | stays | the closed classes of a Spanish grammar | raises | yes |
| `pii_category_families.yaml` | stays, rebuilt | today it lists the 23 detector labels seen in B2AI r6. Rebuilt from every label Presidio and GLiNER document, mapped to the Safe Harbor categories (`text/tasks/pii_detection/data/safe_harbor.yaml`); profiles may add | raises | |
| `quality_join.yaml` | split | `abut_s`, `interference`, `gate` and `capture` stay. `dropout.min_s` (0.5) and `clip.min_s` (0.02) each rest on one listened example. They move, and senselab defaults them to the detection floors already in config (`min_dropout_ms` 10, `min_clip_run` 3), which sends more spans to review. `faults.shutoff.event_kinds` and `streams.decides` move + same default | raises | |
| `recording_vectors/dictionary.yaml` | stays | output schema | | |
| `recording_vectors/evaluation.yaml` | stays | reporting bounds, unfitted, feed no decision | | |
| `review_page.yaml` | stays | display settings | | |
| `session_background.yaml` | split | all stays except `decides` (owner, 10-10), which moves + default `false` | raises | |
| `speaker_vector_profile/` | moves + same default | fitted extent floor for an offline derivative; no decision reads it | raises | |
| `task_content.yaml` | split | `families`, `any_word_families`, run and overlap settings stay. `open_vocabulary_families` moves | raises | |
| `task_events.yaml` | moves + same default | `local` and `decision` (fitted on owner labels) are general event thresholds: the profile records them and senselab keeps the same values | raises | |
| `task_extent.yaml` | split | physics keys and `completeness` stay. The `enabled` switches move + same default: they turn general algorithms on after the owner's listening rounds. `trim.ddk.exclude_families` moves (default empty). `count_in.gap_max_s` (a protocol and UI matter) and `atypical.*` (B2AI quantiles) move with no default; senselab keeps the quantile method, not the numbers | raises | |
| `task_lexicon/` (story and category word lists) | moves | B2AI stimuli | () | |
| `task_minimum_duration.yaml` | split | the per-family table moves with no default. The global floor moves, and senselab defaults it to 0.5 s, enough for one cough; B2AI keeps 1.0 s | raises | |
| `task_speech.yaml` | split | owner-policy word and gap settings and `voice_task_text` move | raises | |
| `temporal_words.yaml` | stays | the words a date, time or age is made of | | |
| `time_release.yaml` | moves | owner release policy (v7/v9) | raises | |
| `triage_table.yaml` | stays | output schema | | |
| `voice_phonation.yaml` | stays | unfitted; general phonation measures | | |
| `word_timing.yaml` | moves + same default (recogniser word width; the voicing strength equals Praat's default) | measured over 1,500 r18c recordings; recogniser timing | raises | |

## Config keys (`data/config/default.yaml`)

**Stay, as engine.** The general sections, plus three instrument settings:
- the general sections: resample, preemphasis, envelope, floor, spans, clipping, normalization, spectrogram,
  band_profile, ppg, gammatone, yamnet, hear, phonation_spans, phonation, praat_features, voice, branch, quality,
  disruptions, residual, task_audio, report, redaction (except `name_approvals`), second_opinion, and the taxonomy
  structure keys;
- `windows.ast.win_length_s` (10.24 s is AST's input length);
- `windows.ast.label_top_k` and `windows.ast.default_threshold`, which nothing reads ("inert").

**Move + same default.** Chosen by hand during B2AI development, never fitted (git history checked 2026-10-22):

| Key | History |
|---|---|
| `windows.yamnet.default_threshold` / `windows.hear.default_threshold` | null → 0.2 on 09-10 (7a5590250) |
| `windows.*.label_top_k` | 4 since 09-10 |
| `taxonomy.consolidation_floor` | 0.1 → 0.2 on 09-05 (67eda3c3e), in a figure commit, reason not given |

**Move. senselab's default becomes empty or null:**
- **Task lists:** `stimulus.residue.cue_families`, `stimulus.task_lexicons`, `verdict.hint_mismatch_exempt_families`,
  `verdict.model_speaker_families`.
- **Per-task values:** `airway.route_by_task_index` and `verdict.gates.by_family`.
- **The cohort profile name:** `verdict.cohort_conditions`.
- **Per-recording approvals:** `redaction.name_approvals` (already empty).

**Move + same default, because each only adds review.** These are the owner release-policy switches. The code
reads every one with "off" or "empty" as its fallback (`vocabulary.py:1451-1475`), so an empty default would mean
*less* review, which is the unsafe direction for de-identification:
- `verdict.llm_reset_redactions`, `condition_categories`, `person_name_review_flags`, `trim_protected_categories`;
- `llm_second_speaker_flags`, `llm_contradiction_flags`, `llm_instructions_spoken_flags`, `uncomputed_reading_flags`;
- `second_opinion_disagreement_flags`, `second_opinion_confident_yes/no`;
- `pii.name_words_max`.

**Move, with a nearby default:**
- `speech.target_match_cosine`: B2AI keeps 0.231, senselab defaults to 0.25 (SpeechBrain's own verification
  default for this model);
- gate `voice.sustained`: 3 s (CAPE-V asks for vowels held 3–5 s).

**Move + same default:**
- `speech.secondary_exclusive_min_s`, `diarization.streams`;
- the count gates (`speech.lexical`, `transcript_agreement`, `bracketed_event`), which are counting conventions.

**Move with no default:**
- the classifier-score and level gates in `taxonomy.ruleset.gates.*` (YAMNet glide/chant, breath residual,
  cough dB, PPG silent fraction). These are sweeps over the B2AI corpus, and the cough gate is "carried over
  unswept".

## Not in this step

- The 48 task rows and the family sets in Python (`nodes/branches.py`, `routing_analysis/families.py`): step 4.
- `b2ai_hints.py` and `task_registry.py`: step 7.
- The PII policy and task guidance in `text/tasks/pii_detection/data/`: step 6.
