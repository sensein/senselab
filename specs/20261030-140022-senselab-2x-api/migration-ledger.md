# Migration ledger (P-39)

One row per 1.x area: the last `alpha` commit whose 1.x changes are reconciled into 2.x. Update it
in the same PR as the reconciliation. A 1.x module may be deleted only when its mark equals
`alpha`.

Base: the design branch was cut from `alpha` at `19834fdd`. No commits had landed on `alpha`
since then as of 2026-10-30. P-39 reconciliation at Phase 2 start (2026-11-01): `origin/alpha` is `22d2c103`, already merged into `2x/phase-2` at `90615bac`. Four commits landed since `19834fdd` (`078471b3`, `1101ca0d`, `e4b7f087`, merge `22d2c103`); all touch only `.github/` CI and `src/tests/scripts/docs_deploy_test.py`, so none touches an area Phase 2 ports (Tasks 13-17) and no port-task rows are added.
P-39 reconciliation at Phase 2 exit (2026-11-02): `origin/alpha` is `36aa9615`, merged into `2x/phase-2`. Of the commits since `22d2c103`, only `48ea265d` (uv-managed interpreters) touches an area Phase 2 ported, and `envs.py` already carries it, so no port-task row is added.
P-39 reconciliation at Phase 3 start (2026-11-03): `origin/alpha` is `d5b1b20a`, merged into `2x/phase-3` by merge commit `06aea676`. Eight commits since `36aa9615` (PR #626, scrub of local paths and personal partitions). Under `src`, `scripts`, `tutorials` only `ef423730`, `666fd4e0`, `781a9af0`, `3eac1c48`, `597bb7fb`, `033e47c5`, `7b8fa8fd` land; none touches an area Phase 2 ported, so no port-task row is added. One merge conflict, in `.pre-commit-config.yaml` (both sides added hooks), kept both; `src/` carries no edit. The new `local-paths` hook flagged five synthetic example paths in the 2.x spec dir, rewritten to the generic `user` account.

P-39 reconciliation at Phase 3 task 21 (2026-11-08): `origin/alpha` is `c4bb902b`, merged into `2x/phase-3`. Two commits since `d5b1b20a`: `3300d8c9` (#631, `drivendata/pasketti-word` routed through the Qwen3-ASR backend) touches only unported 1.x speech_to_text and audio_analysis, so no port-task row is added; `c4bb902b` (#632, `resample_audios` band-limits with a Kaiser sinc filter) is already carried by `signal.resample` (`numpy-sinc`, Kaiser window, 80 dB stop band against #632's >100 dB), so no port-task row is added.

| Area (1.x paths) | Where | Mark | Reconciled changes (commit → 2.x handling) |
|---|---|---|---|
| `src/senselab/audio/tasks/` | on alpha | `c4bb902b` | `3300d8c9` (`speech_to_text/api.py`, `doc.md`: pasketti-word added to `_QWEN_ASR_PREFIXES`) -> none; speech_to_text/Qwen not ported. `c4bb902b` (`preprocessing/preprocessing.py`: `resample_audios` uses a Kaiser sinc) -> already ported: `packages/senselab/src/senselab/signal/resample.py` is Kaiser-windowed (80 dB vs 100 dB) |
| `src/senselab/video/tasks/`, `src/senselab/text/tasks/` | on alpha | `c4bb902b` | — (no change under this area since `19834fdd`) |
| `src/senselab/audio/workflows/audio_analysis/` | on alpha | `c4bb902b` | `3300d8c9` (`stages.py` `stage_asr`: pasketti-word takes the Qwen native-timestamps opt-out) -> none; audio_analysis is not ported |
| `src/senselab/audio/workflows/triage/` | on alpha | `c4bb902b` | `ef423730` (placeholder in `data/breath_pattern.yaml`, a comment only) → none; triage is not ported |
| `src/senselab/utils/` | on alpha | `c4bb902b` | `c4bb902b` (`cached_inference.py` `CACHE_SCHEMA_VERSION` 26 -> 27) -> none; 2.x caches by content identity, not this constant |
| `scripts/` (1.x scripts), `tutorials/` | on alpha | `c4bb902b` | `666fd4e0`, `781a9af0`, `3eac1c48`, `597bb7fb` (new `scripts/check_local_paths.py` and its `local-paths` hook) → kept as merged, passing on the 2.x tree; `ef423730`, `033e47c5`, `7b8fa8fd` (`scripts/triage_review_serve.py` placeholder) → none, 1.x script only |

"On alpha": the module is developed on `alpha`, and its copy under `src/` on the 2.x line is the
source pool it ports from (ADR-009). No 1.x module has moved into `packages/` yet.

**`src/` is 1.x as it stands on `alpha`** (owner decision D-6,
[`decisions-2026-11-01.md`](decisions-2026-11-01.md); ADR-009). Phase 1 rounds edited 97 files under
`src/` (the Task 7 and Task 9 rewires, H9c, H-R42). Those edits are undone by restoring `src/` to the
freeze commit `19834fdd`, and `alpha` is then merged, so each mark above becomes the merged `alpha`
commit. The restore brings back the 1.x modules core replaced before H10 (`file_lock`,
`model_revision`, `prov_store`, `prov_bep028`, `cuda_probe`): core's versions are 2.x code beside
them, not moves, and the 1.x copies leave `src/` only when their last use case moves. **2.x never
edits `src/`:** a file leaves it only by a rename-only `git mv` commit into `packages/`, followed by a
modify commit.

## Moved in Phase 2

| 1.x module (alpha `36aa9615`) | Moved to | Task |
|---|---|---|
| `src/senselab/utils/subprocess_venv.py`, `uv_tools.py`, `venv_lock.py`, `cuda_probe.py` | folded into `packages/senselab-core/src/senselab_core/runtime/envs.py` | 9a1 |
| `src/senselab/utils/vllm_server.py` | `packages/senselab-core/src/senselab_core/runtime/vllm_server.py` | 9b |
| `src/senselab/audio/tasks/preprocessing/` `resample_audios` (torch, speechbrain) | rewritten, not ported: `packages/senselab/src/senselab/signal/resample.py` (numpy Kaiser-windowed sinc, task `signal.resample`, ruling P2-59) | 12 |
| `src/senselab/audio/tasks/speech_to_text/` `huggingface.py`, `token_confidence.py` | `packages/senselab/src/senselab/audio/tasks/speech_to_text/` (backend `transformers`, host; `token_confidence` is the one word-confidence engine, P2-56) | 13 |
| `src/senselab/audio/tasks/speech_to_text/crisperwhisper.py` | `packages/senselab/src/senselab/audio/tasks/speech_to_text/` (`crisperwhisper.py` the isolated worker, `crisperwhisper_backend.py` the declaration, `item_faults.py` the typed faults shared with `huggingface.py`; one backend `crisperwhisper`, `engine: ct2 \| transformers` a backend param) | 14 |
| `src/senselab/text/tasks/pii_detection/` `redaction_review.py`, `redaction_policy.py`, `data/` | rewritten, not ported (moved whole then removed, P2-3): `packages/senselab/src/senselab/text/tasks/pii_spans/` (`engine.py` the prompt and parsing engine, `transformers.py` the isolated worker, `transformers_backend.py` the declaration; task `pii_spans`, Text in, TextSpans out; profile slice `data/profiles/pii_spans.yaml`) | 17a |

Unported in Task 13, each at its `alpha` source `36aa9615:src/senselab/audio/tasks/speech_to_text/`: `huggingface.py`:
`HuggingFaceASR` and its class-level pipeline cache (the model cache and `load` replace them), `_attach_token_confidence`
and `_set_confidence` (word confidence replaces the per-line fields), the `ScriptLine`, `Language`, `DeviceType`, `HFModel`
and `load_hf_resilient` plumbing (kernel types, the catalogue's pinned commit and `ensure_env` replace them);
`api.py` `transcribe_audios` dispatch (never moves, P2-36; the generated `speech_to_text` task replaces it).
`token_confidence.py`: `token_entropy`, `no_speech_prob` and the no-speech token lookup are removed (no 2.x consumer; the 1.x uncertainty loop read them); `torch` is imported lazily and each block gains `tokens` and `token_logprob`.
Open for Task 22a: a mixed batch whose window count differs from its item count gets no word confidence (`_block` returns none), so batch formation owns its alignment.

Unported in Task 14, each at its `alpha` source `36aa9615:src/senselab/audio/tasks/speech_to_text/crisperwhisper.py`: `CrisperWhisperASR` and its `serve_in_venv` call (the worker client replaces them); `_CRISPER_REQUIREMENTS` and the venv name (the committed `crisperwhisper` lock replaces them); the capped-context redecode (`CONTEXT_CAPPED`, `_decode_capped`, `CrisperWhisperDecoderPositionsExceeded`): a position overrun is a typed content fault, since `Transcript` has no `decode_strategy`; `_WordConfidence` (teacher-forced re-scoring), replaced by `token_confidence` (P2-56), so word confidence on the `transformers` engine comes from generate logits and is none where the library's words do not divide the generated tokens; the ct2 torn-entry discard (a conversion is published by one atomic rename); the library-supplied per-word `probability` probing.

Unported in Task 17a, each at its `alpha` source `36aa9615:src/senselab/text/tasks/pii_detection/`: `redaction_review.py` (the review prompt, its REASONING/PROPOSAL/CONDITION/SPEAKERS headings and states, `ReviewResult`, the stdin worker script and its `ReviewWorker` process, `review_transcript`, the transcript block and haystack quoting): the task marks spans and holds no judgement, route or proposal; `redaction_policy.py` and `data/redaction_policy.yaml` (the fold's masking and release by kind: dates, ages, places, kinship) and `data/safe_harbor.yaml` (`safe_harbor_codes`): the policy is the fold's, not a measurement, and the identifier list is condensed into the slice's instruction and categories; `data/task_guidance.yaml` becomes `families` of the slice, its `natures` are dropped. Still in `src/`, unported: `rules.py`, `local_llm.py`, `subprocess_backend.py`, `api.py`, `redaction_review_vllm.py` (Task 17b), `__init__.py`, `doc.md`.

Rewritten in Phase 3 Task 13, not moved (ruling P3-31): `src/senselab/utils/work_pool.py` stays in `src/`, because
`src/senselab/utils/reproduce/{slurm,spec,restricted}.py` and five 1.x tests still import it (R-128). It is the named
source of `packages/senselab-core/src/senselab_core/runtime/execution/queue.py`, rewritten from the ADR-003 design:
items are plan fragments under the fragment digest, claims are `Lease`s (`runtime/locks.py`), claim order comes from
slots in segment directories (`segments/<seq>/slot-<n>`, each made by an exclusive `symlink`, never appended; a full,
resolved segment is retired in place by a `retired` mark) rather than a directory scan, and each item has one
first-writer-wins outcome, `outcome/<digest>`, `done` or `dead`. Not carried over: `Item` and its `payload_digest`,
`WorkPool.create` and its pool settings (lease and heartbeat seconds, attempt budgets, success codes; the profile's
`pool` and `lease` sections replace them), staged claims by rename, `heartbeat` (the lease's renewer replaces it),
`reap` (a stale claim is broken by the next claimer), `fail` exit-code budgets (`Claim.dead` takes a `FaultRecord`;
budgets are the `Pool`'s, P3-33), the separate `done/` and `dead/` trees (one outcome name replaces them), the
host-check failure log, `status`, and `pool_tag` (the label digest replaces it).

Unported in Task 9a1 (ruling P2-3), each at its `alpha` source `36aa9615:src/senselab/utils/<module>`:

- `subprocess_venv.py`: `FileRef`, `call_in_venv`, `_SHIM`, `_pack_value`, `_unpack_value`, `parse_subprocess_result`,
  `stage_portable_audio_io`, `_clean_subprocess_env` (the worker protocol replaces the one-shot IPC);
  `provisioned_venv_dirs`, `record_venv_use`, `_note_venv_use`, `venv_environment` (the env record and
  `EnvHandle.context_fields` replace them); `_resolve_lock` with `compile_lock=True`, `_VenvLockLost` and its retry,
  `SENSELAB_VENV_CACHE`, `SENSELAB_VENV_LOCK_TIMEOUT` (the profile's `env_roots` and `Lease` replace them).
- `venv_lock.py`: `requirements_sha256`, `load_lock`, `lock_path`, `LOCK_DIR`, `Backend`, `discover_backends`,
  `compile_lock`, `_compile`, `resolved_packages`, `torch_only_cuda_packages`, `top_cuda_tag`, `wheel_on_index`,
  `torch_targets`, `IPC_REQUIREMENTS`, `QUARANTINED` (lock checking and compiling; `EnvLock.read` keeps header parsing).
- `uv_tools.py`: `requirements_sha256`, `normalize_name`, `is_torch_family`, `TORCH_NAMES`,
  `TORCH_VARIANT_PREFIXES`, and `find_uv(install=True)`'s `pip install uv` (`Uv.find` never installs).
- `cuda_probe.py`: `SENSELAB_TORCH_INDEX_URL` (the profile's `envs.torch_index` replaces it, P-14, P-18).

Unported in Task 9b, at `36aa9615:src/senselab/utils/vllm_server.py`: `ensure_vllm_venv`, `VLLM_VENV`, `VLLM_VERSION`,
`VLLM_PYTHON`, `VLLM_REQUIREMENTS`, `VLLM_MAX_CUDA` (`ensure_env` over `Env.server("vllm-server")` and
`env_locks/inputs.yaml` replace them, P2-78); `engine_identity` (the attestation replaces it); `utilisation`;
`transcribe` and its multipart helpers; `option_name`, `refused_arg`, `CONTENT_LOGGING_PREFIXES`,
`CONTENT_LOGGING_FLAGS`, `REFUSED_FLAGS` (a server takes only the known settings, each bound as `--flag=value`);
`expected_version` (the reported version is attested, not compared). The start `fcntl.flock` is replaced by `Lease`
(P-37). The 1.x tests `src/tests/utils/vllm_server_test.py` and `reproduce/engine_test.py` stay with 1.x.

1.x importers left under `src/` and `scripts/` (`lock_subprocess_venvs.py`, `extend_*.py`) are 1.x and outside the
2.x checks (D-6).

## Moving a use case

1. Merge `alpha` into the 2.x line, so the source pool carries every 1.x change (P-39). Set the
   area's mark to the merged `alpha` commit.
2. A rename-only commit: `git mv src/<path> packages/<package>/src/<path>` for the modules the use
   case needs, with their tests and data, and nothing else in the commit.
3. A modify commit that ports them onto the kernel and runtime. From here 2.x's checks apply to
   them.
4. When the last use case of a 1.x module has moved, delete what is left of it under `src/` and set
   its row to "moved".

## Triage

Triage is not migrated as a workflow (H-R48, H-R52). Its measurement tasks and its pipelines port
like any other use case; its decision layer is superseded by the flywheel (`roadmap.md`) and is
deleted when its last use case moves. Each module and script below has one class, decided from its
code:

- **task**: ports as a 2.x task (`git mv`, then modify);
- **pipeline**: ports as a generic pipeline, reconfigured per dataset;
- **decision layer**: superseded by the flywheel;
- **split on port**: measurement and decision in one module; the measurement ports as the class
  named, the decision part is decision layer.

Modules under `src/senselab/audio/workflows/triage/`:

| Module | Class | Reason |
|---|---|---|
| `__init__.py` | pipeline | package entry of the graph |
| `b2ai_hints.py` | pipeline | reads a Bridge2AI recording's declared task from its BIDS sidecars: a dataset adapter |
| `background_model.py` | task | stationary floor, mains hum and impulses per recording: measurements |
| `background_speech.py` | task | another voice read off the enhancement residual with YAMNet |
| `breath_pattern.py` | task | breath-band envelope and pattern over the narrowband spectrogram |
| `classifier_ontology.py` | task | maps HeAR labels onto AudioSet nodes; a classifier's label data |
| `cohort.py` | decision layer | sorts reviewer-listed conditions into cohort and other by a study profile |
| `cohort_fold.py` | decision layer | what VERDICT reads off COHORT: grounds and evidence items |
| `cohort_stage.py` | split on port (task) | speaker enrollment and corpus quantiles are measurements; the readings it writes for VERDICT are decision |
| `config.py` | pipeline | loads the workflow's configuration; the fitted values in it are decision layer |
| `consensus.py` | task | consensus word stream over ASR hypotheses |
| `corpus_report.py` | decision layer | counts the decisions of a corpus of runs |
| `cough_pattern.py` | task | cough onsets from the narrowband spectrogram |
| `ddk_task.py` | split on port (task) | syllable and cycle events are measurements; "the decision" is decision layer |
| `decision.py` | decision layer | the decision table's reason vocabulary |
| `decision_tables.py` | decision layer | one row per decision and per evidence item |
| `discard_replaced_task.py` | decision layer | discard rules over fold readings |
| `enrollment.py` | task | one subject's target-speaker vector across recordings |
| `extend.py` | pipeline | read, write and re-export a finished run's store for extension drivers |
| `finding_placement.py` | task | places a detector finding among transcript words |
| `label_membership.py` | task | which labels a classifier window carries (top-K over a floor) |
| `live_evidence.py` | decision layer | the live store read as routing evidence for the ruleset |
| `measure_stats.py` | pipeline | distribution of every measurement over a corpus |
| `nodes/__init__.py` | pipeline | node package |
| `nodes/admit.py` | task | decode, all-zero, constant and changing-bytes checks on input |
| `nodes/airway.py` | split on port (task) | breath and cough detection and alignment are measurements; the branch's task-conformance judgement is decision |
| `nodes/airway_task.py` | task | breath train and cough onsets written as measurements |
| `nodes/background.py` | task | own floor, session floor and background measurements |
| `nodes/branches.py` | split on port (pipeline) | the align/detect dispatch is pipeline structure; expectations and proposals carry branch decisions |
| `nodes/common.py` | split on port (pipeline) | node result shapes and store conventions; the decide/report split encodes the decision layer |
| `nodes/ddk.py` | task | syllable-train events, rate and regularity |
| `nodes/figure.py` | task | draws PREPROCESS and TAXONOMY outputs from the store; decides nothing |
| `nodes/gates.py` | decision layer | gates and their bounds |
| `nodes/preprocess.py` | task | the conditioning pass and every whole-file model (YAMNet, AST, HeAR, diarizer, recognizers) |
| `nodes/quality.py` | split on port (task) | joins background against task spans; its verdict is decision |
| `nodes/redact.py` | task | PII findings padded, merged, masked and verified |
| `nodes/report.py` | split on port (pipeline) | renders a run's store; the decision summary it carries is decision layer |
| `nodes/review.py` | task | LLM reader over releasable transcripts |
| `nodes/routing.py` | decision layer | evaluates the taxonomy ruleset to choose branches |
| `nodes/second_opinion.py` | task | a decision model's answers over the transcript, recorded as readings |
| `nodes/speech.py` | split on port (task) | consensus transcript, speakers, PII and quality are measurements; the branch expectation and mode decision are decision |
| `nodes/taxonomy.py` | task | consolidates classifier scores into measurements; decides nothing |
| `nodes/verdict.py` | decision layer | the graph's only decision |
| `nodes/voice.py` | split on port (task) | sustained phonation proposed from evidence; the branch judgement is decision |
| `quality_join.py` | task | background read against each task span |
| `recording_vectors.py` | pipeline | one row per recording over finished stores; its decision columns are decision layer |
| `replay_diff.py` | decision layer | what re-deciding a run changed |
| `residue.py` | task | the lexical residue: words neither task content nor non-lexical |
| `review_page/__init__.py` | decision layer | review page over decisions and evidence |
| `review_page/page.py` | decision layer | assembles the decision review page |
| `review_page/records.py` | decision layer | one review record per decision |
| `review_page/shards.py` | decision layer | side files of the review page |
| `review_page/spectrogram.py` | task | a quantised spectrogram grid |
| `routing_analysis/__init__.py` | decision layer | scores routing detectors over runs |
| `routing_analysis/detectors.py` | decision layer | candidate routing detectors and their grids |
| `routing_analysis/families.py` | pipeline | groups a BIDS task id into its elicitation family: a dataset adapter |
| `routing_analysis/features.py` | decision layer | routing evidence parsed from stores |
| `routing_analysis/gate_matrix.py` | decision layer | family x gate matrix |
| `routing_analysis/gate_plot.py` | decision layer | draws the gate matrix |
| `routing_analysis/labels.py` | decision layer | labels each routing detector reads |
| `routing_analysis/report.py` | decision layer | scores detectors against reference standards |
| `routing_analysis/ruleset.py` | decision layer | the family taxonomy ruleset |
| `routing_analysis/tables.py` | decision layer | parquet tables of the routing analysis |
| `run.py` | pipeline | drives one recording through the graph over one store |
| `session_background.py` | task | session background read off the speech residual |
| `speaker_share.py` | task | each diarized speaker's seconds in a task extent |
| `speaker_vectors.py` | pipeline | one speaker embedding per subject over finished stores |
| `stimulus.py` | task | consensus words aligned against the declared stimulus |
| `task_audio.py` | task | streams cut to the task extent |
| `task_content.py` | task | which transcript words are misreadings of task events |
| `task_events.py` | task | task events over the background, their evidence and extent |
| `task_extent.py` | task | where a task's extent begins and ends |
| `task_lexicon.py` | task | a task's own vocabulary |
| `task_registry.py` | pipeline | what a Bridge2AI task asks, from the study registry: a dataset adapter |
| `task_speech.py` | task | speech outside the task's own content |
| `task_squim.py` | task | SQUIM over fixed windows of the task extent |
| `triage_table.py` | decision layer | `triage.tsv` over the decision tables |
| `viewer/__init__.py` | pipeline | browser views over the recording vectors |
| `viewer/build.py` | pipeline | inlines the viewer into one HTML file |
| `vocabulary.py` | decision layer | the shared vocabulary and the file-level fold |
| `voice_phonation.py` | task | phonation attempt off the plain, residual and enhanced streams |
| `word_arbitration.py` | task | word timing arbitration across recognizers |
| `data/` | split on port | the b2ai task registry, task lexicon and classifier ontology go with their tasks; config thresholds, detector, disagreement and recording-vector profiles, cohort conditions and decision reasons are decision layer |

Triage scripts under `scripts/`:

| Script | Class | Reason |
|---|---|---|
| `triage_audio.py` | pipeline | runs one recording through the graph |
| `extend_background_floor.py`, `extend_clip_amplitudes.py`, `extend_diarization.py`, `extend_ppg_praat.py`, `extend_reprocessed_outputs.py`, `extend_session_floor.py`, `extend_task_audio.py`, `extend_task_squim.py` | pipeline | add a measurement to finished runs in place |
| `extend_llm_review.py`, `extend_second_opinion.py` | pipeline | run REVIEW or SECOND_OPINION over a finished corpus |
| `extend_cohort.py` | split on port (pipeline) | corpus quantiles and enrollment are measurement; COHORT readings feed VERDICT |
| `extend_quality.py`, `extend_refold.py`, `extend_replay_decisions.py`, `extend_withdraw_clips.py` | decision layer | re-decide or withdraw on fold readings |
| `measure_session_background.py`, `triage_measure_stats.py` | pipeline | measurement distributions over a corpus |
| `triage_recording_vectors.py`, `triage_speaker_vectors.py`, `triage_vectors_viewer.py`, `triage_viewer_fixture.py`, `triage_viewer_measure.mjs` | pipeline | corpus vectors and their viewer |
| `build_classifier_ontology_map.py` | task | generates the classifier-ontology profile |
| `triage_rerender.py` | split on port (pipeline) | re-renders REPORT from stores |
| `select_replay_manifest.py`, `verify_task_squim.py` | pipeline | corpus manifests and a measurement comparison |
| `analyze_routing_evidence.py`, `score_taxonomy_ruleset.py`, `gate_family_matrix.py`, `evaluate_task_events.py`, `triage_corpus_report.py`, `triage_gate_breakdown.py`, `triage_replay_diff.py`, `triage_table.py`, `triage_review_page.py`, `triage_review_serve.py`, `triage_review_cache_backfill.py`, `free_speech_review_page.py` | decision layer | routing, decisions and their review |

## Retired provenance formats

2.x reads only `provenance@3.0` (ADR-010). Two formats are retired, and no converter exists for either (H-R47,
P2-44, P-14).
- 1.x provenance stores (`provenance@1`, including historical `kind` records) remain readable by 1.x (alpha and
  the freeze branch).
- `provenance@2.0` JSONL stores were written by 2.x before ADR-010 was implemented.

Migrating either is re-running the pipeline under 2.x. `ProvStore.read` refuses a file in a run's `prov/` that
declares any format but `provenance@3` with `ProvenanceFormatRefused`, whose message points here.

## Untagged digests in 1.x (H-R20)

The strict form of the tagged-digest data-flow lint (`untagged_digests(..., strict=True)`, deleted in H9e with
R-106), run over every allowlisted module outside `packages/`, reported 28 sites where `json.dumps`, `str`,
`repr` or `format` output reaches a hash call. They are migration debt under ADR-006, not fixed in Phase 1: each
is ported when its module moves (P-39). Line numbers are at `deb84049`. Replacements name the 2.x mechanism:
`identity_digest(<tag>, value)` from `senselab_worker.digest` with a tag registered in `IDENTITY_TAGS` for any
structured value, `file_digest("file@1", path)` for a file as written, `npy_digest("npy@1", array)` for an
array, or a typed identity from `senselab_core.kernel.identity`. There is no public digest of bytes in memory
(H-R36): serialised data is never hashed as a blob. The lint is gone; the table is the list it produced.

An earlier count of 32 included four `dict.update` calls fed by `str()` (`cohort_stage.py:317`,
`recording_vectors.py:1144`, `speaker_vectors.py:479`, `reproduce/source.py:830`). The guard now counts
`.update` only on a receiver a hash constructor made, so they are no longer reported and have no row.

| Site | Function | What it keys | 2.x replacement |
|---|---|---|---|
| `scripts/free_speech_review_page.py:496` | `finding_key` | review-page finding id from stem, categories and extent (`sha1` of a pipe-joined string) | `identity_digest` over `[stem, categories, start, end]` under a registered review-finding tag |
| `scripts/verify_grid_unification.py:232` | `check` | printed fingerprint of word texts and confidences, a diagnostic only | `identity_digest` over the same mapping, or drop: it keys nothing |
| `src/senselab/audio/workflows/audio_analysis/adaptive/policy.py:62` | `load_policy` | `policy_hash` of the merged adaptive policy | `identity_digest` under a registered policy tag; the policy becomes a `Frozen` config with `canonical_digest()` |
| `src/senselab/audio/workflows/audio_analysis/run_config.py:301` | `load_run_config` | `config_hash` of the merged run config (`default=str` stringifies non-JSON values) | `ExecutionContext` config field digested by `identity_digest("context@3", …)`; non-JSON values refused, not stringified |
| `src/senselab/audio/workflows/triage/cohort_stage.py:144` | `_digest` | cohort-stage parameter and input digests (`default=str`) | `identity_digest` under a registered cohort tag; the stage's key becomes a `LogicalKey` |
| `src/senselab/audio/workflows/triage/cohort_stage.py:1715` | `write_quantile_artefact` | sha256 of the artefact file's bytes, written as indented `json.dumps` | key it by `identity_digest` over the record, or hash the written file with `file_digest` |
| `src/senselab/audio/workflows/triage/config.py:203` | `load_triage_config` | `config_hash` of the merged triage config (`default=str`) | as `load_run_config`: one config identity through `identity_digest("context@3", …)` |
| `src/senselab/audio/workflows/triage/review_page/page.py:259` | `index_of` | review-page build id over the page index (`default=str`) | `identity_digest` under a registered review-page tag |
| `src/senselab/audio/workflows/triage/task_audio.py:293` | `cut_key` | cache key of a task-audio cut: input digest, definition, sample range, extras (`default=str`) | `LogicalKey` of the cut node, digested by `identity_digest("logical-key@3", …)` |
| `src/senselab/audio/workflows/triage/task_squim.py:322` | `measure_key` | cache key of a SQUIM measurement: input, extent, windows, weights | `LogicalKey` with the weights as a `ModelIdentity` (`weights@3`) |
| `src/senselab/utils/reproduce/build.py:663` | `build_config_sha256` | key of an sdist build configuration | `identity_digest` under a registered reproduce-build tag |
| `src/senselab/utils/reproduce/env.py:191` | `env_key` | key of a reproduce environment spec | `identity_digest` under a registered reproduce-env tag |
| `src/senselab/utils/reproduce/env.py:263` | `_write_json` | sha256 of a JSON marker file as written (indented `json.dumps`) | write JCS and hash the bytes with `sha256_bytes` |
| `src/senselab/utils/reproduce/env.py:420` | `checks_digest` | digest of the venv checks (`pip_check_expected`, `smoke_imports`) | `identity_digest` under the reproduce-env tag |
| `src/senselab/utils/reproduce/env.py:490` | `kernels_sha256` | digest of an installed kernel tree's listing | a content root over the tree (`content-root@2`) |
| `src/senselab/utils/reproduce/run.py:1736` | `_r_env` | recorded sha of the R session environment | `identity_digest` under a registered run-env tag |
| `src/senselab/utils/reproduce/run.py:2219` | `_record` | `argv_sha` of a step's argument vector in provenance | `identity_digest` under the run tag |
| `src/senselab/utils/reproduce/sandbox.py:413` | `_sha` | sha of a sandbox environment mapping | `identity_digest` under the run-env tag |
| `src/senselab/utils/reproduce/source.py:345` | `patch_set_sha256` | digest of a patch set | `identity_digest` under a registered patch-set tag |
| `src/senselab/utils/subprocess_venv.py:262` | `venv_environment` | `dependencies_digest` over a venv's full distribution listing | done (Task 9a1): `identity_digest("wheel-set@1", …)` as `EnvHandle.wheel_set` in `runtime/envs.py` |
| `src/senselab/utils/tasks/cached_inference.py:353` | `audio_signature` | audio identity: `str(sampling_rate)` into a hash update | `ContentIdentity` of the audio (`audio@4` over the canonical `.npy`, the rate and the channels; `@4` adds the channels, R-181) |
| `src/senselab/utils/tasks/cached_inference.py:355` | `audio_signature` | audio identity: `str(shape)` into the same hash | as the row above: one `audio@4` identity |
| `src/senselab/utils/uv_tools.py:88` | `requirements_sha256` | digest of a requirements set with its header fields | not ported in Task 9a1 (moved and removed); the lock header's tagged digest is Task 9a2's |
| `src/senselab/utils/venv_worker.py:116` | `_digest` | short digest of a worker call payload (`default=str`) | the worker wire's canonical form: `identity_digest("value@2", …)` with non-JSON values refused |
| `src/senselab/utils/vllm_server.py:476` | `_path_label` | path digest shown in an error message instead of the name, a label only | a label, not a key: `identity_digest` under a registered label tag, or `sha256_bytes` of the path's UTF-8 |
| `src/senselab/utils/vllm_server.py:527` | `content_part` | digest of the media directory shown in an error message, a label only | as the row above |
| `src/senselab/utils/work_pool.py:286` | `Item.payload_digest` | idempotence key of a work-pool item | done (Phase 3 Task 13, replaced): a queue item is named by the fragment digest, `identity_digest("plan-fragment@1", …)` of its `plan-fragment@1.0` JSON form, in `runtime/execution/queue.py` |
| `src/senselab/utils/work_pool.py:1315` | `pool_tag` | eight-hex pool name in Slurm job names, from the resolved root path | done (Phase 3 Task 13, replaced): a queue's directory is the label digest of its name, `identity_digest("label@1", …)`, in `runtime/execution/queue.py` |

## Name-based digests in 1.x (H9e, H-R36)

These 1.x sites compute a name-based digest outside the digest module. They are migration debt like the
rows above and are replaced when their module moves. Once `src/` is restored (D-6) they carry no `# noqa`
mark, since 2.x does not lint `src/`.

| Site | Name | What it is | 2.x replacement |
|---|---|---|---|
| `src/senselab/audio/data_structures/audio.py:187` | `uuid.uuid3` | `Audio.generate_id`, a name-based UUID of `str(self.waveform)` (a truncated tensor repr) and the rate | `identity_digest` over the content identity (`audio@4`, which covers the channels since R-181) |
| `src/senselab/video/data_structures/video.py:201-202` | `uuid.uuid3` | `Video.generate_id`, a name-based UUID of `str(self.frames)`, the frame rate and the audio id | a registered identity over the frames' `npy_digest` |
| `src/senselab/utils/constants.py:5` | `uuid.uuid3` | `SENSELAB_NAMESPACE`, the namespace of the two ids above | removed with them |

## Retired wire formats

2.x reads the signal wire formats `recording@2`, `audio@2` and `regular-group@2` (written `@2.1`, ADR-006). Their `@1`
forms, whose source stamps carry no chunk digests, are retired with no converter (P2-44): `from_wire_form` refuses
them with `WireFormatRetired`. Migrating a value is re-reading its source file (`open_audio`) or recomputing it; a
stored output in an `@1` form is recomputed after its store entry is removed (Phase 2 Task 11b).

`metric@1` is retired likewise (Phase 3 Task 3): `Metric` of `metric@2` carries the numerator and denominator of its
ratio. A stored metric output is recomputed. A `benchmark-result@1` `result.json` is refused with
`BenchmarkResultRetired`; the suite is run again to write `benchmark-result@2`.

`transcript@1` is retired likewise (Phase 3 Task 4a): `Transcript` of `transcript@2` declares the extent its spans lie
within. A stored transcript output is recomputed; a hand-built timed transcript gains an `extent`.

- `pii-review-vllm` lock (1.x `redaction_review_vllm`): not ported; superseded by `Env.server("vllm-server")` (Task 17b).

## Retired store formats

2.x reads `result@2` and `execution-context@3` store members, any minor, under `store@5/`. `result@1`,
`execution-context@1` and `execution-context@2` are retired with no converter (P2-44; `@2` by Phase 2 Task 34d, whose
`@3` adds the env license): a store read of a retired context raises `StoreFormatRetired`, naming the member, and
leaves the entry in place. A `STORE_DIR` major holds one member format pair (ADR-006 amendment 2026-11-08), so a
retired `result.json` beside a current `context.json` is corruption: set aside and recomputed. Migrating it is removing the entry and recomputing it under this version. An entry written in a later major, or
a later minor whose context carries fields this version does not know, is another version's: it is skipped, never
quarantined (Phase 2 Task 34a).
