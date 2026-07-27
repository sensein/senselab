# Tasks: Uncertainty-driven adaptive analysis workflow

**Input**: Design documents from `/specs/20260723-225523-dynamic-uncertainty-workflow/`
**Prerequisites**: plan.md, spec.md, research.md (D1–D12), data-model.md, contracts/

**Implementation status (2026-07-24)**: all tasks below are implemented on branch
`20260722-175022-scene-quality-utterance` under `src/senselab/audio/workflows/audio_analysis/`
(`votes.py` + `adaptive/`) and `scripts/{analyze_audio,adaptive_loop,make_degradation_suite}.py`.
The loop is **artifact-driven**: `scripts/adaptive_loop.py` runs over a completed analyze_audio run
dir + the content-addressable cache; in-process invocation from analyze_audio itself (the full
`contracts/cli.md` flag set) is tracked as Phase 8 follow-up T040. An independent implementation
audit (2026-07-24) confirmed every checked claim against the code; its remaining gaps are enumerated
in Phase 8.

## Phase 1: Setup

- [X] T001 Create `adaptive/` subpackage skeleton per plan.md (`__init__.py`, module files, `policy/default.yaml`)
- [X] T002 [P] Policy loader with defaults-merge + `policy_hash` (`adaptive/policy.py`)
- [X] T003 [P] Light-import bootstrap for minimal environments (driver-level stub of heavy package `__init__`, `scripts/adaptive_loop.py`)

## Phase 2: Foundational (blocking) — belief store (Phase A of plan.md)

- [X] T004 Vote model + VoteStore with shadow/coexist/purge semantics per contracts/belief-store.md (`adaptive/belief.py`)
- [X] T005 Ingest existing run artifacts: 6 per-pass uncertainty parquets → votes
  (`adaptive/belief.py`); per-model ASR/alignment/diarization JSONs → word streams
  (`adaptive/interventions.load_outcomes_dir` + `adaptive/fusion.collect_word_streams`)
- [X] T006 Re-aggregation from votes via the existing pure aggregators (`aggregate_presence/identity/utterance`), incremental over covered buckets (FR-006)
- [X] T007 Round-1 parity check: re-aggregated values reproduce stored `aggregated_uncertainty` (harvest/aggregate split proof, D8)
- [X] T008 Harvest/aggregate split inside the comparator: NEW pure module `votes.py`
  (`PassHarvest`, `aggregate_pass`, `compute_pass_deltas` — stdlib-light, unit-tested in
  `votes_test.py`) + `compute.py` refactored to `harvest_pass` (model-touching) with
  `compute_uncertainty_axes` as a thin wrapper. Aggregation math moved verbatim; caller-dict
  mutation now opt-out via `mutate_passes=False` (default True preserves the timeline-plot
  contract byte-for-byte). Verify on full env: `uv run pytest src/tests/audio/workflows/audio_analysis/`
- [X] T009 In-process vote-store integration point: `VoteStore.from_harvests(...)` consumes
  `PassHarvest` objects directly (no parquet round-trip); the parquet ingest path remains for
  artifact-driven runs. Round persistence stays with the loop driver by design.

## Phase 3: US1 — Cost-aware triage & conditional processing (P1)

- [X] T010 [US1] Triage round 0: pure decision module (`adaptive/triage.py` — frame-posterior speech
  gate at ~100 ms aggregation per SPEECH_PRESENCE_CERTAINTY_ANALYSIS.md, Brouhaha SNR with
  posterior-masked DSP fallback) + `run_triage` in `scripts/analyze_audio.py`; validated live with
  segmentation-3.0 on 4 cases (clean/conversation/silent/noise-injected)
- [X] T011 [US1] `--enhancement {auto,always,never}` + `--triage-*` thresholds + no-speech early exit
  (skips diarization/ASR/alignment/PPG, `run_state: "no_speech"`, `triage.json` provenance) in
  `scripts/analyze_audio.py`; `--no-enhancement` kept as alias; default `always` preserves golden compat
- [X] T012 [US1] `run_pass` decomposed into `_stage_{diarization,scene,features,asr,alignment,ppg}`
  (pure code motion; task names/params — hence cache keys — byte-identical). NOTE: any edit to
  `scripts/analyze_audio.py` rotates `wrapper_version_hash` (whole-file sha256 by design), so the
  first post-merge run re-populates the cache once.
- [X] T013 [US1] Budget ledger with light/medium/heavy classes, per-run caps, deferral → `next_actions` (FR-018) (`adaptive/policy.py`)

## Phase 4: US2 — Targeted re-processing (P1) 🎯 prototype MVP

- [X] T014 [US2] Region proposal: seed ≥ θ_high, expand ≥ θ_low, gap-merge, pad, grid-quantize, top-N by uncertainty mass (FR-010) (`adaptive/regions.py`)
- [X] T015 [US2] Deterministic policy engine: trigger→rank→admit within budget, stable total order (FR-011, FR-025) (`adaptive/policy.py`)
- [X] T016 [US2] U2 reserve escalation via **cache replay** (policy `reserve_asr_models`, default
  whisper-large-v3-turbo, replayed from the content-addressable cache; re-harvest with
  `harvest_utterance_votes`, region-scoped votes, family weights) (`adaptive/interventions.py`)
- [X] T017 [US2] Region-scope shadowing of file-scope votes from same (model, stream) on covered buckets (D5)
- [X] T018 [US2] U1 region re-ASR with live backends on cropped audio (`adaptive/audio_io.py` + `adaptive/backends.py`; HF whisper pipeline path with policy `u1_asr_models`; enhanced stream regenerated on demand via SepFormer with recorded raw fallback). Senselab-native backend routing (subprocess venvs) remains follow-up.
- [X] T019 [US2] Convergence marks: converged / irreducible / budget_exhausted + ε-monotonicity + max-region-rounds (FR-017) (`adaptive/convergence.py`)
- [X] T020 [US2] `iterations.json` decision log incl. deferred/blocked entries with trigger values (FR-020)

## Phase 5: US3 — Cross-signal repair (P2)

- [X] T021 [US3] S1 stream election per region with recorded scores; enhancement-artifact guard degraded to available evidence (FR-015) (`adaptive/interventions.py`)
- [X] T022 [US3] P3 hallucination adjudication over existing evidence (available indicators: native word confidence, source masses, presence p_voice); purge semantics on both axes (C10)
- [X] T023 [US3] C9 missed-speech correction vote (`adjudicator/missed_speech`) where phonetic/text evidence contradicts low p_voice
- [X] T024 [US3] I1 boundary refinement (`adaptive/identity_repair.py`): consensus adjacent-cosine change-point trajectory from stored per-window embeddings (live fine-hop re-embedding available in `backends.embed_windows` for full envs); boundary-confidence from prominence
- [X] T024a [US3] I2 re-cluster: change-point + diar-boundary segmentation, p_voice-weighted pooling, deterministic average-linkage cosine clustering, cross-model co-association consensus; refined clusters drive fusion speaker attribution + `final/diarization.json`; per-bucket `__cross_diar_label_disagreement__` recomputed with the new voter
- [X] T025 [US3] I4 overlap posterior via per-class segmentation-3.0 (`backends.overlap_posteriors`) — code-complete; gated-model validation pending (requires HF token; guards to `next_actions` otherwise)
- [X] T026 [US3] Aleatoric floor = max(quality, overlap posterior) in `belief._decompose` (overlap term populated by I4 when available)

## Phase 6: US4 — Robust conclusion (P2)

- [X] T027 [US4] Word-stream extraction from all ASR sources (native chunks + MMS/Qwen alignment words) (`adaptive/fusion.py`)
- [X] T028 [US4] Time-aligned word-slot voting with family weights + confidences; alternates below margin (FR-021, D9)
- [X] T029 [US4] Speaker attribution from unified identity clusters at word midpoint; `final/transcript.json` + segments rollup
- [X] T030 [US4] `final/diarization.json`, `final/presence.parquet`, `final/convergence.json` (FR-022)
- [X] T031 [US4] U3 consensus re-alignment (`backends.consensus_align` via torchaudio MMS_FA;
  SIGALRM-guarded with policy `fusion.consensus_alignment{,_timeout_s}`; fallback = weighted member
  timestamps recorded in `transcript.json.timestamps`). Guard validated live (timeout path);
  full alignment runs after the one-time bundle download on a full env.
- [X] T032 [US4] `adaptive/ls_final.py`: `final__consensus_transcript(+__text)`, `final__diarization`,
  `final__presence` LS tracks (additive copies under `final/`) + `disagreements_resolved.json`
  (round-1 entries annotated with final status, Δ, and touching intervention ids). Validated on run14
  (21 word regions, 100 resolved entries).
- [X] T033 [US4] Calibration mechanism: `fusion.load_calibrator` (logistic / piecewise profiles),
  policy `calibration_profile`, `calibrated: true` flag path; unit-tested. Fitting the profile itself
  remains with the synthetic harness (scene-quality-utterance US5).

## Phase 7: Validation & evaluation

- [X] T034 [P] Ground-truth evaluation harness vs Label Studio export (presence acc, transcript WER fused-vs-per-model, cluster↔speaker mapping, boundary-uncertainty check, untranscribed-region check) (`adaptive/evaluate.py`)
- [X] T035 [P] End-to-end prototype run on `audio_48khz_mono_16bits` run dir + `updated-label-a7a37522.json`
- [X] T036 Unit tests for pure parts: shadowing, region proposal, plan ordering determinism, fusion voting (`src/tests/audio/workflows/audio_analysis/adaptive/adaptive_prototype_test.py`)
- [X] T036a Visual round-by-round timeline `final/timeline.png` — GT vs presence/identity/utterance (round-1 vs final overlay), regions, fired interventions with Δ, irreducible hatching, confidence-colored fused words (`adaptive/plot.py`)
- [X] T037 Determinism e2e (`adaptive_e2e_test.py::test_t037_determinism_byte_identical`,
  env-gated on `SENSELAB_ADAPTIVE_E2E_RUN_DIR`; hermetic — U3 network path disabled) — PASSING
  against the reference run dir (SC-004).
- [X] T038 Golden-compat harness (`test_t038_golden_compat_preexisting_artifacts`, env-gated on
  `SENSELAB_GOLDEN_RUN_DIR`/`SENSELAB_CANDIDATE_RUN_DIR`): value-equality over the 9 uncertainty
  parquets + per-task result payloads (SC-005). Requires two full-pipeline runs → execute on GPU/Mac.
- [X] T039 Degradation suite: `scripts/make_degradation_suite.py` (noise/clip/lowpass/silence/music
  variants + injected-span manifest; suite generated under `artifacts/degradation_suite/`) +
  `test_t039_injected_spans_attacked_or_explained` (SC-001 ≥ 70% attacked-or-explained; env-gated).
  Full-pipeline variant runs execute on GPU/Mac.

## Phase 8: Follow-ups from the 2026-07-24 implementation audit

- [x] T040 In-process adaptive integration in `scripts/analyze_audio.py` per contracts/cli.md:
  `--max-rounds/--policy/--budget-*/--max-region-rounds/--region-top-n/--reserve-asr-models/`
  `--enable-overlap-separation/--no-adaptive-outputs`, in-run `rounds/` + `final/` emission via
  `VoteStore.from_harvests`, `summary.json` `adaptive` block, and the `--skip comparisons` warning.
  - **Implemented 2026-07-27.** All nine flags land; `--enable-overlap-separation` maps to the
    *shipped* `rules.I4_overlap_detection` (contracts/cli.md calls it a "v2 U4 rule, off by default",
    but no U4 rule exists and the packaged policy already enables I4 — so the flag only overrides a
    policy that disabled it). CLI overrides beat `--policy`, and `policy_hash` is recomputed after
    merging so two runs differing only by `--budget-heavy` don't claim the same hash.
    Harvests reach the loop via a new `harvests_out=` out-parameter on
    `compute_uncertainty_axes` (an out-parameter, not a fourth return value, so no existing caller's
    tuple arity changes). **Golden-compat verified**: `--max-rounds 1` vs `--no-adaptive-outputs`
    produced byte-identical uncertainty parquets (3/3), an identical LS bundle, and identical
    `summary.json` keys and values; `disagreements.json` differed only in `generated_at`.
- [x] T041 `P2_fine_posteriors` rule (fine-hop posterior re-analysis on crops) — also unblocks I4's
  contract dependency ("else fires P2 first").
  - **Implemented 2026-07-27.** Registered ahead of I4 so the planner can satisfy I4's
    "else fires P2 first" dependency; reuses `backends.overlap_posteriors`, which already returns the
    `speech` track alongside `overlap` and runs on the crop. Replacement presence votes enter at
    `scope=region:<id>` (superseding, not deleting, the coarse round-1 voters) and `overlap_posterior`
    is written on covered rows, which is what lets I4 then run "light (reuses P2 output)".
  - **Two bugs the live run caught that unit tests did not.** (1) The trigger read vote payloads off
    the belief row, but rows only carry `contributing_sources` (names) — payloads live in the store,
    so `coarse` was invisible and the trigger never fired. Now reads `store.active_votes(...)`; the
    test fixture was rewritten to build a real `VoteStore` so a fabricated row shape cannot hide this
    again. (2) `frame_instability` never reached `row_meta`, making the second trigger branch dead; it
    is now plumbed through `VoteStore.from_harvests` (the artifact path still lacks it — parquet has
    no such column).
  - **NOT observed firing end-to-end**, and the reason is a threshold boundary rather than a defect:
    presence `aggregated_uncertainty` is a decisiveness measure (`1 − |2p−1|`) that maxes at 0.554
    even on the noise degradation fixture, so no presence region is seeded at the default
    `theta_high: 0.66` (verified: 0 regions at 0.66, 1 at 0.30). With the threshold lowered the region
    appears and the trigger evaluates, but `coarse_share` lands at **0.4967** — a hair under the
    contract's `≥ 0.5`. Retuning either `theta_high` for the presence axis or the coarse-share
    threshold is a policy call. **Decided 2026-07-27: defer.** Tuning against the single annotated
    clip available would fit noise; set it against a benchmark set instead. It is adjustable today
    via `--policy` (`thresholds.theta_high`) with no code change — see prototype-results.md
    "Decision on `theta_high`".
- [ ] T042 Final-output schema completion vs contracts/final-outputs.md: `final/diarization.rttm`,
  `diarization.json` `member_labels`/`overlap`, `transcript.json.language`, presence.parquet contract
  columns (`presence_confidence`, `elected_stream`, `overlap_posterior`).
- [x] T043 Execute T038 (golden vs candidate full-pipeline runs) and T039 (degradation-suite pipeline
  runs) on a GPU/Mac environment; record results in prototype-results.md.
  - **Executed 2026-07-27 on macOS ARM64** — see prototype-results.md "T043" for the tables.
    T039: SC-001 **5/5 (100%)**, but 4 of 5 pass via *explained* (irreducible) rather than *improved*;
    only `silence` reduced uncertainty. Region proposal needed `theta_high: 0.30` via `--policy`
    because presence uncertainty peaks at 0.554 (see the T041 note). T038: all **9** parquets
    value-equal at `atol=1e-12` plus identical ASR/diarization result payloads.
  - **The T038 harness itself was broken and could never have passed**: it globbed
    `rglob("uncertainty/*.parquet")`, which cannot match the cross-pass deltas at
    `uncertainty/raw_vs_enhanced/<axis>.parquet`, so its own `assert checked >= 9` was unreachable.
    Fixed. Second task in this spec marked complete that had never actually been run.
- [x] T044 Exercise `VoteStore.from_harvests` (unit test + first in-process caller, with T040).
- [x] T045 Align contracts/interventions.md with implementation: U2 cost class (medium in code vs
  heavy in contract) + family-majority guard; document `I2_recluster` as a catalog addition; U3 as a
  fusion-stage step rather than a RULES entry; `max_region_rounds` enforced via convergence marks.
  - **Completed 2026-07-27.** All four listed items were already recorded in the contract's
    "Implementation notes" section (2026-07-24) — but the *per-rule sections still contradicted it*:
    `## U2_reserve_escalation` said "Cost: heavy" and listed the family-majority guard as if it
    existed, so a reader who didn't scroll to the footnote got the wrong answer. Corrected inline at
    the source. Also added two discrepancies the earlier pass missed: **P2** is implemented but
    unreachable at the default `theta_high` (⚠️ noted with the measured 0.554 ceiling), and **U4 has
    no implementation** — `--enable-overlap-separation` maps to the shipped `I4_overlap_detection`,
    not to separation.

### Architecture follow-ups (see [architecture-review.md](./architecture-review.md); T046–T050 implemented 2026-07-24)

- [X] T046 Lazified `audio/workflows/__init__.py` + `audio_analysis/__init__.py` (PEP-562, incl. new
  `harvest_pass`/`PassHarvest`/`aggregate_pass` exports); `_ensure_light_importable` stub deleted —
  verified: the full `adaptive.loop` import chain loads with ZERO heavy modules in a bare env.
- [X] T047 `adaptive/backends.py` dissolved into guarded task-API gateways: U1 →
  `speech_to_text.transcribe_audios` (`return_timestamps="word"`; policy `u1_backend:
  auto|senselab|pipeline`, backend recorded in iterations.json); `consensus_align` → NEW
  `forced_alignment/mms_fa.py` (fills the dead torchaudio slot); `overlap_posteriors` →
  `FramePosterior.per_class` + `overlap_probs()` via `extract_speech_frame_posteriors(...,
  include_per_class=True)` (closes T041's FR-016 dependency); `embed_windows` → the workflow's
  `extract_per_window_embeddings` on crops.
- [X] T048 `adaptive/audio_io.py` routes through `tasks/preprocessing` (exact `prepare_audio`
  replication ⇒ crop `audio_signature`/cache parity) and `tasks/speech_enhancement.enhance_audios`;
  DSP loader retained as the labeled artifact-driven-outside-senselab fallback, **never silent**:
  loader recorded in `convergence.json → audio_backend`, policy `audio_io_backend:
  auto|senselab|dsp` ("senselab" = strict fail-loudly for production).
- [X] T049 (partial) `ScriptLine.iter_leaves()` added (call-site migration opportunistic);
  `adaptive/evaluate` WER → `speech_to_text_evaluation.calculate_wer` with Levenshtein fallback;
  canonical `normalize_transcript_for_wer` moved to `speech_to_text_evaluation/utils.py`
  (aggregate.py re-exports the historical name); `load_ls_ground_truth` →
  `audio_analysis/labelstudio.py`. REMAINING: `triage.dsp_snr_series` → `quality_control/metrics.py`
  deferred until that module's pure-DSP/model-based split (it imports VAD+diarization at top today).
- [X] T050 Transcript fusion promoted to `tasks/speech_to_text_ensemble/` (`fuse_word_streams` with
  explicit `weights`, `load_calibrator`, `iter_word_leaves`); `adaptive/fusion.py` keeps the
  policy→weights wrapper + artifact collection + final-output writers.
- [x] T051 Cache/provenance layer out of the script → `utils/tasks/cached_inference.py`; `_stage_*`
  functions → `workflows/audio_analysis/stages.py`; `wrapper_version_hash` re-scoped to the stage
  modules (documented invalidation-semantics change). Precondition for T040. (Own PR per review.)
- [x] T052 Typed adaptive internals per house style: `adaptive/types.py` dataclasses (`Region`,
  `PlannedIntervention`, `LoopContext`, `InterventionRule`) replacing dict soup; planner + regions
  first. (Opportunistic, own PR.)
  - **Implemented 2026-07-27 as `TypedDict`, not dataclasses** — a deliberate deviation. `Region` is
    written to `rounds/<n>/regions.json` and read back by `plot.py`, `ls_final.py` and the T039
    harness; `PlannedIntervention` lands in `final/iterations.json`. A dataclass would need
    `to_dict`/`from_dict` at each boundary while the dict stayed the wire format, i.e. a second
    representation to keep in sync. `plan_round` also adds `status`/`error`/`intervention_id` *after*
    construction, which a frozen dataclass cannot express. TypedDict types the existing dicts in
    place: zero runtime change, zero serialization change, and every consumer gets checked.
  - **mypy immediately found three latent defects** the dict soup was hiding: `exec_status` was
    written by `loop.py` but never declared anywhere; `region_id` was read as required while being
    assigned only after ranking (now seeded at construction); and a test fixture built `Region`s
    without `n_buckets`. `AxisName` is now a shared literal so `AXES` keeps its narrowing through
    `propose_regions`.
  - Scope was "planner + regions first" per the task: `regions.py`, `policy.py`, `convergence.py`,
    `loop.py`. `LoopContext` and `InterventionRule` remain untyped — `ctx` is a genuinely
    heterogeneous bag mutated across rounds, and typing it usefully means restructuring it, not
    annotating it.

## Dependencies

Phase 2 blocks everything. US2 (Phase 4) is the MVP and depends only on Phase 2. US3 rules plug into
the same engine. US4 consumes the final belief state. T008/T012 (full-pipeline integration) are the
bridge from prototype to production and precede T010/T011/T018/T024/T025/T031/T032.
