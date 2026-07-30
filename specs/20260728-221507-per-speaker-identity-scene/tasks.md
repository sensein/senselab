# Tasks: Per-Speaker Identity Uncertainty and Background Scene Characterization

**Input**: Design documents from `/specs/20260728-221507-per-speaker-identity-scene/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: Included. Three independent reasons: the repository mandates tests mirroring
package structure under `src/tests/` with `*_test.py` naming; constitution IV requires CI
green; and FR-017b makes an automated regression guard a functional requirement, not an
option. Follow red-green-refactor — write the test, watch it fail, then implement.

**Organization**: Grouped by user story. **Phase order deliberately does not follow story
priority.** The P1 story (US1) is blocked by PR #537, which edits `identity.py`,
`clustering.py`, `stages.py`, and `stage_context.py` — exactly US1's files. Both P2
stories are unblocked and ship first. See plan.md "Implementation Phasing".

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependency on incomplete tasks)
- **[Story]**: Which user story the task serves
- Every task names its exact file path

## Path Conventions

Single Python project. Library code under `src/senselab/`, tests mirroring it under
`src/tests/`, CLI entry points under `scripts/`. All commands via `uv run`
(constitution I).

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Dependency and scaffolding changes every later phase depends on.

- [X] T001 Add `librosa` and `pyloudnorm` to `[project.dependencies]` in `pyproject.toml`, then run `uv sync` and commit the updated `uv.lock`. Note `librosa` is currently transitive only despite a prior spec intending to promote it (plan.md post-design check).
- [X] T002 [P] Create the package-data directory `src/senselab/audio/workflows/audio_analysis/data/detection_margin/` with a `.gitkeep`, and register it in `pyproject.toml` package-data so profiles ship with the wheel.
- [X] T003 [P] Bump `CACHE_SCHEMA_VERSION` in `src/senselab/utils/tasks/cached_inference.py` so stale entries are discarded — two changes in this feature alter outputs (score aggregation, amplify-before-serialize).
- [X] T004 [P] Verify the signal-processing surface with the check in `quickstart.md` ("Setup"), confirming `librosa.pcen`, `librosa.A_weighting`, and `pyloudnorm` import cleanly under `uv run`.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: The policy-profile layer and audio-variant provenance that every story
writes through. Constitution VIII makes this blocking — no threshold may appear as a
literal in any later task.

**⚠️ CRITICAL**: No user story work begins until this phase completes.

- [X] T005 Write failing tests for the detection-margin profile loader in `src/tests/audio/workflows/audio_analysis/detection_margin_profile_test.py`: monotone margin ordering, quantile range, bias correction computed from `q` rather than stored, and a hard error when a `provisional` derivation claim lacks a `note`.
- [X] T006 Implement `DetectionMarginProfile` and `DerivationRecord` loading and validation in `src/senselab/audio/workflows/audio_analysis/calibration.py`, per `contracts/policy-profile.md`. Reject a profile whose only support for a margin value is `provisional`.
- [X] T007 [P] Write failing tests for audio-variant provenance in `src/tests/audio/workflows/audio_analysis/level_provenance_test.py`: every result carries a variant name and gain, gain never exceeds the cap, and clipping or requantization is surfaced rather than silent.
- [X] T008 Implement the `AudioVariant` record and the `<run_dir>/level.json` writer in `src/senselab/audio/workflows/audio_analysis/level.py`, per `contracts/level-verdicts.md`. Include LUFS measurement via `pyloudnorm`, true-peak, and per-segment gain fields.
- [X] T009 Extend `src/senselab/audio/workflows/audio_analysis/stage_context.py` so `StageContext` carries the active variant name and gain, and add a `STAGE_VERSIONS` entry for each new stage. **Coordinate with #537, which also edits this file.**
- [X] T010 [P] Add the `influence` and `speaker_count` key groups to `src/senselab/audio/workflows/audio_analysis/adaptive/policy/default.yaml` per `contracts/policy-profile.md`, with `derivation_gate.derived` strictly below `derivation_gate.independent` (FR-011).
- [X] T011 Write the initial bundled profile `src/senselab/audio/workflows/audio_analysis/data/detection_margin/2026-07-29.json` with the 3/6/10 dB ladder and the derivation record, marking the partial-loudness figure and the derived statistics as `provisional` per research.md open risks 2 and 3 (FR-023).

**Checkpoint**: Policy and provenance layers ready — story phases can begin.

---

## Phase 3: User Story 2 — Pin classifier level sensitivity (Priority: P2)

**Plan phase A. Runs first: unblocked, cheap, and its measured floors feed the margin
derivation that US3 depends on.**

**Goal**: Turn the measured finding — both classifiers are amplitude-sensitive — into a
regression guard, document each classifier's floor and mechanism, and fix the
score-comparability defect that structurally suppresses background categories.

**Independent Test**: Run `scripts/probe_classifier_levels.py` on a cached clip and
confirm `level-verdicts.json` reports `level_sensitive` per classifier with a ≥30 dB gain
range, a floor level, and a code reference corroborating the mechanism.

### Tests for User Story 2

- [X] T012 [P] [US2] Write failing tests for gain-sweep verdict derivation in `src/tests/audio/tasks/classification/level_probe_test.py`: label-stability and score-delta computation, the ≥30 dB range assertion (SC-005), and per-classifier attribution including window length (FR-015).
- [X] T013 [P] [US2] Write failing tests for floor-signature detection in `src/tests/audio/tasks/classification/level_probe_test.py`: a fixed label pattern on digital silence is recognized as a floor response, and detection does **not** rely on the silence score alone (FR-020d) — the AST case where `Silence` peaks at 0.437 while `Music` at 0.350 clears a normal threshold.
- [X] T014 [P] [US2] Write a failing score-comparability test in `src/tests/audio/workflows/audio_analysis/sound_sources_test.py`: with a dominant source plus a quieter secondary source, the secondary category mass is not suppressed as an artifact of one classifier's scores being a mutually-exclusive competition (FR-017c).
- [X] T015 [P] [US2] Write a failing headroom test in `src/tests/audio/tasks/classification/yamnet_test.py`: amplification applied before serialization, and clipping or requantization in the input path detected and reported (FR-017d, FR-019b).
- [X] T016 [US2] Write the offline regression guard in `src/tests/audio/tasks/classification/level_probe_test.py` asserting each recorded verdict and floor still holds, skipping with a clear message when a checkpoint is not cached — it must never require network (FR-017b, constitution VI).

### Implementation for User Story 2

- [X] T017 [US2] Implement `AmplitudeInvarianceVerdict` and the gain-sweep driver in `src/senselab/audio/tasks/classification/level_probe.py`, computing label stability, score deltas, floor level, and floor signature per `data-model.md` §3 (FR-013, FR-014).
- [X] T018 [US2] Implement `scripts/probe_classifier_levels.py` per `contracts/cli.md`: cached checkpoints only, `--gains-db` spanning ≥30 dB, digital-silence probe included by default, results persisted to `artifacts/level_probe/level-verdicts.json`. Include a quiet-recording arm reporting whether amplification changes which background source categories are reported (FR-017).
- [X] T019 [US2] Fix score comparability in `src/senselab/audio/workflows/audio_analysis/sound_sources.py`: stop summing mutually-exclusive and independent per-class scores into the same category masses (FR-017c). Change `function_to_apply` handling in `src/senselab/audio/tasks/classification/huggingface.py` so AST's multi-label head is not softmaxed across 527 classes.
- [X] T020 [US2] Apply gain before the temp-WAV write in `src/senselab/audio/tasks/classification/yamnet.py` and preserve resolution on that path, so faint content is not destroyed before the gain reaches it (FR-019b).
- [X] T021 [US2] Add clipping and requantization detection to `src/senselab/audio/workflows/audio_analysis/level.py`, reporting rather than silently degrading (FR-017d).
- [X] T022 [US2] Record the audio variant and gain on every scene-analysis result in `src/senselab/audio/workflows/audio_analysis/stages.py` (FR-012, SC-006). **Coordinate with #537.**
- [X] T023 [US2] Document each classifier's low-level floor and mechanism in `src/senselab/audio/tasks/classification/level_probe.py` docstrings, including that the short-window floor is a learned decision rather than the log-mel offset (FR-016, FR-017a).

**Checkpoint**: Level sensitivity pinned and guarded; background category masses no longer
structurally suppressed. US3's machine-side floors are now available.

---

## Phase 4: User Story 4 — Background mask with uncertainty (Priority: P2)

**Plan phase B. Unblocked; gates the confidence of US3.**

**Goal**: Emit a mask of target-free regions with per-region uncertainty, driven by task
metadata, and support introspection of what those regions contain.

**Independent Test**: Run on a recording with interleaved target activity and quiet
intervals; confirm three-state regions with uncertainty, a recorded metadata provenance,
and reported total masked duration and fraction.

### Tests for User Story 4

- [X] T024 [P] [US4] Write failing tests for three-state mask construction in `src/tests/audio/workflows/audio_analysis/background_mask_test.py`: `target_free` / `target_active` / `indeterminate` are all reachable, uncertainty is in `[0,1]`, and a binary mask is rejected (FR-032, SC-019).
- [X] T025 [P] [US4] Write a failing guard-interval test in `src/tests/audio/workflows/audio_analysis/background_mask_test.py`: a bucket within the guard interval of target activity is never `target_free`, even when no activity is detected inside it (FR-034).
- [X] T026 [P] [US4] Write failing task-metadata tests in `src/tests/audio/workflows/audio_analysis/background_mask_test.py`: recognized task types select their target event set, and absent or unrecognized metadata triggers the conservative fallback with provenance recorded (FR-033, FR-033b, SC-025).
- [X] T027 [US4] Write the decisive misattribution test in `src/tests/audio/workflows/audio_analysis/background_mask_test.py`: for a breathing- or cough-task recording, **zero target events are reported as background sources** (FR-033a, SC-024). Synthesize a fixture if `src/tests/data_for_testing/` has no suitable clip, and note in the test docstring that a synthesized breath may not exercise the real failure mode.
- [X] T028 [P] [US4] Write failing tests for non-target speech retention in `src/tests/audio/workflows/audio_analysis/background_mask_test.py`: a distant talker inside a mask region stays masked and remains reportable as a background source (FR-033c).
- [X] T029 [P] [US4] Write failing tests for the empty-mask and negligible-fraction cases in `src/tests/audio/workflows/audio_analysis/background_mask_test.py`: continuous target activity yields `is_empty: true` with the limitation stated rather than the field omitted (FR-040, SC-022), and a tiny mask is flagged (FR-038).
- [X] T030 [P] [US4] Write failing tests for mask region sizing in `src/tests/audio/workflows/audio_analysis/background_mask_test.py`: regions shorter than the long-window classifier's window set `supports_long_window: false` (FR-045).

### Implementation for User Story 4

- [X] T031 [US4] Implement `BackgroundMaskRegion` and `BackgroundMask` in `src/senselab/audio/workflows/audio_analysis/background_mask.py` per `data-model.md` §5, deriving the three states from the existing presence axis plus diarization and voice-activity outputs.
- [X] T032 [US4] Implement task-metadata-driven target-event selection in `src/senselab/audio/workflows/audio_analysis/background_mask.py`, reading `mask.target_event_types_by_task` from the profile with the conservative fallback (FR-033, FR-033b).
- [X] T033 [US4] Implement non-speech target detection for breathing and cough tasks in `src/senselab/audio/workflows/audio_analysis/background_mask.py`, so the mask is not built from speech activity alone when the target is a non-speech vocal event (FR-033a).
- [X] T034 [US4] Implement guard-interval trimming in `src/senselab/audio/workflows/audio_analysis/background_mask.py`, recording `guard_trimmed_s` per region (FR-034).
- [X] T034a [US4] Add a background-mask row to the final timeline in `src/senselab/audio/workflows/audio_analysis/adaptive/plot.py`: three-state strip with uncertainty as alpha and the guard-trimmed span hatched, placed above the axis rows so a reviewer sees which spans the findings below can be trusted in (user request; FR-031/FR-034).
- [X] T035 [US4] Implement the mask writers — `background_mask.parquet` and `background_mask.json` — in `src/senselab/audio/workflows/audio_analysis/io.py` per `contracts/background-mask.md` (FR-038, SC-021).
- [X] T036 [US4] Implement `MaskedRegionIntrospection` and the `mask_introspection.json` writer in `background_mask.py`, including `is_noise_floor_only` and the A-weighted summary that is never used as the gate (FR-037).
- [X] T037 [US4] Add `--background-mask`, `--task-type`, `--mask-guard-interval`, and `--mask-introspect` flags to `scripts/analyze_audio.py` per `contracts/cli.md`. **Minor collision with #523.**
- [X] T038 [US4] Wire the mask into `run_pass` in `src/senselab/audio/workflows/audio_analysis/stages.py`, emitting on the same grid as the presence output (FR-031). **Coordinate with #537.**

**Checkpoint**: Mask available with uncertainty and provenance. Regions where background
claims are trustworthy without suppression are now identifiable.

---

## Phase 5: User Story 3 — Background characterization (Priority: P3)

**Plan phase C. Depends on US2's measured floors and US4's mask.**

**Goal**: Detect background sources by per-band floor subtraction with a corroborated
margin ladder, on a foreground-suppressed variant, without fabricating findings from
amplified noise.

**Independent Test**: On a recording with a dominant near-microphone talker over audible
background, report background categories with above-floor margins; on pure noise floor,
report nothing.

### Tests for User Story 3 — noise floor

- [X] T039 [P] [US3] Write failing tests for percentile bias correction in `src/tests/audio/workflows/audio_analysis/noise_floor_test.py`: the correction equals `1/(−ln(1−q))`, is applied, and the corrected floor recovers the true mean noise power on synthetic exponential noise (FR-021d, research risk 2).
- [X] T040 [P] [US3] Write failing tests for patch aggregation in `src/tests/audio/workflows/audio_analysis/noise_floor_test.py`: the gate is evaluated on patch-aggregated energy and a frame-level evaluation is rejected (FR-021e).
- [X] T041 [P] [US3] Write failing tests for iterative event exclusion in `src/tests/audio/workflows/audio_analysis/noise_floor_test.py`: a sustained synthetic source is not absorbed into the floor, and iteration reaches stability (FR-021f).
- [X] T042 [P] [US3] Write failing tests for floor freezing inside detected events in `src/tests/audio/workflows/audio_analysis/noise_floor_test.py` (FR-021g).
- [X] T043 [US3] Write failing tests for activity-conditioned floors in `src/tests/audio/workflows/audio_analysis/noise_floor_test.py`: separate `active` and `quiet` strata, and the conditioned floor does not systematically over-gate quiet stretches relative to an unconditioned one (FR-021h). Mark the test docstring as validating unpublished synthesis (research risk 1).
- [X] T044 [P] [US3] Write failing tests for the stationary parallel pass in `src/tests/audio/workflows/audio_analysis/noise_floor_test.py`: a continuous narrowband hum survives rather than being erased by floor subtraction (FR-021i).
- [X] T045 [P] [US3] Write failing tests for recorder-floor detection in `src/tests/audio/workflows/audio_analysis/noise_floor_test.py`: when a band's floor is within a few dB of the estimated capture-chain self-noise, findings there carry `binding_floor: "recorder"` and no perceptual claim is made (FR-021b).

### Tests for User Story 3 — margins, guards, suppression

- [ ] T046 [P] [US3] Write failing tests for the margin ladder in `src/tests/audio/workflows/audio_analysis/sources_test.py`: tier assignment at 3/6/10 dB, rejection below 3 dB, and that a finding never advances tier on gain alone (FR-021, research D1).
- [ ] T047 [P] [US3] Write failing tests for the noise-character guard in `src/tests/audio/workflows/audio_analysis/sources_test.py`: broadband noise floors are separated from structured sources by spectral flatness, evaluated relatively against the band's own baseline (FR-020b).
- [ ] T048 [P] [US3] Write failing tests for label quarantine in `src/tests/audio/workflows/audio_analysis/sources_test.py`: water-like and broadband environmental labels are suppressed unless the noise-character test passes (FR-020c).
- [ ] T049 [US3] Write the false-positive test in `src/tests/audio/workflows/audio_analysis/sources_test.py`: **amplified pure noise floor yields zero findings** (SC-018), using the synthetic clip recipe from `quickstart.md`.
- [ ] T050 [US3] Write the decisive differential test in `src/tests/audio/workflows/audio_analysis/foreground_test.py`: two clips identical except one contains a faint background source must produce **different** reported categories (SC-015). Document that a 30 dB-suppression baseline fails this by construction (research D6).
- [ ] T051 [P] [US3] Write failing tests for suppression-depth reporting in `src/tests/audio/workflows/audio_analysis/foreground_test.py`: depth always reported when requested, and no detectability claim when residual foreground exceeds the candidate (FR-018a, SC-016).
- [ ] T052 [P] [US3] Write failing tests for leakage margins in `src/tests/audio/workflows/audio_analysis/foreground_test.py`: a `speech` or `people` category from a suppressed variant always carries `leakage_margin_db` (FR-026, SC-008).
- [ ] T053 [P] [US3] Write failing tests for pre-gain rejection and the gain cap in `src/tests/audio/workflows/audio_analysis/level_test.py`: segments below the trust floor are rejected rather than amplified, gain is per-segment, and exceeding the cap is an error not a clamp (FR-019, FR-019a, FR-020a).
- [ ] T054 [P] [US3] Write failing tests for excision routing in `src/tests/audio/workflows/audio_analysis/sources_test.py`: long-window results come from excised mask segments, carry `padding_fraction`, and are never merged with grid results (FR-041, FR-043, FR-044, SC-031, SC-033).
- [ ] T055 [P] [US3] Write failing tests asserting no absolute perceptual quantities are emitted for uncalibrated input (FR-021c) in `src/tests/audio/workflows/audio_analysis/sources_test.py`.

### Implementation for User Story 3

- [X] T056 [US3] Implement the two-pass per-band noise-floor estimator in `src/senselab/audio/workflows/audio_analysis/noise_floor.py`: third-octave bands, bias-corrected percentile, iterative event exclusion, per `research.md` D4 (FR-018b, FR-021a). Leave `quality_control/metrics.py` untouched and note its bias in its docstring.
- [X] T057 [US3] Add floor freezing inside detected events and activity-conditioned strata to `noise_floor.py` (FR-021g, FR-021h).
- [X] T058 [US3] Implement recorder-floor estimation and `binding_floor` resolution in `noise_floor.py` (FR-021b, FR-022a).
- [ ] T059 [US3] Implement the `noise_floor.parquet` writer per `contracts/background-sources.md` in `src/senselab/audio/workflows/audio_analysis/io.py`.
- [ ] T060 [US3] Implement the margin ladder, tier assignment, occupancy, minimum duration, and hysteresis in `src/senselab/audio/workflows/audio_analysis/sources.py` (FR-020, FR-021, FR-021j, FR-027, SC-014).
- [ ] T061 [US3] Implement presence/extent separation in `sources.py` so the margin gate decides presence while boundaries are determined independently (FR-021k, research D12).
- [ ] T062 [US3] Implement the noise-character guard, label quarantine, and floor-response signature rejection in `sources.py` (FR-020b, FR-020c, FR-020d).
- [ ] T063 [US3] Implement modulation-depth computation as an orthogonal event feature in `sources.py`, down-weighting the 3–6 Hz band because the residual may carry inherited talker modulation (research D11).
- [ ] T064 [US3] Implement the foreground-suppressed variant and suppression-depth measurement in `src/senselab/audio/workflows/audio_analysis/foreground.py`, reusing `speech_enhancement` for the residual, with the graceful fallback of FR-029 (FR-018).
- [ ] T064a [P] [US3] Write a failing invariant test in `src/tests/audio/workflows/audio_analysis/foreground_test.py` asserting `stage_asr`, `stage_alignment`, and `stage_diarization` never receive the foreground-suppressed variant — assert on the variant name reaching each stage, not on output quality, since a quality-based test would pass while quietly transcribing suppressed audio (FR-028).
- [ ] T065 [US3] Implement leakage-margin measurement in `foreground.py` (FR-026).
- [ ] T066 [US3] Implement per-segment gain toward the level target with the hard cap, and the `−23 LUFS` normalization applied identically to both variants, in `src/senselab/audio/workflows/audio_analysis/level.py` (FR-019, FR-019a, FR-019c).
- [ ] T066a [US3] Implement the recovery-delta report in `src/senselab/audio/workflows/audio_analysis/sources.py`: per category, which sources the foreground-suppressed variant recovers that the unmodified recording does not, written to `<run_dir>/<pass>/recovery_delta.json` (FR-025, SC-007).
- [ ] T066b [P] [US3] Write failing tests for the recovery-delta report in `src/tests/audio/workflows/audio_analysis/foreground_test.py`: a source audible only under suppression appears in the delta; a source recovered by both does not. Distinct from T050, which varies whether a source exists rather than which variant sees it (FR-025, SC-007).
- [ ] T066c [US3] Implement mask-uncertainty discounting in `src/senselab/audio/workflows/audio_analysis/sources.py`, populating `discounted_reason` when a finding is weakened by mask uncertainty rather than by weak evidence (FR-036).
- [ ] T066d [P] [US3] Write failing tests for cross-level score comparison in `src/tests/audio/workflows/audio_analysis/sources_test.py`: classifier scores are never compared or ranked across segments at different levels, since score varies with level on unchanged audio and non-monotonically in at least one classifier (FR-020e).
- [ ] T066e [P] [US3] Write failing tests in `src/tests/audio/workflows/audio_analysis/sources_test.py` for the three untested mask and distance criteria: two same-type sources separated by at least `guards.min_distance_separation_db` are both reported with their margins (SC-012); every finding states its mask provenance (SC-020); a source in a target-free interval outranks the same source under target activity (SC-023).
- [ ] T067 [US3] Implement excision routing in `sources.py`: long-window classifier on excised mask segments, short-window on the grid, with padding fraction recorded and short regions flagged (FR-024, FR-041, FR-042, FR-043, FR-045, SC-032).
- [ ] T068 [US3] Implement the stationary parallel unsubtracted analysis in `sources.py` (FR-021i).
- [ ] T069 [US3] Implement the `background_sources.parquet` and `suppression.json` writers per `contracts/background-sources.md` in `io.py`.
- [ ] T070 [US3] Implement `scripts/calibrate_detection_margin.py` per `contracts/cli.md`, consuming `level-verdicts.json` as the machine basis and hard-erroring on an unmarked provisional figure (FR-022, SC-017).
- [ ] T071 [US3] Add `--foreground-suppression`, `--suppression-model`, `--detection-margin-profile`, `--level-target-lufs`, `--gain-cap-db`, `--scene-variant`, and `--stationary-pass` flags to `scripts/analyze_audio.py` (FR-030).
- [ ] T072 [US3] Wire background characterization into `run_pass` in `stages.py`, consuming the mask from US4 and preferring mask regions (FR-035, FR-039). **Coordinate with #537.**

**Checkpoint**: Background sources detected by floor subtraction with corroborated
margins, guarded against fabrication, with mask-region confidence differentiation.

---

## Phase 6: User Story 1 guards — mutual influence safety (Priority: P1 infrastructure)

**Plan phase D. MUST complete before Phase 7.** Per spec Dependencies, the guards land
before the influence paths they protect, so the loop is never able to confirm its own
edits even transiently.

**Goal**: Uncertainty-gated influence weighting, revision provenance, self-confirmation
detection, oscillation detection, and deterministic iteration.

**Independent Test**: Unit-testable without any influence path enabled — construct
synthetic signals and assert weighting, attribution, and termination behavior.

### Tests for the guards

- [X] T073 [P] [US1] Write failing tests for uncertainty gating in `src/tests/audio/workflows/audio_analysis/adaptive/influence_test.py`: a high-uncertainty signal has proportionally reduced effective weight (FR-011b).
- [X] T074 [P] [US1] Write failing tests for derivation gating in `src/tests/audio/workflows/audio_analysis/adaptive/influence_test.py`: a `derived` signal's gate is strictly below an `independent` one, and a derived signal alone cannot drive a revision an independent signal contradicts (FR-011c, SC-030).
- [X] T075 [US1] Write failing tests for the self-confirmation guard in `src/tests/audio/workflows/audio_analysis/adaptive/influence_test.py`: a value revised by influence records `resolution_kind: "revision"`, and its subsequent uncertainty drop is **not** reported as improved confidence (FR-011d, SC-027).
- [X] T076 [P] [US1] Write failing tests for oscillation detection in `src/tests/audio/workflows/audio_analysis/adaptive/convergence_test.py`: a constructed alternation terminates with `termination_reason: "oscillation"` and `converged: false` (FR-011e, SC-028).
- [X] T077 [P] [US1] Write failing tests for revision attribution in `influence_test.py`: every revision carries round, cause, weight, and evidence (FR-011g, SC-026).
- [X] T078 [P] [US1] Write failing determinism tests in `src/tests/audio/workflows/audio_analysis/adaptive/influence_test.py`: fixed evaluation order and tie-breaking produce byte-identical results across runs (FR-011f, SC-029).
- [X] T079 [P] [US1] Write failing tests for unresolved-quantity reporting in `convergence_test.py`: a quantity that never converged is not presented as settled (FR-011h).

### Implementation for the guards

- [X] T080 [US1] Implement `InfluenceWeight` with uncertainty and derivation gates in `src/senselab/audio/workflows/audio_analysis/adaptive/influence.py` per `data-model.md` §2.
- [X] T081 [US1] Implement `RevisionRecord` and `ResolutionKind` in `src/senselab/audio/workflows/audio_analysis/adaptive/provenance.py`, generalizing the loop's existing explained-versus-improved distinction (research D13).
- [X] T082 [US1] Implement oscillation and no-improvement detection in `src/senselab/audio/workflows/audio_analysis/adaptive/convergence.py`, extending `ConvergenceReport` with `termination_reason`, `oscillation_states`, and `unresolved_quantities`.
- [X] T083 [US1] Enforce deterministic iteration order and stable serialized key ordering in `src/senselab/audio/workflows/audio_analysis/adaptive/loop.py` (FR-011f).
- [X] T084 [US1] Add `--influence-profile` and `--max-influence-rounds` flags to `scripts/analyze_audio.py`.

**Checkpoint**: Guards in place and independently tested. Influence paths may now be
enabled.

---

## Phase 7: User Story 1 — Per-speaker identity uncertainty (Priority: P1) 🎯

**Plan phase E. BLOCKED on PR #537** — it edits `identity.py`, `clustering.py`,
`stages.py`, `stage_context.py` and adds four diarizers. Do not start until it merges;
then rebase onto `alpha`.

**Goal**: Replace the single per-bucket identity scalar in the final convergence with a
speaker-count distribution and per-speaker presence tracks, with full participation in
the adaptive loop.

**Independent Test**: On a recording where diarizers disagree on speaker count, determine
from `final/speakers.json` alone how many speakers each source claimed and which sources
supported each count, without opening intermediate artifacts.

### Tests for User Story 1

- [ ] T085 [US1] Confirm #537 has merged and rebase this branch onto `origin/alpha`, resolving conflicts in `identity.py`, `clustering.py`, `stages.py`, and `stage_context.py`. Verify with an empty `git diff` against a pre-rebase backup ref that no unintended drift occurred.
- [ ] T086 [P] [US1] Write failing tests for the speaker-count posterior in `src/tests/audio/workflows/audio_analysis/speaker_identity_test.py`: probabilities sum to 1, multi-modal disagreement is representable without collapsing to a majority or mean, and every support key appears in probabilities (FR-002, FR-006, FR-008).
- [ ] T087 [P] [US1] Write failing tests for the single-speaker case in `src/tests/audio/workflows/audio_analysis/speaker_identity_test.py`: all sources agreeing yields ≥0.9 mass on one count and exactly one hypothesis, with no phantom speaker (FR-009, SC-001).
- [ ] T088 [P] [US1] Write failing tests for the zero-speech case in `src/tests/audio/workflows/audio_analysis/speaker_identity_test.py`: mass concentrated on zero, no hypotheses, no presence tracks.
- [ ] T089 [P] [US1] Write failing tests for per-speaker presence tracks in `src/tests/audio/workflows/audio_analysis/speaker_identity_test.py`: full-duration coverage on the presence grid for every hypothesis, gaps as null-confidence rows rather than absent rows, and simultaneous presence for overlapping speakers (FR-003, SC-003).
- [ ] T090 [P] [US1] Write failing tests separating existence uncertainty from presence uncertainty in `src/tests/audio/workflows/audio_analysis/speaker_identity_test.py` (FR-004).
- [ ] T091 [P] [US1] Write failing tests for source-label correspondence in `src/tests/audio/workflows/audio_analysis/speaker_identity_test.py`: each source's own labels map auditably to fused hypotheses across unrelated naming conventions (FR-005).
- [ ] T092 [P] [US1] Write failing tests for independent-versus-derived source classification in `src/tests/audio/workflows/audio_analysis/speaker_identity_test.py` (FR-007).
- [ ] T093 [US1] Write the motivating-case test in `src/tests/audio/workflows/audio_analysis/speaker_identity_test.py` using `audio_48khz_mono_16bits`: two diarizers reporting one speaker while embedding clustering reports five must yield `is_multimodal: true` with per-count support — **not** a collapse to either answer. Assert representation, not accuracy; the spec deliberately does not require resolving it in a particular direction (SC-002).
- [ ] T094 [P] [US1] Write failing contract tests for `final/speakers.json`, `final/per_speaker_presence.parquet`, and the extended `final/convergence.json` in `src/tests/audio/workflows/audio_analysis/adaptive/final_outputs_test.py` per `contracts/speaker-identity.md`.

- [ ] T094a [P] [US1] Write a failing regression test in `src/tests/audio/workflows/audio_analysis/uncertainty_axes_test.py` asserting the three existing uncertainty axes remain loadable and aggregate unchanged after the identity representation is replaced, and that no presence or utterance consumer requires modification (SC-010).
- [ ] T094b [P] [US1] Write a failing reproducibility test in `src/tests/audio/workflows/audio_analysis/adaptive/final_outputs_test.py` asserting `final/speakers.json` and `final/per_speaker_presence.parquet` are byte-identical across two runs with identical inputs and settings, using the `cmp` recipe from `quickstart.md` (FR-010, SC-004).

### Implementation for User Story 1

- [ ] T095 [US1] Implement `SpeakerHypothesis`, `SpeakerCountPosterior`, and `SourceLabelCorrespondence` in `src/senselab/audio/workflows/audio_analysis/speaker_identity.py` per `data-model.md` §1.
- [ ] T096 [US1] Implement `PerSpeakerPresenceTrack` derivation in `src/senselab/audio/workflows/audio_analysis/speaker_identity.py`, on the existing presence grid with overlap support (FR-003).
- [ ] T097 [US1] Implement source-kind classification — independent versus derived — in `src/senselab/audio/workflows/audio_analysis/speaker_identity.py`, covering the clustering-derived pseudo-diarizer (FR-007).
- [ ] T098 [US1] Extend `src/senselab/audio/workflows/audio_analysis/identity.py` to emit per-speaker structure downstream while retaining the per-bucket axis as the evidence-gathering mechanism.
- [ ] T099 [US1] Extend `src/senselab/audio/workflows/audio_analysis/adaptive/belief.py` with per-speaker state and count-posterior tracking.
- [ ] T100 [US1] Add a count-disagreement intervention trigger to `src/senselab/audio/workflows/audio_analysis/adaptive/interventions.py`, reading from the vote store rather than row metadata.
- [ ] T101 [US1] Enable the influence paths in `src/senselab/audio/workflows/audio_analysis/adaptive/loop.py` — identity to diarization, diarization to per-speaker presence, mask to presence, utterance to speaker attribution — each gated through `influence.py` (FR-011a).
- [ ] T102 [US1] Implement the `final/speakers.json` and `final/per_speaker_presence.parquet` writers in `src/senselab/audio/workflows/audio_analysis/adaptive/fusion.py`, **replacing** the single-scalar identity representation rather than adding beside it (FR-001).
- [ ] T103 [US1] Extend the `final/convergence.json` writer in `fusion.py` with resolution kinds, applied influence weights, and unresolved quantities.
- [ ] T104 [US1] Add `--per-speaker-identity` flag to `scripts/analyze_audio.py`.

**Checkpoint**: All four stories independently functional.

---

## Phase 8: Polish & Cross-Cutting Concerns

- [ ] T105 [P] Add the spectrogram-bearing per-speaker row to the final timeline in `src/senselab/audio/workflows/audio_analysis/adaptive/plot.py`, reading the belief parquet rather than `final/presence.parquet`.
- [ ] T106 [P] Add Label Studio tracks for the background mask and per-speaker presence in `src/senselab/audio/workflows/audio_analysis/labelstudio.py`.
- [ ] T107 [P] Update `README.md` with a runbook covering the mask, suppression, and per-speaker identity flags.
- [ ] T108 [P] Update the "Audio analysis script" and "Three-axis uncertainty workflow" sections of `CLAUDE.md` to describe the new outputs and the level findings.
- [ ] T109 Run the full `quickstart.md` validation sequence for every phase and record results under `artifacts/`.
- [ ] T110 Run a full end-to-end pass on both local validation recordings via `scripts/analyze_audio.py`, writing to `artifacts/e2e_runs/`, confirming SC-015 and SC-018 hold on real audio rather than only on fixtures.
- [X] T111 Validate the derived χ²₂ statistics on synthetic noise — bias correction, per-bin σ, patch-variance collapse — in `src/tests/audio/workflows/audio_analysis/noise_floor_test.py`, then update `derived_statistics_status` in `src/senselab/audio/workflows/audio_analysis/data/detection_margin/2026-07-29.json` from `provisional` to `verified` only if they hold (research risk 2).
- [ ] T112 Measure whether an alternative AudioSet classifier with more log-floor headroom performs better on faint content using `scripts/probe_classifier_levels.py`, recording results to `artifacts/level_probe/alternatives.json`. Do **not** switch on the headroom argument alone — it already mispredicted the current pair's ordering (research risk 4).
- [ ] T112a Measure default-run wall-clock against the pre-feature baseline using `scripts/analyze_audio.py` with foreground suppression not requested, recording both timings to `artifacts/timing/default_run.json`, and confirm the increase stays within 10% (SC-009).
- [ ] T113 Run `uv run ruff format && uv run ruff check && uv run mypy . && uv run codespell` and fix all findings.
- [ ] T114 Run `uv run pytest -n auto` and confirm green before opening the PR.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 Setup**: no dependencies.
- **Phase 2 Foundational**: depends on Phase 1. **Blocks all story phases** — every
  threshold must resolve through the profile layer (constitution VIII).
- **Phase 3 (US2)**: depends on Phase 2. Unblocked by external work.
- **Phase 4 (US4)**: depends on Phase 2. Unblocked. Parallel with Phase 3.
- **Phase 5 (US3)**: depends on Phase 3 (measured floors feed the margin derivation) and
  Phase 4 (mask regions).
- **Phase 6 (guards)**: depends on Phase 2 only. Can run parallel with Phases 3–5.
- **Phase 7 (US1)**: depends on Phase 6 **and on PR #537 merging**.
- **Phase 8 Polish**: depends on the phases whose work it polishes.

### Story Dependencies

- **US2 (P2)**: independent.
- **US4 (P2)**: independent.
- **US3 (P3)**: needs US2's floors and US4's mask. Would still function without the mask,
  but with lower-confidence findings only.
- **US1 (P1)**: needs the Phase 6 guards and #537. **This is the only externally blocked
  story, which is why it is sequenced last despite being P1.**

### Parallel Opportunities

Phases 3, 4, and 6 are mutually independent and can proceed simultaneously.

Within phases, tasks marked `[P]` touch different files:

- Phase 2: T007 and T010 alongside T005.
- Phase 3: T012–T015 (four test files) in parallel.
- Phase 4: T024–T026, T028–T030 in parallel; T027 is sequenced after T026 because it
  depends on the metadata plumbing.
- Phase 5: T039–T042 and T044–T048 and T051–T055 in parallel — the largest parallel block
  in the feature.
- Phase 6: T073, T074, T076–T079 in parallel.
- Phase 7: T086–T092 and T094 in parallel, all after T085's rebase.
- Phase 8: T105–T108 in parallel.

### Within Each Story

Tests are written first and must fail before implementation. Entities precede the modules
that assemble them; modules precede the writers; writers precede CLI wiring.

---

## Implementation Strategy

### Suggested MVP

**Phase 1 + Phase 2 + Phase 3 (US2)** — 23 tasks. This delivers a standalone, valuable
increment: the amplitude-invariance verdicts with an offline regression guard, plus the
score-comparability fix that stops background categories being structurally suppressed.
That fix alone improves every existing background-source output, independent of anything
else in this feature.

Note the MVP is **not** the P1 story, because the P1 story cannot start.

### Incremental Delivery

1. **Setup + Foundational** → policy and provenance layers land, nothing user-visible.
2. **US2** → level findings pinned, a real defect fixed. Shippable.
3. **US4** → mask available; regions where background claims are trustworthy identified.
   Shippable.
4. **US3** → background sources detected with corroborated margins. Shippable.
5. **Guards** → influence safety, no behavior change on its own.
6. **US1** → per-speaker identity, once #537 lands. Shippable.

Each of steps 2, 3, 4, and 6 is a viable stopping point.

### Risk-Ordered Notes

- **T050 (SC-015) is the make-or-break task.** If two recordings differing only in a faint
  background source produce identical categories, the pipeline is reporting residual
  foreground and US3 does not work regardless of how much else passes. Write it early in
  Phase 5 and treat a failure as a signal about suppression depth, not about the margin.
- **T043 and T111 validate unpublished synthesis.** Activity-conditioned floors and the
  derived χ²₂ statistics have no published precedent. If either fails validation, the
  margin ladder's evidential basis weakens and the profile must be re-derived rather than
  patched.
- **T027 (SC-024) may lack a fixture.** A breathing- or cough-task clip may not exist
  locally; a synthesized breath may not exercise the real failure mode. Flag rather than
  paper over.
- **T085's rebase is the highest-conflict task.** #537 touches four of US1's files. Verify
  with a pre-rebase backup ref and an empty diff, as was done for the prior stacked-branch
  rebase on this repo.
- **US3 may be reordered by evidence.** If the mask proves to carry most of the
  trustworthy background findings — plausible, since 30 dB suppression measurably failed —
  then deep suppression becomes an enhancement for the co-occurring case rather than the
  primary path. Spec Assumptions already anticipates this.
