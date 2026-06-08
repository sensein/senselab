# Tasks: Speaker Profile Embedding for analyze_audio

**Input**: Design documents from `/specs/20260527-151905-speaker-profile-embedding/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Included — senselab enforces `pytest` + `ruff` + `mypy` gates and ships a `*_test.py` per task module; success criteria are framed as testable. Per-story tests are written before that story's implementation and should fail first.

**Organization**: Tasks grouped by user story (US1=P1, US2=P2, US3=P3) so each is independently implementable and testable.

## Close-out status (2026-06-04)

**Delivered**: Phases 1–5 (Setup, Foundational, US1, US2, US3) and the in-scope Polish tasks (T028 threshold characterization, T029 SC-006 regression lock, T030 docs, T031 quickstart end-to-end, T032 quality gates). The feature builds contamination-tolerant per-subject profiles, flags other-voice windows and emits a recording-level rollup in `analyze_audio`, and produces a target-quality indicator — all as signal producers with no embedded gating policy.

**Deferred (not in scope this iteration)**:
- **Phase 6 (T033–T036) — cross-stage cache reuse (FR-015)**: helper shipped + unit-tested; the global-blast-radius `analyze_audio` keying swap is deferred to real-data deployment. See the Phase 6 banner below and spec.md Clarifications 2026-06-04.
- **T028b — optional per-window centroid confidence weighting**: deferred; SQUIM-gated trust identified as the right lever (2026-06-04 enhancement-probe experiments).

**Empirical record (this iteration, advisory to the downstream triage spec)**: the noise-robustness vs same-gender-discrimination trade-off, the same-gender/similar-timbre blind spot, the detection-window-length sweep, and the speech-enhancement (fix and probe) investigations are documented in research.md. Net: same-gender discrimination needs a stronger discriminator (WavLM-Large SV), not preprocessing or window changes; flag-trust under noise wants SQUIM, not enhancement.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: US1/US2/US3 for story-phase tasks only

## Path Conventions

Single project (senselab): library code under `src/senselab/...`, CLI under `scripts/`, tests under `src/tests/...`.

New package: `src/senselab/audio/workflows/speaker_profile/`. Reuses existing `audio_analysis/{embeddings,clustering,presence,identity}.py` and `tasks/speaker_embeddings/`.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Package scaffolding and the documented-constants module that all later work references.

- [X] T001 Create the `speaker_profile` package skeleton (module docstring + public exports) in `src/senselab/audio/workflows/speaker_profile/__init__.py`
- [X] T002 [P] Define workflow dataclasses (`SpeakerProfile`, `ProfileSourceFile`, `ClusterStats`, `ProfileParams`, `ProfileComparisonResult`, `RecordingQualityIndicator`, `RecordingOtherVoiceSummary`) per data-model.md in `src/senselab/audio/workflows/speaker_profile/types.py` (the two recording-level rollups are internal compute holders whose fields populate the existing `single_speaker`/`quality` claims — not serialized as standalone objects)
- [X] T003 [P] Create the constants module holding every threshold from research.md "Constants & Thresholds" as a named value with a comment giving its value, source (`[reuse]`/`[new]`), and validation status in `src/senselab/audio/workflows/speaker_profile/constants.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Cross-cutting infrastructure required before any story: the WavLM embedding backend (default consensus member, FR-019), shared-cache reuse keying (FR-015/R1), the per-file speech-window extractor, and profile artifact I/O.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T004 Implement the WavLM transformers speaker-embedding backend (`WavLMForXVector`, default `microsoft/wavlm-base-plus-sv`, 16 kHz mono, 512-D, returns `list[torch.Tensor]` matching the SpeechBrain contract) in `src/senselab/audio/tasks/speaker_embeddings/wavlm.py`
- [X] T005 Dispatch a WavLM model handle (alongside `SpeechBrainModel`) and register the compatibility entry in `src/senselab/audio/tasks/speaker_embeddings/api.py` (depends on T004)
- [X] T006 [P] Add WavLM backend tests (loads, embeds, 512-D output, graceful failure → recorded reason not raise) extending `src/tests/audio/tasks/speaker_embeddings_test.py`
- [X] T007 Factor the cache-key "wrapper hash" basis from the analyze_audio script source into a stable shared library helper (module-level hash) so `build_speaker_profile` and `analyze_audio` produce identical keys for diarization/embedding/scene tasks (FR-015/R1) in `scripts/analyze_audio.py` and `src/senselab/audio/workflows/speaker_profile/cache.py` — **partially delivered**: the `cache.py` helper shipped, but the `analyze_audio.py` swap was intentionally deferred "until a second consumer exists" (it still uses `wrapper_version_hash = sha256(script source)`). That consumer now exists; the swap is scheduled in Phase 6 (T035).
- [X] T008 [P] Add a cross-stage cache-reuse test (running the build path then analyze_audio on the same file/params yields `cache: "hit"` for shared tasks) in `src/tests/audio/workflows/speaker_profile/cache_test.py` — **helper-contract only**: `cache_test.py` covers the helper's determinism / caller-agnosticism; the real end-to-end "build → analyze → cache hit" assertion was deferred and is scheduled in Phase 6 (T036).
- [X] T009a Promote the per-window speech mask into a reusable single-file helper (from `compute._speech_window_mask`: diarization + AST/YAMNet speech labels + loudness → per-window speech/non-speech) in `src/senselab/audio/workflows/audio_analysis/presence.py` (prerequisite for T009; resolves the U1 build-time gate)
- [X] T009 Implement the per-file speech-window extractor in `src/senselab/audio/workflows/speaker_profile/build.py`: locate speech via a **best-available presence gate** — cheap floor = diarization (all speakers) + the T009a speech mask; opportunistically fold in cached Whisper `no_speech_prob` / PPG voiced-fraction when already present; **never trigger ASR/PPG solely to gate**. Extract ≥1s windows per model via `extract_per_window_embeddings`, drop sub-1s fragments, tag each `WindowEmbedding` with its `file_id` (FR-002/FR-008 gating; no identity assignment)
- [X] T010 Implement profile artifact load/save (JSON per contracts/speaker-profile.schema.md: `schema_version`, invariants, atomic write, ignore-unknown-keys reader, refuse higher schema) in `src/senselab/audio/workflows/speaker_profile/io.py`
- [X] T010c [P] Add profile-artifact I/O tests in `src/tests/audio/workflows/speaker_profile/io_test.py`: round-trip save/load, atomic write, reader ignores unknown keys, and refuses a higher `schema_version` (per contracts/speaker-profile.schema.md)
- [X] T010a Create the committed synthetic test-audio generator (run once, not in CI) in `scripts/gen_synthetic_test_audio.py`: **SpeechT5** (`microsoft/speecht5_tts` + `microsoft/speecht5_hifigan`, revision-pinned, MIT) with 3 fixed CMU-Arctic x-vector speaker embeddings (A=target, B=intruder, C=similar-timbre spare) + fixed seed; synthesizes public-domain phonetically-rich text (Harvard/IEEE sentences + "The North Wind and the Sun"); writes 16 kHz mono FLAC clips + `manifest.json` (file → speaker_id, transcript, duration, session_id) to `src/tests/data_for_testing/synthetic/`. Produces a confident target subject (3–5 clips, ~30–50s), a thin subject (~15s → low-confidence), an insufficient subject (sub-1s fragments), and standalone intruder/spare clips
- [X] T010b Implement deterministic (seeded) fixture composers in `src/tests/audio/workflows/speaker_profile/conftest.py`: load committed clips via `manifest.json`; compose contamination sets (mix B into A at 10/20/30% — SC-002), other-voice recordings (overlay B on A at known intervals + pure-A control — SC-003/004), and quality-ranked variants (clean A + noise at known SNRs via `data_augmentation` audiomentations — SC-005); expose ground-truth labels (target id, intruder intervals, SNRs)

**Checkpoint**: Embedding backend, cache reuse, speech-mask helper, window extraction, artifact I/O, and labeled synthetic fixtures all in place — story work can begin.

---

## Phase 3: User Story 1 - Build a robust speaker profile for a subject (Priority: P1) 🎯 MVP

**Goal**: For a subject's files, produce exactly one contamination-tolerant profile (dominant-cluster centroid per model) with a usage/preparation record and a confidence indication.

**Independent Test**: Provide a subject's files (some containing a known second voice); confirm exactly one profile is produced, it is closer to clean target audio than to the intruder, non-speech/short files are dropped, and the build needs no manual chunking/silence instructions.

### Tests for User Story 1

- [X] T011 [US1] Add build tests in `src/tests/audio/workflows/speaker_profile/build_test.py` using the T010a–T010b fixtures: one profile + usage record (SC-001/FR-001/FR-004); contamination tolerance with ≤20% intruder, profile closer to held-out target than intruder (SC-002); non-speech/sub-1s files auto-dropped with reason (FR-016); confidence `ok`/`low`/`insufficient` boundaries (FR-005); balanced 50/50 subject → `ambiguous` vs dominant ~85/15 → confident (FR-014/SC-007)

### Implementation for User Story 1

- [X] T012 [US1] Implement cross-file dominant-cluster aggregation → L2-normalized per-model centroid + empirical calibration band, reusing `clustering.cluster_pass_speakers` / `_empirical_calibration_band` over the pooled file-tagged windows, in `src/senselab/audio/workflows/speaker_profile/build.py` (depends on T009)
- [X] T013 [US1] Implement the confidence policy (aggregate speech-seconds vs `min/target_confident_speech_s`; `AMBIGUITY_SHARE_RATIO` for near-equal top-two clusters; `insufficient` → decline) reading thresholds from `constants.py`, in `src/senselab/audio/workflows/speaker_profile/build.py`
- [X] T014 [US1] Implement per-file keep/drop decisions and `ProfileSourceFile` usage records (windows used, speech seconds, kept, drop_reason) (FR-016/FR-004) in `src/senselab/audio/workflows/speaker_profile/build.py`
- [X] T015 [US1] Implement optional same-session weighting (`prefer_session`, up-weight same-session windows; default unweighted; works without session metadata) (FR-013) in `src/senselab/audio/workflows/speaker_profile/build.py`
- [X] T016 [US1] Implement the `build_speaker_profile(...)` orchestration entrypoint tying extraction → aggregation → confidence → records → `io.save` into a `SpeakerProfile`, in `src/senselab/audio/workflows/speaker_profile/build.py`
- [X] T017 [US1] Implement the `build_speaker_profile` CLI per contracts/build-profile-cli.md (positional files + `--files-from`, `--subject-id`, `--output`, model/window/threshold/session flags, shared `--cache-dir`, exit codes, one-line summary) in `scripts/build_speaker_profile.py`

**Checkpoint**: A subject's files → one inspectable profile artifact. MVP complete and testable on its own.

---

## Phase 4: User Story 2 - Flag segments likely containing another voice (Priority: P2)

**Goal**: With a supplied profile, score each analyzed window's similarity to the target and flag likely other-voice regions (foreground/background), integrated into `analyze_audio`'s identity-axis outputs; gated on speech presence; non-breaking when absent.

**Independent Test**: On a recording with known second-speaker intervals, flagged regions overlap them well above chance with low false-positives on target-only audio; non-speech buckets are `unavailable`; leave-one-file-out is applied for contributing files; with no profile, all other outputs are unchanged.

### Tests for User Story 2

- [X] T018 [US2] Add compare tests in `src/tests/audio/workflows/speaker_profile/compare_test.py` using the T010b composers: other-voice detection rate ≥ 2× false-positive on target-only (SC-003); target-only false-flag < 10% duration (SC-004); low-`p_voice` buckets → `unavailable` not `other_voice` (FR-008); leave-one-file-out recomputation applied for a contributing file (FR-012); consensus fusion combines per-model uncertainties; existing `single_speaker` claim extended with profile sub-signals and no PASS/REVIEW verdict (FR-020)

### Implementation for User Story 2

- [X] T019 [US2] Implement leave-one-file-out profile recomputation (exclude scored recording's windows; within-file holdout fallback for single-file subjects) (FR-012/R5) in `src/senselab/audio/workflows/speaker_profile/compare.py`
- [X] T020 [US2] Implement per-window scoring vs profile — consensus fusion of per-model calibrated cosine-uncertainties via `clustering.calibrate_cosine_uncertainty`, on the short-window (~0.5s hop) detection grid for brief-intrusion resolution (FR-017/FR-018/R3/R4) in `src/senselab/audio/workflows/speaker_profile/compare.py`
- [X] T021 [US2] Implement the adaptive other-voice threshold (from the profile's calibration band, with `--profile-other-voice-threshold` fixed override) plus speech-presence gating — reuse the full presence `p_voice` already computed by the analyze_audio run (`unavailable` on low `p_voice`) — and `flag` assignment (FR-008/R6) in `src/senselab/audio/workflows/speaker_profile/compare.py`
- [X] T022 [US2] Add `--speaker-profile` and `--profile-other-voice-threshold` inputs, load+validate the profile (treat `insufficient` as absent with a warning), and match the analyzed file against `sources[]` for leave-one-file-out, in `scripts/analyze_audio.py` (FR-007/FR-011)
- [X] T023 [US2] Integrate profile votes into the identity axis (`model_votes["speaker_profile/<model>"]` + `["speaker_profile/consensus"]` carrying similarity/other_voice_uncertainty/flag) in `src/senselab/audio/workflows/audio_analysis/identity.py` (FR-009)
- [X] T024 [US2] Emit the per-pass `<pass>/speaker_profile.json` sidecar (per-window flags, matching the `embeddings/*.json` convention), let profile-flagged buckets ride the existing `disagreements.json` ranking, and guard the no-profile path so non-profile outputs are byte-identical, in `scripts/analyze_audio.py` (FR-009/FR-011)
- [X] T024a [US2] Extend the existing per-pass `single_speaker` global-summary claim with profile sub-signals (`profile_other_voice_fraction`/`_seconds`/`_peak`/`_p95`, `profile_speech_present_seconds`, `profile_confidence`) and fold a profile-based uncertainty into its headline via the existing `max()`/intensity-weighted aggregation — decision-ready, no verdict (FR-020), computed as a `RecordingOtherVoiceSummary` (types.py) whose fields populate the claim, in `src/senselab/audio/workflows/audio_analysis/global_summary.py` (wire profile inputs from `scripts/analyze_audio.py`)

**Checkpoint**: US1 + US2 both work independently; other-voice flags appear alongside existing analysis.

---

## Phase 5: User Story 3 - Estimate target-speaker recording quality (Priority: P3)

**Goal**: With a supplied profile, produce a per-recording target-speaker quality indicator reflecting how cleanly the target voice is captured, attached to `analyze_audio` output.

**Independent Test**: On recordings independently graded for target-capture quality, the indicator ranks them in the same broad order; a clean target-dominant recording scores higher than a noisy/contaminated one.

### Tests for User Story 3

- [X] T025 [US3] Add quality tests in `src/tests/audio/workflows/speaker_profile/compare_test.py` using the T010b quality-ranked variants: clean target-dominant recording outranks noisy/contaminated one (SC-005); profile sub-signals present under the existing `quality` claim; quality discounted/ignored when profile confidence is `low`/`insufficient`

### Implementation for User Story 3

- [X] T026 [US3] Implement `RecordingQualityIndicator` (target_match_fraction = 1 − other-voice rate over speech-present duration; mean within-profile consistency on matched windows; mean SQUIM STOI/PESQ/SI-SDR reused from `analyze_audio` on matched windows; normalized [0,1] `quality`) (FR-010/R7) in `src/senselab/audio/workflows/speaker_profile/compare.py`
- [X] T027 [US3] Extend the existing per-pass `quality` global-summary claim with profile target-quality sub-signals (`profile_target_quality`, `profile_target_match_fraction`, `profile_mean_target_consistency`, target-matched `profile_squim`, `profile_confidence`) and fold a target-quality uncertainty into its headline via the existing aggregation; include the detail in the per-pass `speaker_profile.json` sidecar, in `src/senselab/audio/workflows/audio_analysis/global_summary.py` (wire from `scripts/analyze_audio.py`)

**Checkpoint**: All three stories independently functional.

---

## Phase 6: Cross-Stage Cache Reuse (finish FR-015 / R1) — ⏸️ DEFERRED (2026-06-04)

> **Status: DEFERRED to real-data deployment** (close-out decision 2026-06-04). The library-side helper (T007) and its unit contract (T008) shipped in Phase 2; the remaining wiring (T033–T036) is **not** done. Rationale: the keying swap (T035) changes cache-invalidation semantics for *all* `analyze_audio` users and forces a one-time global cache miss, and the current script-source hash is *safe* (over-invalidates, never serves stale). The recompute-once performance premise does not pay off during synthetic-data characterization; defer until real-data use makes it worthwhile and the `_TASK_MODULES` map can be completed without under-invalidation risk (the live task surface — `ast`/`yamnet`/`ppgs`/`alignment` — is not yet fully mapped). FR-015 is therefore designed and partially delivered (helper only), not wired or verified. See spec.md Clarifications 2026-06-04.

**Goal**: Make the profile stage and `analyze_audio` actually share cached per-file tasks (diarization, speaker embeddings, scene classification) so running `build_speaker_profile` beforehand spares `analyze_audio` from recomputing them — the performance premise of the two-stage design (R1).

**Why this is in scope (and why now)**: The `cache.py` helper (T007) and its unit contract (T008) shipped in Phase 2, but the author deliberately deferred swapping `analyze_audio.py`'s script-source wrapper hash "until the second consumer exists." That consumer (`build_speaker_profile`) now exists (Phase 3) and `analyze_audio` consumes its artifact (Phase 4), so FR-015/R1 is currently *designed but not delivered or verified*. This phase is sequenced **after US3 (Phase 5)** so its `analyze_audio.py` edits don't collide with US2/US3's edits to the same file, and **before Polish** so the end-to-end and gate runs (T031/T032) cover it.

**⚠️ Blast radius**: This changes cache-invalidation semantics for *all* `analyze_audio` users (not just the profile path) and triggers a one-time global cache miss. Treat it as a deliberate, standalone change — not folded into a story commit. The failure mode to guard against is the inverse of today's: the current script-source hash *over*-invalidates (safe but wasteful); a library-module hash can *under*-invalidate (serve a stale entry) if the task→module map is incomplete.

- [ ] T033 ⏸️ DEFERRED — Audit and complete the task→module map (`_TASK_MODULES`) in `src/senselab/audio/workflows/speaker_profile/cache.py`: for each cached task (diarization, speaker_embeddings, classification/scene, features, asr) include the full set of behavior-determining library modules (api + backends + helpers) so the library-derived hash invalidates on a real implementation change and never under-invalidates. Add a test asserting every mapped module resolves to a real importable file.
- [ ] T034 ⏸️ DEFERRED — Bump `CACHE_SCHEMA_VERSION` (`cache.py`) and `analyze_audio.py`'s `_CACHE_SCHEMA_VERSION` in lockstep so the keying change triggers one clean, intentional cache invalidation; record the bump + reason in both files.
- [ ] T035 ⏸️ DEFERRED — Swap `analyze_audio.py`'s per-task cache keying from `wrapper_version_hash()` (sha256 of the script source) to `cache.task_wrapper_hash(<task>)` per task, so each task's key reflects its implementing modules and is caller-agnostic (build ↔ analyze share entries). Keep the `cache_key` payload shape aligned with `cache.py`; preserve an explicit override path for staged rollout. Depends on T022/T024/T027 (the US2/US3 `analyze_audio.py` edits) landing first.
- [ ] T036 ⏸️ DEFERRED — Implement the deferred end-to-end cross-stage reuse test (the real T008 validation hook): run `build_speaker_profile` then `analyze_audio` on the same file with identical task params and assert `cache: "hit"` for the shared tasks (diarization, speaker embeddings, scene classification), in `src/tests/audio/workflows/speaker_profile/cache_test.py` (or a new `cache_integration_test.py`).

**Checkpoint**: Running `build_speaker_profile` then `analyze_audio` on a subject's files recomputes each shared task once, not twice.

---

## Phase 7: Polish & Cross-Cutting Concerns

- [X] T028 [P] **Characterize** (do not lock in) the `[new]` thresholds via a sensitivity sweep against the T010a–T010b fixtures (`AMBIGUITY_SHARE_RATIO`; `min/target_confident_speech_s`; `OTHER_VOICE_CALIBRATED_CUTOFF`; consensus fusion weights; sub-1s intrusion boundary; `min_contiguous_speech_s`): show which thresholds materially move the outputs and over what range. Per the 2026-06-03 clarification, keep defaults **provisional and configurable** — synthetic-derived values are NOT assumed to transfer to real data, so do not finalize production thresholds or tune a recall-biased operating point here. Record the sensitivity findings + caveat in `constants.py` and research.md "Constants & Thresholds". Run on a GPU compute node (sbatch), not the login node.
- [ ] T028b [P] ⏸️ DEFERRED (2026-06-04) (Optional, research) Implement per-window confidence weighting of the profile centroid — down-weight windows by Whisper `no_speech_prob`/avg_logprob, PPG voiced-fraction (opt-in given its ~1.4 GB venv), and SQUIM — flag-gated and evaluated against fixtures, in `src/senselab/audio/workflows/speaker_profile/build.py`. **Deferred**: optional-research; the 2026-06-04 enhancement-probe experiments concluded SQUIM-gated trust is the right per-window-reliability lever (enhancement-delta is redundant or non-discriminative). Any future per-window weighting belongs to the cross-audio/triage spec, built on SQUIM. See research.md + spec.md Clarifications 2026-06-04.
- [X] T029 [P] Add regression test asserting `analyze_audio` without `--speaker-profile` yields byte-identical non-profile outputs vs. a baseline run (SC-006) in `src/tests/audio/workflows/speaker_profile/regression_test.py`
- [X] T030 [P] Author module documentation in `src/senselab/audio/workflows/speaker_profile/doc.md` (purpose, pipeline, constants, caching note)
- [X] T031 Run the quickstart.md end-to-end validation (build → analyze → review) and the success-criteria smoke checks
- [X] T032 Run full quality gates (`ruff`, `mypy`, `pytest`) across all changed modules and fix findings

---

## Phase 8: Reuse & identity-axis integration (PR #523 review) — 🔜 PLANNED (2026-06-05)

> **Driven by the PR #523 maintainer review.** Verified findings, the **(C) hybrid decision**, and the continuous-signal design are in research.md "PR #523 review — reuse/altitude refactor design" + "Signal design (decided 2026-06-05)". Likely a **follow-up PR** (not #523). The embedding-cache tasks (T048–T049) are the same gap as the deferred Phase 6 (FR-015); land them together. Integration model **decided = C** (T040); still worth confirming with the maintainer but not blocking the unconditional fixes.

### Decision (resolved)

- [X] T040 **Integration model = (C) hybrid, recall-primary** (resolved 2026-06-05; confirm with maintainer). Profile feeds the identity axis as a corroborating reference-based voter **and** keeps an independent presence-gated per-window signal (fires where identity isn't measured, e.g. background voice in non-speech regions); target-quality stays its own claim. "Wrong subject" is a **continuous certainty**, not a flag. Recorded in research.md; `analyze-audio-profile` contract to be updated in T044.

### Unconditional reuse fixes (independent of integration shape) — safe to start now

- [X] T041 Shared `presence.reference_grid_and_speech_mask(...)` (pick reference grid + speech mask, returns `(reference_windows, mask|None)`) called from BOTH `compute.py` (identity-axis clustering) and `speaker_profile/build.extract_speech_windows_for_file`, enforcing the FR-002 "same signal" by one code path. Behavior-preserving (mask is grid-based / vector-independent → compute-once equals the former per-model recompute, which it also removes); each caller keeps its None-fallback. ruff+mypy clean; GPU run confirmed **33/33** incl. `compute_uncertainty_axes_test` (core path) + `embeddings_test` + the profile tests. (commit `642594ae`)
- [X] T042 [P] Promote one shared cosine helper (`embeddings.cos_sim`/`cos_dist`, single degenerate-input contract, `cos_dist(clip=)` covers identity `[0,1]` vs profile unclipped) and replace the three copies (`compare._cosine_distance`, `clustering._cos_sim`, `identity._cosine_similarity`/`_cos_dist`). **Verified**: numerically equivalent (compare/clustering bit-identical; identity 1-ULP, neutral under within-run SC-006), ruff+mypy clean; GPU run confirmed **22/22** model-dependent tests pass — `regression_test` (SC-006 byte-identity), `compare_test`, `success_criteria_test` (SC-002/003/004/005). (commit `4d413184`)
- [X] T043 [P] Single canonical `DEFAULT_SPEECH_PRESENCE_LABELS` in `audio_analysis/presence.py`, imported by `analyze_audio.py` + `speaker_profile/build.py` (removed the sync-by-comment duplicate). `build_speaker_profile --cache-dir` left documented as reserved (real wiring is T049). (commit `fa258eae`)

### Signal shape — continuous composable atoms (per the decided design)

- [X] T044 Continuous atoms confirmed/shaped: per-window `subject_similarity` / `other_voice_uncertainty` are already continuous (`calibrate_cosine_uncertainty`) with the discrete `flag` derived from the cutoff; recording-level certainties are emitted **paired with `profile_confidence`** and stay **within-profile calibrated only**. (Contract doc note pending in the same pass as T047.) *(per-window `voice_present` is conveyed today via the gate/`flag`; a dedicated field can be added if a consumer needs it.)*
- [X] T045 `summarize_other_voice` emits the **continuous `profile_subject_dominance`** (voiced-time-weighted mean subject similarity; complement = wrong/absent-subject uncertainty), `None` when nothing scorable; surfaces as a distinct sub-signal via the existing splat; `nonsubject_voice_fraction / peak / p95` retained as max-like recall atoms. GPU-verified **33/33** (incl. SC-006 byte-identity, compare, success-criteria). (commit `11e32a4d`)
- [X] T046 Scene-based **voice-presence scoring gate** (`compare_recording_to_profile(voice_present_by_window=…)` from `reference_grid_and_speech_mask` on the analyzed recording) — catches background/secondary voice, excludes cough/breath/silence; falls back to `p_voice` when scene classification absent. Dependency kept **acyclic** (profile fed forward, not back into the presence gate). GPU-verified **33/33** incl. SC-006 (profile-path-only/additive: no-profile path byte-identical). (commit `ffa2b1f0`)
- [X] T047 **(C) integration done.** `aggregate_identity` now folds the per-bucket `speaker_profile/consensus` other-voice uncertainty as a real reference-based voter (per-model entries display-only, no double-count); `analyze_audio` re-aggregates each identity bucket after injecting the profile votes, so the **identity parquet + disagreements + Label Studio** reflect the profile instead of carrying inert votes. Named profile sub-signals stay exposed; the independent presence-gated per-window flag is retained; the p95 recall fold is kept as an explicit spike-preserver (max, no double-count). **SC-006 rewritten** to its real invariant (no-profile path byte-identical to baseline; a profile may change only the speaker-identity/quality surface — identity uncertainty/votes/status, single_speaker & quality headlines + identity_axis_mean, disagreements + LS ordering, the sidecar) and **GPU-verified green** (regression + aggregate + compute + embeddings + compare + success-criteria). commits `a0140dad`,`077f4dc0`,`9d1acab7`. **Deferred to a follow-up:** the **diarization-overlap corroborator** (profile-distance is unreliable on mixed windows) — scoped out of this commit to keep it reviewable; tracked as T047a below.
- [ ] T047a (follow-up) Diarization-overlap corroborator: where diarization detects overlapping speech, corroborate/boost the other-voice signal (the profile-distance path is unreliable on mixed-speaker windows — same physics as the same-gender blind spot). Additive to the (C) integration.

### Embedding-compute reuse (== Phase 6 / FR-015 — land together)

- [X] T048 **Handled by the T049 cache** (no artifact vector storage / schema bump needed). The leave-one-file-out path re-extracts siblings at the build grid (2.0/1.0); with the embedding cache (T049) and a shared `--cache-dir`, those re-extractions hit build's cache entries → no re-embedding. Chose the cache over bloating the profile JSON with per-window vectors.
- [X] T049 **Done + GPU-verified.** `extract_per_window_embeddings` gains opt-in `cache_dir`; each model's window list is cached per `(audio_sig, model, window_s, hop_s)` via `cache.py`'s caller-agnostic key. Delivered **without the global keying swap (T035)** — embeddings was an *uncached* task, so a new caller-agnostic cache touches no existing task's keying (no blast radius). float32↔JSON round-trips bit-exactly → **cached == uncached centroids verified bit-exact on GPU**; cache populates (files×models entries) and a 2nd build skips inference. Added `embeddings.py` to the `speaker_embeddings` module hash (T033 under-invalidation guard for this task). Wired through build (`--cache-dir` now active), the analyze LOO path (cross-stage hit), and the identity-axis extraction (re-run reuse). commits `01c2a696`,`915976e3`.

### Deferred hooks (emit raw ingredients now; solve in the triage/metric spec)

- [ ] T050 ⏸️ DEFERRED — Non-subject **cough/breath attribution**: speech speaker-embeddings are OOD on non-speech, so don't force a subject/non-subject call on coughs. Emit the raw ingredients (scene `cough`/`breath` label + the segment's diarization speaker) so a downstream combiner can attempt attribution later.
- [ ] T051 ⏸️ DEFERRED — **Task-aware contextualization** ("voiced speech in a respiration task is anomalous regardless of who"): combines BIDS task metadata + scene classification downstream; this feature only ensures `voice_present` + scene labels are emitted so it stays composable.

**Checkpoint**: profile emits continuous, confidence-paired, within-profile-calibrated atoms (subject-identity certainty, subject_dominance/wrong-subject, nonsubject-voice fraction/peak/p95) — exposed distinctly, fed forward into the identity axis (C) and an independent voice-gated flag — with extraction/cosine/label logic shared, the same audio embedded once per grid, and cough/task-context left as composable hooks.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: no dependencies.
- **Foundational (Phase 2)**: depends on Setup; BLOCKS all stories. Within it: T004→T005; T007→T008; T009a→T009; T009 depends on T002/T003; T010 depends on T002; T010a→T010b (fixtures); T010a may run in parallel with T004–T009.
- **User Stories (Phase 3–5)**: all depend on Foundational. US2 builds on US1's profile artifact; US3 builds on US2's per-window comparison. They can be developed in parallel but share `compare.py` (US2/US3), so coordinate those.
- **Cross-stage cache reuse (Phase 6)**: depends on US2/US3 having landed their `analyze_audio.py` edits (T022/T024/T027), since T035 swaps the cache keying in the same file; standalone from the story logic otherwise. Must land before Polish so T031/T032 exercise it.
- **Polish (Phase 7)**: depends on the stories it touches (T028 after US1/US2 thresholds exist; T029 after US2 wiring) and on Phase 6 (T031 end-to-end / T032 gates cover the cache wiring).

### User Story Dependencies

- **US1 (P1)**: after Foundational. No dependency on US2/US3. ← MVP.
- **US2 (P2)**: after Foundational; consumes a US1 profile artifact (can test against a fixture profile to stay independent).
- **US3 (P3)**: after Foundational; reuses US2's matched-window scoring (can test against fixture comparison output to stay independent).

### Within Each User Story

- Story tests written first and failing → implementation.
- `build.py` tasks (T012–T016) are sequential (same file).
- `compare.py` tasks (T019–T021, T026) are sequential (same file).
- `scripts/analyze_audio.py` tasks (T022, T024, T027, then T035) are sequential (same file); T035 (cache keying swap) lands last, after the US2/US3 wiring.

### Parallel Opportunities

- Setup: T002, T003 in parallel.
- Foundational: T006 ∥ T008 (different test files); T004/T005 (one file chain) can run alongside T009/T010 only after T002/T003.
- Polish: T028 ∥ T029 ∥ T030 (different files).
- Story test-authoring tasks (T011, T018, T025) target different/again files and can be drafted in parallel with the prior story's implementation.

---

## Parallel Example: Setup + Foundational

```bash
# Phase 1 — parallel:
Task: "Define workflow dataclasses in src/senselab/audio/workflows/speaker_profile/types.py"   # T002
Task: "Create documented constants module in .../speaker_profile/constants.py"                  # T003

# Phase 2 — parallel test authoring (different files):
Task: "WavLM backend tests in src/tests/audio/tasks/speaker_embeddings_test.py"                  # T006
Task: "Cross-stage cache-reuse test in src/tests/audio/workflows/speaker_profile/cache_test.py" # T008
```

---

## Implementation Strategy

### MVP First (User Story 1 only)

1. Phase 1 Setup → 2. Phase 2 Foundational (note: WavLM T004–T006 can be deferred if the MVP profile runs ECAPA+ResNet only with graceful degradation) → 3. Phase 3 US1 → 4. **STOP and VALIDATE** the profile artifact against T011 → 5. Demo.

### Incremental Delivery

1. Setup + Foundational → foundation ready.
2. US1 → build + inspect profiles (MVP).
3. US2 → other-voice flags in `analyze_audio`.
4. US3 → target-speaker quality indicator.
5. Cross-stage cache reuse → `build_speaker_profile` and `analyze_audio` share cached tasks (finish FR-015/R1).
6. Polish → finalize thresholds, regression-lock SC-006, docs, gates.

---

## Notes

- [P] = different files, no incomplete dependencies.
- Per the team preference, every threshold lands in `constants.py` as a named, documented, configurable value — `[new]`/TBD ones carry a "validate empirically (T028)" comment.
- WavLM has no official Large-SV checkpoint; default `microsoft/wavlm-base-plus-sv`, configurable.
- Keep `analyze_audio`'s no-profile path byte-identical (SC-006, regression-tested in T029).
- Commit after each task or logical group; stop at any checkpoint to validate a story independently.
