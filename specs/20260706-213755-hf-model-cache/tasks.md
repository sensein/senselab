# Tasks: HuggingFace Model Cache & Version Consistency

**Input**: Design documents from `/specs/20260706-213755-hf-model-cache/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/model-cache-api.md, quickstart.md

**Tests**: INCLUDED — the constitution mandates Test-Driven Development (NON-NEGOTIABLE) and the plan's Constitution Check commits to tests-first. Each behavior gets a failing test before implementation; each backend migration is guarded by a behavior-preserving test.

**Organization**: By user story (US1–US4 from spec.md), in priority order. US1/US2 are P1; US3/US4 are P2.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependency on an incomplete task)
- Paths are single-project: `src/senselab/…`, `src/tests/…`

**Convention**: run everything via `uv` (Constitution II): `cd src && uv run pytest …`, `uv run ruff check .`, `uv run mypy`. Behavioral tests that need GPU/network are marked and run on SLURM; unit tests monkeypatch `HfApi.model_info` and use a temp HF cache (no network).

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the module and test scaffolding for the shared mechanism.

- [ ] T001 Create `src/senselab/utils/hf_model_cache.py` module skeleton with docstring and the named, documented constants: `DEFAULT_FRESHNESS_WINDOW_DAYS = 7`, `SCHEMA_VERSION = 1`, and the env-var names `SENSELAB_HF_FRESHNESS_DAYS`, `SENSELAB_HF_FREEZE` (reuse existing `SENSELAB_HF_MAX_RETRIES`). Document each value + rationale at the definition site (Constitution III / FR-012).
- [ ] T002 [P] Create `src/tests/utils/hf_model_cache_test.py` with shared fixtures: a temp HF cache dir (`monkeypatch` of `HF_HOME`/`HF_HUB_CACHE`), a `model_info` call-counter stub, and a helper to fabricate a `snapshots/<sha>/config.json` cache layout offline.
- [ ] T003 [P] Add module-level `logging.getLogger("senselab")` usage note and confirm `ruff`/`mypy` clean on the empty skeleton (`cd src && uv run ruff check senselab/utils/hf_model_cache.py && uv run mypy senselab/utils/hf_model_cache.py`).

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: The cache record + resolver core + freeze plumbing that ALL user stories depend on.

**⚠️ CRITICAL**: No user-story work can begin until this phase is complete.

- [ ] T004 [P] Write failing tests for `CachedModelRecord` I/O in `src/tests/utils/hf_model_cache_test.py`: round-trip write/read, `schema_version` present, unknown `schema_version` → treated as absent, torn/corrupt JSON → treated as absent.
- [ ] T004a [P] Write failing lock-safety test in `src/tests/utils/hf_model_cache_test.py`: a lock whose heartbeat has lapsed but whose **owner process is still alive** must NOT be broken — assert a waiter never acquires concurrently with a live owner (no double-holder), reproducing the pre-#527 `_HeartbeatLock` unlink-break cache-corruption race (research D7).
- [ ] T005 Implement `CachedModelRecord` (dataclass + `schema_version`) with atomic write-then-`os.replace` read/write in `src/senselab/utils/hf_model_cache.py`, superseding `_read_result_cache`/`_write_result_cache` and adding `resolved_sha` + `last_verified_epoch` (data-model E3).
- [ ] T006 [P] Write failing test for `ResolvedModel` shape (full-SHA invariant, `snapshot_path`, `from_cache`) in `src/tests/utils/hf_model_cache_test.py`.
- [ ] T007 Implement `ResolvedModel` dataclass in `src/senselab/utils/hf_model_cache.py` (data-model E2).
- [ ] T008 Implement cache-layer identity helpers in `src/senselab/utils/hf_model_cache.py`: `resolve_local_sha(repo_id, ref)` (read `refs/<ref>` / `try_to_load_from_cache(...).parent.name`, compare against `_CACHED_NO_EXIST` sentinel) and `snapshot_dir_for(repo_id, sha)`; detect partial/corrupt snapshot via `scan_cache_dir().warnings` **and treat a corrupt/partial snapshot as absent so `resolve_model` re-obtains it** (research D2/D5 gotcha 6). Resolve the SHA **independently** (`model_info` / `refs/<ref>`) — never trust `ensure_hf_model`'s return, which can be a branch name (research D7) — and never let `HF_HUB_OFFLINE=1` fabricate cache presence for an absent snapshot.
- [ ] T009 Implement `resolve_model(repo_id, ref="main", *, repo_type="model", token=None)` orchestration skeleton in `src/senselab/utils/hf_model_cache.py`: freeze-precedence → record-freshness → coordinated re-check → stage → verify (each branch stubbed to be filled by US1–US3), reusing `ensure_hf_model`, `_HeartbeatLock`, `retry_on_transient_error` from `dependencies.py`. Build the coordination core as a **provider-agnostic** `ensure_cached_resource(key, fetch_fn, version_probe=None)` with the HuggingFace adapter (`model_info(...).sha` as `version_probe`) layered on top, so future non-HF Tier-B providers can reuse the download-once/offline/freeze core (see plan.md "Extensibility"). Only the HF adapter is delivered in this feature.
- [ ] T010 Re-export the new public surface (`resolve_model`, `ResolvedModel`, `load_hf_resilient`, `hf_subprocess_env`, `run_version_freeze`) from `src/senselab/utils/dependencies.py` so existing import paths keep working (back-compat).
- [ ] T010a Replace `_HeartbeatLock`'s unlink-based stale-break with a **safe lease** in `src/senselab/utils/dependencies.py`: record owner identity (pid/host) alongside the heartbeat, take over a lock **only when the owner is provably gone** (never unlink a lock held by a live process), and bound the total wait. This is the coordination primitive `resolve_model`'s download/re-check reuse, so it must not corrupt the cache under CPU-starved holders (research D7; makes T004a pass).

**Checkpoint**: Record store + resolver scaffold + a safe cross-process lock exist; user stories can now be implemented.

---

## Phase 3: User Story 1 - Reliable model loading under heavy parallelism (Priority: P1) 🎯 MVP

**Goal**: Many concurrent jobs load the same model with zero 429s and no missing outputs — the originating problem.

**Independent Test**: Simulate N concurrent loads of one already-cached model; assert all succeed, `model_info` is called zero times, and `snapshot_download` at most once for a first-time model.

### Tests for User Story 1 ⚠️ (write first, must fail)

- [ ] T011 [P] [US1] Test `resolve_model` performs zero network calls when the SHA snapshot is cached and record is fresh (`test_fresh_within_window_no_network`) in `src/tests/utils/hf_model_cache_test.py` (C1/SC-002).
- [ ] T012 [P] [US1] Test download-once under concurrency: N threads first-time-load one model → exactly one `snapshot_download` (`test_download_once_concurrent`) in `src/tests/utils/hf_model_cache_test.py` (C2/SC-003).
- [ ] T012a [P] [US1] Test corrupt/partial cache is detected and re-obtained: fabricate a dangling-symlink / `.incomplete` snapshot → assert `resolve_model` re-stages rather than loading a broken model or silently succeeding (`test_corrupt_cache_reobtained`) in `src/tests/utils/hf_model_cache_test.py` (spec Edge Cases).
- [ ] T013 [P] [US1] Test in-process loads use `local_files_only=True` + full-SHA `revision` and never rely on `HF_HUB_OFFLINE` env toggling (`test_local_only_sha_pinned_load`) in `src/tests/utils/hf_model_cache_test.py` (research D1).
- [ ] T013a [P] [US1] Test execution-context parity: the same cached model resolved for an in-process load (`load_hf_resilient`) and for a subprocess worker (`hf_subprocess_env`) yields the same resolved SHA and both make zero Hub calls (`test_inprocess_subprocess_parity`) in `src/tests/utils/hf_model_cache_test.py` (FR-010).

### Implementation for User Story 1

- [ ] T014 [US1] Implement the cached-and-fresh fast path in `resolve_model` (return `from_cache=True`, no network) in `src/senselab/utils/hf_model_cache.py`.
- [ ] T015 [US1] Implement download-once staging in `resolve_model`: `_HeartbeatLock` + `snapshot_download(repo_id, revision=<sha>)` with double-checked re-read, returning `snapshot_path` (`src/senselab/utils/hf_model_cache.py`).
- [ ] T016 [US1] Rework `load_hf_resilient(loader, *args, repo_id, ref, pass_revision=True, **kwargs)` to resolve first, inject `revision=resolved_sha, local_files_only=True` (or hand back `snapshot_path` when `pass_revision=False`), and retry on transient errors — remove reliance on `hf_offline_loading` env toggling (`src/senselab/utils/hf_model_cache.py`).
- [ ] T017 [US1] Rework `hf_subprocess_env` to resolve each referenced model via `resolve_model` and pass the resolved SHA hint + offline env to the child (fresh-import → correct), falling back to unchanged env if staging fails (`src/senselab/utils/dependencies.py`).

**Checkpoint**: Concurrent cached loads make zero Hub calls; first-time loads download once. MVP behavior deliverable.

---

## Phase 4: User Story 2 - Correct, verified model version (Priority: P1)

**Goal**: The loaded model is provably the requested version, or the load fails loudly — never a silent substitution.

**Independent Test**: Request a specific version → confirm loaded SHA matches; request a bogus version → clear error naming model + version.

### Tests for User Story 2 ⚠️ (write first, must fail)

- [ ] T018 [P] [US2] Test version mismatch/unavailable fails loud naming repo+ref, never silently substitutes (`test_version_mismatch_fails_loud`) in `src/tests/utils/hf_model_cache_test.py` (C4/SC-004).
- [ ] T019 [P] [US2] Test missing repo/ref → `RepositoryNotFoundError`/`RevisionNotFoundError` naming repo+ref (`test_missing_model_clear_error`) in `src/tests/utils/hf_model_cache_test.py` (C6/FR-008).
- [ ] T020 [P] [US2] Test `GatedRepoError` is raised and NOT cached as a definitive failure (`test_gated_not_cached`) in `src/tests/utils/hf_model_cache_test.py` (C6, research gotcha 4).
- [ ] T021 [P] [US2] Test two distinct refs of one repo resolve to distinct records/paths (`test_two_refs_no_crosstalk`) in `src/tests/utils/hf_model_cache_test.py` (C7/FR-011).

### Implementation for User Story 2

- [ ] T022 [US2] Implement cache/hub-layer version verification in `resolve_model`: after staging, assert local snapshot SHA == resolved SHA; on mismatch/absence raise a clear error naming repo+ref (`src/senselab/utils/hf_model_cache.py`) (FR-003/015).
- [ ] T023 [US2] Implement definitive-failure caching (Repository/Revision not found) and the gated-repo exception (never cache `GatedRepoError`) in `resolve_model` error handling (`src/senselab/utils/hf_model_cache.py`) (FR-008).
- [ ] T024 [US2] Ensure per-`(repo_id, ref)` record/path keying so distinct refs never cross-contaminate (`src/senselab/utils/hf_model_cache.py`) (FR-011).

**Checkpoint**: Version correctness guaranteed or loud failure; US1+US2 both independently testable.

---

## Phase 5: User Story 3 - Bounded freshness without per-load hub calls (Priority: P2)

**Goal**: Reuse the known version for a window; re-check exactly once when it lapses; freeze scopes for reproducibility.

**Independent Test**: Move upstream "latest"; verify cached version is used until the window elapses, then one coordinated re-check adopts the new SHA; verify run/system freezes hold versions steady.

### Tests for User Story 3 ⚠️ (write first, must fail)

- [ ] T025 [P] [US3] Test stale window triggers exactly one coordinated `model_info` across N threads (`test_stale_triggers_single_recheck`) in `src/tests/utils/hf_model_cache_test.py` (C3/SC-005).
- [ ] T026 [P] [US3] Test a changed upstream SHA is adopted (download + record rewrite) after the window, not before (`test_changed_sha_adopted`) in `src/tests/utils/hf_model_cache_test.py` (C3).
- [ ] T027 [P] [US3] Test Hub-unreachable during a due re-check falls back to cached SHA + warning, no hard failure (`test_hub_unreachable_uses_cache`) in `src/tests/utils/hf_model_cache_test.py` (C5/SC-007).
- [ ] T028 [P] [US3] Test system freeze `SENSELAB_HF_FREEZE=1` → zero Hub calls (`test_system_freeze_no_network`) in `src/tests/utils/hf_model_cache_test.py` (C8/SC-009).
- [ ] T029 [P] [US3] Test `run_version_freeze()` holds SHAs constant across a window lapse within the block (`test_run_freeze_stable_mid_run`) in `src/tests/utils/hf_model_cache_test.py` (C8/SC-008).

### Implementation for User Story 3

- [ ] T030 [US3] Implement the freshness-window gate in `resolve_model` (`now - last_verified_epoch < window`, default 7d via `SENSELAB_HF_FRESHNESS_DAYS`) with the coordinated single re-check via `_HeartbeatLock` + double-checked re-read + `model_info(expand=["sha"])` (`src/senselab/utils/hf_model_cache.py`) (FR-005/006).
- [ ] T031 [US3] Implement changed-SHA adoption (stage new SHA, rewrite record) and timestamp-bump-on-unchanged in `resolve_model` (`src/senselab/utils/hf_model_cache.py`).
- [ ] T032 [US3] Implement Hub-unreachable fallback: on transient/auth error during a due re-check, return cached SHA + `logger.warning`, do not overwrite a good record (`src/senselab/utils/hf_model_cache.py`) (FR-007, research gotcha 4).
- [ ] T033 [US3] Implement system freeze (`SENSELAB_HF_FREEZE`) short-circuit at the top of `resolve_model` (`src/senselab/utils/hf_model_cache.py`) (FR-014).
- [ ] T034 [US3] Implement `run_version_freeze()` context manager backed by a `ContextVar` dict `(repo_id,ref)->sha`; wire its lookup into `resolve_model` with precedence system ⊇ run ⊇ default (`src/senselab/utils/hf_model_cache.py`) (FR-013).

**Checkpoint**: Freshness + freeze semantics complete; US1–US3 independently testable.

---

## Phase 6: User Story 4 - One consistent mechanism across all model backends (Priority: P2)

**Goal**: Every HF-backed backend loads through the shared mechanism; no bespoke caching/version code remains (strangler-fig, behind unchanged public APIs, per-backend behavior-preserving tests).

**Independent Test**: `grep` finds no backend-specific caching/offline/version code; each backend's public output is unchanged vs baseline; a new backend needs only its model-specific load step.

> Each migration task: write/confirm a behavior-preserving test first, then re-point the loader onto `resolve_model`/`load_hf_resilient`/`hf_subprocess_env`. Tasks touch different files → mostly [P]. GPU/network behavior checks run on SLURM.

### Re-point already-migrated backends onto the reworked API

- [ ] T035 [P] [US4] Migrate `src/senselab/utils/data_structures/model.py` (`HFModel` cache path) from `ensure_hf_model` to `resolve_model`, exposing the resolved SHA; test in `src/tests/utils/data_structures/model_test.py`.
- [ ] T036 [P] [US4] Re-point `src/senselab/audio/tasks/speech_to_text/huggingface.py` (`load_hf_resilient` pipeline) to the reworked signature; test `src/tests/audio/tasks/speech_to_text/huggingface_no_timestamps_test.py`.
- [ ] T037 [P] [US4] Re-point `src/senselab/audio/tasks/speech_to_text/granite.py` (was `hf_offline_loading`) to `load_hf_resilient` with `revision=sha`; test `src/tests/audio/tasks/speech_to_text/granite_test.py`.
- [ ] T038 [P] [US4] Re-point `src/senselab/audio/tasks/forced_alignment/forced_alignment.py` MMS + per-language loaders (was `hf_offline_loading`) to `load_hf_resilient`; test `src/tests/audio/tasks/forced_alignment_test.py` + `forced_alignment/mms_test.py`.
- [ ] T039 [P] [US4] Confirm subprocess workers `nemo.py`, `canary_qwen.py`, `qwen.py` use the reworked `hf_subprocess_env` (SHA-pinned); tests `src/tests/audio/tasks/speech_to_text/{canary_qwen_test,qwen_test,granite_test}.py` and `speech_to_text_test.py`.

### Migrate SpeechBrain backends to SHA-pinned local snapshot loads

- [ ] T040 [P] [US4] Migrate `src/senselab/audio/tasks/speaker_embeddings/speechbrain.py` to load from `resolve_model(...).snapshot_path` (`source=snapshots/<sha>/`, keep `speechbrain_savedir`/`speechbrain_loading_cwd`, `local_strategy=COPY`); test `src/tests/audio/tasks/speaker_embeddings_test.py`.
- [ ] T041 [P] [US4] Migrate `src/senselab/audio/tasks/speech_enhancement/speechbrain.py` (SpectralMask/Sepformer selection) to SHA-pinned local path; test `src/tests/audio/tasks/speech_enhancement_test.py`.
- [ ] T042 [P] [US4] Migrate the SpeechBrain SSL loader in `src/senselab/audio/tasks/ssl_embeddings/self_supervised_features.py` to SHA-pinned local path; test `src/tests/audio/tasks/features_extraction_test.py`.
- [ ] T043 [P] [US4] Migrate the SpeechBrain SER loader in `src/senselab/audio/tasks/classification/speech_emotion_recognition/api.py` (`_load_speechbrain_ser_model`) to SHA-pinned local path; test `src/tests/audio/tasks/classification_test.py`.

### Migrate pyannote backends (local checkpoint, NO revision; pre-stage sub-models)

- [ ] T044 [P] [US4] Migrate `src/senselab/audio/tasks/speaker_diarization/pyannote.py` to pre-stage the pipeline AND its sub-models (segmentation, embedding) via `resolve_model`, then `Pipeline.from_pretrained(local_dir)` WITHOUT `revision`; test `src/tests/audio/tasks/speaker_diarization_test.py` (research gotchas 2–3).
- [ ] T045 [P] [US4] Migrate `src/senselab/audio/tasks/voice_activity_detection/pyannote_vad.py` to the same local-checkpoint pattern; test `src/tests/audio/tasks/voice_activity_detection_test.py`.

### Migrate remaining in-process transformers backends

- [ ] T046 [P] [US4] Migrate `src/senselab/audio/tasks/classification/huggingface.py` (audio-classification pipeline) to `load_hf_resilient`; test `src/tests/audio/tasks/classification_test.py`.
- [ ] T047 [P] [US4] Migrate `src/senselab/audio/tasks/text_to_speech/huggingface.py` (TTS pipeline) to `load_hf_resilient`; test `src/tests/audio/tasks/text_to_speech_test.py`.
- [ ] T048 [P] [US4] Migrate `src/senselab/audio/tasks/speaker_embeddings/wavlm.py` (was retry-only) to `load_hf_resilient`; test `src/tests/audio/tasks/speaker_embeddings_test.py`.
- [ ] T049 [P] [US4] Migrate the transformers SSL path in `src/senselab/audio/tasks/ssl_embeddings/self_supervised_features.py` (`AutoModel`/`AutoFeatureExtractor`) to `load_hf_resilient`; test `src/tests/audio/tasks/features_extraction_test.py`.
- [ ] T050 [P] [US4] Migrate `src/senselab/text/tasks/embeddings_extraction/huggingface.py` (`AutoTokenizer`/`AutoModel`) to `load_hf_resilient`; test `src/tests/text/tasks/embeddings_extraction_test.py`.
- [ ] T051 [P] [US4] Migrate the wav2vec2 SER head + config loaders in `src/senselab/audio/tasks/classification/speech_emotion_recognition/api.py` (`EmotionModel.from_pretrained`, `AutoConfig`, and the `hf_hub_download(config.json / safetensors index / hyperparams.yaml)` calls) to resolve/stage via the shared mechanism; test `src/tests/audio/tasks/classification_test.py` + `src/tests/audio/data_structures/audio_classification_result_test.py`.
- [ ] T052 [P] [US4] Migrate the SER diagnostic probe `src/senselab/audio/tasks/classification/speech_emotion_recognition/__main__.py` (`AutoConfig`) to the shared config-resolve helper.

### Migrate sentence-transformers backend

- [ ] T053 [P] [US4] Migrate `src/senselab/text/tasks/embeddings_extraction/sentence_transformers.py` to resolve first, then `SentenceTransformer(str(snapshot_path))` (or `revision=sha, local_files_only=True`); test `src/tests/text/tasks/embeddings_extraction_test.py`.

### Migrate remaining subprocess-venv workers

- [ ] T054 [P] [US4] Migrate `src/senselab/audio/tasks/speaker_diarization/nvidia.py` (Sortformer) to use the reworked `hf_subprocess_env`; test `src/tests/audio/tasks/speaker_diarization_test.py`.
- [ ] T055 [P] [US4] Migrate the continuous-SER subprocess worker in `src/senselab/audio/tasks/classification/speech_emotion_recognition/api.py` (`_CONT_SER_WORKER`) to `hf_subprocess_env`; test `src/tests/audio/tasks/classification_test.py`.
- [ ] T056 [P] [US4] Migrate `src/senselab/audio/workflows/audio_analysis/pii_subprocess.py` (GLiNER) to `hf_subprocess_env`; test `src/tests/audio/workflows/audio_analysis/pii_subprocess_test.py`.

### Verify scope boundary (non-HF loaders explicitly excluded)

- [ ] T057 [US4] Document in `hf_model_cache.py` module docstring and `quickstart.md` that yamnet (TF-Hub), s3prl, SPARC (`sparc.py` ×2), and Coqui (`coqui.py` ×2) are out of scope for HF version verification; confirm they are untouched.

**Checkpoint**: Every HF-backed backend loads through the shared mechanism (SC-010); a new backend needs only its load step (SC-006).

---

## Phase 7: Polish & Cross-Cutting Concerns

- [ ] T058 [P] Add an SC-006/SC-010 inspection test that greps `src/senselab` for residual bespoke caching/offline/version code and asserts none remains outside `hf_model_cache.py`/`dependencies.py` — this is the measurable proxy for "a new backend needs only its model-specific load step" (`src/tests/utils/hf_model_cache_test.py`).
- [ ] T059 [P] Retire or thin the old `hf_offline_loading` env-toggling context manager (deprecate with a docstring pointer to SHA-pinned loading) in `src/senselab/utils/dependencies.py`; keep `_HeartbeatLock`/`ensure_hf_model`/`retry_on_transient_error`.
- [ ] T060 Update `src/tests/utils/dependencies_test.py` for the reworked surface (record schema, resolver, freezes) and remove tests asserting env-toggling behavior.
- [ ] T061 [P] Write the PR description surfacing all thresholds/defaults (7-day window, retry limit, freeze env) per Constitution III / FR-012, and summarizing the migration.
- [ ] T062 GPU/parallel validation on SLURM: run the ≥100-concurrent-job cached-load check (SC-001) and per-backend smoke tests on a GPU node; record results. (Do NOT run on the login node.) This is the out-of-CI validation of SC-001; T012 is the CI proxy (thread-simulated concurrency with a `model_info` call counter).
- [ ] T063 Full quality gate: `cd src && uv run pytest && uv run ruff check . && uv run mypy`.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: no dependencies.
- **Foundational (Phase 2)**: depends on Setup — BLOCKS all user stories.
- **US1 (Phase 3)**: after Foundational. Delivers the MVP (429 fix).
- **US2 (Phase 4)**: after Foundational; independent of US1 but shares `resolve_model` (build after US1 in practice since both extend the same function).
- **US3 (Phase 5)**: after Foundational; extends `resolve_model` freshness/freeze branches.
- **US4 (Phase 6)**: after US1–US3 (needs the finished `resolve_model`/`load_hf_resilient`/`hf_subprocess_env` surface); the 23 migration tasks are mutually parallel (different files).
- **Polish (Phase 7)**: after US4.

### Story Independence

- US1, US2, US3 are behaviors of the shared resolver; each is independently testable via its own test cluster (T011–T013, T018–T021, T025–T029).
- US4 depends on the resolver surface being complete but its per-backend tasks are independent of one another.

### Within Each User Story

- Tests written and failing before implementation (Constitution I).
- Foundational record/resolver before behavior branches; behaviors before backend migration.

---

## Parallel Execution Examples

**User Story 1 tests (write together, must fail):**
```text
Task: T011 test_fresh_within_window_no_network
Task: T012 test_download_once_concurrent
Task: T013 test_local_only_sha_pinned_load
```

**User Story 4 backend migrations (independent files, run in parallel):**
```text
Task: T040 speaker_embeddings/speechbrain.py
Task: T044 speaker_diarization/pyannote.py
Task: T046 classification/huggingface.py
Task: T050 text/embeddings_extraction/huggingface.py
Task: T053 text/embeddings_extraction/sentence_transformers.py
Task: T056 audio_analysis/pii_subprocess.py
```

---

## Implementation Strategy

### MVP First (User Story 1)

1. Phase 1 Setup → Phase 2 Foundational → Phase 3 US1.
2. **STOP and VALIDATE**: concurrent cached loads make zero Hub calls; first-time download-once. This alone fixes the 429 storm.

### Incremental Delivery

- US1 (MVP: 429 fix) → US2 (version correctness) → US3 (freshness + freezes) → US4 (migrate all backends, one family at a time, each behind a green behavior test) → Polish.
- Each US4 migration commits independently; a broken backend never blocks the others.

---

## Notes

- [P] = different files, no incomplete-task dependency.
- Unit tests must not touch the network or GPU — monkeypatch `HfApi.model_info` and use a temp HF cache; heavy behavior checks (T062) run on SLURM.
- Commit after each task or logical group; keep public APIs byte-identical for un-migrated callers throughout (strangler-fig).
- **Total: 67 tasks** — Setup 3, Foundational 9, US1 9, US2 7, US3 10, US4 23, Polish 6.
