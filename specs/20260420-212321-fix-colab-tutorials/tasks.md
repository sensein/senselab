# Tasks: Fix All Tutorials to Run on Google Colab

**Input**: Design documents from `/specs/20260420-212321-fix-colab-tutorials/`
**Prerequisites**: plan.md, spec.md, research.md

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup

**Purpose**: Create standardized setup cell template and audit existing notebooks

- [x] T001 Audit all 17 tutorial notebooks — document current state (working/broken, has setup cell, has badge, CPU/GPU requirement) in `artifacts/tutorial-audit.md`
- [x] T002 Create standardized setup cell template based on research.md patterns — save as `tutorials/_setup_template.py` with pip install --pre, uv install, FFmpeg setup, HF_TOKEN handling, and GPU detection

---

## Phase 2: Foundational

**Purpose**: Fix shared infrastructure that all tutorials depend on

- [x] T003 Verify `pip install --pre senselab[nlp,text,video]` works on a clean Python 3.12 environment and all imports succeed
- [x] T004 Verify `scripts/install-ffmpeg.sh` works when called from a notebook cell (Colab-compatible invocation)
- [x] T005 Verify uv auto-install works via `pip install uv` in a notebook context

**Checkpoint**: Setup cell template verified working. Notebook fixes can begin.

---

## Phase 3: User Story 1 — Every Tutorial Runs on Colab (Priority: P1) MVP

**Goal**: All 17 notebooks execute without errors on a fresh Colab-like environment

**Independent Test**: Run each notebook via papermill on Python 3.12, verify zero cell errors

### CPU Tutorials (can be fixed in parallel)

- [x] T006 [P] [US1] Fix `tutorials/audio/00_getting_started.ipynb` — add setup cell, badge, clear outputs, verify execution
- [x] T007 [P] [US1] Fix `tutorials/audio/audio_data_augmentation.ipynb` — add setup cell, badge, clear outputs, verify execution
- [x] T008 [P] [US1] Fix `tutorials/audio/features_extraction.ipynb` — add setup cell, badge, clear outputs, verify execution
- [x] T009 [P] [US1] Fix `tutorials/audio/forced_alignment.ipynb` — add setup cell, badge, clear outputs, verify execution
- [x] T010 [P] [US1] Fix `tutorials/audio/speech_to_text.ipynb` — add setup cell, badge, clear outputs, verify execution
- [x] T011 [P] [US1] Fix `tutorials/video/pose_estimation.ipynb` — add setup cell, badge, clear outputs, verify execution
- [x] T012 [P] [US1] Fix `tutorials/utils/dimensionality_reduction.ipynb` — add setup cell, badge, clear outputs, verify execution
- [x] T013 [P] [US1] Fix `tutorials/senselab-ai/senselab_ai_intro.ipynb` — add setup cell, badge, clear outputs, verify execution

### GPU Tutorials (can be fixed in parallel, need GPU to fully verify)

- [x] T014 [P] [US1] Fix `tutorials/audio/conversational_data_exploration.ipynb` — add setup cell, badge, HF_TOKEN setup, GPU detection, clear outputs
- [x] T015 [P] [US1] Fix `tutorials/audio/extract_speaker_embeddings.ipynb` — add setup cell, badge, clear outputs, verify execution
- [x] T016 [P] [US1] Fix `tutorials/audio/speaker_diarization.ipynb` — add setup cell, badge, HF_TOKEN setup, GPU detection, clear outputs
- [x] T017 [P] [US1] Fix `tutorials/audio/speaker_verification.ipynb` — add setup cell, badge, clear outputs, verify execution
- [x] T018 [P] [US1] Fix `tutorials/audio/speech_emotion_recognition.ipynb` — add setup cell, badge, clear outputs, verify execution
- [x] T019 [P] [US1] Fix `tutorials/audio/speech_enhancement.ipynb` — add setup cell, badge, clear outputs, verify execution
- [x] T020 [P] [US1] Fix `tutorials/audio/text_to_speech.ipynb` — add setup cell, badge, clear outputs, verify execution
- [x] T021 [P] [US1] Fix `tutorials/audio/voice_activity_detection.ipynb` — add setup cell, badge, HF_TOKEN setup, GPU detection, clear outputs
- [x] T022 [P] [US1] Fix `tutorials/audio/voice_cloning.ipynb` — add setup cell, badge, clear outputs, verify subprocess venv works

**Checkpoint**: All 17 notebooks have setup cells, badges, and execute locally. Ready for CI.

---

## Phase 4: User Story 2 — Automated Tutorial Testing in CI (Priority: P2)

**Goal**: CI automatically executes all tutorials on every PR

**Independent Test**: Push a PR, verify tutorial-tests job runs and reports results

- [x] T023 [US2] Add `papermill` to dev dependencies in `pyproject.toml`
- [x] T024 [US2] Create tutorial manifest file `tutorials/manifest.json` mapping each notebook to requirements (cpu/gpu, timeout, extras)
- [x] T025 [US2] Add `tutorial-cpu-tests` job to `.github/workflows/tests.yaml` — runs CPU notebooks via papermill on ubuntu Python 3.12
- [x] T026 [US2] Add `tutorial-gpu-tests` job to `.github/workflows/tests.yaml` — runs GPU notebooks via papermill on EC2 (label-triggered)
- [x] T027 [US2] Verify CPU tutorial CI job passes on all CPU notebooks
- [x] T028 [US2] Verify GPU tutorial CI job passes on all GPU notebooks

**Checkpoint**: CI runs tutorials automatically. Regressions caught before merge.

---

## Phase 5: User Story 3 — Standardized Tutorial Structure (Priority: P3)

**Goal**: Consistent structure across all tutorials

**Independent Test**: Lint/validate each notebook for required elements

- [x] T029 [US3] Create a tutorial structure validation script `scripts/validate-tutorials.py` that checks each notebook for: Colab badge, setup cell, GPU requirement note, cleared outputs
- [x] T030 [US3] Add validation script to pre-commit hooks or CI
- [x] T031 [US3] Review and standardize section headers across all 17 notebooks (Introduction, Setup, Example, Results pattern)

**Checkpoint**: All notebooks pass structural validation.

---

## Phase 6: Polish & Cross-Cutting

- [x] T032 Merge all tutorial fixes to alpha via PR
- [x] T033 Run full CI (pre-commit + cpu-tests + tutorial-tests) and verify green
- [x] T034 Merge alpha to main via PR

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies
- **Phase 2 (Foundational)**: Depends on T002 (setup cell template)
- **Phase 3 (US1)**: Depends on Phase 2 verification. All 17 notebook tasks are parallel.
- **Phase 4 (US2)**: Depends on Phase 3 (needs working notebooks to test CI against)
- **Phase 5 (US3)**: Depends on Phase 3 (needs notebooks with structure to validate)
- **Phase 6 (Polish)**: Depends on all previous phases

### Parallel Opportunities

- T006-T022: All 17 notebook fixes are independent and can run in parallel
- T023-T024: CI setup tasks are parallel with notebook fixes
- T029: Validation script can be written in parallel with notebook fixes

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Audit + setup template
2. Complete Phase 2: Verify setup cell works
3. Complete Phase 3: Fix all 17 notebooks
4. **STOP and VALIDATE**: Run each notebook locally via papermill
5. Push to alpha, verify on Colab manually

### Incremental Delivery

1. Fix CPU tutorials first (T006-T013) — can verify locally
2. Fix GPU tutorials (T014-T022) — verify on EC2
3. Add CI (T023-T028) — automate verification
4. Standardize structure (T029-T031) — polish
