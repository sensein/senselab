# Tasks: Auditory Scene Analysis with Windowed Classification

**Input**: Design documents from `/specs/20260429-201758-auditory-scene-analysis/`
**Prerequisites**: plan.md, spec.md, research.md, quickstart.md

**Organization**: Tasks grouped by user story.

## Format: `[ID] [P?] [Story] Description`

## Phase 1: Setup

- [ ] T001 Review current classification module: read src/senselab/audio/tasks/classification/api.py and src/senselab/audio/tasks/classification/huggingface.py to understand existing audio-classification pipeline
- [ ] T002 Verify AST model works with HF pipeline: `uv run python -c "from transformers import pipeline; p = pipeline('audio-classification', model='MIT/ast-finetuned-audioset-10-10-0.4593'); print('OK')"`

**Checkpoint**: Existing infrastructure understood, AST model verified

---

## Phase 2: User Story 1 — Windowed Scene Classification (Priority: P1) 🎯 MVP

**Goal**: Classify audio scenes over sliding windows with configurable size/hop, returning per-window labels with timestamps.

**Independent Test**: Run windowed classification on 30s sample with 1s window, 0.5s hop → ~59 per-window results with timestamps and labels.

### Implementation

- [ ] T003 [US1] Implement `classify_audios_in_windows()` in src/senselab/audio/tasks/classification/api.py — accepts audios, model, window_size (float seconds, default 1.0), hop_size (float seconds, default 0.5), top_k (int, default 5), device. Slices each audio waveform into overlapping windows via tensor indexing, creates lightweight Audio objects per window, runs existing HuggingFace classification in batches, returns List[List[Dict]] where each Dict has keys: start (float), end (float), labels (List[str]), scores (List[float])
- [ ] T004 [US1] Handle edge cases in classify_audios_in_windows: audio shorter than window (use full audio as single window), mono check, empty audio list returns empty list
- [ ] T005 [US1] Write test in src/tests/audio/tasks/classification_test.py — test windowed classification with a synthetic audio tensor: verify correct number of windows for given size/hop, verify each result has start/end/labels/scores, verify timestamps are correct
- [ ] T006 [US1] Test with AST model on real audio: `uv run python -c "..."` using tutorial_audio_files/audio_48khz_mono_16bits.wav, verify meaningful AudioSet class labels returned
- [ ] T007 [US1] Run pre-commit and tests: `uv run pre-commit run --all-files && uv run pytest src/tests/audio/tasks/classification_test.py -v -k "window"`

**Checkpoint**: Windowed classification working with AST model

---

## Phase 3: User Story 2 — Multiple Models + Visualization (Priority: P2)

**Goal**: Support multiple audio scene models and provide timeline visualization.

**Independent Test**: Run same audio through AST and at least one other model, visualize both timelines.

### Implementation

- [ ] T008 [US2] Test with second HF audio-classification model on same audio — e.g., `facebook/wav2vec2-base` with audio-classification head, or another AudioSet model. Verify different label set returned through same API
- [ ] T009 [US2] Create helper function or document pattern for converting windowed results to plot_aligned_panels segment format in src/senselab/audio/tasks/classification/api.py — convert List[Dict] results to segment dicts compatible with plot_aligned_panels `{"type": "segments"}` panel
- [ ] T010 [US2] Update src/senselab/audio/tasks/classification/doc.md — add section on windowed classification with example models (AST, YAMNet if available) and AudioSet class list reference

**Checkpoint**: Multiple models supported, visualization integrated

---

## Phase 4: User Story 3 — Tutorial (Priority: P2)

**Goal**: Tutorial demonstrating windowed scene analysis on sample audio.

**Independent Test**: Notebook passes papermill locally.

### Implementation

- [ ] T011 [US3] Create tutorials/audio/auditory_scene_analysis.ipynb — install cell, restart admonition, imports, recording widget + sample audio fallback, device setup
- [ ] T012 [US3] Add "Windowed Scene Classification" section to tutorial — run AST on sample audio with 1s window/0.5s hop, display per-window results as table, explain AudioSet labels
- [ ] T013 [US3] Add "Timeline Visualization" section to tutorial — use plot_aligned_panels with waveform + spectrogram + scene classification segments, show how sound events align with acoustic content
- [ ] T014 [US3] Add "Filtering by Event Type" section to tutorial — demonstrate extracting only windows classified as "Speech" or "Music", show how to use results for segmentation
- [ ] T015 [US3] Add to tutorials/manifest.json with timeout_cpu: 600, timeout_gpu: 300
- [ ] T016 [US3] Test notebook via papermill: `uv run papermill tutorials/audio/auditory_scene_analysis.ipynb /dev/null --cwd . -k python3 --execution-timeout 600`
- [ ] T017 [US3] Update tutorials/README.md with new tutorial

**Checkpoint**: Tutorial complete and passes CI

---

## Phase 5: Polish & Cross-Cutting

- [ ] T018 Run full pre-commit: `uv run pre-commit run --all-files`
- [ ] T019 Run full test suite: `uv run pytest src/tests/ -q --tb=line`
- [ ] T020 Push to branch and create PR to alpha with `test-tutorials` label
- [ ] T021 Check PR comments before merging
- [ ] T022 Verify CI passes and merge to alpha then main

---

## Dependencies & Execution Order

- **Setup (Phase 1)**: Start immediately
- **US1 (Phase 2)**: Depends on Setup
- **US2 (Phase 3)**: Depends on US1 (needs windowed API)
- **US3 (Phase 4)**: Depends on US1 + US2 (tutorial shows both)
- **Polish (Phase 5)**: After all stories

### Parallel Opportunities
- T008 + T010: Different files (model testing + docs)
- T011-T017 are sequential (same notebook)

---

## Implementation Strategy

### MVP (US1 Only)
1. Implement `classify_audios_in_windows()` (T003-T004)
2. Test with AST (T006)
3. **VALIDATE**: Correct windows, timestamps, labels

### Full Delivery
1. MVP → US2 (multi-model + viz) → US3 (tutorial) → Polish
