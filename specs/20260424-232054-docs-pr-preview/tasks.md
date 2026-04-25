# Tasks: Docs PR Preview + Coverage Audit

**Input**: Design documents from `/specs/20260424-232054-docs-pr-preview/`
**Prerequisites**: plan.md, spec.md, research.md, quickstart.md

**Organization**: Tasks grouped by user story.

## Format: `[ID] [P?] [Story] Description`

## Phase 1: Setup

- [ ] T001 Verify existing docs workflow still works: read .github/workflows/docs.yaml and confirm it deploys to `docs` branch on release

**Checkpoint**: Existing workflow understood

---

## Phase 2: User Story 1 — PR Docs Preview (Priority: P1) 🎯 MVP

**Goal**: Every PR gets an auto-built docs preview with a comment link, cleaned up on close/merge.

**Independent Test**: Open a PR, verify preview comment appears with working link, merge PR, verify preview deleted.

### Implementation

- [ ] T002 [US1] Create .github/workflows/docs-preview.yaml with build-and-deploy job — triggers on `pull_request: [opened, synchronize, reopened]`, builds docs with pdoc, deploys to `docs` branch under `pr-{number}/` subdirectory using JamesIves/github-pages-deploy-action@v4 with `target-folder` parameter
- [ ] T003 [US1] Add PR comment posting to docs-preview.yaml — use peter-evans/create-or-update-comment to post/update a comment with the preview URL (`https://sensein.github.io/senselab/pr-{number}/`), include a unique comment identifier so updates replace rather than duplicate
- [ ] T004 [US1] Add cleanup job to docs-preview.yaml — triggers on `pull_request: [closed]`, checks out `docs` branch, removes `pr-{number}/` directory, commits and pushes, updates PR comment to note preview was removed
- [ ] T005 [US1] Verify YAML is valid and pre-commit passes: `uv run pre-commit run --files .github/workflows/docs-preview.yaml`

**Checkpoint**: PR preview workflow ready for testing

---

## Phase 3: User Story 2 — Module Documentation Coverage (Priority: P2)

**Goal**: All modules have doc.md files with meaningful descriptions.

**Independent Test**: `uv run pdoc src/senselab -t docs_style/pdoc-theme --docformat google` generates complete docs with all modules present.

### Implementation

- [ ] T006 [P] [US2] Create src/senselab/audio/tasks/preprocessing/doc.md — describe resample_audios, downmix_audios_to_mono, normalize, chunk_audios; when to use each
- [ ] T007 [P] [US2] Create src/senselab/audio/tasks/input_output/doc.md — describe read_audios, save_audios; supported formats and file I/O patterns
- [ ] T008 [P] [US2] Create src/senselab/audio/tasks/plotting/doc.md — describe plot_waveform, plot_specgram, plot_waveform_and_specgram, plot_aligned_panels, play_audio; include panel types for plot_aligned_panels
- [ ] T009 [P] [US2] Create src/senselab/audio/tasks/quality_control/doc.md — describe QC framework, metrics, checks, taxonomy; reference issue #472 for roadmap
- [ ] T010 [P] [US2] Create src/senselab/audio/tasks/ssl_embeddings/doc.md — describe self-supervised learning embedding extraction; models supported
- [ ] T011 [P] [US2] Create src/senselab/audio/tasks/speaker_diarization_evaluation/doc.md — describe DER and other diarization evaluation metrics
- [ ] T012 [P] [US2] Create src/senselab/text/tasks/embeddings_extraction/doc.md — describe text embedding extraction; HuggingFace and sentence-transformers backends
- [ ] T013 [US2] Build docs locally and verify all modules appear: `uv run pdoc src/senselab -t docs_style/pdoc-theme --docformat google -o /tmp/docs-test && ls /tmp/docs-test/senselab/audio/tasks/`

**Checkpoint**: All modules documented

---

## Phase 4: User Story 3 — README and Docs Consistency (Priority: P2)

**Goal**: README links are correct and feature descriptions match the codebase.

**Independent Test**: All links in README.md resolve; feature list matches current capabilities.

### Implementation

- [ ] T014 [US3] Fix README.md documentation URL — change `https://sensein.group/senselab/senselab.html` (line ~39) to `https://sensein.github.io/senselab`
- [ ] T015 [US3] Verify tutorials/README.md lists all current tutorials — check against `ls tutorials/audio/*.ipynb tutorials/video/*.ipynb tutorials/utils/*.ipynb`, add any missing entries (audio_recording_and_acoustic_analysis, transcription_and_phonemic_analysis, speech_representations_lab)
- [ ] T016 [US3] Verify README.md feature list includes recent additions — check that SPARC articulatory coding, PPG phoneme analysis, speech enhancement, plot_aligned_panels are mentioned or their categories are represented

**Checkpoint**: README and docs consistent

---

## Phase 5: Polish & Cross-Cutting

- [ ] T017 Run pre-commit on all changed files: `uv run pre-commit run --all-files`
- [ ] T018 Push to branch and create PR to alpha
- [ ] T019 Verify CI passes (pre-commit, cpu-tests)
- [ ] T020 Test docs preview by checking the PR's own preview deployment

---

## Dependencies & Execution Order

- **US1 (Phase 2)**: Independent — workflow file only
- **US2 (Phase 3)**: Independent — doc.md files only, all parallelizable
- **US3 (Phase 4)**: Independent — README files only
- **US1 + US2 + US3**: Can all run in parallel (different files)
- **Polish (Phase 5)**: After all stories complete

## Implementation Strategy

### MVP (US1 only)
1. Create docs-preview.yaml
2. Push and test on the PR itself

### Full delivery
All three stories can be done in parallel since they touch different files.
