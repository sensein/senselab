# Tasks: Pedagogical Audio Tutorials and PPG Phoneme Durations

**Input**: Design documents from `/specs/20260423-213942-pedagogical-tutorials/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, quickstart.md

**Tests**: Test tasks included for Phase 2 (PPG phoneme duration implementation requires unit tests).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Setup

**Purpose**: No new project structure needed — senselab repo already exists with tutorials/ directory.

- [x] T001 Review PR #431 code thoroughly: read src/senselab/audio/tasks/features_extraction/ppg.py changes and src/tests/audio/tasks/features_extraction_test.py additions from branch `satra/ppg-phoneme-durations`
- [x] T002 Verify existing senselab APIs work for tutorial needs: run `uv run python -c "from senselab.audio.tasks.features_extraction import extract_pitch_from_audios, extract_praat_parselmouth_features_from_audios; from senselab.audio.tasks.classification import classify_emotions_from_speech; from senselab.audio.tasks.speaker_verification import verify_speaker; print('OK')"`

**Checkpoint**: Codebase understood, APIs verified

---

## Phase 2: Foundational — PPG Phoneme Duration Implementation (Blocking)

**Purpose**: Implement phoneme duration analysis fresh on current codebase. BLOCKS Tutorial 2 and SHBT205-Lab.

**⚠️ CRITICAL**: Tutorial 2 (US2) and SHBT205-Lab (US3) cannot be completed until this phase is done.

### Tests

- [x] T003 [US4] Write test for phoneme duration extraction in src/tests/audio/tasks/features_extraction_test.py — test that given sample audio and its PPG, duration analysis returns dict with phoneme labels and duration values
- [x] T004 [US4] Write test for phoneme timeline plotting in src/tests/audio/tasks/features_extraction_test.py — test that given sample audio and its PPG, timeline plot returns a matplotlib Figure

### Implementation

- [x] T005 [US4] Implement `extract_mean_phoneme_durations(audio, posteriorgram)` in src/senselab/audio/tasks/features_extraction/ppg.py — takes Audio + PPG tensor, returns dict with frame_count, phoneme_durations list (phoneme label, count, mean_duration_seconds, total_duration_seconds)
- [x] T006 [US4] Implement `plot_ppg_phoneme_timeline(audio, posteriorgram, title, show)` in src/senselab/audio/tasks/features_extraction/ppg.py — takes Audio + PPG tensor, returns matplotlib Figure with horizontal bars for each phoneme segment
- [x] T007 [US4] Export new functions in src/senselab/audio/tasks/features_extraction/__init__.py
- [x] T008 [US4] Run tests locally: `uv run pytest src/tests/audio/tasks/features_extraction_test.py -v -k "ppg or phoneme"` and `uv run pre-commit run --all-files`
- [ ] T009 [US4] Close PR #431 with comment "Superseded by fresh implementation in [new branch/PR]"

**Checkpoint**: Phoneme duration functions available in codebase, tests pass

---

## Phase 3: User Story 1 — Audio Recording & Acoustic Analysis (Priority: P1) 🎯 MVP

**Goal**: New Tutorial 1 notebook teaching students to record/load audio, visualize waveform + spectrogram, extract pitch + formants, detect emotion, and compare speakers.

**Independent Test**: Run via papermill with downloaded sample audio — notebook produces waveform, spectrogram, pitch contour, formant values, emotion classification, and speaker similarity score.

### Implementation

- [x] T010 [US1] Create notebook tutorials/audio/audio_recording_and_acoustic_analysis.ipynb — title cell, overview markdown explaining what students will learn (recording, waveform, spectrogram, pitch, formants, emotion, speaker matching)
- [x] T011 [US1] Add install cell and restart admonition in tutorials/audio/audio_recording_and_acoustic_analysis.ipynb — `!pip install -q uv` + `!uv pip install --pre --system "senselab"`, followed by restart runtime markdown cell
- [x] T012 [US1] Add imports and device setup cell in tutorials/audio/audio_recording_and_acoustic_analysis.ipynb — import Audio, plotting functions, feature extraction, emotion classification, speaker verification; auto-detect device
- [x] T013 [US1] Add "Record or Load Audio" section in tutorials/audio/audio_recording_and_acoustic_analysis.ipynb — markdown explaining audio (waveform, sampling rate); code cell with conditional recording widget (Colab JS) OR download sample audio from GitHub raw URL; load as Audio object
- [x] T014 [US1] Add "Waveform Visualization" section in tutorials/audio/audio_recording_and_acoustic_analysis.ipynb — markdown explaining waveforms (amplitude over time); code cells: `plot_waveform(audio)`, `play_audio(audio)`
- [x] T015 [US1] Add "Spectrogram" section in tutorials/audio/audio_recording_and_acoustic_analysis.ipynb — markdown explaining spectrograms (frequency content over time, mel scale); code cells: `plot_specgram(audio)`, `plot_specgram(audio, mel_scale=True)`
- [x] T016 [US1] Add "Pitch Extraction" section in tutorials/audio/audio_recording_and_acoustic_analysis.ipynb — markdown explaining F0 (vocal fold vibration); code cell: `extract_pitch_from_audios([audio])`, matplotlib plot of pitch contour over time
- [x] T017 [US1] Add "Formant Extraction" section in tutorials/audio/audio_recording_and_acoustic_analysis.ipynb — markdown explaining formants F1/F2 (vocal tract resonances, vowel identity); code cell: `extract_praat_parselmouth_features_from_audios([audio])`, display F1/F2 values
- [x] T018 [US1] Add "Speech Emotion Recognition" section in tutorials/audio/audio_recording_and_acoustic_analysis.ipynb — markdown explaining SER (acoustic cues → emotion); code cell: `classify_emotions_from_speech([audio], HFModel(...))`, display emotion labels + scores
- [x] T019 [US1] Add "Speaker Matching" section in tutorials/audio/audio_recording_and_acoustic_analysis.ipynb — markdown explaining speaker embeddings + cosine similarity; code cells: load/download second audio, `verify_speaker([(audio1, audio2)])`, display similarity score with interpretation
- [x] T020 [US1] Test Tutorial 1 locally via papermill: `uv run papermill tutorials/audio/audio_recording_and_acoustic_analysis.ipynb /dev/null --cwd . -k python3 --execution-timeout 600`
- [x] T021 [US1] Clear all outputs and run pre-commit on tutorials/audio/audio_recording_and_acoustic_analysis.ipynb

**Checkpoint**: Tutorial 1 fully functional, passes papermill locally

---

## Phase 4: User Story 2 — Transcription & Phonemic Analysis (Priority: P2)

**Goal**: New Tutorial 2 notebook teaching students ASR, forced alignment, and PPG-based phoneme timeline analysis.

**Independent Test**: Run via papermill with sample audio — notebook produces transcription, word/phone alignment boundaries, PPG phoneme timeline plot.

### Implementation

- [x] T022 [US2] Create notebook tutorials/audio/transcription_and_phonemic_analysis.ipynb — title cell, overview markdown explaining what students will learn (ASR, forced alignment, PPGs, phoneme durations)
- [x] T023 [US2] Add install cell and restart admonition in tutorials/audio/transcription_and_phonemic_analysis.ipynb — `!pip install -q uv` + `!uv pip install --pre --system "senselab[nlp]"`, followed by restart runtime markdown cell
- [x] T024 [US2] Add imports and device setup cell in tutorials/audio/transcription_and_phonemic_analysis.ipynb — import Audio, transcribe_audios, align_transcriptions, extract_ppgs_from_audios, extract_mean_phoneme_durations, plot_ppg_phoneme_timeline, HFModel, Language
- [x] T025 [US2] Add "Load Audio" section in tutorials/audio/transcription_and_phonemic_analysis.ipynb — download sample audio from GitHub raw URL, load as Audio object, display properties, play_audio
- [x] T026 [US2] Add "Automatic Speech Recognition" section in tutorials/audio/transcription_and_phonemic_analysis.ipynb — markdown explaining ASR and Whisper; code cell: `transcribe_audios([audio], model=HFModel("openai/whisper-small"))`, display transcription text
- [x] T027 [US2] Add "Forced Alignment" section in tutorials/audio/transcription_and_phonemic_analysis.ipynb — markdown explaining forced alignment (mapping words/phones to time); code cells: `align_transcriptions([(audio, script, Language.EN)])`, display word and phone boundaries, matplotlib visualization overlaid on waveform
- [x] T028 [US2] Add "Phonetic Posteriorgrams" section in tutorials/audio/transcription_and_phonemic_analysis.ipynb — markdown explaining PPGs (probability distributions over phonemes per frame); code cell: `extract_ppgs_from_audios([audio])`, inspect tensor shape and contents
- [x] T029 [US2] Add "Phoneme Duration Analysis" section in tutorials/audio/transcription_and_phonemic_analysis.ipynb — markdown explaining phoneme timing from PPGs; code cells: `extract_mean_phoneme_durations(audio, ppg)` display as table, `plot_ppg_phoneme_timeline(audio, ppg)` display timeline
- [x] T030 [US2] Test Tutorial 2 locally via papermill: `uv run papermill tutorials/audio/transcription_and_phonemic_analysis.ipynb /dev/null --cwd . -k python3 --execution-timeout 1800`
- [x] T031 [US2] Clear all outputs and run pre-commit on tutorials/audio/transcription_and_phonemic_analysis.ipynb

**Checkpoint**: Tutorial 2 fully functional, passes papermill locally

---

## Phase 5: User Story 3 — Update Existing Course Notebooks (Priority: P2)

**Goal**: Adapt 00_getting_started.ipynb (verify) and create shbt205_lab.ipynb from course materials, replacing all raw library calls with senselab APIs.

**Independent Test**: Both notebooks run via papermill in CI with no errors using senselab API calls only.

### Implementation

- [x] T032 [US3] Verify tutorials/audio/00_getting_started.ipynb uses current conventions — check install cell, restart admonition, device auto-detect, HF_TOKEN setup; fix if needed
- [ ] T033 [US3] Create tutorials/audio/shbt205_lab.ipynb from ~/Downloads/drive-download-20260424T013242Z-3-001/SHBT205-Lab.ipynb — copy and begin adaptation
- [ ] T034 [US3] Update install cell and add restart admonition in tutorials/audio/shbt205_lab.ipynb — `!pip install -q uv` + `!uv pip install --pre --system "senselab"`, restart admonition, auto-detect device
- [ ] T035 [US3] Replace raw Whisper ASR with senselab API in tutorials/audio/shbt205_lab.ipynb — replace `whisper.load_model()`, `whisper.decode()` with `transcribe_audios([audio], model=HFModel("openai/whisper-base"))`
- [ ] T036 [US3] Replace raw SPARC calls with senselab API in tutorials/audio/shbt205_lab.ipynb — replace `coder.encode()` with `SparcFeatureExtractor.extract_sparc_features([audio])`, adapt articulatory feature plotting
- [ ] T037 [US3] Replace raw Promonet PPG calls with senselab API in tutorials/audio/shbt205_lab.ipynb — replace `promonet.preprocess.from_audio()` and `promonet.plot.from_audio()` with `extract_ppgs_from_audios([audio])`, `plot_ppg_phoneme_timeline(audio, ppg)`
- [ ] T038 [US3] Add fallback sample audio download for CI in tutorials/audio/shbt205_lab.ipynb — conditional recording widget (Colab) with downloadable sample audio fallback
- [ ] T039 [US3] Add pedagogical markdown context throughout tutorials/audio/shbt205_lab.ipynb — explanations for each section (audio basics, ASR, articulatory coding, PPGs)
- [ ] T040 [US3] Clear all outputs and run pre-commit on tutorials/audio/shbt205_lab.ipynb
- [ ] T041 [P] [US3] Test shbt205_lab.ipynb locally via papermill: `uv run papermill tutorials/audio/shbt205_lab.ipynb /dev/null --cwd . -k python3 --execution-timeout 1800`
- [ ] T042 [P] [US3] Test 00_getting_started.ipynb locally via papermill: `uv run papermill tutorials/audio/00_getting_started.ipynb /dev/null --cwd . -k python3 --execution-timeout 600`

**Checkpoint**: Both course notebooks work with current senselab, pass papermill

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: CI integration, manifest updates, final validation

- [ ] T043 Add 3 new entries to tutorials/manifest.json — audio_recording_and_acoustic_analysis (timeout_cpu: 600, timeout_gpu: 300), transcription_and_phonemic_analysis (timeout_cpu: 1800, timeout_gpu: 600), shbt205_lab (timeout_cpu: 1800, timeout_gpu: 600, requires_hf_token: true)
- [ ] T044 Run all tutorials locally via papermill — verify all 4 new/updated notebooks plus existing tutorials still pass
- [ ] T045 Run full pre-commit: `uv run pre-commit run --all-files`
- [ ] T046 Run unit tests: `uv run pytest src/tests/ -q --tb=line`
- [ ] T047 Push to branch and create PR to alpha with `test-tutorials` label
- [ ] T048 Verify CI passes (pre-commit, cpu-tests, tutorial-cpu-tests)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational / US4 (Phase 2)**: Depends on Setup — BLOCKS US2 and US3 (they need PPG phoneme duration functions)
- **US1 (Phase 3)**: Depends on Setup only — can start in parallel with Phase 2
- **US2 (Phase 5)**: Depends on Phase 2 completion (needs phoneme duration functions)
- **US3 (Phase 6)**: Depends on Phase 2 completion (SHBT205-Lab needs PPG functions)
- **Polish (Phase 7)**: Depends on all story phases complete

### User Story Dependencies

- **User Story 1 (P1)**: Independent — no dependency on US2/US3/US4
- **User Story 4 (P3)**: Independent — foundational code, no story dependencies
- **User Story 2 (P2)**: Depends on US4 (needs phoneme duration functions in ppg.py)
- **User Story 3 (P2)**: Depends on US4 (SHBT205-Lab needs PPG functions for promonet replacement)

### Within Each User Story

- Markdown explanation cells before code cells
- Each section builds on the previous (load audio → visualize → extract features → analyze)
- Test via papermill after each notebook is complete

### Parallel Opportunities

- **Phase 2 (US4) + Phase 3 (US1)**: Can run in parallel — US1 doesn't need PPG functions
- **Phase 5 (US2) + Phase 6 (US3)**: Can run in parallel after Phase 2 — different notebooks
- **T041 + T042**: Can run in parallel (different notebooks)
- **T003 + T004**: Can run in parallel (different test functions)

---

## Parallel Example: Phase 2 + Phase 3

```bash
# These can run in parallel since US1 doesn't depend on PPG phoneme functions:

# Agent A: PPG phoneme duration (Phase 2)
Task: T003 Write test for phoneme duration extraction
Task: T005 Implement extract_mean_phoneme_durations in ppg.py
Task: T006 Implement plot_ppg_phoneme_timeline in ppg.py

# Agent B: Tutorial 1 (Phase 3)
Task: T010 Create Tutorial 1 notebook
Task: T013 Add record/load audio section
Task: T016 Add pitch extraction section
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (review APIs)
2. Complete Phase 3: Tutorial 1 (US1) — can start immediately
3. **STOP and VALIDATE**: Test Tutorial 1 independently via papermill
4. This delivers the foundational tutorial students need

### Incremental Delivery

1. Phase 1 (Setup) + Phase 2 (US4: PPG functions) + Phase 3 (US1: Tutorial 1) → MVP
2. Phase 5 (US2: Tutorial 2) → Adds transcription + phoneme analysis
3. Phase 6 (US3: Course notebooks) → Updates existing materials
4. Phase 7 (Polish) → CI integration, final validation

### Parallel Strategy

With two agents:
1. **Agent A**: Phase 2 (PPG phoneme duration) → Phase 5 (Tutorial 2)
2. **Agent B**: Phase 3 (Tutorial 1) → Phase 6 (Course notebooks)
3. Both: Phase 7 (Polish)

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Recording cells must be conditional — skip when running via papermill in CI
- Sample audio files downloaded from `https://github.com/sensein/senselab/raw/main/src/tests/data_for_testing/`
- Commit after each completed phase
- Each notebook cleared of outputs before commit
