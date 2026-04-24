# Implementation Plan: Pedagogical Audio Tutorials and PPG Phoneme Durations

**Branch**: `20260423-213942-pedagogical-tutorials` | **Date**: 2026-04-23 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/20260423-213942-pedagogical-tutorials/spec.md`

## Summary

Create two new pedagogical audio tutorials for senselab, update two existing course notebooks to use senselab APIs, and implement fresh PPG phoneme duration analysis (superseding PR #431). Tutorial 1 covers audio recording/loading, waveform/spectrogram visualization, pitch extraction, formant extraction, emotion detection, and speaker matching. Tutorial 2 covers ASR, forced alignment, and PPG-based phoneme timeline analysis.

## Technical Context

**Language/Version**: Python 3.11-3.12 (Colab uses 3.12)
**Primary Dependencies**: senselab (the library being tutorialized), papermill (CI execution), ipywebrtc or JS widgets (recording)
**Storage**: N/A (notebooks are files in the repo)
**Testing**: papermill for notebook execution in CI, pytest for PR #431 unit tests
**Target Platform**: Google Colab (primary), local Jupyter (secondary)
**Project Type**: Library tutorials (Jupyter notebooks)
**Performance Goals**: Each tutorial completes in <30 minutes for students; CI execution within timeout limits
**Constraints**: Must work on CPU (GPU optional); recording cells must be skippable for CI
**Scale/Scope**: 4 notebooks (2 new, 2 updated), 1 PR merge, manifest updates

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. UV-Managed Python | PASS | Tutorials use `!pip install uv` + `!uv pip install` in Colab; CI uses `uv run papermill` |
| II. Encapsulated Testing | PASS | CI runs notebooks via papermill in uv-managed venv |
| III. Commit Early and Often | PASS | Plan calls for incremental commits per task |
| IV. CI Must Stay Green | PASS | All notebooks tested in CI via `test-tutorials` label |
| V. Memory-Driven Anti-Pattern Avoidance | PASS | Checked memory for relevant warnings |
| VI. No Unnecessary API Calls | PASS | Tutorials use local sample files; no redundant model downloads |
| VII. Simplicity First | PASS | Tutorials use existing senselab APIs, no new abstractions |
| VIII. No Hardcoded Parameters | PASS | Device auto-detected; sample files downloaded from URLs |

## Project Structure

### Documentation (this feature)

```text
specs/20260423-213942-pedagogical-tutorials/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code (repository root)

```text
tutorials/
├── audio/
│   ├── audio_recording_and_acoustic_analysis.ipynb   # NEW Tutorial 1
│   ├── transcription_and_phonemic_analysis.ipynb     # NEW Tutorial 2
│   ├── shbt205_lab.ipynb                             # NEW (adapted from course)
│   ├── 00_getting_started.ipynb                      # UPDATED
│   └── ... (existing tutorials unchanged)
├── manifest.json                                     # UPDATED (add 3 entries)

src/senselab/audio/tasks/features_extraction/
├── ppg.py                                            # UPDATED (fresh phoneme duration implementation)
└── __init__.py                                       # UPDATED (phoneme duration exports)

src/tests/audio/tasks/
└── features_extraction_test.py                       # UPDATED (PR #431 tests)
```

**Structure Decision**: No new modules or directories. Tutorials go in existing `tutorials/audio/`. Phoneme duration functions added to existing ppg.py.

## Implementation Phases

### Phase 1: Implement PPG phoneme duration analysis (fresh)

**Goal**: Implement phoneme duration extraction and timeline plotting fresh on the current codebase, reviewing PR #431 for ideas but not merging it.

1. Read PR #431 code thoroughly (`extract_mean_phoneme_durations`, `plot_ppg_phoneme_timeline`)
2. Implement equivalent functionality cleanly against current senselab APIs in `ppg.py`
3. Add exports to `__init__.py`
4. Write tests in `features_extraction_test.py`
5. Run tests locally: `uv run pytest src/tests/audio/tasks/features_extraction_test.py -v -k "ppg or phoneme"`
6. Run pre-commit: `uv run pre-commit run --all-files`
7. Push, verify CI passes, merge to alpha
8. Close PR #431 as superseded

### Phase 2: Create Tutorial 1 — Audio Recording & Acoustic Analysis

**Goal**: New notebook `tutorials/audio/audio_recording_and_acoustic_analysis.ipynb`

**Notebook structure** (sections with markdown + code cells):

1. **Title & Overview** (markdown): What students will learn — recording/loading audio, visualization, pitch, formants, emotion, speaker matching
2. **Install senselab** (code): `!pip install -q uv` + `!uv pip install --pre --system "senselab"`
3. **Restart runtime admonition** (markdown)
4. **Imports & device setup** (code): senselab imports, auto-detect device
5. **Section: Record or Load Audio** (markdown + code):
   - Markdown: Explain what audio is (waveform, sampling rate)
   - Code: JS-based recording widget (conditional on Colab) OR download sample audio
   - Code: Load as `Audio` object, display properties
6. **Section: Waveform Visualization** (markdown + code):
   - Markdown: What a waveform represents (amplitude over time)
   - Code: `plot_waveform(audio)`, `play_audio(audio)`
7. **Section: Spectrogram** (markdown + code):
   - Markdown: What a spectrogram shows (frequency content over time), mel scale
   - Code: `plot_specgram(audio)`, `plot_specgram(audio, mel_scale=True)`
8. **Section: Pitch Extraction** (markdown + code):
   - Markdown: What pitch (F0) is, how it relates to vocal fold vibration
   - Code: `extract_pitch_from_audios([audio])`, plot pitch contour with matplotlib
9. **Section: Formant Extraction** (markdown + code):
   - Markdown: What formants are (F1, F2), relationship to vowel identity
   - Code: `extract_praat_parselmouth_features_from_audios([audio])`, display F1/F2 values
10. **Section: Speech Emotion Recognition** (markdown + code):
    - Markdown: How machines detect emotion from acoustic cues
    - Code: `classify_emotions_from_speech([audio], model)`, display labels + scores
11. **Section: Speaker Matching** (markdown + code):
    - Markdown: How speaker verification works (embeddings + cosine similarity)
    - Code: Load/record second audio, `verify_speaker([(audio1, audio2)])`, interpret score

### Phase 3: Create Tutorial 2 — Transcription & Phonemic Analysis

**Goal**: New notebook `tutorials/audio/transcription_and_phonemic_analysis.ipynb`

**Notebook structure**:

1. **Title & Overview** (markdown): ASR, forced alignment, PPG phoneme timeline
2. **Install senselab** (code): `!pip install -q uv` + `!uv pip install --pre --system "senselab[nlp]"`
3. **Restart runtime admonition** (markdown)
4. **Imports & device setup** (code)
5. **Section: Load Audio** (code): Download sample audio or use recording from Tutorial 1
6. **Section: Automatic Speech Recognition** (markdown + code):
   - Markdown: What ASR is, how Whisper works at a high level
   - Code: `transcribe_audios([audio], model=HFModel("openai/whisper-small"))`, display transcription
7. **Section: Forced Alignment** (markdown + code):
   - Markdown: What forced alignment does (maps words/phones to time)
   - Code: `align_transcriptions([(audio, script, Language.EN)])`, display word and phone boundaries
   - Code: Visualize alignment on waveform (matplotlib overlay)
8. **Section: Phonetic Posteriorgrams (PPGs)** (markdown + code):
   - Markdown: What PPGs are (probability distributions over phonemes per frame)
   - Code: `extract_ppgs_from_audios([audio])`, inspect shape and contents
9. **Section: Phoneme Duration Analysis** (markdown + code):
   - Markdown: How to derive phoneme timing from PPGs
   - Code: `extract_mean_phoneme_durations(audio, ppg)`, display duration table
   - Code: `plot_ppg_phoneme_timeline(audio, ppg)`, interpret the timeline

### Phase 4: Update Existing Course Notebooks

**Goal**: Adapt `00_getting_started.ipynb` and `SHBT205-Lab.ipynb` for current senselab.

**4a: Update 00_getting_started.ipynb** (already in `tutorials/`):
- Verify install cell uses current convention (already done in prior PR)
- Verify all API calls use current senselab functions
- Minor: This notebook is already up to date from the prior tutorial fix PR

**4b: Create shbt205_lab.ipynb** (new file adapted from course materials):
- Start from `~/Downloads/drive-download-20260424T013242Z-3-001/SHBT205-Lab.ipynb`
- Replace raw Whisper ASR → `transcribe_audios()`
- Replace raw promonet PPG → `extract_ppgs_from_audios()` + PR #431 functions
- Replace raw SPARC calls → `SparcFeatureExtractor.extract_sparc_features()`
- Update install cell, add restart admonition, auto-detect device
- Keep recording widget (conditional on Colab)
- Add fallback sample audio download for CI
- Clear all outputs

### Phase 5: Manifest & CI Integration

1. Add 3 new entries to `tutorials/manifest.json`:
   - `audio_recording_and_acoustic_analysis.ipynb` (timeout_cpu: 600, timeout_gpu: 300)
   - `transcription_and_phonemic_analysis.ipynb` (timeout_cpu: 1800, timeout_gpu: 600)
   - `shbt205_lab.ipynb` (timeout_cpu: 1800, timeout_gpu: 600)
2. Test all notebooks locally via papermill
3. Run pre-commit
4. Push and verify CI with `test-tutorials` label

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| PR #431 code outdated | Fresh implementation (Phase 1) avoids merge conflicts entirely |
| Recording widget doesn't work in CI | All tutorials have downloaded sample audio fallback path |
| PPG/SPARC subprocess venvs slow on CPU CI | Generous timeouts (1800s CPU) |
| SHBT205-Lab SPARC voice conversion has no senselab wrapper | Document as known limitation; use SPARC feature extraction (which IS wrapped) and note voice conversion needs future API |
| Colab JS recording approach varies across browsers | Provide clear instructions + sample audio fallback |

## Complexity Tracking

No constitution violations. All work uses existing senselab APIs and established tutorial patterns.
