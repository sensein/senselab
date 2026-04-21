# Implementation Plan: Fix All Tutorials to Run on Google Colab

**Branch**: `20260420-212321-fix-colab-tutorials` | **Date**: 2026-04-20 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/20260420-212321-fix-colab-tutorials/spec.md`

## Summary

Review and fix all 17 tutorial notebooks to execute on Google Colab without errors. Standardize setup cells, add Colab badges, and add CI-based tutorial testing via papermill.

## Technical Context

**Language/Version**: Python 3.11-3.14 (Colab uses 3.12)
**Primary Dependencies**: papermill (notebook execution), senselab (the library being tutorialized)
**Storage**: N/A (notebooks are files in the repo)
**Testing**: papermill for notebook execution, pytest for CI integration
**Target Platform**: Google Colab (Ubuntu, Python 3.12, optional GPU)
**Project Type**: Documentation/tutorials (Jupyter notebooks)
**Performance Goals**: Setup cell completes in <3 minutes on Colab
**Constraints**: Colab free tier (12GB RAM, 100GB disk, optional T4 GPU)
**Scale/Scope**: 17 tutorial notebooks across 4 directories

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. UV-Managed Python | PASS | Setup cell bootstraps uv via `pip install uv`, then uses `uv pip install` for senselab. Fully compliant. |
| II. Encapsulated Testing | PASS | CI runs notebooks in isolated environments (GHA ubuntu, EC2) |
| III. Commit Early and Often | PASS | Each notebook fix is a discrete commit |
| IV. CI Must Stay Green | PASS | Tutorial CI job added alongside existing test jobs |
| V. Memory-Driven Anti-Pattern Avoidance | PASS | Colab notebook testing saved in memory |
| VI. No Unnecessary API Calls | PASS | Notebooks use cached models where possible |
| VII. Simplicity First | PASS | Standardized setup cell, no complex abstractions |
| VIII. No Hardcoded Parameters | PASS | Setup cell uses `pip install --pre senselab`, no hardcoded paths |

No violations. Gate passes.

## Project Structure

### Documentation (this feature)

```text
specs/20260420-212321-fix-colab-tutorials/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── tasks.md             # Phase 2 output
└── checklists/
    └── requirements.md  # Spec quality checklist
```

### Source Code (repository root)

```text
tutorials/
├── audio/
│   ├── 00_getting_started.ipynb
│   ├── audio_data_augmentation.ipynb
│   ├── conversational_data_exploration.ipynb
│   ├── extract_speaker_embeddings.ipynb
│   ├── features_extraction.ipynb
│   ├── forced_alignment.ipynb
│   ├── speaker_diarization.ipynb
│   ├── speaker_verification.ipynb
│   ├── speech_emotion_recognition.ipynb
│   ├── speech_enhancement.ipynb
│   ├── speech_to_text.ipynb
│   ├── text_to_speech.ipynb
│   ├── voice_activity_detection.ipynb
│   └── voice_cloning.ipynb
├── video/
│   └── pose_estimation.ipynb
├── utils/
│   └── dimensionality_reduction.ipynb
└── senselab-ai/
    └── senselab_ai_intro.ipynb

.github/workflows/tests.yaml  # Add tutorial-tests job
```

**Structure Decision**: No new directories needed. Tutorials are edited in-place. CI workflow extended with a tutorial-tests job.
