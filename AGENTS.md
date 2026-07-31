# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

Senselab is a Python package for processing and analyzing behavioral data (primarily voice/speech, but also text and video) using reproducible pipelines. It uses uv for dependency management and requires Python 3.11-3.12.

## Build and Development Commands

```bash
# Install dependencies (full development setup)
uv sync --extra articulatory --extra text --extra video --extra senselab-ai --group dev --group docs

# Install pre-commit hooks (required before committing)
uv run pre-commit install

# Run all tests with coverage
uv run pytest

# Run a single test file
uv run pytest src/tests/audio/tasks/preprocessing_test.py

# Run a specific test function
uv run pytest src/tests/audio/tasks/preprocessing_test.py::test_function_name -v

# Run tests in parallel
uv run pytest -n auto

# Type checking
uv run mypy .

# Linting
uv run ruff check
uv run ruff check --fix  # auto-fix issues

# Spell checking
uv run codespell

# Generate documentation locally
uv run pdoc src/senselab -t docs_style/pdoc-theme --docformat google
```

## Architecture

### Module Structure

```
src/senselab/
├── audio/           # Audio processing (largest module)
│   ├── data_structures/  # Audio, AudioClassificationResult
│   ├── tasks/            # Processing operations by capability
│   └── workflows/        # Composite pipelines (e.g., health_measurements)
├── video/           # Video processing
│   ├── data_structures/  # Video, Pose
│   └── tasks/            # pose_estimation, input_output
├── text/            # Text processing
│   └── tasks/            # embeddings_extraction
├── utils/           # Shared utilities
│   ├── data_structures/  # SenselabDataset, DeviceType, SenselabModel, Language
│   └── tasks/            # batching, dimensionality_reduction, pooling, etc.
└── agentic_interface/    # AI agent for Jupyter notebooks
```

### Key Design Patterns

**Data Containers**: `Audio` and `Video` classes use lazy loading (private `_waveform`/`_frames` with property accessors). All data structures are Pydantic v2 models.

**Model Abstraction**: `SenselabModel` is the base class with provider-specific subclasses (`HFModel`, `SpeechBrainModel`, `PyannoteAudioModel`, etc.) that validate model IDs on instantiation.

**Task Organization**: Each task module has an `api.py` with public functions and backend-specific implementations (e.g., `huggingface.py`, `speechbrain.py`). Functions accept data objects, model specs, device, and return typed results.

**Device Handling**: `DeviceType` enum (CPU, CUDA, MPS) with dtype mapping for cross-platform compatibility.

### Audio Tasks

Key audio processing capabilities in `audio/tasks/`:
- `speech_to_text/` - ASR transcription
- `speaker_verification/`, `speaker_embeddings/`, `speaker_diarization/`
- `speech_enhancement/`, `voice_activity_detection/`
- `voice_cloning/`, `text_to_speech/`
- `forced_alignment/`, `features_extraction/`
- `classification/` - includes speech emotion recognition
- `preprocessing/` - resample, filter, normalize
- `data_augmentation/`, `quality_control/`

## Code Style

- Google-style docstrings (enforced by ruff with `convention = "google"`)
- Line length: 120 characters
- Type hints required (mypy with pydantic plugin)
- Tests located in `src/tests/` mirroring the package structure
- Test files must be named `*_test.py`

## System Requirements

- macOS requires ARM64 (Apple Silicon); Intel Macs are not supported
- FFmpeg must be installed system-wide
- Docker required for some video models (MediaPipe-based estimators)
- CUDA 12.8 libraries for GPU support
- HuggingFace token (`HF_TOKEN` env var) for many models
