# Implementation Plan: Auditory Scene Analysis

**Branch**: `20260429-201758-auditory-scene-analysis` | **Date**: 2026-04-29 | **Spec**: [spec.md](spec.md)

## Summary

Add windowed audio scene classification using HuggingFace AST (Audio Spectrogram Transformer) and other audio-classification models. Sliding windows with configurable size/hop produce per-window event labels with timestamps. Integrates with existing classification and visualization infrastructure.

## Technical Context

**Language/Version**: Python 3.11-3.12
**Primary Dependencies**: transformers (HuggingFace audio-classification pipeline), existing senselab classification module
**Testing**: pytest + papermill for tutorial
**Target Platform**: CPU (GPU optional), Google Colab
**Project Type**: Library feature + tutorial
**Constraints**: Must work on CPU; window processing must be memory-efficient

## Constitution Check

| Principle | Status | Notes |
|-----------|--------|-------|
| I. UV-Managed Python | PASS | All via uv |
| II. Encapsulated Testing | PASS | pytest in uv venv |
| IV. CI Must Stay Green | PASS | New tests + existing pass |
| VII. Simplicity First | PASS | Reuses existing classification pipeline + plot_aligned_panels |
| VIII. No Hardcoded Parameters | PASS | Window/hop/top_k configurable with defaults |

## Project Structure

```text
src/senselab/audio/tasks/classification/
├── api.py                          # UPDATED (add classify_audios_in_windows)
├── huggingface.py                  # UPDATED (windowed iteration support)
└── doc.md                          # UPDATED (scene classification docs)

tutorials/audio/
└── auditory_scene_analysis.ipynb   # NEW

src/tests/audio/tasks/
└── classification_test.py          # UPDATED (windowed classification tests)
```

## Implementation Phases

### Phase 1: Windowed Classification API (P1)

1. Add `classify_audios_in_windows()` to `classification/api.py`:
   - Accepts: audios, model, window_size, hop_size, top_k, device
   - Slices each audio into windows (tensor slicing, not creating new files)
   - Creates lightweight Audio objects per window
   - Runs existing `classify_audios_with_transformers` in batches
   - Returns: List[List[Dict]] — per-audio, per-window results with start/end/labels/scores
2. Test with AST model on sample audio
3. Test with different window/hop configurations

### Phase 2: Multiple Models + Visualization (P2)

1. Test with at least 2 models (AST + another HuggingFace audio-classification model)
2. Create visualization helper or show how to use `plot_aligned_panels` with scene results
3. Update doc.md with supported models

### Phase 3: Tutorial (P2)

1. Create `auditory_scene_analysis.ipynb`:
   - Load audio (recording widget + sample)
   - Run windowed classification with AST
   - Visualize timeline of detected events
   - Compare with different models
   - Show integration with filtering/segmentation
2. Add to manifest.json
3. Test via papermill

### Phase 4: Polish

1. Pre-commit, tests, CI
2. Push, check comments, merge
