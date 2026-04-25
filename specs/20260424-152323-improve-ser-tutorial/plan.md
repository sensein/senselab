# Implementation Plan: Improve SER Tutorial

**Branch**: `20260424-152323-improve-ser-tutorial` | **Date**: 2026-04-24 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/20260424-152323-improve-ser-tutorial/spec.md`

## Summary

Update the speech emotion recognition tutorial to use better-performing SER models that produce meaningful (not near-uniform) emotion scores, compare multiple models side-by-side, and add text-based sentiment analysis from transcription as a complementary signal.

## Technical Context

**Language/Version**: Python 3.11-3.12 (Colab uses 3.12)
**Primary Dependencies**: senselab, transformers (for text sentiment pipeline)
**Storage**: N/A
**Testing**: papermill for notebook execution
**Target Platform**: Google Colab
**Project Type**: Tutorial notebook update
**Performance Goals**: Tutorial completes in <10 minutes on CPU
**Constraints**: Must work on CPU; models must be freely available on HuggingFace
**Scale/Scope**: 1 notebook update

## Constitution Check

| Principle | Status | Notes |
|-----------|--------|-------|
| I. UV-Managed Python | PASS | Tutorial uses uv install in Colab |
| II. Encapsulated Testing | PASS | CI via papermill |
| III. Commit Early and Often | PASS | Single notebook change |
| IV. CI Must Stay Green | PASS | Tested via papermill locally + CI |
| V. Memory-Driven Anti-Pattern Avoidance | PASS | Addresses user feedback directly |
| VI. No Unnecessary API Calls | PASS | Models cached after first download |
| VII. Simplicity First | PASS | Uses existing senselab API + direct transformers for text sentiment |
| VIII. No Hardcoded Parameters | PASS | Device auto-detected |

## Project Structure

### Source Code

```text
tutorials/
└── audio/
    └── speech_emotion_recognition.ipynb   # UPDATED
```

**Structure Decision**: Single notebook update. No new senselab modules — text sentiment uses transformers pipeline directly in the notebook since adding a full senselab text classification API is unnecessary for one tutorial section.

## Implementation Phases

### Phase 1: Rebuild SER Tutorial

**Notebook structure**:

1. **Install + restart** (standard pattern)
2. **Imports + device**
3. **Record or load audio** (recording widget + sample fallback)
4. **Section: Speech Emotion Recognition with Multiple Models**
   - Load 3 models (IEMOCAP-trained, RAVDESS-trained, continuous)
   - Run all three on the same audio
   - Side-by-side comparison table + bar chart visualization
   - Explain which model suits which use case
5. **Section: Understanding Emotion Scores**
   - Explain near-uniform distributions
   - Show acted vs natural speech examples (RAVDESS sample vs user audio)
   - Guidance on interpreting scores (relative ordering > absolute values)
6. **Section: Text Sentiment from Transcription**
   - Transcribe audio with whisper-large-v3-turbo
   - Run text sentiment (cardiffnlp/twitter-roberta-base-sentiment-latest)
   - Compare acoustic emotion vs text sentiment
   - Explain when they agree and disagree
7. **Section: Apply to Your Own Data**
   - How to batch-process multiple files
   - How to choose the right model
8. **Summary**

### Phase 2: Test and Push

1. Test with papermill locally
2. Run pre-commit
3. Push to branch, create PR with `test-tutorials` label
4. Verify CI passes
5. Merge to alpha then main

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| superb/wav2vec2-base-superb-er may not work with senselab's API | Test locally first; the model uses standard HF pipeline format |
| Text sentiment model is large | Use a small RoBERTa-base model (~500MB), not a large model |
| CI timeout with multiple model downloads | Generous timeout (1200s CPU) |
