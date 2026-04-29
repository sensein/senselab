# Implementation Plan: Expand Speech Representation Model Coverage

**Branch**: `20260428-101838-expand-speech-models` | **Date**: 2026-04-28 | **Spec**: [spec.md](spec.md)

## Summary

Expand senselab's speech representation model coverage by integrating S3PRL (30+ SSL models), broadening SpeechBrain access, adding NeMo ASR, enabling dedicated Pyannote VAD, creating a model registry, and providing a tutorial reproducing the speaker identity coding paper's benchmarking pipeline.

## Technical Context

**Language/Version**: Python 3.11-3.12
**Primary Dependencies**: s3prl (subprocess venv), speechbrain, pyannote-audio, nemo_toolkit (subprocess venv), transformers
**Storage**: N/A
**Testing**: pytest for unit tests, papermill for tutorials
**Target Platform**: Linux (CI), macOS (dev), Google Colab (tutorials)
**Project Type**: Library extension + tutorials + documentation

## Constitution Check

| Principle | Status | Notes |
|-----------|--------|-------|
| I. UV-Managed Python | PASS | All execution via uv; subprocess venvs for S3PRL/NeMo |
| II. Encapsulated Testing | PASS | Tests in uv-managed venv |
| III. Commit Early and Often | PASS | Incremental per-phase commits |
| IV. CI Must Stay Green | PASS | New tests + existing tests must pass |
| V. Memory-Driven Anti-Pattern Avoidance | PASS | Follows subprocess venv pattern from prior work |
| VI. No Unnecessary API Calls | PASS | Models cached after first download |
| VII. Simplicity First | PASS | Uses existing patterns (subprocess venv, model classes) |
| VIII. No Hardcoded Parameters | PASS | Model names, venv paths configurable |

## Project Structure

```text
src/senselab/audio/tasks/
├── ssl_embeddings/
│   ├── self_supervised_features.py    # UPDATED (add S3PRL + SpeechBrain backends)
│   ├── s3prl.py                       # NEW (S3PRL subprocess venv worker)
│   └── doc.md                         # UPDATED (comprehensive model list)
├── speech_to_text/
│   ├── nemo.py                        # NEW (NeMo ASR subprocess venv)
│   └── api.py                         # UPDATED (NeMo backend option)
├── voice_activity_detection/
│   ├── pyannote_vad.py                # NEW (dedicated Pyannote VAD)
│   └── api.py                         # UPDATED (VAD model option)

tutorials/audio/
├── ssl_embeddings_comparison.ipynb    # NEW (multi-model embedding comparison)
├── speaker_identity_benchmark.ipynb   # NEW (reproduce paper's 17-model benchmark)

docs/
└── model_registry.md                  # NEW (or generated page)
```

## Implementation Phases

### Phase 1: S3PRL Subprocess Venv Integration (US1 — P1)

**Goal**: Add S3PRL models as a backend for SSL embedding extraction via subprocess venv.

1. Create `s3prl.py` with subprocess worker script that:
   - Loads model via `s3prl.hub`
   - Accepts audio as FLAC files
   - Extracts hidden states / embeddings
   - Returns as numpy arrays via JSON
2. Create `S3PRLModel` or use string-based model selection
3. Update `ssl_embeddings` API to route S3PRL models to subprocess backend
4. Write tests for at least 3 S3PRL models (APC, TERA, CPC)
5. Update `ssl_embeddings/doc.md` with S3PRL model table

### Phase 2: SpeechBrain Embedding Unification (US1 — P1)

**Goal**: Make SpeechBrain speaker encoders accessible through ssl_embeddings in addition to speaker_embeddings.

1. Update `ssl_embeddings` API to accept SpeechBrainModel and route to existing speaker encoder infrastructure
2. This is primarily an API routing change — the actual SpeechBrain encode_batch() already works
3. Document in ssl_embeddings/doc.md

### Phase 3: NeMo ASR (US3 — P2)

**Goal**: Add NeMo Conformer ASR as a subprocess venv option for speech-to-text.

1. Create `nemo.py` in speech_to_text with subprocess worker script
2. Reuse existing NeMo subprocess venv (add nemo ASR dependencies if missing)
3. Update `transcribe_audios` API to accept NeMo models
4. Test with `nvidia/stt_en_conformer_ctc_large`

### Phase 4: Pyannote Dedicated VAD (US3 — P2)

**Goal**: Expose Pyannote's VAD pipeline as a distinct option.

1. Create `pyannote_vad.py` with dedicated VAD pipeline (not diarization-based)
2. Update VAD API to support model selection
3. Test with `pyannote/voice-activity-detection`

### Phase 5: Model Registry + Documentation (US4 — P2)

**Goal**: Create a comprehensive model registry and update all docs.

1. Create YAML data source with all supported models
2. Generate model_registry.md from YAML
3. Update ssl_embeddings/doc.md with complete model table
4. Update README.md features section if needed

### Phase 6: Expand Speaker Embeddings Tutorial (US2 — P2)

**Goal**: Expand existing `extract_speaker_embeddings.ipynb` with multi-backend comparison exploring how different representations capture speaker identity.

1. Add S3PRL, HuggingFace SSL sections to existing tutorial
2. Add comparison visualization (t-SNE, cosine similarity)
3. Pedagogical explanations of supervised vs self-supervised vs generative approaches

### Phase 7: Polish

1. Run all tests locally
2. Run pre-commit
3. Push and verify CI
4. Merge

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| S3PRL venv creation slow | Cache venv; only create on first use |
| S3PRL model download large | Tests use smallest models (APC < 5MB) |
| NeMo ASR model large (~500MB) | CPU timeout generous; test with small model if available |
| Some paper models unavailable | Document which are accessible (15/17 target) |
