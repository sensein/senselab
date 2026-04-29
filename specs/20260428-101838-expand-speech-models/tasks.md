# Tasks: Expand Speech Representation Model Coverage

**Input**: Design documents from `/specs/20260428-101838-expand-speech-models/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md

**Organization**: Tasks grouped by user story. P1 (SSL embeddings) is the MVP; P2 stories can be done in parallel after P1.

## Format: `[ID] [P?] [Story] Description`

## Phase 1: Setup

- [x] T001 Review current ssl_embeddings module: read src/senselab/audio/tasks/ssl_embeddings/self_supervised_features.py and identify extension points for new backends
- [x] T002 Review speaker_identity_coding_paper encoder patterns: read /tmp/speaker_identity_coding_paper/encoder.py to understand S3PRL, HF, and SpeechBrain encoder APIs
- [x] T003 Test S3PRL installation in isolated venv: `uv venv /tmp/test-s3prl --python 3.11 && /tmp/test-s3prl/bin/pip install s3prl torch torchaudio && /tmp/test-s3prl/bin/python -c "import s3prl.hub as hub; m = getattr(hub, 'apc')(); print(type(m))"`

**Checkpoint**: Integration approach validated

---

## Phase 2: User Story 1 — Multi-Backend SSL Embeddings (Priority: P1) 🎯 MVP

**Goal**: Users can extract SSL embeddings from S3PRL models, HuggingFace models, and SpeechBrain speaker encoders through senselab's API.

**Independent Test**: Extract embeddings from APC (S3PRL), wav2vec2 (HF), and ECAPA-TDNN (SpeechBrain) on the same audio, verify each returns a valid tensor.

### S3PRL Backend

- [ ] T004 [US1] Create src/senselab/audio/tasks/ssl_embeddings/s3prl.py — S3PRL subprocess venv worker script and `S3PRLEmbeddingExtractor` class. Worker loads model via `s3prl.hub`, processes FLAC audio, returns numpy embeddings via JSON. Venv requirements: s3prl, torch>=2.0, torchaudio, numpy, soundfile
- [ ] T005 [US1] Define S3PRL venv spec in s3prl.py — `_S3PRL_VENV = "s3prl"`, `_S3PRL_REQUIREMENTS`, `_S3PRL_PYTHON = "3.11"`, reuse `ensure_venv` and `parse_subprocess_result` patterns from sparc.py
- [ ] T006 [US1] Implement `S3PRLEmbeddingExtractor.extract_embeddings(audios, model_name, device)` — accepts list of Audio + model name string (e.g., "apc", "tera"), returns List[torch.Tensor] of embeddings
- [ ] T007 [US1] Test S3PRL embeddings with 3 models locally: run extract_embeddings with "apc", "tera", "cpc" on test audio, verify output shapes

### SpeechBrain Backend Unification

- [ ] T008 [P] [US1] Update src/senselab/audio/tasks/ssl_embeddings/self_supervised_features.py — add routing for SpeechBrainModel inputs that delegates to existing speaker_embeddings infrastructure (EncoderClassifier.encode_batch)
- [ ] T009 [P] [US1] Test SpeechBrain embeddings: extract embeddings via ssl_embeddings API using SpeechBrainModel("speechbrain/spkrec-ecapa-voxceleb"), verify 192-dim output

### API Integration

- [ ] T010 [US1] Update ssl_embeddings API entry point (api.py or __init__.py) — route based on model type: HFModel → existing HF backend, SpeechBrainModel → SpeechBrain backend, str → S3PRL backend
- [ ] T011 [US1] Export new functions in src/senselab/audio/tasks/ssl_embeddings/__init__.py
- [ ] T012 [US1] Write tests in src/tests/audio/tasks/ssl_embeddings_test.py — test S3PRL (mock subprocess), SpeechBrain routing, HF (existing)
- [ ] T013 [US1] Update src/senselab/audio/tasks/ssl_embeddings/doc.md — comprehensive table of tested models from all 3 backends with embedding dimensions
- [ ] T014 [US1] Run pre-commit and all tests: `uv run pre-commit run --all-files && uv run pytest src/tests/audio/tasks/ssl_embeddings_test.py -v`

**Checkpoint**: Multi-backend SSL embeddings working with 3 backends

---

## Phase 3: User Story 2 — Speaker Identity Paper Reproducibility (Priority: P2)

**Goal**: Tutorial notebook demonstrating extraction of embeddings from the paper's 17 models and basic speaker verification benchmarking.

**Independent Test**: Notebook runs via papermill, extracts embeddings from at least 10 models, produces comparison visualization.

### Implementation

- [ ] T015 [US2] Expand tutorials/audio/extract_speaker_embeddings.ipynb — add sections for multi-backend embedding extraction: S3PRL (APC), HuggingFace SSL (wav2vec2/HuBERT), alongside existing SpeechBrain (ECAPA-TDNN). Show how different model families capture speaker identity differently.
- [ ] T016 [US2] Add comparison visualization to tutorials/audio/extract_speaker_embeddings.ipynb — t-SNE or cosine similarity matrix comparing embeddings from 3+ backends on same audio samples, with pedagogical explanations of supervised vs self-supervised vs generative approaches
- [ ] T017 [US2] Verify updated tutorial manifest.json timeout is sufficient (may need increase for S3PRL subprocess venv creation)
- [ ] T018 [US2] Test expanded notebook via papermill locally
- [ ] T019 [US2] Update tutorials/README.md to reflect expanded tutorial scope

**Checkpoint**: Existing tutorial expanded with multi-backend comparison

---

## Phase 4: User Story 3 — NeMo ASR + Pyannote VAD (Priority: P2)

**Goal**: NeMo ASR accessible through speech-to-text API, Pyannote dedicated VAD accessible through VAD API.

**Independent Test**: Transcribe audio with NeMo model, detect voice activity with dedicated Pyannote VAD model.

### NeMo ASR

- [ ] T020 [P] [US3] Create src/senselab/audio/tasks/speech_to_text/nemo.py — NeMo ASR subprocess venv worker. Reuse existing NeMo venv (add `nemo_toolkit[asr]` if needed). Worker loads model via `nemo.collections.asr.models.EncDecCTCModel.from_pretrained()`, transcribes audio, returns text
- [ ] T021 [P] [US3] Update src/senselab/audio/tasks/speech_to_text/api.py — add NeMo model routing in `transcribe_audios()`, accept NeMo model identifiers
- [ ] T022 [US3] Test NeMo ASR: transcribe test audio with `nvidia/stt_en_conformer_ctc_large` (or smaller model), verify text output

### Pyannote Dedicated VAD

- [ ] T023 [P] [US3] Create src/senselab/audio/tasks/voice_activity_detection/pyannote_vad.py — load `pyannote/voice-activity-detection` pipeline, return speech segments (start, end) distinct from diarization-based approach
- [ ] T024 [P] [US3] Update src/senselab/audio/tasks/voice_activity_detection/api.py — add model selection: PyannoteAudioModel for dedicated VAD vs diarization-based
- [ ] T025 [US3] Test Pyannote VAD: detect voice activity on test audio, verify segments returned
- [ ] T026 [US3] Run pre-commit and tests for speech_to_text and voice_activity_detection

**Checkpoint**: NeMo ASR and Pyannote VAD working

---

## Phase 5: User Story 4 — Model Registry + Documentation (Priority: P2)

**Goal**: Comprehensive model registry page and updated module docs.

**Independent Test**: Generated docs include model registry; all module docs list supported models.

### Implementation

- [ ] T027 [P] [US4] Create docs/model_registry.yaml — structured data source listing all supported models across all tasks. Include: name, task, source, model_id, embedding_dim (where applicable), parameters, training_data, recommended_for
- [ ] T028 [P] [US4] Create scripts/generate_model_registry.py — reads model_registry.yaml, outputs a Markdown table for inclusion in docs
- [ ] T029 [US4] Generate docs/model_registry.md from YAML — verify it renders correctly
- [ ] T030 [P] [US4] Update src/senselab/audio/tasks/ssl_embeddings/doc.md — add comprehensive model table from all 3 backends (HF, S3PRL, SpeechBrain) with tested status and embedding dimensions
- [ ] T031 [P] [US4] Update src/senselab/audio/tasks/speech_to_text/doc.md — add NeMo ASR models to supported list
- [ ] T032 [P] [US4] Update src/senselab/audio/tasks/voice_activity_detection/doc.md — add Pyannote dedicated VAD to supported list
- [ ] T033 [US4] Build docs locally and verify model registry appears: `uv run pdoc src/senselab -t docs_style/pdoc-theme --docformat google`

**Checkpoint**: All docs complete with model registry

---

## Phase 6: Polish & Cross-Cutting

- [ ] T034 Run full pre-commit: `uv run pre-commit run --all-files`
- [ ] T035 Run full test suite: `uv run pytest src/tests/ -q --tb=line`
- [ ] T036 Push to branch and create PR to alpha with `test-tutorials` label
- [ ] T037 Check PR comments before merging
- [ ] T038 Verify CI passes and merge to alpha then main

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: Start immediately
- **US1 (Phase 2)**: Depends on Setup — S3PRL venv validated
- **US2 (Phase 3)**: Depends on US1 — needs multi-backend embeddings working
- **US3 (Phase 4)**: Independent of US1/US2 — can start after Setup
- **US4 (Phase 5)**: Depends on US1 + US3 (needs to document all new models)
- **Polish (Phase 6)**: After all stories

### User Story Dependencies

- **US1 (P1)**: Independent — core SSL embedding work
- **US2 (P2)**: Depends on US1 (needs S3PRL + SB backends for paper models)
- **US3 (P2)**: Independent — NeMo ASR + Pyannote VAD are separate modules
- **US4 (P2)**: Depends on US1 + US3 (documents all integrations)

### Parallel Opportunities

- **T008 + T009**: SpeechBrain routing and testing (different from S3PRL work)
- **T020 + T023**: NeMo ASR and Pyannote VAD (completely independent modules)
- **T027 + T028 + T030 + T031 + T032**: All docs tasks (different files)
- **US1 + US3**: Can run in parallel after Setup (different modules)

---

## Parallel Example: US1 + US3

```bash
# Agent A: SSL Embeddings (US1)
T004: Create S3PRL subprocess backend
T008: Add SpeechBrain routing to ssl_embeddings
T010: Unified API routing

# Agent B: NeMo + Pyannote (US3)
T020: NeMo ASR subprocess backend
T023: Pyannote dedicated VAD
```

---

## Implementation Strategy

### MVP First (US1 Only)

1. Complete Setup (Phase 1)
2. Implement S3PRL backend (T004-T007)
3. Add SpeechBrain routing (T008-T009)
4. Unify API (T010-T014)
5. **VALIDATE**: Extract from 3 backends on same audio

### Full Delivery

1. MVP (US1) + US3 in parallel
2. US2 (tutorials) after US1 complete
3. US4 (docs) after US1 + US3
4. Polish + merge

---

## Notes

- S3PRL subprocess venv is the most complex task — follow sparc.py pattern exactly
- NeMo ASR may reuse existing NeMo diarization venv
- Speaker identity paper tutorial is educational, not production — OK if some models fail gracefully
- Model registry YAML is the source of truth; Markdown is generated
- Always check PR comments before merging
