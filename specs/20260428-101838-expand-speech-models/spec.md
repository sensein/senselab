# Feature Specification: Expand Speech Representation Model Coverage

**Feature Branch**: `20260428-101838-expand-speech-models`
**Created**: 2026-04-28
**Status**: Draft
**Input**: Expand coverage of speech representation models from S3PRL, the speaker identity coding paper (sensein/speaker_identity_coding_paper), SpeechBrain, Pyannote, and NeMo toolkit. Improve accessibility to the user and coverage in docs.

## Clarifications

### Session 2026-04-28

- Q: Tutorial scope for speaker identity paper? → A: Not a paper replication — a pedagogical exploration of different approaches to analyzing speaker identity. Merge into existing `extract_speaker_embeddings.ipynb` instead of creating new notebooks.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Researcher Extracts SSL Embeddings from Multiple Model Families (Priority: P1)

A researcher wants to compare how different self-supervised speech models represent a set of audio recordings. They use senselab to extract embeddings from S3PRL models (APC, TERA, CPC, DeCoAR2, MockingJay), HuggingFace models (wav2vec2, HuBERT, WavLM, data2vec, W2V-BERT), and SpeechBrain models (ECAPA-TDNN, x-vector) — all through a unified API. They can then compare, visualize, and use these embeddings for downstream tasks.

**Why this priority**: SSL embeddings are the foundation for many speech analysis tasks. Currently senselab's `ssl_embeddings` module only supports HuggingFace models, missing the S3PRL ecosystem (30+ models) and SpeechBrain speaker encoders. The speaker_identity_coding_paper demonstrated that comparing these representations reveals important differences in what each model captures about speaker identity.

**Independent Test**: Extract embeddings from the same audio using at least 5 different models from 3 different backends (S3PRL, HF, SpeechBrain), verify each returns a tensor of the expected shape.

**Acceptance Scenarios**:

1. **Given** an audio file, **When** the user calls the unified embedding extraction API with an S3PRL model name (e.g., "apc", "tera", "cpc"), **Then** they receive a tensor of embeddings from that model.
2. **Given** an audio file, **When** the user calls the API with a HuggingFace model (e.g., "facebook/wav2vec2-large-lv60"), **Then** they receive hidden-state embeddings (existing behavior, preserved).
3. **Given** an audio file, **When** the user calls the API with a SpeechBrain model (e.g., "speechbrain/spkrec-ecapa-voxceleb"), **Then** they receive speaker embeddings from that model.
4. **Given** embeddings from multiple models, **When** the user compares them, **Then** the tutorial shows how to visualize and analyze representation differences (e.g., t-SNE, cosine similarity).

---

### User Story 2 - Explore Speaker Identity with Multiple Representations (Priority: P2)

A student or researcher opens the existing `extract_speaker_embeddings` tutorial (expanded) and explores how different speech representation approaches capture speaker identity. The tutorial demonstrates extracting embeddings from multiple backends (S3PRL, HuggingFace SSL, SpeechBrain), comparing them visually (t-SNE, cosine similarity), and understanding which representations are best suited for speaker-related tasks. This is not a paper replication — it's a pedagogical exploration of different approaches.

**Why this priority**: The existing speaker embeddings tutorial only shows SpeechBrain ECAPA-TDNN. Expanding it to include SSL models (wav2vec2, HuBERT, APC) and comparing representations helps users understand the landscape and choose the right model for their needs.

**Independent Test**: The expanded `extract_speaker_embeddings.ipynb` tutorial runs via papermill, extracts embeddings from at least 3 backends, and produces a comparison visualization.

**Acceptance Scenarios**:

1. **Given** the existing speaker embeddings tutorial, **When** updated with multi-backend extraction, **Then** users can extract embeddings from S3PRL, HuggingFace, and SpeechBrain models in the same notebook.
2. **Given** embeddings from multiple models, **When** visualized (e.g., t-SNE, cosine similarity matrix), **Then** users can see how different representations capture speaker identity differently.
3. **Given** the tutorial, **When** a student reads it, **Then** they understand the trade-offs between supervised (SpeechBrain), self-supervised (HuBERT/wav2vec2), and generative (APC/TERA) representations for speaker identity.

---

### User Story 3 - Expanded NeMo and Pyannote Model Access (Priority: P2)

A user wants to access NeMo toolkit capabilities beyond diarization (e.g., NeMo ASR models, speaker recognition, language identification) and Pyannote capabilities beyond diarization (e.g., voice activity detection with dedicated VAD models, overlapped speech detection). These are exposed through senselab's standard API patterns.

**Why this priority**: NeMo and Pyannote are major speech toolkits with capabilities beyond what senselab currently exposes. NeMo has state-of-the-art ASR (Conformer-CTC/RNNT), speaker recognition, and language ID. Pyannote has dedicated VAD and overlap detection models.

**Independent Test**: Run NeMo ASR on sample audio through senselab, run Pyannote VAD pipeline through senselab.

**Acceptance Scenarios**:

1. **Given** an audio file, **When** the user runs NeMo-based ASR through senselab, **Then** they get a transcription.
2. **Given** an audio file, **When** the user runs Pyannote's dedicated VAD model, **Then** they get voice activity segments distinct from the diarization-based approach.

---

### User Story 4 - Comprehensive Documentation and Model Registry (Priority: P2)

A user browsing the docs can find a model registry showing all supported models, organized by task, with information about each model's source, size, training data, and when to use it. Each task module's docs clearly list supported backends and example models.

**Why this priority**: Users currently have to read source code to discover which models are supported. A centralized registry improves discoverability and helps users choose the right model for their data.

**Independent Test**: The docs site contains a model registry page listing all supported models by task with at least name, source, size, and recommended use case.

**Acceptance Scenarios**:

1. **Given** the docs site, **When** a user navigates to the model registry, **Then** they find models organized by task with actionable information.
2. **Given** any task module's docs, **When** a user reads it, **Then** they find a list of tested models with usage examples.
3. **Given** the SSL embeddings module, **When** the docs are generated, **Then** the documentation includes a table of tested models from S3PRL, HuggingFace, and SpeechBrain with embedding dimensions.

---

### Edge Cases

- What happens when an S3PRL model is not installed? (Subprocess venv handles isolation; clear error message if model unavailable.)
- What happens when a NeMo model requires GPU but only CPU is available? (Model runs on CPU with degraded performance; docs note this.)
- What happens when model output shapes differ across backends? (Unified API normalizes to a standard format: list of tensors per audio.)

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The SSL embeddings module MUST support S3PRL models (at minimum: APC, TERA, CPC, MockingJay, DeCoAR2) via an isolated subprocess venv, in addition to existing HuggingFace models.
- **FR-002**: The SSL embeddings module MUST support SpeechBrain speaker encoder models (ECAPA-TDNN, ResNet, x-vector) for embedding extraction, not just speaker verification.
- **FR-003**: The existing `extract_speaker_embeddings.ipynb` tutorial MUST be expanded to demonstrate multi-backend embedding extraction (S3PRL, HuggingFace, SpeechBrain) with comparison visualization, exploring how different representations capture speaker identity.
- **FR-004**: NeMo ASR capabilities MUST be accessible through senselab's speech-to-text API via subprocess venv isolation.
- **FR-005**: Pyannote's dedicated VAD model MUST be accessible as a distinct option from the diarization-based VAD.
- **FR-006**: A model registry page MUST be added to the documentation listing all supported models by task, source, size, and recommended use case.
- **FR-007**: The SSL embeddings module MUST have a comprehensive doc.md and a dedicated tutorial notebook demonstrating multi-model embedding extraction and comparison.
- **FR-008**: All new model integrations MUST follow the existing patterns: subprocess venv for conflicting dependencies, lazy loading, caching, mono/16kHz conventions.

### Key Entities

- **Speech Representation**: A dense vector (embedding) extracted from an audio signal by a pre-trained model. Different models capture different aspects (speaker identity, phonetic content, emotion, etc.).
- **Model Registry**: A documentation artifact listing all supported models with metadata (source, task, size, training data, recommended use).
- **S3PRL Model**: A self-supervised speech model from the S3PRL toolkit (30+ models including APC, TERA, CPC, wav2vec2, HuBERT, etc.).
- **Encoder Backend**: The library used to load and run a model (HuggingFace, S3PRL, SpeechBrain, NeMo, Pyannote).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can extract embeddings from at least 3 different backends (S3PRL, HuggingFace, SpeechBrain) through senselab's unified API.
- **SC-002**: The SSL embeddings tutorial demonstrates extraction from at least 3 different backends (S3PRL, HuggingFace, SpeechBrain) with visualization.
- **SC-003**: The model registry page lists at least 30 models across all tasks with source, size, and use case information.
- **SC-004**: NeMo ASR produces transcriptions through senselab's API on sample audio.
- **SC-005**: All new model integrations pass CI (pre-commit + cpu-tests).

## Assumptions

- S3PRL models run in an isolated subprocess venv (s3prl has specific torch/torchaudio version requirements that may conflict with the main environment).
- NeMo models continue to use subprocess venv isolation (already established pattern for NeMo diarization).
- SpeechBrain speaker encoders are already available in the main environment (SpeechBrain is a direct dependency).
- The model registry is a documentation page generated from a structured data source (YAML or Python dict), not a runtime API.
- The speaker identity coding paper's benchmarking pipeline is demonstrated as a tutorial, not reproduced as a production feature.
- Some models (e.g., Hybrid BYOL-S, cochlear models) from the paper may not be reproducible through senselab due to unavailable checkpoints or custom dependencies.
