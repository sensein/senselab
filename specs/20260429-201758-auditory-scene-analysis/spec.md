# Feature Specification: Auditory Scene Analysis with YAMNet and Windowed Classification

**Feature Branch**: `20260429-201758-auditory-scene-analysis`
**Created**: 2026-04-29
**Status**: Draft
**Input**: Add YAMNet and other related models for auditory scene analysis in iterated form over windows.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Classify Audio Scenes Over Time Windows (Priority: P1)

A researcher has a long audio recording (e.g., 30 minutes of environmental audio, a clinical interview, or a classroom recording) and wants to understand what sound events occur and when. They run an auditory scene classifier (YAMNet or similar) over sliding windows, producing a timeline of detected sound events — speech, music, laughter, traffic, silence, dog bark, etc. — with timestamps showing when each event occurs and how confident the model is.

**Why this priority**: Windowed audio classification is a fundamental building block for audio understanding. Current senselab classification is whole-file only — no temporal resolution. Windowed analysis is needed for long recordings where the acoustic environment changes over time.

**Independent Test**: Run YAMNet on a 30-second audio sample with 1-second windows, verify output contains per-window classifications with timestamps and confidence scores.

**Acceptance Scenarios**:

1. **Given** a long audio recording, **When** the user runs windowed scene classification, **Then** they receive a list of per-window results, each containing the top predicted sound event(s) with confidence scores and the time interval (start/end in seconds).
2. **Given** a window size and hop size, **When** the user specifies them, **Then** windows are generated accordingly (e.g., 1s windows with 0.5s hop for overlapping analysis).
3. **Given** default parameters, **When** the user runs without specifying window/hop, **Then** reasonable defaults are used (e.g., 1s window, 0.5s hop).
4. **Given** the windowed results, **When** the user visualizes them, **Then** they see a timeline plot showing sound events over time aligned with the audio waveform.

---

### User Story 2 - Use Multiple Audio Scene Models (Priority: P2)

A user wants to compare different audio scene classifiers on the same recording. They can choose from multiple models (YAMNet, AudioSet-based models, PANNs, BEATs, etc.) through the same API, each providing different label sets and granularities. The API abstracts the model-specific details.

**Why this priority**: Different models have different strengths — YAMNet has 521 AudioSet classes, PANNs have fine-grained event detection, BEATs uses self-supervised pre-training. Users need to compare and choose the right model for their domain.

**Independent Test**: Run the same audio through 2 different scene classifiers and verify both return per-window classifications with their respective label sets.

**Acceptance Scenarios**:

1. **Given** an audio file, **When** the user runs the scene classifier with a YAMNet-family model, **Then** they receive predictions from the 521-class AudioSet ontology.
2. **Given** the same audio file, **When** the user runs with a different model (e.g., a PANN or BEATs variant), **Then** they receive predictions from that model's label set.
3. **Given** results from multiple models, **When** displayed together, **Then** the user can compare which events each model detects.

---

### User Story 3 - Integrate Scene Analysis into Quality Control and Pipelines (Priority: P3)

A researcher uses windowed scene classification as part of a larger pipeline — for example, detecting non-speech segments (music, noise) for quality control, or identifying when a speaker transitions from quiet speech to laughter. The per-window results can be used to filter, segment, or annotate audio programmatically.

**Why this priority**: Scene classification becomes most valuable when integrated into workflows — filtering out non-speech regions before ASR, detecting environmental noise for quality assessment, or segmenting recordings by acoustic context.

**Independent Test**: Use windowed classification to identify and extract only speech segments from a mixed audio recording.

**Acceptance Scenarios**:

1. **Given** windowed classification results, **When** the user filters for "Speech" labels above a confidence threshold, **Then** they get time segments containing speech.
2. **Given** windowed results, **When** used as input to another senselab task (e.g., extract only speech segments for ASR), **Then** the integration works seamlessly with senselab's Audio chunking utilities.

---

### Edge Cases

- What happens when the audio is shorter than the window size? (Use the full audio as a single window.)
- What happens when the model returns many low-confidence predictions? (Return top-k results per window, configurable.)
- What happens with very long audio (hours)? (Process in batches; memory-efficient iteration over windows.)
- What happens with multi-channel audio? (Require mono; raise error with guidance to downmix first.)

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST support windowed audio classification where the user specifies window size (in seconds) and hop size (in seconds), with reasonable defaults (1s window, 0.5s hop).
- **FR-002**: The system MUST support YAMNet (521-class AudioSet ontology) as the primary auditory scene classifier, accessible through senselab's existing classification API or a new dedicated function.
- **FR-003**: The system MUST support at least one additional audio scene model beyond YAMNet (e.g., a PANN, BEATs, or CLAP variant from HuggingFace) to demonstrate model flexibility.
- **FR-004**: Each window's result MUST include: start time (seconds), end time (seconds), top-k predicted labels, and confidence scores.
- **FR-005**: The system MUST provide a visualization function that plots detected sound events over time, aligned with the audio waveform (similar to plot_aligned_panels with a "segments" panel).
- **FR-006**: The windowed classification MUST handle audio of any length efficiently, processing windows in batches rather than loading all windows into memory at once.
- **FR-007**: The system MUST follow established senselab tutorial conventions with a tutorial demonstrating windowed scene analysis on sample audio.

### Key Entities

- **AudioWindow**: A segment of audio with defined start and end time, extracted from a longer recording.
- **SceneClassificationResult**: Per-window classification output containing time interval, predicted labels, and confidence scores.
- **AudioSet Ontology**: The 521-class label hierarchy used by YAMNet and many audio scene classifiers.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Windowed classification on a 30-second sample produces at least 30 per-window results (with 1s window, 0.5s hop = ~59 windows).
- **SC-002**: At least 2 different audio scene models are supported and produce results through the same API.
- **SC-003**: The visualization function produces a timeline plot showing detected events aligned with the waveform.
- **SC-004**: Processing a 5-minute audio file completes within 60 seconds on CPU.
- **SC-005**: A tutorial notebook demonstrates windowed scene analysis and passes CI via papermill.

## Assumptions

- YAMNet is available as a TensorFlow/TFLite model or via a HuggingFace-compatible wrapper. If TensorFlow-only, it may need a subprocess venv to avoid TF/PyTorch conflicts.
- HuggingFace has audio-classification models trained on AudioSet that can serve as alternatives to YAMNet (e.g., MIT/ast-finetuned-audioset-10-10-0.4593).
- The windowed iteration uses senselab's existing Audio chunking utilities (chunk_audios) or a simple sliding window over the waveform tensor.
- The output format is compatible with senselab's existing classification API (AudioClassificationResult) extended with time intervals.
- Window size and hop size are specified in seconds and converted to samples internally based on sampling rate.
