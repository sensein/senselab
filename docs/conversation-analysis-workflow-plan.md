# Conversation Analysis Workflow Plan

## Goal

Build a Zoom-oriented conversation analysis workflow that composes existing Senselab capabilities to produce:

- transcripts with heuristic accuracy estimates
- diarization and speaker-level turn summaries
- acoustic, linguistic, emotional, and engagement-related features
- turn-taking dynamics and interruption statistics
- sparse phonetic posteriorgram summaries with onset/offset timing
- environment/context summaries from captured audio conditions
- automated checks for each processing step

## Existing Senselab Capabilities We Should Reuse

- `senselab.audio.tasks.speech_to_text.transcribe_audios`
- `senselab.audio.tasks.speaker_diarization.diarize_audios`
- `senselab.audio.tasks.features_extraction.extract_features_from_audios`
- `senselab.audio.tasks.features_extraction.extract_ppgs_from_audios`
- `senselab.audio.tasks.speaker_embeddings.extract_speaker_embeddings_from_audios`
- `senselab.audio.tasks.ssl_embeddings.extract_ssl_embeddings_from_audios`
- `senselab.audio.tasks.classification.speech_emotion_recognition.classify_emotions_from_speech`
- `senselab.audio.tasks.quality_control` metrics/checks for environmental and preprocessing validation
- `senselab.video.tasks.input_output` patterns for extracting audio from video recordings
- existing `explore_conversation` workflow as the nearest workflow baseline

## Proposed Workflow

### Inputs

- one or more Zoom recordings as audio or video files
- optional explicit model choices for ASR, diarization, speech emotion recognition, speaker embeddings, and SSL embeddings
- optional toggles for PPG extraction, speaker embeddings, SSL embeddings, and environment context evaluation

### Processing Stages

1. Ingest
   - accept local audio/video paths
   - extract audio from video when needed
   - validate file existence and supported extensions

2. Preprocessing
   - read audio
   - downmix to mono
   - resample to 16 kHz
   - record preprocessing checks

3. Recording-level checks
   - silence/clipping/headroom/dynamic range/SNR checks
   - summarize environment context from QC metrics

4. Diarization and turn segmentation
   - diarize recording into speaker turns
   - extract turn-level audio segments
   - compute turn duration, pause/gap, overlap, interruption, speaker-switch statistics

5. Transcription
   - transcribe each turn
   - support one or more ASR models
   - estimate transcript accuracy heuristically:
     - multi-model transcript consensus when multiple ASR models are used
     - timestamp-coverage heuristic when only one model is used

6. Turn-level features
   - acoustic features from Parselmouth/OpenSMILE/torchaudio outputs
   - linguistic features from transcripts:
     - lexical richness
     - word choice markers
     - syntax-lite sentence/utterance statistics
     - discourse markers
   - dialogue act heuristics:
     - question
     - command/request
     - acknowledgement
     - greeting/closing
     - statement
   - engagement markers:
     - backchanneling
     - politeness cues
     - short acknowledgement turns
   - emotional states:
     - lexical sentiment heuristic
     - optional speech emotion recognition model output

7. Optional speaker/representation outputs
   - speaker embeddings
   - SSL embeddings
   - sparse PPG summaries with phoneme-like onset/offset segments

8. Recording-level summary
   - speaker inventory and speaker statistics
   - turn-taking summary
   - emotion summary
   - engagement summary
   - environment/context summary
   - checks by step

## Output Shape

Return one dictionary per input recording with:

- `source_file`
- `derived_audio_file` when video extraction is used
- `checks`
- `environment_context`
- `speaker_summary`
- `turn_taking`
- `turns`
- `transcript_summary`

Each turn should include:

- `speaker_id`
- `start`
- `end`
- `duration_seconds`
- `transcripts`
- `transcript_accuracy_estimate`
- `transcript_accuracy_estimate_method`
- `acoustic_features`
- `linguistic_features`
- `dialogue_acts`
- `engagement_markers`
- `emotion`
- `speaker_embeddings`
- `ssl_embeddings`
- `ppg_summary`
- `checks`

## TDD Implementation Plan

- [ ] Add a design doc and task checklist
- [ ] Add CPU-safe unit tests for text and turn-taking analytics helpers
- [ ] Add CPU-safe workflow orchestration tests using mocks for heavy models
- [ ] Implement recording/turn summary helpers
- [ ] Implement automated step checks
- [ ] Implement the new conversation-analysis workflow
- [ ] Export the workflow from `senselab.audio.workflows`
- [ ] Run targeted CPU-safe tests
- [ ] Mark heavy integration smoke tests for later CUDA validation when useful

## Testing Strategy

### CPU-safe now

- helper tests for linguistic features, dialogue acts, engagement markers, transcript consensus, turn-taking statistics, and PPG summarization
- orchestration tests with monkeypatched task backends
- error handling tests for invalid paths and empty inputs

### CUDA follow-up

No part of the orchestration layer strictly requires a GPU, but full end-to-end smoke tests with large pretrained backends are still best validated when CUDA is available:

- diarization smoke tests
- multi-model ASR smoke tests
- speech emotion recognition smoke tests
- speaker embedding/SSL embedding smoke tests
- PPG extraction smoke tests

These should be treated as integration validation, not blockers for CPU-first unit coverage.

## Key Design Decisions

- Prefer a new workflow over overloading `explore_conversation`, so we can preserve backward compatibility while adding richer outputs.
- Keep text analytics heuristic and dependency-light for the first version.
- Treat transcript accuracy as an estimate, not a ground-truth metric.
- Reuse existing feature extractors instead of introducing new model dependencies unless clearly necessary.
- Represent automated checks explicitly at both recording and turn granularity so failures are inspectable.

## Risks

- Zoom recordings can be video-only, audio-only, mono, stereo, or noisy, so ingestion and preprocessing checks must be robust.
- Some downstream models are heavy and slow on CPU; unit tests must mock them.
- Dialogue-act and politeness detection will begin as heuristics and should be labeled accordingly in outputs and docs.
- PPG tensor shapes can vary, so summarization should be shape-tolerant and degrade gracefully.
