# Feature Specification: Improve SER Tutorial with Better Models and Text Sentiment

**Feature Branch**: `20260424-152323-improve-ser-tutorial`
**Created**: 2026-04-24
**Status**: Draft
**Input**: User feedback: SER tutorial produces near-uniform probability distributions (~0.12 each class) that don't meaningfully discriminate emotions. The current model gives poor signal on real-world (non-acted) speech. Tutorial needs better models and should also add text-based sentiment detection from transcription.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Researcher Gets Meaningful Emotion Scores from Own Audio (Priority: P1)

A researcher loads their own voice recording into the SER tutorial and runs emotion classification. They receive emotion probability scores where the dominant emotion is clearly distinguishable (e.g., highest score >0.3 vs others <0.15), rather than near-uniform distributions (~0.12 across all classes). The tutorial explains how to interpret the scores and which models perform best on different types of speech (acted vs. natural).

**Why this priority**: This is the core problem — users are getting uninformative results. Without meaningful emotion discrimination, the tutorial fails its educational purpose and provides no practical value for researchers analyzing voice recordings.

**Independent Test**: Run the tutorial with a clearly emotional audio sample (e.g., from RAVDESS acted speech dataset). The top emotion score should be at least 2x higher than the average of other scores.

**Acceptance Scenarios**:

1. **Given** an audio file with clearly expressed emotion, **When** the user runs the best-performing SER model, **Then** the predicted emotion matches the expressed emotion with a confidence score at least 2x higher than the mean of other classes.
2. **Given** a natural (non-acted) speech recording, **When** the user runs the SER model, **Then** the tutorial explains that natural speech produces less dramatic score differences than acted speech, and provides guidance on interpretation.
3. **Given** the tutorial runs on RAVDESS test data, **When** using the recommended model, **Then** accuracy exceeds 70% (compared to ground truth labels).

---

### User Story 2 - User Runs Multiple SER Models for Comparison (Priority: P2)

A user runs the same audio through multiple SER models to compare their strengths. The tutorial shows at least 2-3 models with different characteristics (e.g., one trained on acted speech, one on conversational speech) and presents results side-by-side so the user can choose the most appropriate model for their use case.

**Why this priority**: Different models perform differently on different types of speech. Showing multiple models helps users understand the landscape and make informed choices for their own data.

**Independent Test**: Run the tutorial with sample audio through all listed models and display a comparison table of scores.

**Acceptance Scenarios**:

1. **Given** an audio file, **When** the user runs all recommended models, **Then** they see a side-by-side comparison table of emotion scores from each model.
2. **Given** the comparison table, **When** the user examines the results, **Then** the tutorial explains why different models may disagree and which is most appropriate for their data type.

---

### User Story 3 - User Gets Sentiment from Transcribed Text (Priority: P2)

A user transcribes their audio using ASR, then runs text-based sentiment analysis on the transcription. This complements the acoustic-based SER by providing linguistic sentiment (positive/negative/neutral) from the words spoken, not just how they were spoken.

**Why this priority**: Emotion from voice (paralinguistic) and sentiment from text (linguistic) are complementary signals. A user saying "I'm fine" in a sad voice shows neutral text sentiment but sad vocal emotion. Both perspectives together give a richer picture.

**Independent Test**: Run the tutorial with sample audio, get transcription, and see sentiment scores that reflect the textual content.

**Acceptance Scenarios**:

1. **Given** an audio file, **When** the user runs ASR followed by text sentiment analysis, **Then** they see sentiment labels (positive/negative/neutral) with confidence scores.
2. **Given** both acoustic emotion and text sentiment results, **When** displayed together, **Then** the tutorial explains how they can agree or diverge and what that means.

---

### Edge Cases

- What happens when the audio is very short (<1 second)? (Some models may not produce reliable results; the tutorial should warn about minimum duration.)
- What happens when the audio contains multiple emotions within one clip? (The model averages over the whole clip; the tutorial should mention segmentation for longer recordings.)
- What happens when the transcription is empty or the model fails to transcribe? (Text sentiment falls back gracefully with a message.)
- What happens with non-English audio? (The tutorial should note which models support which languages.)

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The tutorial MUST use at least one SER model that produces clearly differentiated emotion scores (not near-uniform) on both acted and natural speech.
- **FR-002**: The tutorial MUST compare at least 2-3 SER models and present results in a side-by-side table, explaining the strengths and weaknesses of each.
- **FR-003**: The tutorial MUST include a section on text-based sentiment analysis from transcribed speech, showing how linguistic sentiment complements acoustic emotion.
- **FR-004**: The tutorial MUST include guidance on interpreting emotion scores: what "good" discrimination looks like, why near-uniform scores indicate poor model fit, and how acted vs. natural speech affects results.
- **FR-005**: The tutorial MUST allow users to load their own audio file (recording or upload) and run all analyses on it.
- **FR-006**: The tutorial MUST show emotion scores as a clear visualization (bar chart with labeled emotions and scores), not just raw numbers.
- **FR-007**: The tutorial MUST follow established senselab tutorial conventions (install cell, restart admonition, device auto-detect, recording widget with sample fallback).

### Key Entities

- **Audio**: The speech recording being analyzed.
- **Emotion Score**: A probability distribution over emotion classes (e.g., happy, sad, angry, neutral, fearful, surprised, calm, disgust) from acoustic analysis.
- **Sentiment Score**: A classification (positive/negative/neutral) with confidence from text-based analysis of the transcription.
- **SER Model**: A pre-trained model that maps audio features to emotion probabilities.
- **Transcription**: The text output from automatic speech recognition.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On RAVDESS acted speech samples, the best recommended SER model achieves >70% accuracy and the top emotion score is at least 2x the mean of other classes.
- **SC-002**: The tutorial presents at least 3 SER models with a comparison table showing scores on the same audio.
- **SC-003**: Text sentiment analysis produces a clear positive/negative/neutral classification with confidence >0.5 on sample text.
- **SC-004**: Users can load their own audio and see both acoustic emotion and text sentiment results within 5 minutes of the analysis cells.
- **SC-005**: The tutorial includes at least one paragraph explaining why near-uniform scores occur and how to choose the right model for different speech types.

## Assumptions

- Users run the tutorial in Google Colab with access to a microphone or their own audio files.
- SER models are available via HuggingFace and can run on CPU (GPU optional but faster).
- Text sentiment analysis uses a pre-trained text classification model (no fine-tuning needed).
- The tutorial updates the existing `speech_emotion_recognition.ipynb` rather than creating a new notebook.
- The existing senselab `classify_emotions_from_speech()` API is used; model selection is the key improvement.
- For text sentiment, the tutorial uses senselab's ASR for transcription and a standard text sentiment model for classification.
