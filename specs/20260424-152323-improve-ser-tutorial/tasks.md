# Tasks: Improve SER Tutorial with Better Models and Text Sentiment

**Input**: Design documents from `/specs/20260424-152323-improve-ser-tutorial/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, quickstart.md

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Research

**Purpose**: Survey current state-of-the-art in speech emotion recognition to select the best models.

- [x] T001 Research latest conference papers (INTERSPEECH 2025, ICASSP 2025, ACL 2025) on speech emotion recognition — identify top-performing models, datasets (IEMOCAP, MSP-Podcast, MELD, CREMA-D), and evaluation metrics (UAR, WAR, F1). Search for "speech emotion recognition survey 2025" and "SER benchmark leaderboard". Document findings in specs/20260424-152323-improve-ser-tutorial/research.md
- [x] T002 Check HuggingFace model hub leaderboards and papers-with-code benchmarks for speech emotion recognition — identify models with best UAR on IEMOCAP (4-class and 6-class) and MSP-Podcast. Look for emotion2vec, WavLM-based, and HuBERT-based models. Document top 5 models with scores in specs/20260424-152323-improve-ser-tutorial/research.md
- [x] T003 Test candidate SER models locally to verify they work with senselab's classify_emotions_from_speech API — run each model on the test audio file and compare score distributions. Models that produce near-uniform scores on conversational speech should be flagged. Document results in specs/20260424-152323-improve-ser-tutorial/research.md

**Checkpoint**: Best models identified and verified

---

## Phase 2: User Story 1 — Meaningful Emotion Scores (Priority: P1) 🎯 MVP

**Goal**: Users get clearly differentiated emotion scores that meaningfully discriminate emotions, with interpretation guidance.

**Independent Test**: Run the notebook with sample audio — the best model produces top emotion score ≥2x mean of other classes on acted speech.

### Implementation

- [ ] T004 [US1] Create the notebook structure in tutorials/audio/speech_emotion_recognition.ipynb — title, Colab badge, overview, install cell, restart admonition, imports, device setup, recording widget, load audio cell (same pattern as other tutorials)
- [ ] T005 [US1] Add "Speech Emotion Recognition" section in tutorials/audio/speech_emotion_recognition.ipynb — run the best-performing model (from research) on the user's audio, display emotion scores as a labeled bar chart with clear visual hierarchy
- [ ] T006 [US1] Add "Understanding Emotion Scores" section in tutorials/audio/speech_emotion_recognition.ipynb — explain near-uniform distributions, acted vs natural speech, relative ordering vs absolute scores, dimensional vs categorical approaches
- [ ] T007 [US1] Add acted speech comparison in tutorials/audio/speech_emotion_recognition.ipynb — load a RAVDESS sample with clear acted emotion, run the same model, show the contrast in score distribution vs natural speech
- [ ] T008 [US1] Test US1 locally: `uv run papermill tutorials/audio/speech_emotion_recognition.ipynb /dev/null --cwd . -k python3 --execution-timeout 1200`

**Checkpoint**: Tutorial produces meaningful emotion scores with interpretation guidance

---

## Phase 3: User Story 2 — Multi-Model Comparison (Priority: P2)

**Goal**: Users compare 2-3 SER models side-by-side and understand which to use for their data.

**Independent Test**: Comparison table shows scores from all models on the same audio.

### Implementation

- [ ] T009 [US2] Add "Model Comparison" section in tutorials/audio/speech_emotion_recognition.ipynb — run at least 3 models (best discrete, RAVDESS-trained, continuous/dimensional) on the same audio, display side-by-side bar charts
- [ ] T010 [US2] Add comparison table and model selection guidance in tutorials/audio/speech_emotion_recognition.ipynb — print table with model name, training data, best use case, and scores; explain when to use each model
- [ ] T011 [US2] Test US2 locally with papermill

**Checkpoint**: Multiple models compared with clear guidance

---

## Phase 4: User Story 3 — Text Sentiment from Transcription (Priority: P2)

**Goal**: Users see text-based sentiment analysis complementing acoustic emotion analysis.

**Independent Test**: Transcription + sentiment scores displayed alongside acoustic emotion.

### Implementation

- [ ] T012 [US3] Add "Text Sentiment from Transcription" section in tutorials/audio/speech_emotion_recognition.ipynb — transcribe audio with whisper-large-v3-turbo, run text sentiment (cardiffnlp/twitter-roberta-base-sentiment-latest via transformers pipeline), display results
- [ ] T013 [US3] Add "Comparing Acoustic Emotion and Text Sentiment" section in tutorials/audio/speech_emotion_recognition.ipynb — side-by-side display of acoustic emotion scores and text sentiment, explain when they agree/disagree and what that means
- [ ] T014 [US3] Test US3 locally with papermill

**Checkpoint**: Text sentiment complements acoustic emotion

---

## Phase 5: Polish & Cross-Cutting Concerns

- [ ] T015 Add "Applying to Your Own Data" section in tutorials/audio/speech_emotion_recognition.ipynb — how to load own audio, batch-process multiple files, choose the right model, tips for better results
- [ ] T016 Add summary table comparing all approaches in tutorials/audio/speech_emotion_recognition.ipynb
- [ ] T017 Clear all outputs and run pre-commit: `uv run pre-commit run --all-files`
- [ ] T018 Run full test: `uv run papermill tutorials/audio/speech_emotion_recognition.ipynb /dev/null --cwd . -k python3 --execution-timeout 1200`
- [ ] T019 Push to branch, create PR to alpha with `test-tutorials` label
- [ ] T020 Verify CI passes and merge

---

## Dependencies & Execution Order

### Phase Dependencies

- **Research (Phase 1)**: Start immediately — informs model selection for all stories
- **US1 (Phase 2)**: Depends on Research completion
- **US2 (Phase 3)**: Depends on US1 (notebook structure must exist)
- **US3 (Phase 4)**: Depends on US1 (notebook structure must exist), can run in parallel with US2
- **Polish (Phase 5)**: Depends on all stories complete

### Parallel Opportunities

- **T001 + T002**: Research tasks can run in parallel (different sources)
- **US2 + US3**: Can run in parallel (different sections of same notebook, but sequential editing recommended)

---

## Implementation Strategy

### MVP First (User Story 1)

1. Complete Research (Phase 1) — identify best models
2. Build notebook with US1 — meaningful emotion scores + interpretation
3. **STOP and VALIDATE** — does the best model produce clear discrimination?
4. Add US2 (multi-model) and US3 (text sentiment)
5. Polish and push

---

## Notes

- Research phase is critical — model selection determines tutorial quality
- The continuous SER model runs in subprocess venv, may be slow on CPU
- Text sentiment uses transformers pipeline directly (no new senselab API)
- RAVDESS dataset samples used for acted speech comparison
