# Research: Improve SER Tutorial

**Date**: 2026-04-24

## R1: Best SER Models for Non-Acted Speech

**Decision**: Use multiple models to show the range of behavior:
1. `superb/wav2vec2-base-superb-er` — trained on IEMOCAP (semi-natural conversational speech), better for real-world audio
2. `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` — trained on RAVDESS (acted speech), good for dramatic emotions
3. `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` — continuous SER (valence/arousal/dominance), dimensional approach

**Rationale**: The user's near-uniform scores (~0.12 per class) indicate the current model doesn't discriminate well on natural speech. Models trained on conversational data (IEMOCAP) generalize better. Showing multiple models teaches users to match the model to their data type.

**Verified locally**:
- `superb/wav2vec2-base-superb-er`: WORKS — produces clear discrimination (0.67 neutral vs 0.32 happy on test audio)
- `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`: WORKS — confirms near-uniform (~0.127) on same audio
- `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`: WORKS — runs in subprocess venv
- `speechbrain/emotion-recognition-wav2vec2-IEMOCAP`: FAILS — AttributeError, incompatible with current speechbrain
- `emotion2vec/emotion2vec_plus_large`: FAILS — no model_type in config.json, needs custom API (not HF pipeline)

**Literature survey** (INTERSPEECH/ICASSP/ACL 2024-2025):
- emotion2vec (ACL 2024): Best overall SER model, but uses custom API — not compatible with senselab's HF pipeline
- EmoBox (2024): Multilingual SER toolkit, benchmarks multiple models
- Papers with Code IEMOCAP leaderboard: Various approaches, 73-82% UAR typical for audio-only models
- Key insight: Self-supervised models (wav2vec2, HuBERT, WavLM) dominate leaderboards

**Alternatives considered**:
- Single "best" model: Doesn't teach model selection
- Fine-tuning on user data: Out of scope for a tutorial
- emotion2vec: Best SER model but requires custom API, not compatible with senselab's classify_emotions_from_speech
- SpeechBrain IEMOCAP model: Incompatible with current version

## R2: Text Sentiment Analysis

**Decision**: Use HuggingFace `text-classification` pipeline directly in the tutorial (no new senselab API needed). Model: `cardiffnlp/twitter-roberta-base-sentiment-latest` (3-class: positive/negative/neutral).

**Rationale**: Text sentiment is a simple pipeline call. Adding a full senselab API for text classification is overkill for one tutorial section — it can be done directly with the transformers library. The tutorial already installs transformers via senselab dependencies.

**Alternatives considered**:
- Add senselab text classification API: More engineering than needed for the tutorial
- Use sentence-transformers for sentiment: Not designed for classification
- Use a large LLM for sentiment: Overkill, slow, requires API keys

## R3: Tutorial Structure

**Decision**: Restructure the tutorial around practical use:
1. Load your own audio (recording or upload)
2. Run multiple SER models → comparison table
3. Transcribe the audio (ASR)
4. Run text sentiment on transcription
5. Compare acoustic emotion vs text sentiment
6. Guidance section on model selection and interpretation

**Rationale**: The current tutorial focuses on benchmarking accuracy on RAVDESS. While educational, users need to see how to apply SER to their own data and interpret results. The benchmark can be a secondary section.

## R4: Why Near-Uniform Scores Occur

**Decision**: Add an explicit section explaining this phenomenon:
- Models trained on acted speech (RAVDESS) expect exaggerated emotional expression
- Natural speech has subtler cues, so softmax distributes probability more evenly
- The absolute scores matter less than the relative ordering and margin
- Some models are simply not good at certain speech types

**Rationale**: This directly addresses the user feedback. Without this explanation, users lose trust in the tool.
