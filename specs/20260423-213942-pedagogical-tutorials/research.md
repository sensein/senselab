# Research: Pedagogical Audio Tutorials and PPG Phoneme Durations

**Date**: 2026-04-23

## R1: Audio Recording in Colab

**Decision**: Use JavaScript-based recording widget (Google Colab `audio` snippet pattern) with fallback to downloaded sample audio.

**Rationale**: `ipywebrtc` requires widget infrastructure that can be fragile in Colab. The standard Colab JS approach (`google.colab.output.eval_js`) is more reliable and doesn't require extra pip installs. For CI testing via papermill, recording cells are skipped and a downloaded sample file is used instead.

**Alternatives considered**:
- `ipywebrtc`: Widget-based, sometimes breaks with Colab kernel restarts
- `sounddevice`: Requires system audio drivers, not available in Colab
- JavaScript Web Audio API: Native browser support, no extra dependencies (chosen)

## R2: Pitch Contour Plotting

**Decision**: Use `extract_pitch_from_audios()` from torchaudio backend for time-series pitch, plot with matplotlib manually. Praat descriptors for summary statistics.

**Rationale**: torchaudio returns frame-level pitch tensors ideal for contour plotting. Praat functions return summary statistics (mean/std), not time-series. Tutorials need both: contour visualization (torchaudio) and descriptive stats (praat).

**Alternatives considered**:
- Praat only: Returns descriptors, not plottable contours
- OpenSMILE: Returns feature vectors, not time-aligned pitch
- torchaudio only: Good for contours but misses speech-adapted floor/ceiling detection

## R3: Formant Extraction Approach

**Decision**: Use `extract_praat_parselmouth_features_from_audios()` unified API which wraps pitch detection + formant extraction + other features.

**Rationale**: The unified API handles pitch floor/ceiling detection automatically, then extracts F1/F2 formants. Simpler for students than calling individual functions. Returns dict with all features including `f1_mean`, `f2_mean`, etc.

**Alternatives considered**:
- Manual two-step (pitch detection → formant extraction): More code, same result
- OpenSMILE: Less interpretable output for formants

## R4: Emotion Classification Model Selection

**Decision**: Use `classify_emotions_from_speech()` with discrete HFModel (e.g., `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`).

**Rationale**: Discrete emotion models produce clear, interpretable labels (happy, angry, sad, neutral) — ideal for students. Continuous/valence models require more explanation and are less intuitive for a first tutorial.

**Alternatives considered**:
- Continuous SER models: Valence/arousal/dominance — more nuanced but harder to interpret pedagogically
- Multiple models: Overkill for Tutorial 1; the existing SER tutorial already covers this in depth

## R5: Speaker Verification Approach

**Decision**: Use `verify_speaker()` with default SpeechBrainModel and two audio files/recordings.

**Rationale**: Returns `(similarity_score, is_same_speaker)` tuple — immediately interpretable for students. Default model works well on CPU. Threshold of 0.25 is sensible default.

**Alternatives considered**:
- Raw embedding extraction + manual cosine similarity: More educational but too complex for Tutorial 1
- Multiple models: Unnecessary for a demo

## R6: SPARC Integration in Updated SHBT205-Lab

**Decision**: Use `SparcFeatureExtractor.extract_sparc_features()` from senselab's SPARC API, which already wraps the subprocess venv.

**Rationale**: SPARC is fully integrated into senselab with subprocess venv isolation. The SHBT205-Lab currently calls SPARC's raw `coder.encode()/decode()/convert()` — these map directly to senselab's wrapper. Voice conversion (`coder.convert()`) may need a new senselab wrapper or remain as a documented external call.

**Alternatives considered**:
- Keep raw SPARC calls: Contradicts the "senselab API only" clarification
- Skip SPARC section: Loses pedagogical value of articulatory features

## R7: PPG / Promonet Mapping

**Decision**: Promonet's phonetic posteriorgram extraction maps to senselab's `extract_ppgs_from_audios()`. Duration analysis uses `extract_mean_phoneme_durations()` and `plot_ppg_phoneme_timeline()` from PR #431.

**Rationale**: Senselab's PPG extraction uses the same underlying model (espnet-based). PR #431 adds the duration analysis and timeline plotting that the SHBT205-Lab demonstrates manually.

**Alternatives considered**:
- Keep raw promonet calls: Contradicts clarification
- Use only forced alignment for phoneme timing: Misses the PPG probability-based approach

## R8: PR #431 Approach

**Decision**: Do NOT rebase/merge PR #431. Instead, implement phoneme duration analysis fresh on the current codebase, using PR #431 code as reference. Close PR #431 after the new implementation is complete.

**Rationale**: PR #431 was based on an older senselab version. Rebasing risks introducing stale patterns and conflicts. A fresh implementation ensures the code is clean, idiomatic with the current API surface, and thoroughly reviewed.

**Alternatives considered**:
- Rebase and merge PR #431: Risk of stale code patterns and merge conflicts
- Cherry-pick commits: Still carries old code structure

## R9: CI Testing Strategy

**Decision**: All 4 notebooks added to `tutorials/manifest.json` with `benefits_from_gpu: true` and appropriate CPU/GPU timeouts. Recording cells are conditional (skipped when no display/microphone available — detected by checking `os.environ.get("COLAB_RELEASE_TAG")`). Downloaded sample audio used for CI.

**Rationale**: Consistent with established tutorial CI pattern. PPG and SPARC subprocess venvs work on CPU but are slow — generous CPU timeouts needed.

**Alternatives considered**:
- Skip CI for course notebooks: Contradicts clarification that all go in `tutorials/`
- GPU-only testing: Excludes CPU CI which catches most issues

## R10: Tutorial File Layout

**Decision**: 
- `tutorials/audio/audio_recording_and_acoustic_analysis.ipynb` (new Tutorial 1)
- `tutorials/audio/transcription_and_phonemic_analysis.ipynb` (new Tutorial 2)
- `tutorials/audio/00_getting_started.ipynb` (updated, already exists)
- `tutorials/audio/shbt205_lab.ipynb` (updated from course materials)

**Rationale**: Follows existing `tutorials/audio/` convention. Names are descriptive and match the spec's user story titles.
