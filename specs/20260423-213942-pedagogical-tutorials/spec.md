# Feature Specification: Pedagogical Audio Tutorials and PPG Phoneme Durations

**Feature Branch**: `20260423-213942-pedagogical-tutorials`  
**Created**: 2026-04-23  
**Status**: Draft  
**Input**: User description: "update the tutorials in ~/Downloads/drive-download-20260424T013242Z-3-001 from a pedagogical perspective for students using the current senselab features. also revise an update the PR on getting phonemic durations from ppgs. a basic tutorial should show how senselab can be used to record an audio, look at spectrograms, extract pitch, formants and plot them. the second tutorial can focus on transcription and phonemic landmark detection on a recorded audio file. add these as two new tutorials to senselab."

## Clarifications

### Session 2026-04-23

- Q: Should tutorials support only recording or also downloaded audio files? → A: Both — dual-path input (record in browser OR use a downloaded sample audio file).
- Q: Should tutorials include a speaker matching demo? → A: Yes, a speaker matching demo from two recordings, placed in Tutorial 1 (acoustic features).
- Q: Where do the speaker matching demo go — Tutorial 1 or Tutorial 2? → A: Tutorial 1 (acoustic features tutorial).
- Q: Should updated course notebooks (00_getting_started, SHBT205-Lab) go into senselab `tutorials/` with CI testing? → A: Yes, both added to `tutorials/` and tested in CI. SPARC is already in senselab (articulatory coding), Promonet maps to PPG extraction, subprocess venvs handle dependency isolation.
- Q: Should the updated SHBT205-Lab replace raw library calls with senselab API equivalents? → A: Yes, replace all raw library calls (Whisper, promonet, etc.) with senselab API calls throughout.
- Q: Should emotion detection be included, and where? → A: Yes, in Tutorial 1 (acoustic features + speaker analysis).
- Q: How to handle PR #431 (PPG phoneme durations)? → A: Do NOT rebase/merge the old PR. Instead, write fresh phoneme duration analysis code in a new PR (review PR #431 code for ideas, implement cleanly on current codebase). Close PR #431 after the new implementation is complete.
- Q: Visualization style for phonemic analysis? → A: Tutorial 2 should include an aligned multi-panel plot with waveform, spectrogram, and phoneme boundaries stacked vertically on a shared time axis.
- Q: Visualization style for acoustic features? → A: Tutorial 1 should show time-varying features (pitch contour over time, formant tracks over time) rather than static summary statistics.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Student Records or Loads Audio and Explores Acoustic Features (Priority: P1)

A student in a speech/hearing course opens Tutorial 1 ("Audio Recording & Acoustic Analysis") in Google Colab. They either record their own voice or load a provided sample audio file. They then visualize the waveform, spectrogram, pitch contour, and formant frequencies using senselab. They run speech emotion recognition to classify the emotional content of the audio. Finally, they record a second audio sample (or load a second file) and run speaker matching to compare the two.

**Why this priority**: This is the foundational tutorial — it teaches students to capture/load audio, understand basic acoustic representations (waveform, spectrogram), extract/interpret pitch and formant features, detect emotions from speech, and compare speakers. Every subsequent tutorial builds on these skills.

**Independent Test**: Can be fully tested by running the notebook end-to-end in Colab using either microphone recording or the fallback sample audio: the student sees a waveform plot, a spectrogram, a pitch contour, formant values, an emotion classification result, and a speaker similarity score comparing two audio samples.

**Acceptance Scenarios**:

1. **Given** a student opens Tutorial 1 in Colab, **When** they run the install cell and restart the runtime, **Then** senselab is installed and importable with no errors.
2. **Given** the student chooses to record, **When** they speak into their microphone, **Then** a WAV file is saved and an `Audio` object is created from it.
3. **Given** the student chooses to use a sample file, **When** they run the download cell, **Then** a sample audio file is fetched and loaded as an `Audio` object.
4. **Given** an audio (recorded or loaded), **When** the student runs the visualization cells, **Then** they see a waveform plot, a spectrogram (linear and/or mel), and can play back the audio inline.
5. **Given** an audio, **When** the student runs the pitch extraction cell, **Then** they see a pitch (F0) contour plotted over time with labeled axes, showing how pitch varies throughout the utterance.
6. **Given** an audio, **When** the student runs the formant extraction cell, **Then** they see F1 and F2 formant tracks plotted over time, showing how vocal tract resonances change during the utterance.
7. **Given** an audio, **When** the student runs the emotion detection cell, **Then** they see a classified emotion label (e.g., neutral, happy, angry, sad) with confidence scores and an explanation of how speech emotion recognition works.
8. **Given** two audio samples (recorded or loaded), **When** the student runs the speaker matching cell, **Then** they see a similarity score indicating whether the two samples are from the same speaker, with an explanation of how to interpret the result.

---

### User Story 2 - Student Transcribes Audio and Identifies Phoneme Landmarks (Priority: P2)

A student opens Tutorial 2 ("Transcription & Phonemic Analysis") in Colab. Using a pre-recorded audio file (or their own recording from Tutorial 1), they run automatic speech recognition, then extract phoneme-level timing information using forced alignment and PPG-based phoneme duration analysis. They visualize a phoneme timeline showing when each phoneme was produced.

**Why this priority**: This builds on Tutorial 1 by adding language-level analysis (ASR, phoneme timing). It demonstrates senselab's transcription, forced alignment, and PPG phoneme duration features in a pedagogically coherent sequence.

**Independent Test**: Can be fully tested by running the notebook with a sample English audio file: the student sees a transcription, a forced alignment output with word/phone boundaries, and a PPG phoneme timeline plot.

**Acceptance Scenarios**:

1. **Given** an audio file (sample or recorded), **When** the student runs the ASR cell, **Then** they see a text transcription of the speech.
2. **Given** an audio file and its transcription, **When** the student runs forced alignment, **Then** they see word-level and phone-level time boundaries, visualized as an aligned multi-panel plot with waveform, spectrogram, and phoneme labels stacked vertically on a shared time axis.
3. **Given** an audio file, **When** the student runs PPG extraction and phoneme duration analysis, **Then** they see a timeline plot of detected phonemes with their start/end times and durations.
4. **Given** phoneme timing data, **When** the student examines the output, **Then** they can identify specific phonemes and their temporal characteristics in their speech.

---

### User Story 3 - Existing Course Tutorials Updated to Use Current Senselab APIs (Priority: P2)

An instructor updates their SHBT205 course materials. The existing `00_getting_started.ipynb` and `SHBT205-Lab.ipynb` tutorials are revised to: use the current senselab release (not a PR branch), replace all raw external library calls with senselab API equivalents (Whisper → `transcribe_audios`, raw promonet → PPG extraction API, etc.), follow the established tutorial conventions (minimal install, restart admonition, auto-detect device), and be added to the senselab `tutorials/` directory for CI testing.

**Why this priority**: Equal to P2 — the existing course tutorials must work with the current codebase. Students should learn senselab's API, not raw library internals.

**Independent Test**: Both updated notebooks run successfully via papermill in CI (CPU tests) and in Colab, using senselab API calls throughout.

**Acceptance Scenarios**:

1. **Given** the updated 00_getting_started.ipynb, **When** run in Colab or CI, **Then** all cells execute without error using the current senselab release and senselab API calls.
2. **Given** the updated SHBT205-Lab.ipynb, **When** run in Colab or CI, **Then** SPARC sections use senselab's articulatory coding API, Promonet/PPG sections use senselab's PPG extraction API, ASR uses `transcribe_audios`, and no raw external library calls remain.
3. **Given** both updated notebooks, **When** added to `tutorials/manifest.json`, **Then** they pass CI tutorial tests (both CPU and GPU when applicable).

---

### User Story 4 - Fresh PPG Phoneme Duration Implementation (Priority: P3)

New phoneme duration analysis functionality is implemented fresh on the current codebase, informed by but not directly merging PR #431. The code is reviewed thoroughly, written cleanly against the current senselab APIs, and tested. After the new implementation passes CI, PR #431 is closed.

**Why this priority**: This is a dependency for Tutorial 2's phoneme analysis section. PR #431 was based on an older senselab version and should not be rebased — instead, the relevant ideas are reimplemented cleanly.

**Independent Test**: New phoneme duration functions pass all tests locally and in CI on the current codebase.

**Acceptance Scenarios**:

1. **Given** the current codebase, **When** phoneme duration analysis is implemented fresh, **Then** all existing and new tests pass.
2. **Given** an audio file, **When** phoneme duration analysis is called on extracted PPGs, **Then** it returns phoneme labels with duration information.
3. **Given** an audio file, **When** a phoneme timeline plot is generated, **Then** it produces a visual timeline of phoneme segments with labeled boundaries.
4. **Given** the new implementation passes CI, **When** reviewed, **Then** PR #431 is closed as superseded.

---

### Edge Cases

- What happens when the student's microphone is not available or permission is denied? (Tutorial 1 provides a fallback sample audio file that can be downloaded instead.)
- What happens when the recorded audio is silent or too short for feature extraction? (Display a meaningful message.)
- What happens when PPG extraction returns no phoneme segments for very short or noisy audio? (Tutorial 2 should handle empty results gracefully.)
- What happens when forced alignment fails because the transcription doesn't match the audio? (Show how mismatches appear and how to interpret them.)
- What happens when speaker matching compares two very short recordings? (Display similarity score with a note about confidence for short utterances.)

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Tutorial 1 MUST support dual-path audio input: an in-browser audio recording mechanism (for students with microphones) AND a downloadable sample audio file (for students without microphones or for CI testing).
- **FR-002**: Tutorial 1 MUST demonstrate waveform plotting, spectrogram plotting (using `plot_waveform`, `plot_specgram`), inline audio playback (`play_audio`), time-varying pitch contour, time-varying formant tracks, and speech emotion recognition using senselab functions. Acoustic features MUST be shown as plots over time, not just static summary statistics.
- **FR-003**: Tutorial 1 MUST include a speaker matching section that compares two audio samples (recorded or loaded) and displays a similarity score with interpretation guidance.
- **FR-004**: Tutorial 2 MUST demonstrate automatic speech recognition using senselab's `transcribe_audios` with a Whisper model.
- **FR-005**: Tutorial 2 MUST demonstrate forced alignment using senselab's forced alignment API, showing word-level and phone-level boundaries in an aligned multi-panel visualization (waveform + spectrogram + phoneme labels on a shared time axis).
- **FR-006**: Tutorial 2 MUST demonstrate PPG-based phoneme duration analysis and timeline plotting using the newly implemented phoneme duration functions.
- **FR-007**: All tutorials (2 new + 2 updated) MUST follow the established senselab tutorial conventions: minimal `!uv pip install` cell, restart runtime admonition, auto-detect device, cleared outputs, Colab badge.
- **FR-008**: The existing `00_getting_started.ipynb` and `SHBT205-Lab.ipynb` MUST be updated to replace all raw external library calls with senselab API equivalents, install from the current senselab release (not a PR branch), and be added to the `tutorials/` directory with `manifest.json` entries for CI testing.
- **FR-009**: Phoneme duration detection MUST be implemented fresh on the current codebase (reviewing PR #431 code for reference but not merging it). After the new implementation passes CI, PR #431 MUST be closed as superseded.
- **FR-010**: Each tutorial MUST include pedagogical context — brief explanations of what each acoustic feature represents and why it matters, not just code execution.
- **FR-011**: Both new tutorials MUST support a downloaded sample audio file path so that CI can execute them fully without microphone access (recording cells are optional/skipped in CI).

### Key Entities

- **Audio**: The primary data object — a recorded or loaded speech signal with waveform, sampling rate, and metadata.
- **Pitch (F0)**: The fundamental frequency contour extracted from speech, representing vocal fold vibration rate.
- **Formants (F1, F2)**: Resonant frequencies of the vocal tract, key to vowel identification.
- **Spectrogram**: Time-frequency representation of the audio signal.
- **PPG (Phonetic Posteriorgram)**: Frame-level probability distribution over phoneme classes.
- **Phoneme Timeline**: Temporal segmentation showing when each phoneme occurs in the audio.
- **Forced Alignment**: Time-alignment of known transcript text to the audio signal at word and phone level.
- **Speech Emotion**: The classified emotional state (e.g., neutral, happy, angry, sad) detected from acoustic cues in the speech signal.
- **Speaker Embedding**: A fixed-length vector representing a speaker's voice characteristics, used for speaker matching/verification.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All 4 notebooks (2 new + 2 updated) execute successfully via papermill in CI with no errors, using downloaded sample audio files.
- **SC-002**: A student with no prior senselab experience can complete Tutorial 1 (load/record, visualize, extract features, emotion detection, speaker match) within 30 minutes.
- **SC-003**: Tutorial 2 produces a phoneme timeline plot that correctly labels phonemes in a clear English sentence.
- **SC-004**: Fresh phoneme duration analysis implementation passes all CI checks and is merged; PR #431 is closed as superseded.
- **SC-005**: The updated course notebooks use only senselab API calls (no raw Whisper, promonet, or other external library calls) and produce equivalent pedagogical outputs to the originals.
- **SC-006**: Each tutorial includes at least one paragraph of explanatory text per major section helping students understand the acoustic/linguistic concepts, not just the code.
- **SC-007**: Speaker matching in Tutorial 1 correctly identifies same-speaker pairs with a similarity score and provides interpretive guidance for students.

## Assumptions

- Students use Google Colab with Chrome browser (microphone access requires HTTPS, which Colab provides).
- Audio recording in Colab uses `ipywebrtc` or equivalent browser-based widget; the recording widget produces audio that can be converted to WAV format.
- Students may or may not have microphone access; all tutorials are fully functional using downloaded sample audio files alone.
- The PPG subprocess venv (espnet-based) works on Colab's CPU runtime; GPU is not required but benefits from it.
- SPARC articulatory coding is available via senselab's existing API; Promonet PPG extraction maps to senselab's PPG extraction API. Subprocess venvs handle any dependency isolation needed.
- The updated `SHBT205-Lab.ipynb` replaces all raw external library usage with senselab API equivalents — no raw Whisper, promonet, or SPARC library calls remain.
- PR #431 code is used as reference only — the phoneme duration feature is reimplemented fresh on the current codebase. PR #431 is closed after the new implementation is complete.
- Tutorial audio files will be hosted in the senselab repository's test data directory or downloaded from GitHub raw URLs during notebook execution.
- All 4 notebooks are added to `tutorials/` and `manifest.json` for CI testing via papermill.
