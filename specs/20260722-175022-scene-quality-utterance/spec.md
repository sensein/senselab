# Feature Specification: Scene-aware presence axis + improved utterance uncertainty

**Feature Branch**: `20260722-175022-scene-quality-utterance`
**Created**: 2026-07-22
**Status**: Draft
**Input**: User description: "Rework the presence axis into a scene-aware dimension quantifying audio quality (SNR, clipping, reverb, bandwidth) and background sound sources (people/machine/environment), add per-axis temporal grids and frame-level posteriors, and improve the utterance uncertainty estimator"

## Context

The `audio_analysis` uncertainty workflow (`src/senselab/audio/workflows/audio_analysis`) emits three per-bucket uncertainty time series — `presence` (was a speaker present?), `identity` (was it the same speaker?), and `utterance` (what was said?). Today the presence axis collapses many coarse and fine voters into a single binary-vote Shannon entropy on a fixed 0.5 s grid, and background sound events are only used as a binary "speech / not-speech" cue. Users analyzing Bridge2AI-Voice recordings (speech plus cough/breathing tasks, often with people, machines, or environmental noise in the background) need the presence axis to also characterize **how good the signal is** and **what else is in the scene**, and they need the utterance axis to give a more trustworthy "what was said" uncertainty. A handoff note (`SPEECH_PRESENCE_CERTAINTY_ANALYSIS.md`) established that the real bottleneck at short segments is temporal resolution and voter calibration, not tagger accuracy.

This feature keeps the `presence` axis name and its existing `aggregated_uncertainty` output (no breaking change to downstream consumers) and adds scene-quality and sound-source columns alongside it, introduces per-axis temporal grids and frame-level speech posteriors, and reworks the utterance estimator.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Per-bucket audio quality on the presence axis (Priority: P1)

A researcher runs the audio-analysis workflow on a clinical recording and wants to know, over time, how usable the signal is: where it is noisy, clipped, reverberant, or band-limited (e.g. telephone-quality). They read the `presence` parquet and see, per bucket, a set of `[0, 1]` degradation scores (0 = clean) for signal-to-noise ratio, clipping, reverberation, and effective bandwidth, plus a quality-uncertainty score reflecting disagreement among the independent quality estimators.

**Why this priority**: Signal quality is the single most requested addition and it directly gates the trustworthiness of every other axis. It reuses estimators senselab already implements, so it delivers value with the least new machinery.

**Independent Test**: Run the workflow on a tutorial clip and verify the `presence` parquet contains per-bucket `quality_snr`, `quality_clip`, `quality_reverb`, `quality_bandwidth` in `[0, 1]` plus a `quality_uncertainty` column, and that a clean clip scores low degradation while a synthetically noised/clipped clip scores higher.

**Acceptance Scenarios**:

1. **Given** a clean recording, **When** the workflow runs, **Then** every quality degradation score is near 0 across all buckets.
2. **Given** a recording with a segment of added broadband noise, **When** the workflow runs, **Then** `quality_snr` rises in the noised buckets and returns to baseline outside them.
3. **Given** a recording with digital clipping in one region, **When** the workflow runs, **Then** `quality_clip` is elevated only in the clipped buckets.
4. **Given** a telephone-band (≤4 kHz) recording, **When** the workflow runs, **Then** `quality_bandwidth` is elevated across the recording.
5. **Given** multiple independent SNR estimators disagree in a bucket, **When** the workflow runs, **Then** `quality_uncertainty` for that bucket is elevated.

---

### User Story 2 - Background sound-source categorization (Priority: P1)

A researcher wants to know when non-target sound sources are present and what kind they are — other **people** (laughter, cough, chatter), **machines** (engine, HVAC, tools, vehicle), or **environment** (wind, rain, birds, water) — separately from the target speech. They read the `presence` parquet and see, per bucket, the relative mass assigned to `speech`, `people`, `machine`, and `environment`, plus which category dominates.

**Why this priority**: Knowing the interfering source type is essential for interpreting both quality and utterance uncertainty, and it is the second half of the "scene analysis" ask. It reuses the AST and YAMNet outputs the workflow already computes.

**Independent Test**: Run the workflow on a clip containing speech over background traffic and verify the `presence` parquet has per-bucket `src_speech`, `src_people`, `src_machine`, `src_environment` (summing to ~1) and a `src_dominant` label, with `src_machine` elevated during the traffic.

**Acceptance Scenarios**:

1. **Given** a speech-only clip, **When** the workflow runs, **Then** `src_speech` dominates in speech buckets.
2. **Given** a clip with background machinery, **When** the workflow runs, **Then** `src_machine` is the elevated non-speech category in the affected buckets.
3. **Given** a clip with overlapping background talkers or laughter, **When** the workflow runs, **Then** `src_people` is elevated.
4. **Given** every AudioSet class the scene classifiers can emit, **When** the category map is applied, **Then** each class maps to exactly one of `{speech, people, machine, environment}` (complete, non-overlapping coverage).

---

### User Story 3 - Fine temporal resolution for presence (Priority: P2)

A researcher analyzing brief events (a cough onset, an inter-word breath) finds the fixed 0.5 s grid too coarse and the binary VAD segments too smoothed to localize them. They configure a finer presence grid and receive a smooth, per-bucket **speech-presence confidence** derived from continuous frame-level posteriors, with a separate **uncertainty** that rises where the model is genuinely ambiguous or the bucket straddles an onset.

**Why this priority**: Temporal resolution is the handoff note's central technical unlock and underpins the quality and source signals at short spans, but it is more invasive than the additive quality/source columns, so it follows them.

**Independent Test**: Run the workflow with the presence grid set to a 0.1 s window / 0.02 s hop and verify the presence series has the expected finer bucket count, a `presence_confidence` and a `presence_uncertainty` column, and that a hand-marked brief onset is localized to within one hop.

**Acceptance Scenarios**:

1. **Given** the default configuration, **When** the workflow runs, **Then** the presence axis reports on a 0.1 s window / 0.02 s hop grid while identity and utterance keep their own grids.
2. **Given** a bucket fully inside steady speech, **When** the workflow runs, **Then** `presence_confidence` is high and `presence_uncertainty` is low.
3. **Given** a bucket straddling a speech onset, **When** the workflow runs, **Then** `presence_uncertainty` is elevated relative to neighboring steady buckets.
4. **Given** coarse voters (whole-window scene tags, per-30 s no-speech probability, sentence-level transcripts), **When** the presence axis is computed on a fine grid, **Then** those voters do not cast identical per-bucket votes that inflate agreement; they contribute only as a slowly-varying context prior.

---

### User Story 4 - Improved utterance uncertainty estimator (Priority: P2)

A researcher relies on the `utterance` axis to flag where the transcription is untrustworthy. Today boundary effects on the fixed grid inflate word-error-rate disagreement, and the confidence scores are uncalibrated. They want an overlapping word-scale grid that stops penalizing words that straddle boundaries, calibrated confidence, an additional token-level uncertainty signal, and coupling so that utterance uncertainty rises where the scene quality is poor or a competing source masks the target.

**Why this priority**: This is a distinct axis from the scene rework and depends on the quality signal from US1, so it is sequenced after it. It is high value because utterance uncertainty is what most reviewers act on.

**Independent Test**: Run the workflow on a two-speaker clip with a noisy region and verify the `utterance` parquet uses an overlapping word-scale grid, exposes a token-level uncertainty sub-signal, and that utterance uncertainty in the noisy region is higher than the same transcript content in a clean region.

**Acceptance Scenarios**:

1. **Given** a word straddling a bucket boundary, **When** utterance uncertainty is computed, **Then** the straddling word does not inflate the disagreement for either adjacent bucket.
2. **Given** transcripts with per-token confidence available, **When** the workflow runs, **Then** a token-level uncertainty sub-signal is included alongside the existing pairwise and native-confidence sub-signals.
3. **Given** two buckets with identical transcript agreement but different scene quality, **When** utterance uncertainty is computed, **Then** the bucket with poorer quality / stronger competing source reports higher utterance uncertainty, and the coupling factor is recorded (not applied silently).
4. **Given** confidence scores from different ASR backends, **When** they are combined, **Then** they are mapped to a common calibrated `[0, 1]` scale.

---

### User Story 5 - Synthetic calibration of quality and presence signals (Priority: P3)

A developer needs the quality degradation scores and presence/utterance confidences to mean the same thing across recordings. Because no labeled dataset of per-bucket SNR/reverb is on hand, they run a helper that synthesizes calibration data by mixing a clean speech clip with known noise at controlled SNRs and convolving known room impulse responses at target reverberation times, then fits the normalization (and a temperature scaling) so the reported scores track the known ground truth. They review a validation artifact comparing reported vs. true SNR/reverberation.

**Why this priority**: Calibration improves interpretability but the raw signals are usable (documented, uncalibrated) without it, so it is the lowest priority and can land last.

**Independent Test**: Run the calibration helper on a tutorial clip, confirm it produces mixtures at known SNR/RT60, fits the degradation-score normalization, and emits a validation plot/table of reported vs. true values with error within a documented tolerance.

**Acceptance Scenarios**:

1. **Given** a clean clip and a target SNR sweep, **When** the helper runs, **Then** it produces mixtures at each target SNR and reports the estimator's response at each.
2. **Given** the fitted normalization, **When** applied to held-out mixtures, **Then** reported degradation increases monotonically with true degradation.
3. **Given** the calibration outputs, **When** a developer inspects them, **Then** a validation artifact shows reported-vs-true agreement and the fitted parameters are persisted for reuse.

---

### Edge Cases

- **Audio shorter than a bucket window**: the whole clip is treated as a single bucket for the affected axis.
- **A required model is unavailable** (e.g. the gated scene/quality model cannot be loaded without a token): the affected sub-signal columns are emitted as null/NaN with a recorded reason, and the axis still produces its other columns — the workflow does not abort.
- **Non-16 kHz or multi-channel input**: quality estimators that require mono 16 kHz receive a resampled/downmixed view; the original audio is not mutated.
- **Silent or all-zero buckets**: quality degradation is reported as undefined (null) rather than a misleading extreme, and presence confidence is low with low uncertainty.
- **Non-English transcripts**: the phoneme-based utterance cross-check remains gated off (as today); the new token-level and scene-coupling signals still apply where language-independent.
- **Scene classifier emits a class outside the category map**: it is assigned to a documented default category and the omission is logged so the map can be extended.
- **Overlapping speech (target plus background talker)**: presence remains high; the source axis reflects both `speech` and `people` mass rather than forcing a single winner.

## Requirements *(mandatory)*

### Functional Requirements

#### Scene quality (US1)

- **FR-001**: The presence axis MUST emit, per bucket, four audio-quality degradation scores in `[0, 1]` (0 = clean): signal-to-noise ratio, clipping, reverberation, and effective bandwidth.
- **FR-002**: The SNR and clipping degradation scores MUST be derived from senselab's existing audio-quality routines rather than newly implemented signal processing.
- **FR-003**: The reverberation degradation score MUST be derived from the added scene/quality model's room-acoustics (C50) output.
- **FR-004**: The effective-bandwidth degradation score MAY be a new minimal estimator; it MUST be documented and MUST distinguish full-band from band-limited (e.g. telephone-band) signals.
- **FR-005**: The presence axis MUST emit a per-bucket `quality_uncertainty` score reflecting disagreement among the independent quality estimators for that bucket.
- **FR-006**: All new quality columns MUST be additive to the existing `presence` parquet; the existing `aggregated_uncertainty` column and its meaning MUST be preserved.

#### Sound sources (US2)

- **FR-007**: The presence axis MUST emit, per bucket, the relative mass assigned to each of `speech`, `people`, `machine`, and `environment`, summing to approximately 1, plus a `src_dominant` label.
- **FR-008**: Sound-source categories MUST be derived from the scene classifiers (AST and YAMNet) the workflow already computes, via a checked-in, versioned map from the classifier label ontology to the four categories.
- **FR-009**: The ontology-to-category map MUST cover every class the classifiers can emit (complete, non-overlapping), with a documented default for unmapped classes and logging when the default is used.
- **FR-010**: The source categorization MUST be structured so an alternative dedicated sound-source model can be added later as a backend without changing the parquet schema.

#### Temporal resolution & frame posteriors (US3)

- **FR-011**: The workflow MUST support per-axis reporting grids (window/hop), with presence defaulting to a 0.1 s window / 0.02 s hop, utterance to a 1.0 s window / 0.5 s hop, and identity/others to 0.5 s.
- **FR-012**: The presence axis MUST derive speech presence from continuous frame-level posteriors (from the existing pyannote segmentation model's raw scores and the added scene/quality model's voice-activity output) aggregated within each reporting bucket, without routing through the segment-thresholding VAD pipeline.
- **FR-013**: The presence axis MUST report a per-bucket `presence_confidence` (calibrated mean speech probability) separately from a per-bucket `presence_uncertainty` (cross-voter disagreement plus within-bucket temporal instability).
- **FR-014**: On grids finer than the coarse voters' native resolution, coarse voters (whole-window scene tags, per-segment no-speech probability, sentence-level transcripts) MUST NOT be summed as equal per-bucket voters; they MUST contribute only as a slowly-varying context prior.
- **FR-015**: Grid parameters actually used MUST be recorded in each axis's output provenance.

#### Utterance estimator (US4)

- **FR-016**: The utterance axis MUST use an overlapping word-scale reporting grid and MUST exclude words that straddle a bucket boundary from that bucket's disagreement computation.
- **FR-017**: The utterance axis MUST include a token-level uncertainty sub-signal where the ASR backend exposes per-token confidence/entropy, degrading gracefully to the existing sub-signals when it does not.
- **FR-018**: ASR confidence signals from different backends MUST be mapped to a common calibrated `[0, 1]` scale.
- **FR-019**: Utterance uncertainty MUST increase where scene quality is poor or a competing non-speech source is present, and the coupling factor MUST be recorded in the output rather than applied invisibly.

#### Calibration (US5)

- **FR-020**: The system MUST provide a helper that synthesizes calibration data by mixing clean speech with known noise at controlled SNRs and convolving known room impulse responses at target reverberation times.
- **FR-021**: The helper MUST fit the degradation-score normalization (and a temperature-scaling hook for confidences) against the synthesized ground truth and persist the fitted parameters for reuse.
- **FR-022**: The helper MUST emit a validation artifact comparing reported vs. true SNR/reverberation.

#### Cross-cutting

- **FR-023**: When any model or estimator is unavailable, the affected columns MUST be emitted as null/NaN with a recorded reason and the rest of the axis MUST still be produced.
- **FR-024**: The Label Studio bundle, timeline plot, and disagreements index MUST surface the new presence sub-signals (quality and source) without breaking the existing tracks/rows.
- **FR-025**: The added scene/quality model MUST reuse the existing HuggingFace-token access flow and MUST NOT introduce a new isolated subprocess environment.

### Key Entities

- **Presence bucket record**: a time interval on the presence grid carrying the existing `aggregated_uncertainty`, the new `presence_confidence`/`presence_uncertainty`, the four `quality_*` degradation scores plus `quality_uncertainty`, and the four `src_*` masses plus `src_dominant`.
- **Sound-source category map**: a versioned mapping from the scene-classifier label ontology to `{speech, people, machine, environment}`, with a documented default.
- **Per-axis grid configuration**: window and hop per axis, recorded in provenance.
- **Calibration profile**: fitted normalization/temperature parameters derived from synthetic mixtures, persisted for reuse.
- **Utterance bucket record**: existing pairwise-WER, native-confidence, and PPG sub-signals plus the new token-level sub-signal and the recorded scene-quality coupling factor.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On a clean tutorial clip, at least 95% of buckets report all four quality degradation scores below 0.1.
- **SC-002**: On a clip with a synthetically noised region, the mean `quality_snr` degradation inside the region is at least 0.3 higher than outside it.
- **SC-003**: The sound-source category map assigns every class the scene classifiers can emit to exactly one category (100% coverage, 0 overlaps), verifiable by an automated check.
- **SC-004**: A hand-marked brief onset (≤50 ms) is localized by the presence signal to within one reporting hop (≤20 ms at the default presence grid).
- **SC-005**: Boundary-straddling words contribute 0 to adjacent buckets' utterance disagreement, verifiable on a constructed fixture.
- **SC-006**: For matched transcript agreement, utterance uncertainty in a poor-quality region is measurably higher than in a clean region on the same clip.
- **SC-007**: On held-out synthetic mixtures, reported SNR/reverberation degradation increases monotonically with true degradation (rank correlation ≥ 0.9).
- **SC-008**: Existing downstream consumers of the `presence` parquet, Label Studio bundle, timeline plot, and disagreements index continue to work unchanged (existing regression tests pass and `aggregated_uncertainty` values are unchanged when the new features are disabled).

## Assumptions

- Users run the workflow via `scripts/analyze_audio.py` or the importable `compute_uncertainty_axes` API, on Bridge2AI-Voice-style recordings (speech plus cough/breathing, variable background).
- The added scene/quality model is `pyannote/brouhaha`, loaded through the existing pyannote-audio + HuggingFace-token path (gated, no new pip dependency, no subprocess venv).
- AST and YAMNet are already available in a default run and provide the AudioSet-style class scores that feed the source categories; when both are absent, source columns are null.
- The four sound-source categories (`speech`, `people`, `machine`, `environment`) are sufficient for v1; finer taxonomies and a dedicated health-acoustics model (e.g. HeAR) are out of scope but the interface leaves room for them.
- No labeled per-bucket SNR/reverberation dataset is available, so calibration is bootstrapped from synthetic clean+noise+RIR mixtures; a real labeled set can replace it later via the same fitting hook.
- The default effective-bandwidth estimator uses a spectral-rolloff-style measure; reuse of existing Praat spectral moments is an acceptable alternative documented in planning.
- Backward compatibility takes precedence: the `presence` axis keeps its name and `aggregated_uncertainty` output; all new signals are additive columns.

## Out of Scope

- Adding a dedicated sound-source or health-acoustics model (PANNs/BEATs/HeAR) as a backend — the interface is left open but no such model ships in this feature.
- Adding TEN VAD or any new frame-VAD dependency — frame posteriors come from models already in use plus the added pyannote model.
- Renaming the presence axis or restructuring the parquet paths / Label Studio track names.
- A user-facing labeling UI beyond the existing Label Studio bundle.
- Calibration against a real labeled clinical dataset (only synthetic bootstrapping is in scope).
