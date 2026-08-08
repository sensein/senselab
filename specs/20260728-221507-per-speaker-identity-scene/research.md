# Phase 0 Research: Per-Speaker Identity and Background Scene Characterization

**Date**: 2026-07-29 | **Feature**: `20260728-221507-per-speaker-identity-scene`

Research was conducted before planning, in four parallel investigations: human detection
psychophysics; loudness standards, noise-floor estimators and bioacoustics prior art;
AudioSet classifier level/SNR behavior; and a code-level audit of the installed models.
Load-bearing claims were independently spot-checked against the installed source and
checkpoint configs before being adopted here.

Evidence tiers used below: **[CODE]** verified in installed source or checkpoint config
in this repo; **[MEAS]** measured empirically against the real checkpoints;
**[STD]** published standard; **[PUB]** published paper or tool default;
**[SYN]** synthesis or derivation, needs validation.

---

## D1. Detection mechanism: per-band floor subtraction, not amplification

**Decision.** Detect background sources by estimating a per-frequency-band noise floor,
subtracting it, and applying a single margin threshold. Amplification is scoped to
conditioning the classifier's input and capped at ~10 dB. (FR-018b, FR-019)

**Rationale.** Amplification changes no signal-to-noise ratio — it moves the background
and the residual foreground together. **[MEAS]** An oracle test with 30 dB of foreground
suppression, residual amplified to −20 dBFS, produced *identical* output whether a faint
background source was present at −80 dBFS or entirely absent (YAMNet `Speech` 0.99 vs
1.00). Attenuate-then-reamplify is bit-exact in float32 (score L1 = 1e-6 after −100 dB
then +100 dB), so gain never *recovers* information — it only prevents the classifier's
own floor from destroying it.

By contrast **[PUB]** four independent detection toolchains (QUT ecoacoustics, Raven,
PAMGuard, Kaleidoscope) all use a *local* reference rather than a global gain. Towsey
states the design goal directly: after per-bin floor subtraction, "the power in every
frequency bin fluctuates around 0 dB but during an acoustic event it is considerably
higher. Thus it becomes possible to define a single absolute threshold for the detection
of an acoustic event that spans multiple frequency bins." This is what makes the
different-distances problem tractable: the reference becomes local, so a near and a far
source are each judged against their own band floor.

**Alternatives considered.**

- *Fixed dB boost* — meaningless across recordings with different absolute levels.
- *Single global normalization target* — rejected by the user on distance grounds, and
  independently by the measurement above.
- *Per-window adaptive gain (AGC)* — would mimic what the short-window classifier was
  suspected of doing internally, confounding the very question US2 measures.
  **[PUB]** Also a documented spurious-onset generator: DCASE 2019 attributes a 50×
  near-field false-alarm win to ship-engine modulation aliasing against a fast tracker.
- *Multi-gain test-time augmentation* — no published precedent for reconciling frozen
  classifier posteriors across gains; retained only as an optional comparison arm.

---

## D2. Both classifiers are amplitude-sensitive; neither self-normalizes

**Decision.** Treat both AST and YAMNet as level-sensitive. Do not rely on any internal
normalization. Pin the finding with a regression guard. (FR-014, FR-017b)

**Rationale — AST [CODE].** Independently verified in this repo:

```python
# .venv/.../feature_extraction_audio_spectrogram_transformer.py
do_normalize=True, mean=-4.2677393, std=4.5689974      # lines 75-77
def normalize(self, input_values): return (input_values - (self.mean)) / (self.std * 2)  # 156
# waveform = waveform * (2**15)  # Kaldi compliance ...   # 113 — COMMENTED OUT
```

Same values confirmed in the checkpoint's `preprocessor_config.json`. These are **fixed
AudioSet constants, not per-example statistics**, so they remove the *training corpus'*
average level, not this clip's — conferring zero amplitude invariance. A gain becomes a
rigid log-domain shift of every input bin. **[MEAS]** At 0.1× the measured shift was
−0.503958 against a predicted −0.503959, standard deviation 0.000000 across bins; ≈1.1σ
of the input tensor per 20 dB.

**Rationale — YAMNet [CODE].** `log(mel + 0.001)` on a *magnitude* mel, offset read
directly from the TF-Hub graph proto (`const = 0.0010000000474974513`), with no
normalization op anywhere in the executed path and no `Square` op (hence magnitude, not
power). senselab adds nothing: decode, downmix, and resample are all level-preserving.

**The expected asymmetry did not hold.** The working hypothesis was that the ~10 s window
model would not self-amplify while the ~1 s model might. Neither does. Worse, the model
*with* an explicit normalization step (AST) is the one whose reported label set churned at
every tested gain — because its constants are global.

**A mechanism correction.** YAMNet's collapse at −60 dBFS is **not** the log offset
flooring the spectrum: **[MEAS]** at the collapse point only ~4.6% of bins are floored
(the offset does not bite until ~−80 dBFS, 90% floored). It is a **learned,
absolute-level-keyed `Silence` decision** firing ~30 dB above the arithmetic floor, and it
is source-independent — tones, beeps, band noise, and speech all die at −55…−65 dBFS.
Because it is monotone and source-independent, it is the best low-level diagnostic either
model exposes and is retained as a level tripwire (FR-042).

**Neither model is simply "more brittle."** **[MEAS]** AST holds `Speech` ≥0.65 down to
−83 dBFS where YAMNet has returned `Silence = 1.000` for 10 dB; but for non-speech sources
AST's floor is close to YAMNet's and its scores are non-monotonic (a beep's score *rose*
from 0.19 to 0.44 as level dropped 70 dB). Any robustness claim must specify content type
and level.

**Derived operating limits.**

| Limit | Value | Fixable by gain? |
|---|---|---|
| In-window absolute level | **−45 dBFS RMS** | Yes |
| Source-vs-interferer SNR | **~10 dB** (YAMNet 5–10, AST 15–20 non-speech) | **No** |
| Foreground suppression depth | must exceed background-to-foreground ratio | **No** |
| Stable target band | **−35 to −15 dBFS RMS** | — |
| Gain ceiling | **+10 dB** (clipping inflection **[PUB]**) | — |

---

## D3. Detection margin: a 3 / 6 / 10 dB ladder above the per-band floor

**Decision.** Candidate at +3 dB, probable at +6 dB, confident at +10 dB, per third-octave
band against a same-recording locally-estimated floor. Reject below +3 dB. (FR-021)

**Rationale — three independent corroborations.** This is the strongest available evidence
that the values are not arbitrary:

1. **Human psychophysics** — masked-threshold and audibility criteria place minimum
   measurability near +3 dB and confident identification near +10 dB. **[STD]** ISO 1996-2
   requires a source to exceed background by ≥3 dB to be measurable at all; **[STD]**
   ISO 7731 uses 10 dB (octave) / 13 dB (third-octave) above masked threshold;
   **[STD]** BS 4142 treats +5 dB as adverse and +10 dB as significant impact;
   **[STD]** ECMA-74 prominence uses TNR ≥ 8 dB / PR ≥ 9 dB.
2. **Established detection practice** — a dozen traditions converge on 3, 6, or 8–10 dB
   above a per-band floor: **[PUB]** Raven BLED `SNR Threshold: 10`, PAMGuard click
   `dbThreshold = 10` and whistle `thresholdDB = 8`, R `bioacoustics SNR_thr = 10`
   ("8 dB recommended for bird vocalizations"), QUT `DecibelThreshold: 6.0`,
   scikit-maad `spectral_activity` 6 and `temporal_activity` 3, IMCRA's internal
   `γ₀ = 6.6 dB` / `ζ₀ = 2.2 dB`, Martin's `noise_slope_max` 0.8–9 dB.
3. **Measured classifier capability** — **[MEAS]** YAMNet's reliable-detection floor
   against an interferer is 5–10 dB SNR, AST's 15–20 dB for non-speech, with noise-family
   labels contaminating the top-3 from ~20 dB SNR downward.

The +10 dB confident tier is therefore defensible from the human *and* machine sides at
once, which is the balance the spec required (FR-022, SC-017).

**Alternatives considered.**

- *Loudness-model criteria (ISO 532 partial loudness)* — theoretically the best-motivated
  option; **[PUB]** partial loudness predicts masked threshold as the point where the
  signal's partial loudness equals 0.003 sones. **Rejected**: requires absolute dB SPL,
  and because the models are nonlinear an unknown dBFS→SPL offset does not merely rescale
  the answer, it changes which bands fall below threshold. Separately, the model needs
  signal and background supplied *pre-separated* — which our foreground suppression would
  actually provide, making this a genuine upgrade path *if* calibrated recordings ever
  enter scope.
- *Absolute hearing-threshold criteria* — inapplicable to uncalibrated recordings.
- *A single broadband threshold* — rejected unanimously by the room-acoustics and
  ecoacoustics literature; environmental and microphone noise are LF-weighted, so a
  broadband floor is set by the low bands and leaves mid/high-band events ungated.

---

## D4. Noise-floor estimator: two-pass, per-band, bias-corrected, event-excluding

**Decision.** Offline two-pass robust per-band percentile with event exclusion, iterated
to stability, conditioned on target activity, with the floor frozen inside detected
events. (FR-021a, FR-021d–i)

**Rationale.**

- **Percentile, not minimum or mean.** **[STD]** BS 4142 defines background as `L_A90`
  — the 10th percentile — chosen because it is "relatively insensitive to occasional loud
  events," and it tolerates up to 90% event occupancy per band. **[PUB]** Raven's shipped
  preset uses the 20th percentile. Mean-of-log (as SoX `noiseprof` does) absorbs events by
  construction. Raw minimum tracking carries up to **19.8 dB** bias.
- **Bias correction is mandatory.** **[SYN, derived]** For noise-only bins the periodogram
  is exponential, so a *q*-quantile sits `1/(−ln(1−q))` below the true mean noise power:
  **+9.77 dB for q=0.10**, +6.51 dB for q=0.20. Uncorrected, every relative-dB gate is
  silently ~10 dB permissive. **This repo already has the uncorrected version** at
  `quality_control/metrics.py:140` and `:443`.
- **Patch aggregation, not frames.** **[SYN, derived]** A single time-frequency bin's
  log-power has σ = 5.57 dB, so 3σ ≈ 17 dB and a 3 dB frame threshold is meaningless.
  Aggregated over a ~0.96 s patch, σ ≈ 0.2 dB, making 3 dB ≈ 15σ.
- **Event exclusion, iterated.** **[PUB]** IMCRA's two-iteration structure — exclude bins
  exceeding the bias-compensated minimum, then re-estimate on survivors — generalizes
  trivially to non-causal offline windows.
- **Freeze the floor inside events.** **[PUB]** PAMGuard slows its noise tracker 10×
  during a detection (`longFilter2 = longFilter/10`) precisely so an event cannot pull up
  its own floor.
- **Condition on target activity.** **[SYN, unpublished]** Every published estimator
  assumes the floor is independent of the events. Our suppression residual violates this:
  artifact level correlates with the removed talker's level. Estimating separate floors for
  target-active and target-quiet frames, using the diarization already available, is the
  mitigation. **This has no published precedent and must be validated.**

**Alternatives considered and rejected.**

| Estimator | Why rejected |
|---|---|
| Minimum Statistics (Martin) | 1.54 s window + 1.73 s asymmetric latency is pure cost offline; **[PUB]** documented −28% underestimate on non-stationary floors; explicitly "tends to consider small speech-like noise fluctuations as speech" — i.e. leaves suppression artifacts above the floor |
| MMSE / SPP (Gerkmann) | **[PUB]** its anti-stagnation rule *forces* the estimate upward after sustained high speech-presence probability, reclassifying a long vehicle pass-by or air conditioner as floor — the exact inverse of the goal. Retained only as a cross-check for genuine floor *steps* |
| Doblinger spectral-minimum | **[PUB]** Martin's Table II measures +59…+77% error under continuous speech — events absorbed wholesale |
| Histogram mode (Hirsch / Towsey) | Viable and **[PUB]** well-documented (Towsey's bin-95 cap is a good guard), but the percentile route composes more cleanly with the bias correction and event exclusion. Otsu's method on the per-band histogram noted as an upgrade |

---

## D5. Level target and headroom

**Decision.** Normalize the raw pass to about **−23 LUFS** integrated (BS.1770 gated),
apply the same scalar to the suppressed variant, enforce true peak ≤ −1 dBTP, and keep gain
≤ 10 dB. (FR-019, FR-019c)

**Rationale.** **[STD]** EBU R128's −23 LUFS leaves ~23 dB of headroom below full scale,
enough to apply up to +10 dB of classifier-input gain without clipping; a streaming-style
−14 LUFS target leaves only 14 dB. Applying the *same* scalar to both variants preserves
the raw-vs-suppressed delta the uncertainty workflow depends on — independent
renormalization would corrupt it. **[STD]** BS.1770's relative gate (−10 LU) and
**[STD]** EBU Tech 3342's LRA gate (−20 LU, P95−P10) are the closest standards prior art
to "separate the weakest real signal from the noise floor," and Tech 3342 states the goal
almost verbatim: "The lower edge of Loudness Range should not be defined by the noise
floor (which may be inaudible), but should instead correspond to the weakest 'real'
signal."

**Weighting.** **[STD]** A-weighting (inverted 40-phon contour) is the right family for
"would a human notice this," and K-weighting is the wrong tool — it *boosts* HF by 4 dB,
was fitted to programme material at 60 dBA, and BS.1770 itself disclaims applicability to
tones. A-weighted broadband excess is emitted as a human-readable summary only, never as
the gate.

---

## D6. Amplification is bounded by suppression depth, not by gain

**Decision.** Measure and report achieved foreground suppression depth; refuse
detectability claims when residual foreground exceeds the candidate. (FR-018a, SC-015,
SC-016)

**Rationale.** See D1's oracle test. **[MEAS]** 30 dB of suppression left the residual
foreground dominant, and the amplified residual reported leaked speech *more* confidently
(0.90 → 0.99) rather than revealing the background. This reorders the work: suppression
quality is the primary variable and gain is secondary. It is also the strongest argument
for the background mask (D7) — in target-free regions there is no foreground to suppress,
so the constraint vanishes.

**Known risk.** **[PUB]** Aggressive spectral subtraction generates *musical noise*:
"randomly appearing and disappearing tonal components… spurious peaks at random locations
in the time-frequency plane." That is a synthetic event generator feeding the classifier.
Prefer a higher residual noise floor over deeper subtraction.

---

## D7. Background mask derived from existing signals plus task metadata

**Decision.** Derive the mask from the existing presence axis, diarization, and
voice-activity outputs, with target-activity detection selected by **task metadata**, and a
conservative fallback when metadata is absent. (FR-031, FR-033, FR-033a/b)

**Rationale.** The presence axis already distinguishes "confidently no speaker" from
"cannot tell," which is exactly the three-state distinction the mask needs, so the mask is a
derived output rather than a new model — hence its low cost. The exception is decisive:
**for a task whose target is a non-speech vocal event (breathing, cough), speech detection
reports no activity during the target event.** Building the mask from speech activity alone
would admit the target signal into the background mask and then report it as a background
human-sound source — misattributing the very signal being collected. SC-024 makes this
testable: zero target events reported as background sources on a breathing- or cough-task
recording.

**Alternatives considered.** Masking on *any* speech (excluding distant talkers) was
rejected by the user in favour of target-only masking, so a background talker remains in
the mask and is reportable as a source — consistent with the off-target-speaker detection
goal. Emitting both masks was rejected as doubling the surface for a distinction the
pipeline may not reliably make.

---

## D8. Classifier routing for mask regions

**Decision.** Run the long-window classifier on **excised** mask segments; keep the
short-window classifier on the regular grid and retain it as the level tripwire.
(FR-041–045)

**Rationale.** **[MEAS]** Window length matters for *isolation*, not adaptation. With a
loud-then-quiet test signal, YAMNet's 0.96 s windows gave identical quiet-half results
under global-gain-with-clipping and segment-wise gain — the clipped speech lives in
different windows and cannot contaminate the background ones. AST's single 10.24 s window
couples them (0.344 vs 0.548). Excising the quiet segment and running AST on it alone beat
every mixed-window variant: **0.705**.

**Accepted cost.** Excision means a variable number of short segments rather than a fixed
grid, and mask regions shorter than ~10 s are heavily zero-padded — where **[MEAS]** padding
maps to a fixed normalized value while the signal region drifts with gain, making the
pad/signal contrast itself gain-dependent. The trailing short window was AST's least stable
case. Hence FR-043's padding-fraction reporting and FR-045's short-region flag.

---

## D9. Score comparability defect (incidental finding, adopted into scope)

**Decision.** Do not sum mutually-exclusive and independent per-class scores into the same
category masses. (FR-017c)

**Rationale [CODE].** `huggingface.py:131` defaults `function_to_apply="softmax"`, applied
across all 527 AudioSet classes — but AST-AudioSet is a *multi-label* model. Softmax forces
the classes to compete, so a dominant `Speech` score crushes every background class.
YAMNet's graph ends in `Sigmoid` — independent per class. `_window_category_masses` in
`sound_sources.py` then sums both into the same four category masses. This structurally
suppresses exactly the secondary background sources this feature exists to surface, so it
is in scope rather than deferred.

**[MEAS]** Related reliability context: with 5 concurrent sources, AST precision is 69.1%
and YAMNet 61.7%, with YAMNet recall 36.7%. **[PUB]** Google states plainly that YAMNet's
"classifier outputs have not been calibrated across classes, so you cannot directly treat
the outputs as probabilities."

---

## D10. Pipeline artifact: amplify before serialization

**Decision.** Apply gain before any lossy serialization in the classifier input path.
(FR-019b, FR-017d)

**Rationale [CODE + MEAS].** `yamnet.py:117` round-trips the waveform through
`save_to_file`, which writes **PCM_16**. **[MEAS]** That injects a −101 dBFS quantization
floor, annihilates content below ~−100 dBFS (at −120 dBFS the readback is exact zeros →
`Silence = 1.000`), and hard-clips 30.9% of samples at 10× gain. Critically, **[MEAS]**
16-bit quantization noise is statistically indistinguishable from analog white noise
(spectral flatness 0.541 vs 0.563), so amplifying *after* the write amplifies broadband
noise and produces exactly the water-like false categories of D11.

---

## D11. Guarding against fabricated categories

**Decision.** Layer a noise-character test, a pre-gain level reject, a floor-response
signature detector, and a quarantine list for noise-family labels. (FR-020a–e)

**Rationale [MEAS].** Amplifying a −100 dBFS noise floor to −20 dBFS produced
`Waterfall 0.372, Water 0.338, White noise 0.244, Gurgling 0.094` on YAMNet — statistically
indistinguishable from genuine white noise at the same level, and precisely the kind of
label a background characterizer would accept as a real environmental finding. On digital
silence, AST saturates to a fixed `Silence 0.437, Music 0.350` — note that **thresholding
on `Silence` will not catch this** while `Music` at 0.35 clears most practical thresholds.

**The cheap discriminator.** **[MEAS]** Spectral flatness separates broadband noise floors
(0.54–0.56) from every structured source tested (≤0.004) for the cost of one FFT. Use it as
a *relative* test against the band's own long-term flatness — **[PUB]** MPEG-7 standardizes
the descriptor but not any decision threshold, and no ITU-T/ETSI numeric threshold was
verifiable.

**Complementary features, ranked by orthogonality to level.** **[PUB]** Amplitude-modulation
depth is first — stationary noise has near-zero modulation depth at all rates, and AMS
features are worth roughly a 10 dB SNR benefit over MFCCs in 84% of tested conditions. Caveat:
speech suppression operates on the 3–6 Hz modulation band, so the residual may carry
*inherited* talker modulation; down-weight that band. Then multi-frame spectral flux with a
max-filter (**[PUB]** SuperFlux: up to 60% false-positive reduction). Harmonicity is
high-precision/low-recall — use to confirm, never to reject, since rain, wind, HVAC and
crowd murmur are inherently inharmonic.

---

## D12. The SNR gate is novel; adopt posterior-domain machinery alongside it

**Decision.** Layer the margin gate *on top of* standard sound-event-detection
post-processing rather than in place of it. (FR-021j, FR-021k)

**Rationale.** **[PUB]** No DCASE Task 4 system gates detections on estimated SNR or a
noise floor; the community's entire false-positive toolkit is posterior-domain — class-wise
median filtering, minimum durations, hysteresis, class-wise thresholds. Our gate is
therefore not standard practice and should be additive. Two published elements are adopted
directly: class-wise median filter length ≈ `0.55 × avg_class_duration / frame_duration`
(the DESED-derived lengths span 3–50 frames, a 17× spread, so one length is wrong), and
**[PUB]** Sound Event Bounding Boxes, whose core argument maps onto our problem exactly —
frame-level thresholding "entangles detection confidence with temporal extent," and no single
threshold gets both presence and extent right. Hence FR-021k separates the two decisions.

---

## D13. Mutual influence: guards before influence paths

**Decision.** Implement uncertainty-gated influence weighting, revision provenance, and
oscillation detection *before* enabling any influence path. (FR-011a–h, Phase D before E)

**Rationale.** The user chose full loop participation. The dominant risk is
**self-confirmation**: if identity repair revises speaker labels and identity uncertainty is
then recomputed from those revised labels, uncertainty falls *because* it was revised, not
because evidence improved. The existing loop already distinguishes *explained* from
*improved* outcomes (observed during T039 validation, where 4 of 5 successes were
explained rather than improved), so FR-011d generalizes an existing distinction rather than
inventing one.

Two further guards are structural rather than optional. **Derived-signal
down-weighting** (FR-011c): a synthetic voter derived from embeddings that also feed identity
uncertainty would otherwise have its influence double-counted, and its agreement with its
parent is not corroboration. **Determinism** (FR-011f): mutual influence with unordered
evaluation would break the byte-reproducibility the convergence outputs already provide.

**Alternatives considered.** Report-only and trigger-only were both offered to the user and
declined. Note that report-only would have allowed parallel development with #537; the
chosen design does not.

---

## D14. Dependency decisions

| Decision | Rationale | Alternative rejected |
|---|---|---|
| **Promote `librosa` to an explicit dependency** | `pcen` and `A_weighting` are used directly. **[CODE]** librosa 0.11.0 is present but **not declared in `pyproject.toml`** — it is transitive today, so an upstream change would break this feature silently | Continuing to rely on it transitively is the defect, not a simplification |
| **Add `pyloudnorm`** | BS.1770 correctness (K-weighting biquads, two-stage gating, 4× oversampled true peak) is easy to get subtly wrong; numpy/scipy-only, validated to ±0.1 LU | Hand-implementing the standard — ~150 lines of standards-compliance code we would then own and have to validate |
| **No new subprocess venv** | Everything needed is available in the main environment | A separate venv was pre-authorized "only if needed"; it is not needed |
| **Do not change `quality_control/metrics.py`** | Its percentile floor is a published QC metric with its own semantics; changing it would change QC outputs for unrelated consumers | Fixing it in place — rejected under Simplicity First; instead its docstring documents the bias and the corrected estimator lives in `noise_floor.py` |

---

## Open risks carried into implementation

1. **FR-021h has no published precedent.** Conditioning the floor on target activity is
   synthesis. Validate before relying on it.
2. **Derived §D4 statistics are unvalidated.** The −2.51 dB log bias, 5.57 dB per-bin σ, the
   `1/(−ln(1−q))` correction, and the patch-variance collapse are straightforward χ²₂
   results but were not found stated in this form in the noise-estimation literature.
   Validate on synthetic noise before letting them set a threshold.
3. **Some supporting psychophysics figures are provisional** — the partial-loudness
   transition interval and current-generation microphone self-noise values could not be
   verified against primary sources. FR-022 requires the derivation to mark which figures
   are verified and which are not.
4. **An alternative classifier may be better positioned.** **[CODE]** BEATs applies the
   `waveform * 2**15` pre-scale that AST has commented out, giving ~90 dB more headroom
   above the same log floor. But headroom arithmetic already mispredicted the AST-vs-YAMNet
   ordering — the binding constraint turned out to be a learned decision boundary, not the
   arithmetic floor — so this must be measured, not assumed.
5. **For consumer-grade capture the microphone, not human hearing, is often the binding
   floor.** Phone-class self-noise sits at or above a quiet bedroom, so content near human
   threshold was never captured. In the lab-like close-mic scope with a quiet room, this is
   the *likely* binding case, and FR-021b requires the system to say so rather than imply a
   perceptual justification it cannot support.
6. **#537 collision is unavoidable for US1.** It edits `identity.py`, `clustering.py`,
   `stages.py`, `stage_context.py` and adds four diarizers — more voters, which makes the
   count-disagreement and derived-voter-weighting problems richer rather than simpler.
