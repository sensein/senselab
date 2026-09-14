# VOICE branch

What the branch answers: **is there sustained phonation here, what are its acoustic properties, and
what did the voice do across it?**

The frame is [`../20260913-branch-contract-and-hints/design.md`](../20260913-branch-contract-and-hints/design.md);
shared rules are in [`branch-conventions.md`](branch-conventions.md); owed ground truth is in
[`branch-listening-sample.md`](branch-listening-sample.md).

## The state of this branch

**VOICE fails on every recording.** Its subject is every live span whose `family` is `phonation`
(`voice.py:232`, `_PHONATION_FAMILY` at `:39`). Nothing reachable proposes one: the detector that
did was retired on 2026-09-04, and the module's only `prov_type="span"` write (`voice.py:335-348`)
sits downstream of the no-span return at `:237-266`, which every recording takes.

**And VOICE is the worked example of what the contract forbids.** `voice.py:335-348` mints a
*second* span from an input span, re-keyed by period-aligned onset and carrying `onset_kind`. That is
the re-minting the contract replaces with a `refine` assertion, and it is why `report.py:291-308`
needs `_spans_of_family` to split one family into two populations on `("onset_kind" in attributes)`.
Once the minting becomes a `refine`, the split has nothing to separate.

**Consequence for REPORT.** `onset_kind` is written only at `voice.py:343`, so on stores written
under the contract the `voice=True` reads at `report.py:715` and `:1154` return empty. REPORT's VOICE
arms need the span-versus-assertion distinction substituted in, not the family merely widened.

## The tasks this branch serves

Declared families, not ground truth. `VOICE_ELICITING` (`families.py`) holds six families; five
carry counts in the corpus profile:

| family | n | what the task asks for |
| --- | --- | --- |
| `maximum-phonation-time` | 2,696 | sustain a vowel as long as possible on one breath |
| `prolonged-vowel` | 1,604 | sustain a vowel at comfortable pitch and loudness |
| `glides-low-to-high` | 1,596 | glide F0 upward across the range |
| `glides-high-to-low` | 1,554 | glide F0 downward across the range |
| `maximum-phonation-time-v2` | 813 | as above |

These five sum to 8,263 against 8,306 declared VOICE; the residual 43 is `high-to-low`, the sixth
member of `VOICE_ELICITING`.

**`loudness` and `loudness-v2` are not VOICE-declared.** Both are in `LEXICAL_SPEECH`
(`families.py:41-42`), because the task asks the participant to say words. An earlier version of this
document listed them here, which over-counted the table by 1,602. The *measurement* they want is
still V6's — the loudness-condition correlates — which is the ordinary case of a branch running on content rather
than declaration, and it parallels CAPE-V, whose voice quality belongs here while its sentence
conformance belongs to SPEECH.

**Three clinical measurement shapes.** Maximum phonation time is a duration. Glides are an F0
trajectory — range, monotonicity, direction. Prolonged vowel is voice quality: perturbation and
noise over a steady segment.

### A recording routed here whose declared task is not voice

VOICE routed 22,277 recordings against 8,306 declaring a voice family. Sustained phonation occurs
inside sentence reading, free speech and DDK. The branch marks it wherever it finds it and reports
what it measured, asserting nothing about the declared task.

## Capabilities

### V1 — Propose the phonation attempt (**not built; the branch's foundation**)

**Question.** Where did the speaker attempt to phonate?

**The attempt is proposed from the envelope and *qualified* by voicing — not defined by it.** An
earlier version of this document defined a phonation span as "a contiguous region of voiced frames",
which is the same tracker-dependence V2 rejects, one level upstream. A Titze type-3 voice — frankly
aperiodic — yields no voiced run at all, so it would produce **no V1 span, hence no V2 measurement,
hence `VOICE: FAIL`, on a recording containing eighteen seconds of phonation.** Fixing V2 alone
moved that bias rather than removing it.

**Reads.** The energy envelope PREPROCESS computes, plus its `phonation_tracks` measurement — written
by `phonation_tracks` (`preprocess.py:919`, called at `:2147`), which runs `f0_track` over the
pre-emphasised stream for per-frame F0 and voicing strength and `formant_track` over `plain` for four
formants — and the HNR track via `hnr_track` (`tasks/phonation/__init__.py:3-12`).

**Computes.** The attempt's extent from sustained energy above the background. Then, as *properties
of* that extent rather than conditions on it:

- **voiced fraction** — how much of the attempt the tracker found F0 for;
- **interruption structure** — the number, location and duration of unvoiced intervals within it;
- **F0 availability** — whether the tracker produced a usable contour at all;
- **F0 stationarity, formant stationarity and spectral flux** — whether the production held still.

A type-3 voice yields an attempt with a low voiced fraction and no usable F0 contour. That is a
*finding about the voice*, and exactly what a reader needs. Defined the other way it was an absence
of data.

**The stationarity qualifiers are not optional, because the first three do not separate a sustained
vowel from connected speech.** Connected speech has a high voiced fraction, a usable F0 contour and
few interruptions — it passes all three. VOICE routes 22,277 recordings against 8,306 declaring a
voice family, so **connected speech reaching this branch is the common case, not the edge case**.
Without a steadiness qualifier V1 would propose "phonation attempts" over runs of connected speech,
and V4 would compute perturbation and CPPS over segments containing consonants, pauses and changing
vowels — the classic invalid-perturbation error, which none of V4's type-2 or type-3 qualifiers
detects.

`formant_track` is already in `phonation_tracks`, so formant stationarity costs nothing new.

**Steadiness is a covariate, not a gate.** V4 carries it on every perturbation and CPPS value rather
than V1 refusing to propose — the same discipline as the rest of the inversion.

**But the covariate and the value it qualifies are currently computed on different signals**, which
undermines the pairing. `phonation_tracks` runs `f0_track` on **pre-emphasised**, `formant_track` on
**plain**, and derives the range on **plain** (`preprocess.py:941`) — while every V4 value comes from
**enhanced** (finding 0). A steadiness qualifier measured on one signal cannot certify a perturbation
value measured on another.

**And F0 is tracked on a pre-emphasised signal**, which [`branch-ddk.md`](branch-ddk.md) D1 argues
carefully is wrong for the envelope modulation spectrum without the analogous argument ever being
made for autocorrelation pitch tracking. +6 dB/octave attenuates the fundamental relative to the
upper harmonics, **raising octave-error risk upward** — worst on low-F0 and creaky voices — and
Praat's guidance is to track pitch on the unmodified signal. The range is derived on `plain` and
applied to `preemphasised`, which is structurally the same mismatch the audit condemns for the
jitter form defaults, here in the supposedly clean path.

**And `voice_tracks.npz` carries unmasked sentinels.** `voice.py:305` and `:374` write `hnr_db` with
Praat's −200 dB undefined-frame markers unmasked — measured **389 sentinel frames** in a padded 2 s
signal — while `phonation.hnr_floor_interval_db` is null so nothing masks them downstream. Any mean
or percentile over that array is destroyed. [`praat-instrument-audit.md`](praat-instrument-audit.md)
finding 10, which also records the useful negative result that
`extract_harmonicity_descriptors`' own `Get mean` *excludes* them.

**An absent `phonation_tracks` is not an empty one.** `derive_f0_range` **raises**
`F0RangeUnavailable` when the wide search places no pitch (`tasks/phonation/api.py:45-66`), so on a
frankly aperiodic recording the whole measurement is *absent* rather than present-and-null. The "F0
availability" qualifier must handle an absent measurement, which is the case it most needs to
describe.

**Emits.** `propose` spans, `family: "voice"`, subject to the `propose`/`refine` rule in
[`branch-conventions.md`](branch-conventions.md), each carrying the three qualifiers above.

**Parameters.** The claim that this is "parameter-free in its core" is withdrawn — it was false under
the old definition and is still false under the new one:

1. **The envelope threshold and minimum attempt duration are already derived — reuse them.**
   `spans.k_db: 6.0` (`config-derivations.md:106-120`) and `spans.min_duration_ms: 50` (`:234`)
   are both derived, both operate on the same pre-emphasised envelope V1 reads, and both do exactly
   this job. **V1 reuses them and introduces no second pair.** An earlier version marked them owed
   without checking — the same asserting-an-absence this document corrected elsewhere. What V1 may
   still need beyond them is a *minimum sustained* duration distinguishing an attempt from an
   ordinary span, which is a different quantity and is owed;
2. Praat's own undeclared internal voicing threshold inside `to_pitch_cc`, **and the range handed to
   it**. An earlier version said the range "is not a free parameter" because `derive_f0_range`
   narrows per recording; V3 below establishes that it does not — it selects one of two hardcoded
   pairs at a 170 Hz boundary. So both the range and the voicing decision within it are undeclared,
   and the range is worse than undeclared: it is a sex-typed prior nobody wrote down;
3. whether a single unvoiced frame breaks an interruption, which now affects only the *reported
   interruption structure* rather than whether a span exists at all — a much smaller consequence than
   before, and the point of the inversion.

The same config block marks `phonation.hnr_floor_interval_db` and `phonation.rms_floor_interval`
owed (`default.yaml:146-147`, recording that *Praat calibrates no dB floor*).

### V2 — Maximum phonation time (**not built**)

**Question.** How long did the phonation last, and was it continuous?

**MPT is not the longest contiguous voiced run.** The clinical convention is onset to audible
cessation (Kent, Kent & Rosenbek 1987). Praat's tracker drops out precisely on voice breaks,
diplophonia, subharmonics and terminal creak — the signatures of disordered phonation — so a
longest-run measure halves the MPT of any voice with a mid-phonation break and truncates the end of
nearly every effort, worst in the most dysphonic voices.

**Report over the V1 attempt:**

1. the **onset-to-offset extent** of the attempt;
2. the **total voiced duration** within it;
3. the **number, location and total duration of internal interruptions**.

**The extent has its own bias, in the opposite direction.** An envelope or VAD offset does not
distinguish phonation from the voiceless sigh or egressive airflow that terminates most
maximum-effort trials, and AGC extends that tail. **The extent is therefore an upper bound**, and
should either be labelled as one or intersected with voicing evidence — which trades the
overestimate back toward the tracker bias V1 exists to avoid. Report both and name which is which.

**Require two covariates.** Mean F0 and the relative level of the phonation travel with the measure:
an MPT produced at the very bottom of the range or at near-inaudible level is not the same
measurement as one produced at comfortable pitch and loudness.

**A truncated trial is right-censored.** If the attempt runs to the recording boundary the true MPT
is unknown and greater than what was measured. **It must never be pooled with complete trials** —
doing so biases any group mean downward, and again worst where efforts are longest.

**Report the number of attempts.** Taking a maximum silently discards false starts.

**Emit no scalar named `maximum_phonation_time`.** MPT has widely circulated norms — roughly
15–25 s in healthy adults, under 10 s conventionally notable — so a scalar under that name **will be
read against them** whatever a triple elsewhere in the payload says. The verdict's `longest_span_s`
is the retired measure surviving in the payload and goes with it. Either the `counts` entry carries
the onset-to-offset extent under an explicitly qualified name, or it carries no scalar at all and
the triple is the whole output.

**No `declared` half.** "As long as possible" declares no duration.
`voice.task_duration_ranges` is null (`default.yaml:162`) and `_task_range` (`voice.py:131-163`)
returns `not_evaluated`; under the contract the check becomes a report rather than a gate.

**Serves.** `maximum-phonation-time` (2,696), `-v2` (813), `prolonged-vowel` (1,604) secondarily.

**MPT confounds respiratory and laryngeal contributions**, and the measure that separates them — the
s/z ratio — is not in this protocol.

### V3 — F0 trajectory (**detectors exist; branch consumption not built**)

**Question.** What did F0 do across the phonation?

**Computes.** Semitone range, semitone IQR, monotonicity, monotone fraction, sweep rate, rank
correlation with time, direction bias. `BRANCH_DETECTORS = {"VOICE": _PITCH_TRAJECTORY}`
(`detectors.py:1555`) holds the 22 detectors defined at `detectors.py:1477-1549` — 19 under the
`glide` kind and 3 under `voice` — each carrying an empty `thresholds` and absent from `DETECTORS`,
because they are branch measurements rather than gates.

**`derive_f0_range` does not narrow per recording. It is a binary sex-typed bin, and this is a
defect of the function against its own documentation** — [`praat-instrument-audit.md`](praat-instrument-audit.md)
finding 1, which measures the downstream step and notes that `capability-map.md:117` and `:314`
already flag this bin as the exact mistake not to repeat.

`extract_pitch_values` (`praat_parselmouth.py:421-436`) takes the recording's trimmed mean pitch and
selects one of **two hardcoded pairs**:

```
mean_pitch < 170  ->  floor 60,  ceiling 250    # commented 'male' settings
otherwise         ->  floor 100, ceiling 500    # commented 'female' and 'child' settings
```

The `derive_f0_range` wrapper's own docstring (`tasks/phonation/api.py:46`) says *"This recording's
own F0 search range, narrowed from a wide search by the standardization method"*, and
`config-derivations.md:641-648` inherits that description. **Neither matches the code.** A previous
revision of this document repeated the description as fact; it was wrong, and the error came from
reading the derivation rather than the implementation.

**What follows from the bin, all of it structural:**

- **The binding ceiling is 250 or 500 Hz, never 600.** The `voice.f0_search_range_hz` upper bound is
  not what limits anything. For a speaker binned below 170 Hz the ceiling truncates **an ordinary
  male upward glide**, not merely falsetto.
- **The floor is 60 or 100 Hz, never 50**, so downward glides are clipped into fry well above the
  stated bound.
- **Every F0-derived measure across 62,547 recordings carries a discontinuity at 170 Hz**, along a
  sex-typed boundary, in a corpus that deliberately enrols gender-diverse participants. Two speakers
  either side of that mean are analysed under different floors *and* different ceilings — a measured
  **1.67× step** in the intensity window, harmonicity window and minimum analysable segment.
- **The range is derived twice, independently** — `preprocess.py:941` and `voice.py:71` — so the two
  can land in different bins for the same recording.
- **On a glide the sweep decides its own analysis ceiling**, since the bin is chosen from the mean
  over the whole recording — and the ±2 SD trim, applied in linear Hz, preferentially trims the high
  end of an upward sweep.

**The conformance flag therefore needs three levels**, not two: the F0 extremum coincided with the
**derived limit**, with the **outer search bound**, or — the one that matters most here — **which bin
was selected**. Report the bin.

Report also a **robust percentile range** beside the extrema and a **count of octave-scale
frame-to-frame jumps** (interpreted per [`branch-conventions.md`](branch-conventions.md)).

**Report the glide span as a produced range, not a physiological one.** Participants under-produce
without coaching, the tracker clips, and octave errors are asymmetric. "Produced range" is what was
measured; "range" invites a reader to treat it as capacity.

**Add the direction deviation, keyed on the dominant monotone segment.** The sweep ran opposite to
the requested direction — a sign, an observable rather than a classification — over the 3,150 glide
recordings. **Key it on the dominant monotone segment, not on the endpoints**, and distinguish "ran
opposite" from "ran both ways": many participants glide up and then back down, and an
endpoint-to-endpoint sign reports that as a null result.

**A register break is normal, not instability.** A modal-to-falsetto transition in an untrained voice
produces an abrupt F0 discontinuity that the octave-jump count will register. It is the expected
behaviour of a voice asked to glide across its range, and must not be reported as tracking failure
or as instability.

**Emits.** A per-span measurement carrying the trajectory values; a `deviate` assertion for a sweep
running against the declared direction.

**How well any detector separates the two glide families is unmeasured** in any source this document
can stand on, and measuring it against the declared families would fit the declaration.

**What is missing in code.** The trajectory computation lives in the routing-analysis module, not in
`tasks/phonation`. A branch consuming it needs it promoted so the branch and the router read one
definition.

### V4 — Voice quality (**machinery exists; branch consumption not built**)

**Question.** What are the noise and perturbation properties of the phonation?

**Computes.** All in `tasks/features_extraction/praat_parselmouth.py`: `extract_cpp_descriptors`
(`:706`), `extract_harmonicity_descriptors` (`:580`), `extract_jitter` (`:1112`), `extract_shimmer`
(`:1168`), `extract_pitch_descriptors` (`:448`), `extract_slope_tilt` (`:638`),
`measure_f1f2_formants_bandwidths` (`:824`), `extract_spectral_moments` (`:945`).

**CPPS is the primary descriptor as a method — but this implementation is voiced-interval-gated,
which removes the property that motivated promoting it.** `extract_cpp_descriptors` builds a
voiced/unvoiced TextGrid from `to_pitch_ac(..., voicing_threshold=0.3)` and computes CPPS only over
intervals labelled `V` (`praat_parselmouth.py:748-762`). So CPPS is **unavailable on exactly the
aperiodic voices V1's inversion exists to rescue** — the claim that it needs no period extraction is
true of the method and false of this code path.

**And its cepstral peak search is hard-capped at 60–330 Hz** (`praat_parselmouth.py:778-779`), so
CPPS is quietly invalid above 330 Hz: high-F0 women, children, and the upper half of every upward
glide.

Both are properties of the wrapper rather than of the measure, and the audit measured two more that
compound them: the vuv segmentation inflates voiced duration by **70%** on fragmented productions,
carrying silence into the cepstrogram and depressing CPPS; and a **`> 4` dB cut discards every
dysphonic interval**, returning NaN when all of them are — indistinguishable from a crash.

**The four findings multiply**: the vuv inflation depresses CPPS toward the cut, the 330 Hz cap
depresses it by a measured 3.8 dB at F0 420, and what survives is a mean over only the intervals that
already scored well. **[`praat-instrument-audit.md`](praat-instrument-audit.md) findings 2–4** carry
the measurements.

**So CPPS is suppressed, not qualified, until it is reimplemented.** An earlier version said the
right words — "promoted as a method and not currently trustworthy as implemented" — and then kept it
**primary**, which is what an implementer would act on. Those are incompatible.

**A value cut cannot be qualified away.** The `> 4` cut is selection on the dependent variable: no
covariate recovers an estimate from a filtered sample, and with support counts unavailable a reader
cannot see how much was deleted. And 4 dB sits *inside* the published normal/dysphonic decision
region, not below it.

**What replaces it is step 4 of [`praat-instrument-audit.md`](praat-instrument-audit.md)'s
remediation path** — a direct CPPS implementation of roughly thirty lines, with no value cut, no
interval gating, a quefrency band from the recording's own F0, duration weighting, and a support
count. Until that exists, **V4's primary descriptor is unavailable** and the branch says so rather
than reporting a number it has just described as untrustworthy.

**Three further defects in the existing function**, all in the audit: CPPS is averaged **unweighted**
across intervals, so finding 3's short padded intervals dominate a dysarthric recording's mean;
`std_dev_cpp` is a **between-interval** SD carrying a name that reads as the published
within-recording quantity; and `voicing_threshold=0.3` against Praat's 0.45 **compounds** the vuv
inflation rather than being independent of it.

Perturbation is secondary, and qualified rather than suppressed — see below.

**Perturbation is invalid on much of the material it will run on.** Cycle-to-cycle perturbation is
interpretable only on nearly-periodic signals — Titze (1995) signal typing, where type 1 is
nearly-periodic, type 2 has period doubling or subharmonics, and type 3 is frankly aperiodic.

**The gate must detect type 2, and the obvious qualifiers do not.** Period-mark recovery fraction,
HNR and F0 SD catch type 3, which was already obvious from listening or from a spectrogram. **Type 2
passes all three** — the tracker happily marks doubled cycles, recovery is high, HNR is moderate and
F0 SD is small. Type 2 is the class the gate exists for.

Two partial instruments, and neither is sufficient:

- **the octave-scale frame-to-frame jump count** from V3 — restricted to sustained material, never
  glides, per [`branch-conventions.md`](branch-conventions.md)'s interpretive rule;
- **the period-length distribution's modality** from `period_marks`
  (`tasks/phonation/__init__.py:3-12`).

**The modality test fails in the case it is meant to catch.** If the tracker locks to the
subharmonic, every mark is a doubled period and the distribution is **unimodal at 2T**. So
bimodality is evidence *of* type 2; its absence is not evidence against it. The direct instrument is
a subharmonic-to-harmonic ratio, and **it is not in the inventory** — stated rather than assumed
away.

**Type 2 does not reach the perturbation measurement at all, and the audit corrects how.** It is not
the 1.3 maximum-period factor: measured, `To PointProcess (periodic, cc)` places **0 pulses** on an
alternating-period signal and **1** on a period-doubled one, so the factor never gets to act. And
with floor 60 from the sex bin, a 45 or 55 Hz source yields **zero pulses** — vocal fry, low male
voices and Parkinsonian creak are excluded **upstream by the bin**.

**So a diplophonic voice reads as `unmeasurable`, not as `severely disordered`** — while
`default.yaml:145` names `phonation.period_doubling_factor: 2.0` as a phenomenon of interest. See
[`praat-instrument-audit.md`](praat-instrument-audit.md) finding 8, which also records that the
`0.0001 / 0.02 / 1.3 / 1.6` literals are Praat's own form defaults, mismatched against the derived
range in a way Praat's documentation warns about.

**The validity qualifier and the value come from different point processes.** `period_marks` admits
`[1/f0_max, 1/f0_min]`; the jitter and shimmer calls admit `[0.0001, 0.02]` with the 1.3 constraint.
So a recovery fraction computed from `period_marks` **certifies cycles the computation may have
discarded**. Compute the qualifier from the same point process the value came from.

**Report alongside every perturbation value:** the period-mark recovery fraction *from the same point
process*, HNR, F0 SD, the octave-jump count where the material is sustained, the period-length
modality, and V1's steadiness qualifiers.

**HNR is not independent of the noise covariate.** Additive room noise lowers HNR for a perfectly
type-1 voice, so the type-3 qualifier and the SNR covariate are two readings of overlapping evidence,
not corroboration. **Period-mark evidence outweighs HNR** for the signal-typing question.

**And HNR's validity role inherits the sex bin.** The harmonicity window steps **75 → 45 ms** across
the 170 Hz boundary, so HNR values are not comparable across it — and using HNR as the evidence
deciding whether a perturbation value is valid **imports the sex-typed discontinuity into the
validity judgement itself**, not merely into the measurement.

**Two covariates specific to this block**, beyond the shared set in
[`branch-conventions.md`](branch-conventions.md):

- **sample rate.** Period-mark precision is quantised by the sample period, so jitter has a floor
  set by the sample rate, and this corpus has mixed rates. *Effective bandwidth does not substitute*
  — it is about the frequency content, this is about time resolution.
- **segment SNR.** Additive noise inflates both measures.

**Name the variant.** `local` jitter and shimmer are sensitive to slow drift; `ppq5` and `rap` much
less so. On a vowel with vibrato the two diverge materially while both being reported as "jitter".
Emit the variant name with the value, never a bare "jitter".

**Declare an analysis-window convention** excluding attack and decay, per
[`branch-conventions.md`](branch-conventions.md).

**It qualifies; it does not suppress.** The word "gated" was used loosely in an earlier version while
the operative instruction was "report alongside" — those are different designs. **This section
qualifies**: every perturbation value is emitted with the evidence a reader needs to discount it.
Suppression would require a cut on the octave-jump count and a named statistic for modality — a dip
test, a bimodality coefficient — and neither is named nor owed here, because neither is wanted.

**Emits.** A per-span measurement, every value named per
[`branch-conventions.md`](branch-conventions.md)'s convention-in-the-name rule. **No verdict** —
mapping these to normal or disordered needs norms this project does not have. Praat's default
jitter, shimmer and HNR settings (1.04%, 3.81%, 20 dB) are widely mistaken for norms and are not
norms, and no published perturbation norm was collected on AGC'd, band-limited phone audio.

### V4a — Vocal tremor (**not built; the largest missing capability in this branch**)

**Question.** Is there a 4–8 Hz modulation of F0 or amplitude?

Vocal tremor is central in essential tremor, Parkinson's disease and spasmodic dysphonia, and the
**sustained vowel is its standard instrument** — 5,113 recordings here (`prolonged-vowel` plus both
`maximum-phonation-time` families). It is absent from this branch entirely.

**It is nearly free once DDK D1 exists.** Tremor is a modulation-spectrum measurement — the same
machinery [`branch-ddk.md`](branch-ddk.md) D1 specifies, run at a different search band (4–8 Hz
rather than the syllable rate) over the F0 contour and the amplitude envelope of a V1 attempt.

**It also explains a V4 observation.** A 4–8 Hz modulation is exactly what makes `local` jitter and
shimmer diverge from `ppq5` and `apq11`: the short-window variants track the modulation, the
longer-window ones average across it. Reporting tremor turns that divergence from an anomaly into a
measurement.

**Emits.** A per-span measurement: modulation frequency, modulation depth for F0 and for amplitude,
and the search band as a declared convention.

### V5 — Composite severity indices: **moved, and the conclusion is stronger than "the owner decides"**

AVQI and its relatives are **not per-recording capabilities here** — their protocol requires a
sustained vowel *and* continuous speech concatenated, which in this corpus are different recordings.
The capability sits in [`corpus-level-node.md`](corpus-level-node.md) C2, as a **session-level**
capability.

**But the placement is not the interesting conclusion.** An earlier version left the question open
for the owner. It should not be open, because this document's own condition — *the concatenation
protocol must be followed rather than approximated* — **is unsatisfiable in this corpus**:

- the continuous-speech material must be the short standardized sentence set the index was
  calibrated on, in a prescribed vowel:speech proportion. Rainbow, Caterpillar and Harvard are none
  of these, and free speech is disqualifying;
- only `prolonged-vowel` (1,604) can supply the vowel, and only when it is /a/ — which makes **V8 a
  precondition**, not a nicety;
- the two takes are different recordings, so independent AGC states, gain and mic distance put a
  level and spectral discontinuity **at the join, directly into the LTAS terms** the index reads;
  bandwidth and sample rate must match across the pair and do not;
- **`extract_slope_tilt`'s band definitions are not AVQI's**, so following the protocol means
  reimplementing the index's own measurements rather than composing the existing helpers;
- the two takes are different voice states, recorded minutes apart.

**So the choice is not "emit AVQI or not". It is between emitting an AVQI-shaped number that is not
AVQI, and emitting nothing.** A value assembled this way has **no known relationship to the published
0–10 scale or to its ~2.95 cut-off**, and the published smartphone recording-chain sensitivity
applies on top.

The non-diagnostic argument stands beside this one: emitting a 0–10 severity estimate is mapping an
acoustic value to normal-or-disordered, which every other capability here declines to do. It also
ingests shimmer with the largest positive coefficient and no validity gate, while V4 qualifies
shimmer everywhere else.

**What VOICE keeps**: CPPS as the *intended* primary descriptor — it carries the dominant weight in
these composites anyway, and reported alone it is a description rather than a severity estimate. But
**not the current implementation**, which V4 suppresses until the audit's step 4 replaces it.

### V6 — Vocal effort events (**not built; and the two tasks are not one measurement**)

**Reading the sidecar settled what the tasks are.** An earlier version described `loudness` as
connected speech at three instructed levels and proposed clustering on level; both were wrong. From
the protocol's own `_acoustictask-metadata.json`:

> **`loudness`** (897) — *"…shout "hey" as loud as possible 3 times in a single recording."*

> **`loudness-v2`** (705) — *"…say "hey" in your normal voice. Then, shout "hey" as loud as you can.
> Try to reach the target line on the screen."*

Both carry `speech_type: "non-lexical"` and an empty `stimulus_text`. A single syllable, not connected
speech.

**They are two different measurements, and a second correction is needed here.** A previous version
presented one design covering both, but **v1 has only one condition**:

| task | conditions | what is measurable per recording |
| --- | --- | --- |
| `loudness` (897) | **one** — three maximal attempts | absolute tilt / F0 / CPP at maximal effort (uncalibrated), and **dispersion across the three attempts** |
| `loudness-v2` (705) | **two** — normal, then shouted | the **between-condition** change in each correlate |

**Every effort correlate is a between-condition change, so none of them is computable on v1 from the
file alone.** What v1 supports is dispersion across its three attempts, and absolute values that are
not cross-comparable.

**This is methodologically an event task, closer to A5/A6 than to V1–V4.** Event detection plus event
descriptors. A previous version half-saw that and then attached V4-family measures that do not
survive the material.

#### What stays per-recording

Event detection on the envelope — isolated single-syllable events separated by silence, the same
instrument as [`branch-airway.md`](branch-airway.md) A5 — with the **expected count declared** (3 or
2). That makes a `counts` entry with a real `declared` half, which no other VOICE capability has.

Then, per recording: the **count**, the **timing**, and **within-file dispersion** across attempts.

**Report the measured level separation as the result, never as the segmentation criterion.** A
previous version proposed clustering on level, which was the same error V1 and D1 were inverted to
remove — the event defined by the property being measured. Reduced dynamic range **is** the deficit in
hypophonia, and k-means with k=3 always returns three clusters, partitioning noise and reporting it as
condition effects.

**Drop the order check.** A previous version asked both for the measured separation between ordered
events (correct) and for "whether the recovered order matches the declared one" — which is **not
independently observable**: on v2 the only way to tell which event was the shout is the measurement
itself. Circular on v2, vacuous on v1.

#### Clipping — the hazard this task has and V2 handles rigorously

**This is the corpus's most clip-prone task and clipping was absent from the previous version.** Two
consequences, both parallel to hazards V2 treats carefully:

- **The maximum is right-censored by the converter**, exactly as MPT is right-censored by the
  recording boundary. V2 refuses to pool censored trials; V6 must do the same and said nothing.
- **Clipping flattens spectral tilt** — so the alpha-ratio correlate offered as the escape from the
  level confound is corrupted by the same event that corrupts the level.

The generic clipping covariate in [`branch-conventions.md`](branch-conventions.md) does not cover
either; both are task-specific and belong here.

#### Shouting moves F0 across the 170 Hz bin, which destroys the comparison

Maximal effort raises F0 by roughly 3–8 semitones, and `derive_f0_range` selects its bin from the
recording's **own trimmed mean** (V3). So a male speaker at a comfortable 120 Hz shouts at 200–250 Hz:
his `loudness` recording lands in the **100/500** bin while his `prolonged-vowel` lands in
**60/250**.

**The within-participant comparison these correlates exist for is destroyed by the instrument**, not
by the voice. And on v2 the trimmed mean sits *between* the two conditions, so a mixture selects the
analysis range for both.

#### Name it for what it is

An earlier version's table said **"maximum vocal intensity"** for an uncalibrated, gain-unknown,
possibly clipped level — while V2, in the same document, refuses to emit `maximum_phonation_time` for
exactly this hazard class. Per [`branch-conventions.md`](branch-conventions.md)'s naming rule, the
value carries its convention: relative peak level within the recording, with the clipping state
attached. **Absolute SPL is not recoverable** from a file of unknown gain.

**Do not build any of this on `range_ratio_intensity_db`.** `praat_parselmouth.py:566` computes it as
`max_dB / min_dB` — measured **1.000** on a buzz with no silence, **−0.244** once silence is added. It
tracks silent fraction, not dynamic range. [`praat-instrument-audit.md`](praat-instrument-audit.md)
finding 6.

#### The effort correlates are session-level

**F0, tilt and CPP at maximal effort are interpretable only against the same participant's comfortable
phonation in the same sitting** — which is a different recording. That is the grouping
[`corpus-level-node.md`](corpus-level-node.md) already specifies, and moving them there **also
resolves v1's missing second condition**: it comes from the session, not from the file.

So: the count, the timing and within-file dispersion are V6's. The correlates are the node's.

**Owed.** A pause criterion for separating the events.

**A consequence for SPEECH.** Both families carry `speech_type: "non-lexical"` while `families.py`
places them in `LEXICAL_SPEECH` — see [`branch-speech.md`](branch-speech.md).

### V7 — Population-conditioned F0 range (**gated behind null config; and it should stay that way**)

`_f0_range` (`voice.py:48-83`) reads `hint.metadata["population"]` against
`voice.f0_range_by_population`, which is null (`default.yaml:160`), falling back to `derive_f0_range`
(`voice.py:71`).

**The argument in the previous version is defeated on its own premise.** It said a population prior
should stay unpopulated because the derived per-recording range is the better path. But V3 above
establishes that **the fallback *is* a population prior** — implicit, undeclared, binary, sex-typed,
and cruder than any published norm: two hardcoded pairs selected at a 170 Hz boundary.

So the choice is not "population prior versus per-recording derivation". It is **between an implicit
two-bin prior and an explicit declared one**, and the implicit one is worse in every respect except
that nobody had to write it down.

**The hazard the previous version named is real and now applies to the code as it stands.**
Conditioning an F0 analysis range on population clips a speaker whose F0 sits outside their
population's typical range — puberphonia, a trans-feminine voice — and this protocol deliberately
enrols gender-diverse participants. That is exactly what the 170 Hz bin does today, without
declaring it.

**The alternative that needs no norm**: a genuinely wide analysis range, plus V3's octave-jump
diagnostics, plus the three-level conformance flag. That reports when the range was inadequate rather
than assuming one from an inferred demographic.

**So the defect is in the function, not in the null key.** `voice.f0_range_by_population` being null
is not the problem; `derive_f0_range` silently supplying a sex-typed prior is. Fixing the config
would not fix it.

**`population` is not a field the contract's declaration defines** — see Unresolved.

### V8 — Which vowel was produced (**not built**)

Both a covariate — formant and perturbation measures differ by vowel — and a conformance check on
tasks that specify one.

**Use formants, not the PPG.** An earlier version proposed reading vowel identity from the PPG
posteriorgram; by DDK's own argument the posteriorgram is out of domain on sustained productions, and
`measure_f1f2_formants_bandwidths` (`praat_parselmouth.py:824`) is the direct instrument — F1 and F2
are what vowel identity *is*.

**The formant configuration is adult-male, and which function it configures matters.**
`phonation_spans.formant_max_hz: 5000.0` with `max_formants: 5` configures **`formant_track`**
(`preprocess.py:942-948`), which is the function V8 should use. Praat's guidance is roughly 5500 Hz
for an adult female voice and higher for children, so formant estimates carry a sex- and
age-structured error. **That one is a configuration question.**

`measure_f1f2_formants_bandwidths` is a **different** function and a different problem: the wrapper
forwards only four of its eight parameters (`praat_parselmouth.py:1342-1347`), so
`maximum_formant_hz` is **unreachable from any caller** — a code defect, not a configuration choice.
An earlier version of this section cited that function alongside the `phonation_spans` key, which
conflated the two.

**And LPC formant estimation degrades with F0 regardless of configuration.** At F0 250 Hz the
harmonic spacing undersamples the spectral envelope and F1 for a close vowel like /i/ is essentially
unrecoverable. So **V8 fails systematically on high vowels and high-F0 speakers** — and the same
limitation lands on **V1's formant-stationarity qualifier**, which will read "unsteady" for high-F0
speakers for tracker reasons rather than production ones.

**V8 is a precondition for the composite indices**, not an optional covariate: AVQI's protocol
requires the sustained vowel to be /a/, so without vowel identity the pair cannot even be assembled.
See [`corpus-level-node.md`](corpus-level-node.md) C2.

## Deviations

**A deviation is not evidence of a bad recording.** See
[`branch-conventions.md`](branch-conventions.md): `repeat_attempt` and `truncation` are produced by
the conditions this corpus studies, and a consumer filtering on them is filtering on impairment.

| type | evidence |
| --- | --- |
| `sweep_direction_mismatch` | the dominant monotone segment ran opposite to the declared direction (V3) |
| `truncation` | the attempt ran to the recording boundary — the MPT is right-censored (V2) |
| `repeat_attempt` | more than one phonation attempt where the task asked for one (V2) |

**`off_task_extent` is withdrawn from this branch.** It previously marked *"a region with no
phonation where the task asked for one"* — which makes the silence around every maximum-phonation
effort a deviation and needs an undeclared minimum duration to avoid firing constantly. **Absence of
the target is a measurement**: V2's interruption triple reports it, with locations and durations, and
reports it better. The one correct instance of `off_task_extent` is AIRWAY's, which keys on
positively-identified off-task content.

`attempt_count` is a `counts` entry. VOICE emits no `stimulus_mismatch` — no voice task carries a
stimulus text.

## A branch `FAIL` is an absence of detected content

`VOICE: FAIL` means **this branch's detector found no phonation**, never that the speaker produced
none. The distinction is load-bearing at corpus scale: a detector keyed on voiced frames fails most
often on disordered phonation, so `FAIL` would concentrate on the most impaired speakers across
62,547 recordings. **V1's envelope-first proposal removes most of that cause** — an attempt with zero
voiced fraction is still an attempt, and is reported as one.

The residue: a phonation too quiet to clear the envelope threshold still yields no span. See
[`branch-conventions.md`](branch-conventions.md), and Unresolved below on `Outcome.FAIL`'s wording.

## What exists today

| capability | status |
| --- | --- |
| V1 propose the attempt | **not built** — the branch has no subject; reuses `spans.k_db` and `spans.min_duration_ms`; stationarity qualifiers required |
| V2 maximum phonation time | not built; previously specified as a longest voiced run |
| V3 F0 trajectory | 22 detectors held; branch consumption not built; **`derive_f0_range` is a binary sex-typed bin, ceiling 250 or 500 Hz** |
| V4 voice quality | **every Praat scalar is computed on FRCRN-enhanced audio** (audit finding 0); CPPS suppressed until reimplemented; perturbation qualified |
| V4a vocal tremor | **not built**; largest missing capability; nearly free once DDK D1 exists |
| V5 composite severity | **moved** to [`corpus-level-node.md`](corpus-level-node.md) C2, session-level; the protocol is unsatisfiable here |
| V6 vocal effort events | not built; event detection and dispersion stay here, the effort correlates are session-level |
| V7 population F0 range | null key, but an implicit two-bin prior is already in force in `derive_f0_range` |
| V8 vowel identity | not built; use formants; configuration is adult-male; precondition for C2 |

Reachable today: `_f0_range` resolution, `resolve_stream`, the activity write, and the
`gate_interval` tri-state (`voice.py:196-203`). Everything from `:268` on is unreachable because the
branch returns at `:266` — about 153 lines, or roughly 200 counting the helpers only it calls.

## What the branch emits

```
spans        family: "voice" phonation attempts, each carrying voiced fraction,
             interruption structure and F0 availability (V1)
assertions   refine (corrected_extent) where a PREPROCESS span's extent is wrong;
             label naming vowel identity (V8);
             deviate (sweep_direction_mismatch, truncation, repeat_attempt)
measurements per-span trajectory with the selected F0 bin (V3), qualified
             perturbation with its validity evidence and variant names (V4;
             CPPS withheld until reimplemented), vocal tremor (V4a), effort-event
             level and dispersion with clipping state (V6) — each with its
             extent's covariates, steadiness qualifiers and support count
counts       attempt count {found}; the MPT extent under an explicitly qualified
             name, or no scalar at all
verdict      { spans_n, phonation_s, production, ambiguous_spans_n,
               marks_skipped_short_n, task_range, gate_interval, flags }
```

**`longest_span_s` is removed from the verdict — and that is a writer-vocabulary retirement with
live readers.** `report.py:104` keys VOICE's summary on it and `voice_test.py:344-352` asserts it is
a first-class product. Removing it is right: it is the retired longest-voiced-run measure surviving
in the payload, and it will be read as MPT against published norms whatever V2 says elsewhere.

But the same rule this document applies to the `phonation` → `voice` family rename applies here —
**the writer's vocabulary may shrink, the reader's may not.** The migration owes: what
`report.py:104` reads instead, and what the test asserts instead. An earlier version removed the key
and said nothing about either.

**The verdict's basis, exactly.** Today: `FAIL` at `voice.py:242` when no phonation span exists —
every recording. On the unreached path: `FLAG` when flags accumulated, `PASS` when spans were
measured and nothing contested (`voice.py:396-399`).

Under the contract: `FAIL` when no attempt was found; `FLAG` when one measurement contradicts
another; `PASS` otherwise — every number a measurement, no normative judgement in the verdict.

**There is no `contest` capability.** An earlier version listed *"contest where a proposed phonation
span carries no voiced frame"*, which is self-contradictory twice: a V1 span is no longer defined by
voicing at all, and contesting the branch's own proposal has no object. If VOICE is to contest, the
object must be a PREPROCESS span.

## Out of scope

Normative interpretation of any acoustic value. Severity estimates (V5). Absolute intensity.
Attribution of a short MPT to respiratory or laryngeal cause. Any refit against declared families.

## Unresolved

- **Whether to emit an AVQI-shaped number that is not AVQI** (V5). The protocol is unsatisfiable in this corpus, so this is the actual question; the owner decides it, but not as an open choice between AVQI and nothing.
- **Three declaration fields this branch reads are not in the contract's `metadata` contract**:
  V2's expected duration, V7's `population`, V8's expected vowel. The contract froze the entry keys;
  `data/task_expectations/` does not exist yet, so extending it is free but must be asked for.
- Whether the MPT extent is reported as an upper bound or intersected with voicing evidence.
- What `report.py:104` and `voice_test.py:344-352` read once `longest_span_s` is retired.
- `phonation_spans.formant_max_hz: 5000.0` is adult-male; affects V4 and V8.
- **The Praat wrapper audit is complete** — [`praat-instrument-audit.md`](praat-instrument-audit.md). **Finding 0 — every Praat scalar is measured on FRCRN-enhanced audio — governs this whole branch**, and its remediation steps 0 and 1 come before any other repair here.
- Whether V6's effort correlates are built at all, given they are session-level and v1 supplies only one condition.
- **`Outcome.FAIL`'s wording is itself a hazard** — `no_content_found` would carry the meaning — but
  `Outcome` is a closed vocabulary with readers, so this is recorded rather than changed.
