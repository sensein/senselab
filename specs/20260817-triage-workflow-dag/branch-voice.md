# VOICE branch

What the branch answers: **is there sustained phonation here, what are its acoustic properties, and
what did the voice do across it?**

The frame is [`../20260913-branch-contract-and-hints/design.md`](../20260913-branch-contract-and-hints/design.md);
shared rules are in [`branch-conventions.md`](branch-conventions.md); owed ground truth is in
[`branch-listening-sample.md`](branch-listening-sample.md).

## The state of this branch

**VOICE fails on every recording.** Its subject is every live span whose `family` is `phonation`
(`voice.py:232`, `_PHONATION_FAMILY` at `:39`). Nothing reachable proposes one: the detector that
did was retired on 2026-09-04, and the module's only `prov_type="span"` write (`voice.py:336-348`)
sits downstream of the no-span return at `:237-266`, which every recording takes.

**And VOICE is the worked example of what the contract forbids.** `voice.py:336-348` mints a
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
still V5's — intensity dynamics — which is the ordinary case of a branch running on content rather
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
- **F0 availability** — whether the tracker produced a usable contour at all.

A type-3 voice yields an attempt with a low voiced fraction and no usable F0 contour. That is a
*finding about the voice*, and it is exactly what a reader needs. Defined the other way it was an
absence of data.

**Emits.** `propose` spans, `family: "voice"`, subject to the `propose`/`refine` rule in
[`branch-conventions.md`](branch-conventions.md), each carrying the three qualifiers above.

**Parameters, all owed.** The claim that this is "parameter-free in its core" is withdrawn — it was
false under the old definition and is still false under the new one:

1. the envelope threshold separating attempt from background, and a minimum attempt duration;
2. `f0_track` calls Praat's `to_pitch_cc` with floor and ceiling from `voice.f0_search_range_hz`,
   shipping `[50.0, 600.0]` (`default.yaml:159`) — this project's number, no derivation — on top of
   Praat's own undeclared internal voicing threshold;
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

**The search range clips the diagnostically interesting endpoint of both glide families.**
`[50.0, 600.0]` (`default.yaml:159`) excludes female falsetto, which reaches 700–900 Hz and beyond,
and excludes the fry below 50 Hz where downward glides end. The endpoint of the sweep is exactly
where the tracking fails.

Three parameter-free responses, all available now:

- use `derive_f0_range` (`tasks/phonation/__init__.py:3-12`) for **per-recording** limits rather than
  one fixed window — noting that **widening an F0 search range is a trade, not a free win**: it
  increases octave errors across the whole track, so the range-clipping flag below is what makes the
  narrow default honest rather than something to fix by widening alone;
- add a **conformance flag**: the F0 extremum coincided with the search limit, which says the range
  was clipped without asserting what the true extremum was;
- report a **robust percentile range** beside the extrema, and a **count of octave-scale
  frame-to-frame jumps**, which is the tracker's own instability made visible.

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

**CPPS is the primary descriptor.** It requires no period extraction, so it survives exactly the
type-2 and type-3 signals where perturbation measures die; it is far more robust to additive noise
and to compression than jitter or shimmer; and it carries the dominant weight in the composite
instruments that combine these measures. Perturbation is secondary and gated.

**Perturbation is invalid on much of the material it will run on.** Cycle-to-cycle perturbation is
interpretable only on nearly-periodic signals — Titze (1995) signal typing, where type 1 is
nearly-periodic, type 2 has period doubling or subharmonics, and type 3 is frankly aperiodic.

**The gate must detect type 2, and the obvious qualifiers do not.** Period-mark recovery fraction,
HNR and F0 SD catch type 3, which was already obvious from listening or from a spectrogram. **Type 2
passes all three** — the tracker happily marks doubled cycles, recovery is high, HNR is moderate and
F0 SD is small. Type 2 is the class the gate exists for.

Two instruments detect it, and one is already in this document:

- **the octave-scale frame-to-frame jump count** from V3, which is sitting in the glide capability
  and was not wired into the perturbation block;
- **direct subharmonic evidence** — a bimodal period-length distribution from `period_marks`
  (`tasks/phonation/__init__.py:3-12`), which is what period doubling *is*.

**Report alongside every perturbation value:** the period-mark recovery fraction, HNR, F0 SD, the
octave-jump count, and the period-length distribution's modality. The first three say the signal was
not type 3; the last two say it was not type 2.

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

**Emits.** A per-span measurement. **No verdict** — mapping these to normal or disordered needs norms
this project does not have, and Praat's default jitter, shimmer and HNR settings (1.04%, 3.81%,
20 dB) are widely mistaken for norms and are not norms. No published perturbation norm was collected
on AGC'd, band-limited phone audio.

### V5 — Composite severity indices (**not built; and the rule I invoked was the wrong one**)

The Acoustic Voice Quality Index (Maryn et al. 2010) combines CPPS, HNR, shimmer local, shimmer dB
and LTAS slope and tilt. An earlier version of this document proposed emitting it on the grounds
that its **coefficients are published rather than fitted here**, so it discharges the no-refits rule.

That is true and it is beside the point. **AVQI is calibrated against perceptual overall grade and
reported on a 0–10 severity scale with a published cut-off around 2.95.** Emitting it is emitting a
severity estimate — which engages the *non-diagnostic* constraint, not the no-refits one. Every other
capability in this document declines to map an acoustic value to normal or disordered; this one would
do exactly that, with a number that looks authoritative because it is externally anchored.

**Three consequences:**

1. **CPPS is promoted to V4's primary descriptor** on its own merits — it dominates AVQI's weight
   anyway, and reported alone it is a description rather than a severity estimate.
2. **AVQI is not a per-recording capability here.** Its protocol requires a sustained vowel *and*
   continuous speech concatenated; in this corpus those are **different recordings**. So it is a
   corpus-level capability like duplicate detection, not something VOICE computes on one file.
3. **It has no validity gate.** Shimmer carries its largest positive coefficient, and V4 gates
   shimmer everywhere else in this document. A composite that ingests an ungated shimmer inherits
   its invalidity silently.

**Whether this project emits any severity estimate is a decision for the owner**, not one this
document resolves. Recorded as owed.

### V6 — Loudness-task conditions and effort correlates (**not built**)

**Question.** How did the voice change across the instructed loudness conditions?

**The prior version of this section had no subject.** `loudness` (897) and `loudness-v2` (705) are
connected speech produced at instructed levels. Under V1's old definition a phonation span was a run
of voiced frames, so a loudness recording is **dozens of short runs, not one phonation** — and
nothing in any of these documents segmented it into its soft, comfortable and loud conditions.
Without that segmentation there is no loudness measurement at all, only an intensity contour over an
undifferentiated recording.

**Condition segmentation comes first, and it is close to parameter-free.** The conditions are
instructed, ordered, pause-separated, and multimodal in level by construction — three clusters in
level, in a known order, separated by pauses. Segmenting on that structure needs no fitted cut.

**Then measure the level-invariant effort correlates as primary**, not the dB range:

- **spectral tilt / alpha-ratio change per dB** — `extract_slope_tilt` (`praat_parselmouth.py:638`);
- **F0 shift across conditions** — `extract_pitch_descriptors` (`:448`);
- **CPP change across conditions** — `extract_cpp_descriptors` (`:706`).

**dB range is secondary and heavily qualified.** It is precisely what AGC destroys, and it is
corrupted by distance change — participants pull the phone away when asked to be loud, which is
task-correlated rather than random. The level-invariant correlates survive both. An earlier version
of this document made dB range the measurement and the qualifiers a caveat; that inverted the
reliability ordering.

**Absolute SPL is not recoverable** from a file of unknown gain, and `recording_input_gain` does not
recover it.

**Intensity ownership.** SPEECH S4 computes "intensity variability" on these same recordings over
different extents. **V6 owns intensity across instructed conditions; S4 owns it within connected
speech.** Both cite the other so a reader finds one answer.

### V7 — Population-conditioned F0 range (**gated behind null config; and it should stay that way**)

`_f0_range` (`voice.py:48-83`) reads `hint.metadata["population"]` against
`voice.f0_range_by_population`, which is null (`default.yaml:160`), falling back to `derive_f0_range`
over the wide search range.

**Conditioning an F0 search range on population silences the population most likely to be studied.**
A male-registered speaker with unusually high F0 — puberphonia, a trans-feminine voice — falls
outside a population prior built from typical ranges, and the tracker then clips exactly the voice
the measurement was wanted for. This protocol deliberately enrols gender-diverse participants, so
this is not a hypothetical.

**The alternative needs no norm at all**: a wide search range, plus V3's octave-jump diagnostics,
plus the range-clipping conformance flag. That combination reports when the range was inadequate
instead of assuming a range from a demographic label.

So `voice.f0_range_by_population` being null is not a gap to close. If it is ever populated, the
per-population values are published clinical norms and belong in `data/` with a cited source — but
the case for using them at all has not been made.

**`population` is not a field the contract's declaration defines** — see Unresolved.

### V8 — Which vowel was produced (**not built**)

Both a covariate — formant and perturbation measures differ by vowel — and a conformance check on
tasks that specify one.

**Use formants, not the PPG.** An earlier version of this document proposed reading vowel identity
from the PPG posteriorgram. By DDK's own argument the posteriorgram is out of domain on sustained
productions, and `measure_f1f2_formants_bandwidths` (`praat_parselmouth.py:824`) is the direct
instrument for exactly this: F1 and F2 are what vowel identity *is*, and they are already needed as
covariates.

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
| V1 propose the attempt | **not built** — the branch has no subject; envelope parameters owed |
| V2 maximum phonation time | not built; previously specified as a longest voiced run |
| V3 F0 trajectory | 22 detectors held; branch consumption not built; search range clips both glide endpoints |
| V4 voice quality | Praat machinery complete; CPPS primary, perturbation gated; type-2 detection not built |
| V5 composite severity | **not built, and deliberately** — a severity estimate, and a corpus-level protocol |
| V6 loudness conditions | not built; condition segmentation missing entirely |
| V7 population F0 range | gated behind null config, and the case for it is not made |
| V8 vowel identity | not built; use formants |

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
measurements per-span trajectory (V3), CPPS and gated perturbation with its
             validity qualifiers and variant names (V4), loudness-condition
             correlates (V6) — each with its extent's covariates and support count
counts       attempt count {found}; the MPT extent under an explicitly qualified
             name, or no scalar at all
verdict      { spans_n, phonation_s, production, ambiguous_spans_n,
               marks_skipped_short_n, task_range, gate_interval, flags }
```

**`longest_span_s` is removed from the verdict.** It is the retired longest-voiced-run measure
surviving in the payload, and it will be read as MPT against published norms whatever V2 says
elsewhere.

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

- **Whether this project emits a severity estimate at all** (V5). A decision for the owner.
- **Three declaration fields this branch reads are not in the contract's `metadata` contract**:
  V2's expected duration, V7's `population`, V8's expected vowel. The contract froze the entry keys;
  `data/task_expectations/` does not exist yet, so extending it is free but must be asked for.
- Whether the MPT extent is reported as an upper bound or intersected with voicing evidence.
- **`Outcome.FAIL`'s wording is itself a hazard** — `no_content_found` would carry the meaning — but
  `Outcome` is a closed vocabulary with readers, so this is recorded rather than changed.
