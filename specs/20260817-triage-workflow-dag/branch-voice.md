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

### V1 — Propose phonation spans (**not built; the branch's missing foundation**)

**Question.** Where is the sustained phonation?

**Reads.** PREPROCESS's general spans and its `phonation_tracks` measurement — written whole-file by
`phonation_tracks` (`preprocess.py:919`, called at `:2147`), which runs `f0_track` over the
pre-emphasised stream for per-frame F0 and voicing strength and `formant_track` over `plain` for four
formants — plus the HNR track VOICE can compute via `hnr_track`
(`tasks/phonation/__init__.py:3-12`).

**Computes.** A contiguous region of voiced frames.

**Emits.** `propose` spans, `family: "voice"`, `wasDerivedFrom` the phonation track, subject to the
`propose`/`refine` rule in [`branch-conventions.md`](branch-conventions.md).

**Two parameters, both owed.** An earlier version of this document called the capability
"parameter-free in its core" on the grounds that *"F0 exists here"* is Praat's decision rather than
this branch's. That is false and the claim is withdrawn:

1. `f0_track` calls Praat's `to_pitch_cc` with floor and ceiling from `voice.f0_search_range_hz`,
   which ships `[50.0, 600.0]` (`default.yaml:159`) — this project's number, carrying no derivation
   — on top of Praat's own undeclared internal voicing threshold.
2. Nothing says whether a single unvoiced frame breaks a run. With zero tolerance a sustained vowel
   shatters into fragments at every tracker dropout.

Both are owed. The same config block already marks `phonation.hnr_floor_interval_db` and
`phonation.rms_floor_interval` owed (`default.yaml:146-147`, whose comment records that *Praat
calibrates no dB floor*); this extends that honesty rather than departing from it.

### V2 — Maximum phonation time (**not built**)

**Question.** How long did the phonation last, and was it continuous?

**MPT is not the longest contiguous voiced run.** The clinical convention is onset to audible
cessation (Kent, Kent & Rosenbek 1987). Praat's tracker drops out precisely on voice breaks,
diplophonia, subharmonics and terminal creak — which are the signatures of disordered phonation. So
a longest-run measure **halves the MPT of any voice with a mid-phonation break and truncates the end
of nearly every effort, worst in the most dysphonic voices**: a directional bias against the
population the task exists to characterise. An earlier version of this document specified exactly
that measure; it was a scientific error, not a wording problem.

**Report a triple instead**, over the attempt:

1. the **onset-to-offset extent** of the attempt, from the energy envelope or VAD rather than from
   the F0 tracker;
2. the **total voiced duration** within it;
3. the **number, location and total duration of internal interruptions**.

**And report the number of attempts.** Taking a maximum silently discards false starts, which are
themselves informative.

**No `declared` half.** "As long as possible" declares no duration, so the `counts` entry carries
`found` alone. `voice.task_duration_ranges` is null (`default.yaml:162`) and `_task_range`
(`voice.py:131-163`) returns `not_evaluated` immediately; under the contract the check becomes a
report rather than a gate, which is the better form.

**Serves.** `maximum-phonation-time` (2,696), `-v2` (813), `prolonged-vowel` (1,604) secondarily.

**Flag for the reader:** MPT confounds respiratory and laryngeal contributions, and the measure that
separates them — the s/z ratio — is not in this protocol. Nothing here can attribute a short MPT to
one or the other.

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
  one fixed window;
- add a **conformance flag**: the F0 extremum coincided with the search limit, which says the range
  was clipped without asserting what the true extremum was;
- report a **robust percentile range** beside the extrema, and a **count of octave-scale
  frame-to-frame jumps**, which is the tracker's own instability made visible.

**Add the direction deviation.** The sweep ran opposite to the requested direction — a sign, which
is an observable rather than a classification — over the 3,150 glide recordings. That is the
conformance finding the glide families exist to produce and it was previously absent.

**Emits.** A per-span measurement carrying the trajectory values; a `deviate` assertion for a sweep
running against the declared direction.

**How well any detector separates the two glide families is unmeasured** in any source this document
can stand on, and measuring it against the declared families would fit the declaration.

**What is missing in code.** The trajectory computation lives in the routing-analysis module, not in
`tasks/phonation`. A branch consuming it needs it promoted so the branch and the router read one
definition.

### V4 — Voice quality (**machinery exists; branch consumption not built**)

**Question.** What are the perturbation and noise properties of the sustained phonation?

**Computes.** All of this exists in `tasks/features_extraction/praat_parselmouth.py`:
`extract_jitter` (`:1112`), `extract_shimmer` (`:1168`), `extract_harmonicity_descriptors` (`:580`),
`extract_cpp_descriptors` (`:706`), `extract_pitch_descriptors` (`:448`),
`extract_intensity_descriptors` (`:515`), `extract_slope_tilt` (`:638`),
`measure_f1f2_formants_bandwidths` (`:824`), `extract_spectral_moments` (`:945`).

**Jitter and shimmer are invalid on most material they will run on.** Cycle-to-cycle perturbation is
interpretable only on nearly-periodic signals (Titze 1995 signal typing), and a substantial fraction
of a clinical voice corpus is not. Reporting a jitter value with no indication of whether the signal
supported it is worse than reporting nothing.

**A parameter-free fix using what exists.** Report alongside every perturbation value:

- the **fraction of the segment for which consecutive `period_marks` were recovered**
  (`period_marks`, `tasks/phonation/__init__.py:3-12`) — a direct measure of whether cycle
  extraction succeeded;
- **HNR and F0 standard deviation** over the same segment.

Together these say whether the perturbation value means anything, without classifying the signal.

**Declare an analysis-window convention** excluding attack and decay, per
[`branch-conventions.md`](branch-conventions.md).

**Emits.** A per-span measurement. **No verdict** — mapping these to normal or disordered needs norms
this project does not have.

### V5 — AVQI (**not built; the one externally-anchored instrument available**)

The Acoustic Voice Quality Index (Maryn et al. 2010) combines CPPS, HNR, shimmer local, shimmer dB,
and LTAS slope and tilt. **Its coefficients are published rather than fitted here**, so computing it
does not violate the no-refits rule — and it is the only externally-anchored operating point
available to this project without touching the corpus. Every component is in the Praat inventory
above.

**Its caveats must travel with it.** AVQI requires the concatenated sustained-vowel-plus-continuous-
speech protocol it was validated on, and there is published sensitivity to the recording chain
including smartphone offsets — which, given
[`branch-conventions.md`](branch-conventions.md)'s capture-chain findings, is not a footnote here.

### V6 — Intensity dynamics (**not built**)

**Question.** How did intensity vary across the production?

**Computes.** The intensity contour via `extract_intensity_descriptors`
(`praat_parselmouth.py:515`), reported as range and trajectory.

**Only relative dynamics are measurable.** The recording is not calibrated and absolute SPL is not
recoverable from a file of unknown gain — and `recording_input_gain` does not recover it. Worse, AGC
in the consumer capture chain actively destroys the within-recording dynamics that are the point of
the task; see [`branch-conventions.md`](branch-conventions.md).

**Serves** `loudness` (897) and `loudness-v2` (705) on content, though both are SPEECH-declared.

### V7 — Population-conditioned F0 range (**gated behind null config**)

`_f0_range` (`voice.py:48-83`) reads `hint.metadata["population"]` against
`voice.f0_range_by_population`, which is null (`default.yaml:160`), falling back to `derive_f0_range`
over the wide search range.

**Two things are owed, and they are different in kind.** The per-population ranges are published
clinical norms, so the right move is to cite a source in `data/` with its derivation rather than
measure anything. But **`population` is not a field the contract's declaration defines** — see
Unresolved.

### V8 — Which vowel was produced (**not built**)

Available from the PPG posteriorgram. It is both a covariate — formant and perturbation measures
differ by vowel — and a conformance check on tasks that specify one. Cheap, non-normative, currently
absent.

## Deviations

| type | evidence |
| --- | --- |
| `sweep_direction_mismatch` | the F0 sweep ran opposite to the declared direction (V3) |
| `truncation` | the production began at or ran to the recording boundary |
| `repeat_attempt` | more than one phonation attempt where the task asked for one (V2) |

**`off_task_extent` is withdrawn from this branch.** It previously marked *"a region with no
phonation where the task asked for one"* — which makes the pre- and post-phonation silence around
every maximum-phonation-time effort into a deviation, and needs an undeclared minimum duration to
avoid firing constantly. **Absence of the target is a measurement, not a departure**: V2's
interruption triple already reports exactly this, with locations and durations, and reports it
better. The one correct instance of `off_task_extent` is AIRWAY's, which keys on
positively-identified off-task content.

`maximum_phonation_time` and `attempt_count` are `counts` entries, not deviations. VOICE emits no
`stimulus_mismatch` — no voice task carries a stimulus text.

## What exists today

| capability | status |
| --- | --- |
| V1 propose phonation spans | **not built** — the branch has no subject; two parameters owed |
| V2 maximum phonation time | not built; previously specified wrongly as a longest run |
| V3 F0 trajectory | 22 detectors held; branch consumption not built; search range clips both glide endpoints |
| V4 voice quality | Praat machinery complete; validity qualifiers not built |
| V5 AVQI | not built; coefficients published, so available under the no-refits rule |
| V6 intensity dynamics | not built; only relative dynamics measurable |
| V7 population F0 range | gated behind null config; declaration field undefined |
| V8 vowel identity | not built |

Reachable today: `_f0_range` resolution, `resolve_stream`, the activity write, and the
`gate_interval` tri-state (`voice.py:196-203`). Everything from `:268` on is unreachable because the
branch returns at `:266`. An earlier version of this document said *"roughly 380 of its 420 lines are
unreachable"*; the unreachable body is `:268-420`, about 153 lines, or roughly 200 counting the
helpers only it calls.

## What the branch emits

```
spans        family: "voice" phonation spans (V1)
assertions   refine (corrected_extent) where a PREPROCESS span's extent is wrong;
             label naming vowel identity (V8);
             deviate (sweep_direction_mismatch, truncation, repeat_attempt)
measurements per-span trajectory (V3), voice quality with validity qualifiers (V4),
             AVQI (V5), intensity (V6) — each carrying its extent's quality covariates
counts       maximum phonation time {found}, attempt count {found}
verdict      { spans_n, phonation_s, longest_span_s, longest_span_criterion,
               production, ambiguous_spans_n, marks_skipped_short_n, task_range,
               gate_interval, flags }
```

**The verdict's basis, exactly.** Today: `FAIL` at `voice.py:242` when no phonation span exists —
every recording. On the unreached path: `FLAG` when flags accumulated, `PASS` when spans were
measured and nothing contested (`voice.py:396-399`).

Under the contract: `FAIL` when no phonation was found; `FLAG` when one measurement contradicts
another; `PASS` otherwise — with every number carried as a measurement and no normative judgement in
the verdict.

**There is no `contest` capability.** An earlier version of this document listed *"contest where a
proposed phonation span carries no voiced frame"*, which is self-contradictory twice over: a V1 span
is by construction a maximal voiced run, so it cannot carry no voiced frame, and contesting the
branch's own proposal has no object. If VOICE is to contest anything the object must be a PREPROCESS
span; no capability here produces that, so the verb is not in this branch's emit set.

## Out of scope

Normative interpretation of jitter, shimmer, CPP, HNR or AVQI. Absolute intensity. Attribution of a
short MPT to respiratory or laryngeal cause. Any refit against declared families.

## Unresolved

- **Three declaration fields this branch reads are not in the contract's `metadata` contract**:
  V2's expected duration, V7's `population`, and V8's expected vowel. The contract froze the entry
  keys. `data/task_expectations/` does not exist yet, so extending it is free — but it must be asked
  for explicitly rather than assumed.
- Whether the `[50, 600]` clipping is better addressed by widening the fixed range or by relying on
  `derive_f0_range` per recording.
