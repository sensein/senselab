# VOICE branch

What the branch answers: **is there sustained phonation here, what are its acoustic properties, and
what did the voice do across it?**

The frame is [`../20260913-branch-contract-and-hints/design.md`](../20260913-branch-contract-and-hints/design.md);
shared rules are in [`branch-conventions.md`](branch-conventions.md); owed ground truth is in
[`branch-listening-sample.md`](branch-listening-sample.md).

## The state of this branch

**VOICE fails on every recording.** Its subject is every live span whose `family` is `phonation`
(`voice.py:230`, `_PHONATION_FAMILY` at `:40`). Nothing reachable proposes one: the detector that
did was retired on 2026-09-04, and the module's only `prov_type="span"` write (`voice.py:333-346`)
sits downstream of the no-span return at `:235-264`, which every recording takes.

**Measured on real material, 2026-09-15.** All **6** VOICE-routed recordings of a 13-recording b2ai
v3.1 run returned `Outcome.FAIL` with the retired-detector reason (`voice.py:235-239`) — including the
MPT recording carrying **15.89 s** of held phonation and the glide recording carrying **12.27 s**,
both on `voice.sustained` against a 3.0 s cut. VERDICT converts each to
`mismatch: routing routed VOICE, it found no subject` (`vocabulary.py:393-397`), making this **the
largest single flag source in the run, 6 of 13**, and the sole reason the glide recording flags at
all. `default.yaml:181` still ships `voice.hint_tags` marked *"unread as of v2"*, so the branch also
carries a dead second copy of the hint vocabulary while having no subject to apply it to. On that
evidence this is the highest-value next piece of work in the graph: it is the only finding of that run
blocked on nothing but implementation. Values in
[`benchmarks/hints-and-routing-2026-09-15.md`](benchmarks/hints-and-routing-2026-09-15.md) § D.

**And VOICE is the worked example of what the contract forbids.** `voice.py:333-346` mints a
*second* span from an input span, re-keyed by period-aligned onset and carrying `onset_kind`. That is
the re-minting the contract replaces with a `refine` assertion, and it is why `report.py:289-306`
needs `_spans_of_family` to split one family into two populations on `("onset_kind" in attributes)`.
Once the minting becomes a `refine`, the split has nothing to separate.

**Consequence for REPORT.** `onset_kind` is written only at `voice.py:341`, so on stores written
under the contract the `voice=True` reads at `report.py:715` and `:1154` return empty. REPORT's VOICE
arms need the span-versus-assertion distinction substituted in, not the family merely widened.

### VOICE refines and reviews spans, and that is its work — owner decision, 2026-09-15

The owner: *"voice would still need to improve/update/review phonation spans and other tasks that are
assigned to it."*

**The branch does not wait for a correct subject to be handed to it.** It receives spans and improves
them, which in the contract's five verbs is **`refine`** — a corrected extent, corrected metadata, or
both, asserted on an existing span, which keeps its id, its `family` and its measurements. **Only
`propose` mints.** Reviewing is `label`, `contest` or `refine`; it is never a second span.

**`refine` was extent-only until 2026-09-15**, when the owner widened it to cover a span's metadata as
well (the contract's § *`refine` covers metadata as well as extent*). For VOICE the extent half is
still the load-bearing one — the re-mint below is a boundary claim — but a VOICE assertion that
corrects what a span *is* rather than where it runs is now a `refine` too, and not a second span
either.

That is what the re-mint at `voice.py:333-346` should have been, per the paragraph above — the
decision is what makes the substitution *the branch's job* rather than a defect waiting on a proposer.
The frame is
[`../20260913-branch-contract-and-hints/design.md`](../20260913-branch-contract-and-hints/design.md)
§ *A branch refines and reviews, and a correct subject is not a precondition*.

**The inconsistency the decision resolves, stated plainly.** **VOICE is routed by a measurement on
`amplitude` spans and then fails for want of a `phonation` span.** `voice.sustained`'s feature is
`[span_longest, amplitude]` — the longest live amplitude span in seconds, cut 3.0 s
(`default.yaml:252-255`) — and the selector at `voice.py:230` admits only `_PHONATION_FAMILY`
(`:40`). The amplitude spans the route was decided on are live in the store, unexamined, when the
branch takes the no-span return at `:235-264`.

**And the second half of the owner's sentence matters as much.** VOICE must do its best on *other
tasks assigned to it* — a recording routed here whose declared family is not voice-eliciting, which
§ *A recording routed here whose declared task is not voice* below records as 22,277 routed against
8,306 declared. The standing rule: **a branch receiving a task outside its declared families does its
best to find, mark, refine or refute evidence of its own kind, rather than failing for want of a
declared subject.** So **a `FAIL` reading "no span of my family" is the wrong shape of answer for such
a recording** — the right answer is the annotations VOICE made on the spans it was given, and `FAIL`
is reserved for having looked and found no attempt (§ *A branch `FAIL` is an absence of detected
content*).

### The subject is the spans the ruleset labelled, and VOICE refines them — settled, 2026-09-15

**The owner has settled the flow.** A ruleset that fires **may write or refine a span's label**
([`family-taxonomy-ruleset.md`](family-taxonomy-ruleset.md) § *A fired rule may write or refine a
span's label*, and
[`../20260913-branch-contract-and-hints/design.md`](../20260913-branch-contract-and-hints/design.md)
§ *A ruleset that fires may write or refine a span's label*), and a branch **refines and reviews**
the spans it is given — a correct subject is not a precondition, and `refine` asserts a corrected
extent, corrected metadata or both without minting a new span (§ *VOICE refines and reviews spans*
above). **So VOICE's subject is the spans the ruleset labelled, and VOICE refines them.** A
`consensus_taxonomy` rework and a new phonation detector are no longer open questions about where the subject comes from; what is left is
engineering, listed below. **It costs no new model pass and no new detector**: the subject is the
same evidence the route was taken on.

**The one real obstacle is the family scoping, and it is unresolved.**
[`branch-conventions.md`](branch-conventions.md) § *`propose` versus `refine` — scoped by family*
rules that a branch `refine`s only a span of the family it proposes into. `voice.sustained`'s
feature is `[span_longest, amplitude]` (`default.yaml:252-255`), so the span the route was decided
on is an **`amplitude`** span, and a ruleset label on it **does not by itself make it refinable by
VOICE**. Either the label sets the span's `family` — a writer touching a span PREPROCESS minted — or
the scoping widens for the four annotating verbs while staying for `propose`. **Unresolved**, and
named at [`branch-conventions.md`](branch-conventions.md) § *The two owner decisions of 2026-09-15
leave the minting rule alone and open one question*, which is where it is decided.

**Two code changes the flow needs and does not have.** The ruleset records neither the gate's
**value** nor the **span identity** it fired on, so a label cannot yet be written onto *the* span
that fired: `evaluate_gate` returns an enum and drops `gate_value`'s number
(`routing_analysis/ruleset.py:405-409`), and `span_longest_s[measure] = max(durations)`
(`features.py:1134`) drops the `"id"` its own rows carry (`:1084`). **Owed a code change**, both.

**The code and two documents still name the retired plan.** The no-span `why` at `voice.py:237-238`
says the branch is *"pending a rework onto `consensus_taxonomy`"*, and `config-derivations.md:502`
and `taxonomy.md:95` carry the same plan. The plan is superseded; the string and the two references
are **owed a correction**. A third reference is stale for an unrelated reason:
`config-derivations.md:499-503` describes a permanent-`uncertain` `voice` line making TAXONOMY's fold
unable to reach `FAIL` or `PASS`, and that mechanism was **deleted in ruleset stage 2** — nothing in
`src/senselab` reads a `presence_floor` any more ([`dag.md`](dag.md), *"The presence-floor path is
gone from the code"*).

**What the evidence says.** `voice.sustained` read **15.89 s** on the MPT recording and **12.27 s**
on the glide (§ *The state of this branch*), so the amplitude spans the settled flow hands VOICE are
real material on exactly the recordings this branch exists for. The consolidated classifier labels on
that same 20 s held vowel read `Chant` **0.937**, `Music` **0.930**, `Mantra` **0.899** and
`Brass instrument` **0.661**, every one outranking anything voice-specific
([`benchmarks/hints-and-routing-2026-09-15.md`](benchmarks/hints-and-routing-2026-09-15.md) § G) —
which is measured evidence against ever sourcing the *label* from that channel, and it is why the
settled flow does not. **Thirteen recordings license no threshold**, whatever the source.

**V1 keeps its specification and gains a second role.** The ground-truth rule in
[`../20260913-branch-contract-and-hints/design.md`](../20260913-branch-contract-and-hints/design.md)
§ *Refitting `spans.k_db` was considered and rejected* forbids fitting a detection boundary against
declared families, so a detector of the branch's own stays blocked on listening
([`branch-listening-sample.md`](branch-listening-sample.md)). V1 is envelope-first and qualified by
voicing, so **it is the refiner of the spans VOICE is handed**, not only a proposer — unblocked as a
capability in the role the settled flow gives it.

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
by `phonation_tracks` (`preprocess.py:931`, called at `:2152`), which runs `f0_track` over the
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

**The inputs exist; the statistics do not, and they are owed.** `formant_track` is already in
`phonation_tracks`, so formant stationarity costs no new extraction — but that is true of the
**input** and false of the **statistic**. Nothing named `spectral_flux` or `stationarity` exists
anywhere in `src/senselab`, and none of the three has a named statistic (standard deviation? a trend
test? a windowed variance ratio? frame-to-frame spectral distance under which metric?), a window, or
a config key.

[`branch-conventions.md`](branch-conventions.md) rules that a measurement with no stated window is
comparable to nothing, so **all three are owed a statistic and a window, declared as conventions.**
This is the branch's foundational capability and it separates a sustained vowel from connected speech
across 22,277 routed recordings, so the gap is load-bearing rather than a detail.

**Steadiness is a covariate, not a gate.** V4 carries it on every perturbation and CPPS value rather
than V1 refusing to propose — the same discipline as the rest of the inversion.

**But the covariate and the value it qualifies are currently computed on different signals**, which
undermines the pairing. `phonation_tracks` runs `f0_track` on **pre-emphasised**, `formant_track` on
**plain**, and derives the range on **plain** (`preprocess.py:955`) — while every V4 value comes from
**enhanced** (finding 0). A steadiness qualifier measured on one signal cannot certify a perturbation
value measured on another.

**And F0 is tracked on a pre-emphasised signal**, which [`branch-ddk.md`](branch-ddk.md) D1 argues
carefully is wrong for the envelope modulation spectrum without the analogous argument ever being
made for autocorrelation pitch tracking. +6 dB/octave attenuates the fundamental relative to the
upper harmonics, **raising octave-error risk upward** — worst on low-F0 and creaky voices — and
Praat's guidance is to track pitch on the unmodified signal. The range is derived on `plain` and
applied to `preemphasised`, which is structurally the same mismatch the audit condemns for the
jitter form defaults, here in the supposedly clean path.

**And `voice_tracks.npz` carries unmasked sentinels.** `voice.py:372` writes `hnr_db` with
Praat's −200 dB undefined-frame markers unmasked — measured **389 sentinel frames** in a padded 2 s
signal — while `phonation.hnr_floor_interval_db` is null so nothing masks them downstream. Any mean
or percentile over that array is destroyed. [`praat-instrument-audit.md`](praat-instrument-audit.md)
finding 10, which also records the useful negative result that
`extract_harmonicity_descriptors`' own `Get mean` *excludes* them.

**An absent `phonation_tracks` is not an empty one.** `derive_f0_range` **raises**
`F0RangeUnavailable` (`tasks/phonation/api.py:41`, raised at `:93-97`) when no range could be derived
from the recording, so on a frankly aperiodic recording the whole measurement is *absent* rather than
present-and-null. The "F0 availability" qualifier must handle an absent measurement, which is the
case it most needs to describe.

**And an absence and a crash are not yet distinguishable at that boundary.** `extract_pitch_values`
now separates them in its return — `pitch_failed` is 1.0 when the analysis itself raised and 0.0 when
the wide search simply placed no pitch (`praat_parselmouth.py:370-378`, set at `:504` and `:473`) — but
`derive_f0_range` reads only the two range values, so both arrive as the same
`F0RangeUnavailable`. Surfacing the distinction is **owed**, as Task 2 of
[`../20260914-f0-range-and-measurement-streams/plan.md`](../20260914-f0-range-and-measurement-streams/plan.md).
Until it lands, a V1 qualifier reading "no F0 was available" cannot say whether the instrument failed
or the voice was aperiodic — and those are opposite findings.

**Emits.** `propose` spans, `family: "voice"`, subject to the `propose`/`refine` rule in
[`branch-conventions.md`](branch-conventions.md), each carrying the three qualifiers above.

**Parameters.** The claim that this is "parameter-free in its core" is withdrawn — it was false under
the old definition and is still false under the new one:

1. **The envelope, and the threshold that goes with it — both owed.**

   **V1 reads a linear envelope on `plain`, not the stored pre-emphasised dB one.** Two of
   [`branch-ddk.md`](branch-ddk.md) D1's three reasons for refusing that envelope transfer verbatim:
   **+6 dB/octave attenuates the F0 region** where quiet, low-pitched or breathy sustained phonation
   carries most of its energy relative to the broadband floor, so a 6 dB-over-floor test
   **under-detects exactly quiet low-F0 phonation**; and a dB envelope is a nonlinear transform of
   the quantity being thresholded. **That is the cause of the residue V1 names below** — "a
   phonation too quiet to clear the envelope threshold still yields no span".

   **`spans.k_db: 6.0` does not transfer to that envelope, and is owed a re-derivation.**
   `config-derivations.md:106-120` derives 6 dB for the **pre-emphasised** primary pass, and the
   floor derivation immediately above records a **measured** instance of exactly this substitution
   failing: reading the rise statistic off a different signal definition shifted it by ~19.8 dB,
   which "put ordinary background noise above both `spans.k_db` values with no real event present at
   all".

   The physics runs the way that warning describes. `config-derivations.md:35-38` measures
   pre-emphasis raising event-to-floor contrast by **+7.36 to +10.95 dB** on the hardest events, so
   removing it does the reverse for low-frequency energy: on `plain` a 6 dB test becomes more
   sensitive to quiet low-F0 phonation, **which V1 wants**, and simultaneously to HVAC rumble,
   handling noise, breath puffs and DC drift, **which the tilt was suppressing.** Over-proposed
   attempts on room rumble flow straight into V4 across 22,277 routed recordings.

   [`branch-ddk.md`](branch-ddk.md) D1 makes the identical move for `envelope.lowpass_hz` and files
   it honestly as a value derived under one condition applied under another. This does the same.

   **`spans.min_duration_ms: 50` is conventional (`config-derivations.md:234`) and does transfer.**
   What V1 may still need beyond it is a *minimum sustained* duration distinguishing an attempt from
   an ordinary span — a different quantity, also owed;
2. Praat's own undeclared internal voicing threshold inside `to_pitch_cc`, **and the range handed to
   it**. The range is no longer the worse half of that pair: since 2026-09-14 `derive_f0_range`
   narrows per recording from the declared wide search, and the five coefficients it narrows by are
   config keys with written derivations (V3 below). **The voicing threshold inside `to_pitch_cc` is
   still undeclared**, and it is what decides which frames the percentiles are taken over, so it
   conditions the derived range as well as the contour;
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
`voice.task_duration_ranges` is null (`default.yaml:162`) and `_task_range` (`voice.py:129-161`)
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

**`derive_f0_range` narrows per recording, as of 2026-09-14.** `extract_pitch_values`
(`praat_parselmouth.py:381`, the rule at `:465-497`) runs one wide autocorrelation pass over
`voice.f0_search_range_hz` and then narrows off the linear-Hz percentiles of that pass's voiced
contour:

```
floor    = max(search_floor,  p5 / pitch_floor_divisor)
ceiling  = min(search_ceiling, max(pitch_ceiling_quartile_multiplier * q3,
                                   pitch_excursion_multiplier * p95))
fallback = the unnarrowed search range, when p95 < pitch_pinned_octave_ratio * search_floor
```

All five coefficients are `praat_features.pitch_*` config keys with their derivations written beside
them in [`config-derivations.md`](config-derivations.md), and one reader supplies them to all three
call sites — `praat_features` (`preprocess.py:877`), `phonation_tracks` (`:951`) and VOICE's
`_f0_range` (`voice.py:71`) — through `f0_range_parameters` (`nodes/common.py:357`), so no two can
hold values that drift. The rule, its measurement and what it does not reach are in
[`praat-instrument-audit.md`](praat-instrument-audit.md) step 2.

**What follows from the derivation, for this capability:**

- **The binding constraint is now the outer search bound, not the derivation.**
  `2.5 × q3` reaches the 600 Hz search ceiling for any q3 ≥ 240 Hz, so above roughly that F0 every
  recording gets the search bound rather than a narrowing. Whether 600 Hz is the right bound is the
  audit's step 4 owed item, where the same number is contested against the CPPS band's 700 Hz; it is
  not re-argued here.
- **The floor is never below the declared search floor**, 50 Hz, so a 45 Hz fry places no pitch and
  is an absence — but an absence against a **declared** bound, which a reader can look up, rather
  than against an undeclared one.
- **A recording whose contour sits within an octave of the search floor takes the wide range**, and
  `pitch_range_fell_back` says so. That is a larger population than the low-frequency-hum capture it
  was designed for: at a 50 Hz floor it is every voice whose p95 is under 100 Hz. Those recordings
  lose the octave-error robustness narrowing buys.
- **The range is still derived twice, independently** — `preprocess.py:955` and `voice.py:71`. They
  now read one set of coefficients, but they run on separate calls over the same stream, so they
  agree by construction of the rule rather than by sharing a result.
- **On a glide the derived ceiling is near-vacuous rather than circular — corrected 2026-09-14
  against a measurement.** This bullet read that the sweep decides its own analysis ceiling, making
  V3's conformance flag *partly circular*, and that the audit records narrowing as *actively wrong on
  a sweep*. **Measured, narrowing does not clip a glide**: an exponential 100→400 Hz sweep derives
  `[72.8, 600.0]` against produced extremes of 102 and 392 Hz
  (`src/tests/audio/tasks/features_extraction_test.py:373-381`), and the retired bin capped a
  low-binned upward glide at 250 Hz, so this is strictly better than what it replaced. What
  degenerates is **the flag**: "did F0 reach the derived limit" can now only fire when the margin
  pushes past the 50/600 clamps, so on the 3,150 glide recordings it is **near-vacuous**. **That is
  this capability's live open item** — a defect in the flag, not a reason to condition the range on
  the task, which the measurement does not support. See the audit's step 2, corrected on the same
  date.

**The conformance flag therefore needs three levels**, not two: the F0 extremum coincided with the
**derived limit**, with the **outer search bound**, or with neither. Report which, and report the
derived range itself and `pitch_range_fell_back` beside it — a derived limit and a fallback to the
search bound are different facts about the recording, and above q3 240 Hz they are the same number.

**History — what this replaced, and why it must not read as current.** Until 2026-09-14
`extract_pitch_values` took the recording's ±2 SD-trimmed mean pitch and selected one of two
hardcoded pairs: below 170 Hz → floor 60, ceiling 250, commented *'male' settings*; otherwise floor
100, ceiling 500, commented *'female' and 'child' settings*. That bin, not the search range, was what
bound every F0-derived measure: the ceiling truncated an ordinary male upward glide, the floor
clipped downward glides into fry well above the stated bound, and every F0-derived measure in the
corpus carried a **discontinuity at 170 Hz** along a sex-typed boundary — a measured
**1.67× step** in the intensity window, the harmonicity window and the minimum analysable segment —
in a corpus that deliberately enrols gender-diverse participants. `capability-map.md:117` and `:314`
had already named that bin as the exact mistake not to repeat, and the code did it anyway. **The
scalars already written into 60,202 stores were computed under it** (audit step 0), so the diagnosis
governs every existing artifact even though it governs no new one. It is 60,202 and not the corpus
because the pass that wrote them ran over a 60,202-row manifest and 2,376 stores hold no Praat
measurement at all ([`../20260911-ppg-praat-batch/design.md`](../20260911-ppg-praat-batch/design.md)).

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

### V4 — Voice quality (**no trustworthy descriptor exists today**)

**State this once, plainly: VOICE has no trustworthy voice-quality descriptor at all until audit
steps 0–4 land.** Not "machinery exists, consumption not built" — that badly understates it. Every
candidate is compromised at the instrument:

| descriptor | state |
| --- | --- |
| CPPS | **suppressed** — `> 4` cut, 330 Hz cap, vuv inflation, unweighted mean |
| HNR | unmasked −200 dB sentinels (finding 10) **and** an analysis window set by the per-recording F0 floor and never stated beside the value — a 75 → 45 ms step at the 170 Hz bin until 2026-09-14, a continuum since (finding 1) |
| jitter, shimmer | **withheld** pending a bench measurement — time-resolution-unvalidated at 16 kHz, see below |
| slope, tilt | inherit the **same point-process failure** as jitter and shimmer (below) |



**Question.** What are the noise and perturbation properties of the phonation?

**Computes.** All in `tasks/features_extraction/praat_parselmouth.py`: `extract_cpp_descriptors`
(`:706`), `extract_harmonicity_descriptors` (`:580`), `extract_jitter` (`:1112`), `extract_shimmer`
(`:1168`), `extract_pitch_descriptors` (`:448`), `extract_slope_tilt` (`:638`),
`measure_f1f2_formants_bandwidths` (`:824`), `extract_spectral_moments` (`:945`).

**CPPS is the primary descriptor as a method — but this implementation is voiced-interval-gated,
which removes the property that motivated promoting it.** `extract_cpp_descriptors` builds a
voiced/unvoiced TextGrid from `to_pitch_ac(..., voicing_threshold=0.3)` and computes CPPS only over
intervals labelled `V` (`praat_parselmouth.py:807-821`). So CPPS is **unavailable on exactly the
aperiodic voices V1's inversion exists to rescue** — the claim that it needs no period extraction is
true of the method and false of this code path.

**And its cepstral peak search is hard-capped at 60–330 Hz** (`praat_parselmouth.py:837-838`), so
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
cannot see how much was deleted.

**What replaces it is step 4 of [`praat-instrument-audit.md`](praat-instrument-audit.md)'s
remediation path** — a direct CPPS implementation with no value cut, no interval gating, duration
weighting, a support count, both smoothing windows declared, and a **fixed wide peak-search band of
60–700 Hz**.

**Not a band derived from the recording's own F0.** An earlier version of this paragraph said that,
and step 4 retracts it for three reasons — chiefly that `derive_f0_range` **raises** on a type-3
voice, so a per-recording band would be unavailable on exactly the population the reimplementation
exists to serve. Until that exists, **V4's primary descriptor is unavailable** and the branch says so rather
than reporting a number it has just described as untrustworthy.

**Three further defects in the existing function**, all in the audit: CPPS is averaged **unweighted**
across intervals, so finding 3's short padded intervals dominate a dysarthric recording's mean;
`std_dev_cpp` is a **between-interval** SD carrying a name that reads as the published
within-recording quantity; and `voicing_threshold=0.3` against Praat's 0.45 **compounds** the vuv
inflation rather than being independent of it.

Perturbation is secondary, and **withheld** pending the bench measurement — see below.

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

**State it as plainly as CPPS's state is stated: no type-2 instrument works today.** Modality
provably fails in the case it exists to catch (below); the octave-jump count cannot separate period
doubling from the tracker's own octave error on the same material; the subharmonic-to-harmonic ratio
is absent from the inventory; and Sample C's two-floor substitute is confounded (see
[`branch-listening-sample.md`](branch-listening-sample.md)). **So perturbation's type-2 gate has no
working instrument, which is a second reason jitter and shimmer are withheld.**

**The modality test fails in the case it is meant to catch.** If the tracker locks to the
subharmonic, every mark is a doubled period and the distribution is **unimodal at 2T**. So
bimodality is evidence *of* type 2; its absence is not evidence against it. The direct instrument is
a subharmonic-to-harmonic ratio, and **it is not in the inventory** — stated rather than assumed
away.

**Type 2 does not reach the perturbation measurement at all, and the audit corrects how.** It is not
the 1.3 maximum-period factor: measured, `To PointProcess (periodic, cc)` places **0 pulses** on an
alternating-period signal and **1** on a period-doubled one, so the factor never gets to act. And
the floor is now `max(search_floor, p5 / 1.5)`, so at the shipped 50 Hz search floor a 45 Hz source
still yields **zero pulses** — vocal fry and Parkinsonian creak below the search floor are excluded
upstream. What changed on 2026-09-14 is *which* bound excludes them: the retired bin put the floor at
60 Hz, which also took ordinary low male voices, and did it undeclared; the exclusion is now at the
**declared** `voice.f0_search_range_hz` floor, which a reader can look up and an operator can move.
That does not make the exclusion smaller at 45 Hz. It makes it attributable.

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

**And HNR's validity role inherits whatever the F0 floor does.** Praat's harmonicity window is
`periods_per_window / f0_min`, so it is set by the derived floor. Under the retired bin that was a
**step** — 75 → 45 ms across the 170 Hz boundary, a discontinuity along a sex-typed line. Since
2026-09-14 the floor varies continuously per recording, so the step is gone and what remains is a
**continuum**: two recordings still get different HNR windows, just not two classes of them. The
consequence for this block is unchanged in kind and weaker in degree — using HNR as the evidence
deciding whether a perturbation value is valid still **imports the window difference into the
validity judgement itself**, not merely into the measurement, so the window length belongs beside
the HNR value per [`branch-conventions.md`](branch-conventions.md)'s stated-window rule.

**Two covariates specific to this block**, beyond the shared set in
[`branch-conventions.md`](branch-conventions.md):

- **sample rate — and the correction here is worse than the original claim.** An earlier version
  said "this corpus has mixed rates", which is true of the *files* and false at the point of
  measurement: `resample.target_hz: 16000` means **every stream any of these measurements sees is
  16 kHz mono**. So the time-resolution floor on jitter is **uniform across the corpus** — not a
  between-recording covariate but a constant.

  **The arithmetic, corrected.** The step-to-period ratio is the wrong quantity. With independent
  uniform pulse-placement error Δ = 62.5 µs, the induced **local-jitter floor** is
  Δ·√(6/12)·√(2/π)/T ≈ **0.56 Δ/T — about 0.42% at F0 120 Hz and 0.88% at 250 Hz.** An earlier
  version gave 0.75% and 1.6% and said both were "at or above" the 0.2–1% normal range; they sit
  **inside** it. That does not weaken the conclusion — **a floor comparable to the measurand is
  fatal** — but it is the number the bench measurement is specified against.

  **Shimmer needs its own bound, and the timing argument does not give it one.** Shimmer is an
  *amplitude* measure; its sample-rate floor comes through uninterpolated peak-amplitude picking at
  roughly four samples per cycle of a 4 kHz component. **The bench measurement must synthesise known
  shimmer as well as known jitter.**

  *Effective bandwidth does not substitute* — that is frequency content, this is time resolution.
- **segment SNR.** Additive noise inflates both measures.

**Name the variant.** `local` jitter and shimmer are sensitive to slow drift; `ppq5` and `rap` much
less so. On a vowel with vibrato the two diverge materially while both being reported as "jitter".
Emit the variant name with the value, never a bare "jitter".

**Declare an analysis-window convention** excluding attack and decay, per
[`branch-conventions.md`](branch-conventions.md).

**Qualification is the rule; jitter and shimmer are the exception, and are suppressed.** The word
"gated" was used loosely in an earlier version while the operative instruction was "report
alongside" — those are different designs. The rule here is **qualification**: a value is emitted with
the evidence a reader needs to discount it.

**But jitter and shimmer are withheld pending the bench measurement above**, and an earlier version
said "must be measured before jitter is published at all" and then emitted it three paragraphs later.
**That is exactly the incompatibility this section diagnoses for CPPS one page earlier** — saying the
right words and shipping the thing anyway — reproduced for a different measure. A floor that may be
comparable to the measurand is not something a covariate discounts.

So: **CPPS withheld pending audit step 4; jitter, shimmer and HNR withheld; everything else
qualified.**

**HNR is withheld for its own reason**, stated here rather than left implicit in the emit block: it
carries unmasked −200 dB sentinels (`voice.py:372`, audit finding 10) **and** its analysis window is
set by the per-recording F0 floor, so it varies between recordings and is never stated beside the
value. The first is a code fix. The second stopped being a 75 → 45 ms step at the 170 Hz bin on
2026-09-14 and is now a continuous per-recording quantity — which makes it reportable rather than
merely wrong, so the second half of the withholding is discharged by emitting the window, not by
fixing the code.

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

**It also explains a V4 observation — for some variants, and the exception matters.** A 4–8 Hz
modulation makes `local` diverge from the smoothed variants because the short-window measure tracks
the modulation while the longer-window one averages across it.

**The inversion is a shimmer statement, and it is broader than a previous version said.** For a
sinusoidal perturbation of period *P* cycles the local response is 2·sin(π/P) while the 11-point
residual is |1 − D₁₁(1/P)|; the two **cross at P ≈ 30 cycles**. That is a property of the **11-point
smoother**, and the code exposes `local / localDB / apq3 / apq5 / apq11 / dda` for shimmer and
`local / localabsolute / rap / ppq5 / ddp` for jitter (shimmer at `praat_parselmouth.py:1261-1266`,
jitter at `:1205-1209`) — **there is no
`ppq11`**, so the claim as an earlier version wrote it crossed the two families, and **within jitter
no available variant ever crosses `local`.**

**And the boundary sweeps a wider range than stated.** P ≈ 30 means F0 ≈ 30 × f_tremor, so across the
4–8 Hz tremor band it runs **F0 120–240 Hz** — at the top of the band that covers most adult female
voices too, not "most adult male voices in this corpus".

So an `apq11`-versus-`local` divergence cannot be read as a tremor signature **without stating both
F0 and the tremor frequency**, and the direction flips across a boundary that moves with the tremor
being measured.

**Emits.** A per-span measurement: modulation frequency, modulation depth for F0 and for amplitude,
and the search band as a declared convention.

### V5 — Composite severity indices: **moved, and the conclusion is stronger than "the owner decides"**

**The Acoustic Voice Quality Index** (Maryn, Corthals, Van Cauwenberge, Roy & De Bodt, 2010) is a
weighted combination of six measures — **CPPS, HNR, shimmer local, shimmer local dB, LTAS slope and
LTAS tilt** — reported on a 0–10 severity scale with a published cut-off near 2.95. This document is
its home in the set, so the components and the citation live here;
[`corpus-level-node.md`](corpus-level-node.md) C2 defers to it.

AVQI and its relatives are **not per-recording capabilities here** — the protocol requires a
sustained vowel *and* continuous speech concatenated, which in this corpus are different recordings.
The capability sits in C2, as a **session-level** capability.

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

**Dispersion on v1 is n = 3**, with a warm-up or fatigue trend that three points cannot separate from
random variation.

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

#### Shouting moves the derived F0 range, which confounds the comparison

Maximal effort raises F0 by roughly 3–8 semitones, and `derive_f0_range` takes its percentiles from
the recording's **own contour** (V3). So a speaker's `loudness` recording and his `prolonged-vowel`
recording are analysed under **different derived ranges, set by the thing being compared** —
different floors, different ceilings, hence different intensity, harmonicity and minimum-segment
windows on the two halves of the contrast.

**This was sharper under the retired 170 Hz bin and is not gone with it.** The bin turned the
difference into a discrete jump — a male speaker at a comfortable 120 Hz shouting at 170–190 Hz had
his two recordings analysed under **60/250** and **100/500**. The derivation replaced that jump with
a continuous drift, which is smaller and no longer sex-typed, but it is still **an instrument
parameter moving with the measurand**. The within-participant comparison these correlates exist for
is confounded by the instrument, not by the voice.

**The fix available today is to derive one range and apply it to both recordings** of a participant's
contrast, which is a session-level operation and belongs with C3 rather than here. And on v2 both
conditions are in one file, so one derivation already covers both — that is a second reason v2's
within-file contrast is the better measurement.

#### Name it for what it is

An earlier version's table said **"maximum vocal intensity"** for an uncalibrated, gain-unknown,
possibly clipped level — while V2, in the same document, refuses to emit `maximum_phonation_time` for
exactly this hazard class. Per [`branch-conventions.md`](branch-conventions.md)'s naming rule, the
value carries its convention: relative peak level within the recording, with the clipping state
attached. **Absolute SPL is not recoverable** from a file of unknown gain.

**Do not build any of this on `range_ratio_intensity_db`.** `praat_parselmouth.py:625` computes it as
`max_dB / min_dB` — measured **1.000** on a buzz with no silence, **−0.244** once silence is added. It
tracks silent fraction, not dynamic range. [`praat-instrument-audit.md`](praat-instrument-audit.md)
finding 6.

#### The effort correlates are session-level

**`loudness-v2`'s contrast stays here, and only v1's goes to the session.** An earlier version moved
"the correlates" wholesale to C3, which threw away the better measurement: **v2 has both conditions in
one file**, matched on vowel, duration, mic distance, gain state and one derived F0 range. That is a clean
within-file contrast, and C3's session-level reference is confounded on every one of those axes. The
table above already said v2 has two conditions; the later section contradicted it.

So:

| | per-recording (V6) | session-level (C3) |
| --- | --- | --- |
| `loudness-v2` (705) | **the full contrast** — normal against shouted, within file | — |
| `loudness` (897) | count, timing, dispersion across three attempts | the effort correlates, against `prolonged-vowel` |

**Only v1 needs C3**, because only v1 lacks a comfortable-effort condition.

**Owed.** A pause criterion for separating the events.

**A consequence for SPEECH.** Both families carry `speech_type: "non-lexical"` while `families.py`
places them in `LEXICAL_SPEECH` — see [`branch-speech.md`](branch-speech.md).

### V7 — Population-conditioned F0 range (**gated behind null config; and it should stay that way**)

`_f0_range` (`voice.py:49-81`) reads `hint.metadata["population"]` against
`voice.f0_range_by_population`, which is null (`default.yaml:160`), falling back to `derive_f0_range`
(`voice.py:71`).

**The key should stay null, and as of 2026-09-14 the reason the previous two revisions gave is
finally the true one.** The fallback is now a genuine per-recording derivation (V3), so the choice
this key poses is the one it looks like: an explicit declared population prior against deriving the
range from the recording in hand.

**A middle revision of this document argued the opposite, and its premise is gone.** It held that the
fallback *was itself* a population prior — implicit, undeclared, binary and sex-typed, two hardcoded
pairs selected at a 170 Hz boundary — so that the real choice was between an implicit prior and an
explicit one, and the implicit one was worse in every respect except that nobody had to write it
down. **That was true of the code until step 2 landed and is true of nothing now.** It is kept here
because it is why the key was reopened, and because the scalars already in the corpus were computed
under the prior it describes.

**The hazard the first version named is real and now applies only to the declared key.**
Conditioning an F0 analysis range on population clips a speaker whose F0 sits outside their
population's typical range — puberphonia, a trans-feminine voice — and this protocol deliberately
enrols gender-diverse participants. Populating `voice.f0_range_by_population` would reintroduce by
declaration exactly what the retired bin did by accident.

**The alternative that needs no norm is what the code now does**: a wide declared search, narrowed
per recording, plus V3's octave-jump diagnostics and the three-level conformance flag. That reports
when the range was inadequate rather than assuming one from an inferred demographic.

**So there is no longer a defect here to fix.** `voice.f0_range_by_population` being null is the
correct state, and `derive_f0_range` no longer supplies a prior behind it. What survives as owed is
V3's glide item — **the conformance flag is near-vacuous on a sweep**, corrected 2026-09-14 from
"narrowing is task-conditioned", which a measurement did not support — and the search bound the
narrowing sits inside.

**`population` is not a field the contract's declaration defines** — see Unresolved.

### V8 — Which vowel was produced (**not built**)

Both a covariate — formant and perturbation measures differ by vowel — and a conformance check on
tasks that specify one.

**Use formants, not the PPG.** An earlier version proposed reading vowel identity from the PPG
posteriorgram; by DDK's own argument the posteriorgram is out of domain on sustained productions, and
`measure_f1f2_formants_bandwidths` (`praat_parselmouth.py:883`) is the direct instrument — F1 and F2
are what vowel identity *is*.

**The formant configuration is adult-male, and which function it configures matters.**
`phonation_spans.formant_max_hz: 5000.0` with `max_formants: 5` configures **`formant_track`**
(`preprocess.py:965-972`), which is the function V8 should use. Praat's guidance is roughly 5500 Hz
for an adult female voice and higher for children, so formant estimates carry a sex- and
age-structured error. **That one is a configuration question.**

`measure_f1f2_formants_bandwidths` is a **different** function and a different problem: the wrapper
forwards only four of its eight parameters (`praat_parselmouth.py:1424-1429`), so
`maximum_formant_hz` is **unreachable from any caller** — a code defect, not a configuration choice.
An earlier version of this section cited that function alongside the `phonation_spans` key, which
conflated the two.

**And LPC formant estimation degrades with F0 regardless of configuration.** At F0 250 Hz the
harmonic spacing undersamples the spectral envelope and F1 for a close vowel like /i/ is essentially
unrecoverable. So **V8 fails systematically on high vowels and high-F0 speakers** — and the same
limitation lands on **V1's formant-stationarity qualifier**, which will read "unsteady" for high-F0
speakers for tracker reasons rather than production ones.

**V8 has no decision rule, and that is an owed operating point.** F1 and F2 are the *correlate* of
vowel identity, not the label. Mapping a formant pair to a category needs a boundary in a space that
**scales with vocal tract length**, and there are no norms here to place one.

**A session-level formant normalisation would work** — the participant's own connected speech
supplies the scaling — and it uses the grouping [`corpus-level-node.md`](corpus-level-node.md)
already needs for C2, C3 and [`branch-quality.md`](branch-quality.md) Q2.

**V8 is a precondition for the composite indices**, not an optional covariate: AVQI's protocol
requires the sustained vowel to be /a/, so without vowel identity the pair cannot be assembled. It is
a precondition for **C3** as well — see there, where it does not by itself suffice.

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
| V1 propose the attempt | **not built** — the branch has no subject; reads a linear envelope on `plain`, so **`spans.k_db` is owed a re-derivation**; `min_duration_ms` transfers; stationarity qualifiers required |
| V2 maximum phonation time | not built; previously specified as a longest voiced run |
| V3 F0 trajectory | 22 detectors held; branch consumption not built; `derive_f0_range` narrows per recording since 2026-09-14, so **the binding constraint is now the 600 Hz search bound**. Measured, the narrowing **brackets a glide rather than clipping it**, so no task condition is owed; what is owed is **the conformance flag**, which the clamps leave near-vacuous on a sweep |
| V4 voice quality | **every Praat scalar is computed on FRCRN-enhanced audio** (audit finding 0); **CPPS, jitter, shimmer and HNR all withheld** — see the descriptor table above |
| V4a vocal tremor | **not built**; largest missing capability; nearly free once DDK D1 exists |
| V5 composite severity | **moved** to [`corpus-level-node.md`](corpus-level-node.md) C2, session-level; the protocol is unsatisfiable here |
| V6 vocal effort events | not built; event detection and dispersion stay here, the effort correlates are session-level |
| V7 population F0 range | null key, and correctly so: `derive_f0_range` now derives per recording rather than supplying an undeclared prior behind it |
| V8 vowel identity | not built; use formants; configuration is adult-male; precondition for C2 |

Reachable today: `_f0_range` resolution, `resolve_stream`, the activity write, and the
`gate_interval` tri-state (`voice.py:194-201`). Everything from `:266` on is unreachable because the
branch returns at `:264` — about 153 lines, or roughly 200 counting the helpers only it calls.

## What the branch emits

```
spans        family: "voice" phonation attempts, each carrying voiced fraction,
             interruption structure and F0 availability (V1)
assertions   refine (corrected_extent, corrected_attributes, or both) where a
             PREPROCESS span's extent or metadata is wrong;
             label naming vowel identity (V8);
             deviate (sweep_direction_mismatch, truncation, repeat_attempt)
measurements per-span trajectory with the derived F0 range and whether it fell
             back to the search bound (V3), vocal tremor
             (V4a), effort-event level and dispersion with clipping state (V6)
             — each with its extent's covariates, steadiness qualifiers and
             support count.
             WITHHELD: CPPS (pending audit step 4); jitter and shimmer
             (pending the bench measurement); HNR (unmasked sentinels)
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

**The verdict's basis, exactly.** Today: `FAIL` at `voice.py:240` when no phonation span exists —
every recording. On the unreached path: `FLAG` when flags accumulated, `PASS` when spans were
measured and nothing contested (`voice.py:394-397`).

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

- **Whose family a ruleset-written label puts a span in** — the source itself is settled (§ *The
  subject is the spans the ruleset labelled, and VOICE refines them*), and a branch `refine`s only a
  span of the family it proposes into, so a label on an `amplitude` span does not by itself make it
  refinable here. Either
  the label sets the `family` or the scoping widens for the annotating verbs; decided at
  [`branch-conventions.md`](branch-conventions.md) § *The two owner decisions of 2026-09-15 leave the
  minting rule alone and open one question*. The two prerequisites the settled flow still needs —
  the gate's value and the fired span's identity — are each **owed a code change**.
- **Whether to emit an AVQI-shaped number that is not AVQI** (V5). The protocol is unsatisfiable in this corpus, so this is the actual question; the owner decides it, but not as an open choice between AVQI and nothing.
- **Three declaration fields this branch reads are not in the contract's `metadata` contract**:
  V2's expected duration, V7's `population`, V8's expected vowel. The contract froze the entry keys;
  `data/task_expectations/` does not exist yet, so extending it is free but must be asked for.
- Whether the MPT extent is reported as an upper bound or intersected with voicing evidence.
- What `report.py:104` and `voice_test.py:344-352` read once `longest_span_s` is retired.
- `phonation_spans.formant_max_hz: 5000.0` is adult-male; affects V4 and V8.
- **The Praat wrapper audit is complete** — [`praat-instrument-audit.md`](praat-instrument-audit.md). **Finding 0 — every Praat scalar is measured on FRCRN-enhanced audio — is provenance for this whole branch, not a defect in it.** Its remediation **step 1 was withdrawn by the owner on 2026-09-14**: the mechanism it rested on is false, so no repair here is ordered behind a stream switch. The ordering claim that used to sit on this line — *"its remediation steps 0 and 1 come before any other repair here"* — is void for step 1; step 0 still comes first, on the retired F0 bin and the wrapper defects rather than on the stream. Whether `enhanced` or `plain` is the better stream for these scalars is **unmeasured** and recorded as future research in [`branch-listening-sample.md`](branch-listening-sample.md).
- Whether V6's effort correlates are built at all, given they are session-level and v1 supplies only one condition.
- **`Outcome.FAIL`'s wording is itself a hazard** — `no_content_found` would carry the meaning — but
  `Outcome` is a closed vocabulary with readers, so this is recorded rather than changed.
- **A genuine F0 absence errors this branch, and the runner cannot tell that apart from a crash.**
  `_f0_range` raises `F0RangeUnavailable` at `voice.py:71` with no handler anywhere up to `voice()`,
  so the branch leaves at `:193` — before its first store write at `:207-219`, and before the no-span
  `FAIL` at `:235-264` — and `_attempt` records it `ERRORED` like any other failure. The refusal
  itself is correct and pinned by `voice_test.py:295-311`; what is owed is at the runner. The
  mechanism, the affected population and why no shipped config avoids it are in
  [`dag.md`](dag.md) § *5c. VOICE*. **Owed a code change.**
- **A second, independent F0 estimate is a candidate for the type-2 deadlock — and is owed a
  measurement.** SPARC returns per-frame `pitch` and `periodicity` from a neural tracker family
  (`features_extraction/sparc.py:40-41` pins `torchcrepe` and `penn`), not from Praat's
  autocorrelation, so **two independent trackers disagreeing by an octave is evidence a single
  tracker cannot produce** — which is what makes it worth recording against V4's finding that no
  type-2 instrument works today. It is not a solution: disagreement localises without typing,
  SPARC's own octave-error rate is unmeasured here, `periodicity` is not a subharmonic-to-harmonic
  ratio, and any disagreement criterion is an operating point this corpus may not be used to fit.
  Triage calls SPARC nowhere today. The full proposal, its cost and everything it owes are in
  [`branch-ddk.md`](branch-ddk.md) under *SPARC as a shared PREPROCESS derivative*, where DDK rhythm
  is the motivating case; this branch is a second consumer, and its raw-versus-enhanced question is
  its own — sustained phonation, not a DDK train.
- **`hnr_track` is a derivative sitting behind a branch, and that is a placement defect independent
  of whether VOICE runs.** PREPROCESS's job is the shared derivative state; a branch's job is to
  conclude on its own question. Checked against what the branches actually do:

  - **AIRWAY writes no derivative at all.** It reads `silence` (`airway.py:200`), `spans_no_contrast`
    (`:204`) and PREPROCESS's stored `span_hear` windows (`:242`), and emits assertions and a
    verdict.
  - **SPEECH writes findings — spans, speakers, PII entities, assertions — and one derivative it is
    easy to miss**: when separation is configured and the speaker count is exactly two, it writes
    `separated_<index>` WAV sidecars as `stream` entities (`speech.py:752-758`). That is signal
    state, not a conclusion. It is gated on `speech.separation_backend`, which has shipped
    `MossFormer2_SS_16K` since 2026-09-23, so these are written on the recordings the trigger fires
    on — and the same placement question applies to them, in a weaker form, because the stream is
    inherently interval-scoped rather than branch-scoped.
  - **VOICE is the only branch that writes an `.npz`**, at `voice.py:368`, carrying `hnr_db` among
    six arrays.

  `hnr_track` (`tasks/phonation/api.py:101`) is the same *kind* of object as `f0_track` (`:185`) and
  `formant_track` (`:234`): a per-frame track over the same recording on the same configured hop.
  Those two already live in PREPROCESS's `phonation_tracks` (`preprocess.py:931-1005`). This one
  does not, and it has exactly one caller in the tree — `voice.py:267`.

  **The consequence is stronger than "VOICE fails, so it never runs."** Suppose VOICE worked
  end to end. HNR would then exist only for recordings *routed to VOICE*. QUALITY could not read it,
  AIRWAY could not read it, and any cross-branch question about voicing quality would be
  unanswerable by construction rather than by omission. **A derivative whose availability depends on
  routing is not a shared derivative.** It is narrower still than that: `voice.py:267` computes the
  track over the whole stream, then slices it to the phonation spans and concatenates
  (`:297-303`, `:368-376`), so even on a routed recording only in-span frames survive.

  **The rule that separates the two cases**, since it generalises past this one item:

  - A **derivative** is a measurement of the signal that any consumer might want. It is
    branch-independent, so computing it per branch means recomputation and drift between the copies.
    It belongs in PREPROCESS.
  - A **finding** is a branch's conclusion about its own question, in its own family. It belongs to
    the branch.

  By that test `hnr_track` moves into `phonation_tracks`. VOICE's *other* write — proposing
  `family: "phonation"` spans — stays exactly where it is: `propose` is one of the contract's five
  verbs (`../20260913-branch-contract-and-hints/design.md:43-47`) and proposing its own subject is
  what a branch is for.

  **The consumer counts corroborate the test rather than merely illustrating it.**
  `phonation_tracks` has three independent readers — VOICE (`voice.py:275`), REPORT's phonation lane
  (`report.py:79`) and the routing-analysis feature reducer
  (`routing_analysis/features.py:100`, `:664-683`). `voice_tracks` has none: no
  `find_measurement(store, "voice_tracks")` exists anywhere in the tree.

  **And to answer the entity question directly: `voice_tracks.npz` *is* named by a measurement
  entity** — `voice.py:377-386` writes a `measurement` with `name: "voice_tracks"`, the signal, the
  hop and `path_attributes(tracks_path, run_dir)`. The inversion is worth recording because it is the
  opposite of what one would guess: the branch-local sidecar carries its own path, while the shared
  `phonation_tracks` measurement carries no path at all (`preprocess.py:997-1002`) and every reader
  spells the location itself (`features.py:100-102`). Whichever way `hnr_track` is placed, the path
  convention should be made one convention.

  **A known defect travels with the track if it moves.** Audit finding 10: `hnr_db` is written with
  Praat's −200 dB undefined-frame sentinels unmasked, and `phonation.hnr_floor_interval_db` is null
  (`data/config/default.yaml:146`), so nothing masks them downstream. If HNR becomes a PREPROCESS
  derivative the defect becomes a *shared* one, reaching every consumer instead of one — so it must
  be named at the new site and fixed there, not carried across silently. The masking rule is the
  reason to fix it at the writer: `extract_harmonicity_descriptors`' own `Get mean` excludes the
  sentinels, so any consumer that means the raw array will disagree with the scalar on the same
  audio.

  **The consequence for the Praat trajectory item.** If intensity, CPPS or spectral-moment tracks are
  ever added — see [`praat-instrument-audit.md`](praat-instrument-audit.md), *"The wrappers discard
  the trajectory as well as the count"* — **they go in PREPROCESS too**, by the same test, not into
  whichever branch first wants one. Deciding that now is what stops the question being reopened per
  track.
