# Branch conventions

Rules shared by [`branch-airway.md`](branch-airway.md), [`branch-speech.md`](branch-speech.md),
[`branch-voice.md`](branch-voice.md), [`branch-ddk.md`](branch-ddk.md) and
[`branch-quality.md`](branch-quality.md). Stated once so four branches cannot invent four versions.

The contract itself is
[`../20260913-branch-contract-and-hints/design.md`](../20260913-branch-contract-and-hints/design.md).

> **Superseded in one respect, 2026-09-16: a branch no longer concludes an `Outcome`.** It writes a
> `branch_report` carrying task conformance, typed deviations and what it could not measure, and
> proposes spans; VERDICT takes every decision. A branch also never raises over an unmeasured
> operating point. Every rule below still holds except where it names an outcome — see
> [`verdict.md`](verdict.md) § *A branch reports; this fold decides*, and § *A branch `FAIL` is an
> absence of detected content* below, which records what became of that outcome.

## Span family is lowercase

A branch that proposes a span writes `family: "airway"`, `"speech"`, `"voice"` or `"ddk"`.

Not cosmetic. Live convention is lowercase everywhere — `speech.py:883` writes `"speech"`,
`voice.py:40` `"phonation"`, `quality.py:58` `"clip"`, and `report.py:673`, `:715` and `:1154` are
lowercase-keyed. VOICE's entire repair is that its selector and its proposals meet on one string; a
proposal written `"VOICE"` against a selector reading `"voice"` reproduces the bug being fixed, and
reproduces it silently.

## `propose` versus `refine` — **scoped by family**

**A branch proposing in its own family mints when no live span *of that family* covers the ground.
It `refine`s only a span of the family it is proposing into.**

Overlap is the strict interval test `a.start < b.end and a.end > b.start`, used at six sites in the
tree: `_novel` (`preprocess.py:1467`), the `contains_clip` computations (`preprocess.py:1552`,
`:1582`), `figure.py:402`, and the `_overlaps` helper duplicated at `speech.py:137` and
`redact.py:212`. Ties go to the earlier span by extent.

**The family scoping is not a detail — without it the rule undoes the V1/D1 inversion.** An earlier
version said a region overlapping *any* live span mints nothing. PREPROCESS proposes on a 6 dB rise
over the 5th-percentile floor with a 50 ms minimum, so an eighteen-second sustained vowel or a
ten-second DDK train always sits under live spans. VOICE V1's attempt would therefore land as a
`refine`, never as a `family: "voice"` span; the selector V1 changed to read the branch family would
be empty; and the branch would `FAIL` on a recording full of phonation — reintroducing precisely the
bias the inversion removed.

**The rule's purpose is untouched.** It exists to stop a branch re-minting the general span set and
orphaning its per-span measurements. A `family: "voice"` span collides with no `family is None`
reader, so scoping by family preserves the protection and removes the collision.

Three consequences worth stating rather than leaving implied:

1. **A branch measures over its own family span**, so a per-span measurement keys to that span's id
   and the measured extent is that span's own extent.
2. **Covariates and support counts therefore describe the same extent** as the span they hang on —
   there is no mismatch between what was measured and what the covariate qualifies.
3. **Nothing is left unreconciled.** A phonation broken by two 400 ms gaps is three PREPROCESS spans
   under `spans.min_separation_ms: 30`; the branch's single attempt span simply covers all three. The
   branch is not refining them, so they neither need merging nor conflict with it.

### The two owner decisions of 2026-09-15 leave the minting rule alone and open one question

**Neither decision changes this rule.** That a branch *refines and reviews rather than waits for a
correct subject*
([`../20260913-branch-contract-and-hints/design.md`](../20260913-branch-contract-and-hints/design.md)
§ *A branch refines and reviews*) is a statement about what a branch owes, not about which spans it
may mint over; the family scoping above is what stops the same statement collapsing V1 into a
`refine`.

**What it does open: whose family a ruleset-written label puts a span in.** A fired rule may now stamp
or refine a span's label (that design's § *A ruleset that fires may write or refine a span's label*),
and the candidate it would label for VOICE is an `amplitude` span — `voice.sustained`'s feature is
`[span_longest, amplitude]` (`data/config/default.yaml:252-255`). Under the rule above VOICE may
`refine` only a span of the family it proposes into, so a label alone does not make that span
refinable. Either the label sets the span's `family`, which is a writer touching a span PREPROCESS
minted, or the scoping widens for the four annotating verbs while staying for `propose`.
**Unresolved**, and it is now the *only* thing open in VOICE's span source: the owner settled the
rest on 2026-09-15, at [`branch-voice.md`](branch-voice.md) § *The subject is the spans the ruleset
labelled, and VOICE refines them*, which names this section as where the family question is
decided.

### An aggregated span is a `propose`, and the rule above already answers it — 2026-09-15

The owner, on SPEECH and speaker attribution: *"speech may need to resolve across multiple speakers,
so it could fall under refinement (adding/adjusting span metadata, or creating an aggregated span)."*
**Creating an aggregated span is a `propose`** — the contract's only minting verb — and the three
questions that raises are answered by the scoping rule above plus existing practice, so none is
recorded as owed.

**Its family is the branch's own**, `family: "<branch>"`. An aggregate covers ground no live span *of
the branch's family* covers, so the rule at the head of this section permits the mint without
amendment. Consequence 3 above is this case already: three PREPROCESS spans broken by gaps, one
branch span covering all three, no refinement of the three and no conflict with them.

**It is `wasDerivedFrom` every span it aggregates, plus the evidence that built it.** SPEECH does this
today — its `family: "speech"` span is `wasDerivedFrom` every live non-SPEECH span it overlaps
(`speech.py:893-895`, over the `prior_spans` set at `:577-578`) — and `ProvStore.derived_from` returns
a **list** (`../../src/senselab/utils/prov_store.py:525`), so many-to-one derivation is a capability of
the store. `figure.py:580` iterates the relation rather than taking its first element, so the one
production reader that walks it is unaffected.

**The spans it aggregates stay untouched.** Nothing under `nodes/` invalidates anything: the module's
single `was_invalidated_by` call is `extend.py:327`, and `quality.py:27` states the rule outright —
*"PREPROCESS's spans are never invalidated here: the store is append-only"*.

**`refine` widened on the same date and does not change this rule either.** From 2026-09-15 `refine`
covers a span's metadata as well as its extent (the contract's § *`refine` covers metadata as well as
extent*), which is what makes *adding or adjusting span metadata* a `refine` rather than a new verb.
The family scoping is untouched: a branch still `refine`s only a span of the family it proposes into,
whichever of the two things it is correcting, and the open question below is unchanged by the
widening.

## Deviations and counts are stored, and need no new `PROV_TYPE`

| what | stored as |
| --- | --- |
| a deviation | `assertion`, `verb: "deviate"`, plus `deviation_type`, the extent, and the evidence |
| a count | `measurement` named `counts`, each entry carrying `found` and `declared` |

Nothing is added to the `PROV_TYPE` literal at `prov_store.py:17-31`.

**`verb: "deviate"` will need admitting when contract piece 7 widens REPORT's assertion read by
verb**, and that piece now names it. That is a forward statement, not a description: **REPORT reads
no verb set today.**
`report.py:1123-1128` filters by branch and `prov_type` and drops every assertion whose branch is not
AIRWAY; the only verb tests in the file are hard-coded literals — `== "label"` at `:196` and
`!= "label"` at `:327`.

**A count asserts no discrepancy.** `found` and `declared` sit side by side; whether a difference
disqualifies the recording is not the branch's call.

### The deviation-type vocabulary

**This is the authoritative list.** The contract's table carries three rows and reads as exhaustive;
the branch documents mint more. Neither is wrong — the contract established the shape, the branches
supply the types — but the list has to live somewhere, and it lives here.

| type | branch | what it says |
| --- | --- | --- |
| `off_task_extent` | AIRWAY | positively-identified off-task content inside an airway task |
| `stimulus_mismatch` | SPEECH | an aligned word that is not the word the stimulus expected |
| `filler` | SPEECH | a bracketed disfluency where a **read** task expected lexical content |
| `truncation` | SPEECH, VOICE, DDK | the production begins or ends at the recording boundary |
| `language_mismatch` | SPEECH | the transcript language differs from the declaration |
| `repeat_reading` | SPEECH | the stimulus was read more than once |
| `sweep_direction_mismatch` | VOICE | the dominant monotone segment ran opposite to the declared direction |
| `repeat_attempt` | VOICE | more than one attempt where the task asked for one |
| `syllable_sequence_mismatch` | DDK | a produced syllable that is not the one the sequence expected |

A branch adding a type adds a row here. `truncation` is shared deliberately — it means the same
thing in all three.

### Norm-bearing scalars carry their measurement convention in the name

Any scalar with circulating published norms **will be read against them**, whatever a document says
elsewhere. Suppressing one scalar does not scale: it applies verbatim to DDK syllable rate, CPPS in
dB, jitter and shimmer in percent, F0 SD in semitones, and maximum phonation time in seconds.

**So the name carries the convention** — `ddk_syllable_rate_from_envelope_peak_hz`, not `rate`;
`cpps_db_voiced_intervals_60_330hz`, not `cpps` — though note that scalar is one
[`branch-voice.md`](branch-voice.md) V4 currently forbids emitting, and after audit step 4 its name
must also carry the two smoothing windows and the trend range.

**The two largest norm-bearing populations were omitted from this rule and are its clearest case.**
`speaking_rate`, `articulation_rate` and `phonation_ratio` — [`branch-speech.md`](branch-speech.md)
S4's ~25,000 recordings and [`branch-ddk.md`](branch-ddk.md) D2's 7,989 — carry circulating norms in
syllables per second, and come from a helper whose `silence_db −25`, effective `min_dip 2`,
`min_pause 0.3 s` and 0.1 s minimum sounding interval are **all non-Praat**. Published bare over
~33,000 recordings they would be read against norms collected under Praat's own thresholds. And every such value carries a standing statement
that it is not comparable to published norms collected under a different measurement convention, on
different equipment.

### Spectral bands are per measure, and declared per measure

Every spectral measure — CPP, slope and tilt, moments, HNR, formants — carries **its own declared
analysis band**, and any value whose recording does not support that band is marked
**non-comparable** rather than reported. A bandwidth covariate lets a reader *notice* mixed
bandwidths; it does not make the numbers poolable.

**An earlier version declared one common band of 5 kHz "because that is already what is in force".
That was wrong twice over.**

**It is not in force.** The bands actually differ per measure: CPP cepstrogram 0–5000, tilt
100–5000, **slope 50–1000 against 1000–4000**, moments 0–5000, formants 0–5000 — and **HNR is
full-band**, because `to_harmonicity_cc` takes no maximum frequency (`praat_parselmouth.py:679-681`).
Declaring HNR "5 kHz" would mislabel the instrument, and band-limiting it later to comply would
change every HNR value.

**And it ratified a defect.** [`praat-instrument-audit.md`](praat-instrument-audit.md) finding 7
identifies the 5 kHz moments cap as excluding the high-frequency energy that is the breathiness
correlate — the dysphonia signal — and promoting that unchosen Praat default to *the convention* is
the opposite of remediating it. It would also silently kill [`branch-airway.md`](branch-airway.md)
**A6**, since coughs are the most broadband events in the corpus, and **A7**, whose entire mechanism
is reduced high-frequency energy for nasal breathing.

**No single band can serve both** formants (5000 / 5500 / 8000 depending on the speaker) and cough
spectra. So the band is a property of the **measure**, declared with it, and
[`branch-quality.md`](branch-quality.md) Q2's effective bandwidth is a validity check against
whichever band the measure declares.

**And every band sits under an 8 kHz ceiling nobody declared for this purpose**: `resample.target_hz:
16000`, derived from model input requirements — audit finding 13.

**Which resolves HNR.** The function imposes no band; the *signal* does. So **HNR's declared band is
0–8000 Hz** — not because anyone chose it for HNR, but because that is the ceiling the resample
leaves, and Q2 needs a number to compare against rather than the word "full-band". Naming it
`..._0_8000hz` under the convention-in-the-name rule would be misleading in the other direction, so
HNR's name carries **no band** and its declared band travels beside it with the note that it is
inherited from `resample.target_hz`, not chosen.

### The octave-jump count means three different things

It appears in three places with three interpretations, and a reader needs the rule:

| context | interpretation |
| --- | --- |
| any F0 track | **tracker instability** — the estimator jumped, no claim about the voice |
| a glide, mid-sweep | **a normal register break** — modal to falsetto in an untrained voice |
| a sustained vowel | **candidate type-2 evidence, not separable from tracker error** — an octave-scale jump here is equally period doubling and the tracker's own octave error, and this set argues octave-error risk is *elevated* on exactly this material. Partial discriminator: a jump to exactly ½ or 2× that **persists** across frames, against an isolated one |

The count alone distinguishes none of them; the **task and the position within the production** do.
A count used as type-2 evidence is restricted to sustained material, never to glides.

## A deviation is not evidence of a bad recording

`filler`, `repeat_attempt`, `stimulus_mismatch`, `syllable_sequence_mismatch` and `truncation` are
produced **because of** the conditions this corpus exists to study — stuttering, aphasia, apraxia of
speech, Parkinson's disease. Filed under a heading that reads as protocol non-compliance, they
invite a reader to exclude the recording rather than measure it.

**A deviation records that the production differed from what was asked. It says nothing about why,
and nothing about whether the recording is usable.** Every branch document repeats this in its own
deviation section, and any consumer that filters on deviations is filtering on impairment.

## Quality covariates travel with every acoustic measurement

A branch measurement carries the covariates of **its own extent**: clipping, SNR, and — where the
measurement is bandwidth-sensitive — the sample rate.

**Some covariates are file-level and cannot be per-extent.** Effective bandwidth and any AGC or
noise-suppression signature are properties of the capture chain, and an AGC signature is not
definable on a 400 ms span at all. Those are computed once per recording (see
[`branch-quality.md`](branch-quality.md) Q2) and referenced by every extent; only clipping, SNR and
support count are genuinely per-extent. An earlier version of this document required all covariates
per-extent, which contradicted Q2's file-level emission.

### Every distributional summary carries its support count

F0 standard deviation over a segment that was 40% voiced, or CPP over one that was mostly silence,
is not comparable to the same number over a fully-supported segment. **Report the count of frames
actually contributing** beside every mean, SD, median or percentile. This is the cheapest
interpretability guard available and it is currently nowhere in the tree.

### The consumer capture chain is the dominant variance source

Browser and phone capture default to automatic gain control, noise suppression and echo
cancellation. Both are detectable without a threshold:

**AGC** — report the background level in inter-phonation pauses beside the phonation level, and note
when they move oppositely. The noise floor rising as gain is pushed while speech level falls is the
signature.

**Noise suppression is the greater hazard.** Spectral gating manufactures HNR and CPP values
outright. It shows as: the pause noise floor collapsing toward digital silence with very low
variance; level steps at speech boundaries; and a pause noise spectrum unlike the in-speech one.

**This capability has no owner, and that is a gap.** This document requires the covariate of every
measurement; it previously delegated the computation to
[`branch-quality.md`](branch-quality.md) Q2, which specifies **bandwidth only** and has no AGC row —
and QUALITY's own rule sends anything needing the waveform to PREPROCESS. So the covariate is
required of everyone and computed by nobody. **It needs a named capability, an owner and a computing
node before any measurement can claim to carry it.** Recorded in
[`branch-quality.md`](branch-quality.md) as owed.

**And it is not parameter-free either.** "Very low variance" and "a level step" are cuts, however
they are phrased. Comparative framing reduces the number of parameters; it does not remove them.
Whatever owns this declares them.

### AGC has no covariate at shimmer's own extent

AGC acts over tens to hundreds of milliseconds — **inside the shimmer analysis window** — and it
directly modulates cycle-to-cycle amplitude, which is the quantity shimmer measures. On consumer
capture it is the single largest threat to it.

But this document places the AGC signature at **file level**, on the grounds that it "is not
definable on a 400 ms span at all". So **a per-extent shimmer value has no AGC covariate available at
its own extent**, and the file-level one describes a different timescale from the one doing the
damage. Recorded as a hole rather than papered over.

### Mouth-to-mic distance and reverberation

Plausibly the largest uncontrolled variable in unsupervised phone collection, and directionally
harmful: **reverberation inflates shimmer while depressing CPP and HNR — it moves every voice-quality
number toward dysphonia.** Distance change is also task-correlated, since participants pull the
phone away when asked to be loud.

Neither is directly in the inventory. Derivable proxies are a direct-to-reverberant estimate and
level relative to the noise floor. **At minimum this is named as an unmeasured confound** rather
than omitted.

## Analysis-window conventions are conventions, not fits

Where a measurement needs a window — excluding the attack and decay of a sustained vowel, a search
band for a modulation peak, a roll-off reference for bandwidth — the window is **declared as a
convention** and the measurement reported against it. A stated convention is not a fitted threshold.
A measurement with no stated window is comparable to nothing.

**"Parameter-free" is a claim to check, not to assert.** Several capabilities in these documents
claimed it and were wrong: a modulation spectrum needs an envelope extraction, a lowpass cutoff, an
analysis window and a search band; a roll-off is defined relative to something.

## A precondition all four proposing branches share

**`features.py:1079-1091` must gain a `family` filter before any branch-proposed span exists.** It
appends every live span to `live_spans` with no filter, and `_span_statistics`' `all.*` bucket
includes them. In the live pipeline routing precedes the branches, so nothing changes there — but an
offline recompute over finished stores (`scripts/analyze_routing_evidence.py:158`) would pull
branch-proposed spans into `all.duration_*`, `all.rate_per_s` and `all.duty_fraction`.

SPEECH already mints `family: "speech"` spans today, so the exposure predates the contract.

### Non-comparable beats a covariate, and steadiness is the harder case

This document marks a value **non-comparable** rather than reporting it when the spectral band is
unsupported. **Perturbation over non-steady material is the more severe invalidity and currently gets
only a covariate.**

VOICE routes 22,277 recordings against 8,306 declaring a voice family, so perturbation over connected
speech is the common case, not the edge case. Either **steadiness belongs in the value's name** — so
the number cannot be read without it — or unsteady-extent perturbation is marked **non-comparable**,
as an unsupported band would be. A covariate a reader may ignore is not equivalent to a value they
cannot misread.

**The second option is gated on something owed.** Marking a value non-comparable needs a steadiness
**cut**, and [`branch-voice.md`](branch-voice.md) V1's steadiness statistic and window are themselves
owed — no statistic is named and nothing called `stationarity` or `spectral_flux` exists in
`src/senselab`. So **only the naming option is available today**, and the non-comparable option
becomes available when V1's statistic does.

### An aggregate must distinguish absent from normal

The individual capabilities handle absence correctly: an unavailable measurement is an absence, never
a negative. **Nothing in the report or verdict design requires an aggregate to preserve that.**

It matters most where absence is not random. Jitter, shimmer and CPPS all return NaN at the disordered
end, so **a corpus distribution reads conspicuously healthy because the disordered cases are missing,
not because they are absent from the population** — see
[`praat-instrument-audit.md`](praat-instrument-audit.md) step 5, which turns that failure mode into a
measurement.

**So any aggregate — a mean, a distribution, a rate, a figure — reports the count it was computed
over and the count that was unavailable**, and a consumer that cannot show both should show neither.

## Some capabilities are not per-recording

A capability that compares across recordings, or that assembles input from several recordings of one
session, belongs to the **corpus-level node** that runs last over the finished per-recording stores —
[`corpus-level-node.md`](corpus-level-node.md). Duplicate detection and the composite voice-quality
indices are there; a branch that finds another such capability flags it rather than parking it in a
per-recording section where it cannot run.

## The corpus count

These documents quote **62,547** — the recording count in
`runs/ruleset-score-20260912/ruleset_score.json`, the in-tree artifact every branch figure is drawn
from. The governing contract asks for the attested **62,550** until the 28-store divergence is
accounted for.

**There are three numbers, not two.** 62,547 recordings scored; **62,550** attested as the corpus;
and **62,578** stores found on disk. So 62,547 and 62,550 differ by **3**, and the 28-store gap the
contract asks to account for is between 62,550 and the on-disk count. An earlier version of this
section said "they differ by 28", which conflated the two gaps.

Citing the scoring artifact is defensible when every branch figure comes from it, and mixing them
would be worse. Both divergences remain unaccounted for.

**And two measurements reach none of those three numbers.** `praat_features` and `ppg_posteriorgram`
are in **60,202** stores — the row count of the manifest the `ppg_20260911` pass ran over — and
**2,376 of the 62,578 stores hold neither**, measured 2026-09-14. The loss is not uniform across task
families: it runs from **28%** on `respiration-and-cough-v2-breath` to **6%** on `glides-high-to-low`,
so it is deepest on AIRWAY's entire task content and on `maximum-phonation-time`. A per-family
statistic over a Praat scalar or a posteriorgram therefore has a family-dependent hole, which without
the paired counts above reads as a finding about breath recordings rather than about the manifest.
The measurement, what is and is not known about its cause, and what it is owed are in
[`../20260911-ppg-praat-batch/design.md`](../20260911-ppg-praat-batch/design.md).

## What a branch does with an unavailable measurement

An unavailable measurement is an **absence**, never a negative. Gate evidence is `unavailable` on
56,505 of 62,547 recordings for AIRWAY, 2,345 for DDK, 29 for VOICE and 0 for SPEECH
(`runs/ruleset-score-20260912/ruleset_score.json`, `totals.unavailable` — the file sits beside this
document under `runs/`).

## A branch `FAIL` was an absence of detected content — and is now no span at all

**Retired 2026-09-16, and the hazard it named is what retired it.** `Outcome.FAIL` from a branch
meant **this branch's detector found nothing**, not that the recording lacked the content and never
that the speaker failed to produce it. The reading a branch now reports is the spans it proposed —
none of them — and VERDICT reads `findings: absent` off that. The paragraph below is why the
replacement is an improvement rather than a rename, so it is kept.

This matters at corpus scale and in one direction. VOICE's `FAIL` says "no phonation found" and DDK's
says "no train found" — and a detector keyed on voicing or on regular repetition fails most often on
disordered phonation and irregular trains. **Without care, the recordings marked `FAIL` across
62,547 would be disproportionately those from the most impaired speakers**, which is the population
the corpus exists to characterise.

VOICE V1 and DDK D1 are specified to propose from the energy envelope and *qualify* by voicing or
repetition, rather than defining the event by them, which removes most of the cause. The residue is
named in both documents.

**`Outcome.FAIL`'s own wording was a hazard** — `no_content_found` would have carried the meaning
better — and it was recorded here as unresolved because `Outcome` was a closed vocabulary with
readers. The 2026-09-16 change resolved it by removing the vocabulary from the branch rather than
renaming a member: there is no branch outcome to be misread, and "no span proposed in this branch's
own family" says exactly what the old `FAIL` was trying to say. The population risk the paragraphs
above name is unchanged by that — a detector keyed on voicing still finds fewer spans on disordered
phonation — so the residue stays named in `branch-voice.md` and `branch-ddk.md`.
