# Branch conventions

Rules shared by [`branch-airway.md`](branch-airway.md), [`branch-speech.md`](branch-speech.md),
[`branch-voice.md`](branch-voice.md), [`branch-ddk.md`](branch-ddk.md) and
[`branch-quality.md`](branch-quality.md). Stated once so four branches cannot invent four versions.

The contract itself is
[`../20260913-branch-contract-and-hints/design.md`](../20260913-branch-contract-and-hints/design.md).

## Span family is lowercase

A branch that proposes a span writes `family: "airway"`, `"speech"`, `"voice"` or `"ddk"`.

Not cosmetic. Live convention is lowercase everywhere — `speech.py:883` writes `"speech"`,
`voice.py:39` `"phonation"`, `quality.py:58` `"clip"`, and `report.py:673`, `:715` and `:1154` are
lowercase-keyed. VOICE's entire repair is that its selector and its proposals meet on one string; a
proposal written `"VOICE"` against a selector reading `"voice"` reproduces the bug being fixed, and
reproduces it silently.

## `propose` versus `refine` — **scoped by family**

**A branch proposing in its own family mints when no live span *of that family* covers the ground.
It `refine`s only a span of the family it is proposing into.**

Overlap is the strict interval test `a.start < b.end and a.end > b.start`, used at six sites in the
tree: `_novel` (`preprocess.py:1465`), the `contains_clip` computations (`preprocess.py:1539`,
`:1569`), `figure.py:402`, and the `_overlaps` helper duplicated at `speech.py:137` and
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

## Deviations and counts are stored, and need no new `PROV_TYPE`

| what | stored as |
| --- | --- |
| a deviation | `assertion`, `verb: "deviate"`, plus `deviation_type`, the extent, and the evidence |
| a count | `measurement` named `counts`, each entry carrying `found` and `declared` |

Nothing is added to the `PROV_TYPE` literal at `prov_store.py:17-31`.

**`verb: "deviate"` will need admitting when contract piece 7 widens REPORT's assertion read by
verb.** That is a forward statement, not a description: **REPORT reads no verb set today.**
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
`cpps_db_voiced_intervals_60_330hz`, not `cpps`. And every such value carries a standing statement
that it is not comparable to published norms collected under a different measurement convention, on
different equipment.

### One spectral analysis band, declared — and it is 5 kHz

Every spectral measure — CPP, slope and tilt, moments, HNR, formants — is computed over a **declared
common band**, and any value whose recording does not support that band is marked **non-comparable**
rather than reported. A bandwidth covariate lets a reader *notice* mixed bandwidths; it does not make
the numbers poolable.

**The band is 5 kHz, because that is already what is in force.** An earlier version mandated a common
band and declared none — while `praat_parselmouth.py:999` silently limits the spectral moments to
Praat's 5 kHz default and the cepstrogram is likewise bounded. So for those measures the **analysis**
band, not the capture band, is binding, and [`branch-quality.md`](branch-quality.md) Q2's covariate
was qualifying nothing.

Declaring 5 kHz satisfies the rule and turns Q2's effective bandwidth into what it should be: **a
validity check on whether the recording supports the analysis band**, rather than a covariate against
an undeclared one.

### The octave-jump count means three different things

It appears in three places with three interpretations, and a reader needs the rule:

| context | interpretation |
| --- | --- |
| any F0 track | **tracker instability** — the estimator jumped, no claim about the voice |
| a glide, mid-sweep | **a normal register break** — modal to falsetto in an untrained voice |
| a sustained vowel | **type-2 evidence** — period doubling or subharmonics |

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

## What a branch does with an unavailable measurement

An unavailable measurement is an **absence**, never a negative. Gate evidence is `unavailable` on
56,505 of 62,547 recordings for AIRWAY, 2,345 for DDK, 29 for VOICE and 0 for SPEECH
(`runs/ruleset-score-20260912/ruleset_score.json`, `totals.unavailable` — the file sits beside this
document under `runs/`).

## A branch `FAIL` is an absence of detected content

`Outcome.FAIL` from a branch means **this branch's detector found nothing**, not that the recording
lacks the content and never that the speaker failed to produce it.

This matters at corpus scale and in one direction. VOICE's `FAIL` says "no phonation found" and DDK's
says "no train found" — and a detector keyed on voicing or on regular repetition fails most often on
disordered phonation and irregular trains. **Without care, the recordings marked `FAIL` across
62,547 would be disproportionately those from the most impaired speakers**, which is the population
the corpus exists to characterise.

VOICE V1 and DDK D1 are specified to propose from the energy envelope and *qualify* by voicing or
repetition, rather than defining the event by them, which removes most of the cause. The residue is
named in both documents.

**`Outcome.FAIL`'s own wording is a hazard** — `no_content_found` would carry the meaning better —
but `Outcome` is a closed vocabulary with readers, so this is recorded as unresolved rather than
changed.
