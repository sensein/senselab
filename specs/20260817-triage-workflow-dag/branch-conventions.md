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

## `propose` versus `refine`

**A proposed region that overlaps any live span mints nothing. It is a `refine` on the span it most
overlaps. Only a region over ground no live span covers is a `propose`.**

Overlap is the strict interval test `a.start < b.end and a.end > b.start`, used at six sites in the
tree: `_novel` (`preprocess.py:1465`), the `contains_clip` computations (`preprocess.py:1539`,
`:1569`), `figure.py:402`, and the `_overlaps` helper duplicated at `speech.py:137` and
`redact.py:212`. Ties — equal overlap with two spans — go to the earlier span by extent.

Parameter-free, and it matches how PREPROCESS already decides novelty.

## Deviations and counts are stored, and need no new `PROV_TYPE`

| what | stored as |
| --- | --- |
| a deviation | `assertion`, `verb: "deviate"`, plus `deviation_type`, the extent, and the evidence |
| a count | `measurement` named `counts`, each entry carrying `found` and `declared` |

Nothing is added to the `PROV_TYPE` literal at `prov_store.py:17-31`.

**`verb: "deviate"` will need admitting when contract piece 7 widens REPORT's assertion read by
verb.** That is a forward statement, not a description: **REPORT reads no verb set today.**
`report.py:1123-1128` filters by branch and `prov_type` and drops every assertion whose branch is not
AIRWAY; the only verb tests in the file are hard-coded `== "label"` at `:196` and `:327`.

**A count asserts no discrepancy.** `found` and `declared` sit side by side; whether a difference
disqualifies the recording is not the branch's call.

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

**Noise suppression is the greater hazard and has the easier signature.** Spectral gating
manufactures HNR and CPP values outright. It shows as: the pause noise floor collapsing toward
digital silence with near-zero variance; abrupt level steps at speech boundaries; and a pause noise
spectrum unlike the in-speech one. All three are comparisons, not cuts.

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
