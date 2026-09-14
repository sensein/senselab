# Branch conventions

Rules shared by [`branch-airway.md`](branch-airway.md), [`branch-speech.md`](branch-speech.md),
[`branch-voice.md`](branch-voice.md), [`branch-ddk.md`](branch-ddk.md) and
[`branch-quality.md`](branch-quality.md). Stated once so four branches cannot invent four versions.

The contract itself — the five verbs, what a declaration is, the no-refits rule — is
[`../20260913-branch-contract-and-hints/design.md`](../20260913-branch-contract-and-hints/design.md).

## Span family is lowercase

A branch that proposes a span writes `family: "airway"`, `"speech"`, `"voice"` or `"ddk"`. Lowercase,
always.

This is not cosmetic. Live convention is lowercase everywhere — `speech.py:883` writes `"speech"`,
`voice.py:39` `"phonation"`, `quality.py:58` `"clip"`, and `report.py:673`, `:715` and `:1154` are
lowercase-keyed. VOICE's entire repair is that its selector and its proposals meet on one string; a
proposal written `"VOICE"` against a selector reading `"voice"` reproduces the exact bug being
fixed, and reproduces it silently — the branch would simply find nothing.

## `propose` versus `refine`

**A proposed region that overlaps any live span mints nothing. It is a `refine` on the span it most
overlaps. Only a region over ground no live span covers is a `propose`.**

Overlap is the strict interval test the tree already uses in `_novel` and in `contains_clip`:
`a.start < b.end and a.end > b.start`. Ties — equal overlap with two spans — go to the earlier span
by extent.

This is parameter-free, and it matches how PREPROCESS already decides novelty, so a branch and the
proposer agree about what "new" means.

## Deviations and counts are stored, and need no new `PROV_TYPE`

The contract defines a deviation as *an observation with an extent*. That is an `assertion`.

| what | stored as |
| --- | --- |
| a deviation | `assertion`, `verb: "deviate"`, plus `deviation_type`, the extent, and the evidence |
| a count | `measurement` named `counts`, each entry carrying `found` and `declared` |

Nothing is added to the `PROV_TYPE` literal at `prov_store.py:17-31`, so no reader's vocabulary
changes — which is the rule the contract applies to the merged node names and to the `kind` entity
type.

`verb: "deviate"` is a new verb and must be added to the admitted set when the contract's piece 7
widens REPORT's assertion read by verb, alongside `label`, `contest`, `refine`, `trim`, `abstain`
and `flag`.

**A count asserts no discrepancy.** `found` and `declared` sit side by side; whether a difference
disqualifies the recording is not the branch's call.

## Quality covariates travel with every acoustic measurement

A branch measurement must carry the quality covariates of **its own extent**: clipping, SNR,
effective bandwidth, and any AGC signature. Nothing in the store currently says this, and without it
a perturbation value computed over a clipped, noise-suppressed, band-limited span is indistinguishable
from one computed over clean audio.

**The consumer capture chain is the dominant variance source**, and it is invisible in the metadata.
Browser and phone capture default to automatic gain control, noise suppression and echo
cancellation. Spectral gating manufactures HNR and CPP values; AGC destroys the within-recording
dynamics that *are* the loudness task.

**A parameter-free detector exists.** Report the background level in inter-phonation pauses beside
the phonation level, and note when they move oppositely — the noise floor rising as gain is pushed
while speech level falls is the AGC signature, and it needs no threshold to describe.

**Effective bandwidth is not the declared sample rate.** CPP, spectral slope and tilt, spectral
moments, HNR and F3/F4 are all bandwidth-dependent, so an undeclared band limit turns device class
into a pseudo-finding. It is measurable from the long-term average spectrum. See
[`branch-quality.md`](branch-quality.md), which owns it.

## Analysis-window conventions are conventions, not fits

Where a measurement needs a window — excluding the attack and decay of a sustained vowel, say — the
window is *declared as a convention* and the measurement is reported against it. A stated convention
is not a fitted threshold and does not violate the no-refits rule. A measurement with no stated
window is comparable to nothing.

## A precondition all four proposing branches share

**`features.py:1079-1091` must gain a `family` filter before any branch-proposed span exists.** It
appends every live span to `live_spans` with no filter, and `_span_statistics`' `all.*` bucket
includes them. In the live pipeline routing precedes the branches, so nothing changes there — but an
offline recompute over finished stores (`scripts/analyze_routing_evidence.py:158`) would pull
branch-proposed spans into `all.duration_*`, `all.rate_per_s` and `all.duty_fraction`, silently
changing the routing features a finished store reduces to.

The contract records this precondition. AIRWAY A5, VOICE V1, DDK D4 and SPEECH S2 all depend on it,
and SPEECH already mints `family: "speech"` spans today — so the exposure predates the contract.

## What a branch does with an unavailable measurement

An unavailable measurement is an **absence**, never a negative. This matters unevenly: gate evidence
is `unavailable` on 56,505 of 62,547 recordings for AIRWAY, 2,345 for DDK, 29 for VOICE and 0 for
SPEECH (`runs/ruleset-score-20260912/ruleset_score.json`, `totals.unavailable`).
