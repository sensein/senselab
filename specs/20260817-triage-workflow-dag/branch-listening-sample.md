# The listening sample

Nearly every item marked *owed* across the branch documents is owed to the same missing thing:
**nothing in this corpus has been heard.**

This document exists so that observation sits in one place and the others cross-reference it.

## Two kinds of owed, and they are not the same

An earlier version of this document listed thirteen keys under one heading. That conflated two
situations with different risk profiles, and a reader could not tell which was which.

### Owed a number — the key is unset and nothing happens

| key | state |
| --- | --- |
| `airway.contest_labels` | null; AIRWAY's contest path is structurally dead |
| `speech.target_match_cosine` | null; enrollment is refused rather than compared |
| `speech.enrollment_model` | null; same |
| `speech.speech_test_stoi_floor` | null; `squim_vote` is `not_evaluated` |
| `speech.speech_test_si_sdr_floor` | null; same |
| `speech.nontarget` (three legs) | null; `nontarget_speech_s` is always `None` |
| `voice.f0_range_by_population` | null; falls back to the wide search range |
| `voice.task_duration_ranges` | null; `_task_range` returns `not_evaluated` |
| `phonation.hnr_floor_interval_db`, `phonation.rms_floor_interval` | null; the config's own comment records that *Praat calibrates no dB floor* |

A capability behind an unset key does not run. The failure is visible.

### Owed validation — a number is in force right now

| key | shipped value | marked in config? |
| --- | --- | --- |
| `spans.k_db` | `6.0` (`default.yaml:40`) | no |
| `airway.labels_of_interest` | `[Cough, Breathe]` (`:136`) | no |
| `voice.f0_search_range_hz` | `[50.0, 600.0]` (`:159`) | no |
| `ddk.lexical_repetition` | `3` (`:260`) | **yes — UNMEASURED** |
| `ddk.ppg_segment_rate_per_s` | `10` (`:264`) | no |

These are deciding outcomes on all 62,547 recordings today. What is owed is **validation of a value
already in force**, not a value. Only `ddk.lexical_repetition` says so in the config; the other four
read as settled.

**This is the more dangerous group**, because nothing about the run surfaces the fact that an
unvalidated number is load-bearing.

### Owed and not in any config at all

`praat_parselmouth.py`'s syllable-nuclei operating points — `silence_db = -25` (`:142`),
`min_dip = 4` (`:149`) dropped to `2` when mean HNR < 60 (`:155-156`), `min_pause = 0.3` (`:159`).
These reach every rate measure SPEECH S4 and DDK D2 produce and appear in no config. See those
sections; the HNR switch in particular is data-dependent.

## Why the corpus cannot supply any of it

A declared family is what the protocol *asked for*, not what the participant *did*. Fitting against
it produces a detector for the declaration. That is the no-refits rule in
[`../20260913-branch-contract-and-hints/design.md`](../20260913-branch-contract-and-hints/design.md).

## What would supply it: two samples, not one

An earlier version of this document asked for "a few hundred recordings" annotated by "a listener".
Three things were wrong with that and they matter for staffing.

### Sample A — event and content annotation

*Is there a cough here? Was the sentence read as printed? Is there a second voice? Where does the
phonation start and stop?*

Any trained annotator. High throughput. Agreement on these is good, so a second rater on a subset
is enough to establish it. This serves most of the owed items: the span gate, the airway label
vocabulary, the contest definition, the cycle-detection points, the DDK gates, the omission cut.

### Sample B — perceptual voice rating

*How does this voice sound?*

A different instrument and different staffing: CAPE-V or GRBAS, rated by speech-language
pathologists, **multiple raters**, with repeated items to establish intra-rater consistency.
Perceptual voice ratings have notoriously moderate inter-rater agreement, so a single rater produces
a number with no known reliability. This serves only the voice-quality items — and see
[`branch-voice.md`](branch-voice.md) on whether those should be emitted at all.

**Merging A and B into one exercise, as the earlier version did, gets the staffing and the power
wrong for both.**

## Sizing: the positives set the power, not the total

"A few hundred, stratified across device class × task family × measured quality" gives single-digit
cells against roughly thirteen operating points, several of which detect **rare events**. A cough in
a non-airway recording, a second speaker, a contradicted clip — these are a small fraction of
recordings, and a stratified sample that is balanced on *recordings* is sparse on *positives*.

**Size the sample by the rarest positive class each operating point must characterise**, not by a
fraction of the corpus.

## Stratify along the decision variable

Sampling uniformly across device × task × quality spends most of the budget where the answer is
obvious. **Sample densely near each candidate operating point** — recordings whose measured value
sits close to the threshold being validated — and sparsely far from it. That is how a small budget
pins an operating characteristic, and it is the difference between validating one threshold and
validating thirteen.

It also means the sample is **drawn per operating point** rather than once for all of them, though
recordings will be shared between draws.

## What it does not supply

**Norms.** Jitter, shimmer, CPP, HNR, DDK rate and maximum phonation time all have published
normative ranges that a few hundred annotated recordings cannot reproduce.

And no published perturbation norm was collected on AGC'd, band-limited, noise-suppressed phone
audio — so even the published ranges do not transfer cleanly to this corpus. See
[`branch-conventions.md`](branch-conventions.md).

Nor does it supply discourse-content scoring keys for story recall and picture description.

## Status

**Not commissioned.** Recorded here as the prerequisite behind most of the owed items, so a reader
encountering "owed" in any branch document finds one explanation rather than seven.
