# The listening sample

Nearly every item marked *owed ground truth* across the five branch documents is owed to the same
missing thing: **nothing in this corpus has been heard.**

This document exists so that observation sits in one place and the five cross-reference it, rather
than each repeating a variant of "this cannot be fitted".

## Why the corpus cannot supply it

The corpus carries 62,547 recordings and a declared family for each. A declared family is what the
protocol *asked for*, not what the participant *did*. Fitting any threshold against it produces a
detector for the declaration — which recordings the protocol labelled — rather than for the
phenomenon. That is the no-refits rule in
[`../20260913-branch-contract-and-hints/design.md`](../20260913-branch-contract-and-hints/design.md),
and it is why the branch documents mark numbers owed instead of proposing to measure them.

The rule bites hardest where it matters most. `spans.k_db`, `airway.contest_labels`,
`airway.labels_of_interest`, the respiratory-cycle detection parameters, `voice.f0_search_range_hz`,
the voiced-run break tolerance, `speech.speech_test_stoi_floor`, `speech.speech_test_si_sdr_floor`,
`speech.target_match_cosine`, the three `speech.nontarget` legs, `ddk.lexical_repetition`,
`ddk.ppg_segment_rate_per_s`, the DDK train-gap criterion — none can be established from what is
already recorded.

## What would supply it

**A stratified listening sample.** A few hundred recordings, stratified across device class, task
family and measured-quality strata, annotated by a listener.

- It produces ground truth **outside the declared labels**, so using it is not a refit against the
  declaration and does not violate the rule.
- At a few hundred against 62,547 it is roughly **0.5% of the corpus**.
- It unlocks most of the owed items at once, because they are owed to the same absence.

Stratifying by *measured quality* as well as by task matters: a threshold validated only on clean
recordings will be applied to the capture chain described in
[`branch-conventions.md`](branch-conventions.md), where AGC and spectral gating are the norm rather
than the exception.

## What it does not supply

Norms. Jitter, shimmer, CPP, HNR, DDK rate and maximum phonation time all have published normative
ranges that a few hundred annotated recordings cannot reproduce, and the branch documents decline to
invent them. The one externally-anchored instrument available without touching the corpus is AVQI,
whose coefficients are published rather than fitted here — see [`branch-voice.md`](branch-voice.md).

Nor does it supply discourse-content scoring keys for the connected-speech tasks; those are
instruments in their own right.

## Status

**Not commissioned.** Recorded here as the single prerequisite behind most of the owed items in the
five branch documents, so that a reader encountering "owed ground truth" in any of them finds one
explanation rather than five.
