# The DDK branch, dissolved into SPEECH

Owner's decision, 2026-09-17:

> the rule is if speech is detected it should go to speech (e.g. buttercup is a word), just as
> prolonged vowel also goes to speech. but the goals of the speech branch are to determine if the
> speaking task happened properly. if the ddk instrument were put into the speech branch, we could
> just divert all of ddk to speech and dispense with the special ddk branch and routing.

## The principle

Routing is by content, plus the declared task's route. Evaluation is by task, through the
expectation table.

DDK violated both halves. It had a declaration-only routing special case
(`routing.declaration_required: [DDK]` against `branch_gates.DDK: []`), which is routing by
declaration and by nothing else. And it had a branch of its own for what is an expectation entry:
ten rows of `SYLLABLE_REPETITION` whose instruments — the energy envelope, the wideband burst
spectrum, the PPG posteriorgram — are instruments, not a branch.

## What was already true

`taxonomy.ruleset.reference_family_set` carries `SPEECH: speech`, where
`speech = lexical_speech | syllable_repetition`. That set is both the gate-scoring reference and the
family→branch routing map: a recording whose declared family is a positive there routes to that
branch whatever its gates read. All ten `diadochokinesis-*` families were therefore already
unconditional routes to SPEECH. The routing layer already did what the owner asked. What moved is
the evaluation, to where routing already pointed.

That set stays wide. Narrowing SPEECH's routing reference would make SPEECH's run on a DDK
recording contingent on the `speech.lexical` gate (`>= 2` lexical words), and `/pa pa pa/` carries
none. SPEECH is where the PII scan lives, so a narrowed set is a disclosure hole: a recording with
an unexpected disclosure on it would never be scanned. The reference set is wide on purpose.

## What the merge is

`ddk.py` is now an instrument module, not a branch. It keeps every instrument — `ddk_carrier`,
`ddk_places`, `train_rate_hz`, `dispersion`, `dispersion_by_position`, `trend`, the posteriorgram
reader and `ppg_evidence` — and keeps one expectation body, `align_ddk`. `speech.py` imports them.
`align_speech` gained two arms, `SYLLABLE_TRAIN` and `SYLLABLE_SEQUENCE`, both served by
`align_ddk`'s one shared body with `len(sequence)` as the cycle.

This section once described a second body, `_repeated_word`, which the two `buttercup` rows forked
to as `ORDERED_TOKENS`. Both rows are `SYLLABLE_SEQUENCE` now and that body is deleted; the reason is
[`ddk-syllable-template.md`](ddk-syllable-template.md).

`SPEECH_EXPECTATIONS` now carries the ten families with their real expectations — the rows
`DDK_EXPECTATIONS` held — rather than eight generated `NO_LEXICAL` rows and two longhand
`buttercup` rows.

`speech()` loads the three derivatives the syllable body measures over (energy envelope, wideband
spectrogram, PPG posteriorgram) via `read_ddk`, and closes over the result exactly as the removed
`ddk()` did, and as `airway()` and `voice()` close over `run_dir`.

## `Pattern.NO_LEXICAL` and `_speech_no_lexical`: removed

`NO_LEXICAL` was the literal encoding of *this task expects no lexical content*, and it applied to
exactly the eight non-`buttercup` DDK families. That is the claim the owner rejected: `/pa/` is a
production to be evaluated as a syllable train, not an absence to be checked for. The pattern had no
other row, so it goes with the body that served it. `branch.expected_lexical_max`, the count of
lexical words a no-lexical expectation tolerated, had that body as its only reader and goes with it.

The observation `_speech_no_lexical` made — lexical words on a task that asked for none — is not
lost. `align_ddk` measures the train and its conformance; a lexical intrusion on a DDK recording
still travels as SPEECH's own off-task and PII findings, which run on every SPEECH pass.

## `verdict.detection_is_evaluation: [DDK]`: removed

What the idea was. The other three branches detect evidence that occurs incidentally — breath and
cough happen in any recording, sustained phonation happens in any recording, lexical content happens
in any recording — so for them the in-family and out-of-family modes ask genuinely different
questions. A rapid alternating syllable train was held not to occur incidentally: for DDK, *finding
the subject was evaluating the task*, and an out-of-family train was more likely the detector firing
than the participant having produced one. The key named the branches for which that held, and the
fold sent an out-of-family result from such a branch to `detector_covariates` — a fact about the
detector over the corpus — rather than to a flag ground.

Why it went. The key describes an asymmetry between a branch's two modes. Inside SPEECH there is no
such asymmetry to describe: SPEECH's out-of-family mode finds lexical speech, which is exactly the
incidental evidence the key was defined against, and SPEECH's in-family mode now covers the syllable
families. There is no branch left for which detection is evaluation, so the key has no member, and a
membership list with no reachable member is a mechanism that reports it ran.

It was already unreachable before this change: `DDK` was in `routing.declaration_required`, so DDK
ran only when the declaration named it, which is exactly the condition under which `dispatch` takes
the in-family mode. `detect_ddk` therefore never ran under the shipped routing, and the covariate
record it fed was never written. Removing the key costs no reachable behaviour. `detect_ddk` itself
is removed with it: its stated reason for existing was that the two-mode contract required both
arms of a branch, and there is no DDK branch to hold that contract.

`branch.repeat_min_occurrences` — the occurrences of one token that made it a repetition — goes
with `detect_ddk`, which was its only reader. Its derivation argued from the word: two occurrences
are a pair, three are a series, and a repetition train needs a series. Nothing measures a series of
one token any more; SPEECH's own bodies read the transcript against the instruction's tokens
instead.

Nothing here is recreated as a weight or a prior. If a future measurement shows that an
out-of-family syllable train on a SPEECH recording is a detector artefact at a measurable rate, that
rate is what would be fitted — not a membership list.

## `routing.declaration_required`: removed

`[DDK]` was its only entry, and the branch it named is gone. The key ran the routing rule backwards:
a branch it named ran **only** when the declaration named it, so the ruleset alone never routed it.
That is the declaration-only special case the owner's decision dissolves, and leaving the machinery
with no member would leave a second routing rule nothing exercises. `withheld_by_gate`,
`bad_declaration_required` and the `route_*_withheld_pending_declaration` reason go with it. Routing
is once again additive in one direction only: a declaration adds a route and removes none.

## `ungated`: kept

`ungated` is a route state for a branch whose `taxonomy.ruleset.branch_gates` entry is empty — the
ruleset never looked at it, as against having looked and declined. `branch_gates.DDK: []` was the
only empty entry the packaged config shipped, and it goes with the branch.

The state stays reachable. `ungated` is computed from the *configuration*, not from a branch list:
`route_attributes` derives it as the branches whose gate list is empty, and `branch_gates` is a
config mapping any override may deep-merge. Nothing validates a gate list as non-empty. An operator
who ships `SPEECH: []` gets `ungated` for SPEECH, and a reader who could not tell that from
`declined` would read "the ruleset declined this branch" off a ruleset that never evaluated it.
The state describes a configuration, so it survives the branch that happened to be the only one
configured into it.

## Span family: `speech`

The syllable spans are minted into `family="speech"`, not into a second family `speech` keeps for
this body.

`dispatch` raises when a branch mints outside `BRANCH_FAMILY[branch]`, and that check is the thing
that catches a branch overstepping onto another's subject. A branch with two minting families weakens
it by exactly the amount of the second family, for the benefit of a distinction the span's own
attributes already carry: every proposal from these bodies names its `role` (`task_extent`,
`ppg_train`) and its `production` (`syllable_train`, `syllable_sequence`), so
a reader separating a train from connected speech reads an attribute rather than a family.

What it means for the span axes. `report.py`'s lanes read families, and there is no `ddk` lane and
never was. A span left in `family="ddk"` after the branch's removal would be drawn by nothing — the
same defect as the two lanes that read `family="phonation"` that nothing writes. In `family="speech"`
the train and its PPG companion are drawn in the `speech spans` lane, which `_LANE_BRANCH` already
attributes to SPEECH. That lane labels each span by `attributed_to`, which these spans do not carry,
so a train renders as `unattributed`; that is the lane's existing behaviour for any speech span
without a speaker attribution and is not introduced here.

## Report measures

`report.py`'s `_BRANCH_MEASURES` carried a `DDK` entry naming the eighteen measures the removed
`ddk()` put in its report detail. With no DDK report node the entry would be looked up by nothing.
The measures are still measured, by the same instruments, so the entry's keys move onto `SPEECH`,
and `speech()` merges the syllable detail into its own report detail when — and only when — the
in-family mode ran on a syllable family. On every other SPEECH recording the keys are absent rather
than null, which is what both readers of `_BRANCH_MEASURES` already expect (`if key in detail`).

## Known-dead, and not touched here

`report.py`'s `phonation` and `voice` lanes both read `_spans_of_family(store, "phonation", …)` and
nothing writes `family == "phonation"` — VOICE proposes `family: "voice"` — so two of eleven lanes
are permanently empty. That is a real defect, it is adjacent to the lane reasoning above, and it is
a separate change.
