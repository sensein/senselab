# Critical failure: when the graph goes straight to VERDICT

The owner's rule, 2026-09-18:

> a critical failure (e.g. both ASR's failed, or other similar), should route the file to verdict
> directly with a reason. the rest should all go to 1+ branches. not all measurements need to exist,
> but critical ones for rulesets do, including ones based on spans/etc.

Three claims: a critical failure short-circuits with a reason; every non-critical recording reaches
at least one branch; and "critical" is decided by what the ruleset reads, not by a hand-kept list.

## What "critical" is

> **A critical failure is one where some branch of the configured ruleset has a non-empty gate set
> and not one of its gates could be read.**

The ruleset then formed no opinion about that branch at all — not "it declined", but "it never
looked". Every downstream reading of that branch is a reading of an absence dressed as a decision.

Two things this is not:

* **Not "no gate fired".** A gate that read a number and stayed under its threshold is an opinion.
  A recording whose every gate is silent is `empty` or `unexplained`, both of which the fold already
  decides. `ee84b721` is what made the two tellable apart; before it they were not.
* **Not "some gate was unreadable".** AIRWAY names four gates over four independent blocks. Losing
  one of them leaves three opinions, and three opinions are a route. That case is already recorded
  per branch in `unavailable_gates` and reaches VERDICT as `route_state = unavailable`.

## Why not "no gate of any branch can be read"

That was the first-stated reading and it is **unimplementable against the packaged ruleset**,
because one gate can never be unreadable. `voice.sustained` reads
`detector_value(..., ("span_longest", "amplitude"))`, and `detector_value` for `span_longest` is

    return float(features.span_longest_s.get(arguments[0], 0.0))

— a `.get` with a numeric default, never `None`. Measured on a `RecordingFeatures` with nothing at
all written into it, every other gate returns `unavailable` and `voice.sustained` returns `silent`:

    speech.lexical               unavailable
    speech.transcript_agreement  unavailable
    voice.sustained              silent
    voice.glide                  unavailable
    voice.chant                  unavailable
    airway.breath                unavailable
    airway.cough                 unavailable
    airway.bracketed_event       unavailable
    airway.ppg_silent_fraction   unavailable

So under "no gate of any branch", no recording is ever critical, the short-circuit is dead code and
the owner's own example never fires. The branch-wise rule is what the code can actually express.

That `span_longest` reads an absent span table as `0.0` rather than as unread is a separate defect —
it makes "measured, no spans" and "the spans block never ran" the same number. It is not fixed here:
`evidence_blocks(("span_longest", ...)) == ("spans",)` already records the dependency, so the
derivation below sees it even though the reader does not.

## The critical set, derived from the configured ruleset

A block is critical when **every gate of at least one gated branch reads it** — lose it and that
branch has no gate left. `ruleset.critical_blocks` computes exactly that from
`taxonomy.ruleset.branch_gates` and each gate's reader through `detectors.evidence_blocks`. Nothing
is hard-coded: a campaign shipping other gates gets other critical blocks with no code change.

Measured over the packaged ruleset:

| branch | gates | blocks every gate reads |
| --- | --- | --- |
| AIRWAY | `airway.breath` (residual), `airway.cough` (span_yamnet), `airway.bracketed_event` (consensus_transcript), `airway.ppg_silent_fraction` (ppg_posteriorgram) | — |
| SPEECH | `speech.lexical` (consensus_transcript) | `consensus_transcript` |
| VOICE | `voice.sustained` (spans), `voice.glide` (yamnet_windows, yamnet_scores), `voice.chant` (yamnet_windows, yamnet_scores) | — |

**The packaged critical set is `{consensus_transcript}`**, and it is the owner's named example: both
ASR blocks failing leaves SPEECH — a one-gate branch — with nothing to read. The derivation did not
have to be told that. It is also the span-derived case the owner names, from the other side:
`airway.cough` reads `span_yamnet` and `voice.sustained` reads `spans`, and neither is critical only
because each sits beside sibling gates over other blocks. Give SPEECH a second gate over another
block and `consensus_transcript` stops being critical; take `voice.glide` and `voice.chant` away and
`spans` becomes critical. The set follows the ruleset.

Flag gates are excluded. `speech.transcript_agreement` annotates a branch already entered and routes
nothing, so its being unreadable withholds no decision.

Measured, the two readings of a missing transcript stay apart:

* consensus block absent, everything else written → `speech.lexical: unavailable`, SPEECH wholly
  unreadable, **critical**, reason `consensus_transcript: <what PREPROCESS recorded>`.
* consensus block written with an empty transcript → `speech.lexical: silent`, SPEECH `declined`,
  **not critical**.

## What the short-circuit does

`routing` still evaluates the whole ruleset and still records the content reading per branch
unchanged — a critical failure rewrites no `route_state`. What it changes is execution:

* no branch runs, and **the declaration adds none**. `forced_by_declaration` is false everywhere,
  because a declaration adding a route here would be the graph pretending to route on evidence it
  does not have. This is the one place the declaration's additive direction is not honoured, and it
  is not an exception to it: nothing is routed at all.
* every decision carries `withheld_critical = true` and `why = route_<state>_withheld_critical`; the
  branch whose gates were all unreadable also carries `critically_unreadable = true`.
* the runner marks every branch `SKIPPED` with the note `WITHHELD_CRITICAL`.
* VERDICT reads `unreadable` and `unavailable` back off the one `ruleset_routing` measurement, so
  the fold and the node that withheld the run cannot disagree about what was missing, and flags on
  `CRITICAL_ABSENCE` naming each branch, each gate and each recorded absence.

**A critical failure flags; it does not discard.** The two discard grounds are claims about the
recording — `unmeasurable` is ADMIT saying it could not be read, `acoustically_empty` is the
emptiness bypass saying it carried nothing. A block that failed to be written is a claim about the
run. The audio may be perfect. No new threshold is introduced by this choice.

QUALITY still runs. It is the terminal node every recording reaches whatever routed, it reads stored
outputs rather than the routing decision, and withholding it would weaken an existing path for no
stated reason. It is not a branch; no branch report is written on this path.

## The four states a branch can be in

The distinction is load-bearing and each state is readable off the store:

| state | how it reads |
| --- | --- |
| routed and ran | `will_run` true, a `branch_report` present |
| ran and found nothing | a `branch_report` present, `findings` `absent` |
| not selected | `will_run` false, `withheld_critical` false, `route_state` `declined`/`ungated` |
| withheld by a critical failure | `will_run` false, `withheld_critical` true, note `WITHHELD_CRITICAL` |

Without `withheld_critical` the last two are the same row: a branch the ruleset declined and a
branch nobody asked both read as `will_run: false, route_state: declined`.

## The no-hint-no-gate case

`routing.py`'s `empty_set` can still be true on a non-critical recording: no gate fired, the stem
declared no task and no hint tag mapped. Under the owner's rule 2 that should not happen. It is
neither a critical failure nor a case that needs a default branch, and the reasoning is recorded in
config-derivations.md § routing under `routing.default_branch`: both routes to it —
`empty` and `unexplained` — already reach a verdict with a reason, a default branch would convert
`unexplained` from a visible ruleset gap into a branch report that looks like a route, and choosing
which branch is exactly the question the ruleset failed to answer. The key ships null so a campaign
that can answer it for its own corpus can, recorded as `route_<state>_by_default`.

In practice the case is rare rather than merely tolerated: every corpus recording's BIDS stem
carries a `task-` entity, the family resolves through `reference_family_set`, and the declaration
adds a route. Only an undeclared recording with no gate firing gets here.

## What was left alone

* ADMIT's refusal path. A critical failure is evaluated after PREPROCESS and cannot reach a
  recording ADMIT failed: the runner never calls `routing` on that path.
* `route_state` and `BRANCH_ROUTE_STATES`. The content reading is what it was; only execution moved.
* `span_longest`'s `0.0`-for-absent reader, noted above.
