# ROUTING — the branch gate

Decides **which branches run**. It evaluates the family taxonomy ruleset over the store
[`TAXONOMY`](taxonomy.md) has just finished writing, records that reading, and turns it — plus the
hints — into an execution set.

Until 2026-09-13 it read `kind` elements instead. Those emitted a constant and are gone; the staging
is in [`../20260912-ruleset-in-pipeline/design.md`](../20260912-ruleset-in-pipeline/design.md).

## Signature

```
routing(store, source, config, hint=None, *, run_dir) -> pass(decisions)
```

Reads and writes the [element store](store.md). It measures nothing and classifies nothing: it
evaluates eleven declarative gates over the evidence already in the store, writes the
`ruleset_routing` measurement, and writes one `branch_decision` element per branch.

No `fail`. A file every branch declines is a `pass` carrying an empty execution set — see below.

**A reading that cannot be made raises.** The ruleset decides what runs, so a failure to evaluate it
is a failure of the node: the runner records ROUTING `errored` and every branch `skipped`. It used to
be caught and recorded on the measurement, which was correct only while the reading decided nothing.

## What it reads

| element | author | used for |
| --- | --- | --- |
| PREPROCESS's whole derivative set — spans, consensus words, per-span classifier scores, the residual reading, the posteriorgram sidecar | PREPROCESS | the eleven gates' feature paths |
| `yamnet_label_summary` | TAXONOMY | `voice.glide` and `voice.chant` read `plain|yamnet` off it, which is why this node runs after TAXONOMY |
| `taxonomy.ruleset` | config | every gate's feature path, comparison and threshold, and the emptiness bypass |
| `routing.hint_branch_map` | config | which declared tag names which branch |
| `hint.may_contain`, `hint.metadata.speech_type` | caller, optional | the forcing decision |

No number in this node is a literal. The operating points, which of them are provisional, and the
open questions are in [`family-taxonomy-ruleset.md`](family-taxonomy-ruleset.md), keyed by gate name.

## The rule

| route state | meaning | branch |
| --- | --- | --- |
| `routed` | a gate fired for the branch | **runs** |
| `declined` | every gate was evaluated and none fired | does not run, unless a hint forces it |
| `unavailable` | no gate fired and at least one of the branch's gates could not read its feature | does not run, unless a hint forces it |

**`unavailable` is kept apart from `declined` and does not run the branch.** A gate whose measurement
was never written did not decline — that distinction is load-bearing throughout the analysis and is
recorded per branch on the decision, in `unavailable_gates`. It does *not* run the branch: "anything
not absent runs" is the rule that made the old fold inert, and adopting the ruleset is adopting its
declines. What the distinction buys is the fold's agreement axis, which reads `resolved` rather than
`agree`/`mismatch` for a branch nothing could judge ([`verdict.md`](verdict.md)).

**DDK is routable.** The ruleset routes four branches and `BRANCHES` has always named four, so DDK
gets a decision like any other. No node implements it: the runner looks the branch up rather than
indexing it and records `SKIPPED` with `no node implements this branch`. A recording whose DDK gates
fire therefore reaches the fold with `will_run` true and no verdict, and the file **flags**. That is
the graph saying it has no instrument for what is in the recording, and it is the standing argument
for building the branch. See [`dag.md`](dag.md), "DDK".

**Flag gates annotate; they never route.** A fired flag gate is recorded on its branch's decision in
`flag_gates` and changes no `will_run`.

## Hints force execution; they never alter the reading

A hint naming a branch — through `may_contain` or the task's `speech_type` — **forces that branch to
run**, whatever the ruleset made of it. The forcing is recorded as `forced_by_hint`, and:

- the route state is **not** rewritten. A forced branch runs against a `declined` route, and the
  disagreement between the two is what [`verdict.md`](verdict.md) detects as a mismatch;
- forcing adds a branch. It never removes one, never relaxes a threshold, and never makes a branch's
  own conclusion more or less likely.

The mapping from a hint tag or `speech_type` value to a branch is the config key
`routing.hint_branch_map`. A tag with no entry forces nothing and is recorded as unmapped; a tag
whose entry names something that is not a branch is recorded as unmapped **and** named in
`bad_map_values`, because a one-character typo under-claims every file in a run and is a different
thing to chase from a tag the vocabulary does not cover.

The hint layer is due its own design pass. This node's job here was re-keying it from kinds to
branches, and the forcing behaviour is otherwise unchanged.

## REDACT is inside the speech branch

[`REDACT`](redact.md) is a step of SPEECH, not a node beside it. It runs only when SPEECH ran **and**
SPEECH's PII scan over the consensus transcript found something. A file where SPEECH did not run, or
ran and found no PII, has no REDACT verdict at all, and its release axis reads `not_assessed`.

## A file that enters no branch is recorded, not judged

If no gate fires and no hint forces a branch, the execution set is empty. Every decision carries
`will_run: false` with its route state, and the file reaches [`verdict.md`](verdict.md) with no
branch conclusions to fold. **This node does not `flag` it.** A flag here would decide the file,
because the fold tests any node `flag` before it tests the recording's own emptiness, and verdict.md's
"acoustically empty → discard" would be unreachable again.

What the fold reads instead is the evaluation's own file-level state, which distinguishes the two
reasons an execution set can be empty:

| `state` | meaning | the fold's reading |
| --- | --- | --- |
| `routed` | at least one gate fired | — |
| `empty` | the bypass read every tracked stream peak under `emptiness.peak_floor` | `discard`, ground `acoustically_empty` |
| `unexplained` | nothing routed, and the recording was not measurably empty | `flag` — a charge against the ruleset, never against the file |

## When ROUTING itself fails

A runner failure, handled separately from the rule above. When `routing()` raises, the runner records
ROUTING `errored` and does not run AIRWAY, SPEECH, VOICE or REDACT — they are recorded `skipped`,
since running them without the decisions that authorise them would create unaudited conclusions.
VERDICT folds that recorded failure and the absence of decisions into a file `flag` naming the
routing failure; it must not discard the file merely because no decision survived.

## The pass is encapsulated

`PREPROCESS → TAXONOMY → routing` is one unit over **one input stream**. The unit's input type is a
stream: the original recording, or a stream from which an extracted source has been suppressed or
removed.

**The current target runs the unit exactly once, on the original recording.** Every element the unit
writes carries the stream it was computed on, so a second pass over a suppressed-foreground stream is
expressible without any change to the store contract. Nothing in this target invokes one.

## Store contract

One `measurement` named `ruleset_routing`, on this node's `ruleset_routing` step, carrying `state`,
`routed`, `gate_outcomes`, `unavailable`, `flags`, `sources` and `stem` — the shape is in
[`../20260912-ruleset-in-pipeline/design.md`](../20260912-ruleset-in-pipeline/design.md). Then one
`branch_decision` element per branch, written before any branch runs:

```
branch_decision: {
  branch:            "AIRWAY" | "SPEECH" | "VOICE" | "DDK",
  will_run:          bool,
  route_state:       "routed" | "declined" | "unavailable",
  unavailable_gates: [ ... ],   # this branch's gates that could not read their feature
  flag_gates:        [ ... ],   # this branch's flag gates that fired; a flag never routes
  forced_by_hint:    bool,
  hint_tags:         [ ... ],   # the tags naming this branch; forced_by_hint says whether they
                                # changed the outcome
  unmapped_tags:     [ ... ],
  bad_map_values:    { tag: value },
  why:               "route_<state>" | "route_<state>_forced_by_hint",
  stream:            name
}
```

`used(routing, ruleset_routing)` records the reading each decision rests on, and
`wasDerivedFrom(branch_decision, ruleset_routing)` ties the two together.

**This is what lets [`verdict.md`](verdict.md) tell a branch that found nothing from a branch that
never looked.** A branch with `will_run: false` contributes no verdict and the fold reads its
decision instead; a branch with `will_run: true` and no verdict errored, and the fold says so.

## Product

```
outcome:   pass          # always; this node reaches no conclusion about the recording
verdict:   { runs: [...], skipped: [...], forced: [...], empty_set: bool,
             route_state: state, routes: { branch: route_state } }
view:      the ruleset_routing measurement id, then the branch_decision ids
```

## Out of scope

Measuring anything, running any branch, and deciding what an empty execution set means for the file.

Derivations live in [`family-taxonomy-ruleset.md`](family-taxonomy-ruleset.md) and
[`benchmarks/`](benchmarks/).

## Open derivations

| key | what is owed |
| --- | --- |
| `routing.hint_branch_map` | which hint tags and `speech_type` values force which branch; a vocabulary, owed the corpus it was drawn from. **Null** in the packaged config, so every tag is unmapped and nothing is forced |
| the ruleset's operating points | scored 0.97 / 0.95 / 0.96 / 0.94 sensitivity across AIRWAY / SPEECH / VOICE / DDK over 62,547 recordings, 333 of them (0.5%) reaching no branch. Which gates are provisional is in [`family-taxonomy-ruleset.md`](family-taxonomy-ruleset.md) |
| the `unexplained` population | 0.5% of the corpus at the scored operating points, and nobody has looked at what is in those recordings. They now flag rather than passing silently, which is what makes the question askable |
