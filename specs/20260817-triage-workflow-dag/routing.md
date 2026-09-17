# ROUTING — the branch gate

Decides **which branches run**. It evaluates the family taxonomy ruleset over the store
[`TAXONOMY`](taxonomy.md) has just finished writing, records that reading, and turns it — plus the
recording's own declaration — into an execution set.

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
| `taxonomy.ruleset.reference_family_set` | config | which task family declares which branch |
| the `recording` stream's `path` | ADMIT | the `task-` id, hence the declared family |
| `routing.hint_branch_map` | config | which declared tag names which branch |
| `hint.may_contain`, `hint.metadata.speech_type` | caller, optional | a second, optional declaration source |

No number in this node is a literal. The operating points, which of them are provisional, and the
open questions are in [`family-taxonomy-ruleset.md`](family-taxonomy-ruleset.md), keyed by gate name.

## The rule

| route state | meaning | branch |
| --- | --- | --- |
| `routed` | a gate fired for the branch | **runs** |
| `declined` | every gate was evaluated and none fired | does not run, unless the declaration names it |
| `unavailable` | no gate fired and at least one of the branch's gates could not read its feature | does not run, unless the declaration names it |

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

### Measured: recall 13/13, and precision is what produced every flag

On 13 b2ai v3.1 recordings across three subjects, every declared branch was reached — AIRWAY 4/4,
VOICE 3/3, SPEECH 4/4, DDK 2/2 — and every whole-file state read `routed`, so neither `empty` nor
`unexplained` was exercised. Against that, VOICE routed 3 speech tasks, AIRWAY 4 non-airway tasks and
DDK 3 pure-speech tasks; the files came out **5 `pass`, 8 `flag`, no `discard`**. Each flag traces to
one of three things, and none of them is a missed route: a branch running on material it has no
subject in ([`branch-voice.md`](branch-voice.md)), a routed branch with no node
([`branch-ddk.md`](branch-ddk.md)), or a route taken on a label that is wrong
([`benchmarks/hints-and-routing-2026-09-15.md`](benchmarks/hints-and-routing-2026-09-15.md) § G).
Thirteen recordings fit nothing; what they show is which axis to measure next.

## The declared task always adds its branch; it never alters the reading

**Owner decision: the routing always adds a route to branches based on the task declaration.** A
branch the declaration names **runs**, whatever the ruleset made of it, and:

- the route state is **not** rewritten. A declared branch runs against a `declined` route, and the
  disagreement between the two is what [`verdict.md`](verdict.md) detects as a mismatch;
- the declaration adds a branch. It never removes one, never relaxes a threshold, and never makes a
  branch's own conclusion more or less likely. A branch content routed stays routed and is still
  recorded as content-routed;
- both sources naming the same branch is one route, not two. `will_run` is a boolean per branch and
  the execution set is built by one pass over `BRANCHES`, so nothing can be entered twice.

### The declaration is the task, and the task comes off the recording's own path

Two sources could supply it and the authoritative one is the task itself:

1. **The derived task family** (always consulted). ADMIT writes the source `path` onto the
   `recording` stream. `live_evidence.declared_task` reads the stem's `task-` id through
   `families.task_id_of` / `task_family`, and `evaluate_routes` resolves that family through
   `taxonomy.ruleset.reference_family_set` into `RouteEvaluation.declared` — which is the field the
   offline analysis has always filled and the graph deliberately left empty. ROUTING now reads it.
2. **A hint tag** (optional, unchanged). `hint.may_contain` and `hint.metadata.speech_type` through
   `routing.hint_branch_map`.

**Why the family and not the tags.** A family → branch mapping needs no new vocabulary: the graph
already holds exactly one, `reference_family_set`, and it is derived in
[`family-taxonomy-ruleset.md`](family-taxonomy-ruleset.md). Populating `hint_branch_map` instead
would mean inventing one. The corpus's own tag vocabulary is measurable and does not name branches:
`speech_type` takes `non-lexical`, `read`, `elicited` and `recall` — `non-lexical` alone spans
VOICE, DDK and AIRWAY, so no entry for it is derivable — and `recording_profile_name` takes
`Speech`, `Breathe`, `Cough`, which is a coarser second copy of the family knowledge and would be a
second thing to keep in step. `AudioHints` carries no task field at all, so a caller could not name
the task even if asked to. `hint_branch_map` therefore **stays null**, as the optional second
source, additive in the same direction; the derivation is in
[`config-derivations.md`](config-derivations.md).

A recording whose stem carries no `task-` entity and whose caller passed no hint declares nothing
and routes by content alone, exactly as before. A stem spelling `task-unknown` literally is read as
no declaration, because that string is `task_id_of`'s own sentinel for "no task entity"; no family
set names `unknown`, so it could add no route either way.

`reference_family_set` is therefore now **both** the reference standard the routed set is scored
against and the router the declaration goes through. It still skips no gate and rewrites no route
state, so `routed` stays the content reading alone in every consumer.

A tag with no map entry adds nothing and is recorded as unmapped; a tag whose entry names something
that is not a branch is recorded as unmapped **and** named in `bad_map_values`, because a
one-character typo under-claims every file in a run and is a different thing to chase from a tag the
vocabulary does not cover.

### `forced_by_hint` no longer carried the right meaning and was renamed

The added route is now usually a family's, not a hint's, so the flag naming it was wrong. It is
`forced_by_declaration` — declared **and** not content-routed, which is exactly the route the
declaration created — and `declared` beside it is the claim itself, true whether or not it changed
the outcome. `why` reads `route_<state>_forced_by_declaration`. `declared_family` and
`declared_by_family` name *which* source declared it, so a hint-added branch and a family-added
branch stay tellable apart. No parallel field and no alias: pre-alpha.

### Measured: +3,132 routes over 62,547 recordings, and no file-level state changed

Reduced from the finished features shard at
`/orcd/scratch/bcs/002/satra/gatematrix_20260915/evidence/features` through the same
`evaluate_routes` the graph runs, so the added routes are exactly `RouteEvaluation.missed`.

| branch | content routes | declared routes | added by declaration |
| --- | --- | --- | --- |
| AIRWAY | 23,606 | 13,017 | **362** |
| SPEECH | 41,565 | 41,224 | **1,905** |
| VOICE | 22,277 | 8,306 | **361** |
| DDK | 22,363 | 7,989 | **504** |
| total | 109,811 | 70,536 | **3,132** |

2,955 recordings (4.7%) gain at least one route; 3,132 / 109,811 is **+2.85%** on the execution set.
Every one of the 48 families declares at least one branch, so no recording is left declaring
nothing: `recordings_with_no_declared_branch` is 0. Content routes are untouched by construction —
the reduction changes no gate and no threshold.

**The file-level `state` is unchanged**, because it reads `routed` and the emptiness bypass and
neither moves: 62,023 `routed`, 337 `unexplained`, 187 `empty`, before and after. That is the
consequence worth naming: all **524** recordings whose content routed nothing now run their declared
branch while the whole-recording state still says `empty` or `unexplained`. For the 187 `empty`
ones, [`verdict.md`](verdict.md)'s `acoustically_empty` discard is tested *after* any node flag, so a
declared branch that now runs and disagrees with its `declined` route will flag the file instead of
discarding it. How many of the 524 change triage that way is **not measurable from the shard** — it
needs the branches to actually run — and is owed a graph run.

The earlier hint measurement stands and is what this decision replaces: thirteen b2ai v3.1
recordings run twice, hinted and unhinted, differed in exactly one field, `unmapped_tags`, with
`forced_by_hint` false on 52/52 decisions, and replaying a populated `hint_branch_map` offline would
have forced **zero** branches — because the content ruleset had already routed every declared branch
on that material. Thirteen recordings said nothing about the population where content and
declaration disagree; the 3,132 routes above are that population, measured. The values and the diff
method are in [`benchmarks/hints-and-routing-2026-09-15.md`](benchmarks/hints-and-routing-2026-09-15.md) § A.

**Gate values are not recorded anywhere.** `route_attributes` (`live_evidence.py:172-190`) writes
`gate_outcomes` as `{name: outcome}`; the number each gate compared against its cut is in no element,
so every value in that benchmark had to be recovered by re-reducing the finished store through
`extract_features` and `gate_value`. **The owner decided this on 2026-09-15, from the other side:** a
fired rule may write or refine a span's label
([`family-taxonomy-ruleset.md`](family-taxonomy-ruleset.md) § *A fired rule may write or refine a
span's label*), and a label citing neither the value nor the span it came from is indistinguishable
from a classifier's own. So carrying the values is no longer an open question — **owed a code
change**, a store-contract change and not a threshold, and it is also what makes a shipped cut
auditable from a run rather than only from a re-reduction.

## REDACT is inside the speech branch

[`REDACT`](redact.md) is a step of SPEECH, not a node beside it. It runs only when SPEECH ran **and**
SPEECH's PII scan over the consensus transcript found something. A file where SPEECH did not run, or
ran and found no PII, has no REDACT verdict at all, and its release axis reads `not_assessed`.

## A file that enters no branch is recorded, not judged

If no gate fires and the recording declares nothing, the execution set is empty. Every decision carries
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
`routed`, `declared`, `family`, `gate_outcomes`, `unavailable`, `flags`, `sources` and `stem` — the
shape is in [`../20260912-ruleset-in-pipeline/design.md`](../20260912-ruleset-in-pipeline/design.md).
`declared` and `family` are new: the declaration that added a route must be auditable from the run
rather than only from re-deriving it off the stem. Then one `branch_decision` element per branch,
written before any branch runs:

```
branch_decision: {
  branch:                "AIRWAY" | "SPEECH" | "VOICE" | "DDK",
  will_run:              bool,
  route_state:           "routed" | "declined" | "unavailable" | "ungated",  # content only, never rewritten
  unavailable_gates:     [ ... ],   # this branch's gates that could not read their feature
  flag_gates:            [ ... ],   # this branch's flag gates that fired; a flag never routes
  declared:              bool,      # the declaration named this branch, however it ran
  forced_by_declaration: bool,      # declared and NOT content-routed: the route the declaration added
  declared_family:       name,      # the task family the stem declares; "" when it declares none
  declared_by_family:    bool,      # that family named this branch, as against a hint tag naming it
  hint_tags:             [ ... ],   # the tags naming this branch, when a hint supplied any
  unmapped_tags:         [ ... ],
  bad_map_values:        { tag: value },
  why:                   "route_<state>" | "route_<state>_forced_by_declaration",
  stream:                name
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
verdict:   { runs: [...], skipped: [...], forced: [...], declared: [...],
             declared_family: name, empty_set: bool,
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
| `routing.hint_branch_map` | **No longer owed a derivation, and deliberately still null.** The declared route now comes from the task family through `reference_family_set`, which needs no new vocabulary; the corpus's tag vocabulary does not name branches (`non-lexical` spans three) and `recording_profile_name` would be a second copy of the family knowledge. It stays as the optional caller-supplied second source. What is owed is only the corpus a *caller* would draw tags from, if one ever supplies tags a family cannot express |
| the triage effect of the 524 newly-routed empty recordings | 337 `unexplained` and 187 `empty` recordings now run their declared branch while the whole-recording state is unchanged. For the 187, `verdict.md` tests any node flag before the `acoustically_empty` discard, so a declared branch disagreeing with its `declined` route flags the file instead of discarding it. How many change triage needs the branches to run and is **not** derivable from the features shard; owed a graph run |
| the ruleset's operating points | scored 0.97 / 0.95 / 0.96 / 0.94 sensitivity across AIRWAY / SPEECH / VOICE / DDK over 62,547 recordings, 333 of them (0.5%) reaching no branch. Which gates are provisional is in [`family-taxonomy-ruleset.md`](family-taxonomy-ruleset.md). **Recall is not the axis under strain on real material**: 13/13 declared branches were reached in the 2026-09-15 run and every flag traced to over-routing, to a branch with no subject, or to a wrong label |
| the gate values behind an evaluation | the `ruleset_routing` measurement records each gate's outcome and not the number behind it, so a run cannot be audited against its own cuts without re-reducing the store. **Decided 2026-09-15: the evaluation carries them**, because a fired rule may now write a span's label and such a label must cite the rule, the evidence and the value. **Owed a code change**, no longer a decision. The span identity is owed with it: `live_spans` rows carry `"id"` (`routing_analysis/features.py:1084`) and `span_longest_s[measure] = max(durations)` (`:1134`) keeps only the scalar, so no fired gate can name the span it read |
| the `unexplained` population | 0.5% of the corpus at the scored operating points, and nobody has looked at what is in those recordings. They now flag rather than passing silently, which is what makes the question askable |
