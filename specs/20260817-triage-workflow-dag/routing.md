# ROUTING — the branch gate

Decides **which branches run**. [`TAXONOMY`](taxonomy.md) says what kinds are in the recording;
this node turns that classification, plus the hints, into an execution set.

## Signature

```
routing(store, hint?) -> flag(reason, decisions) | pass(decisions)
```

Reads and writes the [element store](store.md). It measures nothing and classifies nothing: it reads
`kind` elements and the hint, and writes one `branch_decision` element per branch.

No `fail`. A file every branch declines is a `flag`, not a failure — see below.

## What it reads

| element | author | used for |
| --- | --- | --- |
| `kind` elements — `speech`, `airway`, `voice` | TAXONOMY | the acoustic decision to run a branch |
| `hint.may_contain`, `hint.metadata.speech_type` | caller, optional | the forcing decision |

## The rule

| kind state | branch |
| --- | --- |
| `present` | **runs** |
| `uncertain` | **runs** |
| `absent` | does not run, unless a hint forces it |

```
speech  present or uncertain  ->  SPEECH runs
airway  present or uncertain  ->  AIRWAY runs
voice   present or uncertain  ->  VOICE runs
```

**Uncertain runs.** The branch is the more precise instrument, and a kind the classification could not
settle is exactly what a branch exists to settle.

## Hints force execution; they never alter the classification

A hint naming a kind — through `may_contain` or the task's `speech_type` — **forces that kind's branch
to run**, whatever the classification said. The forcing is recorded on the branch decision as
`forced_by_hint`, and:

- the `kind` element TAXONOMY wrote is **not** rewritten. A forced branch runs against an `absent`
  classification, and the disagreement between the two is what
  [`verdict.md`](verdict.md) detects as a branch mismatch;
- forcing adds a branch. It never removes one, never relaxes a threshold, and never makes a branch's
  own conclusion more or less likely.

The mapping from a hint tag or `speech_type` value to a kind is the config key
`routing.hint_kind_map`. A tag with no entry forces nothing and is recorded as unmapped.

## REDACT is inside the speech branch

[`REDACT`](redact.md) is a step of SPEECH, not a node beside it. It runs only when SPEECH ran **and**
SPEECH's PII scan over the consensus transcript found something. A file where SPEECH did not run, or
ran and found no PII, has no REDACT verdict at all, and its release axis reads `not_assessed`.

## A file that enters no branch is recorded, not judged

If every kind is `absent` and no hint forces a branch, the execution set is empty. The three
decisions carry `will_run: false` with the kind state and the evidence behind each, and the file
reaches [`verdict.md`](verdict.md) with no branch conclusions to fold. **This node does not `flag`
it.** A flag here would decide the file, because the fold tests any node `flag` before it tests
every-kind-absent, and verdict.md's "acoustically empty → discard" would then be unreachable. The
fold reads `will_run` off the decisions and takes that decision against the hints
([`verdict.md`](verdict.md)).

## When ROUTING itself fails

This is a runner failure, not a classification outcome, and it is handled separately from the rule
above. When `routing()` raises or returns no result, the runner records ROUTING as `errored` and does
not run AIRWAY, SPEECH, VOICE, or REDACT — they are recorded `skipped`, since running them without the
decisions that authorise them would create unaudited conclusions. VERDICT folds that recorded failure
and the absence of decisions into a file `flag` naming the routing failure; it must not discard the
file merely because every kind happened to classify `absent` before ROUTING raised.

## The pass is encapsulated

`PREPROCESS → TAXONOMY → routing` is one unit over **one input stream**. The unit's input type is a
stream: the original recording, or a stream from which an extracted source has been suppressed or
removed.

**The current target runs the unit exactly once, on the original recording.** Every element the unit
writes carries the stream it was computed on, so a second pass over a suppressed-foreground stream is
expressible without any change to the store contract. Nothing in this target invokes one.

## Store contract

One `branch_decision` element per branch, written before any branch runs:

```
branch_decision: {
  branch:        "AIRWAY" | "SPEECH" | "VOICE",
  will_run:      bool,
  kind_state:    "present" | "uncertain" | "absent",     # what TAXONOMY said
  forced_by_hint: bool,
  hint_tags:     [ ... ],                                 # the tags naming this branch's kind;
                                                          # forced_by_hint says whether they changed the outcome
  why:           reason
}
```

`used(routing, kind_element)` records which classification each decision rests on, and
`wasDerivedFrom(branch_decision, kind_element)` ties the two together.

**This is what lets [`verdict.md`](verdict.md) tell a branch that found nothing from a branch that
never looked.** A branch with `will_run: false` contributes no verdict and the fold reads its decision
instead; a branch with `will_run: true` and no verdict errored, and the fold says so.

## Product

```
outcome:   pass          # always; this node reaches no conclusion about the recording
verdict:   { runs: [branch, ...], skipped: [branch, ...], forced: [branch, ...], empty_set: bool }
view:      the branch_decision element ids
```

## Out of scope

Classifying anything, running any branch, reading any acoustic evidence directly, and deciding what an
empty execution set means for the file.

Derivations live in [`benchmarks/`](benchmarks/).

## Open derivations (v2)

| key | what is owed |
| --- | --- |
| `routing.hint_kind_map` | which hint tags and `speech_type` values force which kind's branch; a vocabulary, owed the corpus it was drawn from |
