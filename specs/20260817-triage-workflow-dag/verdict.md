# The file-level verdict

The last fold. Reads the [element store](store.md) — every node's verdict, the branch decisions
[`routing.md`](routing.md) wrote, and the kinds [`TAXONOMY`](taxonomy.md) classified — and answers what
the graph concluded about the recording.

## Signature

```
verdict(store, hint?) -> file_verdict
```

Writes one element. Decides nothing a branch has already decided about its own kind — it
**combines**, and where the graph disagrees with itself it says so.

## Two axes, because they answer different questions

| axis | question | values |
| --- | --- | --- |
| `triage` | what should happen to this recording | `pass` \| `flag` \| `discard` |
| `release` | is a redacted artifact safe to hand on | `releasable` \| `withheld` \| `not_assessed` |

Collapsing them would make a recording with clean measurements and surviving PII look like a
measurement problem, and a recording with an empty transcript and no PII look releasable.

## `discard` is a narrow outcome

**A branch `fail` is not a file `discard`.** A branch `fail` means "this branch has no subject" — a
cough recording has no speech, so SPEECH failing is the expected outcome.

`discard` has exactly two grounds:

| ground | condition |
| --- | --- |
| **unmeasurable** | ADMIT failed. Nothing ran and nothing is claimed about the recording |
| **acoustically empty** | every kind classified `absent`, no branch ran or every branch that ran found nothing, **and no hint claims otherwise** |

**A hint that claims otherwise turns the second ground into a `flag`, never a `discard`.** A file the
graph found nothing in, on which the declaration says there should have been something, is exactly the
case a human must see; discarding it would delete the evidence that the graph was wrong.

The two grounds carry different reasons — "could not measure" and "measured, and there is nothing of
interest in it" — and a consumer that cannot distinguish them treats an empty recording as a broken
one.

## Branch authority is scoped to the branch's own kind

**A branch is the authority on its own kind and on nothing else.** It is the more precise instrument
for the kind it measures; it is not an instrument for the others.

| the branch's conclusion | its reach |
| --- | --- |
| SPEECH | the `speech` kind only. It refutes neither `airway` nor `voice` |
| AIRWAY | the `airway` kind only |
| VOICE | the `voice` kind only |

A branch's conclusion about its own kind stands in the resolved `kinds` map, whatever the
classification said, and whether the branch passed, flagged or failed. A branch that flagged still
resolves its kind — the flag travels beside the resolution and is not a reason to withhold it.

## TAXONOMY is reported beside the branches, never over them

The classification is written into the product next to the branch conclusions, and the fold records
whether the two **agree** or **mismatch**, per kind:

| classification | branch conclusion | recorded | triage |
| --- | --- | --- | --- |
| present | found | `agree` | — |
| absent | found | `mismatch` | `flag` |
| present | not found | `mismatch` | `flag` |
| uncertain | either | `resolved` | — |
| any | branch did not run | see the branch-decision rows below | — |

**A mismatch flags; it never overrides.** The classification cannot overturn a branch on the branch's
own kind, and the branch does not delete the classification: both stay in the store and both appear in
the product, so the disagreement is visible rather than resolved by precedence.

## A branch that never ran is not a branch that failed

`routing.md`'s `branch_decision` elements are what distinguish the two:

| branch decision | branch verdict | reading |
| --- | --- | --- |
| `will_run: false`, kind `absent`, not forced | none | **expected.** The graph declined to look, and said why |
| `will_run: true` | present | folded as above |
| `will_run: true` | absent | **flag** — the branch was asked and left no answer; the reason names `errored without a verdict`, `completed without a verdict` or `never ran` |
| every branch `will_run: false` | none | the empty execution set — see `discard` above |

## Hints are read here, for branch mismatch

The hint's `speech_type` and `may_contain` tags are compared against what the branches concluded.

| case | outcome |
| --- | --- |
| a hinted kind's branch ran and **found nothing** | **`flag`**, naming the mismatch: the kind, the hint that claimed it, and the branch's conclusion |
| a hinted kind's branch found the kind | recorded as agreement |
| a kind found that no hint claimed | recorded; not a flag on its own |

A hint never resolves a kind, never suppresses a branch's conclusion, and never turns a `flag` into a
`pass`. Its one power on this axis is to prevent a `discard` (above) and to name a mismatch.

## The triage fold

Evaluated in order; the first that applies wins.

| order | condition | `triage` |
| --- | --- | --- |
| 1 | ADMIT failed | `discard` — unmeasurable |
| 2 | any node returned `flag`, any mismatch row above fired, or a branch that was asked to run left no verdict | `flag` |
| 3 | every kind absent, nothing found, and no hint claims otherwise | `discard` — acoustically empty |
| 4 | otherwise | `pass` |

**The graph's stated goal is to be accurate about `pass` and `discard` and to minimise `flag`.** A
fold that flags everything transports no information; a reason that fires on nearly every file is a
reason to re-derive, not a reason to keep flagging.

## The release fold

| condition | `release` |
| --- | --- |
| REDACT did not run — no speech branch, no words, or no PII found | `not_assessed` |
| REDACT returned `fail` — a finding survived verification | `withheld` |
| REDACT returned `flag` — verification was incomplete | `withheld`; unresolved is not cleared |
| REDACT returned `pass` | `releasable`, for **its artifacts only** |

Only `pass` clears an artifact, which makes the mapping total.

**`releasable` never applies to the store.** The store holds the unredacted consensus transcript by
design and is append-only. `release` describes REDACT's artifacts and nothing else.

**`not_assessed` is not `releasable`.** A recording with no speech, or with speech and no PII, was
never redacted, and must not be read as cleared of content a transcript could not carry.

**The goal on this axis is to minimise `withheld`**: a withhold is a file no consumer can use, and
every withhold that rests on a scan of text nobody uttered is one the graph created.

## A REDACT non-pass does not flip triage, and is never invisible

Triage answers whether a human must look at the recording; release answers whether an artifact may be
handed on. A surviving PII finding is a release problem, so it does not move the triage axis.

**It appears in the product regardless.** `reasons` carries REDACT's outcome, its surviving
categories and its `unremediable` set on every non-pass, and the [report](report.md) shows it beside
the branch conclusions. A consumer filtering on `triage == pass` sees the release axis in the same
record and cannot mistake one for the other.

## Product

```
triage:   pass | flag | discard
release:  releasable | withheld | not_assessed
reasons:  [ { node, outcome, kind?, why } ]        # every contributing verdict, in order
ran:      { node: "completed" | "skipped" | "errored" }
branches: { branch: { will_run, forced_by_hint, kind_state, verdict? } }
kinds:    { airway: state, speech: state, voice: state }        # after branch resolution
screened: { airway: state, speech: state, voice: state }        # what TAXONOMY classified
agreement:{ kind: "agree" | "mismatch" | "resolved" | "not_run" }
hints:    { kind: "claimed_and_found" | "claimed_not_found" | "found_unclaimed" | "no_claim" }
view:     the verdict element id, and the node verdict ids it folded
```

`reasons` carries **every** node's contribution, not only the deciding one.

`kinds` and `screened` are both present, always. `kinds` is the resolved state after branch authority;
`screened` is what TAXONOMY classified. Keeping both is what makes `agreement` checkable by a reader
rather than asserted by this node.

`ran` comes from two sources and is **merged, the runner's over the store's**: the store derives
`completed` for a node that wrote a verdict, `errored` for one that wrote an activity and no live
verdict, and `skipped` for one that wrote neither; the runner's mapping then overrides per node.
`branches` is the routing decision joined to the branch verdict, so a `skipped` branch carries the
reason it was skipped.

Every read of the store here follows the store's shared rule — an invalidated element is never read,
and of the survivors asserting the same thing the latest write wins, per node for verdicts and per
kind for classifications.

## Out of scope

Ranking recordings, choosing what to do about a flag, overriding a branch on its own kind, and any
threshold that would turn a `flag` into a `pass`.

Derivations live in [`benchmarks/`](benchmarks/).
