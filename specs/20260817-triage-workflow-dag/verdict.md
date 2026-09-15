# The file-level verdict

The last fold. Reads the [element store](store.md) — every node's verdict, the branch decisions
[`routing.md`](routing.md) wrote, and the `ruleset_routing` measurement behind them — and answers what
the graph concluded about the recording.

**The fold is keyed by branch.** It was keyed by kind until 2026-09-13, when TAXONOMY's kind
classification was deleted; the replacement emits a route state per branch, DDK has no kind at all,
and inventing a second branch-to-kind map to keep the old key would be the parallel vocabulary the
change removed. A branch's own verdict joins to its decision by node name — `NodeVerdict.node` **is**
the branch name — and only when the verdict names a `kind`, which keeps a synthesised "outcome nobody
can act on" flag from being folded as a finding.

## Signature

```
verdict(store, hint?) -> file_verdict
```

Writes one element. Decides nothing a branch has already decided about its own subject — it
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
| **acoustically empty** | the ruleset's own file-level state is `empty` — the emptiness bypass read every tracked stream peak under `emptiness.peak_floor` — **and no hint claims otherwise** |

The second ground used to be "every kind resolved `absent`", which could never fire while every kind
read `uncertain`; dag.md recorded it as unreachable. It is now a measurement. Its sibling state,
`unexplained` — nothing routed and the recording was *not* empty — adds a ROUTING **flag**, never a
discard: content no gate could account for is a charge against the ruleset, and the two are kept
apart in the product rather than in a reader's head.

**A hint that claims otherwise turns the second ground into a `flag`, never a `discard`.** A file the
graph found nothing in, on which the declaration says there should have been something, is exactly the
case a human must see; discarding it would delete the evidence that the graph was wrong.

The two grounds carry different reasons — "could not measure" and "measured, and there is nothing of
interest in it" — and a consumer that cannot distinguish them treats an empty recording as a broken
one.

## Branch authority is scoped to the branch's own subject

**A branch is the authority on its own subject and on nothing else.** It is the more precise
instrument for what it measures; it is not an instrument for the others.

| the branch's conclusion | its reach |
| --- | --- |
| SPEECH | lexical speech only. It refutes neither AIRWAY's subject nor VOICE's |
| AIRWAY | airway events only |
| VOICE | phonation only |

A branch's conclusion stands in the `findings` map, whatever the ruleset routed, and whether the
branch passed, flagged or failed. A branch that reached no conclusion reads `uncertain` there: not
routed is not a measurement of absence. The resolution axis is
found/not-found, not the outcome's severity: a branch flags only with a subject in hand, so a flag
resolves its subject `present` and the flag travels beside the resolution; a `fail` is the branch
reporting no subject and always resolves `absent`. A branch therefore never raises its own absence
to a flag over a declaration — that mismatch is the fold's to name, from the `hints` table below.

## The route is reported beside the branches, never over them

The route state is written into the product next to the branch conclusions, and the fold records
whether the two **agree** or **mismatch**, per branch. The axis changed with the key: it compared
TAXONOMY's classification against the branch's conclusion, and it now compares the **route** against
it — which is the quantity the ruleset work cares about, countable per branch over a corpus without a
join.

| route | branch conclusion | recorded | triage | what it means |
| --- | --- | --- | --- | --- |
| routed | found | `agree` | — | — |
| routed | not found | `mismatch` | `flag` | over-routing |
| declined | found | `mismatch` | `flag` | a miss — the branch ran only because a hint forced it |
| declined | not found | `agree` | — | — |
| unavailable | either | `resolved` | — | a branch nothing could judge made no claim to agree with |
| any | branch did not conclude | `not_run` — see the branch-decision rows below | — | — |

**A mismatch flags; it never overrides.** The routing cannot overturn a branch on its own subject,
and the branch does not rewrite the decision: both stay in the store and both appear in the product,
so the disagreement is visible rather than resolved by precedence.

## A branch that never ran is not a branch that failed

`routing.md`'s `branch_decision` elements are what distinguish the two:

| branch decision | branch verdict | reading |
| --- | --- | --- |
| `will_run: false`, route `declined`, not forced | none | **expected.** The graph declined to look, and said why |
| `will_run: true` | present | folded as above |
| `will_run: true` | absent | **flag** — the branch was asked and left no answer; the reason names `errored without a verdict`, `completed without a verdict` or `never ran` |
| every branch `will_run: false` | none | the empty execution set — see `discard` above |

**DDK falls in the third row by construction.** The ruleset routes it, no node implements it, so a
recording whose DDK gates fire flags with "DDK was asked to run and never ran". That is honest and
scoped to recordings with DDK content, and it is the standing argument for building the branch.

## Hints are read here, for branch mismatch

The hint's `speech_type` and `may_contain` tags are compared against what the branches concluded.

| case | outcome |
| --- | --- |
| a hinted branch ran and **found nothing** | **`flag`**, naming the mismatch: the branch, and that it was declared and did not find its subject |
| a hinted branch found its subject | recorded as agreement |
| a subject found that no hint claimed | recorded; not a flag on its own |

The claims are read off ROUTING's own record of reading the declaration — the `hint_tags` on each
decision — not re-resolved here against `routing.hint_branch_map`. Two nodes resolving one tag
independently could disagree whenever the config or the hint handed to them differed.

A hint never resolves a subject, never suppresses a branch's conclusion, and never turns a `flag`
into a `pass`. Its one power on this axis is to prevent a `discard` (above) and to name a mismatch.

### `hints` states a falsehood when `hint_branch_map` is null — **owed a code change**

With the packaged `routing.hint_branch_map: null` every declared tag is unmapped, so no decision
carries `hint_tags` and `_hint_claims` returns `{}` rather than `None` (`nodes/verdict.py:182-184`).
The `hints` table is then built (`vocabulary.py:350-357`) through `_hint_reading(claimed=False, …)`
(`vocabulary.py:279-291`), which writes `found_unclaimed` or `no_claim` for **every** branch.

Measured on four runs handed `may_contain: [cough, airway]`: the verdict reads
**`AIRWAY: found_unclaimed`** — an assertion that nobody claimed AIRWAY, on a recording whose
declaration claimed it. It is not confined to the store. It reaches `summary.json`'s
`recording.declared_hints` (`report.py:1097`) and the PDF header (`report.py:1426-1427`, rendered at
`:1467`), verified in a released summary.

`UNREAD_DECLARATION` (`vocabulary.py:385-386`) exists for exactly this case and does not fire:
`_hint_claims` returns `None` only when a hint was supplied and **no decision survived**
(`nodes/verdict.py:182`), not when the decisions survived and could resolve nothing.

**This is a defect, not an owed derivation, and the fix is blocked on a vocabulary decision.** The
product below pins four `hints` tokens; what the field should say when no map is configured — a fifth
token such as `unresolvable`, or `hints: {}` plus the existing flag — is the owner's call and not a
measurement. The upstream causes are registered at
[`../20260913-branch-contract-and-hints/design.md:288-311`](../20260913-branch-contract-and-hints/design.md);
this consequence on the verdict table and the rendered report is registered here and in
[`benchmarks/hints-and-routing-2026-09-15.md`](benchmarks/hints-and-routing-2026-09-15.md) § B.

## The triage fold

Evaluated in order; the first that applies wins.

| order | condition | `triage` |
| --- | --- | --- |
| 1 | ADMIT failed | `discard` — unmeasurable |
| 2 | any node returned `flag`, any mismatch row above fired, the ruleset read `unexplained`, or a branch that was asked to run left no verdict | `flag` |
| 3 | the ruleset read `empty`, and no hint claims otherwise | `discard` — acoustically empty |
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

## A REDACT `fail` does not flip triage, and no non-pass is invisible

Triage answers whether a human must look at the recording; release answers whether an artifact may be
handed on. A surviving PII finding — REDACT's `fail` — is a release problem, so it does not move the
triage axis. A REDACT `flag` **does** flag the file, by the ladder's row 2 like any other node's flag:
it says verification did not finish, which is a question about the graph rather than about the
artifact.

**It appears in the product regardless.** `reasons` carries REDACT's outcome, its surviving
categories and its `unremediable` set on every non-pass, and the [report](report.md) shows it beside
the branch conclusions. A consumer filtering on `triage == pass` sees the release axis in the same
record and cannot mistake one for the other.

## Product

```
triage:   pass | flag | discard
release:  releasable | withheld | not_assessed
discard_ground: "unmeasurable" | "acoustically_empty" | null
reasons:  [ { node, outcome, kind?, why } ]        # every contributing verdict, in order
ran:      { node: "completed" | "skipped" | "errored" }
branches: { branch: { will_run, forced_by_hint, route_state, verdict? } }
findings: { branch: "present" | "absent" | "uncertain" }        # what each branch found
routes:   { branch: "routed" | "declined" | "unavailable" }     # what the ruleset made of it
route_state: "routed" | "empty" | "unexplained" | null          # what it made of the recording
agreement:{ branch: "agree" | "mismatch" | "resolved" | "not_run" }
hints:    { branch: "claimed_and_found" | "claimed_not_found" | "found_unclaimed" | "no_claim" }
view:     the verdict element id, and the node verdict ids it folded
```

`reasons` carries **every** node's contribution, not only the deciding one.

`discard_ground` names which of the two grounds a `discard` rests on, and is null on every other
outcome. It is in the product because the two grounds are the difference between a broken recording
and an empty one, and a consumer must not have to re-derive that from the reasons.

`findings` and `routes` are both present, always. `findings` is what each branch concluded; `routes`
is what the ruleset made of it. Keeping both is what makes `agreement` checkable by a reader rather
than asserted by this node. `route_state` is null only when ROUTING wrote no evaluation, which is a
reading never made rather than a recording nothing routed.

`ran` comes from two sources and is **merged, the runner's over the store's**: the store derives
`completed` for a node that wrote a verdict, `errored` for one that wrote an activity and no live
verdict, and `skipped` for one that wrote neither; the runner's mapping then overrides per node.
`branches` is the routing decision joined to the branch verdict, so a `skipped` branch carries the
reason it was skipped.

Every read of the store here follows the store's shared rule — an invalidated element is never read,
and of the survivors asserting the same thing the latest write wins, per node for verdicts and per
branch for decisions.

## Out of scope

Ranking recordings, choosing what to do about a flag, overriding a branch on its own subject, and any
threshold that would turn a `flag` into a `pass`.

Derivations live in [`benchmarks/`](benchmarks/).
