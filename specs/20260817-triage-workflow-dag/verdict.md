# The file-level verdict

The last fold. Reads the [element store](store.md) and every node's verdict, and answers what the graph
concluded about the recording.

## Signature

```
verdict(store) -> file_verdict
```

Writes one element. Decides nothing a node has not already decided — it **combines**, and where the
nodes contradict each other it says so rather than choosing.

## Two axes, because they answer different questions

| axis | question | values |
| --- | --- | --- |
| `triage` | does this recording need a human, and is it measurable | `pass` \| `flag` \| `fail` |
| `release` | is a redacted artifact safe to hand on | `releasable` \| `withheld` \| `not_assessed` |

Collapsing them would make a recording with clean measurements and surviving PII look like a
measurement problem, and a recording with an empty transcript and no PII look releasable. They are
independent and stay so.

## A branch `fail` is not a file `fail`

**This is the rule everything else depends on.** A branch `fail` means "this branch has no subject" — a
cough recording has no speech, so SPEECH failing is the expected outcome, not an error. Treating branch
fails as file fails would fail almost every recording.

So a branch's outcome is read **against what TAXONOMY predicted for its kind**. TAXONOMY is advisory and
every branch runs, so both contradiction rows below are reachable — under a gated run they are not, and a
gated run is marked so a reader knows the check did not happen:

| TAXONOMY said | branch outcome | reading |
| --- | --- | --- |
| absent | `fail` | **expected.** Contributes nothing |
| absent | `pass` | the kind was present after all — resolve the `kind` element to present, and **flag**: the screen and the branch disagree |
| present | `pass` | confirmed |
| present | `fail` | **contradiction.** The screen found the kind and the branch found no subject → **flag** |
| undecided | `pass` | resolved to present |
| undecided | `fail` | resolved to absent |
| — | `flag` | **flag**, whatever the screen said |
| present or undecided | **never ran** | **flag** — a kind the graph was asked about has no answer |
| absent | **never ran** | **expected.** Contributes nothing, exactly as `fail` would |

**A branch that never ran is not a branch that failed**, and an earlier version of this table had no row
for it. The difference matters: `fail` is a branch reporting it has no subject, which is evidence; not
running is the absence of evidence, and on a kind the screen called present or undecided that is a gap a
human should see. The two must not collapse, because a graph that skipped a node for an operational
reason — a model unavailable, a budget exhausted, a crash — would otherwise be indistinguishable from one
that looked and found nothing.

The two contradiction rows are the reason this node exists. A graph that disagrees with itself is exactly
what a human should look at, and neither the screen nor the branch is entitled to overrule the other on
evidence this design has not measured.

## The triage fold

Evaluated in order; the first that applies wins.

| order | condition | `triage` |
| --- | --- | --- |
| 1 | ADMIT failed | `fail` — nothing ran, and nothing is claimed about the recording |
| 2 | any node returned `flag`, any contradiction row above fired, or a branch for a present or undecided kind never ran | `flag` |
| 3 | every kind is absent, so no branch had a subject | `fail` |
| 4 | otherwise | `pass` |

`fail` at 1 and `fail` at 3 are different findings and carry different reasons: the first is "could not
measure", the second is "measured, and there is nothing of interest in it". A consumer that cannot
distinguish them will treat an empty recording as a broken one.

## The release fold

| condition | `release` |
| --- | --- |
| REDACT did not run | `not_assessed` |
| REDACT returned `fail` — a finding survived verification | `withheld` |
| REDACT returned `flag` — verification was weakened or contested | `withheld`; unresolved is not cleared |
| REDACT returned `pass` | `releasable`, for **its artifacts only** |

Only `pass` clears an artifact, which makes the mapping total: an `Outcome` member added later
withholds rather than defaulting to cleared.

**`releasable` never applies to the store.** The store holds the unredacted transcript by design and is
append-only, so nothing can make it releasable. `release` describes REDACT's artifacts and nothing else.

**`not_assessed` is not `releasable`.** A recording with no speech has no PII scan and no redaction, and
must not be read as cleared — the audio was never examined for content the transcript could not carry.

## Product

```
triage:   pass | flag | fail
release:  releasable | withheld | not_assessed
reasons:  [ { node, outcome, kind?, why } ]        # every contributing verdict, in order
ran:      { node: "completed" | "skipped" | "errored" }            # so a gap is not read as a finding
kinds:    { airway: state, speech: state, voice_no_words: state }   # after resolution
view:     the verdict element id, and the node verdict ids it folded
```

`reasons` carries **every** node's contribution, not only the deciding one. A `flag` that names one cause
hides the others, and a reader deciding what to do next needs the whole set.

`kinds` is the resolved state after the table above, which may differ from what TAXONOMY wrote. Both
remain in the store — TAXONOMY's assertion and this node's resolution — so the change is visible rather
than silent.

`ran` comes from two sources and is **merged, the runner's over the store's**: the store derives
`completed` for every node that wrote a verdict and `skipped` for every other graph node, and the
runner's mapping then overrides per node. Only the runner can report `errored`, because a node that
raised wrote no verdict and the store cannot tell it from one never asked to run; and only the store
can prove `completed`, so a partial mapping from the runner overrides what it names without erasing
the rest.

Every read of the store here follows the store's shared rule — an invalidated element is never read,
and of the survivors asserting the same thing the latest write wins, per node for verdicts and per
kind for screens. A withdrawn verdict therefore does not vote, and a repaired node contributes once,
as its repair.

## Out of scope

Ranking recordings, choosing what to do about a flag, and any threshold that would turn a `flag` into a
`pass`. This node folds verdicts; it does not weigh them.
