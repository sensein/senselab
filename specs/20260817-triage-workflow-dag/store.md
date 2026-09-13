# The element store

Nodes do not pass products to each other. They **write elements to a store**, and later nodes refine
those elements by asserting over them. Any node may read anything in the store.

## Why not dataflow

A strict `A → B` graph forces two things this design cannot afford. It makes every shared product an
input to declare in advance, so a derivative with one consumer looks like a mistake even when it is
shared machinery. And it gives the last writer the final word, so a node that contests an earlier claim
either overwrites it — destroying the evidence — or has nowhere to put its disagreement.

The store fixes both by being **append-only**.

## The model is W3C PROV

The store is a **PROV document**: three node types and a fixed set of relations, so provenance is the
structure rather than a field bolted onto one. Nothing here invents vocabulary that PROV already has.

| PROV term | here |
| --- | --- |
| **Entity** | something the graph believes exists — a span, a word, a speaker, a measurement, a stream, a kind, a verdict. **An assertion is also an Entity**, which is why it has an id |
| **Activity** | one node's execution, or one step of one — `PREPROCESS`, `AIRWAY.classify`. Carries the parameters it ran with |
| **Agent** | what acted: a model, with its id and resolved commit, or the software itself |

```
entity:   { id, prov_type, extent?, attributes }
activity: { id, node, step?, started, ended, parameters }
agent:    { id, agent_type: "model" | "software", model_id?, commit_sha?, unresolved_reason?, version? }
```

### Relations, all PROV's own

| relation | replaces | meaning |
| --- | --- | --- |
| `wasGeneratedBy(entity, activity)` | the old `author` field | which run produced this |
| `used(activity, entity)` | nothing — it was implicit | what a node **read**. This is what makes ordering inspectable rather than inferred |
| `wasAssociatedWith(activity, agent)` | the old `model` field | which model ran, and at which commit |
| `wasAttributedTo(entity, agent)` | — | who is answerable for the entity |
| `wasDerivedFrom(entity, entity)` | the old `refine` verb | a narrower extent or a better value, with the coarse one retained |
| `wasInvalidatedBy(entity, activity)` | the old `withdraw` verb | this should no longer be read as what it was — and PROV keeps the entity |

**`label`, `confirm`, `contest` and `measure` remain**, as Entities of `prov_type` `assertion`, each
`wasGeneratedBy` the activity that made it and `wasDerivedFrom` the entity it is about. A `confirm` or
`contest` is additionally `wasDerivedFrom` the assertion it answers — which is why an assertion needs its
own id, and the PROV model makes that requirement structural rather than a thing to remember.

**A node's `verdict` entity is `wasGeneratedBy` the step that concluded — its last — not its first.**
AIRWAY attributed its verdict to `classify` and SPEECH to `transcript`, which said each conclusion
was reached before the steps that can change it had run: YAMNet's answers and the lexical read for
AIRWAY, diarization, PII and quality for SPEECH. TAXONOMY and REDACT already attributed theirs to
their last step. For AIRWAY the concluding step is `lexical` when any span carried a label and
`confirm` otherwise, because `lexical` only exists in the first case.

**`refine` and `withdraw` are gone as verbs**, because PROV already has both relations and its semantics
are the ones this design argued for independently: `wasDerivedFrom` keeps the source, and
`wasInvalidatedBy` marks an entity unusable without deleting it.

### What PROV buys beyond tidiness

**`used` closes a hole.** Nodes were told to "record what they read", with nothing to record it in. It is
now a relation, so the graph's real dependency order is queryable — and the finding that the branches are
not concurrent (below) is something the store can now *show* rather than something a reader has to
reconstruct.

**An unresolved commit is representable.** An Agent carries `commit_sha` **or** `unresolved_reason`, so a
Hub outage degrades to an agent whose commit is honestly unknown instead of blocking every write. A
provenance model that cannot say "I could not resolve this" forces a lie or a crash.

**It serialises to a standard.** The JSONL persistence below is PROV-JSON-shaped, so nothing has to be
re-modelled if the store is ever exported or joined with provenance from outside this project.

### Append-only still does the work

Every record is added, never modified. So merging two stores is a set union and is order-independent, and
the current view of an entity is a fold over the relations touching it — the fold being the reader's
choice. PROV changes the vocabulary, not that property.

## Ordering is declared by what a node reads

**Ordering is declared by reads, not by a runner's sequence.** A node that reads another node's
assertions **depends on that node**, and that dependency — the `used` relation — is the only thing
that orders the graph. A node whose reads are all satisfied may run; two nodes neither of which reads
the other may run concurrently.

```
ADMIT → PREPROCESS → TAXONOMY → routing ─┬─→ AIRWAY?             ─┐
                                         ├─→ SPEECH? → REDACT?   ─┼→ VERDICT → REPORT
                                         └─→ VOICE?              ─┘
```

`?` marks a **conditional** node. [`routing.md`](routing.md) writes one `branch_decision` element per
branch before any branch runs, and a branch runs only if its decision says so; [`REDACT`](redact.md)
is a step of SPEECH and runs only when SPEECH's PII scan found something.

**The branches are concurrent.** None reads another: SPEECH withdraws nothing for AIRWAY's labels
([`branch-speech.md`](branch-speech.md) step 4), and VOICE measures PREPROCESS's phonation spans
rather than a residual of the other branches' claims ([`branch-voice.md`](branch-voice.md)). Every
branch reads PREPROCESS and its own `branch_decision`, and nothing else the graph produced.

The real edges are therefore: everything ← PREPROCESS; TAXONOMY ← PREPROCESS; routing ← TAXONOMY;
each branch ← routing; REDACT ← SPEECH; VERDICT ← every verdict and every branch decision; REPORT ←
VERDICT and the whole store.

`PREPROCESS → TAXONOMY → routing` is one unit over one input stream, and every element it writes names
that stream, so a second pass over a suppressed-foreground stream is expressible. The current target
runs the unit once, on the original recording.

## The last fold, and the rendering after it

[`verdict.md`](verdict.md) folds every node's verdict into a file-level one. It is a reader of the store
like any other node, with two properties worth stating here: it reads *verdicts* and `branch_decision`
elements rather than raw measurements, and where two nodes contradict each other it records both and
flags rather than choosing.

[`report.md`](report.md) runs after it and reads the whole store. It **writes no elements**: a
rendering is not evidence, and nothing downstream reads it to learn a fact the store does not hold.

## What this replaces

The per-node "inputs" tables in the design files are now **what each node reads in practice**, not a
contract that constrains it. The admission rule for a PREPROCESS derivative changes accordingly: a
derivative is admitted when it is **written to the store with provenance**, not when a second consumer
appears for it. `spans` having one consumer is therefore no longer an anomaly to resolve.
