# The element store

Nodes do not pass products to each other. They **write elements to a store**, and later nodes refine
those elements by asserting over them. Any node may read anything in the store.

## Why not dataflow

A strict `A → B` graph forces two things this design cannot afford. It makes every shared product an
input to declare in advance, so a derivative with one consumer looks like a mistake even when it is
shared machinery. And it gives the last writer the final word, so a node that contests an earlier claim
either overwrites it — destroying the evidence — or has nowhere to put its disagreement.

The store fixes both by being **append-only**.

## Elements and assertions

An **element** is a thing the graph believes might exist:

```
element:  { id, kind, extent?, author, evidence }
```

| field | meaning |
| --- | --- |
| `id` | stable, assigned once by the node that first proposed the element |
| `kind` | `span`, `word`, `speaker`, `interval`, `measurement` |
| `extent` | `(start, end)` where the element has one. Absent for file-level elements |
| `author` | the node that proposed it, with model and revision where a model was involved |
| `evidence` | whatever the author measured, verbatim |

An **assertion** is a later node's claim *about* an element:

```
assertion: { element_id, verb, value?, author, evidence }
```

| verb | meaning |
| --- | --- |
| `label` | this element is of this kind or class |
| `confirm` | an independent instrument agrees with a named prior assertion |
| `contest` | an independent instrument disagrees. **Both survive** |
| `refine` | a narrower extent or a better value, with the prior extent retained |
| `withdraw` | this element should not be read as what it was proposed as, and why |
| `measure` | attach a measurement without claiming identity |

**Nothing is deleted and nothing is overwritten.** The current view of an element is a fold over its
assertions, and the fold is a reader's choice — a consumer that trusts only confirmed labels and one
that wants every claim both read the same store.

## Consequences

**`contest` does not resolve.** Two instruments disagreeing is a majority for neither, so the store
holds both and the outcome is a `flag`. A node must not invent a tie-break it cannot measure.

**`withdraw` is not deletion.** A pyannote segment withdrawn as an airway event stays in the store with
its reason, because a reader comparing a speaker count against speaker spans needs to see why they
differ.

**`refine` keeps the coarse extent.** A span from the envelope refined by word timings retains both, so
a later reader can tell a locator from an edge.

**Provenance is not optional.** An element or assertion authored by a model carries the model id and
resolved revision. An embedding comparison additionally carries the model that produced the target,
because embeddings from different models are not comparable.

**Anything in the store may be used.** A node reads what it finds useful and is not restricted to a
declared input list. What it *must* do is record what it read, so a claim can be traced to the elements
behind it.

## What this replaces

The per-node "inputs" tables in the design files are now **what each node reads in practice**, not a
contract that constrains it. The admission rule for a PREPROCESS derivative changes accordingly: a
derivative is admitted when it is **written to the store with provenance**, not when a second consumer
appears for it. `spans` having one consumer is therefore no longer an anomaly to resolve.
