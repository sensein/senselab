# TAXONOMY

Consolidates what the classifiers said about the recording. It decides nothing:
[`routing.md`](routing.md) evaluates the family taxonomy ruleset and selects the branches that run.

This node used to classify three kinds — `speech`, `airway`, `voice` — into `present`, `absent` or
`uncertain`, and that classification used to decide execution. It was deleted on 2026-09-13 because
it emitted a constant: every `taxonomy.presence_floor.*` was null, `_line_state` returned
`unavailable` whenever the floor was null, that folded to `uncertain`, and voice was a hardcoded
retired stub, so all three kinds read `uncertain` on every recording and every branch ran. The
deletion and its consequences are in
[`../20260912-ruleset-in-pipeline/design.md`](../20260912-ruleset-in-pipeline/design.md), "Stage 2".

## Signature

```
taxonomy(store, source, config, hint=None, *, run_dir)
    -> flag(nothing to consolidate) | pass(classifiers, n_labels)
```

Reads and writes the [element store](store.md). Writes the two file-scoped measurements under
"Product" and nothing else. `hint` is accepted for the shared node shape and is not read.

**TAXONOMY runs no models.** Every score it reads was written by [`PREPROCESS`](preprocess.md). It is
a consolidation over stored scores, not a detector committee.

**It measures; it does not decide.** Nothing it writes selects a branch, and it writes no
`branch_decision` and no `ruleset_routing`. That boundary is deliberate: a node that both measures
content and decides what runs on it cannot be checked against itself.

**It localises nothing.** Which cough, which vocal task, and where, belong to the branches.

## What it reads

| element | author | used for |
| --- | --- | --- |
| `<classifier>_scores` | PREPROCESS | the whole-file label summaries, off the verbatim sidecar, so no threshold takes part |
| `span_hear` | PREPROCESS | **per-span** HeAR raw scores, consolidated into the file-level taxonomy |
| `span_yamnet` | PREPROCESS | **per-span** AudioSet raw scores, the same |

**Hints are not an input.** A hint may force a branch to run ([`routing.md`](routing.md)) and may be
compared against the branches' conclusions ([`verdict.md`](verdict.md)), but it never enters a
measurement, because a measurement that reads the declaration cannot disagree with it.

## Product

```
outcome:  flag(nothing to consolidate) | pass(classifiers, n_labels)
verdict:  { classifiers: [...], n_labels: int }
view:     each classifier's whole-file label summary, then the consensus taxonomy,
          then this node's verdict
```

Two file-scoped measurements, both with `extent=None`:

| measurement | what it is |
| --- | --- |
| `<classifier>_label_summary` | per label, peak, median and window count over the whole file, off the verbatim scores sidecar, so it needs no threshold. A classifier that never ran gets no summary — a missing summary and an all-zero one stay distinguishable |
| `consensus_taxonomy` | the per-span labels of `span_yamnet` and `span_hear` consolidated into one file-level taxonomy, **one row per AudioSet ontology node** rather than per label string, each classifier's own spellings kept in `labels_by_classifier`. Disagreement is recorded, not resolved: a label only one vocabulary contains is not a vote against it |

`voice.glide` and `voice.chant` — two of the ruleset's eleven gates — read `plain|yamnet`, which the
feature reader populates from `yamnet_label_summary`. **That is why TAXONOMY runs before ROUTING and
not beside it**: evaluated earlier, both gates would read `unavailable` and VOICE would never route.
`GRAPH_ORDER` enforces the order and a test pins it.

**The two measurements do not have the readership this file used to claim for them.** FIGURE prints
the label summaries on its cover (`figure.py:614`) and reads no `consensus_taxonomy` at all. REPORT
reads neither: its classifier panels come from PREPROCESS's `<classifier>_windows`
(`report.py:425`, `:443`), written at `preprocess.py:1662`, not from anything TAXONOMY writes.
`consensus_taxonomy` has **one** production reader, `routing_analysis/features.py:755` — ROUTING's
own feature reduction — plus the `rewrite_consensus_taxonomy` extend driver that recomputes it
(`extend.py:330`). The voice rework that would consume it is unbuilt (`voice.py:240`).
`taxonomy.py:6`'s module docstring still says "ROUTING, FIGURE and REPORT read them". That is code,
and correcting it is a code change, not a change to this file.

That readership is the argument
[`../20260913-branch-contract-and-hints/design.md`](../20260913-branch-contract-and-hints/design.md)
uses to merge this node into SCREEN. Nothing of that design is implemented.

## Its activities

| step | writes |
| --- | --- |
| `<classifier>_label_summary`, one per classifier that produced scores | that classifier's summary |
| `consensus_taxonomy` | the consolidation, when at least one per-span classifier produced scores |
| `conclude` | the node verdict, `used`-edged to everything it wrote |

## Outcome

| outcome | when |
| --- | --- |
| `pass` | at least one per-span classifier produced scores; the reason names them and the label count |
| `flag` | none did — there was nothing to consolidate |

The `flag` is not a claim about the recording. It says every downstream label gate will read
`unavailable`, which is a fact about the run worth surfacing before ROUTING declines everything for
want of evidence. **No decision was invented for this node to make**: it consolidates, and its
verdict says whether the consolidation had anything to work from.

A classifier that ran and matched nothing is a `pass` with a low `n_labels`, not a `flag`: "ran and
found nothing" and "never ran" are different facts and stay apart.

## Every threshold is configurable

| key | governs |
| --- | --- |
| `taxonomy.consolidation_floor` | the score a label needs to enter the consensus taxonomy, every classifier |
| `taxonomy.classifier_ontology_profile` | the one profile label identity, the airway closure and AIRWAY's corroboration all resolve through |
| `taxonomy.airway_ontology_roots` | the AudioSet ontology roots whose closure is the airway vocabulary. Read by AIRWAY, not here |
| `taxonomy.speech_labels` | the speech label family. Read by SPEECH, not here. **Null**, and read `config.get(...) or []`, so it empties that branch's acoustic family without a word anywhere |
| ~~`taxonomy.presence_floor.*`~~, ~~`taxonomy.voice_min_duration_s`~~, ~~`taxonomy.voice_uncertain_duration_s`~~ | **deleted 2026-09-13** with the fold they served |

Each key's derivation is in [`config-derivations.md`](config-derivations.md), keyed by config
section; `data/config/default.yaml` carries a one-line `#` description beside each. A key read with
`config.require` and left null raises; one read with `config.get` returns `None` and its caller
carries on silently, which is the more dangerous of the two.

## Out of scope

Running any model, localising anything, naming which airway event or which vocal task, reading
hints, evaluating the ruleset, and deciding which branches run. The last two are ROUTING's, and the
split is the point.

Derivations live in [`benchmarks/taxonomy.md`](benchmarks/taxonomy.md).

## Open derivations

| key | what is owed |
| --- | --- |
| voice's evidence source | which `consensus_taxonomy` rows express phonation, and what a VOICE branch should conclude from them. The detector that used to propose `phonation_spans` was removed on 2026-09-04 — its criterion parameters had never been measured, so the pass raised on every run, and its glide criterion was separately recorded as unfixable by fitting. The ruleset routes VOICE on its own gates in the meantime, so this is no longer blocking execution; it blocks the branch having a subject |
| `taxonomy.consolidation_floor` | owner-directed at 0.2 and not fitted. What it costs is a row in the consensus taxonomy, not a routing decision, so the exposure is bounded |
