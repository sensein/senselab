# Batched per-span classification, and the file-level consensus taxonomy

Four owner-directed changes, 2026-09-04. Written here rather than in the code, per CLAUDE.md.

## Why the per-span passes were batched

`_span_yamnet` and `_span_hear` classified **one span per call**. Each call crosses a subprocess-venv
boundary, so a recording with 91 spans spent 182 venv spawns on two classifiers.

Measured on `Story-recall-(v2)`, 79.51 s, 91 spans:

| | PREPROCESS wall time |
| --- | --- |
| before the per-span passes ran at all | 169.3 s |
| per-span, one call per span | 949.7 s |
| batched, one call per classifier | **176.4 s** |

The 5.6× was not the classification, it was the spawning. Across the 112-recording b2ai collection
the per-span form would have turned a 1h41m run into roughly nine hours.

Both APIs already took a batch — `classify_audios(audios: List[Audio], ...)` and
`detect_health_acoustic_events(audios: List[Audio], ...)`, each returning one result per input — so
the change is to build every span's input first, make one call, and map results back by index.

### What the batch must not cost

The per-span form gave every span its own outcome: a span the model refused was marked
`_mark_unmeasured(..., type(err).__name__)`, and a span that produced no window was marked
`"no_native_window"`. A naive batch loses that — one exception would blame all 91 spans for a
failure that belonged to one.

`_classify_spans_in_batch` keeps it, in three parts:

1. **Input construction stays per span.** HeAR's 2 s buffering (`span_hear_input`) can refuse an
   individual span, and that refusal is recorded against that span before any classification runs.
   The batch is built only from spans whose input was constructed.
2. **The batch is tried once.** On success, and only when it returns exactly one result per input,
   those results are used directly.
3. **A batch that fails falls back to one call per span.** Then each span's own exception is its own
   fact again, recorded as `"<SpanError> (after batch failed: <BatchError>)"` — which distinguishes
   "the batch call failed" from "this span produced nothing" while naming both. The fallback is
   slow, but it runs only when the fast path has already failed, and a wrong attribution is worse
   than a slow one.

A batch returning the wrong number of results is treated as a batch failure rather than aligned by
position. Aligning a short result by index would silently attribute one span's scores to another,
which is the one outcome worse than failing.

Batching may change how many calls are made and nothing else:
`span_batching_test.py::test_the_batch_result_matches_classifying_one_span_at_a_time` asserts the
batched result equals the looped result over the same inputs.

## The consensus taxonomy

The owner's architecture ruling: **PREPROCESS provides labels; TAXONOMY consolidates them into a
consensus taxonomy for the file, and that is what downstream reads.**

Before this, a downstream node wanting to know what the recording contains had to re-derive it from
per-span windows or from the raw score sidecars. TAXONOMY wrote per-kind states and a
`<classifier>_label_summary`, but nothing that answered "what is in this file" in one place.

`consensus_taxonomy` is a file-scoped measurement (`extent=None`) carrying one row per label:

```
{"label": "Speech",
 "peak": 0.98,                                   # highest over every span and classifier
 "peak_by_classifier": {"yamnet": 0.98, "hear": 0.85},
 "classifiers": ["hear", "yamnet"],              # which reported it
 "n_classifiers": 2}
```

with each classifier's own `{peak, median, n_spans}` retained underneath. Rows are ranked by peak, so
the first row is the file's strongest claim about itself.

### What it consolidates over

Per-span labels, which today means YAMNet and HeAR — the two classifiers PREPROCESS runs per span.
AST is a whole-file classifier at a 10.24 s window with no per-span pass, so it contributes to
`ast_label_summary` and not to this consensus. If a per-span AST pass is ever added, it joins by
appearing in `PER_SPAN_CLASSIFIERS` and needs no other change.

### How disagreement is reconciled: it is recorded, not resolved

Two classifiers reporting the same label produce one row naming both and keeping both peaks.
YAMNet and HeAR share label names (`Cough`, `Speech`) while their vocabularies differ in size and
provenance — 521 AudioSet labels against 8 health-event labels — so a label only one of them can
emit is not evidence of disagreement, and treating a missing label as a vote against would be wrong.
`n_classifiers` therefore counts corroboration and never contradiction.

**This is a measurement, not a decision.** It consolidates evidence and stops. Nothing here turns a
peak into `present` or `absent`; the floors that do that stay under `taxonomy.presence_floor`, where
they already live and where they are still null. A reader of `consensus_taxonomy` gets what was
measured, not what it means.

## `taxonomy.yamnet_consolidation_floor`

Owner-directed: **YAMNet consolidation drops a label scoring under 0.1.**

The failure it fixes was visible on the page. The rasters' rows are the union of each span's top-4
over the file, and with 521 labels a span where nothing fires still contributes its four highest —
which are all near zero. The union came to 17 rows of which 15 read `0.00` in every span, so the
panel was the tallest thing on the page and carried two rows of information.

It is a real threshold and an owner's call, so it is a config key with a description saying so, not
a literal. It applies to YAMNet's consolidation and to the raster union that selects rows.

**HeAR is deliberately exempt.** Its vocabulary is 8 labels, all health events the graph cares
about, and every one of them is shown whatever it scored. A near-zero `Cough` over a span is a
measurement worth seeing; a near-zero `Tick` among 521 AudioSet labels is an artefact of ranking.
The asymmetry is in the vocabularies, not in the classifiers.

## The rasters read higher as lighter

Owner-directed. `BuGn` → `BuGn_r`, `OrRd` → `OrRd_r`; the position mapping is unchanged, so a high
score now lands at the light end of a reversed map.

Reversing breaks a fixed text colour: black reads on a near-white high-score cell and disappears on
a dark low-score one. `_readable_on` therefore picks black or white per cell from the cell's own
Rec. 601 luminance, so the choice follows perceived brightness rather than the colormap's position —
which is what a reversed map needs, since its two ends swap.

## Measured

Re-run on `Story-recall-(v2)`, 79.51 s, 91 spans, after all four changes.

**PREPROCESS 949.7 s -> 176.4 s**, a 5.4x saving. The baseline before the per-span passes ran at all
was 169.3 s, so per-span YAMNet and HeAR over 91 spans now cost about 7 s together rather than 780 s.
The work was never the classification.

**The floor, on this recording:**

| classifier | labels consolidated | raster rows |
| --- | --- | --- |
| YAMNet | 521 -> **2** | 17 -> **2** (`Silence`, `Speech`) |
| HeAR | 8 -> 8 | 8 -> 8 |

Fifteen rows that read `0.00` in every span are gone, and what remains is what the recording holds.

**The consensus, on this recording** — 9 labels over both classifiers:

```
Silence       peak 1.000  n_classifiers=1  {yamnet: 1.000}
Speech        peak 1.000  n_classifiers=2  {yamnet: 0.99994, hear: 0.781}
Laugh         peak 0.717  n_classifiers=1  {hear: 0.717}
Snore         peak 0.677  n_classifiers=1  {hear: 0.677}
Baby Cough    peak 0.414  n_classifiers=1  {hear: 0.414}
```

`Speech` is the one label both classifiers reached independently, which is what `n_classifiers`
exists to record. Every HeAR label survives because HeAR is exempt from the floor.
