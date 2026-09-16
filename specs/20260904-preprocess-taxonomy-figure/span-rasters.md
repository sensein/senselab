# The per-span rasters, and why no threshold takes part in them

## What was wrong

Both per-span classifier panels printed `span_yamnet is absent from the store` on every page of
every run under the packaged config. The cause was a contradiction inside PREPROCESS rather than
anything in the figure.

`_span_yamnet` and `_span_hear` are both documented **"raw scores only — no labelling decision"**,
and both compute two separate things for each window: `raw_scores`, the model's complete output, and
`members`, the subset clearing a threshold. But both opened with

```python
default_threshold = float(config.require("windows.<classifier>.default_threshold"))
```

and that key ships null. `require` raised, PREPROCESS caught it as a cascading absence, recorded the
block in `absent` and **still returned PASS** — so the expensive model output was destroyed for want
of a labelling threshold the block's own docstring says it does not make a decision with.

The same principle was already established one level up and pinned by a test whose docstring reads
*"A null threshold must not cost the expensive model output"*: the whole-file `<classifier>_scores`
step persists raw scores with no threshold, and the separate `<classifier>_windows` fold applies one.
The per-span passes simply had not been brought into line.

## What changed

`_span_window_attributes` builds both entities' attributes in one place. `raw_scores` is written
whatever the configuration says, because the model ran and its output is a measurement. `labels` and
`scores` are a decision taken *over* that measurement, so they appear only when a threshold was
configured.

Two states had to stay distinguishable, and an empty `labels` list cannot carry both:

| store reads | means |
| --- | --- |
| `labelled: false`, no `labels` key | no threshold was set; nothing was decided |
| `labelled: true`, `labels: []` | a threshold was set and nothing cleared it — a real finding |

`default_threshold` is recorded on the entity too, so a reader can see which bar was applied without
consulting the config that was in force.

### The trap this opened, and the guard for it

TAXONOMY's two airway lines count spans carrying a family label, and `_span_label_evidence` decides
`available` from whether the `span_<classifier>` **activity** ran, not from measurement count. Before
this change the block raised, no activity existed, and the line read `unavailable`. After it, the
activity exists — so with a floor configured but a threshold still null, the line would have counted
zero labelled spans and reported **`absent`**: a false negative, and the leniency rule is explicit
that screening must not delete evidence no downstream branch can recover.

`_span_label_evidence` therefore also returns `available: False` when every window it finds says
`labelled: false`. A window is taken as labelled unless it says otherwise, so stores written before
`labelled` existed keep their meaning. Today both floors are null as well, so the state is
`unavailable` either way; the guard is what keeps it correct once floors are set and thresholds are
not.

## What the rasters draw

No threshold takes part in these panels at all.

- **Rows** are the **union of each span's top-`k` labels**, `k = FigureStyle.top_labels = 4`. Each
  span contributes its own four highest-scoring labels, and the union of those sets is the row set.
- **The union is taken per file**, `FigureStyle.raster_rows_scope = "file"`, computed once from every
  span in the recording and drawn identically on every page. A row holds the same vertical position
  on page 1 and page 4, so one label can be scanned down across pages. A page whose spans carry none
  of a row's label shows that row **empty**, which is information: the label is present elsewhere in
  the file and absent here. `_raster_rows` raises on any other scope rather than silently supporting
  a per-page union, which would make a row's position depend on paging.
- **Row order** is the file-wide peak score, descending — deterministic, and matching how the
  whole-file summary ranks.
- **Cells** are that span's own score for the row's label. A label absent from a span's scores draws
  no marker; empty reads as "not in this span's scores", never as zero.
- A label that never scored above zero anywhere earns no row.

`k = 4` is a visualisation parameter and is not the config's retrieval size. `yamnet.top_k` and
`windows.ast.top_k` are the full label space and HeAR passes `top_k=None` to keep all eight — those
govern how much PREPROCESS retrieves and persists, and are untouched. The 4 is how many labels the
figure shows per span.

### Rejected

**Ranking rows by how often a label fired** — what the scratch tool did with
`hear_counts.most_common(4)`. Raw model output carries every label on every window, so a frequency
count ranks them all equally and the rows collapse to whichever labels sort first alphabetically.
That is the same defect that had the whole-file summary printing `A capella peak 0.000`.

**Setting a threshold so the panels would draw.** Forbidden outright: a pipeline parameter must never
be set for a visualisation's benefit, and `default_threshold` feeds the fold that writes label
entities into the store, so changing it changes the store, what TAXONOMY reads, and the config hash.
The point of this change is that the raw scores never needed one.

## The whole-file summary's layout

The readout was one monospaced block drawn from `x=0.0`, and its longest lines ran past the axis —
`Female speech, woman speaking peak 0.510  median 0.257  in 8 windows` being the worst.

The three classifier blocks are now laid out **side by side in columns**, with the kind-states block
below them at full width. The columns are padded **in the text itself** to `_SUMMARY_COLUMN_WIDTH`
(52 characters) with the label field capped at `_SUMMARY_LABEL_WIDTH` (20), so the longest possible
line is a known character count and cannot overrun by construction rather than by luck. Font size is
unchanged; compacting the line is what bought the width, not shrinking the text.

`taxonomy_summary_lines` still returns the vertical form, which is what the `taxonomy_summary.json`
sidecar records; `summary_panel_lines` is the column form the panel prints. Both are built from one
`_summary_sections` so the two cannot drift.

Two tests bound it: one against the declared character width, and one that draws the panel and
measures the rendered artist's extent against the axis extent. The second is the one that matters —
a string test is what let the clipping ship in the first place.
