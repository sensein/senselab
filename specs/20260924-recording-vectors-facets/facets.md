# Faceting the corpus view

The parallel-coordinate page,
`src/senselab/audio/workflows/triage/viewer/recording_vectors_viewer.html`, could narrow its lines
only by brushing an axis. With ten slots and 195 columns that makes filtering compete with reading:
to keep only `story-recall` you first had to spend one of the ten slots on `task`, and the axis you
gave up was one you wanted to look at.

Facets separate the two. The panel on the left of the plot narrows by **what a recording is** — its
task, verdict, release, routing, conformance, the names of the gates it failed, the nodes that
flagged it — and leaves all ten axes for reading.

Parts: `facets.js` (the model, no DOM), the panel in `shell.html` / `app.js` / `styles.css`, and
`CorpusView.setFacetMask` in `corpus.js`. Held by `src/tests/.../viewer/facets.test.mjs` (17 node
tests), `facets.e2e.spec.mjs` (12 Playwright tests) and four drift guards in `viewer_test.py`.

---

## What a facet is, and what it is not

**A facet is a categorical column.** Every `categorical` and `set` column in the axis catalogue is
offered — 52 of them, including all 22 `gate_<name>_passed` outcomes — and nothing else is. A
number is brushed on an axis, where the reader can see its distribution before choosing a range; a
list of ticked numbers would be a worse instrument for the same job.

Two columns are refused, with the reason shown rather than the column silently missing:

| column | why |
| --- | --- |
| `participant` | 1,527 levels, one per person. A list of 1,527 checkboxes is a list, not a facet; `participant` stays a default axis, where a session reads as a bundle. |
| `session` | one level per session, so a facet over it selects one recording at a time. |

**Two `set` columns face by membership, not equality.** `gate_failed_names` and `flag_nodes` hold a
set per recording. Their own catalogue entries already said *"a set of names, not a value — put its
size on the axis, or filter by a term"*; this is the filter that entry pointed at. Choosing
`coverage_min` on `gate_failed_names` keeps every recording whose set contains it.

---

## Composition, stated once

- Within one facet, chosen values are a **union**. `task ∈ {story-recall, cough}`.
- Across facets, they are an **intersection**. `task ∈ {…} ∧ verdict ∈ {…}`.
- The whole facet mask **intersects the axis brushes**. Neither replaces the other, and clearing
  one leaves the other standing.

That last point is why `CorpusView` now keeps `brushMask` separate from `selected`. `brushMask` is
what the brushes alone admit; `selected` is `brushMask ∧ facetMask`. They have to be separate
because `brushMask` is the *denominator the facet panel narrows*, and a panel that counted against
the post-facet set would report the answer the reader already has.

`clearBrushes()` therefore leaves the facets in place, and `facet-clear` leaves the brushes in
place. Both are asserted.

---

## The two counts, and why there are two

Every value shows **`available / total`**.

- **`total`** is the count over the whole corpus. It never moves. This is the number the request
  asked for: *`story-recall` is 889 recordings, before selecting it*.
- **`available`** is the count over the set every **other** facet and the brushes admit — that
  facet's own constraint is left out.

The exclusion is the whole design. If `available` were computed against the fully filtered set,
then the moment you chose `verdict = pass` every sibling would read 0 and the facet would stop being
usable as a way to change your mind. With the facet excluded, `pass`, `flag` and `discard` keep
their real numbers under whatever else is selected, and only *other* facets narrow.

The panel header states the denominator on both sides — `21,304 → 889 drawn, of 62,548
recordings` — so a facet can never silently move it. With nothing chosen it says
`62,548 of 62,548 recordings, no facet applied`.

**Absence is a value.** Each facet carries an explicit `absent` bucket, last, with its own count,
styled like the axis rail's absent chip. This follows the page's existing rule that a null is never
silently kept and never silently dropped: the reader selects nulls if they want them. The bucket's
key is the literal `(absent)`, which a drift guard asserts is not a column name the producer writes.

`gate_failed_names` shows the distinction the page cares about: its `absent` bucket counts rows
whose set is **null**, not rows whose set is **empty**. An empty set is a recording that failed no
gate, and that is a measurement; a null is a recording nothing answered for.

---

## Ordering and the top-K cap

A facet with a declared vocabulary in `SchemaAxes.ORDERINGS` keeps it, so `verdict` reads
`pass < flag < discard` and every gate outcome reads `true < undetermined < false` — severity, not
frequency. An open vocabulary sorts by `available`, then `total`, then name.

At most 12 values are shown, with `show all N values` beneath; a **chosen** value is always hoisted
into the shown set, so a selection can never hide under the cap. The search box filters facet names,
and values within an open facet.

---

## What it costs, measured

Headless Chromium 1600×1000, a **62,540-row** synthetic corpus on the producer's own schema
(`scripts/triage_viewer_fixture.py --rows 62548`), four facet groups open. Medians and maxima over
five repetitions of a four-facet sequence, from `artifacts/measure_facets.mjs`.

| | median | max |
| --- | ---: | ---: |
| **one facet click, whole handler** | **7.2 ms** | 11.7 ms |
| of which: build the mask and re-derive the drawn set over all 62,540 rows | 4.1 ms | 7.3 ms |
| of which: recount and redraw the four open groups | 3.3 ms | 4.6 ms |
| click → the canvas has finished repainting | 210.4 ms | 226.7 ms |
| *the same, for an axis brush change (unchanged control)* | *214.8 ms* | *221.0 ms* |
| encode one column, cold (`task` / `verdict` / `gate_failed_names` / a gate outcome) | 8.6–18 ms | |
| encode **all 52** offered columns | 321 ms | |
| memory held by all 52 encodings | 12.8 MB | |

**The repaint is not the facet layer's cost.** A facet click reaches a painted frame in 210 ms and
an axis brush in 215 ms — the same number, because the paint loop walks every row in 6,000-row
chunks whatever is selected. Faceting added nothing to it. The number the facet layer owns is the
7.2 ms handler, which fits inside one frame.

Three decisions came out of these numbers.

**Encodings are lazy.** Encoding every offered column up front costs 321 ms — a visible stall on a
panel that opens with four groups. A column is encoded the first time its group is opened, so the
first paint pays for four columns (~45 ms) and never for the 48 nobody looked at.

**Only open groups are counted.** `values()` walks the corpus once per column. Counting all 52 on
every click would be 52 passes for values nobody is looking at; counting the open ones is 3.3 ms.

**The mask is typed arrays, not string compares.** A scalar column becomes an `Int32Array` of term
codes (−1 for absent) and the filter is an array read plus a byte lookup, never a property read and
a string comparison. This is the same move the hover hit-test made when it went from 59 ms to
1.6 ms with a `Float64Array`, for the same reason.

A `set` column instead keeps one posting list of row indices per term, because the membership form
would need `terms × rows` bytes — 50 MB for `task`'s 796 levels — and the posting form needs one
entry per membership, which over this corpus is far less than one per row. Choosing a rare term then
costs that term's size rather than the corpus's.

---

## What is not measured

- **The panel against the real 09-23 corpus.** Every number above is over a synthetic corpus of the
  same size on the same schema. The value *distributions* differ — in particular `task`'s real
  vocabulary is 796 levels against the fixture's 10, which makes the real `task` encoding and its
  `values()` sort more expensive than measured here, though both are O(rows) with a small constant.
  Rerun `node artifacts/measure_facets.mjs <parquet>` against the real file.
- **Legibility.** The e2e suite asserts the panel sits beside the plot and that its numbers are
  right; it does not assert the panel is readable, and no screenshot is diffed.
