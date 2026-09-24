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

Headless Chromium 1600x1000, against the **real 62,548-row** `recording_vectors.parquet`
(schema 3, 195 columns, 104.6 MB) built 2026-09-24 from `triage_rerun_20260923`. Four facet groups
open. Medians and maxima over five repetitions of a four-facet sequence, from
`scripts/triage_viewer_measure.mjs`.

| | median | max |
| --- | ---: | ---: |
| **one facet click, whole handler** | **5.6 ms** | 10.8 ms |
| of which: build the mask and re-derive the drawn set over all 62,548 rows | 2.7 ms | 6.2 ms |
| of which: recount and redraw the four open groups | 2.9 ms | 4.3 ms |
| click -> the canvas has finished repainting | 196.3 ms | 201.4 ms |
| *the same, for an axis brush change (unchanged control)* | *206.5 ms* | *209.9 ms* |
| encode one column, cold (`task` 796 levels / `verdict` / `gate_failed_names` / a gate outcome) | 3.3-8.7 ms | |
| encode **all 52** offered columns | 245.7 ms | |
| memory held by all 52 encodings | 12.5 MB | |
| first paint: parquet chosen -> the panel is on screen | 3.9 s | |

**The repaint is not the facet layer's cost.** A facet click reaches a painted frame in 196 ms and
an axis brush in 207 ms -- the same number within noise, because the paint loop walks every row in
6,000-row chunks whatever is selected. Faceting added nothing to it. The number the facet layer owns
is the 5.6 ms handler, which fits inside one 16.7 ms frame with room to spare.

Three decisions came out of these numbers.

**Encodings are lazy.** Encoding every offered column up front costs 246 ms -- a visible stall on a
panel that opens with four groups. A column is encoded the first time its group is opened, so the
first paint pays for four columns (~28 ms) and never for the 48 nobody looked at.

**Only open groups are counted.** `values()` walks the corpus once per column. Counting all 52 on
every click would be 52 passes for values nobody is looking at; counting the open ones is 2.9 ms.

**The mask is typed arrays, not string compares.** A scalar column becomes an `Int32Array` of term
codes (-1 for absent) and the filter is an array read plus a byte lookup, never a property read and
a string comparison. This is the same move the hover hit-test made when it went from 59 ms to
1.6 ms with a `Float64Array`, for the same reason.

A `set` column instead keeps one posting list of row indices per term, because the membership form
would need `terms x rows` bytes -- 50 MB for `task`'s 796 levels -- and the posting form needs one
entry per membership, which over this corpus is far less than one per row. Choosing a rare term then
costs that term's size rather than the corpus's. It is also why `gate_failed_names` encodes in
3.3 ms against `verdict`'s 7.1: almost every recording failed no gate, so there is almost nothing to
post.

The same harness over a 62,540-row *synthetic* corpus of the same schema gives 7.2 ms / 4.1 ms /
3.3 ms and a 210 ms repaint -- slightly worse than the real file on every line, so the fixture is a
conservative stand-in for CI.

---

## What the real corpus showed that a fixture would not

**Genuine zeros are wildly uneven across gates, and the rarest is a single row.** Over the 62,548
recordings the gate readings carry **20,831** genuine `0.0` values, but they are concentrated:

| gate | genuine `0.0` | null |
| --- | ---: | ---: |
| `gate_omissions_max` | 18,422 | 41,689 |
| `gate_expected_tokens_matched_min` | 1,122 | 41,689 |
| `gate_events_min` | 920 | 52,168 |
| `gate_response_min_s` | 221 | 52,401 |
| `gate_repetitions_min` | 69 | 54,564 |
| `gate_coverage_min` | 58 | 61,000 |
| `gate_monotone_tolerance_semitones` | 18 | 59,404 |
| `gate_f0_spread_max_semitones` | **1** | 58,058 |

The zero-versus-absent browser test first picked *the first* gate carrying both a zero and a null,
which is `gate_f0_spread_max_semitones` -- one recording in 62,548, and that one carries
`verdict = flag`. Composing it with `verdict = pass` left no zero drawn at all, and the test failed
on a corpus where the page was behaving correctly. It now picks the gate with the **most** genuine
zeros, so what it measures is the drawing rule and not whether one row survived an unrelated facet.

A synthetic fixture would never have produced that shape: its zeros are a uniform third of each
applied gate.

---

## What is not measured

- **The panel against an r3 corpus.** Every number above is over the `triage_rerun_20260923` tree.
  A third replay (`triage_r3_20260923`, job 23604010) carrying PII near-match, `expected_names` and
  the widened LLM reviewer reach was still running when this was built, so a rebuild against it is
  owed. The facet layer reads only categorical columns, so its shape should not change, but every
  count in the tables above will move.
- **The `verdict` column against the owner's own reading.** The facet counts are what the parquet
  carries: pass 52,255 / flag 10,264 / discard 29 over 62,548. Nothing here checks that those are
  the right verdicts, only that the panel reports them faithfully.
- **Legibility.** The e2e suite asserts the panel sits beside the plot and that its numbers are
  right; it does not assert the panel is readable, and no screenshot is diffed.
