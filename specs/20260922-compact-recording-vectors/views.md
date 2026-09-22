# The two views, and what they had to get right

The page is `src/senselab/audio/workflows/triage/viewer/recording_vectors_viewer.html`, assembled
from the parts beside it by `build.py` and rebuilt by
`uv run python scripts/triage_vectors_viewer.py`. It is **one self-contained file**: 161 KB, no
network reference of any kind, a vendored `hyparquet` 1.31.1 + `fzstd` 0.1.1 bundle (61 KB, esbuild
IIFE) and about 100 KB of our own JavaScript and CSS.

It reads `recording_vectors.parquet` as a `File` the reader hands it, and **inherits that file's
handling**: mode 600, transcripts and PII marks rendered in place, opened from `file://` by the
owner and never hosted. Nothing derived from the parquet is in this repository.

---

## Why a `File`, not a fetch

A `file://` page cannot `fetch()` a sibling file — Chrome gives the page an opaque origin and
blocks it, with no error the page can show. A `File` from a picker or a drop has none of that
problem, and it is also the honest interface: the reader chooses the artefact each session, and the
page has no path to anything it was not handed.

It also happens to be the fast interface. `File.slice(start, end).arrayBuffer()` is a local disk
read, so `hyparquet` can read **only the column chunks a view needs**. The first paint reads 85
columns and **4.06 MB of the 91.3 MB file**; the nine binary blocks are not touched until a line is
selected.

---

## What it costs, measured

Measured on this machine against the real 62,488-row file. The data path is measured in node
against the shipped bundle; the DOM is measured by driving the shipped page in jsdom. **Canvas
paint time is not measured** — no Chromium is installed on this host, and a jsdom canvas is a stub.
What is measured is everything that decides *what* is drawn.

| | |
| --- | ---: |
| parquet footer | 4 ms, 0.52 MB |
| corpus read, 85 columns | **498 ms, 4.06 MB** |
| summarising the ten default axes | 68 ms |
| summarising all 84 assignable columns | 519 ms |
| corpus rows retained (post-GC) | 229 MB |
| first selection, blocks read directly | **1.0–1.1 s** |
| first selection, after the warm | **43 ms** |
| warming every block column | ~1.5 s, ~250 MB |
| decoding every block of all 62,488 rows | 1.4 s, 0 failures |

**A narrow read costs the same as a full one**, and that is the whole reason the page caches. The
file is a single row group of 62,488 rows with **no page index** (`parquet-cpp-arrow` 23.0.1 did
not write one), so `hyparquet` must read and decompress a whole column chunk to reach one row:
reading `wave_minmax` for row 40,000 costs the same 22.4 MB and ~60 ms as reading it for all rows.
One recording therefore costs 1.0 s, and all 62,488 cost 1.5 s. The page pays the 1.5 s once, in
the background, after the first selection — and the warm is a checkbox, for a reader who would
rather spend 1 s a click than 250 MB.

If the producer ever writes the parquet with `write_page_index=True`, this inverts and the cache
should become opt-in rather than opt-out.

---

## The axis defaults, and the numbers behind them

Owner-directed, the first three are `participant`, `task`, `verdict`. The remaining seven were
chosen against the distributions over this corpus, not from the candidate list alone.

| axis | non-null | what it separates |
| --- | ---: | --- |
| `participant` | 100% | 1,527 levels; the only axis that shows a participant's whole session as a bundle |
| `task` | 100% | 796 levels; the dominant covariate of everything else |
| `verdict` | 100% | pass 51,698 / flag 10,761 / discard 29 |
| `duration_s` | 100% | p5 2.7 s, p50 7.3 s, p95 57.1 s, 16,140 distinct — the highest-resolution numeric in the file |
| `conformance_airway` | 38.6% | true 9,942 / undetermined 11,252 / false 2,907 |
| `conformance_speech` | 69.5% | true 36,683 / false 4,014 / undetermined 2,717 |
| `conformance_voice` | 36.2% | true 4,615 / undetermined 17,995 |
| `flags_n` | 100% | 0–4; 10,761 recordings carry at least one |
| `pii_findings_n` | 69.5% | 89 distinct; **and 30.5% null, which is the axis the null rule exists for** |
| `release` | 100% | not_assessed 44,526 / releasable 13,555 / withheld 4,407 |

Rejected for a default slot, with reasons:

- **`conformance_quality`** — 99.9% non-null but only two values, 84.5% `undetermined`. It
  separates almost nothing. `release` takes the slot, and it is also what `pii_findings_n` explains.
- **Every `m_<name>`** — the best-covered numeric measurement,
  `expected_sequence_repeat_fraction`, is present on 29.1%; the median one is present on **7.5%**,
  and `sweep_extent` on 5 recordings out of 62,488. A measurement default would be an axis that is
  absent for nine lines in ten. They are one click away in the picker, which is where a
  measurement belongs when it is a per-family reading in a corpus of 48 families.
- **`route_airway` / `route_speech` / `route_voice`** — 99.9% non-null, and they are exactly the
  *explanation* of the conformance nulls (`declined` ⟺ no branch report). Genuinely useful, but
  paired with the conformance axes they would spend six of ten slots on three facts.

**Under these defaults 59,649 of 62,488 lines break at least once and only 2,839 are complete.**
That is not a flaw in the view; it is the shape of this corpus — AIRWAY declined on 61.4% of
recordings and VOICE on 63.8% — and a view that hid it by choosing denser axes would be hiding the
single largest structure in the data.

---

## Null is absent, and the page never says otherwise

The rule is one function, `CorpusView.segmentsFor`, so it can be held by a test rather than
scattered through a paint loop:

> A segment is drawn only between two consecutive axes that **both** carry a value.

So:

- **The line breaks.** An absent value contributes no vertex on the value band. `vertices()`
  returns `y: null` for it, and no code path substitutes a rail position into that field — the
  rail `y` is computed separately, in `paintOne`, only for a single highlighted line.
- **Absence is visible as mass, not only as a number.** Each axis carries an **absent rail** below
  a drawn axis break (a double-slash), with a bar whose width is the share of the drawn set with no
  value there and the count in words. Dashed stubs run from the neighbouring axes to the rail, so
  the eye sees *where the absent lines come from* without any of them acquiring a position.
- **A single line stays traceable.** Hovering or selecting draws that one recording whole, with the
  absent segments dashed at half alpha and the absent vertex as a **hollow square** on the rail
  rather than a filled dot on the band.
- **Absence is a first-class filter.** Every axis has an `absent N` chip that cycles
  exclude → include → only. The default is exclude, so a numeric brush never silently keeps or
  silently drops the nulls: it always drops them, and says how many.

This matters concretely rather than theoretically. Over this corpus the 17 numeric measurements
carry **14,807 genuine `0.0` readings** — 10,835 of them on
`expected_sequence_repeat_fraction`, against 44,329 nulls on the same column. A view that placed
null at the bottom would put 44,329 lines where 10,835 measured zeros belong.

The invariant the schema declares, `m_<name> IS NULL ⟺ m_<name>_n == 0`, was checked against all
17 × 62,488 cells of the real file: **0 breaks**. The per-axis absent counts the page computes were
checked against `recording_vectors.summary.json` for six columns and match exactly.

---

## Categoricals are categories, and not everything is an axis

**Ten of the 29 measurements are strings, one is a vector and one is a matrix.** The axis picker
lists all 101 columns so that none is silently missing, and refuses the ones that are not a number,
with the reason on the disabled option:

| kind | in the picker | what is assignable instead |
| --- | --- | --- |
| `string` scalar (verdict, conformance, …) | ordered categories | itself |
| `list<string>` (`m_carrier_rejected`, `flag_nodes`) | refused: *a set, not a value* | `· distinct terms` (its size) and `_n` |
| `list<double>` vector (`m_ddk_position_realised_mass`) | refused: *one value per syllable position, each null where the position was not realised* | `_n` |
| `list<double>` matrix (`m_ddk_cv_instrument_reading`) | refused: *repetitions by syllable position, flattened row-major* | `_n`, `_width` |
| `binary` | never offered | — |

A categorical axis draws **ordered discrete bands**, not a number line. Closed vocabularies get a
declared order (`pass < flag < discard`, `true < undetermined < false`, `routed < declined <
unavailable`) so the axis reads as severity; open ones (participant, task, declared_family) order
by frequency with ties broken by name. Where the bands are narrower than a label the axis draws
every *n*th label and says how many categories there are — 1,527 for `participant` — rather than
overprinting.

The nine sentinel-only categoricals say so: their option reads *"this column only ever carries
`NOT_SEPARABLE_BY_THIS_DESIGN`, so it is a presence flag"*. `m_carrier_rejected` is the real one
and its option points at the five gate names it carries.

---

## The reduction is disclosed, not implied

Every `m_<name>` axis carries a caption under its selector that says it is a mean, and how many
rows in the current set fold more than one reading, with the maximum:

> `mean · 6,094 of 6,601 fold >1 reading (max 45)`

and, where nothing folds, it says that instead — *"mean · every reading is a single measurement"* —
rather than leaving the reader to infer it. Only 4 of the 17 ever fold over this corpus:
`breath_peak_over_floor_db` (92.3% of its carriers, max 45), `cough_peak_over_floor_db` (81.2%, max
18), `interruptions` and `phonation_onset_to_offset_s` (31.2%, max 55). The recording view's
measurement table says it per recording: *"arithmetic mean of 9 readings"* against the single
number.

---

## The recording view

Nine lanes on one time axis, drawn from the compact vectors, beside the decision record, the
transcript and the measurement table. The waveform is 256 min/max pairs over ±`wave_peak`; the
envelope is 256 max-per-bucket points over [−100, 0] dBFS with `floor_dbfs` as a horizontal rule;
continuity is 256 mean-per-bucket points over [0, 1.05]; spans are five rows E C A S G; the top
classifier label is three lanes, one per classifier, shaded by score; SQUIM is three lanes of
per-span bars over the ranges the schema declares; the ASR lane is word extents; PII marks overlay
those extents with the category drawn in; branch lanes are AIRWAY SPEECH VOICE REDACT with roles.

Three decisions worth writing down:

**An absent producer is written, not drawn as an empty result.** A lane whose block is null renders
as an amber band saying *"absent — the PII scan did not run"*, and never as an empty white lane
that would read as "ran, found nothing". A block that is present and empty renders as *"scanned,
nothing found (pii_findings_n = 0)"*. Those are different sentences on purpose.

**Word text is not fitted to a word bar.** The bars are acoustic extents; a word's text length has
nothing to do with its duration. The lane stays compact and the words live in the transcript panel,
where the PII marks are shown in place as a highlight with the category as a superscript, and
hovering a bar highlights its word.

**`spans_unrowed_n` is said on the canvas.** The schema records that REDACT's redaction spans carry
no row code and are drawn nowhere by the figure; where a recording has any, the span lane says
*"N span(s) carry no row code and are drawn nowhere"* rather than dropping them silently. The
redactions themselves appear in the REDACT branch lane.

---

## What the schema turned out to be ambiguous about

The schema is a good contract — every layout, every enumeration and every parallel-column rule
decoded first time, and the worked example is byte-exact. Four things a decoder hits that it does
not say, in descending order of how much they would cost to get wrong.

**1. `time_scale_s` and `duration_s` are bit-identical on every row of this build, so the trap it
warns about cannot be caught by testing against this file.** §1 is emphatic — *"`duration_s` is the
*source* recording's duration; `time_scale_s` is the conditioned stream's… Use `time_scale_s` and
nothing else"* — and a reader reasonably infers the two differ in practice. They do not: over all
62,459 rows where both are present, `duration_s == duration_conditioned_s == time_scale_s`
exactly, with zero exceptions. A page that used `duration_s` would produce pixel-identical output
today and break silently on the first recording whose conditioning changes its length. The schema
should say that the divergence is currently latent, because "the two differ" reads as a claim about
the data and is not one.

**2. `hyparquet` returns every `BYTE_ARRAY` as a UTF-8 *string* unless `utf8: false` is passed.**
The schema's own JS example takes `bytes` and builds a `DataView` over `bytes.buffer` — which is
exactly right, but says nothing about how one gets a `Uint8Array` out of a reader. With the
default, `env_dbfs` arrives as a 247-character string instead of 256 bytes (UTF-8 collapses the
high bytes) and nothing throws. This is the single sharpest edge in the whole exercise and it costs
one option. It belongs in §1 beside the `DataView` snippet.

**3. The out-of-range behaviour of a byte enum is specified for `255` and undefined for everything
else.** `span_labels.classifier` documents *"255 = not one of those"*, and `asr_words.outcome`
likewise; `spans.row` and `branch_lanes.lane` document no sentinel at all. So what should a decoder
do with `spans.row == 5`? The producer can never write one — `SPAN_ROWS.index` would raise — but a
decoder has to choose, and choosing "clamp to the last row" or "treat as 0" would draw a real span
on a wrong row. This decoder throws for an undocumented code and returns null for a documented
`255`; the schema should say which it wants, because the two behaviours are not distinguishable by
reading it.

**4. A parallel string column's length is stated as a rule but not as an obligation on the
reader.** §4 says the parallel column has *"one element per record of its block, in the same
order"*, and that it is null exactly when the block is null. It does not say what a reader should
do when that does not hold. It always held here — checked over all 62,488 rows, for all four
pairs — but a decoder that trusts it silently reads `undefined` as a label. This one throws.

A fifth, cosmetic: the Python test the schema cites,
`test_the_worked_example_in_the_schema_document_decodes_as_written`, has a docstring that says
"nine bytes" where the assertion is five. The bytes are right.

---

## Testing

The decoders and the axis model are pure JavaScript with no DOM, and are tested by `node --test`:
**50 tests** in `src/tests/audio/workflows/triage/viewer/{decode,axes}.test.mjs`. They cover the
worked example byte-for-byte (including a big-endian read producing a *different* answer, which a
round trip cannot see), full scale being 65535 and not 65536, the fixed widths as literals, every
declared range, null-vs-empty for every block, the `255` sentinel not collapsing to the first term,
an undocumented enum code throwing, parallel-length mismatch throwing, cross-block index overrun
throwing, nulls surviving inside a vector, and the drawing rule — that an absent value breaks the
line and that an absent vertex carries `y: null`.

Every fixture is synthetic. No byte in any committed file came from a recording, and
`viewer_test.py` asserts that by pattern over the page and every part.

Node is not in this repository's CI and there is no `package.json`, so the pytest side does two
things: it **runs the node suites when node is on PATH and skips when it is not**, and it holds a
set of **drift guards that run everywhere** — the JavaScript's `SPAN_ROWS`, `CLASSIFIERS`,
`WORD_OUTCOMES`, `LANES`, `TIME_SCALE`, `TRACE_POINTS`, `UNKNOWN_CODE`, `SCHEMA_VERSION`, every
value range, every SQUIM range, every record size against `struct.calcsize`, and all 29 measurement
names in their four kinds, each asserted equal to `recording_vectors.py`'s own. A byte enum that
stops matching the producer is the failure those exist for, and it is the failure that would make
the page draw a confident wrong picture.

`viewer_test.py` also asserts the committed page is byte-identical to a fresh build (the build
carries no timestamp), that it references no `http(s)` origin and calls no `fetch` or
`XMLHttpRequest`, that every default axis is an assignable column the producer actually writes, and
that the app's block list covers every `binary` column in the producer's schema.

### What is not tested

Canvas pixels. No Chromium is installed on this host, so the page was driven in **jsdom** with a
recording 2-D stub: that exercises the whole data path, every DOM panel, the axis rail, the list,
brushing and the recording view's lane plan against the real 62,488-row file, and it checks *what*
the canvas was asked to draw — but it cannot say the result is legible at 1400 px. First paint and
selection latency in a real browser are also unmeasured; the numbers above are the node and jsdom
costs of everything except rasterisation.
