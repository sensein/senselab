# The two views, and what they had to get right

The page is `src/senselab/audio/workflows/triage/viewer/recording_vectors_viewer.html`, assembled
from the parts beside it by `build.py` and rebuilt by
`uv run python scripts/triage_vectors_viewer.py`. It is **one self-contained file**: 164 KB, no
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
| one hover hit-test over 62,488 lines | **1.6 ms** (59 ms before the geometry cache) |
| brushing an axis and redrawing | ~1.3 s in jsdom, of which the paint is a stub |

Hover was the one interaction that did not survive the corpus size. `pick()` interpolates every
drawn line at the pointer's x, and doing that through `readValue` — a string-keyed property read
plus a scale computation, per row per axis — cost **59 ms per `mousemove`**. The fix is one
`Float64Array` of y per axis, rebuilt on layout or axis change, with `NaN` for absent: 1.6 ms, and
the same array is what the paint loop walks. `NaN` compares false against everything, so an absent
endpoint can never win a nearest-line contest — which is the behaviour we want anyway, since that
segment was not drawn.

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

## The subject axis draws a key, not the id

A BIDS subject id is `sub-` and a UUID: **40 characters**. On a categorical axis the tick labels
are right-aligned against the axis line and truncated at 16, so the subject column spent about
96 px of a ~140 px gap on a label the reader could not finish and could not tell from its
neighbour. The owner's word for it was *swamps the column*.

The axis now draws the **first eight hex digits of the UUID** — the UUID's own first group, with
the `sub-` dropped. `session` gets the same treatment against `ses-`. Everything else
is drawn exactly as the column carries it.

### The margin, measured on the r3 corpus

`specs/20260922-compact-recording-vectors/axis-discrimination.py` over the 62,548-row file:
1,527 distinct participants and 1,736 distinct sessions, every one 40 characters, every one on
its prefix.

| key length | participant: colliding keys / ids | session: colliding keys / ids | birthday expectation, 1,527 ids |
| ---: | ---: | ---: | ---: |
| 4 | 25 / 51 | 28 / 56 | 17.8 |
| 5 | 2 / 4 | 1 / 2 | 1.11 |
| **6** | **0 / 0** | **0 / 0** | 0.069 |
| 7 | 0 / 0 | 0 / 0 | 0.0043 |
| **8 — shipped** | **0 / 0** | **0 / 0** | **0.00027** |

Six is already collision-free on both columns. Eight is shipped because it is two hex digits —
**256×** — of headroom over the shortest length that happens to work on today's corpus, it is the
UUID's own first group rather than an arbitrary cut, and it holds as the corpus grows: the
birthday expectation is 0.012 collisions at 10,000 subjects and 1.16 at 100,000. Below six the
key is not safe on this corpus, so the margin the page relies on is real rather than assumed.

The observed collisions track the birthday expectation at every length, which is the check that
these UUIDs are random in their leading digits rather than sequential — a sequential id would
collide catastrophically at a prefix and the table would not look like this.

### Display, not a column

The shortening is a **transform in the page**, not a `participant_key` column in the parquet.

- The key's length is a **rendering** decision — how many monospace characters fit beside a canvas
  tick at a given axis spacing. A column would freeze a presentation choice inside a data
  contract, and the next reader with a wider screen could not change it without a rebuild.
- A second identity column is a parallel field: two things that name the same subject and can
  drift. Pre-alpha, this repository renames and replaces rather than adding aliases.
- It is free either way for recoverability, because the full id never leaves the row. A reader
  who wants the recording on disk already has it.
- A rebuild is the expensive half and buys nothing a `str.slice` does not: any pandas or duckdb
  reader can cut the same eight characters, and the parquet stays on schema 4 with no bump.

`SchemaAxes.categoryLabel(col, value)` is the whole of it: the column spec carries `shortKey`
(`'sub-'`, `'ses-'`) and `KEY_CHARS` is 8. Three call sites use it — the axis ticks, the colour
legend and a single-term brush chip. A value that does not start with the declared prefix is
returned untouched, so a label can never invent a key out of something that is not an id.

### Where the whole id survives

Four places, and the page says so rather than leaving the reader to find out:

- the **recording panel** prints it in full, with the key beside it —
  `sub-<uuid>   (on the axis: <its first 8 hex digits>)` — so the axis and the panel can be
  matched up without counting characters;
- the **hover line** under the plot carries the whole `stem`, which contains both ids;
- the **legend swatch** carries the full value as its `title`;
- the **row data** is untouched: `row.participant` is the id, and every filter, facet and brush
  works on the id, not on the key.

And the axis **caption** says what it did: *`1,527 ordered categories · first 8 of the id ·
whole in the panel`*. A silently shortened label would be the wrong
failure — the reader would have no way to know the axis was not showing them everything.

`session` gets the same treatment for the same reason and at the same margin. It is not a default
axis (it is not a fact about the recording that the fold decided on), but it is one `selectOption`
away, it has the same 40-character shape, and an axis that behaved differently from its twin would
be a trap.

---

## The axis defaults, and the numbers behind them

Owner-directed, the first three are `participant`, `task`, `verdict`. The remaining **seven are
chosen by a measured statistic over all 175 assignable columns** of the r3 corpus, not from a
candidate list. `axis-discrimination.py` reproduces every number here.

### What discrimination is, and why it is not spread

An axis earns a slot when a random pair of drawn recordings is **placed and separated** by it, and
when that separation **carries the fold's decision** rather than merely varying.

> **separation**  `sep(X) = c² · (1 − Σ_b p_b²)`
>
> the chance that both members of a random pair carry a value on X *and* fall in different
> resolvable bands. `c` is coverage, so a column absent on most rows is punished **quadratically**
> — a 7.5%-covered measurement cannot exceed 0.0056 however beautifully it varies. `p_b` are the
> shares among the present rows over **64 bands**: the value band is about 446 px at 62vh, so 64
> puts them 7 px apart, comfortably above a 1 px line and short of pretending a 1,527-category
> axis resolves all 1,527. A numeric is banded linearly over its range, because linear is what the
> axis opens as; a categorical merges adjacent categories in its own axis order once there are
> more than 64.
>
> **relevance**  `rel(X) = I(X ; D) / H(D)`,  `D = (verdict, release)`
>
> the share of the fold's published decision that the axis resolves. `H(D) = 1.1872` nats over
> eight joint levels on this corpus. A column that spreads perfectly and is independent of what
> the fold concluded scores ~0 and is decoration on a triage page.
>
> **discrimination**  `disc(X) = sep(X) · rel(X)`

This **betters** the coverage-plus-spread standard the schema-3 defaults were chosen by rather than
abandoning it: coverage and spread are both inside `sep`, and `rel` is the third property that
standard had no measurement for. That it is needed is not a matter of taste. Ranking by `sep`
alone and applying the same redundancy filter gives

> `participant, task, verdict, session, enhanced_over_residual_rms_fitted_db, residual_rms_dbfs,
> residual_peak_dbfs, wave_peak, enhanced_rms_dbfs, floor_dbfs`

— Σsep 6.75 against the schema-3 set's 2.19, **three times the separation and not one axis that
says anything about the fold**. A page that opened on six dBFS columns would be a level dashboard
wearing a triage viewer's chrome.

`verdict` and `release` are components of `D` and score high on `rel` by construction. They are
marked as such in the ranking; neither was *selected* by it — `verdict` is owner-directed and
`release` is the fold's second published outcome.

### Two constraints on the walk

**Redundancy.** Walking the ranking, a candidate is skipped when its symmetric NMI with an
already-chosen axis reaches **0.70**: it is saying something the page already says. This is what
removes `gate_family` (NMI 1.000 with `declared_family`), `flag_nodes.size` (1.000 with
`flags_n`), `duration_conditioned_s` and `time_scale_s` (1.000 with `duration_s` — §1's latent
divergence is still latent), `gate_group` (0.724 with `declared_family`) and `gate_failed_n`
(0.717 with `verdict`). The cut is not tuned: **0.60 and 0.70 select the same seven**, and only at
0.80 does the set change.

**Resolution, as a floor rather than an objective.** A set of ten three-band axes draws at most a
few thousand distinct polylines whatever each one measures, and 62,548 recordings collapsed onto
a few hundred paths is a picture in which the reader cannot tell 200 recordings from one. The
floor is the set being replaced: **a new default set may not draw a coarser picture than the old
one** — at least 21,119 distinct polylines, largest bundle no worse than 124.

The greedy walk on `disc` alone lands on 17,923 polylines, below the floor. The repair considers
**every single swap** that clears it and takes the one with the best total discrimination:
`−gate_node +enhanced_over_residual_rms_db`, which costs 0.0017 of Σdisc and buys 32,337
polylines.

### The seven, and what each beat

| slot | axis | non-null | bands | sep | rel | disc | why it is here |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `participant` | 100% | 64 | 0.982 | 0.011 | 0.011 | owner-directed; 1,527 levels, the only axis that shows a subject's session as a bundle |
| 2 | `task` | 100% | 62 | 0.873 | 0.082 | 0.072 | owner-directed; 796 levels |
| 3 | `verdict` | 100% | 3 | 0.275 | 0.379 | 0.104 | owner-directed; pass 52,255 / flag 10,264 / discard 29 |
| 4 | `declared_family` | 100% | 48 | 0.930 | 0.334 | **0.311** | **the strongest column in the file.** 48 families, never null, and the unit the gates actually resolve against — harvard-sentences-list 13,700 / respiration-and-cough-fivebreaths 3,572 / free-speech 3,073 … |
| 5 | `release` | 100% | 4 | 0.420 | 0.622 | **0.261** | the fold's second published outcome, and independent of `verdict` (NMI 0.00): nothing_to_redact 45,854 / releasable 12,046 / withheld 4,647 / not_assessed 1 |
| 6 | `flags_n` | 100% | 5 | 0.375 | 0.488 | **0.183** | 0–4; 48,125 recordings carry none. Never null |
| 7 | `gate_applied_n` | 100% | 4 | 0.626 | 0.094 | 0.059 | how many of VERDICT's gates were applied at all — the scrutiny the recording got. Never null |
| 8 | `route_speech` | 100% | 3 | 0.446 | 0.130 | 0.058 | routed 41,565 / declined 20,954 / unavailable 29 |
| 9 | `duration_s` | 100% | 62 | 0.778 | 0.073 | 0.057 | p5 2.69 s, p50 7.32 s, p95 57.1 s, 16,159 distinct |
| 10 | `enhanced_over_residual_rms_db` | 100% | 60 | 0.975 | 0.054 | 0.053 | the exact RMS difference in dB between the enhanced and residual streams (§8) — 62,518 distinct, near-uniform over 60 bands, p5 −35.8, p50 28.3, p95 56.4. The resolution slot |

Σdisc **0.981** against the schema-3 set's 0.650, Σsep **4.55** against 2.19.

### What the schema-3 defaults lost, and why

| dropped | sep | rel | disc | why |
| --- | ---: | ---: | ---: | --- |
| `conformance_airway` | 0.077 | 0.110 | 0.009 | **38.6% non-null**: nearly two lines in three break there, and `sep` charges that quadratically. What it was showing — AIRWAY declined — is a value on `route_airway`, not an absence |
| `conformance_speech` | 0.132 | 0.250 | 0.033 | same, at 69.5%. `route_speech` carries the same fact (NMI 0.682) at 100% coverage and beats it 0.058 to 0.033, so the branch structure is shown *better*, not hidden |
| `gate_failed_n` | 0.245 | 0.281 | 0.069 | NMI **0.717** with `verdict`, which is on slot 3. Two slots on one fact |
| `pii_findings_n` | 0.161 | 0.240 | 0.039 | 30.5% null, and NMI 0.273 with `release`, which explains it at full coverage |
| `release_ground` | 0.361 | 0.542 | 0.196 | see below — it ranks 4th and is excluded anyway |

The consequence worth naming: **every one of the ten is 100% non-null, so 62,518 of 62,548 lines
are drawn whole** where the schema-3 set drew only 8,425 (13.5%). That is not the earlier design
hiding absence. Absence is still a first-class value — the absent rail, the `absent N` chip and
the facet panel's `(absent)` are all unchanged, and every axis that has nulls still shows them.
What has changed is that the page no longer *opens* on a picture that is 86% broken lines, and
absence becomes something the reader goes and looks at rather than the first thing the plot is.

### `release_ground` ranks 4th and is excluded, on three measured grounds

It would take a slot on `disc` 0.196. It does not, for three reasons, each of which is a number:

1. **Its labels are sentences.** The grounds `vocabulary.py` declares are prose by design —
   *"SPEECH did not run, so no transcript exists for a redaction to read"* is 67 characters, and
   **three of the five start with `SPEECH `**, so at the axis's 16-character truncation three of
   five are indistinguishable. That is the very complaint the subject key was shortened to fix.
2. **Its rel is circular.** `D` contains `release`, and `release_ground` is `release`'s own
   explanation. Measured against `verdict` alone its rel is 0.143 and its disc 0.052 — rank ~15,
   not rank 4. Nothing else in the top ten has that property.
3. **It pairs with `release` at NMI 0.531**, which is two of seven slots on one fact — the
   objection `views.md` already raised against pairing `route_*` with `conformance_*`.

It is one `selectOption` away. **Promotion is one mechanism away**: a label rule that shows a long
categorical from its point of divergence rather than from its first character would unlock it, and
it is the strongest thing such a rule would unlock.

### Eligibility: a default is a fact about the recording, not about the artefact

Ranked with everything else and then refused a slot, because an axis that measures the encoding
would be a slot spent on our own bookkeeping:

- **`spans_unrowed_n`** — rank 8 on `disc` 0.169, and the closest call in the whole exercise. It
  counts spans the figure *does not draw*; its relevance is a back door onto `release` (NMI
  **0.674**, just under the redundancy cut). Recorded here because it is the one exclusion a
  reader might reasonably dispute.
- **every `m_<name>_n` and `_width`** — how many readings the store held is a fact about how the
  graph ran. The best of them, `m_extent_speaker_count_n`, reaches `disc` 0.049.
- **every `gate_<name>_bound`** — the threshold, not the recording. All thirteen that exist are
  constant over this corpus and score `sep` 0.0000 anyway; the bound already reaches the page as a
  dashed reference line on its reading's axis.
- **`schema_version`, `malformed_store_lines`** — one value each.

### Every `m_<name>` is still one click away, and the numbers say why

The best-covered numeric measurement, `m_expected_sequence_repeat_fraction`, is present on
**29.0%** and scores `disc` 0.006; the median one is present on 7.5%. A measurement default would
be an axis absent for nine lines in ten. They belong in the picker, which is where a per-family
reading belongs in a corpus of 48 families.

### `dominant_speaker_share_min` is degenerate — the prediction, now measured

Schema 3 named it *"the strongest gate candidate by far"* and said its spread was unmeasured, with
the warning that *"if diarization returns one speaker on most single-target recordings it piles at
1.0 and the axis is degenerate"*. Over the r3 corpus it does:

| | |
| --- | ---: |
| `gate_dominant_speaker_share_min` present | 38,990 of 62,548 (62.3%) |
| distinct values | 2,034 |
| exactly 1.0 | **36,944 — 94.8% of the present readings** |
| ≥ 0.99 | 37,499 (96.2%) |
| below 0.95 | 860 (2.2%) |
| p1 / p5 / p50 | 0.815 / 0.998 / 1.000 |
| Simpson over 64 bands, among present | 0.0764 → **1.08 effective bands** |
| `sep` / `rel` / `disc` | 0.030 / 0.132 / 0.0039 |

An axis on which 95% of the drawn lines sit on one tick is not an axis. It is not a default and
should not be proposed as one again without a corpus on which it moves. `extent_dominant_speaker_share_min`
and `m_extent_dominant_speaker_share` are the same numbers (NMI 1.000 with it, all three).

Of the other gates, nothing changed: eight of the 21 carry `reading: None` and have no
per-recording scalar at all, so they cannot be a value axis whatever their coverage;
`expected_tokens_matched_min` and `omissions_max` reach 33.4% each but are integer counts whose
range is the stimulus length and are not comparable across `harvard-sentences-list` and
`word-color-stroop` on one axis; and `coverage_min` was withdrawn on 2026-09-24
(`specs/20260924-recall-conformance-is-production/design.md`).

### How a gate reaches the page

Three columns per gate, in their own picker groups so they do not crowd the measurement list:
`gate_<name>` (the reading, a numeric axis), `gate_<name>_bound` (what it was read against) and
`gate_<name>_passed` (`true` / `false` / `undetermined`, ordered that way).

On a gate reading axis the **bound is drawn as a dashed reference line and the failing side is
shaded** — below the line for `at_least`, above it for `at_most`. The bound resolves family over
group over default, so one axis can carry more than one bound over the drawn set;
`CorpusView.boundsFor` returns each with the number of drawn recordings it governs, and each is
drawn and labelled. Above four distinct bounds nothing is drawn, because a band crossed by five
reference lines reads as noise. In the packaged config no gate's bound varies by group, so in
practice there is one line.

The bound is a property of the *recording*, not of the axis, which is why it travels as a column
rather than as a constant in the page. A future `by_family` layer changes the picture without
changing the page.

### `duration_s` needed a second scale


`duration_s` is the strongest numeric axis and also the worst-behaved one: p5 2.7 s, p50 7.3 s,
p95 57.1 s, **max 333 s**. On a linear axis that puts the median at 1.4% of the band and 95% of
every line in the bottom sixth, which is not a readable axis however honest it is. Every numeric
axis therefore carries a **`linear` / `log10` chip**, the caption says which is active, and the
tick labels are inverted through the same scale so they stay real seconds. Under log10 the median
sits at 21% and p95 at 68%.

The chip is **disabled, with the reason in its tooltip, when the domain reaches zero or below** —
`pii_findings_n`, `flags_n` and `floor_dbfs` among them. Silently falling back to linear would be
the wrong failure: the axis would say `log10` and draw something else.

Linear stays the default. A log axis is a real distortion of distance and should be a thing the
reader turned on.

**Under these defaults 59,649 of 62,488 lines break at least once and only 2,839 are complete.**
That is not a flaw in the view; it is the shape of this corpus — AIRWAY declined on 61.4% of
recordings and VOICE on 63.8% — and a view that hid it by choosing denser axes would be hiding the
single largest structure in the data.

---

## Narrowing is two instruments, not one

Brushing an axis was the only way to narrow the drawn set until 2026-09-24. With ten slots and 195
columns that makes filtering compete with reading: to keep only one task you had to spend a slot on
`task`. A **facet panel** now sits to the left of the plot and narrows by what a recording *is* --
every categorical and `set` column, the 22 gate outcomes included -- leaving all ten slots for
reading.

The two compose rather than replace: `selected = brushMask AND facetMask`, `clear brushes` leaves
the facets standing and `clear facets` leaves the brushes standing. Each facet value carries
`available / total`, and the panel header states the denominator on both sides of the narrowing, so
a facet can never silently move it. Absence is a selectable value there too, for the same reason it
is a first-class chip on an axis.

The design, the counts' semantics and what a click costs at this corpus size are in
[`specs/20260924-recording-vectors-facets/facets.md`](../20260924-recording-vectors-facets/facets.md).

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
**81 tests** in `src/tests/audio/workflows/triage/viewer/{decode,axes,facets}.test.mjs`. They cover the
worked example byte-for-byte (including a big-endian read producing a *different* answer, which a
round trip cannot see), full scale being 65535 and not 65536, the fixed widths as literals, every
declared range, null-vs-empty for every block, the `255` sentinel not collapsing to the first term,
an undocumented enum code throwing, parallel-length mismatch throwing, cross-block index overrun
throwing, nulls surviving inside a vector, log being refused rather than silently ignored on a
domain that reaches zero, the cached geometry agreeing with the direct computation, and the
drawing rule — that an absent value breaks the
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

### Headless Chromium, since 2026-09-24

Chromium is now installed, and `npx playwright test` drives the shipped page over `file://` at
1600 × 1000: **21 tests** in `{viewer,facets}.e2e.spec.mjs`, `workers: 1`, against a synthetic
240-row fixture `scripts/triage_viewer_fixture.py` writes. Every test fails the run if the page
raises or if it requests anything over the network, which is how the no-fetch property is checked
in the browser rather than only by pattern.

The two changes of 2026-09-24 are held there:

- **the short key** — the subject axis's labels are all `KEY_CHARS` long where the ids are all 40,
  the labels are unique across the drawn subjects, a 4-character key would merge a pair the
  fixture deliberately contains, each key is a literal prefix of its id, the caption says the
  labels are short, and the recording panel carries the id whole beside the key;
- **the defaults** — the rail's ten selects are `DEFAULT_AXES` in order, every one of the ten
  places at least one recording, and the ten together resolve more than half the corpus into
  distinct polylines rather than a handful of ribbons;
- **and a genuine `0.0` is still a value** — on `gate_train_min_s` every exact zero owns a real
  vertex with a finite `y`, every null is on the rail and nothing else is, and the `absent N` chip
  counts the nulls and not the zeros.

The fixture had drifted: four assertions were written as literals against schema 3, 22 gates and
an older generator, and are now computed from the file the generator wrote. The generator itself
now writes 40-character BIDS ids, the residual levels, an `undetermined` gate outcome and one span
block, so the page has something real to draw in each of those places.

### What is still not tested

Canvas *pixels*. Playwright captures screenshots to `artifacts/viewer_e2e/` and they are inspected
by eye, but nothing asserts legibility — a test can say the label is eight characters, not that
eight characters fit. First paint and selection latency in a real browser against the full 62,548
rows are also unmeasured; the numbers above are the node and jsdom costs of everything except
rasterisation.
