# The summary is the figure

`summary.pdf` is now the decision record followed by the paginated figure — preprocess, taxonomy
and the branch lanes on one product. The code is `nodes/figure.py` (the drawing) and
`nodes/report.py` (the document and the orchestration); nothing here is repeated in either, per
CLAUDE.md.

## What was asked

Owner, 2026-09-17: *"we don't want the current summary. instead the preprocessed figure should be
extended to be the new summary incorporating branch outputs"* — superseding the sibling
`branch_figure` shipped hours earlier, which is withdrawn.

## Correction to the brief's premise: the old summary was already paginated

The brief framed this as trading a fixed-size product for a paginated one, and asked what that does
to the page count. **The old `summary.pdf` was already one page per ten seconds of audio**, plus
decision pages — `_timeline_windows` at `_TIMELINE_PAGE_SECONDS = 10.0`, one `plot_aligned_panels`
call per window. The "two-page summary" was an artifact of the test fixture's six-second recording,
and the suite said so itself: `test_a_long_recording_gets_one_ten_second_evidence_page_per_interval`
asserted that a 25 s recording renders **four** pages.

So pagination is not a new cost. It is a **reduction**: the figure's pages are `page_seconds = 20.0`
wide, half as many per minute of audio.

| recording | old summary.pdf | new summary.pdf |
|---|---|---|
| 6 s | 1 timeline + 1 decision = **2** | 1 decision + 1 cover + 1 evidence = **3** |
| 25 s | 3 timeline + 1 decision = **4** | 1 decision + 1 cover + 2 evidence = **4** |
| 70 s | 7 timeline + 1 decision = **8** | 1 decision + 1 cover + 4 evidence = **6** |

Both counts are measured, not estimated: the 70 s row is
`test_a_long_recording_gets_one_evidence_page_per_page_width`.

## Page order — the decision record leads

The decision record is **first**, then the figure's cover, then the evidence pages. It used to
trail. Three reasons to move it:

1. It is what a reviewer reads first, which is what the brief asked the cover to carry. Rather than
   duplicate it onto the cover, the record *is* the front matter.
2. On a file with no axis to draw — one ADMIT refused — it is the whole product, and a product whose
   only page is an apology reads better with the decision on it.
3. It is fixed content. Putting variable-length evidence in front of it would move the decision to a
   page number that depends on the recording's length.

The figure's cover follows and carries what the decision record does not: the source path, the
consensus alignment record, the per-classifier whole-file score distributions, the enhanced and
residual stream summaries, and the per-lane report block — conformance, deviations, unmeasured
points, measures, and how many proposals named an initial span.

## What happened to `report()`'s panels

| old layer | now |
|---|---|
| waveform + `envelope dBFS` twin + `spans (dB over floor)` overlay | **kept** — the figure's waveform panel draws the envelope, the floor, the `k_db` line and the continuity trace on their own scales, which is strictly more |
| `yamnet` / `ast` / `hear` window rasters | **replaced** by the figure's per-span rasters, which draw the model's own `raw_scores` per proposed span rather than a thresholded per-window label set. The whole-file label distribution is on the cover. **A deliberate change of content, not a like-for-like move** |
| airway HeAR raster | **dropped** — a second view of the same `span_hear` scores the figure's HeAR raster draws per span |
| `speech spans` paired lane | **kept** — as the SPEECH lane, same `wasDerivedFrom` pairing |
| `consensus ASR` token lane | **kept** — the figure's own ASR lane, which additionally draws each source's extent in its own sub-band |
| `airway` paired lane | **kept** — as the AIRWAY lane |
| `voice` paired lane | **kept** — as the VOICE lane |
| `redacted transcript` token lane | **replaced** by the REDACT lane, which draws the redaction spans REDACT actually planned, captioned by category, rather than the word lane with placeholders substituted. The redacted transcript text stays in `summary.json` and on the PNG form's blocks |
| spectrogram | **kept** — the persisted wideband array PREPROCESS wrote, not a fresh STFT |
| — | **new**: the span-source lane (E/C/A/S/G), SQUIM per span, the padded uniform page width, the per-lane run states |

| old block | now |
|---|---|
| `_decision_blocks` | **kept verbatim**, as the leading pages |
| `_header` — task/context, primary decision, leading evidence, screening summary | **dropped from the PDF.** It was a banner on every timeline page; all four fields are in the decision record now leading the document, so repeating them above each evidence page restated the front matter on every page. Still drawn on the PNG form and on the report-only page |
| `_blocks` — the long form | **unchanged**; it was never in the PDF, only the PNG |

## `summary.json` — one record, not two

`branch_summary.json` is gone. Its content folds into `summary.json` as `evidence.lanes`: one record
per declared lane, present whether or not the lane drew anything, each carrying the lane's state and
every bar with the ids it names in `wasDerivedFrom`. That makes the pairing a page draws as a
connector checkable without reading pixels, and keeps one record rather than two that can disagree.

`REPORT_SCHEMA_VERSION` goes `v6` to `v7`. The sweep found no consumer outside `report.py` and one
test asserting the literal, so the bump costs nothing and the JSON did gain a key.

## Where the code lives, and the cycle it forced

The drawing is in `figure.py`; the document, the decision blocks and the orchestration stay in
`report.py`. `report()` remains the entry point, because `run.py` calls it, `triage_audio.py` prints
its return dict and `run_test.py` asserts on `summary_dir` — changing that ripples into the runner
for no gain.

`figure.py` imported four names from `report.py`, so `report.py` importing the figure would have been
a cycle. All four are product-neutral reads over the store, and they moved to `common.py`, where the
writer they read back already lives: `BRANCH_MEASURES` now sits beside `write_report`, which is what
puts those keys in the store.

`summary_pages` is the single page builder. It **yields** `(name, figure)` unsaved and unclosed, and
the caller closes each one, so only one figure is live at a time however long the recording — the
guarantee `test_every_page_is_saved_and_closed_before_the_next_is_built` has always pinned, now
pinned through the generator.

`preprocess_figure` survives as a second consumer of the same generator, so the pages can still be
regenerated by hand over a completed run directory without rebuilding the decision record.

## The REDACT lane

REDACT is a graph edge, not a routed branch: no `branch_decision`, a `verdict` rather than a
`branch_report`, and spans carrying `name` and `category` rather than a family and a role. Its lane
is therefore built apart from the family loop, its two states read from the verdict alone, and its
reported measures are the four keys `redact.py`'s own `write_verdict` call carries — `redactions_n`,
`verified`, `survived`, `outstanding`. An earlier draft of this lane read `categories` and
`widened_n`; neither is written by anything, and they were caught by checking the writer before
committing rather than after.

## Names read that nobody writes — this pass

1. **`_header` read `decision.get("verdict")`** off `document["routing"]`, which `_branches` builds.
   `_branches` writes no `verdict` key — its own docstring says so, because a branch writes no
   verdict. Every summary ever rendered carried `outcomes: AIRWAY=—; SPEECH=—; VOICE=—` on the banner
   above every evidence page. The conformance it meant to show sat beside it under `conformance`.
   **Fixed.**
2. **`_decision_blocks` read `decision.get("flags")`** for MEASURED BRANCH FINDINGS. ROUTING writes
   `flag_gates`. No flag could appear on that line. **Fixed.**
3. **The figure resolved `preemphasised` then `plain` and refused; `report()` resolved `plain` then
   `recording`.** A run where PREPROCESS wrote no conditioned stream but ADMIT decoded the file —
   what a conditioning failure produces, and what the runner's own faked graph produces — would have
   lost every evidence page from a summary that had them before. The three-step chain is now shared.
   **Fixed**, and it is why four `run_test.py` tests failed before it was.

Carried forward from the previous pass and unchanged: the four run states, the `wasDerivedFrom`
pairing rather than extent overlap, branch-level facts on the cover rather than inside a time page,
`key in attributes` rather than `.get(...) is not None`, and the short-label fallback that keeps a
narrow bar captioned.

## The span axis — one axis, not four lanes

Owner, 2026-09-17, having looked at the rendered PDF: *"just keep the preprocessing figure and
adjust the span axes to initial spans and one row per proposed spans from different branches."*

The four per-branch lanes, each with its own initial and proposed rows, become **one axis**: the
initial spans on the top row, then one row per lane — AIRWAY, SPEECH, VOICE, REDACT.

It is more compact, and it is truer to the structure. Every proposal derives from the same initial
population, so the four-lane layout drew that population up to four times and could not show that
one parent fed two branches: the fact appeared as two unrelated bars in two unrelated panels.
`initial_rows` deduplicates by entity id across every lane, so the shared parent is one bar.

### Connectors from one parent to several rows

This is the case that makes the layout worth making, and it is also where it first went wrong.

Drawn naively — every connector leaving the parent's centre — the lines from one parent to three
branch rows are **collinear**. They overlap exactly, and the last drawn paints over the rest, so a
parent feeding three branches renders as a parent feeding one. The render showed this: what should
have been an orange, a green and a lavender line was one grey-lavender line.

`parent_anchor` fixes it. Each lane leaves the parent at its own fraction of the bar's width,
`(lane_index + 1) / (lane_count + 1)` — strictly inside the bar, evenly spread, symmetric about its
centre, and deterministic. The fan is then visible at any bar width.

Three further rules keep a dense derivation readable:

- **Colour follows the lane.** A lane's bars and its connectors share one fill, so a connector
  crossing two intervening rows can be followed by colour to the row it lands in. `lane_colours` has
  one entry per lane and a test pins that they are distinct.
- **Connectors sit behind every bar** (`zorder=2` against the bars' `3`) and below full opacity, so
  a derivation never obscures the spans it relates. A connector crossing a row disappears under that
  row's bars and re-emerges, which reads correctly: it is passing through, not landing.
- **A connector is drawn only where both ends are on the page**, unchanged from the paired lane.

### A branch that did not run

The distinction survives, and the axis is where it is said. Each lane keeps a row whatever its
state; a row with no bar on this page carries its lane's note instead — *did not run — route
declined*, *ran and proposed no voice span*, *was selected to run and wrote no report*, *ROUTING
wrote no decision*. That is the same four-state vocabulary as before, now drawn in the row rather
than in a panel title.

Consequently **the span axis is never collapsed**, unlike the spectrogram and the rasters above it:
a collapsed row cannot carry its note, and the note is the finding. A mutation dropping rows for
lanes that did not run fails nine tests.

## A branch is one block of rows, not one stacked row

Owner, 2026-09-18, reading the rendered summary: *"the last added panel seems to be showing
multiple speech outputs in a single lane."*

### What was on the page

One row per lane put every span a branch minted on one y. SPEECH mints four kinds — `task_extent`
over the whole response, `phrase_run_{index}` inside it, `structure_{index}` over the realised
stimulus units, and, since DDK's train landed in the `speech` family, `ppg_train`. They nest by
construction, so they were four rectangles drawn over one another. Worse, the caption did not
separate them: `_proposed_span_label` captions a proposal `role/qualifier`, and the qualifier for
all four is the same speaker, so the render read
`phrase_run_0/SPEAKEstructure_0/SPEAKER_00hrase_run_1/SPEAKER_0task_extent/SPEAKER_00` — four
captions written into the same pixels. VOICE had it too, with `task_extent` and `phonation` both
captioned `sustained`.

### The treatment: a sub-row per role kind, inside one banded block

The constraint was to keep one visual block per branch. A lane is therefore a **block of sub-rows,
one per kind of span it proposed**, held together by a band drawn behind the whole block in a very
light tint of that lane's own colour. Bars, connectors and band share a hue, which is also what
answers "which row does this connector land in" without following the line.

**The kind is the role with any trailing `_<number>` stripped** (`span_role_kind`, in `common.py`
so both renderers read it the same way). A proposer that mints one span per realised unit numbers
the role — `phrase_run_0`, `phrase_run_1`, `structure_0` — and the number distinguishes the spans,
not the kind. Keying on the raw role would have been a row per span, which is not more readable
than a row per branch; keying on the kind is three rows for SPEECH's runs and structures however
many of each there are. Spans of one kind are disjoint by construction in every proposer, which is
what makes one row per kind sufficient rather than merely tidier.

**A lane with one kind is the single row it always was**, tick label and all. The role earns a place
in the tick only once the block is more than one line, where it is the only thing distinguishing
them: `SPEECH · phrase_run`. A withheld branch, which proposes nothing, is one row, so the
four-state vocabulary above is untouched.

**The kind leaves the caption once it is on the axis.** The bar now tries `short` (the qualifier)
before `label` (`role/qualifier`), so the width is spent on what the row label does not already
say. This is also why `SPEAKER_00` is now legible where `phrase_run_0/SPEAKER_00` was not.

REDACT's spans carry `name` and `category` rather than a role; they are all one kind of thing — a
planned redaction — differing by the category the caption already carries, so the lane takes
`REDACTION_NAME` as its kind and stays one row.

### The same defect in `report()`'s PNG form

`report.format: png` does not go through the span axis; it draws `_derived_lane`'s token lanes, and
those stacked roles in one `proposed` row for exactly the same reason. They now take the role kind
as the token row, under the same rule — one kind keeps the single `proposed` row, so AIRWAY's lane
is unchanged. The `segments` fallback (taken only when nothing in the lane has a live parent) keys
its rows on the label, so there the kind is prefixed to the label instead.

## The initial row now says what it is

Owner, same reading: *"unclear what the initial lane does."*

Four things were wrong with it, all found by looking at the render rather than at the code:

1. **Its tick said `initial`**, one word with no referent, sitting above four branch names. It read
   as a fifth branch. It now says `initial spans` over `what branches read`.
2. **Its bars said `20 dB`** — PREPROCESS's own reading of an envelope span, which states a level
   and never states what was measured over what. `initial_span_label` now names the kind before the
   reading: `envelope 20 dB`. The family/role form it returns for a branch-minted parent already
   named its kind and is unchanged.
3. **It was drawn exactly like a branch row.** It now carries a band of its own and the rule below
   it is the heavier between-block rule, so the input zone and the output zone are one visible
   split. The panel title names both zones rather than describing the whole thing at once.
4. **It went silent when it held nothing.** An empty row with no note cannot be told from a row
   whose bars are all on another page — the same distinction `lane_note` draws for a branch, which
   the initial row did not have. `initial_row_note` gives it one.

### What the render showed, after

Eight rows where there were five: the input band, AIRWAY (one kind), SPEECH split four ways
(`task_extent`, `phrase_run`, `structure`, `ppg_train`), VOICE split two ways (`task_extent`,
`phonation`), REDACT carrying its did-not-run note. Every caption legible, no bar over another, the
title fitting the page width. The panel's declared height already grows with its row count
(`_page_height_ratios`, `raster_row_ratio`), so the split costs nothing that has to be configured.

### What was not done

- **`BranchRow.row` still has its two values.** The zone (`initial`/`proposed`) is the pairing, and
  `summary.json`'s lane records key on it. The sub-row is a separate field, `role`, and the lane
  records carry it too — the JSON is the machine-readable counterpart of what the page draws, so a
  page drawing a row per role and a record not saying which row would no longer be that.
- **`report.py`'s speech caption is still `attributed_to or 'unattributed'`.** With the role on the
  row this no longer collides, but it reads wrongly for DDK's `ppg_train`, which carries a
  `production` and no speaker and therefore captions itself *unattributed* — suggesting a diarizer
  that failed rather than a span that is not a diarized run. `figure.py`'s `_proposed_span_label`
  already falls back through `label`, `production`, `attributed_to` and gets this right. Aligning
  the two is a caption-vocabulary change in a non-default product, not one of the two defects, and
  is left.
- **No packing within a kind.** Two spans of one kind overlapping in time would still collide. No
  proposer writes that, and inventing rows for a case nobody produces would be a layout fitted to
  nothing.
- **`nodes/ddk.py` and `BRANCH_MEASURES` untouched**, being concurrently edited. DDK's `ppg_train`
  is picked up by the shared rule with no per-branch knowledge, so it needs neither.
