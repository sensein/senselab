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
