# FIGURE — PREPROCESS's and TAXONOMY's output, drawn from the store

What this node is, why the scratch tool it replaces could not stay, and the three decisions the
owner ruled on. The code is `src/senselab/audio/workflows/triage/nodes/figure.py`; nothing here is
repeated in it, per CLAUDE.md.

## Where it came from

A scratch generator, `preprocess_figure_tool.py`, lived in a session job directory
(`~/.claude/jobs/c1a50c05/tmp/`) and called itself "not a shipped script". It was the right figure —
it ran the real `admit()+preprocess()` rather than an approximation, and its panels were sound — but
it could not survive a machine restart, and it broke one rule badly enough to make its output
misleading.

## Decision 1 — it is a node function, not a script

It lives beside `report()` in `nodes/` and takes the same shape:

```python
preprocess_figure(store, figure_dir, config, *, run_dir, style=None, stem=None) -> dict[str, Path]
```

`report()` is the precedent: a renderer that reads a completed `ProvStore`, writes nothing back, and
can therefore be re-invoked over a finished run directory without re-running the graph
(`runs/b2ai-v2/rerender-three-subjects-2026-08-27.sbatch` does exactly that with `report`). FIGURE
reads PREPROCESS's derivatives and TAXONOMY's kind elements, so it is connected to those two nodes by
its declared inputs. It runs no model, reads no hint, and writes no element.

**It is deliberately not wired into `run_triage`.** Doing so would make every production run write a
PNG per 20 s of audio, which is a cost the owner has not asked for. Wiring it is a one-line addition
to `run.py` when that is wanted; leaving it out keeps this change from altering what a run produces.

## Decision 2 — no visualisation override of pipeline config, ever

The scratch tool wrote a `figure_tool_override.yaml` setting

```yaml
speech: {word_gap_ms: 150.0}
voice: {f0_range_hz: [75.0, 500.0]}
windows: {yamnet: {default_threshold: 0.3}, hear: {default_threshold: 0.3}, ast: {default_threshold: 0.3}}
```

and called them "visualization-only overrides". They are not visualisation parameters, and the
owner has rejected the category outright.

**Why it matters, mechanically.** `windows.<classifier>.default_threshold` feeds `_windows()` in
`preprocess.py`, whose docstring is "Fold the thresholds over one classifier's stored scores into
per-window label sets". It decides which labels become window **entities** in the store, so setting
it changes what is in the store, what TAXONOMY subsequently reads, and the config hash stamped on
every artifact. The same value also gates `_span_hear` and `_span_yamnet`, which persist only
`_confident_labels(...)` — so the per-span rasters genuinely cannot draw without it. The override
was load-bearing, not lazy. That does not save it: a figure drawn that way shows a pipeline that is
not the one production runs, while reading as if it were.

**Why `0.3` is specifically wrong.** The config's own derivation records that the six thresholds are
null "because no ROC over this corpus exists", and that v1's `taxonomy.presence_floor.{yamnet,ast,hear}
= 0.5` values were **retracted and deleted** because "they were read off bimodal gaps in one
reference recording's whole-file scores, and a whole-file gap is not a per-window threshold". A `0.3`
invented for a figure is the same error a second time, and CLAUDE.md already forbids threshold
literals outright.

**What replaces it.** Nothing. FIGURE reads the packaged config unchanged. A panel whose element is
absent prints the reason the producing node recorded — the verbatim exception text, which names the
null key — so the page says

```
ValueError: windows.yamnet.default_threshold has no value in senselab-triage/default.
It is null because nobody has measured it
```

where the raster would have been. When the thresholds are set, the panels fill with no code change.

**The split is structural.** Every drawing choice is a field of `FigureStyle`; every pipeline value
is read from `TriageConfig` and never written. `figure_test.py::TestItOverridesNoPipelineValue`
asserts that no `FigureStyle` field name collides with a pipeline key, and that the module writes no
file other than its own output.

## Decision 3 — 20 s pages, the tail padded to a uniform width

The scratch tool used `WINDOW_S = 10.0`. The owner set 20 s, and required that a final page shorter
than that be **padded out to a full 20 s**, so every image spans the same duration.

The reason is comparability: without the pad, a 1.5 s span on a 20 s page and the same span on a
3 s final page are drawn at wildly different widths, and the eye reads the second as far longer.
`pages()` therefore returns windows that are always exactly `page_seconds` wide.

The tool's previous rule is deleted, not kept alongside: `MIN_TRAILING_WINDOW_FRACTION` folded a
negligible remainder into its predecessor, which made the *last* page wider than the others — the
same comparability defect from the other direction.

**The pad is a display device and must read as one.** `_mark_padding` shades the region past the end
of the recording with a hatch, draws a dashed rule at the true end, labels it "padding — recording
ended", and the page title gains "· padded to a uniform page". Nothing else changes: the waveform is
plotted only over real samples, and no extent, count or measurement is touched.
`figure_test.py::test_padding_changes_no_span_extent` asserts the store's extents are identical
before and after rendering, and `test_padding_never_changes_which_audio_a_page_covers` asserts the
padded and ragged forms agree on every page start and on the real audio covered.

## Decision 4 — the taxonomy panel aggregates over the whole file

The owner's instruction was that the panel "highlight relevant aggregations of classifications of
the overall audio file", and that TAXONOMY be upgraded to determine a whole-file summary if none
existed. Two already existed and are read rather than reinvented:

- `<classifier>_windows` — file-scoped, carrying `labels`, `windows_by_label`, `n_windows` and the grid
- TAXONOMY's `kind` elements, written with `extent=None`, each carrying `{kind, state, lines}` with
  every line's `state`, `evidence`, `unit` and `floor`

**What was missing, and was added.** `windows_by_label` carries **counts only**, so a label firing
weakly in many windows is indistinguishable from one firing strongly in a few — and both are
unavailable at all under a null threshold. TAXONOMY now also writes
`<classifier>_label_summary` (`extent=None`), carrying per label its **peak**, **median** and
**window count**, read from the verbatim `<classifier>_scores.json` sidecar.

Two properties make this the right place for it:

1. **It reads no threshold.** `_scores()` in `preprocess.py` persists every window's raw output —
   its docstring is explicit: "no threshold is read here (V3)". So the summary is available under the
   packaged config, where every window fold is absent. This is what gives the panel real content
   today instead of a blank space.
2. **It is a measurement in the store, not a number the figure computed.** The figure prints it; it
   does not derive it. A second reader gets the same aggregation, and it carries provenance back to
   the score measurement it was derived from.

A classifier that never ran gets **no** summary rather than an empty one, so "absent" and "measured
nothing" stay distinguishable — the same distinction the seed fixture's `scores_only` mode exists to
express.

## Panels, top to bottom

| panel | source | note |
| --- | --- | --- |
| wideband spectrogram | `spectrogram_wideband.npz` | displayed, deliberately **not** continuity's input; the title says so |
| waveform + envelope + floor + `k_db` + continuity trace + rank cut | stream WAV, `energy_envelope.npz`, recomputed trace | three scales on one row |
| spans by source (E/C/A/S), clip extents | `span` entities | hatched = kept after dedup |
| YAMNet per-span labels | `span_yamnet` | absent under the packaged config, and says why |
| HeAR per-span labels | `span_hear` | same |
| SQUIM per span | `squim` assertions | |
| consensus ASR words | `word` entities | |
| whole-file taxonomy readout | `*_label_summary` + `kind` elements | text, off the shared time axis |

## Decision 5 — the continuity trace is persisted, so FIGURE recomputes nothing

Owner ruling, 2026-09-04: **the figure should not recompute anything.** The first version recomputed
the continuity trace from the persisted narrowband spectrogram, because PREPROCESS computed it inside
`_spans()` and kept it in local scope. That is deterministic, but it is not the same guarantee as
reading a sidecar: a re-render draws a trace the run's spans were not necessarily proposed against.
The owner's own rendered page hit exactly this — the trace on it was computed at draw time and agreed
with the spans only because the code had not changed in between.

PREPROCESS now writes **`continuity_trace`** as its own block, in the same shape as every other
derivative: `derivatives/continuity_trace.npz` under the key `continuity`, with a measurement
carrying `path`, `sampling_rate`, `cut_percentile` and `cut_level`. `_spans()` reads the array out of
`state` instead of computing it, so the spans and the drawn curve are the same object by
construction, and the trace survives even when span proposal later fails.

`cut_level` is recorded rather than re-derived for the same reason. It is computed by
`rank_cut_level()` in `spans/api.py`, which shares `_n_change_points()` with
`segments_between_change_points()` so the rank rule exists once. The level is an **annotation of
where the cut fell, not an equivalent test**: the cut selects by rank, so where the trace has ties
astride the boundary the level is reached by more samples than the cut marks. That is why ranking is
used in the first place — a value comparison lands on the plateau of a flat trace, where `>` admits
nothing and `>=` admits everything.

### The guard is structural

`figure_reads_only_test.py` sweeps FIGURE's imports and fails if the module imports anything that
measures — `spectral_continuity`, `hilbert_envelope_dbfs`, `extract_spectrogram_from_audios`,
`propose_spans`, `classify_audios` and the rest. Verified against the previous commit: the sweep
reports `['spectral_continuity']` there and nothing here. A data test covers the same contract from
the other side, seeding a ramp no spectral analysis of a silent stream could produce and asserting it
comes back verbatim, with the recorded `cut_level` used even where it is not the rank cut of that
ramp.

Everything else was already read, not derived. The spectrogram panel takes the persisted power array
and only converts it to dB for display; the envelope, its floor, the spans, SQUIM, the words and the
kind entities all come out of the store. The dB conversion and the raster's top-K selection are
display transforms of persisted values, not measurements.

## Decision 6 — six legibility defects, all found by looking at a rendered page

Fixed after rendering a real 79.51 s run rather than reading the code:

| defect | cause | fix |
| --- | --- | --- |
| the waveform was invisible | y-limits pinned at ±1.0 while a conditioned stream peaks near 0.05 | limits track the page's own peak, `waveform_headroom` above it, `waveform_min_amplitude` as a floor so a silent page does not magnify dither |
| the legend covered the traces it labelled | five entries drawn `loc="upper right"` over the continuity curve | the legend is gone; every scalar reading is in the panel title, and the curves are identified by their own coloured axes |
| SQUIM labels collided | the label was measured against its own marker, but at this page width neighbouring *markers* overlap on dense spans | a cell's label may use the space up to its nearest neighbour's midpoint, never more than its marker; it shrinks toward `cell_floor_fontsize` and is dropped if it will not fit. The marker is never dropped |
| two absent rasters took ~20% of the page to say one line | fixed `height_ratios` | an absent panel collapses to `absent_height_ratio` and its remaining share is redistributed to the panels with data, in proportion. The page total is unchanged, so pages stay comparable |
| the `A` (asr) row was silently empty | `preprocess.py` reads `word_gap_ms` with `config.get` and **omits the parameter when null**, so no error is raised and the source is dropped | each span row that contributed nothing names why — the null key, or the producing node's own recorded reason |
| the word lane read as three disconnected bands | 0.8-tall bars on a 1.0 pitch in a lane of height ratio 1.0 | lane ratio 1.0 → 0.62 and bar height 0.8 → `asr_row_height` 0.58. The staggering and the row assignment are untouched |

`cell_floor_fontsize` is **4.0 pt**, not lower: 3.5 pt fitted the space and was unreadable on the
rendered page. 4.0 matches the token-label floor `report.py` already uses, so a label that will not
fit at a legible size is dropped rather than shipped illegible.

The padding label is drawn **rotated inside the padded band**. A padded tail is often a fraction of a
second wide — 0.49 s on the test recording — and a horizontal label centred on it overflowed onto the
recording it exists to be distinguished from.

## Left alone deliberately

- **Not wired into `run_triage`** — see Decision 1.
- **`compare(wav, candidates)`**, the scratch tool's second mode, which monkeypatched
  `preprocess.spectral_continuity` to sweep smoothing candidates. It belongs to the continuity
  investigation that has since concluded (the rank-cut reframe retired the absolute gate), and
  monkeypatching a node's internals is not something to ship inside the package.
- **The word lane does not fit a word's text to its bar.** Text is drawn left-aligned at the bar's
  start at a fixed size, so a word longer than its own bar overruns it — visible on short words in
  running speech. `report.py`'s `_token_lane` already solves this properly, measuring each label at
  draw time against the slot its row leaves it and choosing the row count `R` from the labels' own
  rendered widths. Porting that fitter here is a real piece of work and was not in scope; the
  staggering this lane does have was left exactly as it was, per instruction.
- **The `phonation` lane** has no panel at all yet; TAXONOMY's phonation pass raises on eight null
  keys, so there would be nothing to draw.
