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

## Known limitation: the continuity trace is recomputed

Every other curve is read from a sidecar. The continuity trace is **not persisted** — PREPROCESS
computes it inside `_spans()` and keeps it only in local scope — so FIGURE recomputes it from the
persisted narrowband spectrogram under the run's own configuration. That is deterministic and uses
stored input, but it is not the same guarantee as reading a sidecar: if `spectral_continuity`
changed, a re-render of an old run would draw a trace the run's spans were not proposed against.

`open.md` already records the general form of this hazard ("A fix to a derivative producer does not
reach a completed run, and the page cannot tell"). **Persisting the trace in PREPROCESS would close
it for this panel and is a small change**, but it alters what PREPROCESS writes and so was left for
the owner rather than taken here.

## Left alone deliberately

- **Not wired into `run_triage`** — see Decision 1.
- **`compare(wav, candidates)`**, the scratch tool's second mode, which monkeypatched
  `preprocess.spectral_continuity` to sweep smoothing candidates. It belongs to the continuity
  investigation that has since concluded (the rank-cut reframe retired the absolute gate), and
  monkeypatching a node's internals is not something to ship inside the package.
- **The spectrogram panel keeps its full height when absent**, showing one line of text in a tall
  frame. Honest but wasteful; a height that responded to content would make pages differ in layout,
  which is the thing the uniform page exists to prevent.
