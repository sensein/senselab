# Three report-legibility defects, measured and fixed — 2026-08-26

Each was confirmed on a rendered campaign page (`Free-speech-(v2)-1` for D-1 and D-2,
`17578482/Story-recall` for D-3). None is cosmetic: the first put a constant that is not a
measurement on a measurement panel, the second put 40+ word texts where only a lane name belongs, and
the third drew each of those texts wider than the bar it names.

## D-1 — the envelope's dB trace fabricated −240 dBFS

### What was drawn

`hilbert_envelope_dbfs` clamped the filtered envelope before taking its logarithm:

```python
env = np.maximum(filtfilt(b, a, np.abs(hilbert(x))), 1e-12)
return 20.0 * np.log10(env)
```

`filtfilt` of a rectified signal undershoots to ≤ 0 at a sharp offset, so every undershoot read
`20 * log10(1e-12) = -240.0` dBFS — the clamp's own value, not a level of the recording. Rendered,
each is a single-sample vertical spike to the axis floor.

### Measured

On a 2 s synthetic burst (0.6 amplitude, 220 Hz, 0.5 s long, sharp offset, 16 kHz), of 32 000
samples:

| quantity | value |
| --- | --- |
| samples whose filtered envelope is ≤ 0 | 328 |
| what they read under the clamp | −240.0 dBFS, all 328 |
| smallest **positive** filtered envelope | 1.797e-4 → **−74.91 dBFS** |
| envelope maximum | −3.89 dBFS |

The recording's own dB range on that fixture is −74.9 … −3.9; the clamp put a datum 165 dB below
anything measured. On the triage summary the consequence is the panel's y-scale: with the clamp the
rendered twin axis autoscaled to **−252 dBFS** (`report_test.py`
`TestTheEnvelopePanelsScaleIsTheSignals` fails on the pre-fix code with exactly that number),
squashing the informative −50 … −90 band into the top few percent of the row.

### The policy implemented

**A sample whose filtered envelope is non-positive has no dB value and reads `nan`.** Never a floor,
never a clamp. `nan` is what the toolchain already treats as "not measured": matplotlib leaves a gap
and does not autoscale to it, and every reduction below was made to skip it explicitly.

The function's contract is unchanged: one value per input sample, so the time axis stays aligned. A
gap is a `nan` in place, never a dropped sample.

### Every consumer audited

| consumer | what it does now |
| --- | --- |
| `envelope/api.py::rolling_floor_dbfs` | the percentile is taken over the window's **measured** samples (`window[np.isfinite(window)]`). A window holding none has **no floor** and reads `nan`, and so does every sample `np.interp` derives from it. No warning is emitted for an empty window because the empty case is tested for, not reduced |
| `spans/api.py::propose_spans` — peak search | runs on `above` with unmeasurable samples masked to `-inf`. NaN comparisons are all false, which **suppressed a real event**: a plateau whose right edge is unmeasurable — exactly the sharp-offset case — stopped being a local maximum and the burst vanished from the proposals. `-inf` means "not higher than", which is what an unmeasured neighbour warrants |
| `spans/api.py::propose_spans` — onset walk-back | walks the same `-inf`-masked view, so it stops at the first unmeasurable sample. The onset is where measurement stops; walking through a gap would claim an onset never measured |
| `spans/api.py::propose_spans` — offset hangover | the hangover asks whether the envelope **stayed** below threshold. A `nan` there is not evidence that it rose: pre-fix `window.max()` was `nan`, `nan <= threshold` is `False`, and the offset never closed — a scatter of unmeasurable samples 50 ms apart stretched the span **to the last sample of the recording**. The window is now the `-inf`-masked view, so unmeasurable samples cannot hold the offset open, and a window with nothing measured in it closes the span |
| `spans/api.py::propose_spans` — `NoContrast` reason | `float(above.max())` printed `nan dB` for a wholly unmeasurable envelope. The reason now reads over the finite rises, and an envelope with none says so in words: *"the envelope holds no sample measurable against its local floor"* |
| triage `PREPROCESS::_envelope` | writes both tracks to `derivatives/energy_envelope.npz` verbatim; the sidecar now carries `nan` where nothing was measured and no −240 anywhere |
| triage `REPORT::_envelope_curves` → `plot_aligned_panels` twin | decimates and hands the curves to matplotlib, which breaks the line at each `nan` and autoscales over the finite samples only. No dB **label** is derived from the envelope: the span overlay's `NN dB` labels come from `peak_over_floor_db`, a peak minus a floor of two measured samples, asserted finite |
| triage `VOICE` | records provenance use of the `energy_envelope` measurement and reads none of its values |

A single measured sample marooned inside an unmeasurable run is a local maximum of the `-inf`-masked
view and can be proposed. The existing `min_duration_ms` rule discards it, because both walks
terminate at its unmeasurable neighbours and its extent is one sample.

`speech.py::_dbfs` floors a linear amplitude "so silence is finite" and is a **different function**
over a different input; it is not a consumer of `hilbert_envelope_dbfs`. Untouched here, but it is
the same shape of fabrication and is worth a separate look.

### Consequence for the span merge rate

The honest floor changes which peaks clear the gate, because the clamp had been dragging the rolling
floor down. On the three-burst preprocess fixture (`_merging_bursts`, bursts at 1.00-1.15, 1.25-1.40,
1.50-1.65 s):

| | old (clamped) | new (nan) |
| --- | --- | --- |
| rolling floor at t = 1.4745 s | **−118.31 dBFS** | −74.98 dBFS |
| envelope at t = 1.4745 s | −38.73 dBFS | −38.73 dBFS (unchanged) |
| peaks clearing `k_db = 18` | 4, at 1.012/1.224/1.388/1.638 s | 4, at 1.012/1.261/1.475/1.638 s |
| their rises over the floor | 72.4 / **197.3** / **230.1** / 72.0 dB | 71.7 / 62.4 / 36.3 / 71.4 dB |
| `merged_proposals` on the one surviving span | 3 | 4 |

Rises of 197 dB and 230 dB are what a −118 dBFS floor produces and are not levels this fixture
contains. The post-fix set is four ordinary peaks: the three bursts and the envelope's ripple lobe in
the gap between the second and the third, all absorbed by one span. The preprocess test's expected
count moved 3 → 4 for that reason, and stays exact so a hard-coded field would still fail it.

### Owed

Nothing is owed as a threshold — `nan` is an absence, not a cut. One observation goes to
`benchmarks/open.md`: the 40 Hz envelope's own ringing produces lobes between close events that clear
the airway `k_db`, so `merged_proposals` counts filter ripple alongside events.

## D-2 — the words lane rendered every word as a y-tick

### What was drawn

The words lane was a generic `segments` panel:

```python
panels += _lane("words", _segments((word.extent, _redacted_text(...)) for word in _words(store)))
```

A `segments` panel gives each **distinct label** a y position and prints the label set as y-tick
labels. That is right for a lane of a few repeating labels — `Cough`, `unlabelled` — and wrong for a
lane of many distinct texts: on the campaign page 40+ word texts became 40+ overlapping tick labels
stacked on the left axis, while the bars themselves were unlabelled coloured dashes. The text was on
the page but not beside the time it belonged to.

### The idiom followed

`audio_analysis/plot.py`'s `asr_words` row (≈ lines 745-800): one horizontal bar per token at its own
time extent, the token's text drawn **on** the bar in a small font, `clip_on=True`, and a width below
which the text is skipped so it cannot overflow into the neighbouring token.

That idiom is now a panel type in the shared plotting module rather than a copy in triage:
`{"type": "tokens", "tokens": [{"text", "start", "end", "row"?}], "name", "fontsize"?,
"min_width_s"?}` in `plot_aligned_panels`. The y-axis carries a tick per **declared row** and none at
all when no token declares one, so the axis names the lane and never the tokens. `row` is what lets
the same panel type carry the analyze_audio shape — one stripe per (pass, model) — without triage
needing it.

`REPORT`'s job stays supplying data: `_token_lane(name, entries)` turns `(extent, text)` pairs into
the panel and draws nothing itself.

### The redaction discipline is unchanged

The text drawn per word is exactly what `_redacted_text(marks, word, scanned=scanned)` returns, the
same call the previous lane made: `[CATEGORY]` for a word the scan marked, `[unscanned]` for every
word when no complete scan stands behind the transcript, the word itself otherwise. Two tests assert
this **on the drawn artists** rather than on the panel dict: a marked word draws `[PERSON]` and never
`alice`, and an absent scan draws `[unscanned]` on every bar. The existing withheld-transcript tests
are untouched and green.

### Owed

`TOKEN_LABEL_MIN_WIDTH_S = 0.06` s is inherited from the analyze_audio row and is a **presentation
constant with no derivation**: it is absolute while the axis is not, so on a long recording drawn at
the same figure width every token clears it while none has room for its text, and on a short one the
reverse. The scale-free form compares the bar's width in points against the rendered text's own width
at the chosen point size, which needs the axes geometry at draw time. Registered in
`benchmarks/open.md`. `TOKEN_LABEL_FONTSIZE = 5.0` pt is ordinary style and gates nothing.

*Settled by D-3 below, on the same day.*

## D-3 — a label wider than its bar still overlapped its neighbour

### What was still drawn

D-2 moved the word's text onto its own bar and removed the tick pileup, but left the text's *width*
unchecked. `clip_on=True` clips to the **axes**, not to the bar, and `TOKEN_LABEL_MIN_WIDTH_S` gated
on the bar's extent **in seconds**, which says nothing about whether the text fits: at 14 in of
figure a 0.30 s bar is 24 pt of page on a 12 s recording and 9.7 pt on a 30 s one, while the label it
carries is the same width in points either way. On `17578482/Story-recall` the result at page scale
and at 300 dpi zoom was adjacent labels running into overlapping glyphs.

### Measured

DejaVu Sans, the rendered width of a label against the bar it names, in points (72 pt = 1 in of page,
so both are dpi-free):

| label | at 5.0 pt | at 4.0 pt |
| --- | --- | --- |
| `was` | 9.8 | 7.8 |
| `story` | 12.7 | 10.0 |
| `[PERSON]` | 24.3 | 19.2 |
| `grandfather` | 29.9 | 23.8 |
| `[unscanned]` | 31.6 | 24.9 |

| what carries it | bar in points |
| --- | --- |
| 0.30 s of a 12 s recording, 14 in figure | 24.2 |
| 0.30 s of a 30 s recording, 14 in figure | **9.7** |
| 0.30 s of a 60 s recording, 14 in figure | **4.8** |
| 0.15 s of the seeded 6 s triage page, 14 in figure | 21.1 |

`grandfather` needs 29.9 pt and had 9.7 pt of bar on a campaign-length page: three times its bar,
which is the pileup, and no value of a seconds threshold separates that case from the same word on a
12 s page where it also fails, or from `was` on the same page where it does not.

### The policy implemented

`_FittedTokenLabel` is a `matplotlib.text.Text` whose `draw` decides, against **the renderer that is
about to draw it**, at what size it fits — shrink toward a floor, then drop:

1. the bar's width in points, from the axes transform current for this draw, less
   `TOKEN_LABEL_PADDING_PT = 1.0` pt shared between the two ends;
2. the label's own rendered width in points at `TOKEN_LABEL_FONTSIZE = 5.0` pt. If it fits, it is
   drawn at 5.0 pt;
3. otherwise the size the width scales to, clamped up to `TOKEN_LABEL_FLOOR_FONTSIZE = 4.0` pt and
   re-measured. If it fits there, it is drawn at that size;
4. otherwise the label is not drawn at all. **The bar is always drawn.** A label is never truncated
   and never drawn partially; a bar with no text says nothing rather than something wrong.

The arithmetic is `_fitted_token_fontsize`, which takes the measurement as a callable and is unit
tested against a renderer-free text whose width is proportional to its size. The floor is a
**typographic** constant in points, not a temporal one in seconds: it is what the module is willing
to call legible, and it scales with nothing, which is the point.

What the 5.0 → 4.0 band buys, on a Story-recall-density lane (0.20-0.40 s bars, 3-11 character words):

| lane | bars | labels drawn | of those, only because of the band |
| --- | --- | --- | --- |
| 12 s at 14 in | 34 | 32 | 0 |
| 30 s at 14 in | 86 | 11 | 4 |
| 30 s at 20 in | 86 | 59 | 7 |
| 30 s at 40 in | 86 | 82 | 0 |

The band matters exactly where the page is tight and is inert where it is not, which is why the
policy is shrink-then-drop rather than drop: at 14 in a plain drop policy would place 7 labels where
this places 11.

### How the renderer is obtained

Not by asking for one. `Text.draw(renderer)` is handed the renderer of whatever draw is in progress,
so the same code serves the Agg canvas the cluster renders under (`MPLBACKEND=Agg`), `savefig` to
PNG, and `PdfPages.savefig`'s own renderer for the two-page PDF, including the second draw
`bbox_inches="tight"` performs. The decision is retaken from the full size on **every** draw, so a
figure saved as a PNG and then into the PDF cannot compound its own shrink; two tests assert the
decision is byte-identical across a second draw and across the two output paths.

One trap cost a debugging pass and is worth recording: `Text.get_window_extent` returns `Bbox.unit()`
— **one pixel wide** — for a `Text` whose visibility is off. Measuring a label that the previous
draw had hidden therefore reported that it fitted, and the lane oscillated between hidden and shown
on alternate draws. The measurement makes the label visible before asking its width.

### The redaction discipline under the fit

The fit decides how much of the lane is legible; it never decides what may be legible. The text
handed to the panel is still exactly `_redacted_text(...)`, so a marked word can only ever draw
`[PERSON]` and an unscanned transcript can only ever draw `[unscanned]`. The tests now assert the
drawn set is a **subset** of what is permitted rather than equal to it, which is the invariant that
survives a page of any width.

It follows that a placeholder can be dropped like any other label, and on the seeded page it is:
`[unscanned]` needs 24.9 pt at the floor and the page's 0.15 s bar is 21.1 pt. Nothing leaks — a
bare bar is not a transcript — and the warning still stands twice over, in the summary blocks
(`words_n` / `[unscanned]`) and in the JSON, both covered by their own tests. Special-casing the
placeholder's size by its text would put a hidden decision in the drawing code, so it is not done.

### Owed

`TOKEN_LABEL_MIN_WIDTH_S` is deleted, along with the panel's `min_width_s` key; the panel takes
`floor_fontsize` instead. What the register still carries after this is not a threshold but a
**density**: at campaign length the lane is honest and nearly wordless — 11 labels over 86 bars for
30 s at 14 in, where a 0.30 s bar is 9.7 pt and the shortest word measured is 7.8 pt at the floor.
At 60 s the same bar is 4.8 pt and nothing fits. Reading a word off the lane at that length needs
this change does not attempt: staggered rows (the panel already carries `row`), a detail lane over a
window, or a page wider than the recording is long. Registered in `benchmarks/open.md`.
