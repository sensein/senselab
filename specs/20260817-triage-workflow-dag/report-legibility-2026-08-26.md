# Two report-legibility defects, measured and fixed — 2026-08-26

Both were confirmed on a rendered campaign page (`Free-speech-(v2)-1`). Neither is cosmetic: one put
a constant that is not a measurement on a measurement panel, the other put 40+ word texts where only
a lane name belongs.

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
