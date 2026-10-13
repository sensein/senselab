# How far the enhanced stream sits above the residual

The owner asked for "the dbfs difference between enhanced and residual": a signal-to-background
reading. A large difference means clean speech; a small one means the recording is
background-dominated.

It is a **reading, not a judgment**. No threshold, no gate, no outcome. If it later deserves a
gate that is VERDICT's business and the owner's call.

---

## What already existed, and why none of it was the number

PREPROCESS writes one `residual` measurement per recording
(`src/senselab/audio/workflows/triage/nodes/preprocess.py`, the residual block). It carries:

| attribute | what it is | why it is not the ask |
| --- | --- | --- |
| `peak_dbfs`, `rms_dbfs` | levels **of the residual only** | there is no enhanced counterpart |
| `energy_fraction` | `residual_energy / input_energy` | a fraction, not a level |
| `enhanced_energy_fraction` | `signal_energy / input_energy` | a fraction, not a level |
| `gain_db` | `20·log10|g|` for the fitted `plain = g·enhanced + residual` | the fit's gain, not a level difference |

The two fractions are related to the ask but are not it: they are constrained to roughly sum to
one, and their ratio is not by inspection a dB difference. That was the open question.

## The two fractions *are* the difference, exactly

From `senselab.audio.tasks.speech_enhancement.residual.compute_residual`:

```
ref_aligned, sig_aligned = align(reference, signal, lag)     # both length n
residual        = ref_aligned - gain * sig_aligned           # also length n
input_energy    = Σ ref_aligned²
signal_energy   = Σ sig_aligned²                             # NOT gain-scaled
residual_energy = Σ residual²
signal_energy_fraction   = signal_energy   / input_energy
residual_energy_fraction = residual_energy / input_energy
```

Both fractions are over the **same denominator** and both arrays have the **same length `n`**,
because everything is trimmed to the aligned overlap. So the denominator and the length both
cancel:

```
RMS difference (dB) = 20·log10( rms(enhanced) / rms(residual) )
                    = 10·log10( mean(sig²) / mean(res²) )
                    = 10·log10( signal_energy / residual_energy )
                    = 10·log10( enhanced_energy_fraction / energy_fraction )
```

This is the whole result. **The RMS difference is recoverable exactly from two attributes every
finished store already carries**, with no audio decoded and no change to the graph. The enhanced
stream's own level follows from the residual's, which is stored:

```
enhanced_rms_dbfs = residual_rms_dbfs + 10·log10( enhanced_energy_fraction / energy_fraction )
```

### Which difference to carry

Two are defensible and both are carried, because they answer different questions and cost one
column each:

- `enhanced_over_residual_rms_db` — the raw enhanced stream against the residual. This is the
  enhancer's own output level against what it discarded.
- `enhanced_over_residual_rms_fitted_db` — the same plus `gain_db`, i.e. the **fitted speech
  component** `g·enhanced` against the residual. This is the decomposition's own split of the
  input and is the one to prefer when asking "how much of this recording is speech".

### RMS, not peak

RMS is the more stable of the two for this purpose, and it is also the one that is free. The
enhanced stream's **peak** cannot be reconstructed: peak is not an energy, so no ratio of stored
energies recovers it, and only the residual's `peak_dbfs` is written. `residual_peak_dbfs` is
carried as-is; there is no `enhanced_peak_dbfs` column.

Whether peak-difference adds anything over RMS-difference is measured by the probe below, which
computes both directly from the FLACs.

## Where it belongs

Two routes, and both work:

1. **Parquet-side, available today.** `recording_vectors._residual_levels` reconstructs the
   levels from the store. This needs no replay, so it covers the **whole existing corpus** —
   which matters, because PREPROCESS is not replayed and a graph change would leave the column
   null for every recording already run.
2. **Graph-side, for the future.** A future PREPROCESS pass should compute the enhanced stream's
   own `peak_dbfs` and `rms_dbfs` in the same block that computes the residual's — the array is
   in hand there as `computation.signal_aligned` — and record the difference as its own named
   reading rather than leaving a consumer to subtract two attributes. That closes the peak gap
   and removes the reconstruction. **Not yet done**; recorded here as the intended change.

Route 1 is what shipped, for the reason in its own line: the owner can see the measure today.

## Validation

`recording_vectors_20260923/probe/probe_dbfs.py` on ORCD decodes the enhanced and residual FLACs
for a sample of the rerun corpus, computes each stream's RMS dBFS directly, and compares against
the reconstruction. It reports the absolute error distribution, the measured RMS-difference
quantiles, and the peak-difference quantiles beside them so RMS-vs-peak stability is measured
rather than assumed.

The algebra above predicts an error of zero to floating-point precision. **The probe result is
not yet in this file** — job `23546987`. A non-zero error would mean one of the two assumptions
(shared denominator, shared length) is violated in practice, and the columns would have to come
from the FLACs instead.

One hand-checked recording from the rerun corpus, as a sanity anchor:

| quantity | value |
| --- | --- |
| `enhanced_energy_fraction` | 0.996904 |
| `energy_fraction` | 7.7047e-06 |
| reconstructed difference | 51.12 dB |
| stored `rms_dbfs` (residual) | −73.66 dBFS |
| reconstructed `enhanced_rms_dbfs` | −22.54 dBFS |

## What it is not

It is not a gate, and `gates.py` gains nothing here. It is not `gain_db`, which is the fit's
scalar. It is not `energy_fraction`, which is a share of the input rather than a level. And it is
not a quality score: a recording can sit far above its residual and still be unusable for other
reasons.
