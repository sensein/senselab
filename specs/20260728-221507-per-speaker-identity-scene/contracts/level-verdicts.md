# Contract: amplitude-invariance verdicts

**Files**: `artifacts/level_probe/level-verdicts.json` (probe output),
`<run_dir>/level.json` (per-run variant/gain provenance)

## `level-verdicts.json`

Produced by `scripts/probe_classifier_levels.py`. Consumed by
`scripts/calibrate_detection_margin.py` as the machine half of the margin derivation
(FR-022).

```json
{
  "probe_version": "1",
  "clip": "src/tests/data_for_testing/<clip>.wav",
  "clip_sha256": "…",
  "gains_db": [-40, -20, -10, 0, 10],
  "verdicts": [
    {
      "classifier": "MIT/ast-finetuned-audioset-10-10-0.4593",
      "window_length_s": 10.24,
      "verdict": "level_sensitive",
      "gain_range_db": [-40.0, 10.0],
      "label_stability": {"-40": 0.0, "-20": 0.0, "-10": 0.0, "0": 1.0, "10": 0.0},
      "score_delta_max": {"-20": 0.2852, "10": 0.0956},
      "low_level_floor_dbfs": -55.0,
      "floor_mechanism": "fixed dataset-level affine normalization; log(float32 eps) floor",
      "floor_signature": {"Silence": 0.437, "Music": 0.350},
      "mechanism_source": "transformers/.../feature_extraction_audio_spectrogram_transformer.py:75-77,113,156",
      "notes": "mean/std are fixed AudioSet constants, not per-example statistics"
    },
    {
      "classifier": "yamnet",
      "window_length_s": 0.96,
      "verdict": "level_sensitive",
      "gain_range_db": [-40.0, 10.0],
      "low_level_floor_dbfs": -60.0,
      "floor_mechanism": "learned absolute-level-keyed Silence decision (not the log offset)",
      "floor_signature": {"Silence": 1.0},
      "mechanism_source": "tfhub yamnet/1 graph: log(mel + 0.001), no normalization op",
      "notes": "monotone and source-independent; used as the level tripwire"
    }
  ]
}
```

### Invariants

- `gain_range_db` spans **≥ 30 dB** (SC-005).
- One verdict per classifier; `window_length_s` is recorded on each so a verdict is never
  generalized across classifiers (FR-015).
- `verdict == "self_normalizing"` requires `label_stability` ≈ 1.0 at every probed gain.
  Both current classifiers measure `level_sensitive`.
- `floor_signature` non-null ⟹ that label pattern is treated as a floor response and its
  windows are discarded, and detection does **not** rely on the silence score alone
  (FR-020d) — for AST, `Silence` peaks at 0.437 while `Music` at 0.350 would clear most
  thresholds.
- `mechanism_source` is required (FR-016) so the empirical verdict can be corroborated
  against code.
- `low_level_floor_dbfs` feeds `DetectionMarginProfile.machine_basis`; the **most
  restrictive** floor across classifiers binds (FR-022a, US2 scenario 8).

### Regression guard

A test asserts each recorded `verdict` and `low_level_floor_dbfs` still hold, so a model
or dependency upgrade that changes level handling fails CI rather than silently altering
background categorization (FR-017b, US2 scenario 11). The guard must run without network
access — cached models only, skip with a clear message otherwise (constitution VI).

## `<run_dir>/level.json`

Per-run provenance for every audio variant (FR-012, SC-006).

```json
{
  "target_lufs": -23.0,
  "gain_cap_db": 10.0,
  "variants": [
    {
      "name": "unmodified",
      "gain_db": 0.0,
      "measured_lufs": -23.1,
      "lra_lu": 14.2,
      "true_peak_dbtp": -1.4,
      "clipped_fraction": 0.0,
      "requantized": false
    },
    {
      "name": "foreground_suppressed",
      "gain_db": 8.0,
      "measured_lufs": -31.0,
      "lra_lu": 9.8,
      "true_peak_dbtp": -1.1,
      "clipped_fraction": 0.0,
      "requantized": false,
      "per_segment_gain_db": [{"start": 0.0, "end": 10.0, "gain_db": 8.0}]
    }
  ]
}
```

### Invariants

- Every scene-analysis result references a `variants[].name` and its `gain_db` — 100%
  coverage, no unattributed results (SC-006).
- `gain_db <= gain_cap_db` for every variant; exceeding it is an error (FR-019).
- `clipped_fraction > 0` or `requantized == true` ⟹ reported, never silent (FR-017d,
  US2 scenario 10).
- The **same** normalization scalar applies to `unmodified` and `foreground_suppressed`
  so the cross-variant delta is not corrupted by independent renormalization (FR-019c).
- Gain is recorded **per segment** for the suppressed variant, since a single global gain
  is disallowed (FR-019a).
- `per_segment_gain_db` entries are applied before any lossy serialization in the
  classifier input path (FR-019b).
