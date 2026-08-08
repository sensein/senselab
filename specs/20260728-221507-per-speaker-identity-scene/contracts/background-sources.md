# Contract: background source findings

**Files**: `<run_dir>/<pass>/background_sources.parquet`,
`<run_dir>/<pass>/noise_floor.parquet`,
`<run_dir>/<pass>/suppression.json`

## `background_sources.parquet`

One row per detected background source occurrence.

| Column | Type | Nullable | Notes |
|---|---|---|---|
| `start` / `end` | `double` | no | |
| `category` | `string` | no | `speech` \| `people` \| `machine` \| `environment` |
| `label` | `string` | no | Underlying classifier label |
| `classifier` | `string` | no | Which model produced it |
| `above_floor_db` | `double` | **no** | Required on every finding (FR-027, SC-014) |
| `tier` | `string` | no | `candidate` \| `probable` \| `confident` |
| `binding_floor` | `string` | no | `perceptual` \| `classifier` \| `recorder` (FR-022a) |
| `variant` | `string` | no | Audio variant name (FR-012) |
| `gain_db` | `double` | no | Gain applied to that variant |
| `computed_on` | `string` | no | `grid` \| `excised` (FR-044, SC-033) |
| `padding_fraction` | `double` | yes | Non-null when `computed_on == "excised"` (FR-043, SC-031) |
| `from_mask_region` | `string` | yes | Mask region id (FR-035) |
| `mask_confidence` | `double` | yes | Region confidence (FR-036) |
| `leakage_margin_db` | `double` | yes | Required for `speech`/`people` from a suppressed variant (FR-026, SC-008) |
| `suppression_depth_db` | `double` | yes | Achieved depth (FR-018a) |
| `flatness` | `double` | no | Noise-character statistic (FR-020b) |
| `modulation_depth` | `double` | yes | Orthogonal event evidence |
| `occupancy` | `double` | no | Fraction of patch frames clearing the tier (FR-021j) |
| `stationary_pass` | `bool` | no | From the unsubtracted parallel analysis (FR-021i) |
| `discounted_reason` | `string` | yes | Populated when confidence was reduced |

### Invariants

- `above_floor_db` is never null — a finding without its margin is invalid (SC-014).
- `above_floor_db < reject_below_db` ⟹ the row does not exist (FR-021).
- `tier` is consistent with `above_floor_db` and the active margin profile.
- `category in ("speech", "people")` and `variant == "foreground_suppressed"` ⟹
  `leakage_margin_db` is non-null (SC-008). A human-sound category from a suppressed
  variant is unreadable without knowing whether it may be leaked foreground.
- `computed_on == "excised"` ⟹ `padding_fraction` non-null (SC-031).
- `computed_on == "grid"` rows and `"excised"` rows are never merged or averaged
  together (FR-044, SC-033) — they are computed over different audio extents.
- `label` in the profile's `quarantined_labels` ⟹ the row exists only if the
  noise-character guard passed (FR-020c); otherwise it is suppressed entirely.
- On pure amplified noise floor, this file has **zero rows** (SC-018).
- `from_mask_region` non-null and the region is `target_free` ⟹ higher confidence than an
  otherwise-identical finding outside the mask (FR-035, SC-023).
- `mask_confidence` low ⟹ `discounted_reason` names mask uncertainty (FR-036).

## `noise_floor.parquet`

| Column | Type | Nullable | Notes |
|---|---|---|---|
| `band_low_hz` / `band_high_hz` | `double` | no | Third-octave bounds |
| `target_activity` | `string` | no | `active` \| `quiet` — conditioning stratum (FR-021h) |
| `floor_db` | `double` | no | Bias-corrected (FR-021d) |
| `quantile` | `double` | no | *q* used |
| `bias_correction_db` | `double` | no | `10·log10(1/(−ln(1−q)))` |
| `window_s` | `double` | no | |
| `iterations` | `int32` | no | Event-exclusion passes to stability (FR-021f) |
| `frozen_fraction` | `double` | no | Fraction of the band's frames where the floor was held (FR-021g) |
| `recorder_floor_db` | `double` | yes | Capture-chain self-noise estimate (FR-021b) |

### Invariants

- `bias_correction_db > 0` — an uncorrected floor is a spec violation (FR-021d). The
  repo's existing uncorrected estimator in `quality_control/metrics.py` is **not** used
  here.
- Both `active` and `quiet` strata are present when target activity is non-trivial
  (FR-021h).
- `floor_db` is relative to the recording's own reference (FR-021c), estimated only from
  the recording under analysis (FR-021a).
- A band whose `floor_db` is within a few dB of `recorder_floor_db` ⟹ findings in that
  band carry `binding_floor == "recorder"` and no perceptual claim is made (FR-021b).

## `suppression.json`

```json
{
  "requested": true,
  "model": "speechbrain/sepformer-wham16k-enhancement",
  "achieved_depth_db": 28.4,
  "depth_by_interval": [{"start": 0.0, "end": 2.0, "depth_db": 31.2}],
  "leakage_margin_db": 4.1,
  "musical_noise_risk": "moderate",
  "fallback": null
}
```

### Invariants

- `achieved_depth_db` is always reported when `requested == true` (FR-018a, SC-016), so a
  null background result is attributable to insufficient suppression rather than to the
  absence of background content.
- A finding is not claimed detectable when residual foreground exceeds it — enforced via
  `leakage_margin_db` on affected rows (FR-018a).
- Suppression unavailable or failed ⟹ `fallback` names the reason, background
  characterization continues on the standard variant, and the run does not fail
  (FR-029).
