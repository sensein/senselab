# Contract: background mask outputs

**Files**: `<run_dir>/<pass>/background_mask.parquet`,
`<run_dir>/<pass>/background_mask.json`,
`<run_dir>/<pass>/mask_introspection.json` (when `--mask-introspect`)

The mask marks regions free of **target activity** — activity from the near-microphone
participant — not regions free of speech. What counts as target activity comes from task
metadata (FR-033).

## `background_mask.parquet`

One row per bucket, same grid as the presence output (FR-031).

| Column | Type | Nullable | Notes |
|---|---|---|---|
| `region_id` | `string` | no | Contiguous same-state buckets share an id |
| `start` | `double` | no | |
| `end` | `double` | no | |
| `state` | `string` | no | `target_free` \| `target_active` \| `indeterminate` |
| `uncertainty` | `double` | no | `[0,1]` (FR-032, SC-019) |
| `guard_trimmed_s` | `double` | no | Duration removed as guard interval (FR-034) |
| `contains_nontarget_speech` | `bool` | no | Distant talker present; region stays masked (FR-033c) |
| `supports_long_window` | `bool` | no | Region long enough for an unpadded long-window decision (FR-045) |
| `target_event_types` | `list<string>` | no | From task metadata |

### Invariants

- `state` is exactly one of the three values — a binary mask is a spec violation
  (FR-032).
- `state == "target_active"` ⟹ region is excluded from background characterization.
- A bucket within `guard_interval_s` of detected target activity is never `target_free`
  (FR-034), even when no target activity is detected within the bucket itself.
- Grid matches the presence output exactly.

## `background_mask.json`

```json
{
  "task_type": "breath",
  "target_event_types": ["breath"],
  "metadata_provenance": "recognized",
  "guard_interval_s": 0.5,
  "total_masked_s": 12.4,
  "masked_fraction": 0.31,
  "is_empty": false,
  "negligible_fraction": false,
  "regions_supporting_long_window": 2,
  "regions_total": 9
}
```

### Invariants

- `metadata_provenance` is `recognized` or `fallback`; `fallback` is always recorded
  when task metadata is absent or unrecognized (FR-033b, SC-025).
- `is_empty == true` ⟹ `total_masked_s == 0.0`, and the limitation is stated rather than
  the field being omitted (FR-040, SC-022).
- `total_masked_s` and `masked_fraction` are always present (FR-038, SC-021).
- `negligible_fraction == true` flags a mask too small to support conclusions.
- `regions_supporting_long_window == 0` ⟹ only short-window results are emitted for mask
  regions, never padded long-window results presented as equivalent (FR-045, SC-032).
- For a task whose `target_event_types` excludes `speech` (breath, cough), the mask MUST
  NOT be built from speech activity alone (FR-033a) — validated by SC-024: zero target
  events appear as background sources.

## `mask_introspection.json`

```json
{
  "regions": [
    {
      "region_id": "m3",
      "start": 8.0, "end": 12.5,
      "is_noise_floor_only": false,
      "floor_db_by_band": {"500-630": -61.2, "630-800": -63.4},
      "summary_a_weighted_db": -58.1,
      "findings": ["see background-sources.md"]
    }
  ]
}
```

### Invariants

- `is_noise_floor_only == true` ⟹ `findings` is empty (nothing cleared the margin).
- `summary_a_weighted_db` is a human-readable summary only and is never used as the
  detection gate (research D5).
- `floor_db_by_band` values are relative to the recording's own reference, never dB SPL
  (FR-021c).
