# Contract: Movement Report — JSON

**File**: `<store>/movement/<from_version>__<to_version>.json`.
**Producer**: `ranking/movement.py`. **Consumer**: CLI display / downstream review.

```json
{
  "schema_version": 1,
  "from_version": "v1",
  "to_version": "v2",
  "unit": "segment",
  "band_fraction": 0.20,
  "band_summary": {
    "entered_top": 14, "left_top": 9,
    "entered_bottom": 7, "left_bottom": 11
  },
  "added": [],
  "removed": [],
  "became_unscorable": ["rec88#3.0-4.2"],
  "entries": [
    {
      "item_id": "rec123#12.30-15.80",
      "from_rank": 41210, "to_rank": 512,
      "position_delta": 40698, "percentile_delta": 0.42,
      "from_band": "bottom", "to_band": "middle",
      "delta_kind": "moved",
      "annotated": true, "annotation_label": "poor"
    }
  ]
}
```

## Rules (FR-020–FR-023, SC-006/007)

- Covers **100%** of the union of items across the two versions; each item classified by `delta_kind` ∈ `moved` | `unchanged` | `added` | `removed` | `became_unscorable`.
- Both versions MUST share the same corpus and `unit`; mismatched unit/corpus is rejected.
- `position_delta` / `percentile_delta` null when the item is not scored in both versions.
- `band_summary` is a **coarse** count of region transitions (a lens, not an exact audited ledger); it MUST be consistent with the per-entry `from_band`/`to_band` transitions (SC-007).
- `annotated`/`annotation_label` reflect the current active annotation for highlighting (FR-022).
- No exact per-item "boundary-crossing recomputation" guarantee is made (relaxed per 2026-06-04 clarification).
