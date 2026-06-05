# Contract: Metric Definition / Metric Version — JSON

**File**: `<store>/metric_versions/<version_id>.json` (immutable once written).
**Producer/Consumer**: `ranking/metric.py`, `ranking/store.py`, `ranking/recalibrate.py`.

## Metric Version JSON

```json
{
  "schema_version": 1,
  "version_id": "v2",
  "origin": "recalibrated",
  "parent_version_id": "v1",
  "created_at": "2026-06-04T17:40:00Z",
  "definition": {
    "name": "release_quality",
    "direction": "higher_is_better",
    "combine": "weighted_sum",
    "notes": "demoted PII risk after spot-check round 1",
    "terms": [
      {"signal": "audio_quality",            "weight": 0.4, "transform": "minmax", "transform_params": {"min": 0, "max": 1}, "missing": "neutral"},
      {"signal": "asr_confidence",           "weight": 0.3, "transform": "identity", "missing": "unscorable"},
      {"signal": "single_speaker_confidence","weight": 0.3, "transform": "identity", "missing": "unscorable"},
      {"signal": "pii_presence",             "weight": -0.5, "transform": "threshold", "transform_params": {"at": 0.5}, "missing": "fill:0.0"}
    ]
  },
  "recal": {
    "status": "proposed",
    "n_annotations_used": 42, "n_pairs": 318, "n_distinct_levels": 3,
    "agreement_before": 0.41, "agreement_after": 0.63,
    "message": ""
  }
}
```

## Field rules

- `version_id`: monotonic (`v1`, `v2`, …); unique in the store; never reused or rewritten (FR-018).
- `origin` ∈ `initial` | `manual` | `recalibrated`; `recal` present iff `recalibrated`.
- `direction` ∈ `higher_is_better` | `lower_is_better`.
- `terms`: ≥1; each `signal` MUST exist in the target signal table at scoring time, else scoring rejects the definition (FR-019).
- `transform` ∈ `identity|zscore|minmax|rank|clip|threshold`; `transform_params` validated per transform.
- `missing` ∈ `unscorable` | `neutral` | `fill:<float>`.
- `combine`: `weighted_sum` (only value in v1; field reserved for extension).

## Scoring semantics (`metric.py`)

1. For each term: read the signal column, apply `transform` (fit stats like zscore/minmax/rank computed over the **scorable** population of the current table), multiply by `weight`.
2. If a required signal is missing and policy is `unscorable` → item status `unscorable`, `reason="missing:<signal>"`.
3. `neutral` ⇒ term contributes 0; `fill:v` ⇒ use `v` before transform.
4. Combined score = sum of term contributions. `direction` controls sort order in `rank.py` (it does not change the stored score).
