# Contract: Ranking (output) — parquet

**File**: `<store>/rankings/<version_id>.parquet`.
**Producer**: `ranking/rank.py` → `ranking/io.py`. **Consumer**: `movement.py`, `triage.py`, `evaluate.py`, CLI display.

## Columns (rank order: scored items by rank, then unscorable items)

| Column | Type | Notes |
|---|---|---|
| `item_id` | string | From the signal table. Every input item appears exactly once (SC-002). |
| `score` | float64 | Combined metric score; `NaN` when `status="unscorable"`. |
| `rank` | int64 | 1-based, dense, unique over scored items; `-1` (or null) for unscorable. |
| `percentile` | float64 | Position-based in [0,1] over scored items; null for unscorable. |
| `band` | string | `top` / `middle` / `bottom`; null for unscorable. |
| `status` | string | `scored` / `unscorable`. |
| `reason` | string | Non-empty only when `unscorable` (e.g. `missing:asr_confidence`). |

## Schema metadata (parquet key/value)

- `schema_version`: int
- `version_id`, `unit`, `band_fraction`
- `n_scored`, `n_unscorable`
- `metric_definition_hash`: stable hash of the `MetricDefinition` (for movement / cache identity)
- `tie_break`: `"score_desc,item_id_asc"` (documented determinism rule — SC-003)
- `signal_columns`: JSON list
- `created_at`: ISO-8601

## Invariants

- Row count == input item count; `item_id` set == input `item_id` set.
- Scored ranks are a contiguous `1..n_scored` with no gaps/dupes.
- Re-running over identical inputs + version reproduces a byte-identical table (SC-003).
- Band membership derived from position: top = first `ceil(band_fraction*n_scored)`, bottom = last `ceil(band_fraction*n_scored)`, disjoint; middle is the remainder (possibly empty at small N).
