# Contract: Signal Table (input) — parquet

**Producer**: any (user-supplied, or built by `ranking/harvest.py` from `audio_analysis` outputs).
**Consumer**: `ranking/metric.py` (scoring), `ranking/rank.py` (ranking).
**Format**: a single parquet file (pyarrow). One row per item.

## Columns

| Column | Type | Required | Notes |
|---|---|---|---|
| `item_id` | string | yes | Unique within the table. Stable identifier for the item across versions/movement. |
| `unit` | string | yes | `"file"` or `"segment"`. Must be identical for every row in one table. |
| `source_audio` | string | segment-only | Path/id of the source recording. |
| `start` | float64 | segment-only | Segment start (seconds). |
| `end` | float64 | segment-only | Segment end (seconds). |
| `<signal_name>` | float64 | ≥1 | One column per signal. `NaN` ⇒ signal missing for that item. Examples: `audio_quality`, `asr_confidence`, `single_speaker_confidence`, `pii_presence`. |

## Schema metadata (parquet key/value)

- `schema_version`: int (string-encoded)
- `unit`: `"file"` | `"segment"`
- `signal_columns`: JSON list of the signal column names
- `produced_by`: free-text provenance (e.g. `harvest:audio_analysis@<hash>` or `user`)

## Rules

- `item_id` MUST be unique; loader raises on duplicates.
- For `unit == "segment"`, `(source_audio, start, end)` SHOULD determine `item_id`; the harvester derives a stable `item_id` from them.
- A signal column entirely `NaN` is permitted but a metric referencing it will mark items `unscorable` unless the term's `missing` policy fills it.
- No row is ever dropped downstream: missing signals propagate as item status, not omission (FR-006).
