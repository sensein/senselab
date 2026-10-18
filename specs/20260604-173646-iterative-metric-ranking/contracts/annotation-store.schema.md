# Contract: Annotation Store — JSON

**File**: `<store>/annotations.json` (atomic write-then-replace on every update).
**Producer/Consumer**: `ranking/annotate.py`; read by `evaluate.py`, `triage.py`, `recalibrate.py`, `movement.py`.

```json
{
  "schema_version": 1,
  "unit": "segment",
  "annotations": [
    {
      "item_id": "rec123#12.30-15.80",
      "label": "poor",
      "score": null,
      "reviewed_under_version": "v1",
      "reviewer": "jw",
      "created_at": "2026-06-04T18:02:11Z",
      "note": "second speaker audible 13.1-14.0s",
      "resolution": "active"
    },
    {
      "item_id": "rec123#12.30-15.80",
      "label": "acceptable",
      "score": null,
      "reviewed_under_version": "v0",
      "reviewer": "jw",
      "created_at": "2026-06-01T09:00:00Z",
      "note": "",
      "resolution": "superseded"
    }
  ]
}
```

## Rules

- Quality scale default ordinal `label` ∈ `good` | `acceptable` | `poor`; `score` (float) optional and may co-exist with `label`. At least one of `label` / `score` required.
- **At most one `active` annotation per `item_id`** (FR-013). Adding a new annotation for an existing `item_id` marks prior actives `superseded` (latest-wins, D7) — superseded entries are **retained** for history, never deleted.
- `unit` is fixed per store and must match the rankings' unit.
- Annotations are **version-independent**: every later `MetricVersion` sees the full active annotation set (FR-014). `reviewed_under_version` is provenance only.
- Ordinal→numeric mapping for correlation (`good=2, acceptable=1, poor=0`) is defined in `constants.py`, not in the file.
- **Low-sensitivity store**: this file holds only `item_id`, quality labels/scores, and short reviewer notes — never raw audio, transcripts, or extracted PII content. `item_id` and `note` are expected to be PII-free; PII enters the system only as a numeric indicator signal in the signal table.
