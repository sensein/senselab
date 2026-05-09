# JSON Contract — `<run_dir>/disagreements.json`

The top-level disagreements index. Built once per run by reducing across every comparison parquet.

## Shape

```json
{
  "schema_version": 1,
  "generated_at": "2026-05-08T17:30:00Z",
  "wrapper_hash": "<sha256 of analyze_audio.py>",
  "senselab_version": "1.3.1a27.dev17",

  "config": {
    "top_n": 100,
    "aggregator": "min",
    "phoneme_disagreement_threshold": 0.50,
    "cross_stream_grid": {"win_length": 0.2, "hop_length": 0.1},
    "speech_presence_labels": ["Speech", "Conversation", "Narration, monologue", "Female speech, woman speaking", "Male speech, man speaking", "Child speech, kid speaking", "Speech synthesizer"]
  },

  "missing_confidence_signals": {
    "models_without_native_signal": ["pyannote/speaker-diarization-community-1", "nvidia/diar_sortformer_4spk-v1", "ibm-granite/granite-speech-3.3-8b", "nvidia/canary-qwen-2.5b", "Qwen/Qwen3-ASR-1.7B"],
    "note": "These models did not expose a per-region confidence in v1; their rows have null `confidence_a`/`confidence_b`. Aggregation drops them rather than treating as zero."
  },

  "incomparable_reasons": {
    "asr_vs_ppg/raw_16k": "PPG backend not provisioned",
    "asr/granite/enhanced_16k": "no ScriptLine produced (upstream task failed)"
  },

  "totals": {
    "total_rows": 4218,
    "rows_by_kind": {"raw_vs_enhanced": 312, "within_stream": 1840, "cross_stream": 2066},
    "disagreement_rows": 187,
    "disagreement_rate": 0.0443
  },

  "entries": [
    {
      "rank": 1,
      "region_id": "raw_16k__compare__asr__whisper_vs_granite__0042",
      "pass": "raw_16k",
      "comparison_kind": "within_stream",
      "task": "asr",
      "stream_pair": null,
      "model_a": "openai/whisper-large-v3-turbo",
      "model_b": "ibm-granite/granite-speech-3.3-8b",
      "parquet_path": "raw_16k/comparisons/asr/whisper_vs_granite.parquet",
      "row_index": 42,
      "start": 12.30,
      "end": 12.50,
      "mismatch_type": "text_edit",
      "combined_uncertainty": 0.87,
      "ls_track_name": "raw_16k__compare__asr__whisper_vs_granite",
      "summary": "WER 0.6: 'four little rabbits' vs 'four white rabbits'"
    }
  ]
}
```

## Field semantics

| Field | Required | Notes |
|---|---|---|
| `schema_version` | yes | Bumped when this contract changes. |
| `generated_at` | yes | ISO-8601 UTC timestamp of when the index was built. |
| `wrapper_hash` | yes | sha256 of `scripts/analyze_audio.py` at run time — same value emitted on every other artefact. |
| `senselab_version` | yes | Output of `importlib.metadata.version("senselab")`. |
| `config` | yes | Snapshot of the comparator-relevant CLI flags so reviewers can reproduce. |
| `missing_confidence_signals.models_without_native_signal` | yes | Per FR-007 — every model that contributed to any comparison and lacked a native confidence signal. |
| `incomparable_reasons` | yes | Map of `<comparison_path>` → human-readable reason for any comparison that landed in `comparison_status=incomparable` or `unavailable`. |
| `totals` | yes | Roll-up counts. `disagreement_rate` is `disagreement_rows / total_rows`. |
| `entries` | yes | Sorted list of length ≤ `top_n`, descending by `combined_uncertainty` (NaN-last). When `top_n == 0`, this is an empty list. |
| `entries[].region_id` | yes | Stable ID matching the LS bundle's `result[].id`. |
| `entries[].parquet_path` | yes | Path relative to the run directory; opens to a parquet that contains the row at `row_index`. |
| `entries[].mismatch_type` | yes | Categorical reason — same enum as the parquet column. |
| `entries[].combined_uncertainty` | yes | NaN-or-null when no native signals were available; sort places nulls last. |
| `entries[].summary` | yes | One-line human-readable description so reviewers can triage from the index file alone. |

## Ordering rules

- Primary sort key: `combined_uncertainty` descending (NaN last, treated as least-uncertain so they don't dominate the top of the list).
- Tie-breaker 1: `mismatch_type` priority — `phoneme_disagreement` > `text_edit` > `speech_presence_flip` > `label_flip` > `boundary_shift`.
- Tie-breaker 2: `start` ascending.
- Tie-breaker 3: stable by region_id.

## Backwards compatibility

- The file lives alongside (does not replace) the existing `summary.json` and `labelstudio_*.{json,xml}` files (FR-005, SC-005).
- The existing `<pass>/scene_agreement.json` is preserved and remains the single-purpose file for the AST↔YAMNet matching-grid case; the new comparator generalizes it without removing it.
