# JSON Contract — `<run_dir>/disagreements.json`

Top-level ranked index. Built once per run by reducing across the 9 uncertainty parquets.

## Shape

```json
{
  "schema_version": 1,
  "generated_at": "2026-05-09T17:30:00Z",
  "wrapper_hash": "<sha256 of analyze_audio.py>",
  "senselab_version": "1.3.1a27.dev17",

  "config": {
    "top_n": 100,
    "aggregator": "min",
    "phoneme_disagreement_threshold": 0.50,
    "bucket_grid": {"win_length": 0.5, "hop_length": 0.5},
    "speech_presence_labels": [
      "Speech",
      "Conversation",
      "Narration, monologue",
      "Female speech, woman speaking",
      "Male speech, man speaking",
      "Child speech, kid speaking",
      "Speech synthesizer"
    ]
  },

  "models_without_native_signal": [
    "pyannote/speaker-diarization-community-1",
    "nvidia/diar_sortformer_4spk-v1",
    "ibm-granite/granite-speech-3.3-8b",
    "nvidia/canary-qwen-2.5b",
    "Qwen/Qwen3-ASR-1.7B"
  ],

  "incomparable_reasons": {
    "raw_16k/utterance/ppg": "PPG backend not provisioned",
    "enhanced_16k/utterance/granite": "no ScriptLine produced (upstream task failed)"
  },

  "totals": {
    "total_rows": 1080,
    "rows_by_axis": {"presence": 360, "identity": 360, "utterance": 360},
    "rows_by_pass": {"raw_16k": 360, "enhanced_16k": 360, "raw_vs_enhanced": 360},
    "high_uncertainty_rows": 47,
    "high_uncertainty_rate": 0.0435
  },

  "entries": [
    {
      "rank": 1,
      "axis": "utterance",
      "pass": "raw_16k",
      "start": 12.5,
      "end": 13.0,
      "aggregated_uncertainty": 0.91,
      "contributing_models": ["openai/whisper-large-v3-turbo", "ibm-granite/granite-speech-3.3-8b", "nvidia/canary-qwen-2.5b", "Qwen/Qwen3-ASR-1.7B"],
      "parquet": "raw_16k/uncertainty/utterance.parquet",
      "row_idx": 25,
      "ls_region_id": "raw_16k__uncertainty__utterance__25",
      "summary": "pairwise WER 0.83 across 4 ASR; whisper avg_logprob low"
    }
  ]
}
```

## Ranking rules

1. Primary key: `aggregated_uncertainty` desc. NaN values sort last.
2. Tiebreak 1: axis priority `utterance > identity > presence`.
3. Tiebreak 2: `start` ascending.
4. Truncate to `--disagreements-top-n` (default 100). `--disagreements-top-n 0` skips the
   index (file not written).

## Per-entry fields

| Field | Type | Description |
|---|---|---|
| `rank` | int | 1-indexed rank in the truncated index. |
| `axis` | string | One of `{"presence", "identity", "utterance"}`. |
| `pass` | string | One of `{"raw_16k", "enhanced_16k", "raw_vs_enhanced"}`. |
| `start`, `end` | float | Bucket boundaries in seconds. |
| `aggregated_uncertainty` | float | The headline scalar in `[0, 1]`. |
| `contributing_models` | list[string] | Models that voted on this bucket. |
| `parquet` | string | Path of the source parquet relative to `<run_dir>`. |
| `row_idx` | int | Row index inside the source parquet. |
| `ls_region_id` | string | Region id assigned in `labelstudio_tasks.json`. Joins to the LS bundle. |
| `summary` | string | One-line human-readable explanation of why this bucket scored high. |

## `models_without_native_signal`

Documents which models lacked a native confidence scalar so reviewers know why those
models contribute only via cross-model aggregation rather than per-bucket native signals.
This is a top-level field rather than per-entry to keep the entries terse.

## `incomparable_reasons`

Keyed by `<pass>/<axis>/<sub-signal>` (e.g. `raw_16k/utterance/ppg`). Each value is a
one-line reason the sub-signal was unavailable so the reviewer can audit
`comparison_status="incomparable"` / `"unavailable"` rows without re-running the script.

## `totals`

- `total_rows`: rows across all 9 parquets.
- `rows_by_axis` / `rows_by_pass`: per-axis / per-pass counts.
- `high_uncertainty_rows`: count of rows with `aggregated_uncertainty >= 0.66` (the LS
  `high` bin per FR-005).
- `high_uncertainty_rate`: ratio over `total_rows` (NaN when `total_rows == 0`).
