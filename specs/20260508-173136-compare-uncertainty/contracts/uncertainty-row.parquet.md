# Contract — `uncertainty-row.parquet`

One file per `(pass, axis)` and per `(raw_vs_enhanced, axis)` combination. Per default
two-pass run: 9 parquets total (3 axes × 2 passes + 3 deltas).

## Filesystem layout

```text
<run_dir>/
├── <pass>/                    # one of {"raw_16k", "enhanced_16k"}
│   └── uncertainty/
│       ├── presence.parquet
│       ├── identity.parquet
│       └── utterance.parquet
└── uncertainty/
    └── raw_vs_enhanced/
        ├── presence.parquet
        ├── identity.parquet
        └── utterance.parquet
```

## Columns (canonical order)

| Column | Type | Description |
|---|---|---|
| `start` | `float64` | Bucket start time in seconds (`>= 0`). |
| `end` | `float64` | Bucket end time in seconds (`> start`). |
| `axis` | `string` | One of `{"presence", "identity", "utterance"}` — duplicated on every row for join convenience. |
| `aggregated_uncertainty` | `float64` | The headline scalar in `[0, 1]`. NaN only when `comparison_status != "ok"`. |
| `contributing_models` | `list<string>` | Model ids of every contributor with a non-null vote on this bucket. Empty iff `comparison_status` ∈ {"unavailable"} (in which case the row is typically dropped per FR-012, but kept for "incomparable" / "one_sided" buckets). |
| `model_votes` | `map<string, struct>` | One entry per model_id in `contributing_models`. The struct shape is axis-specific — see "Per-axis vote shape" below. |
| `comparison_status` | `string` | One of `{"ok", "incomparable", "unavailable", "one_sided"}`. |

### Per-axis vote shape

For **presence**:

```jsonc
{
  "model_id": "pyannote/speaker-diarization-community-1",
  "speaks": true,
  "native_confidence": 0.92      // optional; null when the model exposes no scalar
}
```

For **identity**:

```jsonc
{
  "model_id": "pyannote/...",
  "speaker_label": "SPEAKER_00",  // present for diar models
  "embedding_cosine_to_prev": 0.07,  // present for ECAPA / ResNet entries
  "raw_vs_enh_disagrees": true       // present only on raw_vs_enhanced/identity.parquet
}
```

For **utterance**:

```jsonc
{
  "model_id": "openai/whisper-large-v3-turbo",
  "text": "hello world",
  "avg_logprob": -0.18,      // present only when the model exposes it (Whisper)
  "phoneme_per_to_ppg": 0.42 // present only on ASR rows when PPG is provisioned
}
```

The struct columns inside the map use Arrow's nullable struct fields — fields not applicable
to a given model_id are stored as null rather than omitted, so the parquet schema remains
identical across rows.

## Per-axis row-emission rules

- **presence** — emit a row when at least one contributing model has `speaks ≠ null`. Buckets
  where every model dropped out (no diar segment overlap, no ASR token, no AST/YAMNet window
  available) emit no row.
- **identity** — emit a row when at least one contributing model has a usable speaker label
  OR an across-time cosine value. Pure-silence buckets (no diar speaker label from any model)
  emit no row.
- **utterance** — emit a row when at least one contributing ASR model has non-empty text on
  the bucket OR the PPG sub-signal is present.

## `aggregated_uncertainty` computation

Per FR-002 (per-axis rules) and FR-004 (`--uncertainty-aggregator`):

- **presence**: `H_normalized = -Σ p_i log p_i / log(n)` where `p_i` = fraction of votes
  taking the `i`-th value (over `{true, false}` for the binary "speaks?" question), `n` =
  number of contributing models. Edge case `n == 1` → `0.0` (no disagreement possible);
  edge case `n == 0` → no row.
- **identity**: per FR-002 the three sub-signals are `cross_model_label_disagreement`,
  `raw_vs_enh_label_disagreement` (raw_vs_enhanced parquet only), `mean_across_time_cosine_distance`.
  Each sub-signal contributes a value in `[0, 1]` or null; null entries are dropped before
  applying `--uncertainty-aggregator`.
- **utterance**: sub-signals are `asr_pairwise_mean_wer`, `whisper_native_uncertainty`,
  `mean_ppg_per`. Same null-drop + `--uncertainty-aggregator` rule.

## `comparison_status`

| Value | Meaning |
|---|---|
| `ok` | At least one contributing model produced a usable signal; aggregator ran cleanly. |
| `incomparable` | Per-axis precondition violated (e.g. all transcripts empty on utterance, all diar models silent on identity); `aggregated_uncertainty` is `NaN`. |
| `unavailable` | A required upstream backend was not provisioned (e.g. PPG missing for the PPG-only utterance sub-signal in a single-model run). |
| `one_sided` | (raw_vs_enhanced parquets only) the same model produced output on only one of the two passes. |

## Provenance metadata (parquet `schema.metadata`)

```jsonc
"comparator_provenance": {
  "schema_version": 1,
  "wrapper_hash": "<sha256 of analyze_audio.py>",
  "senselab_version": "...",
  "axis": "presence",            // | "identity" | "utterance"
  "pass": "raw_16k",             // | "enhanced_16k" | "raw_vs_enhanced"
  "grid": {"win_length": 0.5, "hop_length": 0.5},
  "contributing_model_set": ["pyannote/...", "nvidia/diar_sortformer_4spk-v1", ...],
  "comparator_params": {
    "uncertainty_aggregator": "min",
    "phoneme_disagreement_threshold": 0.50,
    "speech_presence_labels": ["Speech", "Conversation", ...],
    "disagreements_top_n": 100
  },
  "cache_key": "<sha256>"
}
```

## Reproducibility

A reviewer who reruns `analyze_audio.py` with identical inputs must see byte-identical
parquet files (modulo timestamps in the provenance), per FR-014. The cache key in the
provenance lets a reviewer diff two runs of the same parquet without parsing the data
column-by-column.
