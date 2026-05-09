# Phase 1 Data Model — Comparison & Uncertainty Stage

## ComparisonRow (parquet row, one per (start, end) bucket)

Common columns present in every comparison parquet:

| Field | Type | Nullable | Notes |
|---|---|---|---|
| `start` | float (seconds) | no | Lower bound of the time bucket (inclusive). |
| `end` | float (seconds) | no | Upper bound of the time bucket (exclusive). |
| `comparison_kind` | str | no | One of `raw_vs_enhanced` / `within_stream` / `cross_stream`. Stored once per file in metadata; mirrored on every row for joinability. |
| `task` | str | no for raw_vs_enhanced and within_stream | e.g. `diarization`, `ast`, `yamnet`, `asr`. Null for cross-stream rows where the pair spans tasks. |
| `stream_pair` | str | no for cross_stream | e.g. `asr_vs_diarization`, `ast_vs_diarization`, `asr_vs_ppg`. Null for raw_vs_enhanced and within_stream rows. |
| `model_a` | str | yes | Identifier of the "A" side. For raw_vs_enhanced this is `raw_16k` for both. |
| `model_b` | str | yes | Identifier of the "B" side. |
| `pass_a` | str | no | `raw_16k` or `enhanced_16k` (always equal to `pass_b` for within_stream and cross_stream; differs for raw_vs_enhanced). |
| `pass_b` | str | no | Same. |
| `agree` | bool | no | True iff the two sides match per the task-specific comparator. |
| `mismatch_type` | str | yes | Categorical reason when `agree` is false (e.g. `boundary_shift`, `label_flip`, `text_edit`, `speech_presence_flip`, `phoneme_disagreement`). Null when `agree` is true. |
| `comparison_status` | str | no | One of `ok` / `incomparable` / `one_sided` / `unavailable`. Drives LS label values 1:1. |
| `confidence_a` | float in [0, 1] | yes | Native confidence of the A side at this bucket. Null when no native signal exists. |
| `confidence_b` | float in [0, 1] | yes | Same for the B side. |
| `combined_uncertainty` | float in [0, 1] | yes | `1 − aggregator(confidence_a, confidence_b)`. Null only when both confidences are null. |

Per-comparator extra columns:

**ASR-vs-ASR** (within_stream where `task=="asr"`):
| Field | Type | Notes |
|---|---|---|
| `wer` | float ≥ 0 (clipped to [0, 1] for sortability) | Per-bucket word error rate computed via `jiwer.wer`. |
| `a_text` | str | Concatenated transcript tokens overlapping the bucket on side A. |
| `b_text` | str | Same on side B. |
| `reference_side` | str | `"a"` or `"b"` — which side jiwer treated as ground truth for the WER. |

**Classification (AST/YAMNet) within-stream**:
| Field | Type | Notes |
|---|---|---|
| `top1_a` | str | AudioSet display label, top-1 of side A's distribution. |
| `top1_b` | str | Same on side B. |
| `score_a` | float | Top-1 score on side A (in [0, 1]). |
| `score_b` | float | Same on side B. |
| `entropy_a` | float | Shannon entropy of side A's full distribution. |
| `entropy_b` | float | Same. |

**Diarization within-stream**:
| Field | Type | Notes |
|---|---|---|
| `speaks_a` | bool | True if side A reports any speaker present in the bucket. |
| `speaks_b` | bool | Same for side B. |

**Cross-stream ASR-vs-diarization**:
| Field | Type | Notes |
|---|---|---|
| `asr_says_speech` | bool | At least one transcript token's timestamp range overlaps the bucket (per Q4 clarification). |
| `diar_says_speech` | bool | At least one diarization segment overlaps the bucket. |

**Cross-stream classification-vs-diarization**:
| Field | Type | Notes |
|---|---|---|
| `class_says_speech` | bool | Top-1 label is in the configured speech-presence allowlist. |
| `diar_says_speech` | bool | Same as above. |

**Cross-stream ASR-vs-PPG**:
| Field | Type | Notes |
|---|---|---|
| `asr_phonemes` | str | Space-separated ARPAbet sequence derived via G2P over the bucket's transcript text. |
| `ppg_phonemes` | str | Space-separated ARPAbet sequence from the PPG argmax over the bucket. |
| `phoneme_per` | float ≥ 0 | Per-bucket phoneme error rate (jiwer-style normalized edit distance). |
| `phoneme_disagreement` | bool | True iff `phoneme_per >= --phoneme-disagreement-threshold` (default 0.50). |

## DisagreementsIndex (`<run_dir>/disagreements.json`)

```json
{
  "schema_version": 1,
  "generated_at": "ISO-8601 UTC",
  "top_n": 100,
  "aggregator": "min",
  "missing_confidence_signals": ["pyannote", "sortformer", "granite", "canary-qwen", "qwen3-asr"],
  "incomparable_reasons": {
    "asr_vs_ppg": "PPG backend not provisioned",
    "asr/granite": "no ScriptLine produced (pass=enhanced_16k)"
  },
  "entries": [
    {
      "rank": 1,
      "region_id": "raw_16k__compare__asr__whisper_vs_granite__0042",
      "pass": "raw_16k",
      "comparison_kind": "within_stream",
      "task": "asr",
      "stream_pair": null,
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

## UncertaintyAnnotation (added to existing per-task outputs, additive only)

For each existing per-task region in the cached `summary["passes"][pass][task]`:

| Field | Source | Where it lands |
|---|---|---|
| `confidence` | Native scalar in [0, 1] (see research §2). | New column in any per-task parquet/JSON under `<run_dir>/<pass>/<task>/`. |
| `uncertainty` | `1 − confidence` (or entropy for classification). | Same. |
| Backend-specific extras (`avg_logprob`, `no_speech_prob`, `entropy`, `margin_to_top2`, `mms_trellis_score`) | Native fields plumbed through. | Same. |

Constraint: existing column shapes are unchanged — these are *additions*, not replacements (FR-005 / SC-005).

## ComparisonGrid (in-memory, recorded in parquet metadata)

```python
@dataclass(frozen=True)
class ComparisonGrid:
    win_length: float           # seconds
    hop_length: float           # seconds
    name: Literal["features", "cross_stream"]   # which CLI knob produced it
```

Recorded in each parquet's `provenance` JSON column on every row so downstream tools can verify the grid:

```json
{
  "grid": {"name": "cross_stream", "win_length": 0.2, "hop_length": 0.1},
  "wrapper_hash": "...",
  "senselab_version": "1.3.1a27.dev17",
  "schema_version": 1,
  "upstream_cache_keys": ["...", "..."]
}
```

## Validation rules

- A row with `comparison_status == "ok"` MUST have non-null `agree`.
- A row with `comparison_status == "incomparable"` or `"unavailable"` MAY have null `agree` and null backend-specific extras; `mismatch_type` records the reason category.
- A row with `comparison_status == "one_sided"` MUST have one of (`a_value`/`a_text`/`top1_a`/...) populated and the other null.
- All time fields MUST satisfy `0 <= start < end`.
- For ASR-vs-ASR rows, `wer` is clipped to [0, 1] for sortability; the unclipped value lives only in the LS bundle text annotation if useful.
