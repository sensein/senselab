# Parquet Schema Contract — comparison rows

One file per (pass, comparison_kind, task_or_pair) combination. Files live under:

```text
<run_dir>/
├── raw_16k/
│   └── comparisons/
│       ├── diarization/
│       │   └── pyannote_vs_sortformer.parquet           # within_stream
│       ├── ast/
│       │   └── ast_vs_yamnet.parquet                    # within_stream (also covers AST↔YAMNet via existing scene_agreement semantics)
│       ├── asr/
│       │   ├── whisper_vs_granite.parquet               # within_stream
│       │   ├── whisper_vs_canary_qwen.parquet
│       │   ├── whisper_vs_qwen3_asr.parquet
│       │   ├── granite_vs_canary_qwen.parquet
│       │   ├── granite_vs_qwen3_asr.parquet
│       │   └── canary_qwen_vs_qwen3_asr.parquet
│       └── cross_stream/
│           ├── asr__whisper__vs__diarization__pyannote.parquet
│           ├── asr__whisper__vs__diarization__sortformer.parquet
│           ├── ast__vs__diarization__pyannote.parquet
│           ├── yamnet__vs__diarization__pyannote.parquet
│           └── asr__whisper__vs__ppg.parquet
├── enhanced_16k/
│   └── comparisons/
│       └── (mirror of raw_16k)
└── comparisons/
    └── raw_vs_enhanced/
        ├── diarization/
        │   ├── pyannote.parquet                          # raw_vs_enhanced for one model
        │   └── sortformer.parquet
        ├── ast.parquet
        ├── yamnet.parquet
        ├── asr/
        │   ├── whisper.parquet
        │   ├── granite.parquet
        │   ├── canary_qwen.parquet
        │   └── qwen3_asr.parquet
        ├── alignment/
        │   ├── granite.parquet
        │   └── canary_qwen.parquet
        └── features/
            ├── opensmile.parquet
            ├── parselmouth.parquet
            └── torchaudio_squim.parquet
```

## Common columns (every comparison parquet)

```python
import pyarrow as pa

COMMON_SCHEMA = pa.schema([
    pa.field("start", pa.float64(), nullable=False),
    pa.field("end", pa.float64(), nullable=False),
    pa.field("comparison_kind", pa.string(), nullable=False),  # raw_vs_enhanced / within_stream / cross_stream
    pa.field("task", pa.string(), nullable=True),
    pa.field("stream_pair", pa.string(), nullable=True),
    pa.field("model_a", pa.string(), nullable=True),
    pa.field("model_b", pa.string(), nullable=True),
    pa.field("pass_a", pa.string(), nullable=False),
    pa.field("pass_b", pa.string(), nullable=False),
    pa.field("agree", pa.bool_(), nullable=True),
    pa.field("mismatch_type", pa.string(), nullable=True),
    pa.field("comparison_status", pa.string(), nullable=False),  # ok / incomparable / one_sided / unavailable
    pa.field("confidence_a", pa.float64(), nullable=True),
    pa.field("confidence_b", pa.float64(), nullable=True),
    pa.field("combined_uncertainty", pa.float64(), nullable=True),
])
```

## Per-comparator additional columns

### ASR-vs-ASR (within_stream, task="asr")

```python
ASR_VS_ASR_EXTRA = [
    pa.field("wer", pa.float64(), nullable=True),       # clipped to [0, 1] for sortability
    pa.field("a_text", pa.string(), nullable=True),
    pa.field("b_text", pa.string(), nullable=True),
    pa.field("reference_side", pa.string(), nullable=True),  # "a" or "b"
]
```

### Classification within_stream (AST vs YAMNet)

```python
CLASSIFICATION_VS_CLASSIFICATION_EXTRA = [
    pa.field("top1_a", pa.string(), nullable=True),
    pa.field("top1_b", pa.string(), nullable=True),
    pa.field("score_a", pa.float64(), nullable=True),
    pa.field("score_b", pa.float64(), nullable=True),
    pa.field("entropy_a", pa.float64(), nullable=True),
    pa.field("entropy_b", pa.float64(), nullable=True),
]
```

### Diarization within_stream

```python
DIAR_VS_DIAR_EXTRA = [
    pa.field("speaks_a", pa.bool_(), nullable=True),
    pa.field("speaks_b", pa.bool_(), nullable=True),
]
```

### Cross-stream ASR vs diarization

```python
ASR_VS_DIAR_EXTRA = [
    pa.field("asr_says_speech", pa.bool_(), nullable=True),
    pa.field("diar_says_speech", pa.bool_(), nullable=True),
]
```

### Cross-stream classification vs diarization

```python
CLASS_VS_DIAR_EXTRA = [
    pa.field("class_says_speech", pa.bool_(), nullable=True),
    pa.field("diar_says_speech", pa.bool_(), nullable=True),
    pa.field("top1_label", pa.string(), nullable=True),
]
```

### Cross-stream ASR vs PPG

```python
ASR_VS_PPG_EXTRA = [
    pa.field("asr_phonemes", pa.string(), nullable=True),
    pa.field("ppg_phonemes", pa.string(), nullable=True),
    pa.field("phoneme_per", pa.float64(), nullable=True),
    pa.field("phoneme_disagreement", pa.bool_(), nullable=True),
]
```

## Provenance metadata

Every parquet file MUST carry a Parquet-level metadata blob (key `comparator_provenance`) with:

```json
{
  "schema_version": 1,
  "wrapper_hash": "<sha256 of analyze_audio.py>",
  "senselab_version": "1.3.1a27.dev17",
  "grid": {"name": "cross_stream", "win_length": 0.2, "hop_length": 0.1},
  "upstream_cache_keys": ["..."],
  "comparator_params": {
    "uncertainty_aggregator": "min",
    "phoneme_disagreement_threshold": 0.50,
    "speech_presence_labels": ["Speech", "Conversation", "..."]
  },
  "cache_key": "<sha256 of all of the above>"
}
```

This is the same provenance shape produced by the existing per-task outputs (`features.json` etc.) so consumers can reuse the parsing helpers introduced in PR #510.
