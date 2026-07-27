# Contract: presence parquet — additive columns

**Writer**: `src/senselab/audio/workflows/audio_analysis/io.py::write_axis_parquet`

Existing columns (unchanged): `start, end, axis, aggregated_uncertainty, raw_aggregated_uncertainty, intensity_weight, contributing_models (list<string>), model_votes (JSON string), comparison_status`.

## New columns (appended)

| Column | pyarrow type | Axis populated | Null when |
|---|---|---|---|
| `presence_confidence` | `float64` | presence | no votes |
| `presence_uncertainty` | `float64` | presence | no votes |
| `quality_snr` | `float64` | presence | Brouhaha+DSP SNR all unavailable |
| `quality_clip` | `float64` | presence | — (cheap, always available) |
| `quality_reverb` | `float64` | presence | Brouhaha unavailable |
| `quality_bandwidth` | `float64` | presence | — |
| `quality_uncertainty` | `float64` | presence | <2 SNR estimators available |
| `src_speech` | `float64` | presence | AST+YAMNet absent |
| `src_people` | `float64` | presence | AST+YAMNet absent |
| `src_machine` | `float64` | presence | AST+YAMNet absent |
| `src_environment` | `float64` | presence | AST+YAMNet absent |
| `src_dominant` | `string` | presence | AST+YAMNet absent |
| `token_entropy` | `float64` | utterance | no token-scoring backend |
| `scene_quality_coupling` | `float64` | utterance | no quality span overlap |

## Guarantees

- On the `identity` axis and (for presence-only columns) the `utterance` axis, the columns exist but are all-null — one uniform schema across the three parquets keeps readers simple.
- **Backward compatibility (SC-008)**: existing columns keep identical names, types, order, and values; `model_votes` JSON gains new per-vote keys (`quality_*`, `src_*`, frame-posterior fields) with no schema change. Column-projecting readers (LS bundle, plot, disagreements) ignore the new columns unless updated.
- Provenance (`schema.metadata[b"comparator_provenance"]`) gains: per-axis grid params, quality analysis-window params, category-map version, calibration-profile version, and model ids/revisions (Brouhaha, segmentation-3.0) — per FR-015 and memory `project_model_revision_pinning`.

## Disagreements / LS / plot (FR-024)

- `disagreements.py`: new presence sub-signals become rankable entries (quality/source spikes surface alongside the existing axis disagreements).
- `labelstudio.py`: add `<pass>__presence__quality` and `<pass>__presence__sources` tracks (additive; existing `presence` track unchanged).
- `plot.py`: optional extra rows for quality and dominant-source strips; existing 5 rows unchanged when new signals are null.
