# Data Model: Scene-aware presence axis + improved utterance uncertainty

**Feature**: `20260722-175022-scene-quality-utterance` | **Date**: 2026-07-22

Entities below extend existing structures in `src/senselab/audio/workflows/audio_analysis/types.py`, `io.py`, and `src/senselab/utils/data_structures/script_line.py`. **All new fields are additive with defaults** (slots dataclasses / Pydantic), so positional construction and existing readers keep working (D10).

---

## 1. `UncertaintyRow` (extended)

`types.py` — `@dataclass(slots=True)`. Existing fields unchanged: `start, end, axis, aggregated_uncertainty, contributing_models, model_votes, comparison_status, intensity_weight, raw_aggregated_uncertainty`.

**New fields (all default `None`)** — populated only on the `presence` axis (and the utterance-only pair on the `utterance` axis):

| Field | Type | Axis | Range / values | Meaning |
|---|---|---|---|---|
| `presence_confidence` | `float \| None` | presence | `[0,1]` | calibrated mean P(speech) across voters (= `presence_p_voice`) |
| `presence_uncertainty` | `float \| None` | presence | `[0,1]` | decisiveness uncertainty `1−│2p−1│` (= existing `aggregate_presence`) |
| `quality_snr` | `float \| None` | presence | `[0,1]` degradation | 0 = clean high-SNR; 1 = fully noise-dominated |
| `quality_clip` | `float \| None` | presence | `[0,1]` degradation | proportion/severity of clipping |
| `quality_reverb` | `float \| None` | presence | `[0,1]` degradation | from Brouhaha C50 (low C50 → high degradation) |
| `quality_bandwidth` | `float \| None` | presence | `[0,1]` degradation | 0 = full-band; 1 = strongly band-limited |
| `quality_uncertainty` | `float \| None` | presence | `[0,1]` | spread among independent SNR estimators in the analysis window |
| `src_speech` | `float \| None` | presence | `[0,1]` mass | share of scene mass = target/other speech |
| `src_people` | `float \| None` | presence | `[0,1]` mass | non-speech human sounds (laughter, cough, chatter) |
| `src_machine` | `float \| None` | presence | `[0,1]` mass | engine/HVAC/tools/vehicle |
| `src_environment` | `float \| None` | presence | `[0,1]` mass | wind/rain/water/birds/etc. |
| `src_dominant` | `str \| None` | presence | one of the 4 category names | argmax of the four masses |
| `token_entropy` | `float \| None` | utterance | `≥0` | mean per-token ASR entropy over the bucket |
| `scene_quality_coupling` | `float \| None` | utterance | `≥1.0` | multiplier applied to utterance uncertainty from scene quality/source (recorded, not hidden) |

**Validation rules**:
- `src_speech + src_people + src_machine + src_environment ≈ 1.0` (±1e-6) when any source classifier ran; all `None` when both AST and YAMNet absent.
- `src_dominant == argmax(masses)`.
- All `quality_*` in `[0,1]` or `None` (null when Brouhaha/estimator unavailable, per FR-023) — never NaN masquerading as a value.
- `presence_confidence`/`presence_uncertainty` set on every presence row; `None` only if presence produced no vote at all.
- Detailed per-estimator raw values (Brouhaha dB SNR/C50, each DSP SNR, per-category class contributions) live in `model_votes` JSON, not as columns.

---

## 2. Parquet columns (`io.py::write_axis_parquet`)

Each new `UncertaintyRow` field above maps to one pyarrow column (`float64`, or `string` for `src_dominant`), appended to the existing column set. Column-projecting readers ignore unknown columns; the frozen contract already tolerates extras (`intensity_weight`, `raw_aggregated_uncertainty` precedent). Full schema in `contracts/presence-parquet-columns.md`.

---

## 3. `SoundSourceCategoryMap` (new, versioned JSON)

`workflows/audio_analysis/data/audioset_source_map.json`

```json
{
  "version": "1",
  "categories": ["speech", "people", "machine", "environment"],
  "default": "environment",
  "map": { "Speech": "speech", "Vehicle": "machine", "Wind": "environment", "Laughter": "people", "...": "..." }
}
```

- **Keys**: AudioSet display-name strings, covering the union of AST (`id2label`, 527) and YAMNet (class CSV, 521) vocab.
- **Invariant (SC-003)**: every class the classifiers can emit maps to exactly one category; unmapped → `default` with a logged warning.
- **Loaded once**, cached; consumed by `sound_sources.py`.

---

## 4. `PerAxisGrids` (config, not a new type)

Three `BucketGrid` instances threaded through `compute_uncertainty_axes`:

| Axis | win_length | hop_length | Source |
|---|---|---|---|
| presence | 0.1 s | 0.02 s | new `presence_grid` param (default) |
| utterance | 1.0 s | 0.5 s | existing `utterance_grid` param |
| identity / shared | 0.5 s | 0.5 s | existing `grid` param |
| quality analysis (internal) | 0.5 s | 0.25 s | fixed constant in `quality.py`, recorded in provenance |

Grid params recorded in each axis's `AxisResult.provenance` (FR-015).

---

## 5. `CalibrationProfile` (new, versioned JSON)

`workflows/audio_analysis/data/scene_quality_calibration.json` (or under `artifacts/` when freshly fit)

```json
{
  "version": "1",
  "snr": {"type": "linear_db_to_unit", "clean_db": 30.0, "floor_db": 0.0},
  "reverb_c50": {"type": "linear_db_to_unit", "clean_db": 30.0, "floor_db": -5.0},
  "bandwidth": {"nyquist_ref_hz": 8000.0, "rolloff_pct": 0.85},
  "temperature": {"presence": 1.0, "utterance": 1.0}
}
```

- Fitted by `scripts/calibrate_scene_quality.py` from synthetic mixtures (D9); consumed by `quality.py`/`calibration.py` to map dB→`[0,1]`.
- Absent profile → documented default normalization (uncalibrated but bounded).

---

## 6. `ScriptLine` (extended — utils)

`src/senselab/utils/data_structures/script_line.py` — add optional fields (Pydantic v2, default `None`):

| Field | Type | Meaning |
|---|---|---|
| `avg_logprob` | `float \| None` | Whisper per-chunk avg logprob (revives dead signal) |
| `no_speech_prob` | `float \| None` | Whisper no-speech probability |
| `token_entropy` | `list[float] \| float \| None` | per-token softmax entropy (or mean) |

- Populated only by the Whisper HF path (D7); `None` for all other backends → graceful degradation.
- `from_dict` extended to map these keys; existing construction unaffected.

---

## Relationships

```
compute_uncertainty_axes
  ├─ presence_grid / utterance_grid / grid ──> BucketGrid (×3)
  ├─ frame_posteriors(seg-3.0, brouhaha) ─┐
  ├─ quality.py ── brouhaha(SNR,C50)+DSP+librosa+CalibrationProfile ─┐
  ├─ sound_sources.py ── AST/YAMNet full dist × SoundSourceCategoryMap ─┤
  ├─ presence.py/aggregate.py ── voters (+frame posteriors, −coarse) ──┼─> UncertaintyRow(presence, +cols)
  └─ utterance.py/aggregate.py ── ScriptLine(token_entropy)+coupling ───┴─> UncertaintyRow(utterance, +cols)
                                                                          └─> write_axis_parquet (+cols)
```
