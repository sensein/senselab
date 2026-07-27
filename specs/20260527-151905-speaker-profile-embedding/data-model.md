# Phase 1 Data Model: Speaker Profile Embedding

**Feature**: Speaker Profile Embedding for analyze_audio
**Date**: 2026-05-27

Workflow-internal types follow the existing `audio_analysis` convention: plain `@dataclass(slots=True)` for hot/internal structures, serialized to JSON via explicit encoders (see `contracts/speaker-profile.schema.md`). Entities map to the spec's Key Entities.

---

## Entity: `SpeakerProfile`

The persisted, reusable representation of one subject's target voice. Maps to spec **Speaker Profile**.

| Field | Type | Notes / Validation |
|-------|------|--------------------|
| `subject_id` | `str` | Required, non-empty. Identifies the subject (spec **Subject**). |
| `centroids` | `dict[str, list[float]]` | `{embedding_model_id → L2-normalized centroid vector}`. ≥1 entry. Vector length = model dim (e.g., 192 for ECAPA). |
| `confidence` | `Literal["ok","low","ambiguous","insufficient"]` | `ok` ≥ floor & coherent; `low` below ~20s floor; `ambiguous` near-equal top-two clusters; `insufficient` → no usable profile (centroids may be empty). |
| `aggregate_speech_seconds` | `float` | Total speech-present seconds in the dominant cluster. ≥0. Drives `confidence` vs ~20–30s policy (FR-005). |
| `dominant_cluster` | `ClusterStats` | Stats of the selected cluster (see below). |
| `runner_up_cluster` | `ClusterStats \| None` | Present when `confidence == "ambiguous"`. |
| `calibration_band` | `dict[str, tuple[float,float]]` | `{model_id → (same_speaker_floor, diff_speaker_floor)}` from `_empirical_calibration_band`; used by comparison thresholds (R6). |
| `sources` | `list[ProfileSourceFile]` | One per ingested file (kept or dropped). Auditability (FR-004, FR-016). |
| `params` | `ProfileParams` | Models, window/hop, thresholds, session preference. |
| `provenance` | `dict[str, Any]` | senselab version, schema version, cache key basis, build timestamp. |

**Lifecycle**: `insufficient` is terminal (declined). `ok`/`low`/`ambiguous` are usable; the consumer must respect `confidence` (FR-005, FR-014).

---

## Entity: `ProfileSourceFile`

Per-file record within the enrollment set. Maps to spec **Enrollment Set** (membership detail).

| Field | Type | Notes |
|-------|------|-------|
| `file_id` | `str` | Stable id (path or dataset id). Used for leave-one-file-out (FR-012). |
| `audio_signature` | `str` | sha256 of post-resample PCM (matches cache key); ties the source to cached tasks. |
| `session_id` | `str \| None` | Optional; enables `--prefer-session` weighting (FR-013). |
| `speech_seconds_used` | `float` | Speech-present seconds contributed (post-gating). |
| `windows_used` | `int` | Count of ≥~1s windows contributed. |
| `kept` | `bool` | Whether the file contributed to the dominant cluster (FR-016). |
| `drop_reason` | `str \| None` | e.g., `"insufficient_speech"`, `"outside_dominant_cluster"`, `"non_speech_task"`. |

---

## Entity: `ClusterStats`

| Field | Type | Notes |
|-------|------|-------|
| `n_windows` | `int` | Windows in the cluster. |
| `speech_seconds` | `float` | Aggregate seconds. |
| `silhouette` | `float \| None` | Best silhouette of the chosen partition (None in single-cluster regime). |
| `share` | `float` | Fraction of all clustered windows in this cluster (dominance / ambiguity signal). |

---

## Entity: `ProfileParams`

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `embedding_models` | `list[str]` | `["…ecapa…","…resnet…","microsoft/wavlm-base-plus-sv"]` | Consensus models (R3): ECAPA + ResNet (SpeechBrain, existing) + WavLM (transformers, FR-019). Configurable; single-model fallback allowed; degrades if WavLM unavailable. |
| `profile_window_s` / `profile_hop_s` | `float` | `2.0` / `1.0` | Long windows for the centroid (clean embeddings). |
| `detect_window_s` / `detect_hop_s` | `float` | `1.0` / `0.5` | Short windows for detection resolution (R4). |
| `min_confident_speech_s` | `float` | `20.0` | Floor below which `confidence="low"` (FR-005). |
| `target_confident_speech_s` | `float` | `30.0` | Target for `ok`. |
| `ambiguity_share_ratio` | `float` | `0.80` (provisional) | `ambiguous` when `runner_up_speech_s / dominant_speech_s ≥` this (≥2 clusters); validate in T028 (R11). |
| `prefer_session` | `str \| None` | `None` | Optional same-session up-weighting (FR-013). |

---

## Entity: `ProfileComparisonResult` (per analyzed recording)

Per-window scoring output. Maps to spec **Segment Comparison Result**. Emitted into the identity axis (R10).

| Field | Type | Notes |
|-------|------|-------|
| `start` / `end` | `float` | Window/bucket bounds (seconds). |
| `similarity` | `float \| None` | Consensus calibrated similarity to profile (None if N/A). |
| `other_voice_uncertainty` | `float \| None` | Calibrated identity uncertainty vs leave-one-file-out profile (R6). |
| `flag` | `Literal["target","other_voice","unavailable"]` | `unavailable` when speech-presence gate fails (FR-008). |
| `p_voice` | `float \| None` | Reused presence gate value. |
| `per_model` | `dict[str, float]` | Per-model uncertainties before consensus (audit). |

---

## Extension: profile target-quality sub-signals on the existing `quality` claim (per analyzed recording)

Per FR-010 (symmetric with the `single_speaker` extension above), the target-speaker quality rollup is **not** a new standalone object — it is added as sub-signals to `analyze_audio`'s existing per-pass `quality` claim (`global_uncertainty.by_pass[<pass>]`), alongside its current `pesq_mean` / `stoi_mean` / `sisdr_mean` / `uncertainty`. Maps to spec **Recording Quality Indicator** (R7).

| Added field (under `quality`) | Type | Notes |
|-------|------|-------|
| `profile_target_quality` | `float` | Normalized [0,1] target-capture quality (higher = cleaner). |
| `profile_target_match_fraction` | `float` | 1 − other-voice rate over speech-present duration. |
| `profile_mean_target_consistency` | `float` | Mean within-profile cosine consistency on target-matched windows. |
| `profile_squim` | `dict[str,float] \| None` | STOI/PESQ/SI-SDR restricted to target-matched windows (vs. the existing all-window means). |
| `profile_confidence` | `str` | Echo of `SpeakerProfile.confidence` (target-quality is meaningless on `insufficient`). |

The claim's headline `quality.uncertainty` additionally folds in a profile target-quality uncertainty (e.g., `1 − profile_target_quality`) when a profile is supplied, via the existing aggregation. When no profile is supplied these fields are absent and the claim is unchanged (SC-006). Computed internally as a `RecordingQualityIndicator` dataclass (types.py); its fields are written into the claim — not serialized as a standalone object.

---

## Extension: profile sub-signals on the existing `single_speaker` claim (per analyzed recording)

Per FR-020, the recording-level other-voice rollup is **not** a new standalone object — it is added as sub-signals to `analyze_audio`'s existing per-pass `single_speaker` claim (`global_uncertainty.by_pass[<pass>]` in `summary.json`), alongside its current `n_speakers` / `identity_axis_mean` / `single_speaker_uncertainty`. Decision-ready signals only; **no** PASS/REVIEW verdict.

| Added field (under `single_speaker`) | Type | Notes |
|-------|------|-------|
| `profile_other_voice_fraction` | `float` | Fraction of speech-present duration flagged `other_voice`. |
| `profile_other_voice_seconds` | `float` | Total flagged duration (seconds). |
| `profile_peak_other_voice_uncertainty` | `float` | Max per-window other-voice uncertainty. |
| `profile_p95_other_voice_uncertainty` | `float` | High-percentile (p95) uncertainty (robust to single spikes). |
| `profile_speech_present_seconds` | `float` | Denominator for the fraction (gated duration). |
| `profile_confidence` | `str` | Echo of `SpeakerProfile.confidence` (lets a downstream gate fail-safe on `low`/`ambiguous`/`insufficient`). |

The claim's headline `single_speaker.uncertainty` (currently `max(n_speakers-based, identity_axis_mean)`) additionally folds in a profile-based uncertainty when a profile is supplied, via the existing `max()`/aggregator pattern. When no profile is supplied these fields are absent and the claim is unchanged (SC-006). Computed internally as a `RecordingOtherVoiceSummary` dataclass (types.py); its fields are written into the claim — not serialized as a standalone object.

## Reused existing types (no change)

- `WindowEmbedding` (`embeddings.py`) — pooled across files for profile build.
- `UncertaintyRow` / `AxisResult` (`types.py`) — profile votes attach to the `identity` axis rows' `model_votes`.
- `Audio` (`data_structures/audio.py`) — inputs; `Audio.metadata` may carry `session_id`/task hints when the caller provides them.

## Relationships

```
Subject (1) ──< ProfileSourceFile (N)  ──aggregate──>  SpeakerProfile (1)
SpeakerProfile (1) ──used by──> analyze_audio run on each file
   → emits ProfileComparisonResult (N windows); per-recording rollups
     (RecordingOtherVoiceSummary, RecordingQualityIndicator) populate analyze_audio's
     existing single_speaker / quality claims
Leave-one-file-out: scoring recording F excludes F's ProfileSourceFile windows from the centroid
```
