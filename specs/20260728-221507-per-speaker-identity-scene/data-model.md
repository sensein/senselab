# Phase 1 Data Model

**Feature**: `20260728-221507-per-speaker-identity-scene` | **Date**: 2026-07-29

Entity definitions derived from the spec's Key Entities section and functional
requirements. Field types are Python-level; persisted forms are in
[contracts/](./contracts/).

Conventions used throughout:

- Times are seconds (float) relative to recording start.
- Levels are **dB relative to the recording's own reference** — never dB SPL (FR-021c).
- `None` means "not computed / not applicable", and is never zero-imputed (established
  convention of the existing uncertainty aggregators, FR-007 of the prior spec).
- Uncertainty values are in `[0, 1]`, matching the three existing axes.

---

## 1. Per-speaker identity (User Story 1)

### `SpeakerHypothesis`

One person the analysis believes is present. Run-local; no cross-recording linking.

| Field | Type | Notes |
|---|---|---|
| `speaker_id` | `str` | Run-local stable id, e.g. `S0`. Not comparable across runs |
| `existence_uncertainty` | `float` | `[0,1]` — that this speaker exists at all (FR-004) |
| `supporting_sources` | `list[str]` | Source names that proposed this speaker (FR-006) |
| `source_kinds` | `dict[str, SourceKind]` | Per source: independent vs derived (FR-007) |
| `presence_track_ref` | `str` | Key into the per-speaker presence tracks |
| `first_seen` / `last_seen` | `float` | Bounds of any supporting evidence |
| `total_active_s` | `float` | Summed duration where presence is above threshold |
| `converged` | `bool` | Whether this hypothesis stabilized (FR-011h) |
| `revisions` | `list[RevisionRecord]` | Ordered edit history (FR-011g) |

**Validation.** `existence_uncertainty` low ⟺ multiple independent sources agree.
When all sources agree on one speaker, exactly one hypothesis exists (FR-009).

### `SourceKind` (enum)

`independent` — observes speaker identity directly (a diarizer).
`derived` — labels are a by-product of another signal already in the system (e.g. a
clustering-derived pseudo-diarizer). Down-weighted in influence (FR-011c).

### `SpeakerCountPosterior`

Belief over how many speakers are present. Must represent multi-modal disagreement
(FR-002, FR-008) — a mean or majority collapse is a spec violation.

| Field | Type | Notes |
|---|---|---|
| `probabilities` | `dict[int, float]` | count → probability; sums to 1.0 |
| `support` | `dict[int, list[str]]` | count → sources that supported it (FR-006) |
| `modal_count` | `int` | Highest-probability count |
| `is_multimodal` | `bool` | True when ≥2 counts exceed a policy threshold |
| `converged` | `bool` | |

**Validation.** Zero-speech recording ⟹ mass concentrated on `0`. All-sources-agree
⟹ ≥0.9 mass on one count (SC-001).

### `PerSpeakerPresenceTrack`

Time-aligned belief that one specific speaker was talking. Same grid as the existing
presence output (FR-003).

| Field | Type | Notes |
|---|---|---|
| `speaker_id` | `str` | |
| `start` / `end` | `float` | Bucket bounds |
| `presence_confidence` | `float \| None` | `[0,1]` that *this* speaker is active here |
| `presence_uncertainty` | `float \| None` | `[0,1]` — distinct from existence (FR-004) |
| `overlap_with` | `list[str]` | Other speaker ids concurrently active |
| `contributing_sources` | `list[str]` | |

**Validation.** Covers full recording duration for every hypothesis (SC-003).
Concurrent speakers may both be present (overlap edge case).

### `SourceLabelCorrespondence`

Maps each source's own labels to fused hypotheses, making cross-source fusion auditable
(FR-005).

| Field | Type | Notes |
|---|---|---|
| `source` | `str` | e.g. a diarizer name |
| `source_label` | `str` | That source's own label, e.g. `SPEAKER_00` |
| `speaker_id` | `str` | Fused hypothesis it maps to |
| `cluster_id` | `str \| None` | Embedding cluster that mediated the mapping |
| `source_kind` | `SourceKind` | |
| `confidence` | `float \| None` | Mapping confidence |

---

## 2. Mutual influence and convergence (spans all stories)

### `InfluenceWeight`

How much one signal may move another (FR-011b, FR-011c).

| Field | Type | Notes |
|---|---|---|
| `signal` | `str` | Emitting signal |
| `target` | `str` | Quantity influenced |
| `base_weight` | `float` | From policy |
| `uncertainty_gate` | `float` | `[0,1]` multiplier derived from the signal's own uncertainty |
| `derivation_gate` | `float` | `[0,1]` multiplier; < 1 for `derived` sources |
| `effective_weight` | `float` | Product; the value actually applied |

**Validation.** A `derived` signal alone cannot drive a revision an `independent` signal
contradicts (SC-030).

### `RevisionRecord`

Every state change is attributable (FR-011g).

| Field | Type | Notes |
|---|---|---|
| `round` | `int` | Loop round that made the change |
| `quantity` | `str` | What changed |
| `before` / `after` | `Any` | Values |
| `caused_by` | `str` | Signal that caused it |
| `effective_weight` | `float` | Weight applied |
| `evidence` | `dict[str, Any]` | Supporting evidence snapshot |
| `resolution_kind` | `ResolutionKind` | **The self-confirmation guard** (FR-011d) |

### `ResolutionKind` (enum)

`new_evidence` — uncertainty fell because independent evidence arrived.
`revision` — uncertainty fell because the value was overwritten. **Must not be reported
as improved confidence** (FR-011d, SC-027).
`unresolved` — still uncertain.

### `ConvergenceReport`

| Field | Type | Notes |
|---|---|---|
| `converged` | `bool` | |
| `rounds_run` | `int` | |
| `termination_reason` | `str` | `converged` \| `oscillation` \| `no_improvement` \| `budget` |
| `oscillation_states` | `list[Any]` | Populated when alternation detected (FR-011e) |
| `unresolved_quantities` | `list[str]` | Never presented as settled (FR-011h) |
| `per_quantity` | `dict[str, ResolutionKind]` | |

---

## 3. Level and amplitude (User Story 2)

### `AmplitudeInvarianceVerdict`

Per classifier, from the gain sweep (FR-014, FR-015).

| Field | Type | Notes |
|---|---|---|
| `classifier` | `str` | |
| `window_length_s` | `float` | Attribution requirement (FR-015) |
| `verdict` | `"self_normalizing" \| "level_sensitive"` | Measured, both currently the latter |
| `gain_range_db` | `tuple[float, float]` | Range over which the verdict holds (≥30 dB, SC-005) |
| `label_stability` | `dict[float, float]` | gain → top-k list agreement with unity |
| `score_delta_max` | `dict[float, float]` | gain → max abs score change |
| `low_level_floor_dbfs` | `float` | Level beneath which nothing is reported (FR-017a) |
| `floor_mechanism` | `str` | e.g. learned silence decision vs arithmetic log floor |
| `floor_signature` | `dict[str, float] \| None` | Fixed label pattern on digital silence (FR-020d) |
| `mechanism_source` | `str` | Code location corroborating the empirical verdict (FR-016) |

### `AudioVariant`

A named version of the recording that stages consume (FR-012).

| Field | Type | Notes |
|---|---|---|
| `name` | `"unmodified" \| "speech_enhanced" \| "foreground_suppressed"` | |
| `gain_db` | `float` | Applied gain; 0.0 for unmodified |
| `target_lufs` | `float \| None` | Normalization target if applied |
| `true_peak_dbtp` | `float` | Must be ≤ policy ceiling |
| `clipped_fraction` | `float` | Detected clipping (FR-017d) |
| `requantized` | `bool` | Whether a lossy serialization occurred (FR-019b) |
| `segment_rms_dbfs` | `dict[str, float]` | Per-segment pre-gain level, for FR-020a rejection |

**Validation.** Every scene-analysis result references exactly one variant and its gain
(SC-006). Gain never exceeds the policy cap.

---

## 4. Noise floor and background sources (User Story 3)

### `NoiseFloorEstimate`

| Field | Type | Notes |
|---|---|---|
| `band_hz` | `tuple[float, float]` | Third-octave band bounds |
| `floor_db` | `float` | Bias-corrected (FR-021d) |
| `quantile` | `float` | *q* used |
| `bias_correction_db` | `float` | `10·log10(1/(−ln(1−q)))` |
| `window_s` | `float` | Estimation window |
| `iterations` | `int` | Event-exclusion passes to stability (FR-021f) |
| `target_activity` | `"active" \| "quiet"` | Conditioning stratum (FR-021h) |
| `frozen_intervals` | `list[tuple[float,float]]` | Where the floor was held (FR-021g) |
| `recorder_floor_db` | `float \| None` | Capture-chain self-noise estimate (FR-021b) |
| `binding_floor` | `"perceptual" \| "classifier" \| "recorder"` | Which limit binds (FR-022a) |

**Validation.** Estimated from the same recording only (FR-021a). Must not absorb a
stationary source — FR-021i's parallel unsubtracted analysis is the check.

### `DetectionMarginProfile`

Versioned, configurable policy (FR-023) with a written derivation (FR-022).

| Field | Type | Notes |
|---|---|---|
| `profile_version` | `str` | |
| `candidate_db` / `probable_db` / `confident_db` | `float` | Defaults 3.0 / 6.0 / 10.0 |
| `reject_below_db` | `float` | Default 3.0 |
| `target_lufs` | `float` | Default −23.0 |
| `gain_cap_db` | `float` | Default 10.0 |
| `reject_below_pregain_dbfs` | `float` | Default −45.0 |
| `min_occupancy` | `float` | Fraction of patch frames clearing the tier |
| `min_duration_s` | `dict[str, float]` | Class-wise (FR-021j) |
| `flatness_max` | `float` | Noise-character guard (FR-020b) |
| `quarantined_labels` | `list[str]` | FR-020c |
| `derivation` | `DerivationRecord` | |

### `DerivationRecord`

| Field | Type | Notes |
|---|---|---|
| `human_basis` | `list[Citation]` | Psychophysical support |
| `machine_basis` | `list[Citation]` | Measured classifier capability |
| `agreement_note` | `str` | How the two reconcile (SC-017) |
| `verification_status` | `dict[str, "verified" \| "provisional"]` | Per figure (FR-022) |

### `BackgroundSourceFinding`

| Field | Type | Notes |
|---|---|---|
| `start` / `end` | `float` | |
| `category` | `"speech" \| "people" \| "machine" \| "environment"` | |
| `label` | `str` | Underlying classifier label |
| `above_floor_db` | `float` | **Required on every finding** (FR-027, SC-014) |
| `tier` | `"candidate" \| "probable" \| "confident"` | |
| `binding_floor` | `str` | Which limit was closest (FR-022a) |
| `variant` | `str` | Audio variant name |
| `gain_db` | `float` | |
| `from_mask_region` | `str \| None` | Mask region id, if any (FR-035) |
| `mask_confidence` | `float \| None` | Region confidence (FR-036) |
| `leakage_margin_db` | `float \| None` | For `speech`/`people` from a suppressed variant (FR-026, SC-008) |
| `suppression_depth_db` | `float \| None` | Achieved depth (FR-018a) |
| `flatness` | `float` | Noise-character statistic |
| `modulation_depth` | `float \| None` | Orthogonal event evidence |
| `computed_on` | `"grid" \| "excised"` | Never conflated (FR-044, SC-033) |
| `padding_fraction` | `float \| None` | For excised segments (FR-043, SC-031) |
| `stationary_pass` | `bool` | From the unsubtracted parallel analysis (FR-021i) |

**Validation.** A `speech` or `people` finding from a suppressed variant without
`leakage_margin_db` is invalid (SC-008). Pure noise floor yields zero findings (SC-018).

---

## 5. Background mask (User Story 4)

### `BackgroundMaskRegion`

| Field | Type | Notes |
|---|---|---|
| `region_id` | `str` | |
| `start` / `end` | `float` | Same grid as presence output (FR-031) |
| `state` | `"target_free" \| "target_active" \| "indeterminate"` | Three states (FR-032) |
| `uncertainty` | `float` | `[0,1]` (FR-032, SC-019) |
| `guard_trimmed_s` | `float` | Duration removed as guard interval (FR-034) |
| `contains_nontarget_speech` | `bool` | Distant talker retained (FR-033c) |
| `supports_long_window` | `bool` | Long enough for an unpadded decision (FR-045) |
| `usable_for` | `list[str]` | Classifiers that can decide here |

### `BackgroundMask`

| Field | Type | Notes |
|---|---|---|
| `regions` | `list[BackgroundMaskRegion]` | |
| `total_masked_s` | `float` | FR-038, SC-021 |
| `masked_fraction` | `float` | Flagged when negligible |
| `is_empty` | `bool` | Continuous-target-activity case (FR-040, SC-022) |
| `task_type` | `str \| None` | From task metadata |
| `target_event_types` | `list[str]` | e.g. `["speech"]`, `["breath"]`, `["cough"]` (FR-033) |
| `metadata_provenance` | `"recognized" \| "fallback"` | FR-033b, SC-025 |
| `guard_interval_s` | `float` | Policy value applied |

**Validation.** For a breathing/cough task, zero target events appear as background
sources (SC-024). Mask built via fallback is always distinguishable (SC-025).

### `MaskedRegionIntrospection`

| Field | Type | Notes |
|---|---|---|
| `region_id` | `str` | |
| `findings` | `list[BackgroundSourceFinding]` | |
| `is_noise_floor_only` | `bool` | Nothing cleared the margin (FR-037) |
| `floor_db_by_band` | `dict[str, float]` | |
| `summary_a_weighted_db` | `float \| None` | Human-readable summary only, never the gate |

---

## Entity relationships

```text
SpeakerCountPosterior ──1:n── SpeakerHypothesis ──1:1── PerSpeakerPresenceTrack
                                    │
                        SourceLabelCorrespondence (n per hypothesis)
                                    │
                              RevisionRecord ── ResolutionKind
                                    │
                            ConvergenceReport (run-level)
                                    │
                              InfluenceWeight (signal → quantity)

AudioVariant ──1:n── BackgroundSourceFinding ──n:1── BackgroundMaskRegion
                                │                            │
                     DetectionMarginProfile          BackgroundMask ── task metadata
                                │                            │
                       NoiseFloorEstimate          MaskedRegionIntrospection
                        (per band, per stratum)

AmplitudeInvarianceVerdict ── informs ──> DetectionMarginProfile.machine_basis
```

## State transitions

**Speaker hypothesis lifecycle** — `proposed` → `corroborated` (a second independent
source agrees) → `converged`, or → `revised` (an influence path changed it; a
`RevisionRecord` is appended) → back to `corroborated`/`converged`, or → `unresolved`
if the loop terminates without stabilizing.

**Mask region lifecycle** — `indeterminate` → `target_free` or `target_active` as
evidence accumulates. Guard-interval trimming may shrink a `target_free` region or
eliminate it. A region may be revised by influence like any other quantity.

**Finding lifecycle** — `rejected` (below margin) | `candidate` → `probable` →
`confident` as margin evidence strengthens. Never advances on gain alone, since gain
changes no SNR (D1).
