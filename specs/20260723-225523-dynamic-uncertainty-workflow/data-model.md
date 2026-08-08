# Data Model: Uncertainty-driven adaptive analysis workflow

**Branch**: `20260723-225523-dynamic-uncertainty-workflow` | **Date**: 2026-07-23

All entities are Pydantic v2 models or frozen dataclasses (repo convention), serialized to parquet/JSON
under `<run_dir>/rounds/<k>/` and `<run_dir>/final/`. Existing entities (`BucketGrid` `grid.py:9`,
`UncertaintyRow` `types.py:24`, `AxisResult` `types.py:76`, `WindowEmbedding` `embeddings.py:45`) are
reused, not modified — new fields land on new entities.

## Vote (row in VoteStore)

The atom of evidence. One model's statement about one bucket on one axis.

| Field | Type | Notes |
|---|---|---|
| `axis` | str | `presence` \| `identity` \| `utterance` |
| `bucket_start`, `bucket_end` | f64 | On that axis's reporting grid |
| `source_model` | str | Model id or synthetic source (`embedding_silhouette/<m>`, `embedding_changepoint/<m>`, `adjudicator/missed_speech`) |
| `family` | str | From policy family map (FR-008) |
| `stream` | str | `raw_16k` \| `enhanced_16k` |
| `scope` | str | `file` \| `region:<region_id>` |
| `round` | int | Round that produced the vote |
| `payload` | struct | Axis-specific: presence `{speaks, native_confidence, weight, coarse, hallucinated}`; identity `{cluster_id, cosine_to_prev, ...}`; utterance `{text, phoneme_sequence, avg_logprob, alignment_ctc_score}` — same shapes the aggregators consume today |
| `shadowed_by` | str? | Vote id of the shadowing region-scope vote; shadowed votes are kept, not aggregated (D5) |
| `status` | str | `active` \| `shadowed` — nothing else. No rule removes a vote from aggregation |
| `evidence_weight` | f64 (0,1] | Floored product of every measured corroboration factor; `1.0` = *nothing measured*, never "measured as fully corroborated" |
| `provenance` | struct | cache_key, crop bounds (if region scope), intervention id, timestamp, `evidence_weight_factors: [{reason, round, corroboration, corroboration_pooling, evidence_sources, measured_on, weight_map, floor, factor, evidence_weight_after}]` (appended, never overwritten — two rules may each have something to say about one vote) |

Persisted: `rounds/<k>/belief/votes_<axis>.parquet` (append-only across rounds; each round file holds
that round's new/updated rows).

## BeliefRow (aggregated state per bucket per axis)

Extends the semantics of `UncertaintyRow` (all its columns retained for FR-024 compatibility) with:

| Field | Type | Notes |
|---|---|---|
| `round` | int | Last round that changed this row |
| `status` | str | `open` \| `converged` \| `irreducible` \| `budget_exhausted` |
| `epistemic` | f64 [0,1] | Cross-source disagreement component (D7) |
| `aleatoric_floor` | f64 [0,1] | max(quality floor, overlap posterior) (C7, D7) |
| `overlap_posterior` | f64? | From per-class segmentation posteriors (FR-016) |
| `elected_stream` | str? | From S1 (FR-015); null before election |
| `irreducible_reason` | str? | `overlapping_speech` \| `snr_floor` \| `non_speech_vocalization` \| `single_model_coverage` \| ... |
| `history` | list<struct{round, aggregated_uncertainty}> | Uncertainty trajectory for monotonicity checks |

Persisted: `rounds/<k>/belief/{presence,identity,utterance}.parquet`. The **final round's** rows are
also written to the existing paths (`<pass>/uncertainty/<axis>.parquet`) with the pre-existing column
set unchanged and new columns additive.

## Region

| Field | Type | Notes |
|---|---|---|
| `region_id` | str | `r<round>_<axis>_<idx>`, deterministic |
| `axis` | str | Proposing axis |
| `core_start`, `core_end` | f64 | Grid-quantized (D2) |
| `crop_start`, `crop_end` | f64 | Core ± pad, trough-snapped |
| `uncertainty_mass` | f64 | Σ (u − θ_low) · hop over seed+expanded buckets (ranking key, FR-010) |
| `elected_stream` | str | From S1 |
| `interventions_remaining` | int | Starts at `--max-region-rounds` |
| `status` | str | `open` \| `converged` \| `irreducible` \| `budget_exhausted` |

Persisted: `rounds/<k>/regions.json`.

## InterventionRecord (entry in iterations.json)

| Field | Type | Notes |
|---|---|---|
| `intervention_id` | str | `<round>_<rule>_<region_id>`, deterministic |
| `rule` | str | e.g. `U1_region_reasr` (contracts/interventions.md) |
| `region_id` | str? | Null for file-scoped rules (e.g. enhancement decision) |
| `trigger` | struct | Predicate inputs at decision time (reproducible from belief store, SC-008) |
| `action` | struct | Models run, stream, crop bounds, cache keys |
| `cost_class` | str | `light` \| `medium` \| `heavy` |
| `status` | str | `fired` \| `deferred_budget` \| `blocked_guard` \| `failed` |
| `error` | str? | repr(exc) when failed (D11) |
| `delta` | struct? | Per-axis mean/max uncertainty change over covered buckets, post re-aggregation |

## StreamElection

| Field | Type | Notes |
|---|---|---|
| `region_id` | str | |
| `scores` | map<stream, struct{presence_conf, quality, utterance_agreement, total}> | Weighted per policy |
| `elected` | str | |
| `guard_fired` | bool | Enhancement-artifact guard (FR-015) forced raw |
| `guard_evidence` | struct? | Raw-side PPG/ASR values that contradicted enhanced-side speech |

## RoundSummary (`rounds/<k>/summary.json`)

`round`, `regions_proposed`, `interventions {fired, deferred, blocked, failed}`, `budget_spent by
class`, `uncertainty_mass {before, after} per axis`, `buckets {converged, irreducible} delta`.

## ConvergenceReport (`final/convergence.json`)

| Field | Type | Notes |
|---|---|---|
| `run_state` | str | `converged` \| `budget_exhausted` \| `max_rounds` \| `no_speech` |
| `rounds` | list<RoundSummary> | |
| `per_axis` | map | buckets total/converged/irreducible/open; residual uncertainty mass |
| `irreducible_regions` | list<struct{region_id, axis, start, end, reason, residual, floor}> | The "explained" residual (D7) |
| `budget` | struct{caps, spent, by_class, by_rule} | Σ must equal iterations.json costs (SC US5-3) |
| `next_actions` | list | Top deferred interventions with priority — what more budget would buy (FR-018) |
| `policy_hash`, `wrapper_hash`, `senselab_version` | str | Determinism provenance (FR-025) |

## FinalWord (row in `final/transcript.json.words[]`)

| Field | Type | Notes |
|---|---|---|
| `text` | str | Winning candidate |
| `start`, `end` | f64 | From C8 consensus re-alignment when available, else weighted vote of member timestamps |
| `speaker` | str? | Unified cluster id at word midpoint; null in overlap unless v2 attribution ran |
| `confidence` | f64 [0,1] | Calibrated when profile exists; else raw weighted vote share (`calibrated` flag at document level) |
| `alternates` | list<struct{text, share, models}> | Present when winner margin < policy threshold |
| `sources` | list<str> | Contributing model ids (post family-weighting) |
| `corroboration` | f64? | Independent evidence for the winning text; `null` = unmeasured, never read as 0 |
| `member_corroboration` | map<str, f64?> | Every model with a member in the slot, winner or not — where the losing evidence stays visible |
| `flags` | list<str> | `overlap`, `single_source`, ... — `low_presence` and `hallucination_purged_nearby` are gone, replaced by the `corroboration` number |

Document level: `{calibrated: bool, policy_hash, generated_from_round, words: [...]}`.

## Relationships

```text
VoteStore ──aggregate_all──▶ BeliefRow ──propose──▶ Region ──match/rank──▶ InterventionRecord
    ▲                                                                          │
    └────────────── new votes (shadow/coexist per D5) ◀── execute ─────────────┘
BeliefRow(final) ──▶ fusion ──▶ FinalWord / final diarization / fused presence
RoundSummary* ──▶ ConvergenceReport
```
