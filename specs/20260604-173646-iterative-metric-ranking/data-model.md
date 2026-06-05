# Phase 1 Data Model: Iterative Metric-Driven Ranking

Internal types are **plain dataclasses** (`@dataclass(slots=True)`), matching the `audio_analysis` / `speaker_profile` convention — they are workflow-internal and serialized explicitly to parquet/JSON, not via Pydantic. Literals constrain enumerated fields. Persisted shapes are defined in `contracts/`.

Module: `src/senselab/audio/workflows/ranking/types.py`

---

## Enumerations / Literals

```python
RankUnit = Literal["file", "segment"]
QualityLabel = Literal["good", "acceptable", "poor"]   # default ordinal scale (D-clarify)
ScoreStatus = Literal["scored", "unscorable"]          # per-item scoring outcome
Band = Literal["top", "middle", "bottom"]
MissingPolicy = Literal["unscorable", "neutral"]        # plus "fill:<value>" parsed form
SignalTransform = Literal["identity", "zscore", "minmax", "rank", "clip", "threshold"]
Direction = Literal["higher_is_better", "lower_is_better"]
RecalStatus = Literal["proposed", "refused", "warned"]
ItemDelta = Literal["moved", "unchanged", "added", "removed", "became_unscorable"]
```

---

## Core entities

### `SignalTerm`
One signal's contribution to a metric.
- `signal: str` — column name in the signal table.
- `weight: float` — combination weight.
- `transform: SignalTransform = "identity"`
- `transform_params: dict[str, Any] = {}` — e.g. `{"min": 0, "max": 1}` for `clip`, `{"at": 0.5}` for `threshold`.
- `missing: str = "unscorable"` — `MissingPolicy` value or `"fill:<float>"`.

### `MetricDefinition`
Declarative, serializable combination of signals (contract: `metric-definition.schema.md`).
- `name: str`
- `terms: list[SignalTerm]`
- `direction: Direction = "higher_is_better"`
- `combine: Literal["weighted_sum"] = "weighted_sum"`
- `notes: str = ""`
- **Validation**: ≥1 term; weights finite; every referenced `signal` must exist in the target signal table (else reject — FR-019); `fill:` value parses to float.

### `MetricVersion`
Immutable snapshot of a `MetricDefinition` plus provenance.
- `version_id: str` — e.g. `v1`, `v2` (monotonic).
- `definition: MetricDefinition`
- `origin: Literal["manual", "recalibrated", "initial"]`
- `parent_version_id: str | None` — lineage.
- `created_at: str` — ISO-8601 (timestamp passed in by caller; not generated inside pure functions).
- `recal: RecalibrationResult | None` — present when `origin == "recalibrated"`.
- `schema_version: int`

### `SignalTable` (logical; persisted as parquet, contract: `signal-table.parquet.md`)
- `unit: RankUnit`
- rows of `item_id: str` → `{signal_name: float | NaN}`; segment rows also carry `source_audio: str`, `start: float`, `end: float`.
- **Identity**: `item_id` unique within a table; for segments, `(source_audio, start, end)` resolves to a stable `item_id`.

### `RankingItem`
One row of a ranking.
- `item_id: str`
- `score: float | None` — None when `status == "unscorable"`.
- `rank: int | None` — 1-based, dense over scored items; None when unscorable.
- `percentile: float | None` — position-based, in [0, 1].
- `band: Band | None`
- `status: ScoreStatus`
- `reason: str | None` — why unscorable (e.g. missing signal name).

### `Ranking`
Ordered result for one metric version (contract: `ranking.parquet.md`).
- `version_id: str`
- `unit: RankUnit`
- `band_fraction: float` — the fraction used (default 0.20).
- `items: list[RankingItem]` — scored items in rank order, then unscorable items.
- `n_scored: int`, `n_unscorable: int`
- `provenance: dict[str, Any]` — metric definition hash, signal columns used, tie-break rule, created_at.
- `schema_version: int`
- **Invariant**: every input item appears exactly once (FR-002 / SC-002); ranks dense & unique over scored items; deterministic `(score, item_id)` order (SC-003).

### `Annotation`
A reviewer's ground-truth quality judgment for one item (contract: `annotation-store.schema.md`).
- `item_id: str`
- `label: QualityLabel | None` — ordinal (default scale).
- `score: float | None` — optional numeric judgment.
- `unit: RankUnit`
- `reviewed_under_version: str | None` — metric version shown at review time.
- `reviewer: str | None`, `created_at: str`
- `note: str = ""`
- `resolution: Literal["active", "superseded"] = "active"` — latest-wins; superseded ones retained for history.
- **Rule**: at most one `active` annotation per `item_id`; newer supersedes older (D7).

### `SeparationResult`
Output of the ranking-quality check (contract embedded in ranking provenance + returned object).
- `version_id: str`
- `rank_agreement_spearman: float | None`
- `rank_agreement_kendall_tau_b: float | None`
- `band_pairwise_agreement: float | None` — AUC-style top-vs-bottom (primary band stat).
- `band_quality_margin: float | None`
- `n_annotated: int`, `n_annotated_top: int`, `n_annotated_bottom: int`
- `evaluable: bool`, `reason: str | None`
- `meets_separation_target: bool | None` — vs configured target (default 0.80).

### `TriageThreshold`
A release-vs-review cut point with its annotation readout (contract: in `rank-cli.md` / movement docs).
- `version_id: str`
- `cut: float` — rank position or percentile (per `cut_kind`).
- `cut_kind: Literal["rank", "percentile"]`
- `n_auto_accept: int`, `n_human_review: int`
- `n_unscorable_routed: int` — unscorable items forced into `human_review` (auto-fail) regardless of the cut; included in `n_human_review`.
- `above_counts: dict[QualityLabel, int]`, `below_counts: dict[QualityLabel, int]`
- `auto_accept_poor_rate: float | None` — share of annotated-poor among annotated items above the cut.
- **Rule (FR-010b)**: unscorable items are **never** placed in the auto-accept region — they count as auto-fail and are always routed to human review, independent of where the cut falls.

### `RecalibrationResult`
- `status: RecalStatus`
- `proposed_definition: MetricDefinition | None`
- `n_annotations_used: int`, `n_pairs: int`, `n_distinct_levels: int`
- `agreement_before: float | None`, `agreement_after: float | None` — Spearman on annotated set.
- `message: str` — guard/warn explanation when refused/warned.

### `MovementReport`
Comparison of two rankings over the same corpus+unit (contract: `movement-report.schema.md`).
- `from_version: str`, `to_version: str`, `unit: RankUnit`
- `entries: list[MovementEntry]`
- `band_summary: dict[str, int]` — counts entering/leaving top & bottom bands (coarse).
- `added: list[str]`, `removed: list[str]`, `became_unscorable: list[str]`
- `schema_version: int`

### `MovementEntry`
- `item_id: str`
- `from_rank: int | None`, `to_rank: int | None`
- `position_delta: int | None`, `percentile_delta: float | None`
- `from_band: Band | None`, `to_band: Band | None`
- `delta_kind: ItemDelta`
- `annotated: bool`, `annotation_label: QualityLabel | None` — for highlight (FR-022).

---

## Relationships & lifecycle

```
SignalTable (per corpus, per unit)
   └─ scored by ─▶ MetricVersion (v1 ◀─parent── v2 ◀── …)   [immutable chain]
                      └─ produces ─▶ Ranking (1:1 per version)
                                        ├─ evaluated against ─▶ Annotations ─▶ SeparationResult
                                        └─ cut by ─▶ TriageThreshold
   Annotations (corpus-scoped, version-independent; latest-wins, history retained)
   Ranking(vA) × Ranking(vB) ─▶ MovementReport         [same corpus + unit only]
   RecalibrationResult ─uses─▶ Annotations ─proposes─▶ next MetricVersion
```

**Lifecycle (the iterative loop)**:
`initial MetricVersion → Ranking → spot-check sample → Annotations → (manual revise | recalibrate) → new MetricVersion → new Ranking → MovementReport(prev, new)` — repeated; all versions/rankings/annotations retained.

## Requirement → entity/field traceability (selected)

| Requirement | Where satisfied |
|---|---|
| FR-002 / SC-002 completeness | `Ranking` invariant; `RankingItem.status` |
| FR-003 unit per run | `Ranking.unit`, `SignalTable.unit` |
| FR-005 / SC-003 deterministic | `Ranking` `(score, item_id)` order; `provenance.tie_break` |
| FR-006 missing signals | `SignalTerm.missing`; `RankingItem.status/reason` |
| FR-008–010a separation/agreement | `SeparationResult` |
| FR-010b/c triage | `TriageThreshold` |
| FR-012 annotation scale | `Annotation.label/score` |
| FR-013/014 provenance + retention | `Annotation.*`, `resolution` |
| FR-015/016/018 versioning | `MetricVersion`, `RecalibrationResult` |
| FR-019 invalid signal ref | `MetricDefinition` validation |
| FR-020–023 movement | `MovementReport`, `MovementEntry` |
