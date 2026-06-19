# Phase 0 Research: Iterative Metric-Driven Ranking

All Technical-Context unknowns were either resolved during `/speckit.clarify` (band framing, primary quality measure + triage purpose, annotation scale, corpus scale) or are method choices resolved below. No `NEEDS CLARIFICATION` markers remain.

---

## D1. Rank-agreement measure (primary ranking-quality indicator)

**Decision**: Report **Spearman's ρ** as the headline rank-agreement measure between the metric order and the available quality annotations, with **Kendall's τ-b** (tie-aware) reported alongside. Compute via `scipy.stats.spearmanr` / `kendalltau` (already present transitively through `scikit-learn`), falling back to a numpy rank-then-Pearson implementation if `scipy` import fails.

**Rationale**: The spec explicitly wants ordinal trend, not exact neighbor order, and annotations are ordinal (good/acceptable/poor) with many ties — τ-b handles ties correctly, while ρ is the familiar headline. Both are scale-free and robust to imperfect metrics. Using the annotated subset only is correct because annotations are the ground truth.

**Alternatives considered**:
- *Pearson on scores vs. label codes* — rejected: assumes linear, interval-scaled labels; ordinal labels violate that.
- *AUC / Mann–Whitney (good-vs-poor separability)* — kept but demoted: it answers "can the metric tell good from poor?" which is exactly the **secondary band-separation** check (D2), not the full-ranking trend.
- *NDCG* — rejected: assumes graded relevance with position discounting (a search-results frame); overkill and harder to explain to a non-technical stakeholder.

---

## D2. Top/bottom-band separation (secondary, coarse check)

**Decision**: Band fraction defaults to **0.20** (configurable), surfaced in `constants.py` as `DEFAULT_BAND_FRACTION`. The separation check reports, over annotated items, whether top-band quality exceeds bottom-band quality, expressed as (a) the **pairwise-agreement rate** (probability a random annotated top-band item outranks a random annotated bottom-band item in quality — i.e. an AUC-style statistic) and (b) the **mean-quality margin**. Target per SC-001: ≥ 0.80 pairwise agreement.

**Rationale**: Matches the clarified "bands are a coarse lens" decision and SC-001 phrasing directly. The pairwise-agreement statistic is exactly what a human reviewer doing top-vs-bottom comparisons would measure, making the success criterion verifiable.

**Evaluability guard**: If the corpus is too small for the band fraction to yield ≥1 item per band, or there are too few annotated items in either band, the check returns `evaluable=false` with a reason rather than a misleading pass/fail (FR-010, SC behavior). Threshold constants: `MIN_BAND_ITEMS`, `MIN_ANNOTATED_PER_BAND`.

---

## D3. Metric definition representation

**Decision**: A **declarative JSON metric definition** — an ordered list of signal terms, each `{signal, weight, transform, missing}` plus a global `direction` (higher = better | worse) and an optional `combine` op (default `weighted_sum`). `transform` ∈ {`identity`, `zscore`, `minmax`, `rank`, `clip`, `threshold`} with documented params; `missing` ∈ {`unscorable`, `fill:<value>`, `neutral`}. The combined score is a single float per item.

**Rationale**: Declarative (not code) keeps metric versions serializable, diffable, and safe to persist/replay — essential for immutable versioning and movement tracking. It is expressive enough for "combine audio-quality + ASR-confidence + single-speaker-confidence + PII-presence" while staying transparent to a non-technical stakeholder. Per-signal `transform` lets heterogeneous signals (probabilities, distances, counts) be combined sanely; `direction` lets the same definition rank best-first or worst-first.

**Alternatives considered**:
- *Arbitrary Python callable* — rejected: not serializable/replayable, unsafe to persist, breaks reproducibility and version diffing.
- *Full expression DSL / parser* — rejected for v1: more surface than needed; weighted-sum-of-transformed-signals covers the stated use cases. A `combine` field leaves room to add ops later without a schema break.

---

## D4. Assisted recalibration method

**Decision**: Frame recalibration as **learning-to-rank-lite via pairwise logistic regression** over annotated items. Build all annotated pairs with *distinct* quality levels; the feature vector for a pair is the **difference of (transformed) signal vectors**, the label is which item is better; fit `sklearn.linear_model.LogisticRegression` (with L2 regularization) to obtain new per-signal **weights**, then keep the existing transforms/direction. The proposal is returned for the researcher to review and accept (it never auto-replaces a version). Maximize/report the resulting Spearman/τ on the annotated set to confirm improvement.

**Rationale**: Pairwise logistic on signal differences is a standard, well-understood learning-to-rank reduction; it directly optimizes ordinal agreement (the primary measure), uses an existing dependency, is deterministic given a fixed seed/solver, and yields interpretable linear weights that drop straight back into the declarative metric definition (D3). It degrades gracefully with few pairs.

**Guards (FR-017)**: Refuse with a clear message when `n_annotations < MIN_ANNOTATIONS_RECAL` or the number of **distinct quality levels < 2** (no orderable pairs) or distinct-level spread is below `MIN_QUALITY_LEVELS_RECAL`. Warn (not refuse) when pair count is low enough to risk overfit; report effective pair count.

**Alternatives considered**:
- *Direct Spearman/Kendall maximization (Nelder-Mead / coordinate search)* — rejected as default: the objective is non-smooth and search is slower / seed-sensitive; pairwise logistic is a smooth, convex surrogate for the same goal.
- *Isotonic calibration of the combined score to labels* — kept as an optional post-step (maps score→calibrated quality for the triage readout) but not the weight-fitting mechanism; it does not change rank order so it cannot fix a bad combination.
- *Gradient-boosted ranker (LightGBM/XGBoost ranker)* — rejected: new heavy dependency, opaque, overkill at this scale and annotation volume.

---

## D5. Deterministic tie-breaking & band assignment

**Decision**: Primary sort key is the metric score (descending for higher-is-better); ties broken by a **stable secondary key = item id** (lexicographic). Implemented with a stable sort over `(score, item_id)`. Band assignment uses the configured fraction on the final ranked positions: top `ceil(band_fraction * N)` → `top`, bottom `ceil(band_fraction * N)` → `bottom`, remainder → `middle` (documented rounding; bands never overlap even at small N — if the two bands would meet, the middle is empty and overlap is resolved in favor of disjoint top/bottom by position).

**Rationale**: Guarantees SC-003 byte-identical reproducibility regardless of input row order or platform sort instability. Item-id tie-break is meaningful and stable (vs. random or input-order, which break reproducibility). Position-based bands keep the "coarse lens" semantics and avoid value-threshold ambiguity when many items share a score.

---

## D6. Signal ingestion / input format

**Decision**: The canonical input is a **per-item signal table** (parquet, via `pyarrow`) — one row per item, an `item_id` column, an optional `unit` column (`file` | `segment`) plus segment locator columns (`source_audio`, `start`, `end`) when unit is segment, and one float column per named signal (NaN = missing). A `harvest.py` adapter builds this table from existing `audio_analysis` outputs (per-axis parquets / `disagreements.json`), so the ranker is **signal-source-agnostic** and can also ingest a user-supplied table.

**Rationale**: A flat item→signals table is the minimal contract that decouples ranking from how signals were produced (the spec assumes signals are computed upstream). Parquet matches the repo's existing tabular persistence and handles ~100k rows trivially. NaN-as-missing lets `metric.py` apply the per-signal `missing` policy (D3).

**Alternatives considered**:
- *Directly coupling to `audio_analysis` internal types* — rejected: ties the ranker to one producer and one unit granularity; the adapter keeps that as an optional convenience instead.

---

## D7. Versioning, persistence & movement scope

**Decision**: A **ranking store** is a directory: `metric_versions/<vN>.json` (immutable), `rankings/<vN>.parquet`, `annotations.json` (append/merge, latest-wins conflict resolution with full history retained), `movement/<vA>__<vB>.json`, and `manifest.json` (index of versions, parent links, corpus/unit, timestamps). All writes are atomic (`*.tmp` → `os.replace`) and stamped with `schema_version`. Movement is computed between any two versions **over the same corpus and unit**; cross-unit or cross-corpus comparison is rejected.

**Rationale**: Mirrors `speaker_profile/io.py` (atomic JSON, schema_version) and `audio_analysis` parquet+provenance. Immutable, parent-linked versions give the audit trail and let movement tracking (D-) compare any pair without recomputation. Latest-wins with retained history satisfies the annotation-conflict edge case while keeping provenance.

**Movement report contents** (FR-020–FR-023): per shared item — old rank, new rank, ordinal shift (positions and percentile), old/new band; band-region movement summary (counts entering/leaving top and bottom bands, as a coarse signal); annotated-item highlights; and lists of added / removed / became-unscorable items. No exact per-item boundary-crossing audit is asserted (per the 2026-06-04 clarification; SC-007 relaxed to consistency-with-shifts).

---

## D8. Triage threshold semantics

**Decision**: A triage threshold is a **cut point on the ranking** (expressed as a rank position or percentile, default selectable) partitioning items into `auto_accept` (release-ready) above and `human_review` below. `triage.py` reports, for a chosen threshold, the counts of annotated-good / annotated-acceptable / annotated-poor items above vs. below, plus the implied poor-rate in the auto-accept region — so the researcher can move the threshold to bound the share of poor items they are willing to auto-accept (SC-009).

**Rationale**: This is the stated real purpose (PII / multiple-speaker / quality checks for dataset release). Expressing the threshold on the ranking (not on the raw score) keeps it stable under the coarse-metric assumption and directly answers "above here I trust it, below here a human looks." Counts against annotations make the confidence concrete and tunable without committing to a probabilistic calibration (isotonic calibration from D4 is an optional enhancement to the readout).
