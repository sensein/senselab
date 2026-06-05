# `senselab.audio.workflows.ranking`

Iterative **metric-driven ranking** of a corpus of audio items (whole files or
segments). Order every item by a versioned metric that combines already-computed
signals, refine that metric over rounds of spot-checking, and track how items
move between metric versions. The driving use case is **dataset-release triage**:
place a cut point above which items are confidently good (auto-accept) and below
which they go to human review — for PII / multiple-speaker / general-quality
checks.

The workflow is **signal-source-agnostic**: its input is a generic per-item
signal table (one row per item, one float column per signal), so it ranks
anything reducible to a row of numbers. It performs **no model inference** — it
consumes signals computed upstream (e.g. by `audio_analysis`).

## The refinement loop

```
rank a corpus (metric v1)
   → spot-check a sample (across the ranking / near a candidate threshold)
   → annotate items with quality (good / acceptable / poor, numeric optional)
   → evaluate (rank-agreement + band separation against annotations)
   → update the metric:
        • manual revision (edit weights / transforms / thresholds), or
        • assisted recalibration (fit weights from annotations)
   → new immutable metric version + ranking
   → movement report (how items shifted v_prev → v_new)
   → place a triage threshold (auto-accept vs. human-review)
repeat
```

Every metric change is a new **immutable version**; annotations persist across
versions (latest-wins, history retained).

## Pipeline

**Score → rank** (`metric.py`, `rank.py`):

```
signal table (parquet) → score each item under a MetricDefinition
   (per-signal transform × weight; missing → unscorable by default,
    or an explicit per-signal fallback)
 → deterministic order: stable sort by (score, item_id), honoring `direction`
 → dense 1-based ranks + percentile; position-based bands (default top/bottom 20%)
 → unscorable items reported separately (never dropped)
 → Ranking (parquet), one per immutable MetricVersion
```

**Evaluate** (`evaluate.py`): against active annotations, report **rank
agreement** (Spearman ρ + Kendall τ-b) as the primary quality measure and
**top-vs-bottom band separation** (AUC-style pairwise agreement + margin) as a
coarse secondary check; report `evaluable=false` when there is too little
annotated data.

**Refine** (`annotate.py`, `recalibrate.py`): record quality annotations
(latest-wins store) and sample items to review; assisted recalibration frames
the problem as pairwise logistic regression over annotated pairs to propose new
per-signal weights (advisory — never auto-adopted; refuses on too-few / too-uniform
annotations).

**Triage** (`triage.py`): partition a ranking at a rank/percentile cut into
auto-accept vs. human-review, and report annotated good/acceptable/poor counts on
each side plus the auto-accept poor-rate. **Unscorable items are auto-fail** —
never auto-accepted, always routed to human review.

**Movement** (`movement.py`): compare two versions over the same corpus + unit —
per-item ordinal shift, coarse band-region movement (a lens, not an exact
ledger), annotated-item highlights, and added / removed / became-unscorable
accounting.

## Key concepts

- **Metric is declarative.** A `MetricDefinition` is a serializable list of
  `{signal, weight, transform, missing}` terms plus a `direction` — so versions
  are diffable and replayable, which is what makes immutable versioning and
  movement tracking possible.
- **Bands are a coarse lens.** Top/bottom bands (default 20%, configurable) are a
  triage/evaluation device; neighbor order is not asserted exact. Given imperfect
  metrics, movement is tracked as ordinal/rank shift, not precise boundary
  crossings.
- **Determinism.** Identical inputs + version reproduce a byte-identical ranking
  (ties broken by `item_id`).
- **Low-sensitivity store.** Holds only signal values, item ids, and quality
  labels/notes — never raw audio, transcripts, or extracted PII content.

## Persistence (a ranking store)

```
<store>/
    manifest.json                # versions, lineage, unit
    metric_versions/<vN>.json    # immutable metric definitions (+ recal provenance)
    rankings/<vN>.parquet        # one ranking per version
    annotations.json             # latest-wins, history retained
    movement/<vA>__<vB>.json     # two-version comparisons
```

Writes are atomic (`*.tmp` → `os.replace`) and stamped with `schema_version`.
Schemas are the contracts under
`specs/20260604-173646-iterative-metric-ranking/contracts/`.

## Inputs

The signal table is supplied by the caller, or built from an `audio_analysis`
run via the optional `harvest.py` adapter (pivots per-axis uncertainty parquets
into a per-segment signal table). Higher uncertainty = worse, so such a metric
typically uses negative weights or `lower_is_better`.

## CLI

`scripts/rank_audio.py` exposes the loop as subcommands: `rank`, `evaluate`,
`sample`, `annotate`, `update-metric`, `recalibrate`, `threshold`, `movement`.
See `contracts/rank-cli.md`.

## No new dependencies

Pure `numpy` / `scikit-learn` (`LogisticRegression`; `scipy.stats` for rank
correlations) / `pyarrow`. CPU-only.
