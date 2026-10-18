# Quickstart: Iterative Metric-Driven Ranking

End-to-end of the refinement loop. Assumes you have a **signal table** (parquet) with one row per item and one column per already-computed signal. See `contracts/signal-table.parquet.md`.

## 0. Install / run

```bash
uv run python scripts/rank_audio.py --help        # CLI
# or, as a library:
uv run python -c "from senselab.audio.workflows.ranking import rank_corpus"
```

## 1. Build (or harvest) a signal table

```bash
# Option A: you already have one item→signals parquet.
# Option B: harvest from an existing analyze_audio run:
uv run python -c "
from senselab.audio.workflows.ranking.harvest import harvest_from_audio_analysis
harvest_from_audio_analysis(run_dir='artifacts/analyze_audio/recXYZ',
                            unit='segment', out='signals.parquet')
"
```

## 2. Define a metric and produce the first ranking

```jsonc
// metric.json — combine signals; see contracts/metric-definition.schema.md
{
  "name": "release_quality",
  "direction": "higher_is_better",
  "terms": [
    {"signal": "audio_quality",             "weight": 0.4, "transform": "minmax"},
    {"signal": "asr_confidence",            "weight": 0.3},
    {"signal": "single_speaker_confidence", "weight": 0.3},
    {"signal": "pii_presence",              "weight": -0.5, "transform": "threshold", "transform_params": {"at": 0.5}, "missing": "fill:0.0"}
  ]
}
```

```bash
uv run python scripts/rank_audio.py rank \
  --store ./ranking_store --signals signals.parquet --metric metric.json
# → creates v1, writes rankings/v1.parquet; prints n_scored / n_unscorable / band sizes
```

## 3. Spot-check + annotate

```bash
# Sample across the ranking (and near a candidate threshold):
uv run python scripts/rank_audio.py sample --store ./ranking_store --version v1 --n 30 --strategy spread
# Review each, then record quality judgments (good/acceptable/poor):
uv run python scripts/rank_audio.py annotate --store ./ranking_store \
  --item "rec123#12.30-15.80" --label poor --note "second speaker 13.1-14.0s"
```

## 4. See how good the ranking is

```bash
uv run python scripts/rank_audio.py evaluate --store ./ranking_store --version v1
# → Spearman ρ, Kendall τ-b (primary); top-vs-bottom pairwise agreement + margin (secondary, default band=20%)
#   or: evaluable=false (reason: too few annotated items in bottom band)
```

## 5. Update the metric — two ways

```bash
# (a) Manual revision: edit metric.json (e.g. raise the PII penalty), then:
uv run python scripts/rank_audio.py update-metric --store ./ranking_store --metric metric.json   # → v2

# (b) Assisted recalibration from annotations (proposes new weights):
uv run python scripts/rank_audio.py recalibrate --store ./ranking_store
# → prints agreement_before/after, pair count; refuses if too few annotations / <2 quality levels
uv run python scripts/rank_audio.py recalibrate --store ./ranking_store --accept                 # → v3
```

## 6. Track movement between versions

```bash
uv run python scripts/rank_audio.py movement --store ./ranking_store --from v1 --to v3
# → band_summary (entered/left top & bottom), top movers, added/removed/became-unscorable,
#   annotated items highlighted so you can see if they landed where expected
```

## 7. Place a triage threshold (release vs. human review)

```bash
uv run python scripts/rank_audio.py threshold --store ./ranking_store --version v3 --at-percentile 0.7
# → auto_accept vs human_review counts; annotated good/acceptable/poor above vs below;
#   auto_accept poor-rate → move the cut until the release region's poor-rate is acceptable
```

## Acceptance smoke checks (map to spec Success Criteria)

- **SC-002**: rerun `rank`; every input item appears once; unscorable items listed separately.
- **SC-003**: run `rank` twice on identical inputs → byte-identical `rankings/vN.parquet`.
- **SC-004**: `evaluate` v(new) ≥ v(prev) on the annotated set before adopting.
- **SC-006**: `movement` accounts for 100% of items (moved/unchanged/added/removed/became-unscorable).
- **SC-009**: `threshold` readout lets you bound the auto-accept poor-rate.

## Quality gates

```bash
cd src && uv run pytest && uv run ruff check . && uv run mypy .
```
