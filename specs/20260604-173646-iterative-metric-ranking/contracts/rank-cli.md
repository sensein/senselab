# Contract: `rank_audio` CLI

Thin wrapper (`scripts/rank_audio.py`) over `senselab.audio.workflows.ranking`. Mirrors the `analyze_audio.py` / `build_speaker_profile.py` style: argparse subcommands, JSON + human-readable output, errors to stderr, non-zero exit on failure. Heavy logic lives in the importable package; the script only parses args and calls functions.

Common options: `--store <dir>` (ranking store path, required), `--json` (machine-readable output).

## `rank` — produce a ranking for a metric version

```
rank_audio rank --store S --signals signals.parquet \
                --metric metric.json [--as-version vN] [--band-fraction 0.20]
```
- Loads the signal table + metric definition; creates the next immutable `MetricVersion` (or `--as-version`); writes `rankings/<vN>.parquet`; prints summary (`n_scored`, `n_unscorable`, band sizes).
- Rejects a metric referencing a signal absent from the table (FR-019).

## `annotate` — record quality judgments

```
rank_audio annotate --store S --item ITEM_ID --label good|acceptable|poor [--score X] [--note "..."] [--reviewer ID]
rank_audio annotate --store S --from-file annotations.csv      # batch
rank_audio sample   --store S --version vN --n 30 [--strategy spread|near-threshold|disagreement] [--threshold T]
```
- `sample` selects items to spot-check (across-ranking spread, near a candidate triage threshold, or disagreement zones — FR-011) and prints them for review.
- `annotate` writes to `annotations.json` with latest-wins supersession (FR-013/014).

## `update-metric` — manual revision

```
rank_audio update-metric --store S --metric revised_metric.json
```
- Creates a new `manual`-origin version + ranking (FR-015).

## `recalibrate` — assisted recalibration

```
rank_audio recalibrate --store S [--accept] [--min-annotations N]
```
- Fits pairwise-logistic weights from active annotations; prints `agreement_before/after`, pair count, distinct levels.
- Refuses with a clear message when annotations are too few / too few distinct levels (FR-017); only writes a new `recalibrated` version when `--accept` (FR-016/018).

## `evaluate` — ranking-quality report

```
rank_audio evaluate --store S --version vN [--separation-target 0.80]
```
- Prints rank-agreement (Spearman/τ-b) + band pairwise-agreement + margin; reports `evaluable=false` with reason when insufficient data (FR-008–010a).

## `threshold` — triage cut readout

```
rank_audio threshold --store S --version vN (--at-rank K | --at-percentile P)
```
- Prints auto-accept vs human-review counts and annotated good/acceptable/poor on each side + auto-accept poor-rate (FR-010b/c, SC-009).
- **Unscorable items are never auto-accepted**: they count as auto-fail and are always routed to human review regardless of the cut (reported as `n_unscorable_routed`, included in the human-review count).

## `movement` — compare two versions

```
rank_audio movement --store S --from vA --to vB [--json]
```
- Writes `movement/<vA>__<vB>.json`; prints band_summary + top movers + added/removed/became-unscorable (FR-020–023).

## Exit codes
`0` success · `2` usage / invalid metric (e.g. unknown signal) · `3` not-evaluable where a value was required · `4` recalibration refused (insufficient annotations).
