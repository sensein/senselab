# Implementation Plan: Iterative Metric-Driven Ranking

**Branch**: `20260604-173646-iterative-metric-ranking` | **Date**: 2026-06-04 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/20260604-173646-iterative-metric-ranking/spec.md`

## Summary

Add a **standalone ranking workflow** that takes a per-item **signal table** (one row per item — a whole audio file *or* a segment — with one column per already-computed signal: audio-quality measures, ASR-confidence indicators, single-speaker-confidence indicators, PII-presence indicators, etc.) and a **versioned metric definition** that combines those signals into a single comparable score. It emits a complete, deterministic **rank ordering** of the items, assigns each a coarse **band** (configurable, default top/bottom 20%), and reports **ranking quality** as a **rank-agreement measure** against any available quality annotations (primary) plus **top/bottom-band separation** (secondary).

On top of the ranking it supports the **iterative refinement loop** that is the point of the feature: the researcher **spot-checks** items (sampled across the ranking and near a candidate triage threshold), records **quality annotations** on a small fixed ordinal scale (good / acceptable / poor, numeric optional), and then **updates the metric** either by **manual revision** (edit weights / thresholds / which signals are combined) or by **assisted recalibration** (fit the combination to maximize rank agreement with the accumulated annotations). Every metric change creates a new **immutable version** with its own ranking; annotations persist across versions. The researcher can place a **triage threshold** — items above it are treated as confidently good (release-ready / auto-accepted) and the rest are routed to human review — and read how annotated-good vs. annotated-poor items fall on each side. Finally, a **movement report** compares any two versions: per-item **ordinal/rank shift**, **band-region movement** (a coarse lens, not an exact ledger), annotation highlights, and added/removed/became-unscorable accounting.

The design **reuses existing senselab patterns**: it consumes the kind of per-window / per-file signals already produced by the `audio_analysis` workflow (the 9 uncertainty-axis parquets + `disagreements.json` ranked index are the direct conceptual ancestor of this generalized, annotation-refinable ranking), follows the **dataclass-internal + parquet/JSON-on-disk + atomic-write + `schema_version`-stamped** persistence conventions from `speaker_profile` and `audio_analysis`, and ships a thin CLI mirroring `analyze_audio.py` / `build_speaker_profile.py`. The metric layer is **signal-source-agnostic** (input is a generic item→signals table), so it ranks anything that can be reduced to a row of numbers, with an adapter to harvest signals from `audio_analysis` outputs.

## Technical Context

**Language/Version**: Python ≥3.11,<3.15 (per `pyproject.toml`)
**Primary Dependencies**: scientific stack already in the repo — `numpy`, `scikit-learn` (≥1.7; `LogisticRegression` for pairwise/learning-to-rank-lite recalibration, plus its transitive `scipy.stats` for `spearmanr` / `kendalltau`), `pyarrow` (parquet rankings + signal tables). No new third-party dependency. Optional integration adapter reads `audio_analysis` per-axis parquets / `disagreements.json`. No model inference in this feature — it operates on already-extracted signals.
**Storage**: Filesystem only — a per-corpus **ranking store** directory holding: metric-version JSON files, ranking parquet files (one per version), an annotations JSON store, movement-report JSON files, and a `manifest.json` index. Atomic write-then-`os.replace`; explicit `schema_version` on every artifact (mirrors `speaker_profile/io.py`).
**Testing**: `pytest` (with `pytest-xdist`, `pytest-mock`, `pytest-cov`); `ruff` lint (line-length 120); `mypy` (pydantic plugin) — matches repo gates. No model loads in tests; signal tables are synthesized in fixtures.
**Target Platform**: Linux (CPU is sufficient — pure numeric/sort/optimize work); clinical-research batch use on HPC (the senselab usage context).
**Project Type**: Single Python library + CLI scripts (senselab `src/` package + `scripts/` entry points).
**Performance Goals**: Rank a corpus of up to ~100k items (full re-rank per metric version) in a few seconds on a single CPU core (dominated by an `O(n log n)` sort); recalibration runs only over the annotated subset (typically O(10²–10³) items) so it stays sub-second to a few seconds; deterministic, reproducible output is prioritized over incremental-update performance.
**Constraints**: Imperfect metrics — neighbor order is not asserted exact; only macro-scale separation / rank trend. Bands are a coarse lens (default 20%, configurable), not an exact per-item ledger. Assisted recalibration refuses/warns below a minimum annotation count and minimum distinct-quality-level spread. Missing signals never silently drop an item (reported unscorable by default, or scored via an explicit per-signal fallback). **Low-sensitivity store**: the ranking store holds only derived signal values, item identifiers, and quality labels/notes — never raw audio, transcripts, or extracted PII content; PII enters only as a numeric indicator signal, and item IDs / annotation notes are expected to be PII-free (raw-media inspection happens in the separate human-review step, outside this system).
**Scale/Scope**: ~100k items per ranking run; single corpus per ranking store; unit (file vs segment) chosen per run and never conflated; streaming / distributed scale out of scope for v1.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

The project constitution at `.specify/memory/constitution.md` is an **unpopulated template** (placeholder principles only) — there are no ratified gates to evaluate against, so the gate passes vacuously. In its place, this plan adheres to the **established senselab repo conventions** observed in the codebase (and followed by the prior `speaker_profile` feature):

- **Library-first**: core logic lives under `src/senselab/audio/workflows/ranking/` as importable, independently testable functions; the CLI script is a thin wrapper (mirrors `analyze_audio.py` + `workflows/audio_analysis/`).
- **Reuse over reinvention**: consume `audio_analysis` signal outputs via an adapter rather than recomputing them; use existing `scikit-learn` / `scipy.stats` for correlation and recalibration rather than bespoke math.
- **Graceful degradation**: missing signals / unscorable items / not-enough-annotations are captured as structured statuses and reasons, not aborts (matches the existing `failures`/`comparison_status` patterns).
- **Auditability & immutability**: every metric change is a new immutable version; provenance (signals used, weights, recalibration method, parent version) is stamped into artifacts; outputs carry `schema_version` (matches parquet provenance / profile JSON stamping).
- **Surfaced constants**: thresholds (default band fraction 0.20, min annotations for recalibration, rank-correlation method, tie-break rule) live in a documented `constants.py`, not as buried magic numbers.
- **Quality gates**: `ruff` + `mypy` + `pytest` must pass.

**Result**: PASS (no gates defined; repo conventions adopted). Re-checked post-design — still PASS (pure-numeric, additive new package; no cross-cutting changes to existing modules — the `audio_analysis` adapter only *reads* existing outputs).

## Project Structure

### Documentation (this feature)

```text
specs/20260604-173646-iterative-metric-ranking/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
│   ├── signal-table.parquet.md       # INPUT: per-item signal table schema
│   ├── metric-definition.schema.md   # Versioned metric definition (JSON)
│   ├── ranking.parquet.md            # Ranking output table schema
│   ├── annotation-store.schema.md    # Annotations store (JSON)
│   ├── movement-report.schema.md     # Two-version movement report (JSON)
│   └── rank-cli.md                   # rank_audio CLI subcommand schemas
└── checklists/
    └── requirements.md  # Created by /speckit.specify
```

### Source Code (repository root)

```text
src/senselab/audio/workflows/
├── audio_analysis/                 # EXISTING — read-only signal source (adapter target)
│   ├── disagreements.py            #   conceptual ancestor of the ranked index
│   └── types.py / io.py            #   per-axis parquet shapes the adapter harvests
└── ranking/                        # NEW package
    ├── __init__.py                 # public exports
    ├── types.py                    # MetricDefinition, MetricVersion, RankingItem, Ranking,
    │                               #   Annotation, SeparationResult, TriageThreshold, MovementReport,
    │                               #   QualityLabel literal, ScoreStatus literal
    ├── constants.py                # DEFAULT_BAND_FRACTION=0.20, ordinal scale, MIN_ANNOTATIONS_*,
    │                               #   RANK_CORRELATION method, tie-break rule
    ├── metric.py                   # evaluate a MetricDefinition over a signal table → scores + statuses
    ├── rank.py                     # scores → deterministic ranking + band assignment
    ├── evaluate.py                 # rank-agreement (primary) + band separation (secondary); evaluability
    ├── recalibrate.py              # assisted recalibration (pairwise/learning-to-rank-lite) + guards
    ├── triage.py                   # threshold placement + annotated good/poor above-vs-below counts
    ├── movement.py                 # two-version ordinal shift + band-region movement + add/remove
    ├── annotate.py                 # annotation store load/save + conflict resolution (latest-wins)
    ├── store.py                    # ranking-store layout, immutable version management, manifest
    ├── io.py                       # parquet/JSON read+write, atomic writes, schema_version stamping
    └── harvest.py                  # adapter: audio_analysis outputs → signal table (optional)

scripts/
└── rank_audio.py                   # NEW thin CLI: rank | annotate | update-metric |
                                    #   recalibrate | threshold | movement subcommands

src/tests/audio/workflows/ranking/  # NEW tests
├── conftest.py                     # deterministic synthetic signal tables + annotations
├── metric_test.py                  # combination, direction, missing-signal/unscorable, fallback
├── rank_test.py                    # completeness, deterministic ties, band assignment
├── evaluate_test.py                # rank-agreement + band separation + not-evaluable reporting
├── recalibrate_test.py             # agreement improves; refuses on too-few/low-variety annotations
├── triage_test.py                  # threshold counts (good/poor above vs below)
├── movement_test.py                # ordinal shift, band-region movement, add/remove/unscorable
├── annotate_test.py                # provenance, latest-wins conflict resolution, cross-version retention
├── store_io_test.py                # version immutability, atomic write, schema_version round-trip
└── regression_test.py             # SC-003 reproducible (identical re-rank incl. tie-break)
```

**Structure Decision**: Single-project senselab layout. Core logic is a new importable package `src/senselab/audio/workflows/ranking/`, parallel to `speaker_profile/` and `audio_analysis/`. It depends only on the scientific stack already present (`numpy`, `scikit-learn`, `pyarrow`); the `harvest.py` adapter *reads* `audio_analysis` outputs but no existing module is modified, so the feature is fully additive. A thin `scripts/rank_audio.py` mirrors the `analyze_audio.py` / `build_speaker_profile.py` pattern and exposes the loop as subcommands.

## Complexity Tracking

> No constitution gates are defined, and the design introduces no flagged complexity violations. The one non-trivial design choice (the assisted-recalibration method) is resolved in `research.md` toward the simplest approach that uses an existing dependency. Table intentionally empty.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| — | — | — |
