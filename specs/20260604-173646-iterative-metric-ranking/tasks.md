# Tasks: Iterative Metric-Driven Ranking

**Input**: Design documents from `/specs/20260604-173646-iterative-metric-ranking/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Included — the senselab repo enforces `pytest`/`ruff`/`mypy` gates and signals are synthesizable (no model loads), so each story ships with fast unit tests. Write tests first within each story and let them fail before implementing.

**Organization**: Tasks are grouped by user story (P1 → P3) so each story is an independently testable increment.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependency on incomplete tasks)
- **[Story]**: US1 / US2 / US3 (Setup/Foundational/Polish carry no story label)
- All paths are repo-relative to `/home/wilke18/senselab`.

## Path Conventions

- Package: `src/senselab/audio/workflows/ranking/`
- CLI: `scripts/rank_audio.py`
- Tests: `src/tests/audio/workflows/ranking/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project skeleton and test fixtures.

- [X] T001 Create the ranking package skeleton with `__init__.py` (empty public-export stub) in `src/senselab/audio/workflows/ranking/__init__.py`
- [X] T002 [P] Create the test package and deterministic fixtures (synthetic item→signals tables with controllable rank-vs-quality alignment, plus synthetic annotation sets covering good/acceptable/poor, missing-signal rows, and tie rows) in `src/tests/audio/workflows/ranking/conftest.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Types, constants, persistence, store layout, and signal ingestion that every story depends on.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T003 [P] Define all dataclasses + Literals (`SignalTerm`, `MetricDefinition`, `MetricVersion`, `RankingItem`, `Ranking`, `Annotation`, `SeparationResult`, `TriageThreshold`, `RecalibrationResult`, `MovementReport`, `MovementEntry`; `RankUnit`/`QualityLabel`/`ScoreStatus`/`Band`/`Direction`/`SignalTransform`/`ItemDelta` literals) per data-model.md in `src/senselab/audio/workflows/ranking/types.py`
- [X] T004 [P] Define surfaced, documented constants — `DEFAULT_BAND_FRACTION=0.20`, ordinal→numeric map `{good:2, acceptable:1, poor:0}`, `MIN_BAND_ITEMS`, `MIN_ANNOTATED_PER_BAND`, `MIN_ANNOTATIONS_RECAL`, `MIN_QUALITY_LEVELS_RECAL`, `DEFAULT_SEPARATION_TARGET=0.80`, `TIE_BREAK="score_desc,item_id_asc"`, `SCHEMA_VERSION` — in `src/senselab/audio/workflows/ranking/constants.py`
- [X] T005 Implement persistence primitives — atomic write-then-`os.replace`, JSON load/save with `schema_version` guard, and pyarrow parquet read/write helpers (table+schema-metadata) — in `src/senselab/audio/workflows/ranking/io.py`
- [X] T006 Implement the ranking-store layout — directory bootstrap (`metric_versions/`, `rankings/`, `movement/`, `annotations.json`, `manifest.json`), `manifest` index, monotonic `next_version_id`, immutable metric-version write/read (refuse overwrite — FR-018), and version/lineage retrieval — in `src/senselab/audio/workflows/ranking/store.py` (depends on T005)
- [X] T007 Implement the signal-table loader (read parquet per `contracts/signal-table.parquet.md`: validate unique `item_id`, fixed `unit`, NaN=missing, expose signal-column list; raise on duplicate ids) in `src/senselab/audio/workflows/ranking/io.py` (depends on T005)

**Checkpoint**: Types, constants, store, and signal ingestion ready — user stories can begin.

---

## Phase 3: User Story 1 — Rank a corpus by a quality metric (Priority: P1) 🎯 MVP

**Goal**: Given a signal table + a metric definition, produce a complete, deterministic ranking (with bands) for an immutable metric version, and report ranking quality (rank-agreement primary, band separation secondary) against any available annotations.

**Independent Test**: Provide a synthetic signal table and a metric; confirm a complete ranking is emitted (every item once, unscorable items reported separately), re-running is byte-identical, and the separation/agreement check reports correctly (or `evaluable=false` when no annotations).

### Tests for User Story 1 ⚠️ (write first, ensure they fail)

- [X] T008 [P] [US1] Metric evaluation tests — weighted-sum combination, each transform, `direction`, missing-signal `unscorable`/`neutral`/`fill:` policies, and rejection of a metric referencing an unknown signal (FR-019) — in `src/tests/audio/workflows/ranking/metric_test.py`
- [X] T009 [P] [US1] Ranking tests — 100% item coverage (FR-002/SC-002), dense unique ranks, deterministic `(score, item_id)` tie-break, band assignment math at small/large N, unscorable items placed/flagged — in `src/tests/audio/workflows/ranking/rank_test.py`
- [X] T010 [P] [US1] Evaluation tests — Spearman/Kendall τ-b rank-agreement, band pairwise-agreement + margin, and `evaluable=false` reasons (too few items/annotations per band) (FR-008–010a) — in `src/tests/audio/workflows/ranking/evaluate_test.py`
- [X] T011 [P] [US1] Store/IO tests — metric-version immutability (overwrite refused), atomic write, `schema_version` round-trip, ranking parquet schema+metadata per `contracts/ranking.parquet.md` — in `src/tests/audio/workflows/ranking/store_io_test.py`
- [X] T012 [P] [US1] Reproducibility regression test — rank twice over identical inputs+version → byte-identical `rankings/<vN>.parquet` (SC-003) — in `src/tests/audio/workflows/ranking/regression_test.py`

### Implementation for User Story 1

- [X] T013 [US1] Implement metric scoring — validate `MetricDefinition` (≥1 term, signals exist in table else reject — FR-019), apply per-term transform (zscore/minmax/rank stats fit over scorable population) × weight, apply missing policy, produce per-item `score`+`status`+`reason` — in `src/senselab/audio/workflows/ranking/metric.py` (depends on T003, T004, T007)
- [X] T014 [US1] Implement ranking — stable sort by `(score, item_id)` honoring `direction`, dense 1-based ranks + percentile over scored items, position-based band assignment (disjoint top/bottom via `DEFAULT_BAND_FRACTION`), unscorable items appended (FR-002/004/005) — in `src/senselab/audio/workflows/ranking/rank.py` (depends on T013)
- [X] T015 [US1] Implement the ranking parquet writer (columns + schema metadata incl. `metric_definition_hash`, `tie_break`, `band_fraction` per `contracts/ranking.parquet.md`) in `src/senselab/audio/workflows/ranking/io.py` (depends on T005, T014)
- [X] T016 [US1] Implement the ranking-quality check — `spearmanr`/`kendalltau` via `scipy.stats` (numpy rank-then-Pearson fallback), band pairwise-agreement (AUC-style) + mean-quality margin, evaluability guards, `meets_separation_target` vs `DEFAULT_SEPARATION_TARGET` → `SeparationResult` (FR-008–010a) — in `src/senselab/audio/workflows/ranking/evaluate.py` (depends on T003, T004, T014)
- [X] T017 [US1] Implement the public `rank_corpus` entrypoint — create the initial/next immutable `MetricVersion`, score → rank → persist ranking + version via store (FR-007/018) — in `src/senselab/audio/workflows/ranking/__init__.py` + `rank.py` (depends on T006, T014, T015)
- [X] T018 [US1] Wire `rank` and `evaluate` CLI subcommands (argparse, `--store/--signals/--metric/--band-fraction`, JSON + human output, exit codes 0/2/3 per `contracts/rank-cli.md`) in `scripts/rank_audio.py` (depends on T016, T017)

**Checkpoint**: A corpus can be ranked, persisted as an immutable version, and its quality reported — MVP usable on its own.

---

## Phase 4: User Story 2 — Iteratively refine the metric via spot-checking and annotation (Priority: P2)

**Goal**: Spot-check (sampled) items, record quality annotations (persisted with provenance, latest-wins), update the metric by manual revision *or* assisted recalibration (new immutable version + ranking), and read a triage-threshold confidence cut.

**Independent Test**: From an existing ranking, sample + annotate items, then (a) manually revise the metric → new version, and (b) recalibrate from annotations → proposed version; confirm new versions/rankings are created, prior versions/annotations persist, agreement is no worse (SC-004), and recalibration refuses on too-few/low-variety annotations.

### Tests for User Story 2 ⚠️ (write first, ensure they fail)

- [X] T019 [P] [US2] Annotation-store tests — write/read per `contracts/annotation-store.schema.md`, at-most-one `active` per item with prior marked `superseded` and retained (FR-013), full set visible to every version (FR-014), batch ingest — in `src/tests/audio/workflows/ranking/annotate_test.py`
- [X] T020 [P] [US2] Recalibration tests — pairwise-logistic weights improve Spearman on the annotated set (`agreement_after ≥ before`), refusal when `n < MIN_ANNOTATIONS_RECAL` or `<2` distinct levels, warn on low pair count (FR-016/017) — in `src/tests/audio/workflows/ranking/recalibrate_test.py`
- [X] T021 [P] [US2] Triage-threshold tests — auto-accept vs human-review counts, annotated good/acceptable/poor above-vs-below, `auto_accept_poor_rate` at rank and percentile cuts, and **unscorable items never auto-accepted (auto-fail → human-review, counted as `n_unscorable_routed`)** (FR-010b/c, SC-009) — in `src/tests/audio/workflows/ranking/triage_test.py`

### Implementation for User Story 2

- [X] T022 [US2] Implement the annotation store — load/save `annotations.json` (atomic), record ordinal `label` and/or numeric `score` with provenance, latest-wins supersession with retained history, single-file batch ingest (FR-012/013/014) — in `src/senselab/audio/workflows/ranking/annotate.py` (depends on T005, T003)
- [X] T023 [P] [US2] Implement spot-check sampling strategies — `spread` (across rank regions/bands), `near-threshold` (around a candidate cut), `disagreement` — returning items to review (FR-011) — in `src/senselab/audio/workflows/ranking/annotate.py` (depends on T014)
- [X] T024 [US2] Implement manual metric update — accept a revised `MetricDefinition`, create a `manual`-origin immutable version, re-score + re-rank via US1 path (FR-015) — in `src/senselab/audio/workflows/ranking/__init__.py` (depends on T017, T006)
- [X] T025 [US2] Implement assisted recalibration — build distinct-level annotated pairs, fit `sklearn.linear_model.LogisticRegression` on signal-difference vectors → new weights folded into the metric definition, compute `agreement_before/after`, apply guards → `RecalibrationResult` (proposed, not auto-adopted) (FR-016/017) — in `src/senselab/audio/workflows/ranking/recalibrate.py` (depends on T013, T016, T022)
- [X] T026 [P] [US2] Implement triage thresholding — partition ranking at a rank/percentile cut, **force unscorable items into human-review (auto-fail), never auto-accept, and count them as `n_unscorable_routed`**, count annotated good/acceptable/poor on each side, compute `auto_accept_poor_rate` → `TriageThreshold` (FR-010b/c) — in `src/senselab/audio/workflows/ranking/triage.py` (depends on T014, T022)
- [X] T027 [US2] Wire `annotate`, `sample`, `update-metric`, `recalibrate` (`--accept`), and `threshold` CLI subcommands (exit code 4 on recalibration refusal) per `contracts/rank-cli.md` in `scripts/rank_audio.py` (depends on T022, T023, T024, T025, T026)

**Checkpoint**: A full refinement cycle (sample → annotate → update/recalibrate → re-rank) works; all versions/annotations retained (SC-005).

---

## Phase 5: User Story 3 — Track how items move when the metric changes (Priority: P3)

**Goal**: Compare any two versions over the same corpus+unit: per-item ordinal shift, coarse band-region movement, annotation highlights, and added/removed/became-unscorable accounting.

**Independent Test**: With two versions present, request a movement report; confirm it accounts for 100% of items with correct old/new ranks, band-region counts consistent with per-item shifts, annotated items highlighted, and unit/corpus mismatch rejected.

### Tests for User Story 3 ⚠️ (write first, ensure they fail)

- [X] T028 [P] [US3] Movement tests — 100% item coverage with `delta_kind` ∈ moved/unchanged/added/removed/became_unscorable (FR-023/SC-006), ordinal+percentile shift, `band_summary` consistent with per-entry band transitions (SC-007), annotation highlight (FR-022), unit/corpus-mismatch rejection — in `src/tests/audio/workflows/ranking/movement_test.py`

### Implementation for User Story 3

- [X] T029 [US3] Implement movement comparison — union of items across two rankings, per-item `from/to` rank+percentile+band and `delta_kind`, coarse `band_summary` (entered/left top & bottom), added/removed/became-unscorable lists, annotation highlight; reject mismatched unit/corpus (FR-020–023) — in `src/senselab/audio/workflows/ranking/movement.py` (depends on T006, T014, T022)
- [X] T030 [US3] Implement the movement-report JSON writer per `contracts/movement-report.schema.md` (atomic, `schema_version`, path `movement/<vA>__<vB>.json`) in `src/senselab/audio/workflows/ranking/io.py` (depends on T005, T029)
- [X] T031 [US3] Wire the `movement` CLI subcommand (`--from/--to`, prints band_summary + top movers + add/remove lists) per `contracts/rank-cli.md` in `scripts/rank_audio.py` (depends on T029, T030)

**Checkpoint**: All three stories functional and independently testable.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Optional adapter, public surface, docs, and gate validation.

- [X] T032 [P] Implement the optional `audio_analysis` adapter — `harvest_from_audio_analysis(run_dir, unit, out)` reading per-axis parquets / `disagreements.json` into a signal table (`contracts/signal-table.parquet.md`) — in `src/senselab/audio/workflows/ranking/harvest.py`
- [X] T033 [P] Add an adapter smoke test (synthesized analyze_audio-shaped inputs → valid signal table) in `src/tests/audio/workflows/ranking/harvest_test.py`
- [X] T034 [P] Finalize public exports in `src/senselab/audio/workflows/ranking/__init__.py` and write `src/senselab/audio/workflows/ranking/doc.md` (module overview + the refinement loop)
- [X] T035 Run and fix `cd src && uv run pytest && uv run ruff check . && uv run mypy .` until green
- [X] T036 Execute `quickstart.md` end-to-end against a synthetic store and confirm the SC smoke checks (SC-002/003/004/006/009)

---

## Dependencies & Execution Order

**Phase order**: Setup (P1) → Foundational (P2) → US1 (P3) → US2 (P4) → US3 (P5) → Polish (P6).

**Blocking**:
- Foundational (T003–T007) blocks **all** stories.
- US1 (T013–T018) is the MVP and underpins US2/US3 (they reuse the score→rank→persist path and the band logic).
- US2 depends on US1 (`update-metric`/`recalibrate` re-rank via US1; `triage`/sampling need a ranking + annotations).
- US3 depends on US1 (needs ≥2 rankings) and reuses annotations from US2 for highlights (but is testable with annotations alone).

**Story independence**: US1 is fully standalone. US2 and US3 each layer on US1 but do not depend on each other (US3 highlights annotations if present, otherwise reports movement without them).

## Parallel Opportunities

- **Setup/Foundational**: T002 ∥ T003 ∥ T004 (different files); then T005 → (T006 ∥ T007).
- **US1 tests**: T008 ∥ T009 ∥ T010 ∥ T011 ∥ T012 (all different test files).
- **US2 tests**: T019 ∥ T020 ∥ T021; impl T023 ∥ T026 can proceed alongside T022/T025.
- **US3**: single-file impl; T028 test in parallel with US2 polish.
- **Polish**: T032 ∥ T033 ∥ T034.

Example parallel batch (after Foundational):
```
T008, T009, T010, T011, T012   # write all US1 tests together (they should fail)
```

## Implementation Strategy

- **MVP = User Story 1** (Phases 1–3): a deterministic, persisted, quality-checked ranking of a corpus. Ship and validate this first.
- **Increment 2 = User Story 2** (Phase 4): the iterative loop (annotate → update/recalibrate → re-rank) + triage threshold — the feature's core value.
- **Increment 3 = User Story 3** (Phase 5): movement tracking for confidence/regression-catching.
- **Polish** (Phase 6): the `audio_analysis` adapter and gate/quickstart validation.
- Keep every metric change immutable and every artifact `schema_version`-stamped from US1 onward so US2/US3 inherit auditability for free.
