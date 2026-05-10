# Tasks: Comparison & Uncertainty Stage

**Feature**: Comparison & Uncertainty Stage — three-axis uncertainty (presence / identity /
utterance) for `analyze_audio.py`, with the reusable comparator extracted into
`senselab.audio.workflows.audio_analysis`.

**Branch**: `20260508-173136-compare-uncertainty` | **Plan**: [plan.md](./plan.md)
| **Spec**: [spec.md](./spec.md)

## Architectural directive

Per the user instruction at `/speckit.tasks` time:

> **anything common and useful should be added to senselab, including an analysis workflow,
> with the script as a lightweight wrapper as appropriate**

The reusable comparator (axis aggregators, per-axis vote harvesters, per-segment speaker
embeddings, BucketGrid, disagreements index, timeline plot) lives in a new senselab module
`src/senselab/audio/workflows/audio_analysis/`. The CLI script
`scripts/analyze_audio.py` is thinned to:

1. Argparse + run-directory layout (script-specific I/O).
2. Per-task orchestration + caching (existing — out of scope for this PR's refactor).
3. **One call to `senselab.audio.workflows.audio_analysis.compute_uncertainty_axes(...)`**
   that replaces the per-pair `compare_*` functions and the `_diff_*` differencers.
4. Parquet writers + LS-bundle generator that consume the workflow's return value.

## User-story mapping recap (from spec.md, re-aligned to the three-axis design)

| Story | Priority | Output that delivers it |
|---|---|---|
| **US1** — "did enhancement help here?" at-a-glance | P1 | 3 raw-vs-enhanced parquets + 3 LS `pass_pair__uncertainty__*` tracks |
| **US2** — "where do my models disagree on this clip?" | P2 | 6 per-pass parquets + 6 LS `<pass>__uncertainty__*` tracks |
| **US3** — cross-stream sanity (ASR↔diar, scene↔diar, ASR↔PPG) | P2 | Cross-task signals contribute to the three axes (no separate parquets); ASR token-overlap → presence; AST/YAMNet Speech allowlist → presence; ASR-vs-PPG PER → utterance |
| **US4** — per-region uncertainty + ranked discovery | P3 | `disagreements.json` index + `timeline.png` 5-row plot + utterance TextArea sibling on LS |

US3 is fully *integrated* into the same parquets as US1 and US2 — there are no
cross-stream-specific files. The maximally-inclusive contribution policy from FR-002
makes US3 a property of the aggregator, not a separate output channel.

---

## Phase 1: Setup

- [X] T001 Create senselab workflow module skeleton at `src/senselab/audio/workflows/audio_analysis/__init__.py` with placeholder `compute_uncertainty_axes` signature, `__all__` export list, and one-line module docstring pointing back to `specs/20260508-173136-compare-uncertainty/`. No logic yet — just the package boundary.
- [X] T002 Create matching test directory `src/tests/audio/workflows/audio_analysis/__init__.py` and a placeholder `test_smoke.py` that imports the module and asserts `compute_uncertainty_axes` exists.
- [X] T003 [P] Add a `doc.md` next to `__init__.py` (mirrors `health_measurements/doc.md`) summarizing the three-axis design with a pointer to `specs/20260508-173136-compare-uncertainty/spec.md` and `quickstart.md`.

---

## Phase 2: Foundational (blocks all user stories)

- [X] T004 [P] Implement `BucketGrid` dataclass + `iter_buckets()` generator in `src/senselab/audio/workflows/audio_analysis/grid.py`. Mirror the in-script implementation but make `win_length` / `hop_length` validated in `__post_init__` (raise `ValueError` on `hop > win`).
- [X] T005 [P] Implement the four uncertainty aggregators (`min`, `mean`, `harmonic_mean`, `disagreement_weighted`) in `src/senselab/audio/workflows/audio_analysis/aggregators.py`. Each takes `list[float | None]` (sub-signal uncertainties), drops `None`s, and returns a single `float | None` (None when all inputs were None). Match the formulas in `data-model.md`.
- [X] T006 [P] Define typed dataclasses for the workflow's domain in `src/senselab/audio/workflows/audio_analysis/types.py`: `UncertaintyAxis` (Literal["presence","identity","utterance"]), `UncertaintyRow`, `PerSegmentEmbedding`, `AxisResult` (in-memory holder for one axis's rows + provenance). Use Pydantic v2 to match the rest of senselab's data structures.
- [X] T007 Implement `comparator_cache_key()` in `src/senselab/audio/workflows/audio_analysis/cache.py` — sha256 over `(audio_signature, axis, pass_set, model_set, params, wrapper_hash, senselab_version, schema_version)`. `_CACHE_SCHEMA_VERSION = 1`.
- [X] T008 [P] Write smoke tests for grid / aggregators / cache_key in `src/tests/audio/workflows/audio_analysis/test_grid_aggregators_cache.py`. Cover: bucket iteration over a 2.5 s window with 1.0 s grid; aggregator math vs hand-computed expected values; cache key changes when any input field changes.
- [X] T009 [P] Drop the v1 per-pair helpers from `scripts/analyze_audio.py`: `_diff_diarization`, `_diff_classification`, `_diff_asr`, `_diff_asr_vs_diarization`, `_diff_classification_vs_diarization`, `_diff_asr_vs_ppg`, `compare_raw_vs_enhanced`, `compare_within_stream`, `compare_cross_stream`, `_attach_comparator_regions_to_ls_tasks`, `_disagreement_severity_rank`, plus the v1 `_diff_*` test fixtures in `src/tests/scripts/analyze_audio_test.py` (helpers `_mk_diar_result`, `_mk_asr_result` and the Phase 3/4/5 per-pair tests). **Preserve** `test_comparator_cli_flags_parse` and `test_comparator_cli_flag_validation` — they still verify the FR-010 default grid (0.5 / 0.5) and the `--speech-presence-labels nargs="+"` parsing, which the workflow rewrite must not regress. Leave the harvesters (`_whisper_chunk_confidence`, `_whisper_bucket_confidence`) in place — T013 / T013b move them into the workflow module.
- [X] T009b [P] Add explicit FR-008 / SC-005 coverage in `src/tests/scripts/analyze_audio_test.py`: a smoke test asserting `parse_args(["audio.wav", "--skip", "comparisons"])` sets `args.skip` such that no `<pass>/uncertainty/` subtree is created and no `disagreements.json` is written. Pair it with a `--disagreements-top-n 0` test that confirms the index file is skipped while the per-axis parquets and timeline plot are still produced.

---

## Phase 3 (US1, P1): Raw-vs-enhanced delta uncertainty parquets

**Goal**: Reviewer can answer "did enhancement help here?" by reading three
`<run_dir>/uncertainty/raw_vs_enhanced/{presence,identity,utterance}.parquet` files (and
their LS counterparts).

**Independent test**: Run the script with the default two-pass mode on a clip; verify three
parquets exist under `<run_dir>/uncertainty/raw_vs_enhanced/`, each with non-empty rows for
buckets where the two passes disagreed; verify three matching `pass_pair__uncertainty__*`
tracks land in the LS bundle.

- [X] T010 [P] [US1] Implement per-segment speaker embedding extraction in `src/senselab/audio/workflows/audio_analysis/embeddings.py`. API: `extract_per_segment_embeddings(audio: Audio, segments: list[ScriptLine], models: list[str], device: DeviceType | None) -> dict[str, list[np.ndarray]]` — returns `{model_id → list-of-embedding-vectors-one-per-segment}`. Reuses `senselab.audio.tasks.speaker_embeddings.extract_speaker_embeddings` per slice. Slice the audio waveform between `seg.start` and `seg.end` before calling the existing API.
- [X] T011 [P] [US1] Implement presence-axis vote harvest in `src/senselab/audio/workflows/audio_analysis/presence.py`. Function: `harvest_presence_votes(pass_summary, grid, speech_presence_labels, alignment_by_model) -> list[BucketVotes]`. For each bucket emit a dict `{model_id → {"speaks": bool, "native_confidence": float | None}}` from diar (segment overlap with the bucket window), ASR (token overlap — for text-only ASR backends without per-token chunks, consult the post-MMS `alignment_by_model` block per FR-011), AST/YAMNet (top-1 label in `speech_presence_labels` allowlist, projected onto the bucket grid via `floor(start / native_hop)` per FR-010).
- [X] T012 [P] [US1] Implement identity-axis vote harvest in `src/senselab/audio/workflows/audio_analysis/identity.py`. Function: `harvest_identity_votes(pass_summary, grid, per_segment_embeddings) -> list[BucketVotes]`. For each bucket emit `{diar_model → {"speaker_label": str}}` plus `{embedding_model → {"embedding_cosine_to_prev": float | None}}` (looked up from per-segment embeddings via diar segment overlap, cosine to prior bucket on the same speaker track).
- [X] T013 [P] [US1] Implement utterance-axis vote harvest in `src/senselab/audio/workflows/audio_analysis/utterance.py`. Function: `harvest_utterance_votes(pass_summary, grid, ppg_block) -> list[BucketVotes]`. For each bucket emit `{asr_model → {"text": str, "avg_logprob": float | None, "phoneme_per_to_ppg": float | None}}` using the alignment-fallback path from FR-011 and the harvesters in `src/senselab/audio/workflows/audio_analysis/harvesters.py` (T013b).
- [X] T013b [P] [US1] Move the reusable harvester helpers from `scripts/analyze_audio.py` into `src/senselab/audio/workflows/audio_analysis/harvesters.py`: `whisper_chunk_confidence` (formerly `_whisper_chunk_confidence`), `whisper_bucket_confidence` (formerly `_whisper_bucket_confidence`), `g2p_phonemes` (formerly `_g2p_phonemes`), `phoneme_per` (the PER core extracted from the deleted `_diff_asr_vs_ppg`). Remove the underscore prefixes since they're now public workflow API. Update T011 / T013 / T028 to import from `senselab.audio.workflows.audio_analysis.harvesters`. Per the user's "anything common and useful goes in senselab" directive at `/speckit.tasks` time.
- [X] T014 [US1] Implement axis aggregation in `src/senselab/audio/workflows/audio_analysis/aggregate.py`. Functions: `aggregate_presence(votes) -> float`, `aggregate_identity(votes, raw_vs_enh, aggregator) -> float`, `aggregate_utterance(votes, aggregator) -> float`. Match the math in `data-model.md`'s pseudocode block.
- [X] T015 [US1] Implement `compute_uncertainty_axes(passes, grid, params, *, audio, speaker_embedding_models, aggregator) -> dict[(pass, axis) → AxisResult]` in `src/senselab/audio/workflows/audio_analysis/__init__.py`. This is the public entry point: orchestrates per-segment embeddings → per-axis vote harvest → aggregation, returns nine `AxisResult`s (3 axes × 2 passes + 3 raw_vs_enhanced deltas). Pure function; no I/O.
- [X] T016 [P] [US1] Add unit tests covering each axis's aggregation in `src/tests/audio/workflows/audio_analysis/test_aggregate.py`. Hand-build vote dicts, call the aggregators, assert against expected uncertainties (presence: 50/50 votes → 1.0; identity: agreeing labels → 0.0; utterance: identical transcripts → 0.0).
- [X] T017 [US1] Implement parquet writer `write_axis_parquet(axis_result, dest, provenance)` in `src/senselab/audio/workflows/audio_analysis/io.py`. Schema per `contracts/uncertainty-row.parquet.md`. Uses pyarrow; serializes `model_votes` as a struct map; embeds `comparator_provenance` JSON in `schema.metadata`.
- [X] T018 [US1] Wire `scripts/analyze_audio.py` to call `compute_uncertainty_axes` after the per-task pipeline runs and before `disagreements.json`/LS-bundle assembly. Pass `args` → `params` dict; pass cached `passes` summary; receive nine `AxisResult`s; write each via `write_axis_parquet` to the FR-003 layout (`<run_dir>/<pass>/uncertainty/...` and `<run_dir>/uncertainty/raw_vs_enhanced/...`).
- [X] T019 [P] [US1] Add an end-to-end smoke test in `src/tests/audio/workflows/audio_analysis/test_compute_uncertainty_axes.py` driven by SimpleNamespace-built passes summaries (no real models). Cover: a 4 s clip with two diar models agreeing on speech/silence + two ASR models with one transcript edit; verify all three axes' parquets land with the right row counts and `aggregated_uncertainty` ranges.

---

## Phase 4 (US2, P2): Per-pass uncertainty parquets

**Goal**: Reviewer can answer "where do my models disagree on this clip?" by reading the six
per-pass parquets and the matching LS tracks. Builds directly on US1 — same workflow call,
different output slice.

**Independent test**: Run with the default model set (≥2 ASR, 2 diar, 2 scene); verify six
parquets land under `<run_dir>/<pass>/uncertainty/` (3 axes × 2 passes); verify each row's
`contributing_models` lists the actual models that voted; verify the LS bundle has six
`<pass>__uncertainty__*` tracks plus the utterance TextArea sibling.

- [X] T020 [US2] Extend `compute_uncertainty_axes` in `src/senselab/audio/workflows/audio_analysis/__init__.py` to also emit per-pass results (currently US1 only handled raw_vs_enhanced). The same per-axis aggregation runs once per pass with that pass's vote dicts.
- [X] T021 [US2] Update the `scripts/analyze_audio.py` wiring to write the six per-pass parquets to `<run_dir>/<pass>/uncertainty/{presence,identity,utterance}.parquet`.
- [X] T022 [US2] Implement LS-bundle integration in `src/senselab/audio/workflows/audio_analysis/labelstudio.py`. API: `attach_uncertainty_tracks_to_ls(ls_tasks, ls_config, axis_results)` adding three Labels tracks per pass + three `pass_pair__*` tracks + utterance TextArea sibling tracks per `contracts/ls-bundle.md`.
- [X] T023 [P] [US2] Add the bin-mapping helper `_uncertainty_to_label_bin(u: float, status: str) -> str` in `labelstudio.py` (returns one of `low`/`medium`/`high`/`incomparable`/`unavailable`).
- [X] T024 [US2] Wire `scripts/analyze_audio.py` to call `attach_uncertainty_tracks_to_ls(...)` after writing the parquets and before `write_json` of the LS bundle.
- [X] T025 [P] [US2] Add tests for LS attachment in `src/tests/audio/workflows/audio_analysis/test_labelstudio.py`. Cover: bin mapping at the threshold edges (0.33, 0.66); track-name format; TextArea sibling carrying transcript consensus; label-set is the fixed 5-value enum.

---

## Phase 5 (US3, P2): Cross-stream contributions to the three axes

**Goal**: Verify the cross-stream sanity checks from spec US3 are *integrated* into the
three axes — not as separate parquets. Concretely: ASR token-overlap and AST/YAMNet
Speech-allowlist contribute to presence votes; ASR-vs-PPG PER contributes to utterance.

**Independent test**: Run on a clip where pyannote says silence in [12.0, 12.3] but Whisper
returns "hello" with timestamps in that range; verify the presence row at the 12.5-bucket
has `pyannote.speaks=False` AND `whisper-turbo.speaks=True` in `model_votes`, with
`aggregated_uncertainty` reflecting the disagreement.

- [X] T026 [US3] Verify the ASR token-overlap path in `src/senselab/audio/workflows/audio_analysis/presence.py` correctly resolves text-only ASR through the alignment block per FR-011. Add a regression test in `src/tests/audio/workflows/audio_analysis/test_compute_uncertainty_axes.py` that builds a Granite-style text-only result + a successful alignment block and asserts the presence votes split between the diar (silent) and the ASR (speech) signals.
- [X] T027 [US3] Verify the AST/YAMNet Speech-allowlist projection onto the bucket grid in `src/senselab/audio/workflows/audio_analysis/presence.py` uses floor-based window indexing (`win_idx = floor(start / native_hop)`); add a regression test covering AST 10.24 s → 0.5 s grid mapping.
- [X] T028 [US3] Verify the ASR-vs-PPG PER contribution to utterance in `src/senselab/audio/workflows/audio_analysis/utterance.py`: when PPG is provisioned, each ASR's `phoneme_per_to_ppg` is computed via `senselab.audio.workflows.audio_analysis.harvesters.g2p_phonemes` + `phoneme_per` (T013b); when PPG is absent, the field is null and the sub-signal drops out cleanly. Test both branches in `src/tests/audio/workflows/audio_analysis/test_compute_uncertainty_axes.py`.
- [X] T028b [US3] Add a graceful-degrade integration test in `src/tests/audio/workflows/audio_analysis/test_compute_uncertainty_axes.py` covering FR-013: build a `passes` summary where (a) one diar model has `status="failed"`, (b) one ASR model has `status="ok"` but empty `result`, (c) PPG block is absent. Call `compute_uncertainty_axes`; assert the function returns 9 AxisResults without raising; assert affected rows carry `comparison_status` ∈ {"incomparable", "unavailable", "one_sided"} with the failure reasons surfacing in the `incomparable_reasons` dict the function returns alongside the axis_results.
- [X] T029 [P] [US3] Add a multi-AudioSet-label-survival test in `src/tests/audio/workflows/audio_analysis/test_compute_uncertainty_axes.py` confirming the AST/YAMNet allowlist parser handles `"Narration, monologue"` etc. correctly (regression for the comma-string parsing bug).

---

## Phase 6 (US4, P3): Disagreements index + timeline plot

**Goal**: Reviewer can find the worst N buckets across the entire run via
`disagreements.json` and at-a-glance read all three uncertainty axes via `timeline.png`.

**Independent test**: Run on a clip with at least one high-uncertainty bucket per axis;
verify `disagreements.json` ranks them with `aggregated_uncertainty desc` and tiebreaks by
axis priority (utterance > identity > presence) then start; verify `timeline.png` is a
5-row figure (PNG present, ≥ 30 KB, dimensions match the expected layout).

- [X] T030 [US4] Implement `build_disagreements_index(axis_results, top_n, run_dir, *, config, incomparable_reasons, models_without_native_signal) -> dict` in `src/senselab/audio/workflows/audio_analysis/disagreements.py`. Output shape per `contracts/disagreements.json.md`: aggregates across the 9 axis results, ranks by `aggregated_uncertainty desc` with axis-priority + start tiebreaks, truncates to `top_n`.
- [X] T031 [US4] Wire `scripts/analyze_audio.py` to call `build_disagreements_index(...)` after the per-axis writes and write its return value to `<run_dir>/disagreements.json`. Skip when `--disagreements-top-n 0`.
- [X] T032 [US4] Implement `build_aligned_timeline_plot(run_dir, axis_results, *, save_path) -> Path | None` in `src/senselab/audio/workflows/audio_analysis/plot.py`. Five rows: presence (raw solid + enhanced dashed in [0, 1]); identity (same overlay); utterance (same overlay); raw-vs-enhanced delta strip with one band per axis; reference (raw diar speakers + raw ASR token spans). Replaces the in-script v1 implementation.
- [X] T033 [US4] Wire `scripts/analyze_audio.py` to call `build_aligned_timeline_plot(...)` after the disagreements index and write to `<run_dir>/timeline.png`. Skip cleanly when no axis_results were emitted.
- [X] T034 [P] [US4] Add tests for the disagreements index in `src/tests/audio/workflows/audio_analysis/test_disagreements.py`: ranking + axis-priority tiebreak + `top_n=0` opt-out + `top_n` truncation.
- [X] T035 [P] [US4] Add a smoke test for the timeline plot in `src/tests/audio/workflows/audio_analysis/test_plot.py` that builds a tiny `axis_results` dict + summary, calls `build_aligned_timeline_plot`, and asserts the output PNG exists with `> 5000 bytes`.

---

## Phase 7: Polish & Cross-Cutting Concerns

- [X] T036 Delete the v1 plot helpers (`_color_for_speaker`, the matplotlib code paths) and any v1 LS-bundle helpers now subsumed by the workflow module from `scripts/analyze_audio.py`. Confirm `wc -l scripts/analyze_audio.py` drops by ≥ 800 LOC vs. the pre-pivot baseline.
- [X] T037 [P] Update `CLAUDE.md` (project root) "Audio analysis script + ASR backend extensions" section to point at the new `senselab.audio.workflows.audio_analysis` API and add a brief example of calling `compute_uncertainty_axes` standalone (without the script).
- [X] T038 [P] Add a tutorial notebook at `tutorials/audio/uncertainty_axes.ipynb` demonstrating: load a cached run's `passes` summary → call `compute_uncertainty_axes` → print top-5 uncertain buckets per axis → render the timeline plot. Mirrors the structure of existing `tutorials/audio/*.ipynb` and is wired into the papermill notebook CI per the existing project conventions.
- [X] T039 Run `uv run pytest src/tests/audio/workflows/audio_analysis/ src/tests/scripts/analyze_audio_test.py -q` and confirm all green; fix any regressions before moving on.
- [X] T040 [P] Run `uv run ruff check`, `uv run ruff format`, `uv run mypy scripts/analyze_audio.py src/senselab/audio/workflows/audio_analysis/`, `uv run codespell` across changed files — confirm clean.
- [X] T041 E2E validation: rerun `uv run python scripts/analyze_audio.py artifacts/amie_sample_12apr2007/twin-1.wav` (the same clip from the prior validation note); record per-axis bucket counts, top-10 disagreements.json entries, plot screenshot in `artifacts/compare_uncertainty_e2e_validation.md` (replace the v1 section).
- [X] T041b SC-003 + SC-004 measurement: in the same E2E note, record (a) the cache-hit rate on a second-pass rerun of the same clip — every axis should report `cache="hit"` and the rate should be ≥ 95 % (per SC-003); (b) the wall-clock delta with comparator stage on vs. `--skip comparisons` for the same upstream-cached run, asserting ≤ 30 % overhead (per SC-004). Document the numbers and the wall-clock measurement methodology.
- [X] T042 Update `specs/20260508-173136-compare-uncertainty/checklists/requirements.md` — verify all checklist items pass against the rebuilt spec/plan/research artifacts.

---

## Dependency graph

```text
Phase 1 (Setup)
        │
        ▼
Phase 2 (Foundational) — T004…T009 must all complete before any user story
        │
        ▼
        ├─────────────────────────┬──────────────────────┬─────────────┐
        ▼                         ▼                      ▼             ▼
Phase 3 (US1, P1)           Phase 4 (US2, P2)     Phase 5 (US3, P2)   Phase 6 (US4, P3)
T010…T019                    T020…T025            T026…T029           T030…T035
   │                           │                    │                   │
   └──────────┬────────────────┴────────────────────┴───────────────────┘
              ▼
Phase 7 (Polish): T036…T042
```

US2, US3, US4 all depend on the workflow module and `compute_uncertainty_axes` from US1.
US3 is a *verification* phase (it just confirms the maximally-inclusive contribution policy
works) — its tasks are tests, not new code paths. US2 and US4 can run in parallel after
US1 lands (different files: per-pass parquet writes + LS attachment for US2 vs. disagreements
+ plot for US4).

## Parallel execution examples

Within Phase 2 (after T001/T002 land):

```text
T004 (grid.py) [P]   T005 (aggregators.py) [P]   T006 (types.py) [P]   ⟶ all 3 in parallel
T007 (cache.py)                                                         ⟶ depends on T006
T008 (tests) [P]     T009 (delete v1 helpers) [P]                       ⟶ independent of T004-T007
```

Within Phase 3 (after Phase 2):

```text
T010 (embeddings.py) [P]   T011 (presence.py) [P]   T012 (identity.py) [P]   T013 (utterance.py) [P]
T014 (aggregate.py)                                                                ⟶ depends on T011-T013
T015 (__init__.py orchestrator)                                                    ⟶ depends on T010, T014
T016 (tests) [P]     T017 (io.py) [P]                                              ⟶ depends on T015
T018 (script wiring)                                                               ⟶ depends on T017
T019 (e2e smoke test) [P]                                                          ⟶ depends on T018
```

Within Phase 6 (after Phase 3):

```text
T030 (disagreements.py) [P]   T032 (plot.py) [P]
T031 (script wiring)                          ⟶ depends on T030
T033 (script wiring)                          ⟶ depends on T032
T034 (tests) [P]   T035 (tests) [P]
```

## Implementation strategy

**MVP scope** — deliver Phase 1 + Phase 2 + Phase 3 (US1 only). At that point the script
emits the three raw-vs-enhanced parquets, the workflow module is importable from senselab,
and a reviewer can answer "did enhancement help?" via either the parquets or the LS tracks.
Phases 4–6 layer on incremental value but US1 alone is a complete, shippable cut.

**Incremental delivery**: small commits at each task boundary per Constitution §III. The
strict ordering (Phase 1 → 2 → 3 → 4/5/6 → 7) means CI stays green at each commit boundary
because each phase's artifacts are independently testable.

**Test-first only where it pays**: T016, T019, T029, T034, T035 are written *with* the
implementation tasks they validate (tests live alongside, not before, since the workflow
module is under active design). T008 is written before T004-T007 are wired into the
workflow because the grid / aggregators / cache_key are pure helpers with stable contracts.

## Format validation

All 42 tasks above follow the strict checklist format:

- ✅ start with `- [ ]`
- ✅ have a sequential ID (T001…T042)
- ✅ user-story-phase tasks carry `[USn]` labels
- ✅ parallel-eligible tasks carry the `[P]` marker
- ✅ all tasks reference an exact file path

Setup, Foundational, and Polish phase tasks have NO story label per the format rules.
