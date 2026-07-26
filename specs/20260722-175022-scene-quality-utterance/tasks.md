---
description: "Task list for scene-aware presence axis + improved utterance uncertainty"
---

# Tasks: Scene-aware presence axis + improved utterance uncertainty

**Input**: Design documents from `/specs/20260722-175022-scene-quality-utterance/`
**Prerequisites**: plan.md, spec.md, research.md (D1–D10), data-model.md, contracts/

**Tests**: INCLUDED — the spec's User Scenarios & Success Criteria are mandatory and constitution II requires encapsulated testing. Tests are written before implementation within each story.

**Organization**: Grouped by user story (US1–US5) for independent implementation and testing. Task IDs are sequential in execution order.

## Format: `[ID] [P?] [Story] Description with file path`

- **[P]**: parallelizable (different files, no incomplete-task dependency)
- Absolute repo root: `/Users/satra/software/sensein/senselab`

---

## Phase 1: Setup (Shared Infrastructure)

- [X] T001 Promote `librosa` from transitive to an explicit dependency in `pyproject.toml` (align with pinned `librosa==0.11.0`) and refresh `uv.lock` via `uv add librosa` (D8).
- [X] T002 [P] Scaffold new modules with docstrings + `__future__` imports and no logic yet: `src/senselab/audio/tasks/scene_quality/__init__.py`, `src/senselab/audio/tasks/scene_quality/brouhaha.py`, `src/senselab/audio/tasks/voice_activity_detection/frame_posteriors.py`, `src/senselab/audio/workflows/audio_analysis/quality.py`, `src/senselab/audio/workflows/audio_analysis/sound_sources.py`, `src/senselab/audio/workflows/audio_analysis/calibration.py`, and `src/senselab/audio/workflows/audio_analysis/data/` (dir).

---

## Phase 2: Foundational (Blocking Prerequisites)

**⚠️ CRITICAL**: The additive-output contract must exist before any story can emit its columns.

- [X] T003 Add new defaulted fields to `UncertaintyRow` in `src/senselab/audio/workflows/audio_analysis/types.py` per data-model.md §1 (`presence_confidence`, `presence_uncertainty`, `quality_snr`, `quality_clip`, `quality_reverb`, `quality_bandwidth`, `quality_uncertainty`, `src_speech`, `src_people`, `src_machine`, `src_environment`, `src_dominant`, `token_entropy`, `scene_quality_coupling`) — all `= None`, after existing defaulted fields (slots dataclass).
- [X] T004 Add matching `pa.array` columns (float64, `src_dominant` string) to `write_axis_parquet` in `src/senselab/audio/workflows/audio_analysis/io.py` per contracts/presence-parquet-columns.md, and extend the `comparator_provenance` metadata with per-axis grid params + model id/revision placeholders.
- [X] T005 [P] Foundational regression guard: run `uv run pytest src/tests/audio/workflows/audio_analysis/{compute_uncertainty_axes_test,disagreements_test,labelstudio_test,plot_test}.py` and confirm they still pass with the new all-null columns present (SC-008 baseline). Fix any reader that breaks on unknown columns.

**Checkpoint**: Rows/parquet can carry the new columns (all null) without breaking any existing consumer.

---

## Phase 3: User Story 1 - Per-bucket audio quality (Priority: P1) 🎯 MVP

**Goal**: `presence` parquet gains `quality_snr/clip/reverb/bandwidth` + `quality_uncertainty` per bucket.
**Independent Test**: run workflow on a clip; clean → all quality < 0.1; noised region → `quality_snr` up ≥0.3; clipped region → `quality_clip` up; telephone-band → `quality_bandwidth` up.

### Tests for User Story 1

- [X] T006 [P] [US1] Write `src/tests/audio/workflows/audio_analysis/quality_test.py`: synthetic fixtures (clean tone/speech; +broadband noise at known SNR; hard-clipped; low-pass ≤4 kHz) asserting each `quality_*` responds correctly and stays in `[0,1]`; assert `quality_uncertainty` rises when SNR estimators are made to disagree (SC-001, SC-002). Mock Brouhaha frames so the test is model-free.

### Implementation for User Story 1

- [X] T007 [P] [US1] Implement `scene_quality/brouhaha.py`: `extract_brouhaha_frames(...) -> list[BrouhahaFrames|None]` via `pyannote.audio.Model.from_pretrained` + `Inference`, loaded through `ensure_hf_model`/`local_files_only` + `get_huggingface_token()`, cached per `(repo,revision,device)`; returns per-frame `(vad, snr_db, c50_db)` + hop; `None` on load failure `(ImportError, RuntimeError)` (contracts/quality.md A; FR-025, FR-023).
- [X] T008 [US1] Implement `quality.py::harvest_quality_scores`: 0.5 s/0.25 s analysis window (constants), slice via the `embeddings.py:_slice_audio` + tail-anchored `_window_starts` pattern; `quality_snr` (Brouhaha SNR primary, cross-checked with `spectral_gating_snr_metric` + `peak_snr_from_spectral_metric`), `quality_clip` (`proportion_clipped_metric`), `quality_reverb` (Brouhaha C50), `quality_bandwidth` (`librosa.feature.spectral_rolloff` 85% vs Nyquist), `quality_uncertainty` (normalized SNR-estimator spread); broadcast analysis-window values onto presence buckets; raw values into `_raw` (contracts/quality.md B).
- [X] T009 [US1] Wire quality into `compute.py`: call `extract_brouhaha_frames` once per pass and `harvest_quality_scores` per presence bucket; populate the `quality_*` columns on presence `UncertaintyRow`s and stash `_raw` into `model_votes`. Guard behind `scene_quality: bool = True` kwarg.
- [X] T010 [US1] Null-safe degradation: when Brouhaha is `None`, emit `quality_reverb`/`quality_snr` null (keep `quality_clip`/`quality_bandwidth` from DSP), never abort (FR-023); add a targeted test case.
- [X] T011 [US1] Record quality analysis-window params + Brouhaha model id/revision in presence `AxisResult.provenance` (FR-015, memory `project_model_revision_pinning`).

**Checkpoint**: US1 fully functional and independently testable (the MVP).

---

## Phase 4: User Story 2 - Background sound-source categorization (Priority: P1)

**Goal**: `presence` parquet gains `src_speech/people/machine/environment` + `src_dominant`.
**Independent Test**: clip with background traffic → `src_machine` elevated; speech-only → `src_speech` dominates; coverage test green.

### Tests for User Story 2

- [X] T012 [P] [US2] Write `src/tests/audio/workflows/audio_analysis/sound_sources_test.py`: (a) `test_category_map_covers_all_classes` — load AST `id2label` (527) + vendored YAMNet class list (521), assert every class resolves to exactly one of the 4 categories with zero silent gaps (SC-003); (b) masses from a synthetic window sum to ~1 and `src_dominant == argmax`; (c) background-machine window → `src_machine` dominant.

### Implementation for User Story 2

- [X] T013 [P] [US2] Author `workflows/audio_analysis/data/audioset_source_map.json` (schema data-model.md §3): union of AST 527 + YAMNet 521 display names → `{speech,people,machine,environment}`, `default="environment"`, `version="1"`. Include a small vendored YAMNet class-name list if the model asset isn't importable in-process.
- [X] T014 [US2] Raise persisted scene classes: add `--scene-top-k` (default full 527/521) in `scripts/analyze_audio.py` and pass `top_k` to the AST + YAMNet `classify_audios(...)` windowed calls so full per-window distributions persist; verify top-1 consumers (`speech_presence_labels`, YAMNet veto) unaffected (contracts/sound_sources.md B).
- [X] T015 [US2] Implement `sound_sources.py::harvest_source_categories`: load + cache the map; per presence bucket, sum AST+YAMNet window `scores` into category masses (mean of available classifiers), normalize to ~1, `src_dominant = argmax`; project via center→nearest-window; log unmapped classes once each (contracts/sound_sources.md C).
- [X] T016 [US2] Wire `src_*` columns into `compute.py` presence rows behind `sound_sources: bool = True`; stash per-category class contributions into `model_votes`.
- [X] T017 [US2] Null-safe when both AST and YAMNet absent → all `src_*` null (FR-023); add test case.

**Checkpoint**: US1 + US2 both work independently on the presence parquet.

---

## Phase 5: User Story 3 - Fine temporal resolution for presence (Priority: P2)

**Goal**: per-axis grids + continuous frame posteriors with `presence_confidence`/`presence_uncertainty`.
**Independent Test**: `--presence-grid-win 0.1`; finer bucket count; hand-marked ≤50 ms onset localized within one hop (SC-004).

### Tests for User Story 3

- [X] T018 [P] [US3] Write `src/tests/audio/workflows/audio_analysis/frame_posteriors_test.py`: synthetic `FramePosterior`, assert `mean_posterior_in_window` returns correct mean + within-window std over overlapping frames; empty overlap → nan handled.
- [X] T019 [P] [US3] Write `src/tests/audio/workflows/audio_analysis/grid_test.py`: `presence_grid` default (0.1/0.02) yields expected bucket count vs shared grid; per-axis grids coexist; provenance records each.

### Implementation for User Story 3

- [X] T020 [P] [US3] Implement `frame_posteriors.py`: `extract_speech_frame_posteriors(...)` from `pyannote/segmentation-3.0` raw scores (`Model.from_pretrained`+`Inference`, max over speaker/powerset axis), `FramePosterior` dataclass, `mean_posterior_in_window` helper; `ensure_hf_model`/`local_files_only`; null-safe (contracts/frame_posteriors.md).
- [X] T021 [US3] Add `presence_grid: BucketGrid | None = None` to `compute_uncertainty_axes` (default 0.1/0.02), mirroring `utterance_grid` plumbing (resolve, thread into `harvest_presence_votes`, record in provenance).
- [X] T022 [US3] Add frame-posterior voters to `presence.py` per-bucket votes: seg-3.0 (T020) + Brouhaha VAD head (reuse T007 `BrouhahaFrames.vad`), each `mean_posterior_in_window` → `_vote_from_pvoice(mean, mean>=0.5)`; within-window std feeds uncertainty.
- [X] T023 [US3] Coarse-voter demotion in `presence.py`/`aggregate.py`: down-weight voters whose native resolution (via `_native_classification_grid`) is coarser than the reporting grid to a single context-prior term instead of an equal per-bucket vote (FR-014); add aggregate test.
- [X] T024 [US3] Surface `presence_confidence` (= `presence_p_voice`, already computed at `compute.py:259`) and `presence_uncertainty` (= `aggregate_presence`) as columns on presence rows; keep `aggregated_uncertainty` identical (FR-013, SC-008).
- [X] T025 [US3] Add `--presence-grid-win`/`--presence-grid-hop` to `scripts/analyze_audio.py`, wiring the new `presence_grid` (contracts/cli.md).

**Checkpoint**: US1–US3 independently functional; presence reports on the fine grid with the split.

---

## Phase 6: User Story 4 - Improved utterance uncertainty (Priority: P2)

**Goal**: overlap grid + token-level entropy + calibrated confidence + scene-quality coupling.
**Independent Test**: run with a Whisper model on a noisy two-speaker clip → `token_entropy` populated, boundary-straddle words excluded (SC-005), utterance uncertainty higher in the noisy region (SC-006).
**⚠️ Highest risk** (touches core `ScriptLine` + Whisper path) — see plan Complexity Tracking; splittable to a follow-up PR without blocking US1–US3.

### Tests for User Story 4

- [x] T026 [P] [US4] Write `src/tests/audio/tasks/speech_to_text/` token test: Whisper HF path with `output_scores` populates `ScriptLine.token_entropy`/`avg_logprob`/`no_speech_prob`; non-Whisper backend leaves them `None`.
- [x] T027 [P] [US4] Extend `src/tests/audio/workflows/audio_analysis/aggregate_test.py`: token-entropy sub-signal folds into `aggregate_utterance`; `scene_quality_coupling` multiplier raises reported uncertainty under poor quality; falls back exactly to today's signals when token entropy is `None`.

### Implementation for User Story 4

- [x] T028 [US4] Add optional fields `avg_logprob`, `no_speech_prob`, `token_entropy` (default `None`) to `ScriptLine` in `src/senselab/utils/data_structures/script_line.py` and map them in `from_dict` (data-model.md §6).
- [x] T029 [US4] In `src/senselab/audio/tasks/speech_to_text/huggingface.py` (Whisper), request `generate_kwargs={"return_dict_in_generate": True, "output_scores": True}`, compute per-token softmax entropy + `avg_logprob` + `no_speech_prob`, attach onto emitted `ScriptLine`s (contracts/utterance.md A). This also revives the dead `avg_logprob` presence/utterance reads.
  - **Deviation from contract A, as built**: passing `output_scores` through `pipe(generate_kwargs=...)` cannot work — transformers 5.5.4's `AutomaticSpeechRecognitionPipeline._forward` keeps only `sequences`/`token_timestamps` from the generate output and drops the scores before `postprocess`. Implemented instead as `speech_to_text/token_confidence.py`: a context manager that temporarily wraps `pipe.model.generate` to observe its output, leaving pipeline behavior untouched. It requests `output_logits` (raw) rather than `output_scores` (post-processor) because Whisper's logits processors suppress `<|nospeech|>` to `-inf`, which would pin `no_speech_prob` at 0 and distort the entropy.
- [x] T030 [US4] Add `mean_token_entropy_in_window` + a `token_entropy` vote field in `utterance.py::harvest_utterance_votes` (None-safe); confirm overlap grid + `fully_contained=True` boundary exclusion remain (SC-005).
- [x] T031 [US4] Fold the token-entropy sub-signal into `aggregate.py::aggregate_utterance` via the existing `--uncertainty-aggregator` mechanism (contracts/utterance.md C).
- [x] T032 [US4] Map ASR confidences (Whisper `avg_logprob`, CTC score) to a common calibrated `[0,1]` scale using the `CalibrationProfile.temperature` hook (FR-018).
- [x] T033 [US4] Compute + record `scene_quality_coupling` multiplier (from `quality_snr` + `src_machine`+`src_environment` over the bucket span, weights `w_q`/`w_s` params) in `compute.py`/`utterance.py`; apply to reported utterance uncertainty, keep pre-coupling value in `model_votes` (FR-019).
- [x] T034 [US4] Add `--utterance-scene-coupling-weights` CLI flag with documented defaults (contracts/cli.md; constitution VIII).

**Checkpoint**: US1–US4 independently functional.

---

## Phase 7: User Story 5 - Synthetic calibration (Priority: P3)

**Goal**: fit dB→[0,1] normalization + temperature from synthetic mixtures; persist profile + validation artifact.
**Independent Test**: run the helper; reported degradation increases monotonically with true SNR/RT60 (rank corr ≥0.9, SC-007); profile JSON + plot emitted.

### Tests for User Story 5

- [ ] T035 [P] [US5] Write calibration test in `src/tests/audio/workflows/audio_analysis/` (or `src/tests/scripts/`): synthetic SNR/RT60 sweep → estimators → assert monotonic reported-vs-true and profile round-trips (load/apply) (SC-007).

### Implementation for User Story 5

- [ ] T036 [P] [US5] Implement `calibration.py`: `CalibrationProfile` load/apply (`linear_db_to_unit` etc., data-model.md §5) with a documented default when no profile is present.
- [ ] T037 [US5] Implement `scripts/calibrate_scene_quality.py`: synthesize noise (white/pink, numpy) at an SNR sweep + convolve synthetic exponential-decay RIRs at an RT60 sweep (numpy, no new dep), run the quality estimators, fit normalization + temperature, persist to `data/scene_quality_calibration.json`; all inputs/outputs are CLI params with defaults (D9, constitution VIII).
- [ ] T038 [US5] Emit a reported-vs-true validation plot/table under `artifacts/` (FR-022).
- [ ] T039 [US5] Add `--calibration-profile` CLI flag (default bundled profile) and thread the profile into `harvest_quality_scores` + utterance calibration (contracts/cli.md).

**Checkpoint**: all five stories independently functional.

---

## Phase 8: Polish & Cross-Cutting Concerns

- [ ] T040 [P] FR-024: add `<pass>__presence__quality` and `<pass>__presence__sources` tracks to `labelstudio.py` (additive; existing `presence` track unchanged) + test in `labelstudio_test.py`.
- [X] T041 [P] FR-024: add optional quality + dominant-source rows to `plot.py` (existing 5 rows unchanged when new signals null) + test in `plot_test.py`.
- [ ] T042 [P] FR-024: rank the new presence sub-signals in `disagreements.py` + test in `disagreements_test.py`.
- [ ] T043 [P] Docs: update `workflows/audio_analysis/doc.md` and the CLAUDE.md workflow section describing the scene-aware presence columns + new CLI flags; cross-check against quickstart.md.
- [X] T044 Run quickstart.md end-to-end on a `tutorial_audio_files/` clip; confirm all new columns populate and acceptance scenarios hold.
- [ ] T045 `uv run ruff format && uv run ruff check --fix && uv run mypy . && uv run codespell` (memory `feedback_ruff_format`).
- [ ] T046 Full regression: `uv run pytest src/tests/audio/workflows/audio_analysis/` + confirm `aggregated_uncertainty` values unchanged vs baseline (SC-008).

---

## Dependencies & Execution Order

### Phase dependencies

- **Setup (P1)** → **Foundational (P2)** → **User Stories (P3–P7)** → **Polish (P8)**.
- Foundational T003/T004 block every story's column emission; do them first.

### Story dependencies

- **US1 (P1)** — MVP; depends only on Foundational. Introduces `scene_quality/brouhaha.py` (T007), reused by US3.
- **US2 (P1)** — depends only on Foundational; independent of US1 (different harvester file). Shares `compute.py` as an integration point with US1 (sequence T009 and T016 to avoid edit conflicts).
- **US3 (P2)** — seg-3.0 posteriors (T020) are independent, but the Brouhaha VAD voter (T022) reuses US1's T007. Do US1 before US3's T022.
- **US4 (P2)** — scene-quality coupling (T033) reuses US1's quality columns. Do US1 before US4's T033. Otherwise independent; highest risk, can be deferred to a follow-up PR.
- **US5 (P3)** — calibrates US1's estimators; do after US1.

### Within each story

- Tests (T006, T012, T018/19, T026/27, T035) written first and expected to fail.
- Model/loader before harvester before `compute.py` wiring before provenance.
- `compute.py` is a shared integration file: tasks that edit it (T009, T016, T021, T024, T033) are **not** mutually `[P]` — sequence them.

### Parallel opportunities

- Setup T002 ∥ (after T001).
- Across stories after Foundational: US1 and US2 harvesters (`quality.py` vs `sound_sources.py`) develop in parallel; only their `compute.py` wiring serializes.
- Within a story, `[P]` tasks touch distinct files: e.g. T006 (test) ∥ T007 (loader); T012 (test) ∥ T013 (map JSON); T018 ∥ T019 ∥ T020.

---

## Parallel Example: User Story 1

```bash
# Test + Brouhaha loader in parallel (distinct files):
Task: "Write quality_test.py with synthetic clean/noised/clipped/band-limited fixtures"
Task: "Implement scene_quality/brouhaha.py extract_brouhaha_frames"
# Then serialize the compute.py wiring:
Task: "Implement quality.py harvest_quality_scores"
Task: "Wire quality into compute.py presence rows"
```

---

## Implementation Strategy

### MVP first (US1 only)

1. Phase 1 Setup → 2. Phase 2 Foundational → 3. Phase 3 US1 → **STOP & VALIDATE** (`quality_*` columns on a real clip) → demo. This alone delivers the most-requested capability.

### Incremental delivery

US1 (quality) → US2 (sources) → US3 (temporal) → US4 (utterance) → US5 (calibration). Each merges as an independently testable increment; `aggregated_uncertainty` and existing consumers stay green throughout (SC-008).

### Risk containment

US4 is the only phase touching core `ScriptLine` + the Whisper path. If it destabilizes CI, split US4 to a follow-up PR — US1–US3 + US5 remain a complete, shippable increment.

---

## Notes

- `[P]` = different files, no incomplete dependency. `compute.py` edits never run `[P]` together.
- Commit after each task or logical group (constitution III); one focused commit per task ID where practical.
- No `print` — use `logger` (constitution anti-pattern #3). Mock isolation via `monkeypatch.setattr` (memory `feedback_mock_cache_pollution`). Optional-import guards use `(ImportError, RuntimeError)`.
- Gated pyannote models load via `ensure_hf_model` + `local_files_only` (constitution VI); pin revisions (memory `project_model_revision_pinning`).
- All commands via `uv run` (constitution I).
