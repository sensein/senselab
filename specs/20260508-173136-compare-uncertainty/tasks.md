# Tasks: Comparison & Uncertainty Stage for analyze_audio.py

**Input**: Design documents from `/specs/20260508-173136-compare-uncertainty/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Smoke / unit test tasks ARE included because the spec requires them (SC-006: "Smoke tests cover at least one happy-path scenario for each comparison kind ... plus the four documented degradation modes; tests run in under 30 seconds total without venv provisioning or model downloads").

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1 / US2 / US3 / US4)

## Path Conventions

All comparator code lives in `scripts/analyze_audio.py` (single CLI script, per plan §"Structure Decision"). Tests live in `src/tests/scripts/analyze_audio_test.py`. Spec artefacts live under `specs/20260508-173136-compare-uncertainty/`.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Bring the dev environment in line with the design before any comparator code lands.

- [X] T001 Update /Users/satra/software/sensein/senselab/pyproject.toml in a single edit: (a) add `g2p_en` to the `[nlp]` extra (research.md §3), (b) confirm `jiwer >= 3.0` is already in the `[nlp]` extra and add it if missing (research.md §4), (c) extend `[tool.codespell].ignore-words-list` with `g2p` and `g2p_en` so the new references don't trip codespell on CI. Then run `uv lock` to refresh /Users/satra/software/sensein/senselab/uv.lock.
- [X] T002 Run `uv sync --extra nlp --group dev` to install the new G2P dep into the active venv

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: The plumbing every user story shares — cache key, ComparisonRow schema, parquet writer, LS XML/JSON helpers, CLI flag wiring. None of these alone produce reviewer-visible output; together they make all four user stories possible.

- [X] T005 Add the new comparator CLI flags to `parse_args` in /Users/satra/software/sensein/senselab/scripts/analyze_audio.py per /Users/satra/software/sensein/senselab/specs/20260508-173136-compare-uncertainty/contracts/cli.md: `--skip-comparisons`, `--cross-stream-win-length`, `--cross-stream-hop-length`, `--uncertainty-aggregator`, `--phoneme-disagreement-threshold`, `--speech-presence-labels`, `--asr-reference-model`, `--diarization-boundary-shift-ms` (default `50.0`; per Constitution §VIII), `--disagreements-top-n`, plus `comparisons` as an accepted value of the existing `--skip` flag. Add argparse validation per cli.md "Validation" section.
- [X] T006 Add `_CACHE_SCHEMA_VERSION = 3` (bumping from 2 introduced in PR #510) and a `comparison_cache_key()` helper to /Users/satra/software/sensein/senselab/scripts/analyze_audio.py per research.md §6 — sha256 over `(audio_signature, comparison_kind, task_or_pair, sorted upstream_cache_keys, comparator_params, wrapper_hash, senselab_version, schema_version)`. Reuse the existing `cache_lookup` / `cache_store` from PR #510.
- [X] T007 [P] Define the `ComparisonRow` schema and per-comparator extra column lists as module-level pyarrow schemas inside /Users/satra/software/sensein/senselab/scripts/analyze_audio.py per /Users/satra/software/sensein/senselab/specs/20260508-173136-compare-uncertainty/contracts/comparison-row.parquet.md (`COMMON_SCHEMA`, `ASR_VS_ASR_EXTRA`, `CLASSIFICATION_VS_CLASSIFICATION_EXTRA`, `DIAR_VS_DIAR_EXTRA`, `ASR_VS_DIAR_EXTRA`, `CLASS_VS_DIAR_EXTRA`, `ASR_VS_PPG_EXTRA`).
- [X] T008 [P] Add a `write_comparison_parquet(rows, schema, dest_path, provenance)` helper to /Users/satra/software/sensein/senselab/scripts/analyze_audio.py that writes the rows + the `comparator_provenance` Parquet-level metadata blob per comparison-row.parquet.md "Provenance metadata" section.
- [X] T009 Add a `ComparisonGrid` dataclass + `_iter_grid(duration_s, win, hop)` generator to /Users/satra/software/sensein/senselab/scripts/analyze_audio.py per /Users/satra/software/sensein/senselab/specs/20260508-173136-compare-uncertainty/data-model.md "ComparisonGrid" section. Yields `(start, end, idx)` tuples that respect `start + win <= duration_s + epsilon`.
- [X] T010 [P] Add a `_token_overlaps_window(scriptline, win_start, win_end)` helper to /Users/satra/software/sensein/senselab/scripts/analyze_audio.py that walks `result.chunks` (or fallback to top-level `start`/`end`) and returns True if any token's range overlaps the window. Tolerant to both Pydantic objects and JSON-restored dicts (reuse the `_seg_attr` helper from PR #510). This is the core building block for the Q4 clarification.
- [X] T011 [P] Add an `_aggregate_uncertainty(confidences: list[float | None], aggregator: str)` helper to /Users/satra/software/sensein/senselab/scripts/analyze_audio.py implementing `min` / `mean` / `harmonic_mean` / `disagreement_weighted` per cli.md. Drops None entries before aggregating; returns None when no signals are available.
- [X] T012 [P] Add a `_build_disagreements_index(parquet_paths, top_n, aggregator)` helper to /Users/satra/software/sensein/senselab/scripts/analyze_audio.py that reads each parquet, sorts by `combined_uncertainty` (NaN-last), applies the tie-breaker hierarchy from /Users/satra/software/sensein/senselab/specs/20260508-173136-compare-uncertainty/contracts/disagreements.json.md "Ordering rules", and emits the JSON shape from that contract.
- [X] T013 Add `_compare_ls_label_region(...)` and `_compare_ls_textarea_region(...)` helpers to /Users/satra/software/sensein/senselab/scripts/analyze_audio.py mirroring the existing `_ls_label_region` / `_ls_textarea_region` so comparator regions land in the LS bundle with the correct `from_name` / `id` per /Users/satra/software/sensein/senselab/specs/20260508-173136-compare-uncertainty/contracts/ls-bundle.md.
- [X] T014 Extend `build_labelstudio_config` and `build_labelstudio_task` in /Users/satra/software/sensein/senselab/scripts/analyze_audio.py to append the new `<Labels>` (and ASR-pair `<TextArea>`) blocks per ls-bundle.md "XML config additions" and "Tasks JSON additions". A run with no comparator outputs MUST produce bit-identical LS bundle output to current main (FR-005, SC-005) — verify by snapshotting before/after.
- [X] T015 [P] Add a smoke test `test_comparator_cli_flags_parse` to /Users/satra/software/sensein/senselab/src/tests/scripts/analyze_audio_test.py asserting that `parse_args` accepts the new flags with their documented defaults and rejects out-of-range values.
- [X] T016 [P] Add a smoke test `test_comparator_skip_no_op_preserves_outputs` to /Users/satra/software/sensein/senselab/src/tests/scripts/analyze_audio_test.py that asserts a synthetic `summary` produces an identical LS bundle whether `--skip comparisons` is set or not (uses fixture data, no model invocations) — this anchors SC-005.

**Checkpoint**: Foundational phase complete. None of the user stories have shipped yet, but all four are unblocked.

---

## Phase 3: User Story 1 — Spot raw-vs-enhanced disagreements at a glance (P1)

**Story Goal**: A reviewer running `analyze_audio.py` gets a per-(task, model) timeline of every region where raw and enhanced passes disagreed.

**Independent Test**: Run the script twice on the same audio (one `--no-enhancement`, one default). The first run produces no comparisons; the second emits `<run_dir>/comparisons/raw_vs_enhanced/...` parquets plus matching LS Labels-track regions on the timeline. Reviewer can scrub and see the disagreements highlighted.

- [X] T017 [US1] Add `compare_raw_vs_enhanced(passes, grid, args)` to /Users/satra/software/sensein/senselab/scripts/analyze_audio.py iterating each task × model present in BOTH passes; emits one parquet per task / model under `<run_dir>/comparisons/raw_vs_enhanced/<task>/<model>.parquet` per /Users/satra/software/sensein/senselab/specs/20260508-173136-compare-uncertainty/contracts/comparison-row.parquet.md.
- [X] T018 [US1] Implement the per-task differencer registry inside `compare_raw_vs_enhanced` in /Users/satra/software/sensein/senselab/scripts/analyze_audio.py: `_diff_diarization_passes`, `_diff_classification_passes`, `_diff_asr_passes`, `_diff_alignment_passes`, `_diff_features_passes`. Each maps a (raw_result, enhanced_result, grid) pair to a list of ComparisonRow dicts. The diarization differencer takes ``args.diarization_boundary_shift_ms`` (default 50 ms, configurable per Constitution §VIII) as the boundary-shift threshold; label-flip for classification compares top-1; ASR diff uses jiwer.wer per bucket on the cross-stream grid.
- [X] T019 [US1] Wire `compare_raw_vs_enhanced` into `main()` in /Users/satra/software/sensein/senselab/scripts/analyze_audio.py after both passes finish; gate on `"raw_vs_enhanced" not in args.skip_comparisons and "comparisons" not in args.skip`. Cache each parquet via `comparison_cache_key`.
- [X] T020 [US1] Wire the LS export in /Users/satra/software/sensein/senselab/scripts/analyze_audio.py: for every emitted raw_vs_enhanced parquet, declare a `pass_pair__compare__<task>__<model>` Labels track in the XML and emit Label regions for each row where `agree == False`, capped at the disagreements top-N. ASR rows additionally emit a paired TextArea region carrying WER + both transcripts per ls-bundle.md.
- [X] T021 [P] [US1] Add smoke test `test_raw_vs_enhanced_diarization_diff` to /Users/satra/software/sensein/senselab/src/tests/scripts/analyze_audio_test.py with a hand-built two-pass diarization fixture where pyannote raw covers [0, 10] and enhanced covers [0, 5] + [6, 10]; assert the parquet has a row for the [5, 6] bucket with `agree=False`, `mismatch_type="boundary_shift"`.
- [X] T022 [P] [US1] Add smoke test `test_raw_vs_enhanced_asr_text_diff` to /Users/satra/software/sensein/senselab/src/tests/scripts/analyze_audio_test.py with synthetic Whisper-style ScriptLine results in raw vs enhanced; assert ASR-vs-ASR rows carry `wer`, `a_text`, `b_text`, and that bucketing on the cross-stream grid produces the expected number of rows.
- [X] T023 [P] [US1] Add smoke test `test_raw_vs_enhanced_handles_failed_pass` to /Users/satra/software/sensein/senselab/src/tests/scripts/analyze_audio_test.py covering edge case "a pass failed entirely" — passes one with `status="failed"` and asserts rows carry `comparison_status="incomparable"` rather than crashing (FR-010, US1 acceptance scenario 3).

**Checkpoint**: US1 ships independently — reviewer can answer "did enhancement help?" without US2/US3/US4.

---

## Phase 4: User Story 2 — Within-stream model disagreements (P2)

**Story Goal**: With multiple models per task (the default), the reviewer also sees per-pair within-stream disagreement tracks: pyannote vs Sortformer, Whisper vs Granite vs Canary-Qwen vs Qwen3-ASR (all C(n,2) pairs), AST vs YAMNet.

**Independent Test**: Run with the default model set; output bundle includes one parquet per (task, model_a, model_b) pair under `<pass>/comparisons/<task>/<a>_vs_<b>.parquet` plus matching LS tracks. The existing `<pass>/scene_agreement.json` is preserved (subset of the new output).

- [X] T024 [US2] Add `compare_within_stream(pass_summary, grid, args)` to /Users/satra/software/sensein/senselab/scripts/analyze_audio.py that, for each task with ≥2 successful models, iterates `itertools.combinations(models, 2)` and calls task-specific differencers (T018 helpers, generalized to take two arbitrary results rather than raw/enhanced).
- [X] T025 [US2] Implement `_diff_asr_within_stream(result_a, result_b, grid, reference_side)` in /Users/satra/software/sensein/senselab/scripts/analyze_audio.py honoring `--asr-reference-model` to pick the reference side; emits ASR-vs-ASR rows with `wer`, `a_text`, `b_text`, `reference_side`.
- [X] T026 [US2] Implement `_diff_classification_within_stream(result_a, result_b, grid)` in /Users/satra/software/sensein/senselab/scripts/analyze_audio.py emitting rows with `top1_a`, `top1_b`, `score_a`, `score_b`, `entropy_a`, `entropy_b`. The output MUST be a strict superset of the data currently in `<pass>/scene_agreement.json`; existing scene_agreement.json continues to be emitted unchanged (FR-005).
- [X] T027 [US2] Implement `_diff_diarization_within_stream(result_a, result_b, grid)` in /Users/satra/software/sensein/senselab/scripts/analyze_audio.py emitting rows with `speaks_a`, `speaks_b` per cross-stream-grid bucket.
- [X] T028 [US2] Wire `compare_within_stream` into `main()` in /Users/satra/software/sensein/senselab/scripts/analyze_audio.py running per pass; gate on `"within_stream" not in args.skip_comparisons`. Cache each parquet via `comparison_cache_key`. LS export per ls-bundle.md.
- [X] T029 [P] [US2] Smoke test `test_within_stream_asr_pair_emits_wer` in /Users/satra/software/sensein/senselab/src/tests/scripts/analyze_audio_test.py: two synthetic ASR results agree on bucket 0 and differ on bucket 1; assert `agree=True` and `wer=0.0` on row 0, `agree=False` and `wer>0` on row 1.
- [X] T030 [P] [US2] Smoke test `test_within_stream_single_model_no_op` in /Users/satra/software/sensein/senselab/src/tests/scripts/analyze_audio_test.py: pass `summary` containing one task with only one successful model; assert no parquet is written and no LS track appears (graceful no-op per US2 acceptance scenario 2).
- [X] T031 [P] [US2] Smoke test `test_within_stream_classification_superset_of_scene_agreement` in /Users/satra/software/sensein/senselab/src/tests/scripts/analyze_audio_test.py: AST + YAMNet on a matching grid; assert the new parquet contains every (start, end, agree) tuple that scene_agreement.json contains for the same grid.

**Checkpoint**: US2 ships independently — reviewer can audit cross-model disagreement without enabling cross-stream or uncertainty.

---

## Phase 5: User Story 3 — Cross-stream sanity checks (P2)

**Story Goal**: Surface regions where ASR/diarization, AST or YAMNet/diarization, or ASR/PPG should agree but don't.

**Independent Test**: On a clip with brief silence and brief speech, the cross_stream parquets flag (a) regions where Whisper text overlaps a window pyannote called silence, (b) regions where AST/YAMNet's top-1 is in the speech allowlist but diarization disagrees, (c) regions where the ASR-implied phoneme sequence diverges from the PPG sequence above the threshold.

- [X] T032 [US3] Add `compare_cross_stream(pass_summary, grid, args)` to /Users/satra/software/sensein/senselab/scripts/analyze_audio.py that runs three sub-comparators per pass and emits one parquet per (asr_model, diar_model) pair under `<pass>/comparisons/cross_stream/asr__<m>__vs__diarization__<d>.parquet`, etc.
- [X] T033 [US3] Implement `_compare_asr_vs_diarization(asr_result, diar_result, grid)` in /Users/satra/software/sensein/senselab/scripts/analyze_audio.py. For each grid bucket, set `asr_says_speech` via `_token_overlaps_window` (T010) and `diar_says_speech` via diarization-segment overlap; compute `agree`.
- [X] T034 [US3] Implement `_compare_classification_vs_diarization(class_result, diar_result, grid, allowlist)` in /Users/satra/software/sensein/senselab/scripts/analyze_audio.py. For each bucket, set `class_says_speech = top1 in allowlist`. Emit one parquet per (class_model, diar_model) pair.
- [X] T035 [US3] Implement `_compare_asr_vs_ppg(asr_result, ppg_result, grid, threshold)` in /Users/satra/software/sensein/senselab/scripts/analyze_audio.py. Use `g2p_en` to convert ASR transcript-overlap-with-bucket to ARPAbet; read the bucket's PPG argmax from the existing PPG output; compute `phoneme_per` via jiwer; set `phoneme_disagreement = phoneme_per >= threshold`. Skip with `phoneme_status="g2p_unsupported_language"` for non-English transcripts (research.md §3).
- [X] T036 [US3] Add a `--speech-presence-labels` parser + default constant `_DEFAULT_SPEECH_PRESENCE_LABELS = ("Speech", "Conversation", "Narration, monologue", "Female speech, woman speaking", "Male speech, man speaking", "Child speech, kid speaking", "Speech synthesizer")` to /Users/satra/software/sensein/senselab/scripts/analyze_audio.py per research.md §5.
- [X] T037 [US3] Wire `compare_cross_stream` into `main()` in /Users/satra/software/sensein/senselab/scripts/analyze_audio.py running per pass; gate on `"cross_stream" not in args.skip_comparisons`. PPG-vs-ASR sub-comparator gracefully skips when `args.skip` contains `ppgs` or no PPG result is available (records `comparison_status="unavailable"` with a reason in disagreements.json — FR-010, US3 acceptance scenario 4).
- [X] T038 [P] [US3] Smoke test `test_cross_stream_asr_speech_in_silent_window` in /Users/satra/software/sensein/senselab/src/tests/scripts/analyze_audio_test.py: synthetic Whisper output with a token at [12.0, 12.3]; pyannote output with no segment in that window. Assert the parquet has a row where `asr_says_speech=True`, `diar_says_speech=False`, `agree=False`.
- [X] T039 [P] [US3] Smoke test `test_cross_stream_classification_speech_allowlist` in /Users/satra/software/sensein/senselab/src/tests/scripts/analyze_audio_test.py: AST top-1 = "Speech" in window 0, "Music" in window 1; diarization speaks in window 1 only. Assert two disagreement rows.
- [X] T040 [P] [US3] Smoke test `test_cross_stream_ppg_unavailable_degrades_gracefully` in /Users/satra/software/sensein/senselab/src/tests/scripts/analyze_audio_test.py: pass `summary` with no PPG output; assert the ASR-vs-PPG parquet is empty (or absent) and `disagreements.json` records `comparison_status="unavailable"` with reason "PPG backend not provisioned".
- [X] T041 [P] [US3] Smoke test `test_cross_stream_phoneme_per_two_tier` in /Users/satra/software/sensein/senselab/src/tests/scripts/analyze_audio_test.py: synthetic ASR + PPG outputs producing `phoneme_per = 0.30` and `phoneme_per = 0.60`; assert both rows carry the continuous value but only the 0.60 row has `phoneme_disagreement=True` (default threshold 0.50, Q5 clarification).

**Checkpoint**: US3 ships independently — cross-stream sanity checks run with or without US4 uncertainty plumbing.

---

## Phase 6: User Story 4 — Per-region uncertainty (P3)

**Story Goal**: Every existing per-task region grows `confidence` and `uncertainty` columns from the underlying model's native signal where available; `disagreements.json` ranks regions by combined uncertainty.

**Independent Test**: Run the script. Each existing per-task output gains `confidence` / `uncertainty` columns; disagreements.json ranks the top-N by combined uncertainty using the configured aggregator.

- [ ] T042 [US4] Add `_extract_whisper_confidence(scriptline)` to /Users/satra/software/sensein/senselab/scripts/analyze_audio.py that converts `avg_logprob` and `no_speech_prob` from each Whisper chunk into per-chunk `confidence = 1 - exp(-avg_logprob)` and `no_speech_prob`. Confirm the HF pipeline kwarg surface needed (research.md §2 — may need `return_token_timestamps`/equivalent; verify against current `huggingface.py`).
- [ ] T043 [US4] [P] Add `_extract_classification_confidence(window_dist)` to /Users/satra/software/sensein/senselab/scripts/analyze_audio.py returning `(top1_score, entropy, margin_to_top2)` from a full per-window distribution dict (AST + YAMNet share shape).
- [ ] T044 [US4] [P] Add `_extract_mms_alignment_confidence(aligned_segment)` to /Users/satra/software/sensein/senselab/scripts/analyze_audio.py returning the per-segment trellis log-probability if the senselab forced-alignment output exposes it; otherwise return None and document the gap in disagreements.json.
- [ ] T045 [US4] Plumb the per-region confidence into the existing per-task outputs **in-memory only** (do not write the enriched regions back to the cache directory; the existing cache schema is preserved verbatim per FR-005 / SC-005). Extend the in-memory `summary["passes"][pass][task]["by_model"][model]["result"]` regions with `confidence` and `uncertainty` columns BEFORE the comparator stage runs, so within-stream and cross-stream rows pick them up automatically. The on-disk per-task JSON / parquet files emitted in earlier stages of `run_pass()` MUST NOT change shape. For models without a native signal, the in-memory columns are null with the model id added to `disagreements.json.missing_confidence_signals`.
- [ ] T046 [US4] Hook the `--uncertainty-aggregator` flag through `compare_raw_vs_enhanced`, `compare_within_stream`, and `compare_cross_stream` in /Users/satra/software/sensein/senselab/scripts/analyze_audio.py so every emitted row carries `combined_uncertainty` computed via `_aggregate_uncertainty` (T011).
- [ ] T047 [US4] Wire `_build_disagreements_index` (T012) into `main()` in /Users/satra/software/sensein/senselab/scripts/analyze_audio.py after all comparison stages finish; emit `<run_dir>/disagreements.json` per /Users/satra/software/sensein/senselab/specs/20260508-173136-compare-uncertainty/contracts/disagreements.json.md. Skip with `top_n=0`.
- [ ] T048 [P] [US4] Smoke test `test_uncertainty_whisper_avg_logprob_extracted` in /Users/satra/software/sensein/senselab/src/tests/scripts/analyze_audio_test.py: synthetic Whisper ScriptLine with `chunks[].avg_logprob = -0.5` and `no_speech_prob = 0.1`; assert `confidence ≈ 1 - exp(-0.5) ≈ 0.393` lands on the result.
- [ ] T049 [P] [US4] Smoke test `test_uncertainty_aggregator_min_default` in /Users/satra/software/sensein/senselab/src/tests/scripts/analyze_audio_test.py: rows with `confidence_a=0.9, confidence_b=0.4`; assert `combined_uncertainty = 1 - 0.4 = 0.6` under default `min` aggregator. Repeat for `mean`, `harmonic_mean`, `disagreement_weighted`.
- [ ] T050 [P] [US4] Smoke test `test_uncertainty_missing_signal_dropped_not_zero` in /Users/satra/software/sensein/senselab/src/tests/scripts/analyze_audio_test.py: row with `confidence_a=0.9, confidence_b=None`; assert `combined_uncertainty = 1 - 0.9 = 0.1` (None dropped, not treated as zero).
- [ ] T051 [P] [US4] Smoke test `test_disagreements_index_top_n_and_ordering` in /Users/satra/software/sensein/senselab/src/tests/scripts/analyze_audio_test.py: feed three synthetic parquets into `_build_disagreements_index` with `top_n=2`; assert the entries are sorted by combined_uncertainty descending and ties resolve via the documented mismatch_type priority + start ascending (disagreements.json.md "Ordering rules").

**Checkpoint**: US4 ships — the disagreements index now ranks every disagreement region from all three earlier user stories.

---

## Phase 7: Polish & Cross-Cutting

**Purpose**: Lint sweep, type-check, run the existing senselab test suite to confirm SC-005, full E2E on twin-1.wav, update CLAUDE.md.

- [ ] T052 Run `uv run ruff format scripts/analyze_audio.py src/tests/scripts/analyze_audio_test.py` then `uv run ruff check scripts/analyze_audio.py src/tests/scripts/analyze_audio_test.py`; fix any violations.
- [ ] T053 Run `uv run mypy scripts/analyze_audio.py`; fix any violations.
- [ ] T054 Run the targeted test suite `uv run pytest src/tests/scripts/ -v` and confirm all new comparator tests pass (≥27 passed counting the existing 12 + ≥15 new) within 30 seconds total (SC-006).
- [ ] T055 Run the full senselab test suite `uv run pytest --ignore=src/tests/scripts -x -q` to confirm SC-005 (no regression in existing senselab outputs from the additive `confidence`/`uncertainty` columns added in T045).
- [ ] T056 Run the script E2E on `~/Downloads/twin-1.wav` (cached upstream tasks): `uv run python scripts/analyze_audio.py ~/Downloads/twin-1.wav --output-dir artifacts/e2e_runs > "artifacts/e2e_runs/run_$(date +%Y%m%d-%H%M%S)_comparator.log" 2>&1`. Verify: parquets land under each pass's `comparisons/` subtree, `disagreements.json` exists at run-dir top, LS bundle contains the new tracks, comparator stage adds ≤30 % wall-clock overhead vs a `--skip comparisons` run (SC-004).
- [ ] T057 [P] Document the comparator stage in /Users/satra/software/sensein/senselab/CLAUDE.md by appending a new sub-section "Comparison & uncertainty stage" under the existing "Audio analysis script + ASR backend extensions" section, summarizing the new flags, the parquet/JSON/LS outputs, and a one-line link to the spec.
- [ ] T058 [P] Mark all completed tasks `[X]` in /Users/satra/software/sensein/senselab/specs/20260508-173136-compare-uncertainty/tasks.md.
- [ ] T059 Open a PR against `alpha` from this branch with the title "comparison & uncertainty stage for analyze_audio.py" and a summary linking the spec, plan, and key acceptance scenarios. Confirm CI green before requesting review.

---

## Dependencies

- Phase 1 (T001–T002) blocks Phase 2.
- Phase 2 (T005–T016) blocks all user-story phases.
- US1 (Phase 3, T017–T023) is independent of US2/US3/US4 once Phase 2 is done — it's the MVP.
- US2 (Phase 4, T024–T031) and US3 (Phase 5, T032–T041) both depend on Phase 2; they are independent of each other and of US1.
- US4 (Phase 6, T042–T051) augments US1+US2+US3 outputs but does not block their MVP delivery — they ship with `confidence` columns null until US4 plumbs them in. T045 specifically can land before US1 ships if we want every comparator row to carry confidence from day one; alternative plan (recommended): ship US1 first with null confidences, then layer US4 in.
- Phase 7 polish runs last.

## Parallel execution opportunities

Within Phase 2: T007, T008, T010, T011, T012 are all independent helpers in the same file but touch non-overlapping symbols — they can be drafted in parallel by separate agents and merged. T015 and T016 are independent test files.

Within US1: T021, T022, T023 are independent test cases.

Within US2: T029, T030, T031 are independent test cases.

Within US3: T038, T039, T040, T041 are independent test cases.

Within US4: T043, T044, T048, T049, T050, T051 are independent helpers/tests.

In Phase 7: T057 (CLAUDE.md) and T058 (tasks.md checkmarks) are independent of T052–T056 and each other.

## Implementation strategy (MVP first)

1. **Phase 1 + Phase 2 + Phase 3 (US1)** = MVP. After this lands, a reviewer can answer "did enhancement help here?" via the LS bundle. Confidence columns are null; that's fine for a usable MVP.
2. **Phase 4 (US2) and Phase 5 (US3) in parallel** — both add comparison parquets without disturbing US1. Either can ship first. Recommend US2 first because it requires no new dependency (G2P) and therefore no `[nlp]` extra change; US3 is gated on T001/T004.
3. **Phase 6 (US4)** layers uncertainty across everything and is the final piece that makes `disagreements.json` ranked rather than arbitrary.
4. **Phase 7 polish** wraps the whole thing for PR.

## Format validation

All tasks above follow `- [ ] T### [P?] [Story?] description with file path` per the strict format requirement. Total: 57 tasks (T003–T004 are intentionally unused — Phase 1 was collapsed during /speckit.analyze remediation). Per-story counts:
- Setup (Phase 1): 2
- Foundational (Phase 2): 12
- US1 (Phase 3): 7
- US2 (Phase 4): 8
- US3 (Phase 5): 10
- US4 (Phase 6): 10
- Polish (Phase 7): 8
