# Tasks: Audio Analysis Script + ASR Backend Extensions + Forced Alignment

**Input**: Design documents from `/specs/20260506-154425-audio-analysis-asr-extensions/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/cli.md, quickstart.md

**Tests**: Plan calls for light smoke tests with `@pytest.mark.skipif` guards so default CI is unaffected. Test tasks included per phase.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story. Note that US6 (multilingual aligner) is a hard prerequisite for US4 (Canary-Qwen) and for Granite Speech (no separate story; covered as a validation target inside US6).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4, US5, US6)
- Include exact file paths in descriptions

---

## Phase 1: Setup

**Purpose**: Land the existing analyze_audio.py script as the baseline; declare the new optional dep.

- [x] T001 Commit the existing scripts/analyze_audio.py at /Users/satra/software/sensein/senselab/scripts/analyze_audio.py (already staged on this branch from prior session) as the starting baseline; this gives every later phase a stable reference point and validates that the script runs against the current senselab install.
- [x] T002 [P] Add `uroman>=1.3` to the `nlp` extra in /Users/satra/software/sensein/senselab/pyproject.toml. Used later by the MMS aligner for ja/zh romanization (US6). Do not add to base deps — the analyze_audio script itself does not import uroman; senselab's forced_alignment imports it lazily only when the MMS path is exercised on those languages.
- [x] T003 [P] Verify default `uv sync` (no extra) does not pull `uroman` and that all existing tests still collect (`uv run pytest --collect-only -q | tail -3`). Validates that the pyproject.toml change is non-breaking.

---

## Phase 2: Foundational

**Purpose**: Cache schema bump and dispatch-table additions that all later phases assume in place.

- [x] T004 Bump `_CACHE_SCHEMA_VERSION` from 1 to 2 in /Users/satra/software/sensein/senselab/scripts/analyze_audio.py and add a one-line comment explaining "v2 introduces alignment as a separate cache entry". This invalidates prior cache entries cleanly (intentional — old entries lack the new alignment shape).
- [x] T005 Extend the dispatch tables in /Users/satra/software/sensein/senselab/src/senselab/audio/tasks/speech_to_text/api.py: add `_CANARY_PREFIXES = ("nvidia/canary-",)`, `_QWEN_ASR_PREFIXES = ("Qwen/Qwen3-ASR",)`, and `_TIMESTAMP_LESS_HF_MODELS = ("ibm-granite/granite-speech-",)`. Add stub branches in `transcribe_audios` that route to the new modules but raise NotImplementedError with a clear "see canary_qwen.py/qwen.py" message — the actual modules land in US4/US5. Order: existing NeMo prefixes → Canary → Qwen-ASR → HF default. This task lands the dispatch shape so US4 and US5 can be implemented independently without re-touching api.py.

**Checkpoint**: Foundational dispatch in place; the script runs unchanged against existing models; cache schema bump invalidates v1 entries on next run.

---

## Phase 3: User Story 1 - One-Command End-to-End Audio Analysis (Priority: P1) 🎯 MVP

**Goal**: A single command runs the full senselab task suite on an input audio file with and without enhancement, producing per-(pass × task × model) JSON outputs and a top-level summary.

**Independent Test**: `uv run python scripts/analyze_audio.py path/to/audio.wav` produces `<output-dir>/<stem>_<timestamp>/summary.json` plus per-task JSON files for both `raw_16k/` and `enhanced_16k/`. Per-model failures are captured as structured error records without aborting.

### Implementation for User Story 1

- [x] T006 [US1] Verify the existing scripts/analyze_audio.py satisfies User Story 1 acceptance scenarios end-to-end against a 30-second sample (any tutorial WAV under /Users/satra/software/sensein/senselab/tutorials/audio/) and document any gaps in artifacts/analyze_audio_us1_validation.md. The script's main flow (read → resample → run six tasks per pass → write summary) is largely working from the prior session; this task is about confirming the contract.
- [x] T007 [US1] Audit /Users/satra/software/sensein/senselab/scripts/analyze_audio.py against contracts/cli.md to confirm every documented option is implemented; add or update any missing flags. Per-model failure capture (status/error/traceback) and the summary.json shape must match the data-model.md ModelRun schema.
- [x] T008 [P] [US1] Write src/tests/scripts/analyze_audio_test.py at /Users/satra/software/sensein/senselab/src/tests/scripts/analyze_audio_test.py with smoke tests for: argparse default values, audio_signature stability over identical waveforms, cache_key changes when any keyed input changes, and the auto-align skip-condition logic for ScriptLines that already have timestamps. No model loads.

**Checkpoint**: US1 complete — the analyze_audio script can run end-to-end against any WAV and produce the documented output layout.

---

## Phase 4: User Story 2 - Re-Run Without Re-Computation (Priority: P1)

**Goal**: Re-running the script with identical inputs replays prior outputs from the content-addressable cache without invoking models. Changes to any one model/parameter invalidate only that one entry. ASR and alignment have independent cache entries.

**Independent Test**: Run the script twice on the same audio with identical args; verify the second run reports `cache: hit` for every successful task-model combination and completes in under 5% of the first run's wall-clock time.

### Implementation for User Story 2

- [x] T009 [US2] Verify the existing cache implementation in /Users/satra/software/sensein/senselab/scripts/analyze_audio.py satisfies the cache_key composition required by data-model.md (`sha256(audio_signature, task, model_id, params, wrapper_hash, senselab_version, schema_version)`). Confirm `cache_lookup`, `cache_store`, and `run_task_cached` all wire to that key.
- [x] T010 [US2] Add separate alignment-cache support in /Users/satra/software/sensein/senselab/scripts/analyze_audio.py — a new helper `align_cache_key(audio_sig, transcript_sha, language, aligner_model_id, aligner_params, wrapper_hash, senselab_ver)` and a `run_alignment_cached(...)` wrapper mirroring `run_task_cached`. The alignment cache entry includes a `parent_asr_cache_key` field for traceability. Per FR-024 and FR-026.
- [x] T011 [US2] Update the per-task JSON output and the summary.json so each entry includes `cache: hit|miss|disabled`, `cache_key`, and (on miss) a `provenance` block. For alignment outcomes, provenance includes `transcript_sha`, `language`, and `parent_asr_cache_key`. Update existing serializers in /Users/satra/software/sensein/senselab/scripts/analyze_audio.py.
- [ ] T012 [US2] Validate end-to-end: run the script, then re-run with identical args; confirm via the summary.json that every successful entry shows `cache: "hit"` on the second run. Document the timings in artifacts/analyze_audio_us2_validation.md.

**Checkpoint**: US2 complete — re-runs are cache-driven; ASR and alignment cache entries are independent.

---

## Phase 5: User Story 3 - Hierarchical Annotations Imported into Label Studio (Priority: P2)

**Goal**: The script produces a Label Studio import bundle (tasks JSON + labeling config XML) where each model's output appears as its own timeline track per audio variant.

**Independent Test**: Open the produced `labelstudio_config.xml` in Label Studio's labeling configuration editor; import the produced `labelstudio_tasks.json`; verify each task shows parallel timeline tracks for every analyzer × model combination.

### Implementation for User Story 3

- [x] T013 [US3] Verify the existing LS export in /Users/satra/software/sensein/senselab/scripts/analyze_audio.py emits one LS task per audio variant with parallel `from_name` tracks per analyzer × model. Confirm shape against contracts/cli.md.
- [x] T014 [US3] Update the LS export's ASR-region branch in /Users/satra/software/sensein/senselab/scripts/analyze_audio.py to handle three cases: (a) ASR with native timestamps → per-segment regions; (b) ASR text-only with successful alignment → per-aligned-segment regions; (c) ASR text-only with no alignment (or alignment failed) → single full-audio TextArea region. The shape change goes into `build_labelstudio_task()` and the helpers below it.
- [ ] T015 [US3] Validate by importing into a Label Studio sandbox project: run the script, paste `labelstudio_config.xml` into the project's labeling-config editor, import `labelstudio_tasks.json`, confirm each model produces its own timeline track. Document in artifacts/analyze_audio_us3_validation.md.
- [ ] T015a [US3] Verify FR-014 (`scene_agreement.json` emission) is preserved after the LS-export changes in T014. Run /Users/satra/software/sensein/senselab/scripts/analyze_audio.py with matching grids `--ast-win-length 0.96 --ast-hop-length 0.48` (overrides the per-model defaults so AST and YAMNet share YAMNet's native grid), confirm `<run-dir>/raw_16k/scene_agreement.json` is produced with `windows_compared > 0` and an `agreement_rate` field. With default grids (AST 10.24 / YAMNet 0.96) the file should be ABSENT — record both observations in artifacts/analyze_audio_us3_validation.md.

**Checkpoint**: US3 complete — LS bundles import cleanly, one track per (pass × analyzer × model); scene_agreement.json behavior is preserved.

---

## Phase 6: User Story 6 - Forced-Alignment Backends Cover Multilingual ASR Output (Priority: P2)

**Goal**: senselab's `align_transcriptions(audio, transcript, language)` API picks an appropriate aligner backend automatically for any of 1100+ languages. Granite Speech 3.3 (English + 7 translation languages) becomes runnable end-to-end.

**Independent Test**: Call `senselab.audio.tasks.forced_alignment.align_transcriptions(audio, transcript, language='ja')` on a Japanese transcript; receive timed segments without the caller specifying any aligner-model id.

### Implementation for User Story 6

- [x] T016 [US6] Add the MMS model registry to /Users/satra/software/sensein/senselab/src/senselab/audio/tasks/forced_alignment/constants.py: a `MMS_MODEL_ID = "facebook/mms-1b-all"` constant, an ISO-639-1 → ISO-639-3 map covering at minimum {en, fr, de, es, pt, it, ja, zh}, and a `LANGUAGE_TO_BACKEND` registry that defaults any language not in `DEFAULT_ALIGN_MODELS_HF` to MMS.
- [x] T017 [US6] Implement the new `model_type = "mms"` branch in /Users/satra/software/sensein/senselab/src/senselab/audio/tasks/forced_alignment/forced_alignment.py: in the model-loading section of `align_transcriptions`, after `Wav2Vec2ForCTC.from_pretrained(MMS_MODEL_ID)` call `processor.tokenizer.set_target_lang(iso3)` and `model.load_adapter(iso3)`. Cache loaded (model_id, iso3) pairs in a module-level dict to avoid reloading per call. Reuse the existing `_get_trellis`, `_backtrack`, `_merge_repeats`, and `_assign_timestamps` code unchanged — MMS produces compatible CTC posteriors.
- [x] T018 [US6] Add the romanization path in /Users/satra/software/sensein/senselab/src/senselab/audio/tasks/forced_alignment/forced_alignment.py: extend `_preprocess_segments` (or a new helper) to call `uroman` (lazily imported via `importlib.import_module`) on transcripts when `align_model_metadata["romanize"]` is True. Set `romanize=True` for ja/zh in the registry from T016. If `uroman` is not installed at call time, raise an actionable ImportError pointing at the `nlp` extra.
- [x] T019 [US6] Modify /Users/satra/software/sensein/senselab/src/senselab/audio/tasks/speech_to_text/huggingface.py to accept a `return_timestamps: bool = True` keyword. When False, omit `return_timestamps` from the underlying `pipeline(...)` call so timestamp-less HF models (Granite Speech) don't trigger the pipeline's safety check.
- [x] T020 [US6] Wire the timestamp-less default in /Users/satra/software/sensein/senselab/src/senselab/audio/tasks/speech_to_text/api.py: when the model id matches `_TIMESTAMP_LESS_HF_MODELS` (added in T005), pass `return_timestamps=False` to the HF backend by default. Caller can still override.
- [x] T021 [P] [US6] Wire the script's auto-align stage in /Users/satra/software/sensein/senselab/scripts/analyze_audio.py: after the ASR family completes, iterate each ASR ModelRun. Skip when `--no-align-asr` is set or when the ASR result already has timestamps (any ScriptLine with `start is not None` or non-empty `chunks`). Otherwise call `align_transcriptions(audio, transcript, language=resolved_language)` via `run_alignment_cached(...)`. On failure, preserve the ASR text and mark the alignment as failed in its own JSON. Add the new flags `--no-align-asr`, `--aligner-model`, `--asr-language` per contracts/cli.md.
- [x] T022 [P] [US6] Write src/tests/audio/tasks/forced_alignment/mms_test.py at /Users/satra/software/sensein/senselab/src/tests/audio/tasks/forced_alignment/mms_test.py with a `@pytest.mark.skipif` guard checking whether `facebook/mms-1b-all` is in the local HF cache. When present, run alignment on the existing real-speech fixture /Users/satra/software/sensein/senselab/src/tests/data_for_testing/audio_48khz_mono_16bits.wav (resample to 16 kHz mono first via senselab) with a plausible English transcript, then exercise the romanize path with a synthetic Japanese transcript against the same audio. Assertions are SHAPE-ONLY: `result` is `List[List[ScriptLine | None]]`; per-segment ScriptLines have `start`/`end` populated; chunks list non-empty for at least one segment. Do NOT assert transcription accuracy or alignment quality (synthetic transcripts on real audio will not align meaningfully — the test verifies the API contract, not the model's accuracy).
- [x] T023 [P] [US6] Write src/tests/audio/tasks/speech_to_text/huggingface_no_timestamps_test.py at /Users/satra/software/sensein/senselab/src/tests/audio/tasks/speech_to_text/huggingface_no_timestamps_test.py with a `@pytest.mark.skipif` guard checking transformers availability. Verify that `transcribe_with_huggingface(audios, model, return_timestamps=False)` runs a small CTC model (use a tiny one already in tests, not Granite) without raising; verify the resulting ScriptLine list has `start=None`/`end=None`/empty chunks.
- [ ] T024 [US6] Validate end-to-end by running the analyze_audio script with `--asr-models ibm-granite/granite-speech-3.3-8b openai/whisper-large-v3-turbo` against a sample audio. Confirm Granite produces text via the senselab HF path (no timestamp-pipeline error), the script's auto-align stage adds segments via MMS, and the LS export contains per-segment regions for Granite. Document in artifacts/analyze_audio_us6_validation.md.

**Checkpoint**: US6 complete — multilingual aligner works; Granite Speech runnable end-to-end with auto-aligned segments.

---

## Phase 7: User Story 4 - Use NVIDIA Canary-Qwen 2.5B as a Senselab ASR Model (Priority: P2)

**Goal**: Pass `nvidia/canary-qwen-2.5b` to senselab's `transcribe_audios` and have it run successfully via a new NeMo subprocess venv. The script's auto-align stage adds timestamps via MMS (US6 prerequisite).

**Independent Test**: `senselab.audio.tasks.speech_to_text.transcribe_audios([audio], model=HFModel(path_or_uri="nvidia/canary-qwen-2.5b"))` returns `[ScriptLine(text="...", start=None, end=None, ...)]`.

### Implementation for User Story 4

- [X] T025 [US4] Create /Users/satra/software/sensein/senselab/src/senselab/audio/tasks/speech_to_text/canary_qwen.py mirroring the pattern in /Users/satra/software/sensein/senselab/src/senselab/audio/tasks/speech_to_text/nemo.py. Constants: `_VENV_NAME = "nemo-canary-qwen"`, requirements `nemo_toolkit[asr,tts]` from a pinned NeMo trunk ref. Class `CanaryQwenASR` with method `transcribe_with_canary_qwen(audios, model, language, device)` that serializes audios to 16 kHz WAVs in a tempdir and spawns the worker subprocess. Use `senselab.utils.subprocess_venv.ensure_venv` and `venv_python` as in nemo.py.
- [X] T026 [US4] Implement the worker script (triple-quoted string inside canary_qwen.py per the existing nemo.py convention): import `from nemo.collections.speechlm2.models import SALM`, load via `SALM.from_pretrained(model_uri)`, build chat-style prompts using `model.audio_locator_tag` with `{"audio": [path]}` per the model card, call `model.generate(prompts=...)`, decode via `model.tokenizer.ids_to_text(ids)`. Return the decoded text per audio as a JSON dict on stdout. Errors wrapped per the existing `parse_subprocess_result` contract.
- [X] T027 [US4] Replace the NotImplementedError stub for the Canary branch in /Users/satra/software/sensein/senselab/src/senselab/audio/tasks/speech_to_text/api.py (added in T005) with a real call into `CanaryQwenASR.transcribe_with_canary_qwen`. Map the returned text into senselab's ScriptLine shape: `[ScriptLine(text=text, start=None, end=None, speaker=None) for text in texts]`.
- [X] T028 [P] [US4] Write src/tests/audio/tasks/speech_to_text/canary_qwen_test.py at /Users/satra/software/sensein/senselab/src/tests/audio/tasks/speech_to_text/canary_qwen_test.py with a `@pytest.mark.skipif` guard checking whether the `nemo-canary-qwen` venv exists at `~/.cache/senselab/venvs/nemo-canary-qwen/`. When present, run `transcribe_with_canary_qwen` on the real-speech fixture /Users/satra/software/sensein/senselab/src/tests/data_for_testing/audio_48khz_mono_16bits.wav (resample to 16 kHz mono first via senselab). Assertions are SHAPE-ONLY: result is `List[ScriptLine]`, `text` is a non-empty string, `start is None`, `end is None`. Do NOT assert specific transcription content — the test verifies the senselab API contract for the new backend, not Canary-Qwen's accuracy.
- [ ] T029 [US4] Validate end-to-end by running the analyze_audio script with `--asr-models nvidia/canary-qwen-2.5b` against a sample audio. Confirm: the NeMo subprocess venv provisions on first run; transcription returns text; the script's auto-align stage (US6) adds MMS-based segments; LS export contains per-segment regions. Document in artifacts/analyze_audio_us4_validation.md.

**Checkpoint**: US4 complete — Canary-Qwen 2.5B runs end-to-end through the senselab API and produces aligned timeline annotations in the LS export.

---

## Phase 8: User Story 5 - Use Alibaba Qwen3-ASR 1.7B as a Senselab ASR Model (Priority: P3)

**Goal**: Pass `Qwen/Qwen3-ASR-1.7B` to senselab's `transcribe_audios` and have it run successfully via a new Qwen subprocess venv. The companion forced-aligner provides word timestamps natively, so the script's auto-align stage skips Qwen3 output.

**Independent Test**: `senselab.audio.tasks.speech_to_text.transcribe_audios([audio], model=HFModel(path_or_uri="Qwen/Qwen3-ASR-1.7B"))` returns `[ScriptLine(text="...", start=..., end=..., chunks=[...])]` with word-level timestamps in the chunks.

### Implementation for User Story 5

- [X] T030 [US5] Create /Users/satra/software/sensein/senselab/src/senselab/audio/tasks/speech_to_text/qwen.py modeled on canary_qwen.py. Constants: `_VENV_NAME = "qwen-asr"`, requirements include `qwen-asr` (Alibaba's wrapper, pin to a known-good version). Class `QwenASR` with method `transcribe_with_qwen(audios, model, language, device, *, return_timestamps: bool = True)`.
- [X] T031 [US5] Implement the worker script (triple-quoted string inside qwen.py): import `from qwen_asr import Qwen3ASRModel, AlignerModel`, load via `Qwen3ASRModel.from_pretrained(model_uri)`. When `return_timestamps=True`, also load the companion `AlignerModel.from_pretrained("Qwen/Qwen3-ForcedAligner-0.6B")` and call `model.transcribe(audio_path, forced_aligner=aligner, return_time_stamps=True)`. Map per-word `{text, start_time, end_time}` into senselab `ScriptLine.chunks`; the parent ScriptLine carries the full text and the chunks list.
- [X] T032 [US5] Replace the NotImplementedError stub for the Qwen-ASR branch in /Users/satra/software/sensein/senselab/src/senselab/audio/tasks/speech_to_text/api.py (added in T005) with a real call into `QwenASR.transcribe_with_qwen`. Forward a new `return_timestamps` kwarg from the caller (default True) plus a `--qwen-asr-no-timestamps` script-side flag added to the analyze_audio CLI per contracts/cli.md.
- [X] T033 [P] [US5] Write src/tests/audio/tasks/speech_to_text/qwen_test.py at /Users/satra/software/sensein/senselab/src/tests/audio/tasks/speech_to_text/qwen_test.py with a `@pytest.mark.skipif` guard checking whether the `qwen-asr` venv exists at `~/.cache/senselab/venvs/qwen-asr/`. When present, run `transcribe_with_qwen` once with `return_timestamps=True` and once with `False` on the real-speech fixture /Users/satra/software/sensein/senselab/src/tests/data_for_testing/audio_48khz_mono_16bits.wav (resample to 16 kHz mono first via senselab). Assertions are SHAPE-ONLY: with `return_timestamps=True` the result `ScriptLine.chunks` list is non-empty and chunks have `start`/`end` populated; with `return_timestamps=False` chunks are empty/null and `start`/`end` are None. Do NOT assert specific transcription content.
- [ ] T034 [US5] Validate end-to-end by running the analyze_audio script with `--asr-models Qwen/Qwen3-ASR-1.7B` against a sample audio. Confirm: the qwen-asr subprocess venv provisions on first run; transcription returns ScriptLines with word-level chunks; the script's auto-align stage skips this model (already timestamped); LS export contains per-word regions. Document in artifacts/analyze_audio_us5_validation.md.

**Checkpoint**: US5 complete — Qwen3-ASR 1.7B runs end-to-end with native word timestamps via its companion forced aligner.

---

## Phase 9: Polish & Cross-Cutting Concerns

- [X] T035 Update /Users/satra/software/sensein/senselab/CLAUDE.md to add a new "Audio analysis script + ASR backend extensions" section pointing at scripts/analyze_audio.py and the new senselab APIs (MMS aligner, Canary-Qwen, Qwen3-ASR, Granite Speech). Place it after the existing "Profiling with Scalene" section.
- [X] T036 Run full lint and type-check on all changed files: `uv run ruff check scripts/analyze_audio.py src/senselab/audio/tasks/forced_alignment/ src/senselab/audio/tasks/speech_to_text/`, `uv run ruff format` on the same set, and `uv run mypy` on the same set. Fix any new violations.
- [X] T037 Run the existing senselab test suite (`uv run pytest --ignore=src/tests/scripts -x -q`) to confirm SC-003 (no impact on existing tests). Then run all new tests separately (`uv run pytest src/tests/scripts/ src/tests/audio/tasks/forced_alignment/ src/tests/audio/tasks/speech_to_text/`) and confirm all pass or skip correctly per their `skipif` guards.
- [ ] T038 Run the analyze_audio.py script end-to-end against a real tutorial audio (e.g., `tutorials/audio/speech_to_text.ipynb`'s sample WAV) with the full default model list and verify all six ASR-related success criteria (SC-001 through SC-012) are satisfied. Document in artifacts/analyze_audio_e2e_validation.md.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately.
- **Foundational (Phase 2)**: T004, T005 depend on T001 (script committed).
- **US1 (Phase 3)**: Depends on Phase 2 (T004 cache schema bump).
- **US2 (Phase 4)**: Depends on US1 (caches the script's outputs).
- **US3 (Phase 5)**: Depends on US1 + US2 (LS export aggregates ModelRuns).
- **US6 (Phase 6)**: Depends on Phase 2 (T005 dispatch updates). Independent of US1/US2/US3.
- **US4 (Phase 7)**: Depends on US6 (auto-align needs MMS) + Phase 2 (T005 dispatch).
- **US5 (Phase 8)**: Depends on Phase 2 (T005 dispatch). Independent of US6 (Qwen has its own aligner).
- **Polish (Phase 9)**: Depends on all user stories.

### User Story Dependencies

- **US1 (P1)**: Independent.
- **US2 (P1)**: Builds on US1 (same script).
- **US3 (P2)**: Builds on US1, US2.
- **US6 (P2)**: Independent of US1/US2/US3 implementation; provides API used by US1's auto-align stage.
- **US4 (P2)**: Depends on US6 (script-side auto-align). senselab-side parts (T025–T028) are independent of US6 and can be done in parallel; only the validation task T029 needs US6 in place.
- **US5 (P3)**: Independent of US6 (Qwen has its own aligner).

### Within Each User Story

- Implementation tasks before validation task.
- Tests can run in parallel with implementation tasks marked [P].
- Module creation (canary_qwen.py, qwen.py) before api.py wiring.

### Parallel Opportunities

- T002 + T003 (Setup): pyproject change + verification.
- T008 (US1 tests) parallel with T006/T007 (US1 implementation).
- T021 (script auto-align wiring) parallel with T022/T023 (US6 tests).
- All four [P]-marked test files (T008, T022, T023, T028, T033) can be authored in parallel since they live in different files.
- US6 (Phase 6) and US5 (Phase 8) can be done in parallel since US5 doesn't depend on US6.

---

## Parallel Example: US6 implementation + tests

```bash
# Once T016, T017 (constants + MMS branch) are done, these can run in parallel:
Task: "T021 — wire auto-align stage in scripts/analyze_audio.py"
Task: "T022 — write MMS smoke test in src/tests/audio/tasks/forced_alignment/mms_test.py"
Task: "T023 — write huggingface_no_timestamps_test.py"
```

## Parallel Example: US4 + US5 in parallel

```bash
# Once Phase 2 lands, US4 and US5 can be developed in parallel by different developers:
Task: "T025-T028 — Canary-Qwen module + tests in src/senselab/audio/tasks/speech_to_text/canary_qwen.py"
Task: "T030-T033 — Qwen3-ASR module + tests in src/senselab/audio/tasks/speech_to_text/qwen.py"
```

---

## Implementation Strategy

### MVP First (US1 + US2 + US6 — minimum useful end-to-end)

1. Phase 1: Setup (T001-T003)
2. Phase 2: Foundational (T004-T005)
3. Phase 3: US1 (T006-T008) — script working with Whisper today
4. Phase 4: US2 (T009-T012) — separable ASR/alignment cache
5. Phase 6: US6 (T016-T024) — multilingual aligner; Granite usable end-to-end
6. **STOP and VALIDATE**: full E2E with Whisper + Granite working through the script and the LS import.

### Incremental Delivery

1. MVP above unblocks the team's primary use case (Granite + Whisper).
2. Add US4 (Canary-Qwen) — second high-accuracy ASR option.
3. Add US5 (Qwen3-ASR) — multilingual coverage, native word timestamps.
4. Add US3 polish if not done in MVP — LS importability validated against real LS instance.
5. Polish phase locks everything in (lint, full test sweep, e2e validation, docs).

### Parallel Team Strategy

With multiple developers, after Phase 2:

- Developer A: Phase 3+4 (US1+US2 — script polish + cache).
- Developer B: Phase 6 (US6 — MMS aligner + Granite path).
- Developer C: Phase 7 (US4 — Canary-Qwen) — waits on Developer B's MMS for end-to-end validation but can land senselab module + tests independently.
- Developer D: Phase 8 (US5 — Qwen3-ASR) — fully independent.

---

## Notes

- All script changes go in `scripts/analyze_audio.py` (single file).
- All senselab forced_alignment changes go in `src/senselab/audio/tasks/forced_alignment/{constants.py, forced_alignment.py}` (two existing files).
- Each new ASR backend is its own new file; no edits to existing ASR modules other than `api.py` and `huggingface.py` (the small `return_timestamps` patch).
- Tests use `@pytest.mark.skipif(...)` guards so default CI install (no new venvs, no MMS download) skips them silently.
- Per-task validation tasks (T012, T015, T024, T029, T034, T038) write a short markdown to `artifacts/analyze_audio_*_validation.md` — these are working notes, not committed.
- Commit per phase (or per logical sub-group inside a phase) to keep diffs reviewable.
