# Tasks: Test Classification, Dependency Updates, and Modular Architecture

**Input**: Design documents from `/specs/20260419-133236-test-classification-deps/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md

**Organization**: Tasks grouped by user story. All changes target `alpha` branch.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to

---

## Phase 1: Setup

**Purpose**: Branch preparation.

- [x] T001 Checkout `alpha` branch, pull latest, create working branch `alpha-test-class-deps`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Infrastructure that all user stories depend on.

- [x] T002 [P] Create `src/senselab/utils/subprocess_venv.py` — implement `ensure_venv(name, requirements, python_version)` using uv to create/cache isolated venvs at `~/.cache/senselab/venvs/{name}/`, with file locking for concurrent access, and `run_in_venv(name, module, function, args, kwargs)` that executes a function via subprocess with JSON IPC over stdin/stdout. Handle file paths for large data (audio waveforms passed as temp files, not serialized).

- [x] T003 [P] Create `src/senselab/utils/compatibility.py` — implement a compatibility matrix as a Python dict mapping each public API function name to `{required_deps, python_versions, torch_versions, isolated, venv_name, gpu_required}`. Add `check_compatibility(function_name)` that raises a clear `ImportError` naming the missing package + install command. Add `get_matrix()` to return the full matrix for documentation.

- [x] T004 [P] Generate `docs/compatibility-matrix.md` — script or function that reads the matrix from `compatibility.py` and produces a human-readable markdown table. Can be a standalone script at `scripts/generate-compat-matrix.py` or integrated into the pdoc build.

- [x] T004a Test `src/senselab/utils/subprocess_venv.py` — verify: (a) `ensure_venv("test", ["requests"], "3.11")` creates a venv at `~/.cache/senselab/venvs/test/`, (b) calling `ensure_venv` again reuses the existing venv (no reinstall), (c) `run_in_venv("test", ...)` executes a function and returns JSON result, (d) a venv with a different Python version from the host can be created (e.g., 3.11 when host is 3.12).

**Checkpoint**: Subprocess venv utility and compatibility matrix ready. Backend isolation and dependency upgrades can proceed.

---

## Phase 3: User Story 1 — CPU-only tests on labeled PRs (Priority: P1)

**Goal**: Re-enable macOS tests as label-triggered (`macos-test` label). GPU tests auto-skip via existing pytest markers.

**Independent Test**: Open a PR, verify macOS-tests run and GPU tests are skipped.

- [x] T005 [US1] Re-enable macOS test job in `.github/workflows/tests.yaml` — replace `if: false` with label-triggered condition (e.g., `contains(github.event.pull_request.labels.*.name, 'macos-test')`). Create the `macos-test` label in GitHub. Verify the job runs CPU-safe tests only when labeled.

- [x] T006 [US1] Run the macOS test suite locally (`uv run pytest src/tests`) and document the pass/skip/fail counts as baseline. Record which tests skip due to `torch.cuda.is_available()` vs `TORCHAUDIO_AVAILABLE` vs other markers.

- [x] T007 [US1] Push to `alpha`, label PR with `macos-test`, verify macOS-tests pass on GitHub Actions, confirm GPU tests show as "skipped" not "failed", and verify total test duration is under 10 minutes (check Actions log timing for SC-001).

**Checkpoint**: CPU tests run on every PR. GPU tests skip cleanly.

---

## Phase 4: User Story 2 — GPU tests on EC2 when labeled (Priority: P1)

**Goal**: Verify the EC2 GPU pipeline still works after dependency changes. Already functional — this phase validates baseline.

- [x] T008 [US2] Label a test PR with `ec2-gpu-test` on `alpha`, verify all tests run including GPU tests, confirm pass count >= 491 (current baseline).

**Checkpoint**: GPU test infrastructure verified.

---

## Phase 5: User Story 3 — Full dependency upgrade with conflict resolution (Priority: P2)

**Goal**: Upgrade ALL dependencies, resolve conflicts, isolate incompatible packages.

### Phase 5a: GitHub Actions bumps (sequential — merge each after CI passes)

- [x] T009 [US3] Retarget PR #421 (actions/checkout v5→v6) to `alpha` and merge
- [x] T010 [US3] Retarget PR #432 (actions/upload-artifact v5→v7) to `alpha` and merge
- [x] T011 [US3] Retarget PR #433 (aws-actions/configure-aws-credentials v5→v6) to `alpha` and merge
- [x] T012 [US3] Retarget PR #434 (astral-sh/setup-uv v5→v7) to `alpha` and merge
- [x] T013 [US3] Retarget PR #435 (actions/download-artifact v6→v8) to `alpha` and merge

### Phase 5b: Core dependency upgrade

- [x] T014 [US3] Remove version upper bounds from core dependencies in `pyproject.toml` — change `~=X.Y` pins to `>=X.Y` for: torch, torchaudio, torchvision, transformers, datasets, huggingface-hub, pydantic, speechbrain, pyannote-audio, scikit-learn, matplotlib. Keep lower bounds for API compatibility.

- [x] T015 [US3] Run `uv lock --upgrade` and identify which packages have version conflicts. Document the conflict set (expected: coqui-tts, ppgs/espnet/snorkel/lightning, possibly sentence-transformers).

- [x] T016 [US3] Remove conflicting packages from `pyproject.toml` core/extras dependencies — move coqui-tts, ppgs, snorkel, lightning, and any other packages that prevent resolution to the compatibility matrix as isolated backends.

- [x] T017 [US3] Re-run `uv lock --upgrade` with conflicting packages removed. Verify clean resolution. Commit the updated `pyproject.toml` and `uv.lock`.

- [x] T018 [US3] Unpin `sentence-transformers` (remove `<5.4` upper bound). The torchcodec import issue was mitigated by removing `local_files_only` — verify sentence-transformers 5.4+ installs cleanly.

### Phase 5c: Fix breakage from upgrades

- [x] T019 [US3] Run `uv run pytest src/tests` locally (CPU) and identify tests that fail due to API changes in upgraded packages. Fix senselab wrapper code for each failure.

- [x] T020 [US3] Run EC2 GPU tests (label PR with `ec2-gpu-test`) and fix any GPU-specific failures from upgraded packages.

- [x] T021 [US3] Resolve cv2/av `libavdevice` duplicate symbol conflict — ensure `opencv-python-headless` is used (not `opencv-python`) and verify no runtime crashes from the FFmpeg library duplication.

**Checkpoint**: All core dependencies upgraded to latest. Conflicting packages removed from core. Tests pass.

---

## Phase 6: User Story 4 — Feature/dependency compatibility matrix (Priority: P2)

**Goal**: Populate the compatibility matrix with real data and wire it into the API.

- [x] T022 [US4] Populate `src/senselab/utils/compatibility.py` matrix entries for all public API functions in `src/senselab/audio/tasks/`, `src/senselab/text/tasks/`, and `src/senselab/video/tasks/`. For each function, record: required_deps, python_versions, torch_versions, isolated (bool), venv_name, gpu_required.

- [x] T023 [US4] Add `check_compatibility()` calls to all public API entry points (the `api.py` files in each task module). When a required dep is missing, raise `ImportError("package X is required for function Y. Install with: pip install senselab[extra] or uv pip install X")`.

- [x] T024 [US4] Run `scripts/generate-compat-matrix.py` to produce `docs/compatibility-matrix.md`. Verify it covers all functions.

- [x] T025 [US4] Verify that calling a function with a missing optional dep produces a clear error message (not a traceback).

**Checkpoint**: Every function has compatibility metadata. Missing deps produce clear errors.

---

## Phase 7: User Story 5 — Legacy backends in subprocess venvs (Priority: P3)

**Goal**: Move coqui-tts and ppgs/espnet to isolated subprocess venvs.

- [x] T026 [US5] Update `src/senselab/audio/tasks/voice_cloning/coqui.py` — replace direct `TTS()` import with `run_in_venv("coqui", ...)` call. The subprocess venv installs `coqui-tts~=0.27` with `torch~=2.8` (or whatever version coqui needs). Data flow: save source/target Audio waveforms to lossless temp files (e.g., FLAC for audio, .pt for tensors), pass file paths via JSON args, isolated venv reads from paths, writes result to lossless temp file, host loads result. Use the most efficient lossless format for each data type.

- [x] T027 [US5] Add compatibility matrix entry for `clone_voices` marking it as `isolated=True, venv_name="coqui"`.

- [x] T028 [US5] Update `src/senselab/audio/tasks/features_extraction/` ppgs-related code — replace direct ppgs/espnet imports with `run_in_venv("ppgs", ...)` call. The subprocess venv installs `ppgs>=0.0.9`, `espnet`, `snorkel`, `lightning`. Data flow: save Audio waveform to lossless temp file (FLAC or .pt), pass path + parameters via JSON, isolated venv computes features and writes result to temp .npy or .pt, host loads result. Use efficient lossless formats throughout.

- [x] T029 [US5] Add compatibility matrix entries for ppgs/articulatory functions marking them as `isolated=True, venv_name="ppgs"`.

- [x] T030 [US5] Test isolated backends on EC2 — label PR with `ec2-gpu-test`, verify voice_cloning and ppgs tests pass via subprocess venvs.

**Checkpoint**: Legacy backends work via subprocess venvs. Core is conflict-free.

---

## Phase 8: User Story 6 — Dual-mode CPU/GPU tests (Priority: P3)

**Goal**: Identify and update tests that unnecessarily skip on CPU.

- [x] T031 [US6] Audit the 21 GPU-required test files. For each, determine if the test can run on CPU (model loads on CPU, inference works on CPU). Document findings.

- [x] T032 [US6] For tests identified as dual-mode: change `@pytest.mark.skipif(not torch.cuda.is_available())` to test-specific logic that runs on CPU when no GPU, runs on GPU when available. Use `DeviceType.CPU` as fallback.

- [x] T033 [US6] Verify dual-mode tests pass on both macOS (CPU) and EC2 (GPU).

**Checkpoint**: More tests run on CPU, increasing GitHub Actions coverage.

---

## Phase 9: Polish & Cross-Cutting

- [x] T034 Merge all changes from working branch to `alpha` via PR
- [x] T035 Run full EC2 GPU test suite on `alpha` — verify pass count >= baseline
- [x] T036 Merge `alpha` → `main` via PR
- [x] T037 Update `docs/compatibility-matrix.md` with final data
- [x] T038 Update `.github/EC2_GPU_RUNNER.md` if any workflow changes were made

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No deps
- **Foundational (Phase 2)**: T002, T003, T004 are parallel
- **US1 (Phase 3)**: Depends on Phase 1 only (re-enable macOS)
- **US2 (Phase 4)**: Independent (just verification)
- **US3 (Phase 5)**: Depends on Phase 2 (subprocess_venv.py needed for isolating conflicting packages)
  - 5a (Actions bumps): independent, can start immediately
  - 5b (Core upgrade): depends on 5a completion
  - 5c (Fix breakage): depends on 5b
- **US4 (Phase 6)**: Depends on Phase 2 (compatibility.py) + Phase 5 (need final dep list)
- **US5 (Phase 7)**: Depends on Phase 2 (subprocess_venv.py) + Phase 5 (conflicting packages identified)
- **US6 (Phase 8)**: Depends on Phase 3 (macOS tests working) + Phase 5 (deps upgraded)
- **Polish (Phase 9)**: Depends on all prior phases

### Parallel Opportunities

- T002, T003, T004 are fully parallel (different files)
- T009-T013 can merge sequentially but are independent of code tasks
- US1 (Phase 3) can run in parallel with Phase 5a (Actions bumps)
- T031 (audit) can start before T032 (implementation)

---

## Implementation Strategy

### MVP First (US1 + US3a)

1. T001 (setup) + T005 (re-enable macOS) → immediate CI improvement
2. T009-T013 (merge Actions bumps) → modernize CI infra
3. **STOP and VALIDATE**: macOS tests green, Actions up to date

### Core Delivery (US3b + US3c + US4)

4. T002-T004 (foundational: subprocess_venv, compatibility matrix)
5. T014-T021 (upgrade deps, resolve conflicts, fix breakage)
6. T022-T025 (populate matrix, wire into API)
7. **STOP and VALIDATE**: all deps upgraded, matrix complete

### Full Delivery (US5 + US6)

8. T026-T030 (isolate legacy backends)
9. T031-T033 (dual-mode tests)
10. T034-T038 (merge and polish)

---

## Notes

- Total tasks: 38
- Tasks per story: US1: 3, US2: 1, US3: 13, US4: 4, US5: 5, US6: 3, Foundational: 3, Setup: 1, Polish: 5
- US3 is the heaviest phase — break it into sub-phases (a/b/c) for incremental progress
- Commit after each sub-phase, not at the end
- All changes target `alpha` first
