# Tasks: Scalene-Based Profiling Tooling

**Input**: Design documents from `/specs/20260503-235625-scalene-profiling/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/cli.md, quickstart.md

**Tests**: Tests for the profiling tool itself are permitted by the spec clarification (Session 2026-05-04). One smoke test is included to prevent regressions; it is skipped on default installs (where Scalene is not present), keeping the existing CI suite unaffected.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Add the opt-in dependency group and prepare the output directory pattern

- [x] T001 Add new `profiling` dependency group to pyproject.toml at /Users/satra/software/sensein/senselab/pyproject.toml under the existing `[dependency-groups]` table — add `profiling = ["scalene>=2.2", "nbconvert>=7"]` immediately after the `docs = [...]` group; preserve all existing groups exactly as-is. `nbconvert` is required so the wrapper can convert `.ipynb` targets without depending on the `senselab-ai` extra.
- [x] T002 [P] Verify default install is unchanged by running `uv sync` (no flag) and confirming `uv pip list | grep scalene` returns nothing — this validates SC-003 before any other work begins
- [x] T003 [P] Verify opt-in install works by running `uv sync --group profiling` and confirming `uv pip show scalene` succeeds

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Skeleton for the wrapper script — argparse + scalene availability check that all user stories depend on

- [x] T004 Create scripts/profile_with_scalene.py at /Users/satra/software/sensein/senselab/scripts/profile_with_scalene.py with: shebang, module docstring, argparse setup matching contracts/cli.md (positional TARGET, --output-dir, --format html|json, --cpu-only, --gpu, --exclude SUBSTR, --keep-intermediate, plus a mutually-exclusive group containing --no-thirdparty and --scope SUBSTR). Args after a literal `---` (three dashes) on the command line are forwarded to the target script. Define the `main()` entry point under `if __name__ == "__main__":`. Do NOT include `--include-children` — Scalene 2.2.1 has no equivalent flag.
- [x] T005 Implement Scalene availability check in scripts/profile_with_scalene.py — try `import scalene` at the top of `main()`; on ImportError, print `ERROR: Scalene is not installed in this environment.\nTo install: uv sync --group profiling` to stderr and exit with code 3 (matches FR-007 and contracts/cli.md exit code table)
- [x] T006 Implement target validation in scripts/profile_with_scalene.py — verify the TARGET path exists and is readable; on missing/unreadable, print clear error to stderr and exit with code 4; classify by suffix into `python_script` (.py) or `jupyter_notebook` (.ipynb). When the target is a notebook, also verify `nbconvert` is importable; if not, print install hint and exit 5 (matches contracts/cli.md exit code table).
- [x] T007 Implement output path computation in scripts/profile_with_scalene.py — build `<output-dir>/<target_stem>_<YYYYMMDD-HHMMSS>.<ext>` (where ext is html or json based on --format); create parent directory with `Path.mkdir(parents=True, exist_ok=True)`

**Checkpoint**: Foundation ready — script can parse args, validate target, compute output path, and gracefully handle missing scalene

---

## Phase 3: User Story 1 - Profile a Tutorial or Code Snippet (Priority: P1)

**Goal**: Single command to profile any Python script or Jupyter notebook with Scalene and produce an HTML report

**Independent Test**: Run `uv run python scripts/profile_with_scalene.py path/to/script.py` against any senselab usage example and verify an HTML report appears under artifacts/scalene/ with line-level CPU and memory data

### Implementation for User Story 1

- [x] T008 [US1] Implement Scalene `run` step in scripts/profile_with_scalene.py — Scalene 2.2.1 produces JSON; HTML requires a separate `view` step (see T011). Build the command list as `[sys.executable, "-m", "scalene", "run", "-o", <stem>.json, *flags, <target>, "---", *target_args]`. Append flags conditionally: `--cpu-only` (if requested), `--gpu` (if requested). Pass `<target>` as the converted .py path when the original was a notebook. Run via `subprocess.run(...)` with no timeout, capture returncode and stderr.
- [x] T009 [US1] Implement notebook conversion in scripts/profile_with_scalene.py — when target is `.ipynb`, create a `tempfile.TemporaryDirectory()`, run `[sys.executable, "-m", "nbconvert", "--to", "script", "--output-dir", tmpdir, target]` (using `python -m nbconvert` directly to avoid the heavier `jupyter` namespace) via subprocess, locate the resulting `<stem>.py` in tmpdir, then proceed to T008 with that converted path; if --keep-intermediate is set, copy the converted .py next to the final report before tempdir cleanup.
- [x] T010 [US1] Implement exit-code mapping in scripts/profile_with_scalene.py — when the `run` step returncode is 0 AND the JSON file exists, the run succeeded; when `--format html`, proceed to T011's view step; otherwise (run returncode non-zero or JSON missing), print Scalene's stderr to the wrapper's stderr and exit 1. Surface the final report path with `Scalene profile written to: <abs_path>` plus a platform-appropriate open hint (`open` for darwin, `xdg-open` for linux). The earlier "partial report" branch (exit 2) is dropped — Scalene's `run` produces a complete JSON or no JSON.
- [x] T011 [US1] Implement HTML view step and JSON format selection in scripts/profile_with_scalene.py — Scalene 2.2.1 outputs JSON natively from `run`; HTML requires a second invocation. When `--format html` (default), after a successful `run` step, invoke `[sys.executable, "-m", "scalene", "view", "--standalone", <stem>.json]` which produces `<stem>.html` next to the JSON. Move the HTML to the final output path (`<output-dir>/<target_stem>_<timestamp>.html`). Delete the intermediate JSON unless `--keep-intermediate` is set. When `--format json`, skip the view step entirely and report the JSON path as the result.
- [x] T012 [US1] Run end-to-end validation: create a tiny example script at /tmp, run `uv run python scripts/profile_with_scalene.py /tmp/tiny.py` (after `uv sync --group profiling`), verify an HTML file is produced under artifacts/scalene/ and opens in a browser
- [x] T013 [US1] Run end-to-end notebook validation: pick a small senselab tutorial (e.g., tutorials/utils/dimensionality_reduction.ipynb), run `uv run python scripts/profile_with_scalene.py tutorials/utils/dimensionality_reduction.ipynb`, verify the notebook converts to .py, Scalene runs, and an HTML report appears under artifacts/scalene/

**Checkpoint**: User Story 1 complete — wrapper profiles both scripts and notebooks end-to-end

---

## Phase 4: User Story 2 - Profile Specific Functions or Code Regions (Priority: P2)

**Goal**: Allow scoping the profile report to a specific function or named region

**Independent Test**: Create a test script with a slow `target_function()` and a slow `setup_function()`, run the wrapper with `--scope target_function`, and verify the resulting report's top entries are only inside `target_function`'s call stack

### Implementation for User Story 2

- [x] T014 [US2] Implement --scope flag handling in scripts/profile_with_scalene.py — when --scope SUBSTR is provided, append `--profile-only SUBSTR` to the Scalene `run` command. Scalene's `--profile-only` matches files whose **path** contains SUBSTR (substring match, comma-separated for multiple values). Document this in argparse `--help` text as: `Profile only files whose path contains SUBSTR (substring match against file paths, not function names).`
- [x] T015 [US2] Implement --no-thirdparty flag handling in scripts/profile_with_scalene.py — when set, append `--profile-only senselab` to the Scalene `run` command. Scalene's `--profile-only` is a path substring match, so `senselab` matches `src/senselab/`, the installed package, and any tutorial path containing "senselab". Document in argparse `--help` as: `Restrict profiling to files with 'senselab' in their path. Mutually exclusive with --scope.` The mutual exclusion is enforced by the argparse group defined in T004.
- [x] T016 [US2] Implement --exclude flag handling in scripts/profile_with_scalene.py — when --exclude SUBSTR is provided, append `--profile-exclude SUBSTR` to the Scalene `run` command. Useful for hiding noisy modules (e.g., `--exclude transformers`). May be combined with --scope or --no-thirdparty. Document in argparse `--help` as: `Exclude files whose path contains SUBSTR (substring match).` (Originally planned --include-children dropped: Scalene 2.2.1 has no --profile-all-children equivalent. Document this limitation in T018's docs section: "Children process profiling is not supported by this wrapper; profile each subprocess script separately.")
- [x] T017 [US2] Validate scoped profiling end-to-end: create two tiny scripts in different directories — one with `heavy_func` doing CPU work, one with `light_func` doing nothing — and a driver that imports and calls both. Run `uv run python scripts/profile_with_scalene.py --scope heavy script.py` (where the heavy script lives in a path containing "heavy"). Open the report. Confirm entries from the "heavy" path dominate, and the "light" path is filtered out. (Note: `--scope` is path-substring matching, not function-name matching — set up the test accordingly.)

**Checkpoint**: User Story 2 complete — scoped profiling produces focused reports

---

## Phase 5: User Story 3 - Discoverable Documentation and Examples (Priority: P3)

**Goal**: New contributors can profile their changes using only project documentation

**Independent Test**: A reader of CLAUDE.md (or the new docs section) can locate the profiling instructions and run the worked example successfully on the first attempt

### Implementation for User Story 3

- [x] T018 [US3] Add a "Profiling with Scalene" section to CLAUDE.md at /Users/satra/software/sensein/senselab/CLAUDE.md (place it as a new section after "System Requirements") with: install command (`uv sync --group profiling`), basic invocation (`uv run python scripts/profile_with_scalene.py <target>`), one worked notebook example, the constraint that notebooks must be non-interactive (no `input()` or widget event waits), the limitation that child-process profiling is not supported by this wrapper in Scalene 2.2.1, and a pointer to specs/20260503-235625-scalene-profiling/quickstart.md for full options.
- [x] T019 [US3] Add a top-of-file docstring to scripts/profile_with_scalene.py with: one-line summary, install hint, example invocation for a Python script, example invocation for a notebook, example with --scope, and a pointer to contracts/cli.md
- [x] T020 [US3] Verify discoverability: confirm `grep -F "Profiling with Scalene" CLAUDE.md` returns the new section heading (specific match avoids false positives from earlier features that mention "profiling"), and `uv run python scripts/profile_with_scalene.py --help` prints all CLI options matching contracts/cli.md (--output-dir, --format, --cpu-only, --no-thirdparty, --scope, --exclude, --gpu, --keep-intermediate)

**Checkpoint**: User Story 3 complete — profiler is documented and discoverable

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Smoke test, platform notes, final verification

- [x] T021 [P] Create src/tests/scripts/__init__.py at /Users/satra/software/sensein/senselab/src/tests/scripts/__init__.py as an empty file. pytest discovers test files without an __init__.py via testpaths config, but the init file ensures consistent package structure for mypy and IDE tooling, matching the pattern used elsewhere under src/tests/.
- [x] T022 Create src/tests/scripts/test_profile_with_scalene.py at /Users/satra/software/sensein/senselab/src/tests/scripts/test_profile_with_scalene.py with one test function `test_wrapper_produces_report(tmp_path)` that: detects scalene via `importlib.util.find_spec`, uses `@pytest.mark.skipif(not scalene_available, reason="scalene not installed")`, writes a tiny throwaway .py script under tmp_path, runs the wrapper as a subprocess, and asserts an HTML file is produced in the supplied output directory
- [x] T023 [P] Add a one-line platform note to scripts/profile_with_scalene.py — when `sys.platform == "darwin"` and `--gpu` was passed, print to stderr: `Note: GPU profiling is not available on macOS; GPU columns will be empty.` (does not change behavior, just informs)
- [x] T024 Run the existing test suite via `uv run pytest -x --ignore=src/tests/scripts` to verify nothing else is affected (validates SC-003); then run `uv run pytest src/tests/scripts/` separately — should pass when scalene is installed, skip when it's not
- [x] T025 Run quickstart.md validation end-to-end on a clean environment: `uv sync --group profiling`, then run all example commands from specs/20260503-235625-scalene-profiling/quickstart.md, confirm each produces the documented output

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: T001 (pyproject change) blocks T003 and Phase 2+; T002 can verify status before any change
- **Foundational (Phase 2)**: Depends on T003 (scalene installable); T004-T007 must be sequential within the same file
- **User Story 1 (Phase 3)**: Depends on Phase 2; T008-T011 build on T004's argparse skeleton
- **User Story 2 (Phase 4)**: Depends on Phase 2 (argparse skeleton from T004); independent of US1 implementation but easier to validate after US1 is functional
- **User Story 3 (Phase 5)**: Depends on the wrapper being usable (Phase 3 minimum) so docs reference real behavior
- **Polish (Phase 6)**: Smoke test (T022) depends on the wrapper being functional (Phase 3); CI verification (T024) depends on all changes being in place

### User Story Dependencies

- **US1 (P1)**: Independent — only needs Phase 2 foundation
- **US2 (P2)**: Independent of US1 implementation but shares the same wrapper file (sequential edits to the same file)
- **US3 (P3)**: Depends on US1 producing real outputs that the docs can describe

### Within Each User Story

- All edits to scripts/profile_with_scalene.py are sequential (single file)
- Validation tasks (T012, T013, T017, T020) come last in their phase
- The smoke test (T022) is independent of user stories — runs in Polish phase

### Parallel Opportunities

- T002 and T003 can run in parallel after T001 (verifying default vs opt-in install)
- T021 (init file) is independent of all other tasks
- T023 (platform note) is independent of US1/US2/US3 implementation work

---

## Parallel Example: Phase 1 Setup

```bash
# After T001 (pyproject change) is committed, T002 and T003 verify in parallel:
Task: "Verify default install: uv sync && (uv pip list | grep scalene; test $? -ne 0)"
Task: "Verify opt-in install: uv sync --group profiling && uv pip show scalene"
```

## Parallel Example: Polish Phase

```bash
# T021 and T023 can run in parallel (different files):
Task: "Create empty src/tests/scripts/__init__.py"
Task: "Add macOS GPU note to scripts/profile_with_scalene.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T003)
2. Complete Phase 2: Foundational (T004-T007)
3. Complete Phase 3: User Story 1 (T008-T013)
4. **STOP and VALIDATE**: Profile a real tutorial; confirm HTML report is usable
5. Deliverable: Working profiler for any .py or .ipynb target

### Incremental Delivery

1. Setup + Foundational → wrapper skeleton can fail-fast on missing scalene
2. Add US1 → wrapper profiles scripts and notebooks (MVP!)
3. Add US2 → scoped profiling for targeted investigation
4. Add US3 → docs make it discoverable for new contributors
5. Polish → smoke test pins behavior; CI confirms unchanged

### Parallel Team Strategy

For one developer, sequential execution is correct because nearly all tasks edit the same file (scripts/profile_with_scalene.py). For two developers, one could work on the wrapper (Phases 2-4) while the other prepares docs (Phase 5) and the smoke test (T022).

---

## Notes

- All wrapper code lives in a single file: scripts/profile_with_scalene.py
- All commands run via `uv run`
- Reports written to artifacts/scalene/ (gitignored via existing artifacts/ patterns)
- No edits to src/senselab/ (preserves API stability per spec clarification)
- The smoke test is the only addition under src/tests/; it self-skips when scalene is absent
- Commit after each phase or each completed user story
