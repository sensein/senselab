# Tasks: Optimize Import Times

**Input**: Design documents from `/specs/20260501-154228-optimize-import-times/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, quickstart.md

**Tests**: Not requested in the feature specification. This is a development-time profiling tool.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the script file, CLI argument parsing, and shared utilities

- [x] T001 Create scripts/profile_imports.py with CLI arg parsing (argparse: --threshold default 2.0, --output default artifacts/import_profile_report.md, --tutorials-dir default tutorials/)
- [x] T002 Implement notebook import extractor function in scripts/profile_imports.py — parse notebook JSON, extract import lines from code cells, skip google.colab imports, deduplicate while tracking source notebooks
- [x] T003 Implement import categorization helper in scripts/profile_imports.py — classify each import as senselab, third_party, stdlib, or platform_specific based on module path

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core subprocess isolation mechanism that ALL user stories depend on

- [x] T004 Implement isolated import timer function in scripts/profile_imports.py — spawn subprocess with `sys.executable -c "import time; t=time.perf_counter(); {import_line}; print(time.perf_counter()-t)"`, parse stdout for elapsed time, capture stderr on failure, handle timeout (60s), return (wall_clock_seconds, status, error_message)

**Checkpoint**: Foundation ready — can time any single import in isolation

---

## Phase 3: User Story 1 - Profile All Tutorial Import Bottlenecks (Priority: P1)

**Goal**: Produce a ranked list of all distinct imports from tutorials, sorted by cold-start time, with bottlenecks flagged

**Independent Test**: Run `uv run python scripts/profile_imports.py` and verify a Markdown report is produced with a "Ranked Imports" table covering all distinct imports, sorted slowest-first, with >2s imports flagged

### Implementation for User Story 1

- [x] T005 [US1] Implement main orchestrator for US1 in scripts/profile_imports.py — call extractor (T002), iterate distinct imports, call isolated timer (T004) for each, collect results into list of dicts with fields from data-model.md ImportStatement
- [x] T006 [US1] Implement ranked imports report section in scripts/profile_imports.py — generate Markdown table with columns: #, Import, Time (s), Category, Status, Bottleneck (YES/NO), write to output file with header and generation timestamp
- [x] T007 [US1] Add progress output to stdout in scripts/profile_imports.py — print "Timing import N/M: {import_line}..." during execution so the user sees progress, print summary count at end (total, bottlenecks, failed, skipped)
- [x] T008 [US1] Run the script via `uv run python scripts/profile_imports.py` and verify the ranked imports table is produced correctly in artifacts/import_profile_report.md

**Checkpoint**: User Story 1 complete — ranked import report with bottleneck flags is generated

---

## Phase 4: User Story 2 - Identify Internal vs External Bottlenecks (Priority: P2)

**Goal**: For each bottleneck import, show a dependency breakdown revealing which transitive imports (torch, speechbrain, etc.) consume the most time

**Independent Test**: Run the script and verify that each bottleneck import (>threshold) has a "Dependency Breakdown" subsection listing the top 15 transitive imports by self-time

### Implementation for User Story 2

- [x] T009 [US2] Implement dependency breakdown profiler in scripts/profile_imports.py — for a given import line, spawn subprocess with `sys.executable -X importtime -c "{import_line}"`, parse stderr for import time tree (format: `import time: self [us] | cumulative [us] | module_name`), extract depth from leading whitespace, return sorted list of (child_module, self_time_us, cumulative_time_us, depth)
- [x] T010 [US2] Integrate breakdown profiler into orchestrator in scripts/profile_imports.py — after US1 timing pass, iterate bottleneck imports (>threshold), call breakdown profiler (T009) for each, store results
- [x] T011 [US2] Implement dependency breakdown report section in scripts/profile_imports.py — for each bottleneck import, append a subsection to the Markdown report with the top 15 transitive deps sorted by self-time, showing module name, self time (ms), cumulative time (ms)
- [x] T012 [US2] Run the script and verify dependency breakdown sections appear for all bottleneck imports in artifacts/import_profile_report.md

**Checkpoint**: User Story 2 complete — bottleneck imports have transitive dependency attribution

---

## Phase 5: User Story 3 - Per-Tutorial Import Cost Summary (Priority: P3)

**Goal**: Produce a per-tutorial table showing aggregate cold-start import time for each notebook's full import block

**Independent Test**: Run the script and verify a "Per-Tutorial Summary" table is produced with all 20 tutorials, sorted by total import time

### Implementation for User Story 3

- [x] T013 [US3] Implement tutorial aggregate timer in scripts/profile_imports.py — for a given notebook's import list, concatenate all lines into a single script wrapped with perf_counter, run in one subprocess, return (total_seconds, status)
- [x] T014 [US3] Integrate tutorial timer into orchestrator in scripts/profile_imports.py — after per-import timing, iterate each notebook, call aggregate timer (T013), collect per-tutorial results
- [x] T015 [US3] Implement per-tutorial summary report section in scripts/profile_imports.py — generate Markdown table with columns: Tutorial, Import Time (s), # Imports, sorted by import time descending, insert between ranked imports and dependency breakdowns
- [x] T016 [US3] Run the script end-to-end and verify the complete report in artifacts/import_profile_report.md has all three sections: ranked imports, per-tutorial summary, dependency breakdowns

**Checkpoint**: All user stories complete — full profiling report generated

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final cleanup and validation

- [x] T017 Add skipped/failed imports summary section at end of report in scripts/profile_imports.py — list all imports with status skipped or failed, with error messages
- [x] T018 Run quickstart.md validation — execute `uv run python scripts/profile_imports.py` and `uv run python scripts/profile_imports.py --threshold 3.0` to verify both default and custom threshold work as documented

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on T001 (CLI parsing exists) — BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on T002 (extractor) and T004 (timer)
- **User Story 2 (Phase 4)**: Depends on T005 completion (US1 orchestrator produces bottleneck list)
- **User Story 3 (Phase 5)**: Depends on T002 (extractor) — can run in parallel with US2 after US1
- **Polish (Phase 6)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) — no dependencies on other stories
- **User Story 2 (P2)**: Depends on US1 results (needs bottleneck list from T005) — must follow US1
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) — independent of US2, but US1 orchestrator must exist

### Within Each User Story

- Implementation before validation run
- Core logic before report formatting
- Commit after each task

### Parallel Opportunities

- T002 and T003 can run in parallel (different functions, no dependencies)
- T013 (tutorial timer) and T009 (breakdown profiler) can be developed in parallel since they're independent components
- US2 and US3 implementation could overlap once US1 is done (T009-T011 in parallel with T013-T015 if US1 orchestrator is extended for both)

---

## Parallel Example: User Story 1

```bash
# T002 and T003 can be developed in parallel (Setup phase):
Task: "Implement notebook import extractor in scripts/profile_imports.py"
Task: "Implement import categorization helper in scripts/profile_imports.py"
```

## Parallel Example: After US1 Complete

```bash
# T009 and T013 can be developed in parallel (different components):
Task: "Implement dependency breakdown profiler in scripts/profile_imports.py"
Task: "Implement tutorial aggregate timer in scripts/profile_imports.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T003)
2. Complete Phase 2: Foundational (T004)
3. Complete Phase 3: User Story 1 (T005-T008)
4. **STOP and VALIDATE**: Run script, verify ranked imports report
5. Deliverable: A ranked list of all tutorial imports with bottleneck flags

### Incremental Delivery

1. Complete Setup + Foundational -> Script can time individual imports
2. Add User Story 1 -> Ranked imports report (MVP!)
3. Add User Story 2 -> Dependency breakdowns for bottlenecks
4. Add User Story 3 -> Per-tutorial summary table
5. Each story adds a new section to the same report without breaking previous sections

---

## Notes

- All code goes in a single file: scripts/profile_imports.py
- All commands run via `uv run python scripts/profile_imports.py`
- Report output to artifacts/import_profile_report.md (already gitignored directory)
- No new dependencies — script uses only Python stdlib
- Commit after each task or logical group
