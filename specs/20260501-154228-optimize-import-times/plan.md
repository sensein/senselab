# Implementation Plan: Optimize Import Times

**Branch**: `20260501-154228-optimize-import-times` | **Date**: 2026-05-01 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/20260501-154228-optimize-import-times/spec.md`

## Summary

Build a development-time profiling script that extracts all import statements from senselab's 20 tutorial notebooks, times each in an isolated subprocess to measure cold-start cost, captures transitive dependency breakdowns for bottleneck imports using Python's `-X importtime`, and produces a Markdown report with ranked imports, per-tutorial summaries, and internal-vs-external attribution.

## Technical Context

**Language/Version**: Python 3.11-3.12 (managed via uv)
**Primary Dependencies**: stdlib only (subprocess, json, time, re, pathlib) — the profiling script itself has no heavy deps; it invokes senselab imports in child processes
**Storage**: File-based (Markdown report output to `artifacts/`)
**Testing**: Manual verification (dev tool, not a library feature); optional pytest for the extraction/parsing logic
**Target Platform**: macOS ARM64 (developer workstation), Linux (CI)
**Project Type**: Development script (not packaged)
**Performance Goals**: Complete profiling of ~50 distinct imports within 15 minutes (each subprocess takes 2-10s)
**Constraints**: Each import must be isolated in its own subprocess; no module cache sharing between measurements
**Scale/Scope**: ~80 import lines across 20 notebooks, ~50 distinct top-level modules

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. UV-Managed Python | PASS | Script run via `uv run python scripts/profile_imports.py`; child processes use the same uv-managed Python |
| II. Encapsulated Testing | PASS | Dev tool, not a packaged feature; optional tests would use `uv run pytest` |
| III. Commit Early and Often | PASS | Script is a single logical unit; can be committed as one focused change |
| IV. CI Must Stay Green | PASS | No CI changes; script is development-time only |
| V. Memory-Driven Anti-Pattern Avoidance | PASS | No mocking, no cache manipulation, no circular imports |
| VI. No Unnecessary API Calls | PASS | No external service calls; all profiling is local |
| VII. Simplicity First | PASS | Single script, no abstractions, stdlib-only for the profiler itself |
| VIII. No Hardcoded Parameters | PASS | Threshold, output path, tutorial directory are CLI parameters with sensible defaults |

**Post-Phase 1 Re-check**: All gates still pass. The design uses a single script with parameterized inputs, subprocess isolation, and file-based output. No new dependencies introduced.

## Project Structure

### Documentation (this feature)

```text
specs/20260501-154228-optimize-import-times/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0 research output
├── data-model.md        # Phase 1 data model
├── quickstart.md        # Phase 1 quickstart guide
└── checklists/
    └── requirements.md  # Spec quality checklist
```

### Source Code (repository root)

```text
scripts/
└── profile_imports.py   # The profiling script (new file)

artifacts/
└── import_profile_report.md  # Generated report (gitignored output)
```

**Structure Decision**: Single script in `scripts/` directory. No new packages, no new test files required. The script is a standalone development tool that reads notebooks, spawns subprocesses, and writes a Markdown report. The `artifacts/` directory already exists for generated outputs.

## Implementation Design

### Component 1: Notebook Import Extractor

Parses tutorial notebook JSON files to extract all import statements.

**Input**: Path to `tutorials/` directory
**Output**: List of (import_line, source_notebook) tuples, deduplicated import lines

**Logic**:
- Glob for `tutorials/**/*.ipynb`
- For each notebook, parse JSON, iterate `cells` where `cell_type == "code"`
- For each source line, match `^import ` or `^from .+ import ` patterns
- Skip lines starting with `#` or containing `google.colab`
- Deduplicate import lines while tracking which notebooks use each

### Component 2: Isolated Import Timer

Measures wall-clock time of a single import in a fresh subprocess.

**Input**: A single import line (string)
**Output**: (wall_clock_seconds, status, error_message)

**Logic**:
- Construct a Python one-liner: `import time; t=time.perf_counter(); {import_line}; print(time.perf_counter()-t)`
- Run via `subprocess.run([sys.executable, '-c', one_liner], capture_output=True, timeout=60)`
- Parse stdout for the elapsed time
- If subprocess returns non-zero or times out, record as failed with stderr

### Component 3: Dependency Breakdown Profiler

For imports flagged as bottlenecks, captures the `-X importtime` tree.

**Input**: A single import line (string), already identified as >threshold
**Output**: List of (child_module, self_time_us, cumulative_time_us, depth) tuples

**Logic**:
- Run `subprocess.run([sys.executable, '-X', 'importtime', '-c', import_line], capture_output=True)`
- Parse stderr (importtime outputs to stderr) line by line
- Each line format: `import time: self [us] | cumulative [us] | module_name`
- Indentation indicates depth
- Sort by self_time descending, keep top 15 entries

### Component 4: Tutorial Aggregate Timer

Times each tutorial's full import block as a single unit (sequential imports in one process).

**Input**: List of import lines for a specific notebook
**Output**: (total_seconds, status)

**Logic**:
- Concatenate all import lines for the notebook into a single script
- Wrap with `time.perf_counter()` before and after
- Run in a single subprocess
- This measures the realistic user experience (where torch is loaded once and subsequent imports reuse it)

### Component 5: Report Generator

Produces the Markdown report from collected data.

**Input**: All timing results from Components 2-4
**Output**: Markdown file written to `artifacts/import_profile_report.md`

**Sections**:
1. Header with generation timestamp and threshold
2. Ranked imports table (all imports, bottlenecks flagged)
3. Per-tutorial summary table (sorted by total time)
4. Dependency breakdowns (one subsection per bottleneck import, top 15 transitive deps)
5. Skipped/failed imports list

### Main Orchestrator

**Flow**:
1. Parse CLI args (threshold, output path, tutorials dir)
2. Extract imports from notebooks (Component 1)
3. Time each distinct import individually (Component 2) — can parallelize with ProcessPoolExecutor
4. For bottleneck imports, run dependency breakdown (Component 3)
5. Time each tutorial's full import block (Component 4)
6. Generate report (Component 5)
7. Print summary to stdout

## Complexity Tracking

No constitution violations to justify.
