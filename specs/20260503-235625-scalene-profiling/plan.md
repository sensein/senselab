# Implementation Plan: Scalene-Based Profiling Tooling

**Branch**: `20260503-235625-scalene-profiling` | **Date**: 2026-05-04 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/20260503-235625-scalene-profiling/spec.md`

## Summary

Add a Scalene-based general-purpose profiling tool to senselab as an opt-in developer dependency. A single CLI wrapper script (`scripts/profile_with_scalene.py`) accepts any Python script or Jupyter notebook, transparently converts notebooks via `nbconvert`, invokes Scalene 2.2's `run` and `view` subcommands, and produces an interactive standalone HTML report under `artifacts/scalene/`. No changes to `src/senselab/` runtime code, no changes to existing tutorials or tests, no impact on default `uv sync`.

## Technical Context

**Language/Version**: Python 3.11-3.14 (managed via uv) — matches senselab's `requires-python`
**Primary Dependencies**: scalene (new optional dep, opt-in via `--group profiling`); nbconvert (also added to the new profiling group so notebook profiling works without requiring the `senselab-ai` extra). Verified against Scalene 2.2.1.
**Storage**: File-based — HTML/JSON reports written to `artifacts/scalene/`
**Testing**: pytest smoke test under `src/tests/scripts/` with `@pytest.mark.skipif` guard so default CI install (without profiling group) passes unchanged
**Target Platform**: macOS ARM64 (developer workstation), Linux (CI is unaffected since profiling is opt-in)
**Project Type**: Development tool (additive; not part of the senselab package API)
**Performance Goals**: Profiling overhead <20% of unprofiled runtime (Scalene's design target)
**Constraints**:
- API stability: zero changes to `src/senselab/` (FR-005, SC-003)
- Default `uv sync` unchanged (SC-003)
- Existing CI test suite must pass without modification (SC-003)
**Scale/Scope**: One wrapper script (~150-200 LOC), one `pyproject.toml` entry, one smoke test, one docs section

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. UV-Managed Python | PASS | All invocations via `uv run`; install via `uv sync --group profiling` |
| II. Encapsulated Testing | PASS | Smoke test runs under `uv run pytest`; uses `skipif` guard so default install is unaffected |
| III. Commit Early and Often | PASS | Plan splits into discrete commits: pyproject change, wrapper script, test, docs |
| IV. CI Must Stay Green | PASS | Profiling group is opt-in; CI does not install it; existing tests untouched |
| V. Memory-Driven Anti-Pattern Avoidance | PASS | No mocking, no monkey-patching of caches, no print debugging — wrapper uses subprocess only |
| VI. No Unnecessary API Calls | PASS | No external service calls; Scalene runs locally |
| VII. Simplicity First | PASS | Single wrapper script; no abstractions over Scalene's existing primitives; reuses `nbconvert` and `subprocess` |
| VIII. No Hardcoded Parameters | PASS | Output dir, format, scope, exclude, cpu-only, gpu all CLI parameters with sensible defaults |

**Post-Phase 1 Re-check**: All gates still pass. The design adds one script, one optional dep entry, one optional smoke test, and one docs section — no source modifications required.

## Project Structure

### Documentation (this feature)

```text
specs/20260503-235625-scalene-profiling/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0 research output
├── data-model.md        # Phase 1 data model
├── quickstart.md        # Phase 1 quickstart guide
├── contracts/
│   └── cli.md           # CLI contract for the wrapper script
└── checklists/
    └── requirements.md  # Spec quality checklist
```

### Source Code (repository root)

```text
scripts/
├── profile_imports.py            # Existing (from previous feature)
└── profile_with_scalene.py       # New — Scalene wrapper

src/tests/scripts/
└── test_profile_with_scalene.py  # New — smoke test (skipped when scalene not installed)

artifacts/
└── scalene/                      # New — output dir for generated reports

pyproject.toml                    # Modified — add `[dependency-groups.profiling]`

CLAUDE.md or docs                 # Modified — add a "Profiling" section pointing to scripts/profile_with_scalene.py
```

**Structure Decision**: Single Python wrapper in the existing `scripts/` directory, mirroring the existing `profile_imports.py` pattern. Tests live under `src/tests/scripts/` (matches the existing test structure). `pyproject.toml` gains one new entry under `[dependency-groups]`. No package code changes — `src/senselab/` is untouched.

## Implementation Design

### Component 1: pyproject.toml — New Dependency Group

Add to `[dependency-groups]`:

```toml
profiling = [
  "scalene>=2.2",
  "nbconvert>=7"
]
```

`nbconvert` is required so the wrapper can convert `.ipynb` targets to `.py` without depending on the `senselab-ai` extra. Default `uv sync` is unchanged. To install: `uv sync --group profiling`.

### Component 2: Wrapper Script (`scripts/profile_with_scalene.py`)

Single Python file with the following responsibilities:

1. **Argument parsing**: argparse with the options defined in `contracts/cli.md`
2. **Scalene availability check**: import `scalene` at startup; on `ImportError`, print install hint and exit 3
3. **Target validation**: ensure target file exists; classify as `.py` or `.ipynb`
4. **Notebook handling**: when target is `.ipynb`, run `jupyter nbconvert --to script <notebook> --output-dir <tmpdir>` to produce a `.py` in a temp directory; profile the converted file
5. **Output path computation**: build `artifacts/scalene/<target_stem>_<timestamp>.<ext>` (where `ext` is `html` or `json`); create parent directory if missing
6. **Scalene invocation (two-step)**: Scalene 2.2.1 uses `run` (produces JSON) and `view` (renders HTML/CLI from JSON) subcommands.
   - **Step 1 (always)** — Profile to JSON:
     `python -m scalene run -o <stem>.json [scalene-flags...] <target> [--- TARGET_ARGS]`
     - `--cpu-only` → append `--cpu-only`
     - `--gpu` → append `--gpu` (auto-detects CUDA; no-op on macOS, with a warning)
     - `--scope X` → append `--profile-only X` (file-path substring match)
     - `--no-thirdparty` → append `--profile-only senselab` (mutually exclusive with `--scope`)
     - `--exclude X` → append `--profile-exclude X`
     - TARGET_ARGS forwarded after the `---` separator (Scalene's convention)
   - **Step 2 (when `--format html`)** — Render standalone HTML from JSON:
     `python -m scalene view --standalone <stem>.json`
     Produces `<stem>.html` next to the JSON. The wrapper deletes the JSON afterwards unless `--keep-intermediate` is set.
   - With `--format json`: skip Step 2 and report the JSON path as the result.
7. **Run via subprocess**: `subprocess.run([...])`; on Step 1 returncode 0 + Step 2 success, exit 0; on either step failing, exit 1 with Scalene's stderr surfaced. (Earlier "partial report" heuristic dropped — Scalene's `run` produces a complete JSON or no JSON.)
8. **User-facing output**: on success, print absolute path to report and `open`/`xdg-open` hint; on macOS, print one-line note about GPU columns being empty if `--gpu` was passed (no CUDA available).

### Component 3: Smoke Test (`src/tests/scripts/test_profile_with_scalene.py`)

Single pytest file with one test that:

- Detects whether `scalene` is importable; if not, the test is skipped via `@pytest.mark.skipif`
- When Scalene is available, writes a tiny throwaway script, invokes the wrapper via subprocess, and asserts an HTML file is produced under the supplied output directory

Key property: when `scalene` is not in the environment (default CI install), the test is silently skipped — the existing CI test suite is unaffected (SC-003).

### Component 4: Documentation

Add one section to `CLAUDE.md` (or to a new `docs/profiling.md` if a docs directory grows for this purpose). The section briefly states:

- How to install: `uv sync --group profiling`
- How to invoke: `uv run python scripts/profile_with_scalene.py <target>`
- Pointer to `specs/20260503-235625-scalene-profiling/quickstart.md` for full usage

This satisfies FR-009 (worked example) and SC-004 (no external docs required).

### Out of Scope

- No senselab API additions (no `senselab.profiling` module)
- No optimization of senselab itself (this feature is the *measurement* tool; optimizations are separate features)
- No CI integration (profiling is a developer activity; CI runs without the profiling group)
- No GPU profiling on macOS (CUDA is not available there; documented as a known limitation)
- No automatic report comparison/regression-detection across runs (could be a future feature)

## Complexity Tracking

No constitution violations to justify. This is a deliberately small, additive change.
