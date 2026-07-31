# Research: Scalene-Based Profiling Tooling

**Branch**: `20260503-235625-scalene-profiling` | **Date**: 2026-05-04

## Profiler Choice

### Decision: Scalene (already named in user input)

**Rationale**: Scalene was explicitly chosen by the user. It provides line-level CPU, memory, and GPU profiling with low overhead, native HTML and JSON output, programmatic API, and a `@profile` decorator for scoped profiling. It is well-maintained and supports Python 3.11+.

**Alternatives considered**:
- `cProfile` + `snakeviz`: stdlib, but no memory or line-level info
- `py-spy`: sampling-only, no memory profiling
- `memray`: excellent memory profiling but no CPU breakdown
- `line_profiler`: line-level CPU only, no memory

## Dependency Group Placement

### Decision: New `profiling` group under `[dependency-groups]`

**Rationale**: senselab uses two dependency systems:
- `[project.optional-dependencies]` for user-facing feature extras (nlp, text, video, senselab-ai)
- `[dependency-groups]` for developer-only tools (dev, docs)

The Scalene profiler is purely a developer tool — never invoked by end users of senselab. It belongs in `[dependency-groups]` alongside `dev` and `docs`.

**Install command**: `uv sync --group profiling`

**Default `uv sync`**: unchanged — Scalene is opt-in only, satisfying SC-003.

**Alternatives considered**:
- Adding to `[project.optional-dependencies]`: would imply user-facing extra; mismatched semantics
- Adding to existing `dev` group: forces all developers to install Scalene; users running tests don't need it
- Documenting `pip install scalene` outside the project: violates FR-008 (must use existing dep workflow)

## Scalene Invocation Pattern

### Decision: Use the `scalene` CLI directly via `uv run`, wrapped by a single Python launcher script

**Rationale**: Scalene exposes itself as both a console script (`scalene script.py`) and a module (`python -m scalene script.py`). The `uv run scalene` form picks up the project virtualenv automatically. A thin Python wrapper at `scripts/profile_with_scalene.py` parses high-level options (target file, output format, output path, GPU flag) and dispatches to `scalene` via subprocess, so developers don't need to memorize Scalene's full flag set.

**Alternatives considered**:
- Programmatic API (`scalene_profiler.start()/stop()`): only useful for in-script scoped profiling; not suitable for one-shot CLI invocation of arbitrary scripts
- Direct `scalene` CLI without wrapper: works but means developers need to remember flags; defeats SC-001 (5-minute cold-start usability)

## Notebook Profiling Approach

### Decision: Convert `.ipynb` to `.py` via `jupyter nbconvert`, then run Scalene on the `.py`

**Rationale**: Scalene's `%%scalene` IPython magic only works inside an interactive Jupyter session. For batch profiling of tutorial notebooks (which is the primary use case for senselab), conversion to `.py` is reliable, cross-platform, and gives full Scalene memory profiling support. The `nbconvert` tool is already a transitive dependency via Jupyter (in `senselab-ai` extra) — no new dependency required.

**Workflow**: when the wrapper script detects a `.ipynb` extension, it runs `jupyter nbconvert --to script <notebook>` to produce a `.py` file in a temp directory, then profiles that.

**Alternatives considered**:
- Papermill execution + Scalene wrapping: unnecessary complexity; papermill adds parameterization features we don't need
- IPython magic: not usable for non-interactive batch runs
- Manual conversion: defeats SC-001 (5-min usability) and SC-004 (one-command workflow)

## Scoped Profiling Mechanism

### Decision: Use Scalene's `--profile-only` and `--profile-exclude` flags via wrapper CLI options (`--scope`, `--no-thirdparty`, `--exclude`); do not build custom abstractions

**Rationale**: Scalene 2.2.1's `run` subcommand exposes `--profile-only SUBSTR` and `--profile-exclude SUBSTR` as the supported scoping mechanism. Both match against **file paths** (substring match, comma-separated for multiple values), not function names. Path-substring matching is sufficient for senselab's use case — developers typically want to scope by module path (e.g., `senselab.audio.tasks.speech_enhancement`) rather than by function name.

**Update from earlier draft (verified against Scalene 2.2.1 CLI)**: An earlier version of this research mentioned an `@profile` decorator and a `scalene_profiler.start()/stop()` programmatic API as alternative scoping mechanisms. These primitives may exist in older Scalene releases or in undocumented form, but they do not appear in Scalene 2.2.1's `--help` output. The wrapper does not depend on them. Quickstart and contracts/cli.md document only the verified `--scope` / `--no-thirdparty` / `--exclude` path.

**Alternatives considered**:
- Custom `@senselab_profile` decorator wrapping Scalene primitives: depends on undocumented internals; rejected
- Custom `with profile_region(name):` context manager: same risk; rejected

## Output Format and Location

### Decision: Default to HTML output in `artifacts/scalene/<target_name>_<timestamp>.html`; allow `--format json` for machine-readable output

**Rationale**:
- HTML is interactive (sortable, drill-down) and works in any browser — matches FR-002 ("human-readable")
- The `artifacts/scalene/` subdirectory keeps profiling outputs grouped, separate from the existing `artifacts/import_profile_report.md` from the previous feature (FR-010)
- Timestamped filenames prevent overwrites when developers profile iteratively
- JSON output is supported for integrations (CI tooling, custom dashboards) but is not the default

**Alternatives considered**:
- CLI/text default: less useful for non-trivial profiles; HTML is industry standard
- Single-overwrite filename: loses history; iterative profiling needs distinct output paths
- Storing under `scripts/` or another location: violates FR-010 (separation from source)

## macOS ARM64 Compatibility

### Decision: Use Scalene with default settings on macOS ARM64; document the known limitations

**Rationale**: Scalene works on macOS ARM64. Known limitations:
- No child-process profiling under Jupyter on macOS — but our approach converts `.ipynb` to `.py` first, sidestepping this
- GPU profiling unavailable (no CUDA on Mac) — but the spec says GPU profiling is opt-in (FR-006)

The wrapper script will detect macOS and emit a one-line note in stdout reminding the user that GPU columns will be empty.

**Alternatives considered**:
- Disabling memory profiling on macOS: not necessary; works fine
- Documenting "Linux-only": unnecessarily restrictive

## Error Handling for Missing Scalene

### Decision: The wrapper script imports `scalene` at startup; if `ImportError` is raised, print a one-line install hint and exit non-zero

**Rationale**: This satisfies FR-007 — when Scalene is not installed (default `uv sync` without the profiling group), the wrapper fails fast with the actionable message: `"Scalene is not installed. Run: uv sync --group profiling"`.

**Alternatives considered**:
- Lazy import + auto-install: violates the principle of explicit user consent for installs
- Silent fallback to cProfile: changes semantics; user asked for Scalene specifically

## Test Strategy

### Decision: Add a small smoke test under `src/tests/scripts/test_profile_with_scalene.py` that:
1. Invokes the wrapper on a tiny throwaway script (e.g., a script that sleeps for 0.1s and allocates a list)
2. Asserts the expected HTML report file is created
3. Asserts the wrapper exits 0
4. Marks the test as `@pytest.mark.skipif(...)` when Scalene is not installed (default CI install path)

**Rationale**: This satisfies the clarification's "tests for the profiling tool itself" while ensuring it does not break the existing CI suite (which runs without the `profiling` group). The `skipif` guard means the test silently skips on default installs.

**Alternatives considered**:
- No tests: clarification permits but doesn't require tests; smoke test is cheap and prevents regressions
- Snapshot-test the full HTML report: brittle; Scalene's HTML format may change between versions
