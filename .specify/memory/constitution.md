<!--
Sync Impact Report
  Version change: 1.0.1 → 1.1.0 (add principle VIII: No Hardcoded Parameters)
  Modified principles: None
  Added sections: Principle VIII (No Hardcoded Parameters)
  Removed sections: None
  Anti-Patterns Registry: added #5 (hardcoded paths/values in scripts)
  Templates requiring updates:
    - .specify/templates/plan-template.md: ✅ compatible (Constitution Check section)
    - .specify/templates/spec-template.md: ✅ compatible
    - .specify/templates/tasks-template.md: ✅ compatible
  Follow-up TODOs: None
-->
# Senselab Constitution

## Core Principles

### I. UV-Managed Python (NON-NEGOTIABLE)

All Python execution MUST go through `uv` environments. No bare `python`,
`pip`, or `conda` commands. Every command uses `uv run`, `uv sync`, or
`uv build`. This ensures reproducible dependency resolution across
developer machines, CI runners, and SLURM nodes.

- `uv run pytest` — not `pytest`
- `uv run ruff check` — not `ruff check`
- `uv sync --extra ...` for environment setup
- `uv build` for package building, `uv publish` for release

### II. Encapsulated Testing

All tests MUST run in encapsulated environments. Acceptable isolation
boundaries: uv-managed virtualenvs, Docker containers, Podman, or
Apptainer/Singularity. Tests MUST NOT depend on host-installed packages
or global Python installations.

- Unit/integration tests: `uv run pytest` (virtualenv isolation)
- Tests requiring GPU, Docker, or specialized hardware: containerized
  runners (EC2 with Docker, Apptainer on SLURM)
- Pre-commit hooks: `uv run pre-commit run --all-files`

### III. Commit Early and Often

Commits MUST be small, focused, and frequent. Each logical change gets
its own commit. Do not batch unrelated changes. Commit messages MUST
describe what changed and why.

- Commit after each completed sub-task, not at the end of a session
- Prefer multiple small commits over one large commit
- Never skip pre-commit hooks (`--no-verify` is forbidden unless
  explicitly authorized by the user)

### IV. CI Must Stay Green

Every PR MUST pass CI (pre-commit + macOS-tests at minimum) before
merge. Do not merge with known failures. If CI fails, diagnose and
fix before proceeding.

- Monitor CI status after every push
- Do not set up polling/watch loops for CI — check once, fix if
  broken, re-push
- Flaky tests MUST be fixed, not ignored or retried blindly

### V. Memory-Driven Anti-Pattern Avoidance

Behaviors, mistakes, and patterns that caused problems MUST be
recorded in Claude Code memory (feedback type) so they are not
repeated in future sessions. Check memory before starting work.

- After a bug fix or failed approach: save what went wrong and why
- Before making a change: check if memory has relevant warnings
- Anti-patterns include: untested changes pushed to CI, mocking
  leaks between tests, circular imports, debug print statements
  left in code

### VI. No Unnecessary API Calls

External service calls (HuggingFace Hub, PyPI, GitHub API) MUST be
minimized. Use local caches, file-based locks, and
`local_files_only` modes when models are already available.

- HuggingFace models: use `ensure_hf_model()` for coordinated
  download with cross-process locking
- All `from_pretrained()` / `pipeline()` calls: pass
  `local_files_only=True` when model is cached
- Retry transient errors with exponential backoff; cache
  definitive failures

### VII. Simplicity First

Start with the simplest solution that works. Do not add abstractions,
configuration options, or indirection until they are needed. YAGNI.

- No feature flags or backward-compatibility shims when you can
  just change the code
- Three similar lines of code is better than a premature abstraction
- Do not add error handling for scenarios that cannot happen

### VIII. No Hardcoded Parameters

Values that vary by environment, user, or deployment MUST be
configurable — not embedded as literals in code or scripts. During
code review, actively scan for hardcoded paths, URLs, version
strings, instance types, usernames, and similar values. Extract
them as CLI parameters, environment variables, or config file
entries.

- File paths: accept as parameters or derive from environment
  variables (e.g., `$HOME`, `$HF_HOME`, `$WORKING_DIR`)
- Cloud resources: AMI IDs, instance types, regions, key names
  MUST be parameters, not inline constants
- SSH/connection details: usernames, key paths, ports MUST be
  auto-detected from context or accepted as parameters
- When adding a parameter: provide a sensible default so the
  common case requires zero configuration
- When reviewing code: ask "would this break on a different
  machine, account, or OS?" — if yes, parameterize it

## Environment & Tooling

- **Python**: version(s) as specified in `pyproject.toml`, managed via uv
- **Package manager**: uv (not pip, not conda, not poetry)
- **Linting**: ruff (check + format), mypy, codespell
- **Testing**: pytest with coverage, run via `uv run pytest`
- **Pre-commit**: mandatory, installed via `uv run pre-commit install`
- **CI**: GitHub Actions (macOS arm64 for unit tests, EC2 for GPU tests)
- **Versioning**: hatch-vcs (git tags), Intuit Auto for release automation
- **Docs**: pdoc with Google-style docstrings

## Anti-Patterns Registry

Known anti-patterns discovered in this project. Violation of any
of these MUST be flagged during review:

1. **Mock cache pollution**: Test-scoped mocks that modify class-level
   caches (`_pipelines`, `_hf_cache`) MUST use `monkeypatch.setattr`
   to swap in a fresh dict, not `.clear()` on the shared dict.
2. **Circular imports**: `dependencies.py` MUST NOT import from
   `senselab.utils.data_structures` (use stdlib `logging` directly).
3. **Debug print statements**: `print()` calls MUST NOT be committed.
   Use the `logger` from `senselab.utils.data_structures.logging`.
4. **Broad exception catches for optional imports**: Use
   `(ImportError, RuntimeError)` not just `ModuleNotFoundError` for
   imports that may fail due to native library issues (torchcodec).
5. **Hardcoded paths and values in scripts**: SSH key paths
   (`~/.ssh/name.pem`), SSH usernames (`ec2-user`), device names
   (`/dev/xvda`), and AMI-specific assumptions MUST be auto-detected
   or accepted as parameters. Discovered in `setup-gpu-ci.sh` where
   the `--build-ami` mode originally hardcoded all of these.

## Governance

This constitution supersedes ad-hoc practices. All PRs and code
reviews MUST verify compliance with these principles. Amendments
require:

1. Documentation of the change and rationale
2. Version bump (semantic: major for removals/redefinitions, minor
   for additions, patch for clarifications)
3. Update to this file and propagation to dependent templates

Anti-pattern additions do not require version bumps — they are
operational notes appended as discovered.

**Version**: 1.1.0 | **Ratified**: 2026-04-18 | **Last Amended**: 2026-04-18
