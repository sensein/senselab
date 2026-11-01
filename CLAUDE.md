# CLAUDE.md

Guidance for work on **senselab 2.x** in this branch. 1.x guidance lives on `alpha`.

## Where things are

- **2.x code** is the uv workspace under `packages/`:
  - `senselab-worker`: stdlib + numpy, Python ≥3.10. Wire protocol, canonical encoding, digests.
  - `senselab-core`: Python ≥3.11. Kernel values and runtime: store, fetch, locks, provenance, worker client.
- **Specs and state** are in `specs/20261030-140022-senselab-2x-api/`. Resume from `STATE.md`. The
  principles are P-1..P-41 in `principles.md`, the decisions are ADR-001.. in `adr/`, and open findings
  are in `review-register.md`.
- **`src/` is the 1.x source pool, frozen and untested here.** A use case is ported by merging `alpha`, then a
  rename-only `git mv` commit into `packages/`, then a modify commit (`migration-ledger.md`). Nothing in
  `packages/` imports it. Triage's tasks and pipelines port like any other use case; its decision layer is
  superseded by the flywheel.

## Branches and CI

- 2.x work merges into the `2.x` integration branch, never `alpha`.
- CI is `.github/workflows/2x-tests.yaml` and `2x-docs-preview.yaml`. Previews appear under
  `/senselab/2x/pr-N/`. The 1.x workflows skip the 2.x bases.

## Commands

- Setup: `uv sync --all-extras --group dev`. `uv sync` is subtractive, so always pass the full set.
- Tests: `uv run --no-sync pytest <dirs you changed> -q`, run serially. Never `-n auto`: each worker
  loads the ML stack separately.
- Lint: `uv run lint-imports` and `uv run pre-commit run --all-files`.

## Local machine versus ORCD

- Locally, run only scoped tests and lint, in the existing `.venv`.
- Do not run full suites, create new venvs or download models locally. The full `packages` suite and the
  release-gate builds run on ORCD.
- Every probe bounds itself: finite input, a file-size limit and a timeout.

## Conventions

- **No backwards compatibility (P-14).** No shims, aliases, legacy readers or "historical" vocabulary in
  `packages/`. Migration is documented, not coded.
- **One implementation per behaviour (P-37).**
- **Review and compact before committing.** Read each change against `principles.md` and the `engineering.md`
  checklist, fold duplicates and cut verbosity during development. No model reviews in CI.
- **README states 2.x's current status;** update it in the same PR as any change to what 2.x delivers.
- **Rationale belongs in `specs/`.** Docstrings say what a thing is and how to call it.
- **Thresholds live in `data/`** with a written derivation, never as literals.
- **Every digest is tagged and registered** in `DIGEST_TAGS` (ADR-006). A value change bumps its tag.
- **Model loads pass a commit SHA, never a ref.**
- **Tests** are named `*_test.py` and carry exactly one tier marker. Each defect gets a regression plus a
  generated or planted contract over its class, not a one-case patch.
