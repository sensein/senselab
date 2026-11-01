# Full review, delivery surface (CI, docs, releases, branches, state) at 3a468aff

Read-only review. Authority: owner-requests.md items 11, 25, 26, 30, 31, then principles, ADRs, design/docs/testing.md.
Spec dir below is `specs/20261030-140022-senselab-2x-api/` (abbreviated `S/`).

Counts: Critical 0, Important 6, Minor 7.

## What landed correctly (checked, no finding)
- Items 30/31: #621 merged into `2.x` (2f32aa8e); #620 merged into alpha. `2.x` carries `2x-tests.yaml`, `2x-docs-preview.yaml`, the namespaced `docs-deploy.sh`, `fetch-1x-freeze.sh`, `check-2x-packages.sh`. `.github` on `2x/phase-1` differs from `2.x` only by `publish.yaml` and a trimmed `2x-tests.yaml`.
- Separation: `tests.yaml` and `docs-preview.yaml` use `branches-ignore: [2.x, senselab-2x-api, 2x/**]`; the 2x workflows trigger only on those bases. Previews are namespaced (`/senselab/2x/pr-N/`), and cleanup of one namespace cannot touch the other (`docs-deploy.sh`).
- `2x/phase-1` contains `2.x` and `senselab-2x-api`. No 2.x PR targets alpha: #617 -> 2.x, #619 -> senselab-2x-api. #618, #612, #613, #608, #609 and #607 are triage PRs stacked on alpha, which is the 1.x line.
- Leakage in the 2.x tree: `git grep` of `packages/`, `scripts/` (2.x ones), `.github`, CLAUDE.md, README and the 2.x spec dir shows no participant ids, personal paths or emails. The `/home/x/a.wav` strings are synthetic test inputs. The only emails are the three author emails in `pyproject.toml`, which are public authorship.

## Important

### D-1. `senselab-2x-api` (PR #617) still does not carry the 2.x workflows, so #617 runs only 1.x CI
- Owner item 31: "any workflow updates for the 2.x work. this should be kept separate from any 1.x workflows".
- `git merge-base --is-ancestor origin/2.x origin/senselab-2x-api` is false, and `origin/senselab-2x-api` has no `2x-*.yaml`. `gh pr view 617` rolls up `macOS-tests`, `tutorial-cpu-tests`, `cpu-tests`, `gpu-tests`, `start-runner`/`stop-runner` and a 1.x `build-and-deploy` (SUCCESS, so it deployed a 1.x root `pr-617/` preview), and no `2x-tests` jobs. The 1.x `branches-ignore` fix is on `2.x` only.
- Deviation (a 2.x branch needs `2.x` merged in). Correction: merge `origin/2.x` into `senselab-2x-api`, or merge #617 into `2.x` first. Then check #617 produces `2x-*` runs and delete the stray root `pr-617/` preview.

### D-2. STATE.md is stale and wrong on the branch model, PR bases, ledger path and ADR count
- Owner item 26: "make sure a state is maintained where this or future sessions can pick up from". Item 30: "should be the former [2.x]".
- `S/STATE.md`:
  - "Design branch ... draft PR sensein/senselab#617 to `alpha`" (line 7). #617 targets `2.x`.
  - "one PR per task to `alpha`" and `gh pr list --base alpha --head '2x/'` (Next actions 1, How to resume 3). 2.x work merges into `2.x`.
  - "SDD ledger `.superpowers/sdd/phase-1/progress.md`" (line 12). The live ledger is `phase-1-hardening/progress.md`.
  - "ADR-001..009" (Document map). ADR-010 exists and the log omits it.
  - Phase row says "T0-T15a done ... H1-H8 before Phase 1 exit". The ledger shows H1-H10 done (H9a-f, H9-compat, H10), H11 test conformance queued, and CI green at 97b617c5. STATE does not mention H-R1..H-R52, #619, #620, #621, the `2.x` integration branch, H11, or the full review now running.
  - Log entry for 2026-11-01 stops at H-R50.
  - Resume step 1 points at `~/software/sensein/senselab-2x` on `senselab-2x-api`, not the impl worktree on `2x/phase-1`, and gives personal paths (`~/software/...`) rather than a clone recipe.
- The ledger and STATE disagree, and STATE is the one a fresh session reads. Deviation. Correction: rewrite Where-things-are, Status, Next actions and How-to-resume from the current facts, and log H-R48..H-R52 and ADR-010.

### D-3. Plans and engineering.md still direct 2.x work to `alpha`
- Owner item 30: 2.x merges into `2.x`. (Item 11.2 "in place on alpha" was superseded by it.)
- Still saying alpha:
  - `S/engineering.md:127-128` ("2.x is built in place on `alpha`"; "Source: owner ... 'in place on alpha'" with no supersession note).
  - `S/plans/phase-1.md:5,63,200` (`Land ... on alpha`, "one PR from `2x/p1-<slug>` to `alpha`").
  - `S/plans/phase-1-hardening.md:67` (PRs to `alpha`), `:775` (`tests.yaml` adds `push ... 2x/**`; the CI was built differently, in `2x-tests.yaml`), `:796` (protect `alpha` and `main`; no mention of `2.x`).
  - #619's body: "Merging into `alpha` is left to the owner."
  - #617's body: "P-1..P-29", "Docs only" (now P-1..P-41, plus the code beneath it).
- Drift. A later owner ruling contradicts earlier text that was never amended. Correction: mark the owner source line as superseded by item 30, replace `alpha` with `2.x` in the plans, add `2.x` to the branch-protection step (A-39 owner action), and refresh the PR bodies.

### D-4. docs.md says the preview is "the existing `docs-preview.yaml`" at `pr-<N>/`; the branch built a separate workflow at `2x/pr-<N>/`
- `S/docs.md:47` ("a preview at `sensein.group/senselab/pr-<N>/`, which the existing `docs-preview.yaml` provides. 2.x extends it from the pdoc API reference to the full site"). The branch has `2x-docs-preview.yaml` at `/senselab/2x/pr-N/`. This follows the owner's separation (item 31) but docs.md was not amended. Drift. Correction: edit docs.md to the 2x workflow and URL. docs.md also says nothing of the namespacing.
- Content check against item 25 ("clear structure ... for scientists/clinicians/engineers with rich examples using public data"): the preview is only `pdoc senselab_core senselab_worker` (2x-docs-preview.yaml). There is no audience structure, no example, no data fetch in the docs build, and no 2.x status page. docs.md phases the full site at Phase 2 and examples at Phase 3 onwards, so this is "Phase 1 never meant to build it", and the preview-for-communication goal of item 25 is met only for API reference. Phase 1 per docs.md owes the manifest and fetch (built, `senselab data fetch`), the fixture budget check (built, pre-commit `check_fixture_budget.py`) and generators (built). The toy-versus-pulled decision is implemented as policy and tooling, and no CI job fetches a public dataset yet, so it has not been exercised end to end. Note this explicitly in STATE, because the owner asked for the preview as a way to convey progress.

### D-5. CI cadence differs from testing.md
- `S/testing.md:47-52,176-182`: `limits` is label-triggered and nightly, `usecase` is nightly plus ORCD before release, and PRs run logic and contract. The full Python matrix, scale ladder and CPU smoke benchmarks are nightly, and the benchmark smoke runs on PRs touching a task.
- `2x-tests.yaml` `core-only` runs `-m "logic or contract or limits or usecase"` on every PR (3.11 and 3.14) and `core-env` runs the `env` tier on every PR. There is no `schedule:` trigger and no `workflow_dispatch` anywhere in `.github/workflows/2x-*`; no nightly, ORCD job or benchmark smoke exists. H-R49 already queues "CI cadence per testing.md", so this is an open omission against a ruled task, not a surprise. Correction: tier-split the selection and add the scheduled workflow, or amend testing.md to the cadence actually built.

### D-6. Everything a later session needs about the owner's words and rulings lives only in a git-ignored ledger
- Owner item 26. `.superpowers/` is excluded through `.git/info/exclude` only (not a tracked `.gitignore`), so it is neither in the repo nor in any clone: `owner-requests.md` (38 verbatim items, the highest authority in this review's brief), `progress.md` (H-R1..H-R52), the task reports and reviews.
- The tracked tree cites those ids in 43 places (`H-R29`, `H-R30`, `H-R37`, `H-R40` in test docstrings in `packages/`; 14 in ADR-006; also ADR-003, 009, 010, design, migration-ledger, review-register). A reader of the repository cannot resolve them. STATE.md names the ledger once and gives the wrong path. CLAUDE.md does not mention the ledger or owner-requests.
- Correction: commit the owner's verbatim requests as `S/owner-requests.md` (they are already referenced as the source of principles) and fold the H-R rulings into `review-register.md` or an ADR, or stop citing H-R ids from `packages/` (P-21 puts rationale in `specs/`, and a bare ruling id in a test docstring points nowhere). Test docstrings with round ids is the same class H-R49 already lists.

## Minor

### D-7. `publish.yaml` and the release gate are far larger than design §5's statement of the gate
- Design §5 ("Release gate (H7, H9f)") is itself a multi-page list of mitigations, not a statement of requirements. `scripts/release_gate.py` is 1,092 lines (names, versions, member lists, environment allowlists, dirty-tree handling, sdist-to-wheel rebuilds, install from `dist/` with `--no-index`). `publish.yaml` does: build all, gate, `uv publish` only on `release`, with `workflow_dispatch` running the gate only. That matches §5 and does nothing more. Against P-41/owner item 14 ("code itself should be clear to read and compact") the gate is the largest single script on the line. Not a delivery bug; recommend a size/scope review in H11 and moving the §5 rationale to a spec file.
- No 2.x release trigger exists. `release.yaml` fires only on merged PRs to `main` or `alpha`, so a 2.x tag or release can only be made by hand, and `docs.yaml` (release) still builds `src/` pdoc with the compat matrix on the 2.x branch. Decide: add a 2.x release workflow or write the manual procedure in STATE. A 1.x release created from `alpha` is unaffected because release events use the tagged ref's workflow files.

### D-8. The umbrella root distribution is consistent with item 11.2 but is not what design §5 describes
- Item 11.2: "core can be its own distribution". `senselab-core` and `senselab-worker` each have their own `pyproject.toml`, are built by `uv build --all-packages`, and are built alone in CI (`core-only`, `worker-floor`, the sdist-to-wheel loop). Consistent.
- Root `pyproject.toml` is metadata-only: `description = "installs the framework-free core and the subprocess worker runtime"`, `bypass-selection = true`, deps written by `hatch_build_pins.py` for exactly those two. design.md §5 still says the root "holds backends, the catalogue and pipelines, and whose facade (`import senselab as sl`) imports from core." Phase 1 does not build that, so this is Phase 1 "not built yet", but the design text does not say the root is empty until Phase 2, and `[project.scripts] senselab = "senselab_core.cli:main"` is declared by the root while the code lives in core. Correction: one sentence in §5 on the interim umbrella (H-R48/H-R50 are the only record).
- The root `pyproject.toml` still carries the `[tool.mypy] exclude = "^src/"` and `[tool.ruff]` blocks of the 1.x tree and a `[dependency-groups] profiling` group (scalene, nbconvert) that nothing in `packages/` uses.

### D-9. README and CLAUDE.md accuracy
- README and CLAUDE.md are 2.x-specific and consistent with `2.x`, `2x-*.yaml` and the package layout (`senselab-worker` >=3.10 and `senselab-core` >=3.11 match the package `pyproject.toml` files). No 1.x sections remain.
- Stale or unverifiable: README "API docs of a pull request: .../2x/pr-<N>/" is correct; README "Setup: `uv sync --all-extras`" with no extras defined (harmless but cargo-culted from 1.x); CLAUDE.md "Lint with `uv run lint-imports`"; CLAUDE.md says "Resume from `STATE.md`", and STATE is stale (D-2), so the resume path leads a fresh session wrong. CLAUDE.md does not name the ledger, owner-requests or the review-register's open items. README says "Phase 1 delivers ... `senselab-core`" and does not mention ADR-010 W3C PROV (the model is under change); say so under Status when ADR-010 is implemented.
- CLAUDE.md "Local machine versus ORCD" gives no pointer to the ORCD recipe or partitions (owner item 11.4 "orcd partitions in mind"), which exist only in the owner's memory.

### D-10. Personal paths and owner username remain in the tracked 1.x tree
- Owner item 11: "ensure no identifiers leak in". The 2.x paths are clean (above). The inherited 1.x pool and scripts still carry them: `scripts/triage_review_serve.py:10` (<cluster path>), `scripts/triage_review_page.py:20`, `scripts/verify_derivatives_copy.py:5,167`, `scripts/speaker_ceiling/*.sbatch` (`~<cluster path>, `hf-cache-$USER`), `.github/workflows/tests.yaml:60,273` (runner paths; harmless), and 20+ lines in 1.x spec task files (`specs/20260503-235625-scalene-profiling/tasks.md`, `specs/20260506-154425-audio-analysis-asr-extensions/tasks.md` with `<home>/software/sensein/senselab/...`). `src/` files `triage/data/breath_pattern.yaml`, `crisperwhisper_test.py`, `verify_derivatives_copy_test.py` and `review.test.mjs` match <cluster path> or an 8-hex `sub-` pattern, but pre-commit excludes `^src/`, so `check_participant_ids.py` (whose hook is repo-wide otherwise) does not guard that pool. The script's own scan (`uv run python scripts/check_participant_ids.py`) was not run here (read-only, no venv). Correction: run it against `src/` as a one-off before any use case is ported out of the pool, and scrub the personal paths from the 1.x spec files and scripts.

### D-11. #619 CI state
- `gh pr view 619` shows `core-env (release gate and packaging)`: FAILURE next to a skipped duplicate. The rollup may predate 97b617c5 (the ledger reports all green there). Re-check on the head SHA before merge.

### D-12. 2x-docs-preview has no PR-comment or status link
- Item 25 wants the preview to convey ongoing work. The workflow sets an environment URL only. Add the preview URL as a PR comment or job summary so a reviewer finds it.

### D-13. `2x-tests` `pre-commit` job installs with `uv sync --all-extras --group dev` and fetches the 1.x freeze
- This installs the whole 1.x dependency set (torch etc. via the root metadata? the root has no 1.x dependencies, so the cost is dev-only). The ledger shows it green. No action beyond confirming run time stays low as the dev group grows.
