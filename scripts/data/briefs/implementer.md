# Implementer brief: task ${TASK_ID}

## Principles

Read `${SPEC_DIR}/engineering.md` before coding: the principles with their dated amendments, the per-commit
checklist and "What 'special-casing' means". Spec authority is `${SPEC_DIR}` (design.md, adr/, execution.md,
testing.md, config-derivations.md, the phase plan's rulings); read what the task cites.

- P-31 Minimal verbosity: say each thing once, in the fewest exact words; reasons live in `specs/`.
- P-32 Clean, robust, reusable, efficient code: deep modules, functional core, algorithms fit the cost model.
- P-33 Profiling at every scale: per-item cost is measured and stays flat with scale.
- P-34 Brittleness is checked: bad-input and fault tests; no fixed parameters (config with a derivation, or a
  declared invariant); execution is measured before it is scaled.
- P-35 Adversarial breaks are fixed by architecture, never by a patch for the found case.
- P-36 The flywheel informs the roadmap; it does not force it.
- P-37 One implementation per behaviour; variants are parameters, strategies or backends.
- P-38 Hashes establish exactness; reuse across contexts is measured.
- P-39 1.x work is reconciled: check the ledger mark of every 1.x area touched.
- P-40 Agents are sized to the task and reviewed by a different model in a fresh context.
- P-28 No identifiers, no local paths: placeholders from `references.example.yaml`. P-29 Tests are scoped and
  serial, never `pytest -n`; cluster limits hold.
- The task's other P-ids and ADRs: ${TASK_PIDS}. Test tier expected (`testing.md`): ${TIER}.

## Checklist

The report carries one line per item (`pass | fail | n/a — reason`) and lists every numeric or duration literal
added with its verdict (invariant / config key / measured).

- [ ] Prose is minimal: no history, no ids, docstrings say what and how (P-31).
- [ ] No duplicated logic; surfaces are thin over shared engines (P-32).
- [ ] No second function for an existing behaviour, and no copied variant. Variants are
  parameters, strategies or backends, with no caller-specific branches (P-37).
- [ ] Per-item time and peak memory at 1 and 10² items are measured for touched modules and quoted in the PR
  (`benchmarks/scale/` once Phase 3 Task 20 lands; until then `scripts/profile_with_scalene.py` output or a timed
  test). An unmeasured change says so (P-33).
- [ ] New boundaries have bad-input, fault and property tests; there is no special-casing (P-34).
- [ ] No fixed parameters: every numeric literal in runtime, execution and test-bound code is either an invariant
  in the form P-34 defines or config (profile, run config or data file) with its derivation; sizing comes
  from measurement before scaling (P-34, amended 2026-11-07).
- [ ] No identifiers, no local paths and no content copied from a referenced local file in any committed text: code, specs, tests, data, commit messages, PR bodies; placeholders instead (P-28). Tests are scoped and serial (P-29).
- [ ] Every new task, backend or profile is declared, so contract tests cover it (P-30).
- [ ] The PR records the implementer and reviewer models; the reviewer ran in a fresh context on a different model, or both are the top tier on a kernel item and the PR says so (P-40).
- [ ] The ledger mark is current for any 1.x area touched (P-39).

## Files

- Requirements: `${BRIEF}`. Exact values come from it verbatim.
- Worktree: `git -C ${IMPL_CHECKOUT} worktree add -b ${BRANCH} ${WORKTREE} ${BASE}`, then link the shared
  venv: `ln -s ${IMPL_CHECKOUT}/.venv ${WORKTREE}/.venv`. Commit there, one commit per unit of work. Do not push,
  merge, tag, touch other branches, `git worktree prune` or `git stash`.
- Rename-only commits `git mv` whole files and nothing else; edits follow in the next commit. Never edit `src/`.
- Run Python as `PYTHONPATH=<each packages/*/src of the worktree> uv run --no-sync ...`; never `uv sync`, never a
  new venv, never a model download, never a full suite, never `pytest -n`. Run the test files touched and the
  package's `-m "logic or contract"`, serially. A run whose venv prints `STALE VENV` is not evidence.
- Steps marked `orcd` are the controller's: list the command, partition, GPU type and the output that proves it.
- Before each commit: review the diff against the checklist and compact it (fold duplicates, cut verbosity, no
  rationale in code, no shims, thresholds in data with a derivation). Run `ruff check`, `ruff format` and
  `lint-imports` on what changed; hooks run unless the worktree breaks one, and then the report says which and
  its check is run by hand. A changed Frozen or model type runs the value-contract and testing suites.
- Tests: test-first, one tier marker each, no patching senselab internals, each regression fails on the old code.
  Optional frameworks are imported inside the test function with `importlib.import_module`.
- Spec edits go to `${BRIEF_DIR}/task-${TASK_ID}-spec.patch` (git diff, paths from the repository root), never
  into the code worktree. Commit messages name no reviewer and no review status.
- No subagents. Report to `${BRIEF_DIR}/task-${TASK_ID}-report.md`: commits, tests and results, decisions,
  deviations, `orcd` steps, concerns. Reply with only: status (DONE / DONE_WITH_CONCERNS / BLOCKED /
  NEEDS_CONTEXT), the commit list and a one-line test summary. Be terse.
