# Reviewer brief: task ${TASK_ID}

A review that does not return the checklist table is returned unread.

## Principles

Read `${SPEC_DIR}/engineering.md` in full before the diff: the principles with their dated amendments, the
per-commit checklist, "What 'special-casing' means" and P-40. Check every changed line against them. Spec authority
is `${SPEC_DIR}`.

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

## Checklist

Return this table, one row per item, each `pass`, `fail` or `n.a.` with its evidence; a `pass` carries tool output
where a tool gives it (for P-34, the output of `scripts/check_fixed_parameters.py`). For the P-34 fixed-parameter
item, list every numeric or duration literal in the changed code: invariant (why), config (proposed key and home)
or measured (what to measure).

| Item | Verdict | Evidence |
|---|---|---|

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
- [ ] No user scenario in [`user-scenarios.md`](user-scenarios.md) that the change's area touches is broken; the
  PR names the scenarios checked, and an unchecked one is recorded as unchecked, never assumed to pass. A break
  is blocking unless the owner amends the scenario (added 2026-11-08, owner).

## Files

- The brief `${BRIEF}`, the implementer's report, the review package (commits, stat, full diff against
  `${BASE}`) and the worktree `${WORKTREE}`, read-only: no edits, no git writes, no subagents.
- Verdicts: Spec (every requirement met; nothing extra without a stated reason) and Quality (the table, plus:
  tests assert real behaviour and fail on the old code, with how that was checked; no patching of senselab
  internals; layering and import contracts hold; no 1.x-workflow concept without a stated 2.x need). Accept or
  reject each deviation the implementer listed, with the reason.
- Tests: targeted, serial, never `-n`, never `uv sync`, never full suites. A local run is evidence only in a venv
  without `STALE VENV`; otherwise rely on CI.
- Write the review to `${BRIEF_DIR}/task-${TASK_ID}-review.md`: findings ranked, each with a quoted `file:line`,
  severity (blocking / important / nit) and a concrete fix. Reply with only `Spec: pass|fail`,
  `Quality: approved|changes-needed` and one line per blocking or important finding. Be terse.
