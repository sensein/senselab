# senselab 2.x: working state

Read this first when resuming. Update it at the end of every work session, and whenever a phase,
task or decision changes state. Newest entries go first in the log.

## Where things are

| What | Where |
|---|---|
| Design branch (specs only) | `senselab-2x-api` → draft PR sensein/senselab#617 to `alpha` |
| Durable worktree | `~/software/sensein/senselab-2x` (branch `senselab-2x-api`) |
| 1.x freeze tag (Phase 0) | `freeze/1x-2026-10-30` → `19834fdd` (= `1.4.0-alpha.11`) |
| pydra proposal | branch `enh/execution-extensions` → draft PR nipype/pydra#891; worktree `~/software/nipype/pydra-ext` |
| Implementation branch | `2x/phase-1` (worktree `~/software/sensein/senselab-2x-impl`), stacked on `senselab-2x-api`; one reviewed commit series per task; all spec edits during Phase 1 go here too (SDD ruling R2) |
| SDD ledger | `.superpowers/sdd/phase-1/progress.md` in the impl worktree (git-ignored; task completion lines and rulings) |
| Phase plans | `plans/phase-1.md` (this directory) |

## Document map (this directory)

- **Rules:**
  - `principles.md`: P-1..P-41; the register.
  - `engineering.md`: the standards, the per-commit checklist, the agent and model roster (P-40),
    and P-27..P-35, P-39, P-40.
- **What to build:**
  - `design.md`;
  - `adr/` (ADR-001..009);
  - `execution.md`;
  - `testing.md`;
  - `docs.md`.
- **Why:**
  - `review-register.md` (R-1..R-54, all resolved);
  - `architect-review.md`;
  - `engine-choice.md`;
  - `workflow-engines.md`;
  - `use-cases.md`;
  - `benchmarks.md`;
  - `roadmap.md` (the flywheel).
- **Migration:** `migration-ledger.md`, with a mark per 1.x area (P-39).

## Status

| Phase | State |
|---|---|
| Design | done; owner approved 2026-10-30 ("go ahead") |
| 0: freeze tag | done (`freeze/1x-2026-10-30`) |
| 1: kernel, worker runtime, identity, store, data fetch | executing (subagent-driven); T0–T3 done (workspace, wire protocol, canonical serialisation) |
| 2–5 | not started |
| pydra PRs (A0…K1) | proposed in #891; none implemented |

## Next actions

1. Finish `plans/phase-1.md` and execute it task by task with subagents (roster in
   `engineering.md`), one PR per task to `alpha`.
2. Run the P-39 reconciliation scan before Phase 1 starts (`migration-ledger.md`).
3. Start pydra PR A0 (Slurm `--error` fix) and D1 (`ResultStore` interface) in parallel. Neither
   blocks Phase 1.

## Open owner decisions

None open. Owner calls from 2026-10-30 are folded into the docs: reuse default `equivalent`,
vLLM-omni kept, license taint kept and combinators deferred, prose review by model and agents.

## How to resume

1. `cd ~/software/sensein/senselab-2x && git pull`. Read this file, then `principles.md`, then
   `engineering.md` (the roster and checklist).
2. Check drift: `git fetch origin alpha && git log <ledger mark>..origin/alpha --oneline`.
3. Open the current phase plan and continue from the first unchecked task. Its PR state is on
   GitHub (`gh pr list --base alpha --head '2x/'`).
4. Brief every delegated task as `engineering.md` requires.

## Log

- **2026-10-30 (Phase 1 start):** P-39 scan found no drift (`origin/alpha` = `19834fdd` = every
  ledger mark). Execution began on `2x/phase-1`.

- **2026-10-30:**
  - Design completed through five reviews and an architect review.
  - ADR-001..009 written.
  - Phase 0 tag pushed.
  - Worktrees moved to durable paths.
  - Phase 1 planning started.
