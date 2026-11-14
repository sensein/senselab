# Phase 3 pydra decision gate: evidence (2026-11-03)

Gate of design §4.9, run as Phase 3 Task 0. Pinned commit `P` = nipype/pydra `enh/execution-extensions`
`0c1bbcd9f5606651c238672ff241912e7778c8d7`.

Line format: `<item> <commit or PR> <CI run> <ORCD job> <verdict>`.

| Item | Commit or PR | CI run | ORCD job | Verdict |
|---|---|---|---|---|
| D1 | #894 (closed), commits merged by `3189d7f2` | 41606617714 (full tier at `P`) | 27757266 | in the branch at `P` |
| G1 | #895 (closed), commits merged by `3189d7f2` | 41606617714 | 27757266 | in the branch at `P` |
| K1 | #896 (closed), commits merged by `3189d7f2` | 41606617714 | 27757266 | in the branch at `P` |
| A1 | #897 (closed), commits merged by `3189d7f2` | 41606617714 | 27757266 | in the branch at `P`; real Slurm submissions pass |
| F1 | #898 (closed), commits merged by `3189d7f2` | 41606617714 | 27757266 | in the branch at `P`; real Slurm submissions pass |
| F2 | #899 (closed), commits merged by `3189d7f2` | 41606617714 | 27757266 | in the branch at `P` |
| E1a | #900 (closed), commits merged by `3189d7f2` | 41606617714 | 27757266 | in the branch at `P` |
| E1b | #901 (closed), commits merged by `3189d7f2` | 41606617714 | 27757266 | in the branch at `P` |
| K2 | #905, merge `0c1bbcd9` (Task 40, P3-1) | 41605877229 (PR head), 41606617714 | 27757266 | merged at `P` |
| workflow-construction cache | #903, merge `2b24ac27` | 41598460942 | — | merged; its run's one failure fixed by #904 |
| cf start method | #904, merge `277bee99` | 41599491906 | — | merged |
| CI tiers | #902, in the branch; open to `main` | — | — | merged into the branch |
| branch to `main` | #891, draft | — | — | open |

## Criteria

1. **Merged at `P`:** A1, D1, E1a, E1b, F1, F2, G1, K1 and K2: met.
2. **pydra full CI tier at `P`:** run 41606617714, dispatched manually on the branch (Python 3.11–3.15 on three OSes, `cf`, Slurm, singularity, lmod,
   docs), success: met.
3. **ORCD at `P`:** job 27757266, CPU node, locked uv-managed Python 3.13.14 venv built in the job, `pytest pydra -k
   "store or checksum or reuse or readiness or on_error or resources or router or slurm"` with `debug`, `cf` and
   `slurm` workers: 234 passed, 11 skipped (singularity, SGE, two upstream skips), 0 failed, with real Slurm jobs
   on a shared partition: met. Job 27755977 failed 9 Slurm tests because pytest's base temp was on node-local
   `/tmp`, unreadable from the Slurm job's node; 27757266 sets it on shared scratch.
4. **senselab spike at `P`:** `pydra_spike_test.py` on throwaway branch `2x/p3-gate-spike` (`c4392cbe`, not
   pushed), 8 passed (four contracts on `debug` and `cf`), each failing when its pydra feature is removed: met.

## Decision

Pass: **Route A**. Task 9b builds `PydraRunner`; `NativeRunner` is not built (design §6.3).

## Findings for later tasks (no criterion fails)

- pydra's `ResultStore` protocol has no error record, so errored results are not stored (Task 9b, D2 Task 46).
- pydra's Slurm path (`load_and_run`) writes `_result.pklz` directly on failure, bypassing the store (D2 Task 46,
  B2 Task 44).
- The workflow job's checksum is not a logical key; only leaf jobs carry senselab keys (Task 9b).
- The map from pydra's job context to `ExecutionContext` is defined in Task 9b.
- K2's split-consistency test does not cover a task fed by an upstream split inside a workflow; the spike's
  map-task-gather graph exercises it.
