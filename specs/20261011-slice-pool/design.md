# slice_pool: a worker pool over a stage's slices

Owner-approved 2026-10-11 for the r36 round of the r18d triage campaign on ORCD. This directory is a copy of the
run-tree tool (`/orcd/scratch/bcs/002/satra/r18d_20261010/pool/`) and of the r36 chain built on it. It is not
package code.

## Problem

Every sliced stage submitted one array task per slice (128-task arrays). A multi-partition array counts against
the MaxSubmitPU of every partition it lists (pi_satra 256, ou_bcs_low 256, ou_bcs_normal 256, mit_preemptable
448, mit_normal 448, ou_bcs_high 4). Submitting a chain up front therefore capped out at about 256 tasks, and
a slice whose task died stayed missing until someone resubmitted that index by hand.

## Design

A stage is a **pool directory** holding N slice indices and the per-slice command
(`pool.json`; `{i}` and `{n}` are substituted). One small worker array per stage (one array per GPU type for a
GPU stage, all on the same pool) runs `slice_pool.py work POOL`. Each worker loops:

1. **claim** the next free index;
2. **run** the stage's existing per-slice command with that index and the slice count (the drivers already take
   `--slice-index/--slice-count` and resume per recording);
3. **mark** it: on an exit code in `success_codes` (the replay declares `0,3`), `claims/<i>/done.json` and
   `done/<i>`; otherwise `failed/<i>.<k>` (exit code, host, job, log path) and the claim is released;
4. repeat until nothing is left to run or to wait for.

The queue holds the worker count, not the slice count, so the r36 chain's peak is one lane's workers plus one link
job (66 jobs at the default worker counts), whatever the slice counts are.

## Atomicity

`/orcd/scratch/bcs/002` is **NFS v4.2** (`fstor003.ib:/data1`), not Lustre. Two primitives are atomic there,
because both are a single server-side operation that fails with `EEXIST`/`ENOTEMPTY` when the name is taken:
`mkdir`, and `rename(2)` of a directory onto a name that is absent or names a non-empty directory. NFS 4.1+
sessions give exactly-once semantics, so a retransmitted create does not report a false `EEXIST` to its winner.

Measured, not assumed: `pool/tests/race.py`, 32 processes on 4 nodes racing for the same 400 names at the same
second, gave exactly one winner for every name with both `mkdir` and staged rename (400/400, 0 with none,
0 with two).

The pool claims with the **staged rename**: a worker builds `tmp/<token>/` holding `owner.json` and `heartbeat`,
then renames it onto `claims/<i>`. A claim is therefore never an empty directory (which a rename could silently
replace) and is never visible without its owner. After the rename the worker reads `claims/<i>/owner.json` back
and holds the slice only if the token is its own, which also covers an error reported for a rename that landed.

A successful slice keeps its claim directory (with `done.json`) for good, so "is it done" is answered by the
server-side create failing, not by a possibly cached `stat` of `done/<i>`. Releasing (failure, SIGTERM) renames the
claim into `trash/` before deleting it, so a claim disappears atomically.

Liveness reads go through `open()` + `fstat()` rather than `stat()`: an open forces NFS close-to-open
revalidation, where a bare `stat` may answer from an attribute cache up to `acregmax` (60 s) old.

## Heartbeat and stale claims

- The worker touches `claims/<i>/heartbeat` every `heartbeat` seconds (default 60) from a thread while the child
  runs. On each beat it re-reads `owner.json`; if the token is no longer its own (taken over), it kills the child's
  process group and records the attempt as `lost`.
- A claim with no `done.json` whose heartbeat is older than `stale_minutes` (default 10; `init` refuses less than
  three heartbeats) is **stale**. Any worker scanning for work renames it into `trash/` (only one renamer wins),
  writes `interrupted/<i>.<id>` naming the previous owner, and the slice becomes claimable again. The age is the
  reader's clock against the writer's mtime; node clocks are NTP-synced to well under a second, against a 600 s
  threshold.
- If a stale claim turns out to hold `done.json` after the rename (the owner finished in that instant), it is
  renamed back.
- SIGTERM (preemption with requeue on mit_preemptable, time limit, a plain `scancel`) stops the child (TERM,
  then KILL after 20 s), records `interrupted/<i>.<id>` with reason `sigterm`, and releases the claim at once.
  The requeued worker rejoins the pool as a new worker. SIGKILL leaves the claim to go stale.
- Interruptions do not use a slice's retry budget, but `max_interrupts` (default 6) of them do exhaust it, so a
  slice that always outlives the time limit cannot loop forever.

The protocol relies on the drivers being resumable per recording: in the one window where two attempts could
overlap (an owner alive but unable to heartbeat for `stale_minutes`), the second attempt redoes nothing the first
finished, and the first is stopped within one heartbeat of noticing.

## Retries

A slice is retried until it has failed `1 + retries` times (default `retries` 2). Never-tried slices are taken
before retries, so a retry usually lands on another worker. A worker defers a slice that failed on its own host
within `same_host_grace` seconds (default 300) and takes it after that, so a retry goes to another host where one
exists and still runs where none does.

## Ending, standby, and the downstream gate

When nothing is claimable, a worker exits unless claims are still live elsewhere (or deferred). Then up to
`linger` (default 2) workers stay as standby, holding `linger/<k>` slots claimed the same way as slices, so a claim
that goes stale at the tail of a stage is taken over without new submissions. The rest exit.

The next stage's link job runs `afterany` on the worker array(s) and calls `slice_pool.py gate POOL`:

- every index done: exit 0, the link goes on;
- incomplete and no worker array of the pool still queued: orphan claims are released, slices that used their
  budget get one more (their records move to `failed/round1/`), every recorded worker array is resubmitted once
  with its own sbatch arguments, `resubmitted` is written, and the gate exits 10 printing `RESUBMITTED <ids>`; the
  link queues itself again `afterany` on those arrays;
- incomplete after that one resubmission: `halted` is written (workers seeing it stop), the gate exits 1 and the
  chain stops.

`slice_pool.py status POOL` writes `status.json`: one state per slice (`done`, `running`, `stale`, `failed` = budget
used, `retry`, `unclaimed`), the counts, attempts, and the missing indices.

## Test on ORCD (2026-10-11)

Per-slice command `pool/tests/dummy_slice.sh` (sleep, then: slices i%13==0 fail their first two attempts,
i%5==0 their first, i%7==0 succeed with exit 3, a declared success code). Every attempt writes its own start/end
record, so overlap of two attempts on one slice is checked directly. Each test ends with the gate link
(`pool/tests/test_gate.sbatch`) afterany on the workers.

| test | setup | outcome |
|---|---|---|
| race | `race.py`, 8 tasks x 4 processes on 4 nodes, 400 names | mkdir 400/400 exactly one winner; staged rename 400/400 |
| t1 | 40 slices, 6 workers (ou_bcs_low), stale 2 min, heartbeat 10 s; one worker `scancel`led mid-slice | 40/40 done; 15 failed attempts over 11 slices all retried to success (4 slices failed twice, succeeded on the third attempt); 6 done on exit 3; the cancelled worker's slice released on SIGTERM (`interrupted`, reason sigterm) and rerun; 0 overlapping attempts; gate exit 0 |
| t2 | 6 slices, 2 workers (mit_quicktest), both workers cancelled mid-slice | gate exit 10: 2 claims released, workers resubmitted once, gate queued again; second gate exit 0; 6/6 done, 3 failed attempts retried, 0 overlapping |
| t3 | 12 slices, 3 workers, stale 1 min; slice 4's first attempt SIGKILLs its own worker (job ends, no release) | claim went stale (heartbeat age 70 s) and was taken over by another worker (`interrupted`, reason stale); 12/12 done; 4 failed attempts retried; 0 overlapping; gate exit 0 |

`scancel --full --signal=KILL` on an array task reached the worker as SIGTERM (t1, t2), which exercised the release
path; the stale path needed a real SIGKILL of the worker (t3).

## The r36 chain

`r36/` holds the chain as it sits in the run tree: `common_r36.sh` (commit placeholder), `r36_slice.sh`
(one slice of one lane: replay, bypass, review, opinion, cohort, refold, task_audio, rvec), `r36_link.sbatch`
(gate the lane before, submit the next lane and link), `make_r36.sh` (single-job scripts from r35's), and
`SUBMIT_R36.sh`, `CANCEL_R36.sh`, `PUBLISH_R36.sh`. Pitfalls carried from earlier rounds: no variable named
`STAGE` (install_r34.py reads it; the lane variable is `LANE`), no `squeue | grep -q` under pipefail (output is
captured, then tested), temporary files carry the job id or pid, and cancelling goes links first, then lanes from
the end of the chain back to its start.
