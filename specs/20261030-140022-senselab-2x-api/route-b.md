# Route B analysis: what of senselab 2.x moves to pydra

2026-11-15. Read-only analysis. senselab at `3c3dfced` (2x/phase-3 with the cluster evidence), pydra at
`76d03f84` (`enh/execution-extensions`), proposal 0001 as on that branch. Every "pydra has X" below means
that branch; nothing of it is on pydra `main`.

## Summary

**The destination in the owner's direction is the spec's own.** `execution.md` §2 gives every execution layer
a pydra home (L0 env: G1/G2, L1 resident resource: H0/H, L2 pool: I, L3 placement: F1/F2), and its §5 list of
what stays senselab-only is short: env building, the wire codec, the model cache and servers, key policy, the
ProvStore schema, placement tables, site profiles. Route A's ruling P3-52 ("senselab keeps its own queues") is
what departed from that, and the earlier review (`pydra-upstream-review.md`) defended the departure.

**Under Route A senselab uses none of pydra's execution features.** `runner.py` imports `python.define`,
`workflow.define`, `Submitter`, `Worker`, `Result`, `Lease`, `context_key`, `WorkflowError`,
`ComputeResources` and `tag_router`. pydra's lease is bypassed by `_EpochLease` (`runner.py:102`), its store
by the in-memory `SenselabStore` (`:118`), its `cf` and Slurm workers by `InProcess` and `substrates.Slurm`,
its cache by senselab's `ResultStore`. So the lease (#924), Slurm arrays (#925), cheap submission (#927) and
the success record (#929) change nothing in a senselab run. Two execution layers are being built and tested
side by side by the same people.

**The earlier review is right on timing and on four points of design, and wrong on the destination.**
Right: the policy was still moving (ADR-009 amended three times in two weeks, the queue item format three
times in Phase 3); a pydra pool that ships pickled jobs would lose senselab's identity-only items; J1's
one-record-per-job is not ADR-010's activity model; placement fitting is senselab's. Wrong: its test was
"does pydra have a caller today", which is circular for H, I and G2. Its own rule, "a feature goes to pydra
when it means something for a task that is not senselab's", moves the queue, the pool loop, the epoch, the
bisection engine and process supervision. Its advice for the lease ("propose the protocol, let pydra
implement a smaller version") produced a second, larger implementation: pydra `engine/lease.py` (704 lines)
plus `utils/storage.py` (203) beside senselab `locks.py` (651) and the probe in `storage.py:989-1108`. The
two already differ: senselab's never follows a symlink and works relative to a directory descriptor
(`locks.py:3-5`); pydra's is path-based (no `dir_fd` in `lease.py`).

**Route B here means two things, and only the second moves code.** (1) pydra's `Submitter` splits into a
`GraphRunner` (`ready()`, `complete()`, `plan()`) and an `Executor`; senselab drives the graph and stops
pretending to be a pydra `Worker`. (2) The executor side is pydra's: senselab's `Executor` protocol
(`executor.py:355-389`) becomes pydra's, and `InProcess` and `Pool` become pydra's batching worker (H) and
`pydra-workers-pool` (I), with senselab's policy passed in through hooks.

| Subsystem (senselab, lines) | Moves to pydra as | Hook for senselab policy | When |
|---|---|---|---|
| Lease and storage probe: `locks.py` 651, `storage.py:989-1108` | C, landed: `DirectoryLease`, `LeaseSettings`, `pydra.utils.storage` (#924, #928 open) | `LeaseSettings` built from `profile.lease` (R-210); a `broke` report for the event log | M1, after Phase 4. Needs descriptor-relative operation in pydra first |
| Graph bridge: `runner.py` 925 | §3.10 RFC: `GraphRunner.ready/complete/plan` | node `checksum=` (K2, in use); senselab's executor behind the `Executor` protocol | RFC written before Phase 4 extends `compile_plan`; switch after Phase 4 |
| Batching and bisection: split engine in `job.py:291-380`, `batching.py` 181 | H: batching worker, "a failing batch is bisected" (proposal §3.6) | `trigger_of` (which exception splits), fault class of an exception, `FaultBudgets`, the bucket grid | M3, after Phase 5 |
| Queue, pool worker loop, controller, epoch, substrates: about 4,760 | I: `pydra-workers-pool` | opaque item payload and `run_claim(claim)`; scaling caps; the site placement hook; item weight | M4, after Phase 5. Largest and last of the execution moves |
| Process supervision: `supervisor.py` 626, `worker_pipes.py` 335, process half of `worker_client.py` | G2: the `Venv` runner | the wire codec (`senselab-worker`), shutdown graces | M5, with G2 |
| Result store: `store.py` 1198, `StorageDir` in `storage.py` | D2: `PortableStore` (staged write, rename commit, first writer wins, versioned layout) | entry format; key and context records; `ReusePolicy.accept` carrying `acceptable()`, the license policy and verify sampling | M6. Before 2.0.0 final or at a schema bump, not between |
| Identity: `kernel/identity.py` 390, `keys.py` 209, `equivalence.py` 261 | Already there: K1, K2, G1 (context recorded, not keyed) | `checksum=`, `ReusePolicy` | Done. Nothing more moves |
| Provenance: `provenance.py` 563, `records.py` 163, `provenance_check.py` 145 | J1, restated: an execute or reuse record handed to a sink before the commit | the sink is `ProvStore`; vocabulary and BEP028 stay | M7, after Phase 5 |
| Placement, sizing, calibration, probe machine: about 6,500 | Stays. pydra has F1 and F2 | none needed | Not moved |
| Planner, licenses, env building, run directory and events, wire codec | Stays | none needed | Not moved |

**Three things cannot wait for the deferral without cost.** Write the §3.10 RFC before Phase 4 puts nesting,
routing and loops into `compile_plan`. Add a lint rule now that keeps senselab types out of the modules that
will move. Decide whether the store layout changes before 2.0.0, because after it a layout change recomputes
every user's store.

**Needs the pydra maintainers:** the §3.10 RFC, E2, D2, G2, H (`provided=` naming), and descriptor-relative
storage. I and the supervisor can be a POSIX-only plugin the owner controls.

---

## 1. What is left

### 1.1 The senselab roadmap

Phase 3 closed 2026-11-14 at `f37f1ae4`. From `STATE.md` and `design.md` §6.2:

- **Phase 4** (8-12 person-weeks): `Pipeline` with ports, nesting, routing, loops and `Absent`; dataset
  adapters and profiles; the merged config loader; the generic `sl.signal` tier; env grouping; ports of
  `audio_analysis`, `explore_conversation`, `health_measurements`; one cross-modal pipeline. Exit: one
  pipeline on two dataset profiles, pipelines nest unchanged, one generic task on audio and accelerometry.
- **Phase 5** (10-16 person-weeks): the remaining ~30 tasks; MCP server, CLI, docs; tutorials as use-case
  tests; 1.x deleted. Exit: no 1.x name, README quick start in CI. This is the point at which the refactor
  is usable.
- **Carried:** R-303 (calibration cleanup), R-304 (GPU per-request measurement), R-264 and R-281 to Phase 4,
  R-297 to Phase 5, R-305 (within-item dedup), R-306 (per-commit provenance delta at 10x).
- **Deferred in §6.4:** Route B ("added when pydra RFC 3.10 lands"), the env-merging solver, the packed store
  tier and GPU token arbiter, license-set combinators, pixi envs.

### 1.2 The pydra track

The register rows R-238..R-250 all read "not delivered". Six are out of date. On the branch since the old
pin `c8541fc5` (33 files, +3786/-247):

| Row | Item | State at `76d03f84` |
|---|---|---|
| R-238 | A0 Slurm `--error` | landed, #920 |
| R-239 | A2 one `squeue` per cycle, `sbatch --array` | landed, #925 |
| R-240 | B1 benchmarks | landed, #921 |
| R-241 | B2 cheap submission | #922 and #927 landed; the success record is #929, open |
| R-242 | C lease | landed, #924; takeover fix #928 open |
| R-246 | H0 no `chdir` | landed, #926 (`Submitter(chdir=False)`) |
| R-243, R-244, R-245 | D2, E2, G2 | not started |
| R-247, R-248 | H, I | not started |
| R-249, R-250 | J1, J2 | not started |
| none | §3.10 RFC | not written |

Also landed and not in those rows: `WorkflowError.entries` and `Submitter(telemetry=False)` (#923), the
storage capability probe (with #924). These were items 1-4 of the earlier review's "propose now" list.

### 1.3 senselab's subsystems

Lines are `wc -l` at `3c3dfced`. `runtime/execution/` is 13,028 lines; the modules of `runtime/` itself are
19,089; `senselab-worker` is 2,470.

| Subsystem | Modules (lines) | Encodes |
|---|---|---|
| Graph bridge | `execution/runner.py` 925 | ADR-001 (checksum equals key), design §4.9 |
| Executor contract and in-process | `executor.py` 416, `in_process.py` 767, `job.py` 658, `batching.py` 181 | ADR-009 (fault classes, one bisection engine), ADR-002 (key per request formation) |
| Pool | `pool.py` 1126, `pool_parts.py` 255, `pool_worker.py` 1034, `queue.py` 1052, `epoch.py` 265, `substrates.py` 1029 | ADR-003 (queues, claims, epoch fence), ADR-008 |
| Run and events | `run.py` 1075, `events.py` 624, `observing.py` 278, `telemetry.py` 225, `plan_reads.py` 205, `outputs.py` 102 | ADR-008, ADR-007 (private run directory) |
| Model workers | `workers.py` 576, `services.py` 343, `supervisor.py` 626, `worker_client.py` 1143, `worker_pipes.py` 335, `vllm_server.py` 759, `envs.py` 914; `senselab-worker` 2,470 | ADR-003 (services), I-8 (attestation), ADR-006 (wire) |
| Placement and sizing | `planner.py` 965, `placement.py` 118, `site_placement.py` 492, `fit.py` 211, `sizing.py` 406, `calibration.py` 509, `measures.py` 1078, `probe.py` 949, `units.py` 268, `probe_machine.py` 953, `hardware.py` 564 | ADR-004, ADR-009 amendment 2026-11-12 |
| Storage | `store.py` 1198, `storage.py` 1148, `locks.py` 651 | ADR-003, ADR-006, ADR-007 |
| Identity | `kernel/identity.py` 390, `keys.py` 209, `equivalence.py` 261, `revision.py` 491, `licenses.py` 402 | ADR-002, I-1..I-9 |
| Provenance | `provenance.py` 563, `kernel/prov.py` 685, `records.py` 163, `provenance_check.py` 145, `bep028.py` 329 | ADR-010 |
| Profiles | `profiles.py` 472, `execution_settings.py` 360 | `execution-profile@5.0` |

The brief names `runtime/identity.py` and `execution/provenance.py`, `fit.py`, `planner.py`. They do not
exist under those paths: identity is `kernel/identity.py` with `runtime/keys.py`; provenance, fit and planner
are in `runtime/`.

## 2. Mapping

Each row splits a subsystem into (a) generic mechanism, (b) policy that is not senselab's, useful to any
pydra user as a default or option, and (c) senselab's own policy with the hook pydra would expose.

### 2.1 Lease and storage (C)

- **(a)** A directory lease by `mkdir`, a renewer independent of the work, `prove()` before a commit, age
  judged on the storage clock, takeover by rename. senselab: `locks.Lease`, `LeaseState`, `BrokenLease`.
  pydra: `DirectoryLease`, `_Heartbeat`, `LeaseLost`, `storage_now`, `probe_capabilities`.
- **(b)** The two inequalities that keep a live holder from being taken over. pydra checks them in
  `LeaseSettings`; senselab checks them in `JobLease` (`execution_settings.py:208`, `LeaseBoundError`). Legacy
  `SoftFileLock` files are taken over once stale.
- **(c)** The values. Hook: `PickleStore(lease_settings=LeaseSettings(...))`, which exists.
- **Gap to close before senselab deletes `locks.py`:** pydra's lease is path-based. senselab's opens the
  parent once and never follows a symlink, which ADR-003 (amendment 2026-10-31) and ADR-007 require. pydra's
  also gives no record of the leases it broke; senselab logs each as a `LeaseBreak` event.
- **R-210 as written is moot.** It says to pass `profile.lease` to `LeaseSettings`. Since T24-9 senselab's
  pydra store takes no lease at all (`_EpochLease`), so there is nothing to pass it to. The row becomes "use
  pydra's lease in place of `locks.Lease`", which is M1.

### 2.2 Graph bridge (§3.10)

- **(a)** Expansion, state, lazy resolution, per-element readiness, `on_error`. pydra has these in
  `Submitter.expand_workflow_async` and `NodeExecution` (`submitter.py:591`, `:889`).
- **(b)** A ready set handed out whole, so an executor can batch it. Today pydra awaits `Worker.run(job)`
  once per job.
- **(c)** How a ready set becomes requests: `PlanFragment`, `fragment_size`.
- What Route A pays for the missing seam, all in `runner.py`:
  - `ExecutorWorker._submit` and `_flush` (`:503-600`) rebuild the ready set by collecting jobs until
    `loop.call_soon` fires, so a fragment's boundary depends on asyncio scheduling.
  - `_upstream` (`:734`) and `_element` (`:498`) recover senselab's fragment from a pydra job's resolved
    inputs. The plan is compiled to a pydra workflow and then read back out of it.
  - `serve_shared` (`:404`) sends again, after the run, each job pydra folded into another of the same
    checksum, because ADR-010 wants a Reuse activity for it.
  - `SenselabStore` (`:118`) holds results in memory only to answer pydra's `lookup`. pydra's cache therefore
    knows nothing on a resume, and J2's expected hits cannot come from it.
  - `Follower` (`:632`) and the sender pool bridge a blocking executor to pydra's loop.

### 2.3 Batching and bisection (H)

- **(a)** Group ready jobs that share a resident object; send one request; return one result per item;
  split a failed request in halves until the failing item is alone. senselab: `form_batches`
  (`batching.py:162`), `_sent_in_halves`, `_halves`, `_half` (`job.py:291-380`), `Abandoned`.
- **(b)** Three fault classes with different effects: content fails the item, resource moves it, a crash is
  retried and charged to the item after N lone losses. A load failure is not an item's. Failures are not
  stored. These are ADR-009's rules and none of them mentions audio or models.
- **(c)** Which exception is which: `trigger_of` (`job.py:128`), `WorkerFatal.from_load`, CUDA OOM. The
  budgets (`FaultBudgets`). The length buckets (`BucketGrid`). Hooks: a classifier
  `(exception) -> content | resource | infrastructure`, a budget object, a `group_key(job)`.
- The review's objection was that pydra has no batched request. That is what H adds; proposal §3.6 already
  states the bisection rule. Its other objection, that the policy changed three times in two weeks, is a
  reason to wait until ADR-009 has been quiet for a phase, and M3 is ordered accordingly.

### 2.4 Queue, pool, epoch, substrates (I)

- **(a)** Append-only index segments, oldest-first claims, one lease per claimed item, outcome records
  (`done`, `dead`), holds, reconcile: `Queue`, `QueueSet`, `Claim`, `settle` (`queue.py`). The claim loop
  with a quantum and yield: `PoolWorker.serve` (`pool_worker.py:476`). The controller that scales workers
  from queue depth: `Pool.workers_wanted`, `_scale_rounds`, `substrates.scale`, `Fleet`. Starting workers:
  `Local`, `Slurm`, `ExternalWorkers`. The coordinator epoch: `Epoch`, `StaleCoordinator` (`epoch.py`).
- **(b)** Safety by idempotence, with the lease and the epoch as efficiency only. Provenance durable before
  an outcome is published. A worker that yields when another queue it can serve has older work.
- **(c)** What an item is and how a claim runs. The queue uses `PlanFragment` for four things only: its
  digest, its bytes, `len(fragment.jobs)` as the item's weight, and validation on read
  (`queue.py:130, 208-216, 545-555, 832`). Hooks: an item codec `(digest, bytes, weight)`, and
  `run_claim(claim) -> outcomes`. The probe marks (`put_probe`, `probe_mark`, `set_probe_unit`, about 80
  lines) and the site hook (`site_placement.Placer`, `SiteRefused`) stay senselab's; the controller takes a
  `place(request) -> answer` callable.
- **The review's strongest point holds and becomes a requirement.** senselab's pool worker never imports
  pydra and carries identities, not pickles. Proposal §3.7 sketches pool processes that "claim, run and
  commit through the `ResultStore`", which with `PickleStore` means unpickling jobs from a shared directory.
  That would break ADR-007. So I must be built on an opaque payload, or wait for D2 and G2's
  import-reference jobs. This is the design decision I depends on.
- `substrates.Slurm` (sbatch for worker jobs, its own `squeue` polling) and pydra's `SlurmWorker`
  (`_squeue_all`, `_submit_array`) now both poll one user's queue once per cycle. One of them goes at M4.

### 2.5 Process supervision (G2)

- **(a)** A child in its own process group, shutdown by graces, signals deferred during a critical write,
  orphans reaped: `Supervisor`, `Keeper` (`supervisor.py`), `SharedFds`, `WakeEvent`, `Pump`, `Lines`
  (`worker_pipes.py`), the process half of `WorkerClient` (`worker_client.py:182-360`).
- **(b)** A lost worker is an infrastructure fault; a worker that is OOM-killed is a resource fault.
- **(c)** The JSON-lines protocol and its codec (`senselab_worker.serve`, `codec.py`, `canonical.py`), which
  `pydra-extension-points.md` §2 already names as the pluggable runner.
- The review called this "code without a caller" in pydra. G2's runner and I's pool processes are the
  callers. It is POSIX-only (`prctl`, process groups), and pydra runs CI on Windows, so it belongs in the
  plugin, not in core.

### 2.6 Result store (D2)

- **(a)** A content-addressed entry staged beside its place and committed by rename, first writer wins; a
  corrupt or misfiled entry refused; arrays as `.npy` members: `_commit`, `_seal`, `Staging`, `publish`,
  `StorageDir` (`store.py:503`, `storage.py:350-950`).
- **(b)** A versioned layout (`schema-<N>/`). No pickle for registered types. The context beside the result
  and a reuse policy: pydra has `<checksum>/<context key>/`, `ExactReuse`, `ConsistentReuse`, `AnyReuse`.
- **(c)** `LogicalKey` and its fields; the equivalence table; verify sampling; license refusal. Hook:
  `ReusePolicy.accept(stored, current, task)`, which exists.
- **Two pydra behaviours senselab's store must not inherit.** pydra keeps an errored slot and reruns it;
  ADR-003 stores no failure. And #929 lets the submitter read a succeeded slot without dispatching the job,
  which under a senselab store would skip the license check, the verify sample and the Reuse activity. #929
  reaches its methods through `getattr`, so a store without them is unaffected; a senselab store on pydra
  must either not offer them or route the read through the hit hook.

### 2.7 Identity (K1, K2, G1)

This layer has already moved as far as it should. pydra has declared task identity, a node-supplied
`checksum=` (used at `runner.py:258-264`), an execution context that is recorded and not keyed, and
`Environment.checksum`. What stays is senselab's: `build_key`, `job_checksum`, `acceptable`
(`kernel/identity.py:361`), `equivalence.py`, model revision pinning (`revision.py`). The one generic piece
not yet in pydra is a provided object's key entering the checksum (`Arg(provided=...)`), which comes with H.

### 2.8 Provenance (J1)

- **(a)** One record per execution and one per reuse, written by the process that did it, durable before
  the result is published; one document per writer per claim; a reader that unions them; a check that every
  committed item has an activity: `ProvStore.restart`, the write-before-commit order in `PoolWorker`
  (T24-10), `check_provenance`.
- **(b)** The record's content: used checksums, generated outputs, environment checksum, host, times,
  status, plan and node. pydra's `Audit` already writes PROV-shaped messages through messengers.
- **(c)** The `senselab:` vocabulary (`kernel/prov.py`), model and weights agents, licenses, BEP028.
- The review's objection stands against J1 as proposed: one record per job counts a store hit as an
  execution. So J1 is restated: pydra emits `executed` or `reused`, to a sink, before `commit`. senselab's
  sink maps them to PROV. `ProvStore` stays; `records.py` shrinks to that mapping.

### 2.9 What stays

- **Placement and sizing.** `fit.choose`, `sizing`, `calibration`, `measures`, `probe`, `ProbeMachine`. The
  least settled part of the runtime (R-303, R-304 open) and the review's "general too early" applies in
  full. pydra's side is `ComputeResources` and the router, both in use.
- **Planner and licenses.** ADR-004 and I-9. J2's `plan()` gives pydra the listing; `explain()` stays.
- **Env building.** `envs.py`, uv locks, CUDA routing (`execution.md` §5).
- **Run directory and events.** `run.py`, `events.py`. A run's attempts, inputs and `run-event@2.3` log are
  senselab's observability (ADR-008).
- **Profiles.** `ExecutionProfile` stays as the loader. Its sections follow their mechanisms: `lease` to
  `LeaseSettings`, `pool` to the plugin's settings, `faults` to H's budgets.

## 3. Route B

### 3.1 What it means

Proposal §3.10 is one sentence: separate `ready()`, `complete()` and `plan()` from execution, as its own
RFC, after 3.1 and 3.3. `pydra-extension-points.md` §7 gives the shape: `Submitter` splits into `GraphRunner`
(ready set, lazy resolution, state expansion, reads from the `ResultStore`) and `Executor`
(`submit(jobs) -> futures`), touching `expand_workflow_async`, `NodeExecution`, `engine/job.py` and
`engine/lazy.py`.

Read with the owner's direction, Route B is that split with **senselab's execution on pydra's side of it**:

1. **The seam.** senselab calls `ready()`, groups the jobs by node into fragments, and calls `complete()`
   per result. `ExecutorWorker` and the stand-in tasks go.
2. **The executor.** pydra's `Executor` protocol is senselab's, generalised: `admit`, `release`, `submit`,
   `poll`, `cancel`, `bind_epoch`. pydra ships implementations: its existing workers behind an adapter, a
   batching executor (H) that is `InProcess` without senselab's types, and `pydra-workers-pool` (I) that is
   `Pool`, `Queue` and `PoolWorker` without them.

The earlier review considered only part 1 and concluded "nothing of senselab moves". That is true of part 1
and is why part 1 alone does not answer the owner.

### 3.2 Where it makes sense

- **The seam, first.** It removes the double representation in §2.2 and it is the precondition for every
  executor move, since an executor that batches needs whole ready sets. It also decides Phase 4: nesting,
  routing and loops will be written either into `compile_plan` or against `GraphRunner`.
- **The pool.** It is what pydra lacks most (R-248) and what senselab has proved most: the cluster scale
  ladder, 71 abrupt-stop cases, resume on another host. A pydra user with ten thousand small jobs on Slurm
  has the same need.
- **Batching with bisection.** pydra has no way to run several jobs in one call. H0 is already in.
- **The lease.** Both sides exist; one should remain.

### 3.3 Where it does not

- **Key composition, equivalence, licenses.** The hooks exist and the policy is senselab's science.
- **Placement fitting and the probe machine.** Unsettled, and specific to memory-bound model inference.
- **The PROV vocabulary and BEP028.** Only the emission point and its ordering are generic.
- **Env building and the wire codec.** `execution.md` §5.
- **The planner.** pydra's graph runs what the planner placed; it does not place.

### 3.4 What senselab would delete, and keep

Estimates from the class outlines, not measured by a trial deletion.

| Module | Goes | Stays |
|---|---|---|
| `runner.py` 925 | `_EpochLease`, `SenselabStore`, `_never`, `_record`, `ExecutorWorker`, `_dispatched`, `Follower`, `serve_shared`, `_upstream`: about 500 | `compile_plan` reduced to a graph description, `_NodeKey`, `_Run` bookkeeping |
| `locks.py` 651 | all | an adapter from `JobLease` to `LeaseSettings` |
| `queue.py` 1052 | about 850 | probe marks, the `PlanFragment` codec |
| `pool.py` 1126, `pool_worker.py` 1034 | controller and claim loop, about 800 | admission, probing, `run_claim`, observations |
| `epoch.py` 265, `substrates.py` 1029 | about 900 | the site hook wiring, ready marks |
| `supervisor.py` 626, `worker_pipes.py` 335, `worker_client.py` 1143 | about 1,500 | the request and response codec |
| `job.py` 658 | the split engine, about 120 | keying, serving, recording |
| `store.py` 1198, `storage.py` 1148 | staging and commit, `StorageDir`, about 1,200 | entry format, records, hit rules |
| `provenance.py` 563, `provenance_check.py` 145 | writer mechanics and the check, about 400 | the PROV mapping, rebuild, BEP028 |

About 6,000 to 7,000 lines leave senselab, of the 16,000 in `execution/` and storage. Their tests leave with
them; senselab keeps the tests that state an ADR and runs them against pydra's implementation.

### 3.5 Risks

- **pydra's maturity.** 1.0 alpha. The whole track is one branch and nothing is on `main`; the PR #891
  description is stale. Each recent PR needed review rounds that found real defects (the #928 review found
  that a full disk now waits without end where it used to raise). If `main` does not take the branch,
  senselab depends on a fork.
- **Generalisation cost.** pydra supports Windows and three Python versions. #928 is a Windows rename bug
  senselab would never have met. senselab's storage and supervision are POSIX-only by design. Moving them
  into pydra core means a Windows story; moving them into a plugin does not.
- **Maintenance load.** The same people maintain both. Moving code does not shrink the total; it removes
  the second copy and adds pydra's review and CI to every change.
- **Release coupling.** It exists already: Route A needs `checksum=`, `ResultStore`, `tag_router`,
  `ComputeResources` and `on_error`, all branch-only, and `pyproject.toml:101` pins a git SHA, which a PyPI
  release of senselab cannot depend on. senselab 2.0 on PyPI needs a pydra release with the branch in it
  whichever route is taken. Route B widens what that release must contain.
- **Guarantees senselab must not lose:**
  - **Identity (ADR-001, ADR-002, I-2).** The checksum is the key's digest; context is recorded, never
    keyed. Held by `checksum=` and `<checksum>/<context key>/`. At risk if a pydra hit path skips
    `ReusePolicy`.
  - **Licenses (I-9).** Refusal happens when a job is keyed and when a hit is served
    (`job.py:401 _code_license`, `_serve`). Any pydra-side hit, #929's included, must pass the hook.
  - **Provenance completeness (ADR-010, T24-10, T24-11).** An activity for every committed item, on disk
    before the commit. pydra's `commit` must call the sink first and fail the commit if the sink fails.
  - **Fault classes (ADR-009).** Three classes, no failure stored, a load is not an item, a crash charged
    after `crashes_charged` lone losses. pydra's errored slot is a stored failure.
  - **Storage and trust (ADR-003, ADR-007).** Descriptor-relative operations, no symlink followed, private
    run directories, no pickle read from a shared store. pydra's `Submitter` docstring says loading from a
    cache unpickles its files.

## 4. Sequencing

senselab stays usable throughout: its `Executor` protocol is the seam, and behind it an implementation is
senselab's until pydra's passes senselab's tests, then pydra's. No move has a period with both in use.

**M0. Now, with the repin (another agent's PR). No agreement needed.**
- Retire what §5 lists. Pass `Submitter(telemetry=False, chdir=False)`.
- Rewrite R-210 as M1. Mark R-238..R-242 and R-246 with their PRs.
- Proof: `runner_test.py`, `resume_test.py`, `pool_follow_test.py`, `abrupt_stop_test.py`,
  `tests/lint/host_bound_locks_test.py`.

**M0b. Before Phase 4 starts. Needs the maintainers for the RFC.**
- Write the §3.10 RFC: the signatures of `ready()`, `complete()`, `plan()` and the `Executor` protocol,
  taken from `executor.py:355-389`. Implementation can wait; the shape cannot, because Phase 4 extends
  `compile_plan`. Not checked here: whether pydra's `State` can express Phase 4's routing and loops. That
  is the RFC's first question.
- Add a lint test beside `tests/lint/dependencies_test.py`: `locks.py`, `storage.py`, `queue.py`,
  `epoch.py`, `supervisor.py` and `worker_pipes.py` gain no new import of a senselab type. It costs an hour
  and keeps Phases 4 and 5 from making the later moves harder.

**During Phases 4 and 5. pydra only; senselab untouched.** Design §6.3 already budgets 4-8 person-weeks.
- E2 (maintainers), D2 (maintainers), J2.
- Descriptor-relative operation and a `broke` report in `DirectoryLease` (maintainers; POSIX-only path).
- Merge #928 and #929; fix the #928 review's I-1.

**M1. Lease. After Phase 4.** Small, both sides exist.
- Precondition: the lease gaps of §2.1 closed in pydra.
- Proof: `locks_test.py` (945 lines) run against pydra's lease through the adapter;
  `probe_lease_cross_node_test.py`, `run_lease_test.py`, `queue_test.py`, the lease stress fixture.

**M2. The seam. After Phase 4.** Needs the RFC accepted and E2.
- senselab deletes the bridge of §2.2.
- Proof: `runner_test.py`, `in_process_test.py`, `start_order_test.py`, `resume_test.py`,
  `pool_split_test.py`, `provenance_check_test.py` (the Reuse that `serve_shared` wrote must still be there).

**M3. Batching and bisection. After Phase 5.** Needs H accepted (maintainers; `provided=` is an open
question in the proposal).
- Precondition: ADR-009 unamended for one phase.
- Proof: `bisection_engine_test.py`, `in_process_bisect_test.py`, `fault_class_test.py`,
  `job_engine_test.py`, `batching_test.py`, `pool_backend_oom_test.py`.

**M4. Pool. After Phase 5.** A plugin; the open question is only where it lives.
- Precondition: M1, M2; the opaque payload agreed; `plan-fragment@3` unchanged through Phase 5.
- Proof: `queue_test.py`, `pool_test.py`, `pool_worker_test.py`, `pool_scaler_test.py`,
  `pool_follow_test.py`, `substrates_test.py`, `durable_state_test.py`, `abrupt_stop_test.py`,
  `resume_test.py`, and the cluster scale ladder rerun.

**M5. Supervision. After Phase 5, with G2** (maintainers).
- Proof: `supervisor_test.py`, `worker_client_test.py`, `worker_fds_test.py`, `worker_set_test.py`,
  `lifecycle_probes_test.py`.

**M7. Provenance emission. After Phase 5.** Needs J1 restated (maintainers).
- Proof: `provenance_test.py`, `pool_provenance_test.py`, `provenance_check_test.py`, `bep028_test.py`.

**M6. Store. Timed by the release, not by risk.** Needs D2 (maintainers).
- Before 2.0.0 a layout change is a schema bump and costs a recompute. After it, users hold stores. So
  either D2 lands and senselab's entry format moves onto it before 2.0.0, or the store stays senselab's
  until a major version.
- Proof: `store_test.py`, `store_formats_test.py`, `storage_test.py`, `storage_writes_contract_test.py`,
  `addressed_formats_test.py`, `tests/lint/digest_contract_test.py`.

## 5. The pin

The pin (`pyproject.toml:101`) should move with every merge to `enh/execution-extensions`: one PR to `2.x`
per merge that changes the SHA and the lock, runs `runtime/execution/`, and deletes whatever the merge makes
unnecessary. Each workaround should carry a test that fails when its reason is gone, as
`runner_test.py:553-558` does for `LeaseSettings`; that test will fail on this repin and is the signal to
rewrite R-210. Repinning to `76d03f84` retires: `_no_telemetry` and `NO_TELEMETRY` (`runner.py:66, 895-905`)
for `Submitter(telemetry=False)`; `_errored_job` and its regular expression (`:886-892`) for
`WorkflowError.entries`; the stated reason for the per-attempt pydra root, since `result.save` now takes no
lock and writes atomically (the per-attempt root may stay as policy, but T24-9's unpruned-directory concern
can then be closed by reusing one root); the caveat in `tests/lint/host_bound_locks_test.py:7` about
pydra's `save`; and the T24-9 nit on `hash.py:223`, whose `SoftFileLock` is gone. It does not retire
`_EpochLease`, which is senselab's choice to take no run-long lease, nor yet the two `__getstate__` methods,
because a failed job is still pickled (`pydra/engine/job.py:387-390`); try deleting them and run the failure
cases of `runner_test.py`. #928 changes nothing for senselab while it takes no pydra lease. #929 is reached
by `getattr` and `SenselabStore` does not offer it, which is correct until the hit hook of §2.6 exists.

## What was and was not verified

- Read in code: all of `runner.py`; the outlines (classes, functions, line numbers) of every module named;
  `locks.py:1-175`, `job.py:60-265`, `queue.py:359-411, 666-700`; pydra `store.py`, `lease.py`,
  `submitter.py`, `job.py`, `result.py`, `workers/` at `76d03f84` by outline and targeted reads.
- Not read: the bodies of `pool.py`, `pool_worker.py`, `substrates.py`; ADR-006, ADR-007, ADR-008 in full;
  `execution.md` §6. #928 is known from its review, #929 from its PR body and diff stat.
- Estimates, not measurements: every "goes" figure in §3.4.
- Not known: the result of the repin PR, which was still running.
