# senselab 2.x execution design: pools, GPUs, substrates, and the pydra boundary

Status: draft, revised after review. This document refines `design.md` §4.4, §4.7, §4.8 and §4.9.
Identity rules (the logical key, execution context and reuse policy, I-1..I-9) are in [ADR-002](adr/ADR-002-identity-v2.md) and `design.md` §4.8, and are not repeated here.
The case for pydra is in [`engine-choice.md`](engine-choice.md). pydra changes are cited by their
ids in
[proposal 0001](https://github.com/nipype/pydra/blob/enh/execution-extensions/proposals/0001-execution-extensions.md).
Extension points: [`pydra-extension-points.md`](pydra-extension-points.md).

Principles: P-2, P-5, P-6, P-7, P-8, P-10, P-22, P-25, P-34.

## 1. The question

A single run can mix:
- CPU nodes and GPU-only nodes;
- nodes served by a long-lived engine (vLLM, vLLM-omni);
- nodes in isolated envs, some on end-of-life Python;
- local and Slurm substrates.

This document says how they mix, and which parts are generic (pydra) and which are senselab's own.

## 2. Layers

| Layer | What it is | Home |
|---|---|---|
| **L0 Environment** | A locked env (uv venv, or pixi for R/Java/native stacks) with the worker runtime as an overlay. Numerics and protocol identities follow I-2. | pydra: `Environment.checksum()` (G1), `Venv` + runner (G2). senselab: env building, CUDA routing, `senselab_worker` |
| **L1 Resident resource** | State a worker loads once and serves to many jobs: a model on a device, or an engine service. Keyed by identity, with LRU under a budget. | pydra: `provided=` objects + batching (H0, H). senselab: the model-cache provider, `serve_in_venv`, `VllmServer` |
| **L2 Pool** | Worker processes of one resource class, pulling from a lease queue on shared storage, started by a substrate. | pydra plugin `pydra-workers-pool` (I). senselab: site profiles |
| **L3 Placement** | Assigns each job a resource class, env, device, batch and stack, with `explain()`. | pydra: `ComputeResources` (F1), routing (F2). senselab: placement policy |
| **L4 Graph** | Nesting, ports, fan-out, routing, loops, the ready set. | pydra, Route A (section 4), with a fallback (`design.md` §4.9) |

Boundary rule: a feature goes to pydra when it means something for a task that is not senselab's.

## 3. A mixed run

```
 LogicalPlan + Placement ──► coordinator ──► queue/cpu   queue/gpu:a100#<model id>   queue/svc:<service id>
                                 ▲  epoch file (efficiency fence) │                          │
                                 └──── results (store) ◄── pool workers on local / Slurm substrates
```

### 3.1 Coordinator
- **What it holds:** the plan and the graph. It releases a job to its queue when the job's inputs
  exist in the store, and marks descendants complete, typed-absent or failed.
- **The `LogicalPlan`** carries every resolution, and its hash is the run identity
  ([ADR-004](adr/ADR-004-logical-plan.md)).
  - A restart is `resume(logical_plan_hash)` with a fresh `Placement`.
  - A restart never re-resolves `@main`. It may re-probe hardware and re-place nodes.
- **Correctness comes from idempotence, not fencing** ([ADR-003](adr/ADR-003-storage-leases.md)).
  - Jobs are content-addressed, and results are first-writer-wins by rename.
  - A second coordinator therefore causes duplicate work, never a wrong result.
  - An epoch file makes a stale coordinator stop releasing; it is an efficiency fence only.
- **Route choices and loop rounds are stored as results,** so a restart replays them rather than
  recomputing them.
- **It runs as a small CPU job,** never on a login node.

### 3.2 Queues and workers
- **Queues:** one per (resource class, resident identity): `cpu`, `gpu:<type>`, `mps`, `svc:<id>`.
  Items are plan fragments (job, input refs, env, tunables), never pickled objects.
- **Workers:** `senselab execute --pool <dir> --classes <list>` runs inside the env the queue
  needs. A worker claims a batch from one queue, runs it, commits per-item results by atomic
  rename, and renews its lease.
- **Batches return a result per item.** A failing batch is bisected so one item cannot fail its
  batch-mates.

**As built (Phase 3 Task 15a, P3-35, P3-36).** `runtime/execution/pool_worker.py`; `senselab execute --pool <run
dir>/queues --classes <list> [--profile NAME]` runs `serve_pool`.

- **One host process, one protocol.** The pool worker is a host `senselab_core` process. It builds each claimed
  item's jobs from the run's plan (`run_plan`, latest placement) and bound inputs (`bind_sources`), groups the jobs
  of one claim by plan, node and placement, and sends each group through `execute` and the process's `WorkerSet`,
  so every backend is reached by the one worker protocol. It logs each job `claimed` under the current epoch's
  attempt (`Epoch.attempt`), and commits each item with `report` once all its jobs have outcomes. An item whose
  jobs cannot be built (no plan, another plan, an input that no longer binds) is given the outcome `dead` with
  that failure, so the `Pool` applies the class's effect.
- **Quantum.** The coordinator splits a fragment into items of its unit's sized quantum, each item's placement carrying
  the sized batch (`size`, Task 20). The worker claims from the class whose oldest waiting item (`Queue.oldest_age_s`)
  is oldest: one item, or one-job items gathered up to `pool.quantum_items` jobs (`Queue.claim(max_jobs=,
  singles=True)`); on a queue it has not run yet, one item alone, so the model load falls on one item. It stays on that class until it has claimed
  `quantum_items` jobs or `quantum_s` seconds have passed (the ceilings), then takes the class with the oldest work
  again, which may be the same one. Each item's outcome carries the worker's `Sample` of it (wall, jobs, input
  seconds, baseline and peak memory, cold or warm), from which a calibration round's measure is read. Items of a held env are not claimable. It exits once nothing has been claimable for `pool.idle_polls` polls of `pool.poll_s`.
- **Renewal.** Claim leases beat from their own renewers (Task 12); the worker also proves them (`Claim.renew`)
  every renewal interval (`lease.renew_fraction × lease.stale_after_s`) while it waits for a claim's outcomes,
  which run on a thread. An item whose lease was lost is not reported; another claimer takes it.
- **Load failures are the env's (fix round 1).** A worker that fails to start or load raises `WorkerLoadFailed`
  (an `infrastructure` fault, its cause chained) from `WorkerSet`, whatever it raised: a `WorkerFatal` at load, a
  load timeout, a worker lost before ready, and a server that did not start or attests another model
  (`VllmStartError`, `AttestationMismatch`; fix round 2). A server's start lease that is busy or unreadable
  (`LeaseTimeout`, `LeaseUnavailable`) is not a load failure and keeps its own class (fix round 3); a pool worker
  gives an item whose start lease is busy back unreported, so a busy lease never charges an env's
  `infrastructure_replacements`. The public API's envelope for a load failure is `worker_load_failed`, naming its
  cause's type and message, or `server_start_failed` or `attestation_mismatch` when the cause is one of those, so a
  load failure is never reported as `invalid_call`. It is never bisected: the jobs of the request get it once, and the
  `Pool` counts it toward the env's infrastructure budget and holds the env. A broken env costs one worker start
  per claim, not 2n-1.
- **Bisection (P3-36).** Only a request to a loaded worker that fails as a whole is bisected: `WorkerLost`,
  `WorkerFatal`, `ProtocolError`, or any exception whose class is `resource` (`WorkerTimeout`, `WorkerOOMKilled`,
  `ValueTooLarge`, a `MemoryError` in the host). The jobs are split in halves, each sent again, until the failing
  job is alone; it is reported `Failed` with that failure under its own key (`job_key`), and the `Pool` applies the
  effect. A failure of one item inside a request is reported as it is. Any other exception, a load failure among
  them, fails every job of the request that raised it and only those: a half that completed keeps its outcomes and
  is committed. `ProtocolError` is included beyond the ruling's list because the host discards the worker after
  one, exactly as after `WorkerLost`.
- **A lost claim stops its request (fix round 1).** When a claim's lease is lost (`Claim.renew` gives up an
  item), the worker releases the claim's other items at once, so an in-process request that cannot be stopped never
  holds their leases past the stale bound (fix round 2); it then cancels the claim's remaining work, abandons the
  request in flight (`WorkerSet.abandon(thread)` closes the worker processes that the claim's own runner thread holds,
  so their `run` raises `WorkerLost`, and leaves every other thread's workers alone; an in-process host or server
  backend cannot be stopped and is waited for), commits nothing for it, and takes no new claim until the request's
  thread has stopped.
- **SIGTERM.** On the main thread `serve_pool` installs a SIGTERM handler for the call (`PoolWorker.halt`): it sets
  `stop` and wakes the wait for outcomes. The worker abandons the request in flight, waits up to `join_s` (a worker
  client's shutdown bound by default) for its thread, commits the items whose jobs finished, releases the rest
  (their leases are removed, so they are claimable at once) and returns 0.
- **Idle exit and respawn.** Items under another worker's live claim, and items of a held env, are not claimable,
  so a worker with only such work exits after its idle polls even if that holder later dies. Keeping workers alive
  is the substrate's job: the coordinator's substrate (`Substrate.ensure`, Task 16) starts workers again while its
  queues hold work.

**As built (Phase 3 Task 20R-7a).** `runtime/execution/pool.py`, with its items, followed fragments and unit states in
`pool_parts.py`. The Pool no longer calibrates: a unit's measure is an `Estimate` read through `MeasureSource`, and its
computed jobs move through the `ProbeState` of `calibration-probe.md` §7.1.

- **Units.** Every unit the Pool reads or writes is built by `UnitResolver.unit` (the env record and the hardware
  index only), under the decoder `decoder_of` gives the fragment's jobs. A unit that cannot be built is `Unresolved`
  and runs the safe values (`size(None, …)`); a slurm env is one such, since its node, platform and numerics are not
  known before a worker starts. On a local substrate the Pool resolves the class as this host's report, the rule the
  resolver and `explain` use; when that class is not in the hardware index the unit is `Resolving` and its computed
  jobs wait for a worker's ready mark. A worker's hello (`workers/<id>`, its `HardwareClass` per queue) is written
  into the index (`MeasureRoots.note_seen`, write-once); the first hello of a class decides it, and where it
  disagrees with the prediction the hello wins: units are built with it from then on. A `Resolving` queue wants one
  worker, so the wait costs no extra start.
- **Servable jobs bypass the state machine.** `ResultStore.admits` is asked of each job's key digest in the context its
  env gives (`servable_jobs`, on `pool.admit_threads` threads, the predicate `explain`'s counts use); jobs it admits
  are put at once, in every state, in items of `quantum_items` that end on request boundaries. Where the env is not
  known every job is computed. An entry under an old context is not admitted, so its job is computed and probed.
- **Lookup.** The estimate is `MeasureSource.get(unit)`. It is used as it is when its unit of input is `none`, or it
  measures a slope and every known length of the jobs lies in the lengths it measured; otherwise `Probe.plan` plans
  over the held jobs against the roots' pooled observations and regime records, so only the strata they do not cover
  are probed, and `bucket_edges` come from the backend's formation (`bucket_edges`) whether or not an estimate exists.
  A plan with no `SINGLE` item, and no warm-up a batched phase could use (a batched phase needs a device memory fit),
  is not run: the unit is `Measured` by its estimate, else `Unmeasured`, and a unit probes at most once per run.
- **PeerWait.** `MeasureRoots.probe_lease` returns the lease or `PeerProbing`. A unit that finds a peer's lease waits,
  re-reading once per `pool.poll_s` on the Pool's clock, until the peer's lease is gone or `peer_wait_s` has passed;
  it then returns to `New` and plans only what the peer's observations leave uncovered. Having waited once, it takes
  a lease still held by the peer as `Unleased` and probes without waiting again.
- **Probing.** A probe has an id and a queue `<queue>~probe-<id>`. The Pool records the unit's resolved class on that
  queue (`Queue.set_probe_hardware`) before it puts anything, with `put_probe`; a worker takes the serve lease only
  when its own class is the one recorded, so a queue with none recorded is served by none. `WARMUP` and `SINGLE` items
  are put together; a request that comes back `served` has its item replaced from the stratum
  (`Probe.replace_served`); when they have outcomes `Probe.estimate` runs and `Probe.plan_batched` adds the `BATCHED`
  items inside the seconds left. The budget counts the walls of warm requests (those after the model's load) in the
  observations read: past `probe_s` no further phase is planned and the probe ends with what it has, its unclaimed
  items left to run as the ordinary work they are. The model load and the waits between phases do not count.
  After the lease's stale window (`lease.stale_after_s`, on the Pool's clock) with no item claimed or finished, the
  unit is `Unmeasured`: its unclaimed probe items are dropped from the followed fragment and their jobs run as
  production. A retry is never a probe item.
- **End of a probe.** The observations are written (`MeasureRoots.put_observations`, `obs/<probe id>.json`) once the
  probe lease is proven (`Lease.prove`), the lease is released, and the held jobs are put cut and batched by
  `size`; each probe job's result is yielded like any other. `Unmeasured` writes its warm observations too. The
  held jobs of one unit are put in one `size` call per fragment; an item's jobs of one batch share its placement's
  `batch_size`, so the Pool and `InProcess` give each job the same batch.
- **Workers wanted.** A probe queue wants none. The queue a probe measures wants, beyond its sized work, one worker
  per probing unit; sized work is the slowest measured unit's seconds per job over `target_wall_s`, within the items
  waiting and the class's workers; an unmeasured unit's is `unmeasured_workers`.
- **One extrapolation log per run**, made again by `bind_epoch`. `Pool.probes` maps each unit probed to its probe id,
  the digests of its items and their jobs.
- **Removed here, left to Task 20R-7b:** a resource fault no longer grows the unit's measure (`_grow`); the `oom`
  record that replaces it is 7b's. Resource faults still move a job to the next larger class.

**As built, fix rounds 1 and 2 (Task 20R-7a).** `runtime/execution/probe_machine.py` holds `ProbeMachine`, a class over a
`Host` (queues, worker marks, live workers, fragment reads, `produce`, `wake`); its other inputs are the clock, the
`UnitResolver`, the measures and roots, and `Probe`. `Pool` composes it and is its host (`_PoolHost`), so the machine is
tested with a host of fake queues (`probe_machine_test.py`). What changed against the block above:

- **Pending causes wait in `Resolving`.** A unit that cannot be built because its node, hardware or env is not yet known
  (`Pending`: a slurm node, hardware not in the index, an unbuilt env) is held and built from the first hello's
  `worker-ready@1.0` mark; only a final cause (the mark names no class, an env failed to build or is unreadable, two
  decoders) runs the safe values. A worker reads the backends of its queues from the run's placement of its latest
  attempt (`private/placements/<attempt>.json`, written by `prepare`) and builds each env before it writes the mark.
  Each env entry is `env_entry` (the §3.1 fields but the decoder, held to `MeasureEnv`'s filter; `driver` is the torch
  build only); a server's `driver` is completed from `service_of` by the coordinator. An entry is keyed by
  `<task>~<backend name>`, not by env name, since two backends of one env name carry their own overlay. A mark lacking
  the backend's entry demotes only a pending unit: a unit this host resolves is placed as it is. Only another major of
  the mark's format is no mark (logged once); a later minor is read.
- **Held parts are checked like flights**: a part held in `Resolving`, `PeerWait` or `Probing` raises `QueueHeld` and
  `WorkersBroken` as a flight does. `calibration.resolve_s` bounds `Resolving`, from the first poll with a live,
  non-pending worker of the queue; past it the unit runs the safe values and the Pool logs why, once per queue. A
  `Resolving` unit wakes the scaler, so a worker starts. A mark is placed against its parts once, when read. Every
  change to the unit states and the resolving parts is made under the machine's lock, the one `counts` reads under.
- **A closed probe queue.** The stale window runs from the probe queue's last claim or outcome. Past it the Pool writes
  the queue's `closed` mark (`Queue.close_probe`), which a worker checks before it takes the serve lease or claims
  (and drops a lease it holds on a closed queue); items under a live claim finish there, and their jobs are put as
  production only if the claim goes stale.
- **The F2 filter covers the env.** A probe queue records the unit's class and the backend's env
  (`Queue.set_probe_unit`, which replaces `set_probe_hardware`); a worker takes the serve lease only when its own class
  and its env for that backend (`env_identity`: lock, wheels, platform, numerics, threads, torch build) equal them. A
  queue recording none is served by none.
- **A disagreeing hello** places every unit built from the prediction again: a probing unit's queue is closed, its
  finished items' observations are written under the class of the worker that made each, and its parts return to
  `Resolving`; a finished unit is dropped, its put items keeping their sizing.
- **Workers wanted** count quanta of jobs (`ceil(work / quantum_items)`), not items, so one-job items start no more
  workers than a quantum of jobs does. The unmeasured unit's lengths and bucket edges are read in the kind its jobs
  hold.
- **Evidence.** `local_probe_test.py` runs real `senselab execute` workers through `Local`: a run probes, a second run
  over the same measures root plans none and its estimate's source names the first's probe, a run after the env's
  `wheel_set` changed is another unit and probes again, and a run whose env is unbuilt when planned resolves at its
  worker's mark and probes. The no-identifier test builds the mark from a `local:` model's plan and an env whose index
  is a mirror URL with userinfo.

### 3.3 Fault taxonomy

Faults are classified at the source by the worker runtime's error table
([ADR-009](adr/ADR-009-migration-kernel.md)). An unclassified exception counts as `infrastructure`
and is reported as a classification gap.

| Class | Examples | Effect |
|---|---|---|
| content | decoder error on this input, invalid params | charges the item's retry budget; after the budget, the item goes to `dead/` and its descendants become typed failures |
| infrastructure | handshake refusal (version skew), lost service, preemption, storage error | the item is re-placed without charging its budget; the plan reports the fault; claims of the env stop once the `Pool` holds it (a pool worker itself keeps claiming other envs' items) |
| resource | OOM, timeout from contention | re-placed with a larger grant or a smaller batch; not charged |

When one env or one worker is broken, the fault is reported against the env or worker, never
spread across a million items.

**As built (Phase 3 Task 14, P3-33, P3-34, P3-36).** The `Pool` applies each effect once, with the budgets of the
profile's `faults` section. A fault with `gap: true` is treated as infrastructure whatever its class. A retry is a
one-job fragment of the next `attempt`, logged `failed` then `queued` by the coordinator; only the final failure is
yielded to the runner, read back from `dead/<key digest>.json`.

| Class | Budget key | As built |
|---|---|---|
| content (not a gap) | `content_attempts` | charged; queued again until the item has that many charges, then dead |
| resource (not a gap) | `resource_replacements` | uncharged; moved to the class with the least memory above the current one, on the same device and GPU type; at most that many moves, then (or with no larger class) dead |
| infrastructure, or any gap | `infrastructure_replacements` | uncharged; queued again on the same class; once one env has that many failures in the current epoch (counted from that epoch's `failed` events on disk, so a coordinator restarted within the epoch keeps the count, and a new epoch starts afresh) it is reported once as `BrokenEnv` and held on each queue it is seen on (`held-<epoch>-<env label>`), only its own items: claimers skip items of a held env and other envs on the class keep running. `QueueHeld` stops the run; no item is failed (a held job is logged `queued` again), so a later epoch runs them |

A worker's reported result is yielded only when it names the job's own logical key (`job_key`; any other key is a
content fault `WrongKey` of that job, never a yield) and the store holds its outputs; one it does not hold is a
`MissingOutput` gap and is retried. The resource budget counts moves, so the laptop profile's
`resource_replacements: 0` fails a resource fault at once. The class decides the effect: a worker's `charged` flag is
not read, so a content record that is not a gap is charged whatever it says. A `WrongKey` is a charged content
fault and charges replay from the queues' outcomes, so a worker bug that reported wrong keys leaves its jobs in
`dead/` for the run; those `dead/<key>.json` files are removed by hand once the worker is fixed. Holding an env passes the epoch fence
first, and a hold counts only for its own epoch or a newer one, so a stale coordinator can never hold work of the
current one.

### 3.4 Scheduling and memory
- **Claim order.** Queues keep time-ordered index segments, and workers claim the oldest item
  first, in O(1) amortised (ADR-003).
- **A resident model is a cache, not a reservation.** A worker yields after N jobs or T seconds
  when another queue it can serve has older work.
  - Workers being alive is what guarantees progress.
  - Latency is not bounded when the cluster grants no workers.
- **The plan checks** that the caps cover the models that must be resident at the same time.
- **Device memory in 2.0** uses static caps per resource class. An OOM is a resource fault, so the
  job is re-placed with a smaller batch and the item is not charged.
  - A host-local `flock` token budget per device is the designed next step, adopted when the scale
    ladder shows the need (design §6.4).
- **A GPU-only task with no GPU class** in the execution profile fails planning and names the
  node. It never falls back to CPU silently.

### 3.5 Services
- **What a service is:** a resident resource in a pool worker of class `svc`. Its process runs in
  an `Env.server` env.
- **Registration:** a service is a registry entry `{identity, endpoint}` under a renewed lease,
  whose beats are its liveness (ADR-003, amended 2026-11-07).
- **Lifecycle:** each consumer holds a lease on the service. The service exits when no consumer
  lease has been live for `services.idle_exit_s`. Queue depth does not decide this.
- **Identity:** every response carries the service's attested identity (I-8).

### 3.6 Substrates
- A substrate starts pool workers with a CPU, GPU or memory grant: local processes, or Slurm jobs
  (`work_pool.Controller`, with the partition chosen by resource class).
- The controller scales pools from queue depth within the profile's caps, including the QOS caps
  (P-29).
- Kubernetes and Ray launchers are post-2.0. The design admits them as substrates, never as graph
  executors.
- **Storage capability probe.** The lease protocol needs atomic rename, exclusive `link(2)` and
  the server's clock. These are probed as a `SystemRequirement` at plan time, and a store that
  fails the probe is refused (sshfs, SMB and some object-store mounts).
- **Mixed placement** of laptop and cluster pools requires a store both can reach; otherwise the
  laptop plans and submits only.
- **Placement values and the site hook** ([calibration-probe.md](calibration-probe.md) §10, T20R-12).
  senselab runs no scheduler query: it never calls `sinfo`, `squeue` (beyond listing its own jobs),
  `sacctmgr`, `scontrol` or `sbatch --test-only` to choose a value. `Slurm` runs only its declared
  clients (`sbatch`, `squeue`, `scancel`) and raises `UndeclaredClient` for any other command. A class's partition, time, worker
  cap and excluded nodes come from three layers in order: the profile (`substrate.exclude_nodes` is
  the profile's layer of the excluded nodes; senselab adds none), the explicit override
  (`sl.plan(..., placement=mapping_or_yaml)`, `senselab plan|coordinate --placement FILE`), then the
  hook's answer. `Plan.placement_request()` and `senselab plan RUN --json` give each class's needs as
  `placement-request@1.0`, read from existing measures and starting nothing.
  - **One placer, every substrate.** The pool owns one `Placer` (built from the first admitted plan's
    overrides and hook) and reads each class's worker cap from it (`Pool.granted`); `Slurm` reads the
    partition, time and excluded nodes from the same object. A substrate that starts no scheduler jobs
    (`Local`, `InProcess`) records the values with a note that they are recorded, not applied, so no
    explicit value is silently ignored.
  - **The hook** is `substrate.placement_hook` (`module:callable`), replaced by the run's
    `placement_hook` (`sl.plan(..., placement_hook=)`, `--placement-hook`). It is called once per class
    per epoch, before that class's first worker is submitted, with the request narrowed to the class,
    bounded by `substrate.placement_hook_s` on the injected `Clock`. The request is built per epoch from
    the input values the pool bound, never by binding inputs again.
  - **Validation** (`check_answer`): a time below `time_per_worker_s` (below `time_floor_s` when that
    is unmeasured, accepted and noted when both are) and a `max_workers` below 1 or above the
    request's `workers_max` raise `PlacementAnswerRefused`, naming the class, field and both values,
    before any submission. A hook that raises, answers with another type or outlasts its bound is
    `PlacementHookFailed`: the class keeps its explicit and profile values and the event says so.
    A hook path that does not import is `PlacementHookFailed` at start, not a fallback.
  - **Events.** A `placement` event is a run-scoped kind of `run-event@2.1`, appended to the epoch's
    `EventLog` (no node), one per class per epoch: the class's needs, the answer, the hook's path (and
    whether it is importable) and distribution version, the verdict (`no-hook`, `accepted`, `refused`,
    `failed`), the values used with the layer that set each, and the `--exclude` list submitted.
    Partitions, accounts, nodes (as written and as expanded), the answer's `why`, the request's
    unmeasured reasons and the notes render under the reader's audience rule (`rendered`, one helper
    shared with `SiteRefused`) from one set of names, `ClassPlacement.names`: the `Placer` collects every
    layer's partitions and excluded nodes (profile, override, request, hook answer), a replaced one too,
    and the account; the operator's text goes to the log. `senselab status` and `senselab explain` print them from the log; status
    ignores them. The closing actuals record (T20R-7b) is a further `placement`-scoped event appended
    to the same log.
  - **A site's refusal.** An `sbatch` that exits with a reply raises `SiteRefused` (a `SlurmError`): the
    class, the `sbatch` line, the site's message and the layer of each value, rendered from the
    class's `ClassPlacement.names` under the audience rule, naming `--placement` and the hook as the inputs that change it. It is counted against its
    class (`Fleet`), so the class is held and others keep starting; `resume(..., placement=)`
    continues from the store. senselab never rewrites the request.

### 3.7 Result store at scale
- **Store budgets.** The store has its own per-item budgets for files and stats (P-33). The scale
  ladder measures them.
- **Packed tier (deferred, design §6.4).** Small results go to per-shard append-only segments,
  written only by the shard's lease holder, plus an index committed by rename (ADR-003). Expected
  hits are then estimated from the index.
- **Reuse across contexts** follows the run's reuse policy ([ADR-002](adr/ADR-002-identity-v2.md)).

## 4. pydra for the graph

What the extension-point report established:
- a Submitter has one worker (`engine/submitter.py:160`);
- completion is read from the cache;
- expanding a graph into jobs needs a Submitter.

**Route A, built first:**
- senselab's graph format ([ADR-001](adr/ADR-001-graph-format.md)) compiles to a pydra workflow.
  Each node carries its senselab logical key, so pydra never hashes senselab code.
- The coordinator is a pydra Submitter.
- A pool worker plugin (I) publishes ready jobs to the queues and completes them from the store.
  Routing (F2) and resources (F1) map nodes to queues.
- The prerequisites are E1 (per-element readiness), D1 (`ResultStore` interface) and F2.
- After D1/D2, completion reads senselab's store, and pydra's pickle cache is not used.
- Isolated envs need only the worker runtime, not pydra.

**Route B, the end state:** pydra's graph runner (`ready`, `complete`, `plan`) separated from
execution, proposed as RFC 3.10 once E and D land. Moving from A to B changes no senselab user
code.

**Delivery:** used from the branch through `[tool.uv.sources]`, with the commit pinned in
`uv.lock`, and released in 1.0 alphas.

**Decision gate at the start of Phase 3:** if E1, D1, F2 (with F1), K1 and K2 (with G1) are not working on the branch, build
the fallback graph layer behind the same `Executor` protocol (`design.md` §4.9).

| pydra id | Change | Needed for |
|---|---|---|
| E1, E2 | per-element dataflow, keep-going | Route A at scale |
| B1, B2 | benchmarks, cheap submission | 10⁴–10⁶ items |
| D1, C, D2 | store interface, leases + atomic commit, `PortableStore` | shared NFS; envs on different Python versions |
| F1, F2 | `ComputeResources`, routing | mixed pools |
| G1, G2 | env checksum (recorded as context, not keyed), `Venv` + runner | L0 |
| H0, H | no-`chdir` option, provided objects + batching | L1 |
| I | `pydra-workers-pool` | L2 |
| A0, A1, A2 | Slurm fixes, budget, arrays | Slurm |
| J1, J2 | provenance via messengers, `plan()` | P-3, P-10 |
| K1, K2 | declared task identity; node-supplied checksum, `lookup(checksum, context)`, reuse-policy hook | ADR-001 (checksum equals key), ADR-002 (`equivalent`) |

## 5. What stays senselab-only

- env building (uv/pixi locks, CUDA routing, `SystemRequirement` probes);
- the worker runtime codec for senselab types;
- the model-cache provider, `serve_in_venv` and `VllmServer`;
- the cache-key policy (`design.md` §4.8), supplied through pydra's hashing hooks;
- the ProvStore schema and BEP028 export;
- the placement policy and the memory admission tables;
- site profiles.

## 6. The job engine (Phase 3 Task 9)

`run_call` is removed (P2-16, P3-25). Its body is `runtime.execution.job.execute`, which every executor calls;
`InProcess` is the executor that runs a fragment in the calling process, and `InProcess.call` is the one-node plan a
configured task submits. What `run_call`'s docstring carried as reasons, and the reasons for the engine's own choices,
are recorded here (R-200).

- **Model pinning.** A catalogue model at its catalogue revision is the catalogue's pin, read without a lookup, so a
  call on the default model needs no network and the license comes from the reviewed catalogue entry. Any other model
  is `resolve()`d. The planner then rewrites a hub reference to its commit, so the plan names exactly what is loaded.
- **One worker per model.** A worker process loads one model; host backends share one `ModelCache` per budget, so two
  tasks on one model and one backend code share the weights. A host cache key names its backend code,
  `<name>:<worker module>@<version>`, so backends of different code never share a loaded state, and its identity is
  the model's; a host backend without a model takes the digest of its task, name, version and env as its identity.
  Pooled workers are keyed by task, name, version and module too (Task 15a fix rounds 2 and 3; before them, a
  same-named backend of another task got the first one's state). The pool is keyed by backend, env, init and budget (host) or network
  (isolated): a worker started under another budget lacks that budget's sizing, and a worker started offline must not
  serve an online run.
- **Server attestation without a start.** A server backend's context carries the attestation its env lock, model and
  settings declare, computed without starting the server. A store hit therefore needs no server; the server starts on
  the first job the store does not serve, and a started server must attest exactly what was declared, or the job
  raises `AttestationMismatch` rather than record a context that is not the one that ran.
- **One activity per execution.** An execution writes one PROV activity associated with its agent (env lock, license
  and `senselab:modelSource` included, R-110), recording the key, the context and the profile's digest. A store hit
  writes no execution; the store records the reuse. Every activity records `senselab:verifyFraction`,
  `senselab:verifySampling` and `senselab:verifyThresholds`, so a reader can tell what verification the run asked for.
- **Lost commit race.** When `put` finds that another writer committed different outputs first, the job is served
  that entry as a reuse and writes no execution: the store keeps one entry per key, and the provenance must name the
  entry that is kept. The license rule applies to that entry as to any hit.
- **Same key in one run.** Jobs of one run sharing a key run once. Each other one is served that job's stored entry
  as an exact reuse (`cache_hit`, recorded as a reuse) or given its failure; only a descendant of a failed job is
  `blocked` (R-232). The store's hit rules still apply: an entry they refuse, such as one whose context records no
  env license under a policy that does not allow unlicensed hits, is not served, and the job runs again as its own
  execution.
- **Outputs on identity clocks.** Outputs are stored on the clocks their recording inputs' identities name and served
  re-bound to the caller's anchors, so the key holds no anchor and one entry serves one content under every anchor.
  Two recording inputs of one content under two anchors are refused (`AmbiguousClock`): there is no single clock to
  re-bind to.
- **Kernel types as wire forms.** An output on a port declared as a kernel type is stored as its wire form and
  returned as the kernel value, so the store is readable without the backend and a wrong type is refused
  (`OutputTypeMismatch`) before anything is stored.
- **Verify sampling.** A cross-context hit in the sampled fraction `run.verify` is not served: it is recomputed and
  compared under the run's thresholds before anything is stored. The comparison is recorded as
  `EquivalenceEvidence`; a disagreement splits the store's class and is logged by path, never by value, because the
  values may be participant data.
- **Licenses.** Only the run's license policy refuses a license. A refused env or model license raises before any env
  opens. A result from a refused env is never served, and a result whose context records no env license is recomputed
  unless the policy allows unlicensed hits; a reuse so allowed records `senselab:unlicensedReuse`, on every hit path
  including the lost race.
- **The key is the request sent (R-187, P3-51, I-3).** A key holds the batch formation of the batch a job is sent
  in (`request_components(backend, formation)`) and the backend's chunk grid. `form_batches` groups one node's jobs:
  a batch-invariant backend takes any jobs up to the batch size as `mixed`, which is placement only; a backend that
  declares `batch_formation` output-affecting is batched per length bucket of `batching@1.0`, keyed
  `bucket:<grid id>:<index>`; any other backend, or a job with no recording input, is sent alone as `single`. The
  formation comes from the job's own inputs, so a key never depends on batch-mates (Task 18).
- **File-backed inputs are checked at run (R-121, executor part).** A file-backed input is checked against its stamp
  (`audio_io.check`) before its job is keyed; a rewritten or vanished file fails that job with `StaleSource`, a typed
  content fault (charged, never re-placed, since re-placing cannot make the file match), and nothing is sent or
  stored. A stamp this process took is checked by `stat` alone, so the check costs
  nothing for an input opened in this process.
- **Failures are run-scoped (P3-26).** A failed item gives `Failed(FaultRecord)` beside its request's completed items
  and is never stored, so a later run retries it. Only a typed content fault is charged against the item's retry
  budget; a resource or infrastructure fault re-places the job uncharged, and a gap fault, one the fallback table
  classified, is never charged (ADR-009), because its class is a guess.
- **Bounded workers (R-167, R-166).** The worker set keeps at most the profile's workers alive at once (the summed
  `max_workers` of its resource classes); beyond that the least recently used idle worker is closed, and a held worker
  never is. A process has one worker set: the first profile sets its bound, a later profile with a smaller bound
  tightens it, and a larger one does not loosen it. Host model caches are per process and per budget, shared by
  every worker set, so two profiles in one process load a model once. A server start holds a lease of its (env,
  backend, model) only, so a slow start (up to the startup timeout) blocks no other server. Servers are not counted in
  the bound and are not evicted; they stop when the set is closed.
- **Model cache (R-160).** A key loaded before, whose size is known, evicts what it needs before its loader runs, so
  the peak is the budget rather than old plus new; a first load still loads and then evicts, since its size is unknown
  until it is loaded. The cache lock is reentrant and `used_bytes` is read under it. The in-process input copy R-160
  also names is not done: in-process backends still receive the caller's values.
- **Plan fragments are documents.** `plan-fragment@3.0` holds identities, never values or pickled objects, so a fragment
  can cross a process or a scheduler. `admit` is part of the `Executor` protocol: it holds a plan and the values of its
  graph-input identities, and returns an admission (a unique id) that each fragment names, so two admissions of
  one plan (one content under two anchors, say) never replace each other's values. An `output:` input is read from the result
  store inside `execute`, through what the fragment records for it (`upstream`), so a runner deals in
  identities only and a node fed by an upstream node is a fragment like any other. A substrate executor reads graph
  inputs from the records instead of an admission's values.
- **An `output:` input names one port.** A downstream input's identity is `port_output_identity(upstream key
  digest, port)`, identity tag `port-output@1`, so two edges from different ports of one upstream job into one
  downstream port key differently. The fragment's `upstream` records, for each such identity, the upstream key
  digest, the stored entry's output identity (which context's outputs to read) and the port; `execute` reads that
  entry from the store and takes the port's value. Which context computed the upstream entry never reaches the
  downstream key, as a context never reaches any key.
- **A gathered input names every element, in order.** A `gather` edge's input identity is `gather:` and
  `identity_digest("port-gather@1", [element port identities in element order])`, so changing any element or the
  order changes the downstream key. The fragment's `gathered` maps it to the ordered element identities, each of
  which `upstream` maps to its stored entry; `execute` reads them into one list in element order.
- **Upstream reads are recorded.** An execution that reads an upstream input records `prov:used` of the stored
  entry's output entity (the one the upstream activity generated) on the downstream port, once per element for a
  gathered input, so the PROV graph links each downstream activity to the upstream execution it read.
- **A bad upstream fails one element.** An upstream entry missing from the store fails that job alone as an
  infrastructure gap (re-placed, not charged); an entry that lacks the port, holds a value not of its declared type,
  or gives inputs that break the task's constraints fails that job alone as a content fault. Its siblings complete.
- **Servers are per worker set.** A second `WorkerSet` (such as a test's own) starts its own server for a key the
  process set already serves, and a `close()` racing a `server()` may hand back a stopped server. Neither arises with
  the one process set `process_workers` returns.
- **Queued against running across writers (Task 14).** A job's non-terminal state is counted, never ordered: in the
  latest attempt it is running while its claims (`claimed`, `started`) outnumber its `yielded` events and at least
  match its `queued` events, else queued. A `failed` its writer then logged `queued` again (a retry within one
  attempt) is not terminal, so the job reads as the retry's state. While the latest attempt has no terminal event, a
  result an earlier attempt gave stands only when its key is the key the latest attempt gives the job: the runner
  logs every job `queued` with its key under the current plan and placement (`plan_keys`; an upstream input is
  identified by its upstream key, so every key is known before any job runs), and the Pool logs its retries with the
  key under the retry's placement. A resume onto the same placement therefore shows completed work as done, and one
  onto another backend is not done until recomputed; `Run.results` never serves an output under another key.
  `Run.status` and `Run.results` read under the run's current epoch: a job with no event of it yet is queued. `logs_claims` is a member of the `Executor` protocol (`InProcess`
  False, `Pool` True); when it is set the runner logs no `started` as it sends a fragment: the worker's `claimed` is
  the start.
- **The epoch precedes the attempt (Task 14 fix round).** `prepare` takes the epoch, then claims the attempt ordinal,
  and records the attempt in `epoch/<n>/attempt.json` (`Epoch.bind`, read by `Epoch.attempt`, which is also how a
  pool worker finds the attempt and ordinal to log `claimed` under). Every event carries its attempt's `epoch`, and a
  job's latest attempt is ranked by (epoch, ordinal), so a stale coordinator that claimed a higher ordinal cannot
  override the live epoch's results; its failures and blocks never decide a job's state, though a result it gave
  stands as an earlier result. The runner treats `StaleCoordinator` and `QueueHeld` as interruptions: no
  `fail_batch` and no `blocked` or `cancelled` for unsent jobs, which a later epoch runs.
- **Epochs and the Pool (Task 14, ADR-003).** `runtime/execution/epoch.py`: `Epoch(run_dir).take()` makes
  `epoch/<n>` by `mkdir` (a number taken by another coordinator is skipped), writing the lease bound to
  `epoch/<n>/lease.json` at once, and `current()` is the highest. A coordinator (`start`, or `senselab coordinate
  <run>`) holds `epoch/<n>/coordinator.lease`, kept alive under the profile's `lease` section, and marks
  `epoch/<n>/ended` when it stops. `Run.wait` reports `interrupted` only while no thread of that `Run` is alive, the
  latest epoch ended or its lease (or, never held, its directory) is older than its bound (`Epoch.stopped`), and no
  claim on any of the run's queues is live; a running run is never reported interrupted. `Pool.submit`, every retry
  and every poll of `results` raise `StaleCoordinator` once a newer epoch exists. `start` (and `Plan.run`) hands its
  own epoch and attempt log to the executor through the protocol's `bind_epoch`, so a `Pool` built beforehand
  coordinates as that epoch and is never stale for the first coordinator. `results` polls from `pool.first_poll_s`,
  doubling up to `pool.poll_s`. A coordinator holds its epoch's lease from right after `prepare` (before
  `reconcile`, the prov write and, in `start`, the thread launch), so a slow start never reads as a stale one. A
  `QueueSet` built without `epoch=` reads `Epoch(root.parent).current`, the run its queues belong to, so a hold an
  older epoch made after passing the fence is never honoured by any claimer. The Pool's expected key is computed
  under the fragment's own placement (`job_key(..., placement)`), the request actually sent. Correctness does not rest on the fence: a stale
  coordinator's put of a fragment already put is a no-op (an item is named by its digest, first writer wins), a put of
  a new fragment is work a worker does once per outcome, and every result is first-writer-wins in the store. The Pool's
  admission id is the run's (`run-<plan hash>`), since workers bind graph inputs from `inputs.json`, so two
  coordinators of one plan put identical items, and a restarted coordinator replays the retry chain from the
  outcomes on disk (`test_content_and_resource_charges_replay_after_restart`).
- **Open: queue names lack `#<model identity>` (P3-32).** `NodePlacement` carries no resident-model identity yet, so
  a `Pool` queue is named by `resource_class` alone; the suffix arrives with resident models and `svc:<id>` queues
  (Task 16). `senselab coordinate` runs the plan's latest placement under the
  packaged profile (or `--profile`), with workers started separately (`ExternalWorkers`) until Tasks 15 and 16 add
  substrates; `Plan.run` on a slurm substrate runs on a `Pool` over `Slurm` (Task 16).
- **A queue item is at most one quantum of jobs (Task 16 split).** The Step 5 probe on ORCD ran a 200-job node with
  `pool.quantum_items` 10 and a cap of 4 workers: `Pool.submit` put the whole fragment as one item, the queue read one
  deep, one worker was started and it ran all 200 jobs serially (about 11 minutes). `Pool.submit` now puts a fragment
  as items of at most `quantum_items` jobs each, in job order (`split`); each item is a `plan-fragment@3.0` with its
  own digest, so a re-put under the same profile (the same coordinator, another one, or a restart) makes no new item.
  A run re-placed under a profile that changes only `quantum_items` splits into other digests; the old items stay
  queued and are served as store hits or as duplicate compute, as after a `batch_size` change. `results` yields each
  job's result as its item's outcome is read, items read in one poll in job order (across polls and retries, in the
  order they land; the runner settles by element), and every fault effect (content charge, resource move, broken env)
  is applied per item and per job as before, with the key computed under the item's own placement. An item ends only
  between jobs. When the node's request size (`node_request_size`) is at most the quantum, items end on request
  boundaries (`quantum - quantum % request` jobs); a larger request, which only a batch-invariant backend can have, is
  cut at the quantum, so the item size caps that backend's effective batch. A backend that is not batch-invariant is
  sent one job per request, so no item holds part of one. `scale` did need a change: it divided the queue's depth in
  items by `quantum_items`, so 20 items wanted 2 workers. It now divides `Queue.work()`, the jobs of the unresolved
  items, summed from `segments/<seq>/jobs-<n>-<count>` marks the put that took each slot writes beside it (first
  writer wins), so neither `scale` nor `submit` reads an item; a slot without a mark (a put interrupted between the
  slot and the mark) is counted from its item, as one job when the item cannot be read, and `reconcile` takes the
  count from the item it slots. A pool worker's quantum and claim are bounded in jobs, so 200 jobs want `min(20, cap)`
  workers and each claims one 10-job item, while a queue of single-job fragments scales and claims as before.
- **The runner follows; the Pool scales (Task 16 scale-root fix rounds).** On ORCD a 200-job node ran on one worker
  because each fragment held one of asyncio's default-executor threads until its last result landed: a 2-CPU
  coordinator has `min(32, cpus + 4)` = 6, so at most 6 one-job items were ever queued and `scale` wanted 1 worker; a
  downstream node, whose elements become ready one upstream item at a time, kept that cap after the first fix. Now a
  fragment is submitted on the run's sender threads and returns at once, and one `Follower` thread per run polls every
  open handle through `Executor.poll` (never waits; the results landed and not yet given, and whether the fragment is
  done) and settles each element as its result lands, so only the queue and the workers bound the fragments in
  flight. `Executor.results` is gone; `follow(executor, handle)` is the blocking loop over `poll`. An executor that
  splits fragments (`splits_fragments`, the Pool) is sent one fragment per ready set, which keeps the follower's
  handles few; `split` dumps a fragment once and slices it, so it is linear in jobs. A cancel ends every followed
  handle within `pool.cancel_check_s` and leaves its items queued. The Pool owns one scaler thread: it ticks every
  `pool.poll_s` while a fragment is followed, a submit or retry wakes it, `close` stops it, and it is the only caller
  of `scale`, so a slow `sbatch` never delays settling or a cancel and concurrent submits cannot over-start. `poll`
  reads the epoch fence at most once per `first_poll_s`. The sender threads are explicit: the profile's summed
  `max_workers` (3 under `laptop`, 48 under `orcd`), or `PydraRunner(senders=...)`. For `InProcess` they are where
  the jobs run, so they bound its concurrency at the `WorkerSet` bound; before, the default executor gave
  `min(32, cpus + 4)` threads (14 on a 10-CPU laptop), whose extras only waited on leases.
- **Worker report (Task 14, for Task 15a).** A pool worker commits each item with `report(claim, digest, outcomes)`:
  the outcome record carries `jobs`, one `ResultRef` per job. A queue `dead` outcome (a corrupt item) is a content
  fault of each of its jobs. A held queue (`Queue.hold`) gives no claim.
- **`inputs.json` under a lease (R-211).** `record_sources` merges and writes `inputs.json` under a lease on it,
  proven before the write and retried up to `lease.record_attempts` times, so two coordinators starting one plan both keep
  every durable source.
- **Queues (Task 13, ADR-003).** `runtime/execution/queue.py` holds one queue per resource class, with `#<model
  identity>` for a resident model, in a directory named by `identity_digest("label@1", name)`. An item is a plan
  fragment's JCS under `identity_digest("plan-fragment@3", …)`, first writer wins. The index is slots, never appended
  lines: a put takes `segments/<seq>/slot-<n>`, a symlink to the item's digest made by an exclusive `symlink` at the
  next free `n` (`n + 1` on `EEXIST`), and makes `segments/<seq+1>` once `pool.segment_items` slots are taken; a slot
  is made only in a segment with a free number, so nothing lands in a full one. `slotted/<digest>`, made after the
  first slot, keeps a second put from taking another. Claims walk segments, then slots, in order: the first slot not
  marked resolved (`res-<n>`) and not under a live claim is taken under a `Lease` on the item, and its outcome checked
  again under it. Each item has one outcome, `outcome/<digest>`, `done` or `dead`, first writer wins: the record is
  staged and fsynced, the lease proven, and the record linked in the one call after the proof; an `EEXIST` whose
  entry is this claim's own record (a lost reply) is this claim's outcome. A slot is marked resolved only once
  `outcome/<digest>` exists and the slot, read again, names that digest. A full segment whose slots are all resolved
  is retired in place by an exclusive `retired` mark inside it and is never renamed, so a segment number is made
  once. A missing item file is an infrastructure fault: the item stays ready and the
  queue records the fault; an item that is not its digest's fragment is dead with a content fault. `depth()` is an
  upper bound from slot counts; `oldest_age_s()` counts an item behind a stale claim as waiting. A run's queues live
  in `<run dir>/queues/` (`queue.RUN_QUEUES`), where the pool and `senselab execute --pool` also look. A run's `start`
  puts again, in digest order and at O(items) cost, every item in its `queues/` with neither a slot nor an outcome (`QueueSet.reconcile`), which repairs a
  put interrupted between its item and its slot.
