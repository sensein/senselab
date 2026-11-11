# ADR-009: Migration by use case, the kernel boundary, and faults typed at the source

**Status:** accepted 2026-10-30; amended 2026-10-31, 2026-11-01, 2026-11-11.

## Context
- Architect review §13 (R-49): the migration had no boundary and no budget for coexistence.
- §10 (R-46): core was a runtime, not a kernel, and the dependency direction below it was not
  enforced.
- §16 (R-52): faults were classified in the coordinator, far from where they occurred.

## Decision

**Migration by use case (strangler).**
- Migrate one use case at a time, end to end. Use cases are listed in `use-cases.md`; tutorials
  are the first ones.
- Each use case has a **coexistence budget**: the dates by which its 1.x path is deleted.
- **A 1.x module is deleted when its last use case moves.** No 1.x module is kept for a use case
  that has already moved (P-14).
- **The triage decision layer** stays on the frozen 1.x tag, with a dated review against the
  flywheel roadmap at the end of Phase 4 (P-36).

**Kernel and runtime.**
- `senselab_core` is split into:
  - a **kernel**: data protocols and types (ADR-005), identity (ADR-002), the graph format
    (ADR-001), serialisation (ADR-006);
  - a **runtime**: planner, executors, store, provenance writer, observability.
- **Allowed dependency direction:**
  - `senselab_worker` ← kernel ← runtime ← backends: the worker runtime is stdlib plus numpy and
    depends on nothing in senselab; core imports its codec and weight hashing (design §4.12, §5);
  - nothing depends on 1.x.
- **Enforced by import-linter** in CI.

**Faults typed at the source.**
- The worker runtime's error table classifies each fault where it happens:
  - `content`: bad input, or a model error on this item;
  - `resource`: OOM, timeout;
  - `infrastructure`: environment, handshake, network, lost service, preemption.
- Only `content` faults charge an item's retry budget. The others re-place the job
  (`execution.md`).
- An unclassified exception is `infrastructure` and is reported as a classification gap.

## Consequences
- Half-migrated time is bounded per use case, not global.
- The kernel is small enough to review whole.

## Alternatives rejected
- **Migrating by phase across all use cases.** That is a long global coexistence period.
- **Classifying faults in the coordinator.** It has no context about the failure.

## Amendment 2026-10-31: backends raise typed faults (R-83)

Replaces "An unclassified exception is `infrastructure` and is reported as a classification gap."
Finding R-83 ([`review-register.md`](../review-register.md)).

- **Backends raise typed faults.** A backend states the class by raising `ContentFault`, `ResourceFault`
  or `InfrastructureFault` (from `senselab_worker.faults`), or a subclass; the runtime's own faults are
  typed the same way. A typed fault's record has `gap: false`.
- **The exception table is a fallback, and it always marks a gap.** Any other exception is classified by
  type (`ValueError`, `TypeError`, `KeyError` → content; `MemoryError`, `TimeoutError`,
  `*OutOfMemoryError` → resource; anything else → infrastructure), and its record has `gap: true`. The
  class of an untyped exception is a guess: a `KeyError` from a backend bug is not a verdict on the input.
- Each backend's contract is zero gaps over its fault fixtures (Phase 2), and the executor never caches a
  fault with a gap (Phase 3).
- **Alternative rejected: classifying by exception type alone.** A bug and bad input both raise
  `KeyError`.

## Amendment 2026-11-01: where 1.x lives, how a use case moves, triage

Replaces "The triage decision layer stays on the frozen 1.x tag, with a dated review against the
flywheel roadmap at the end of Phase 4." Rulings H-R48, H-R51 and H-R52 ([`rulings.md`](../rulings.md)),
and owner decision D-6 ([`decisions-2026-11-01.md`](../decisions-2026-11-01.md)).

- **Where 1.x lives.** 1.x is developed on `alpha` and kept on the freeze branch; 2.x is developed on
  `2.x`. On the 2.x line, `src/` is the 1.x source pool: not packaged (the root `senselab` distribution
  is the 2.x umbrella), not tested and not linted. Nothing in `packages/` imports it (import-linter
  `packages-not-1x`).
- **`src/` is 1.x as it stands on `alpha`** (D-6). It is restored to the freeze commit (`19834fdd`),
  undoing the edits 2.x rounds made to 97 files, and then `alpha` is merged in (P-39). **2.x never edits
  `src/`.** 1.x code that breaks because core changed is not fixed there.
- **How a use case moves.** Merge `alpha` into the 2.x line first, so the module carries every 1.x change
  (P-39); then a rename-only `git mv` commit from `src/` into `packages/`; then a modify commit that
  ports it. The rename-only commit keeps `git log --follow` and blame across the move. This is the only
  way a file leaves `src/`.
- **Triage.** Its tasks and pipelines port like any other use case. Its decision layer (routing,
  branches, verdicts, and the dataset-fitted thresholds and policies) is superseded by the flywheel
  (`roadmap.md`) and is deleted when its last use case moves. `migration-ledger.md` classifies each
  triage module.

## Amendment 2026-11-01: the kernel does no I/O

Ruling H-R57 and the F3 and F4 tasks ([`rulings.md`](../rulings.md)).

- **The kernel does no I/O.** An import-linter contract forbids `senselab_core.kernel` from importing the
  file, OS and import machinery; the kernel keeps the value types and the `SourceReader` protocol, and the
  runtime supplies the reader and loads the equivalence table.

## Amendment 2026-11-11: one bisection engine, and a crash is charged to the item that repeats it (R-288, Task 24-4)

Finding R-288 ([`review-register.md`](../review-register.md)); owner ruling of 2026-11-10. Extends the per-item isolation
this ADR's Decision gives a fault: a worker that dies on one item of a request must not fail its request-mates, and must
not be retried without end.

- **One engine.** `execute` (`runtime/execution/job.py`) is the only place a request is split. The flag
  `bisect_resource` and the Pool's own split (`pool_worker.bisect`, `whole`, `_split_requests`) are deleted. `InProcess`
  and a pool worker send through the same `execute`; a pool worker adds only the bookkeeping that is its own (the
  `Sent` records, the handover of outcomes, the claim's cancellation).
- **Two triggers split a request to a loaded worker.** A *resource* trigger: a request that raised a resource-class
  exception (`splits` of 2026-11-09, now `trigger_of`) or answered a resource fault for each of several jobs. A
  `WorkerFatal` the worker typed `resource` is among these (the worker's verdict wins: the lone job fails `resource`
  and moves to a larger class, uncharged). A *crash* trigger: the request raised `WorkerLost`, `ProtocolError`, or any
  other `WorkerFatal` reported for a request the worker had loaded for (a line it could not read, an exception outside
  its handler). A request of several jobs that fires either is sent again as two requests, each split in turn.
- **A load is not an item.** What checkout, start, load or reload raised (`WorkerLoadFailed`, a `WorkerFatal` that is a
  load: `from_load`) never fires a trigger and is never split or retried here: it is the env's, as 2026-11-10's
  `execution.md` §3.4 states. A `WorkerFatal` raised by a request to a worker that had loaded is a crash. This reverses
  what Task 24-3 left the Pool: it stopped bisecting any `WorkerFatal`, because the type could not tell a load from a
  request; the type now says which (`from_load` is set where the worker reports it, true at load and on a reload, false
  on a request), so only the load case is excluded.
- **Charging.** A request of one job whose worker was lost is sent again. A job that is lost
  `faults.crashes_charged` times in a row, each time alone, is **charged to the item**: it is `Failed` with
  `ItemCrashedWorker`, a `content`-class typed fault (`gap: false`) chained to the last loss, under its own key, and the
  Pool applies the content effect (`content_attempts`, then `dead/`). The jobs that did not repeat the crash complete:
  a request that crashed several jobs is halved until every job is alone, and a job that completes after a loss was
  not the cause. An item charged here is the item's fault and so does not count toward holding the env.
- **A loss the jobs survive is infrastructure, and the env is told.** A lost request all of whose jobs completed
  (a lone job that completes on the retry, or a request whose halves all complete) is no job's failure. The item
  reports each such loss in its outcome record (`survived`, the worker's infrastructure fault record), and the Pool
  records it as an uncharged `failed` event of the env (not `replaced`) counted in `infrastructure_replacements`, so a
  host that keeps killing workers reaches `BrokenEnv` as it did when the Pool's own split returned the lone job's
  `WorkerLost`. The event names the item's first job and its key. A loss that a charged job caused is the item's, and
  is not reported.
- **`faults.crashes_charged` is a profile key** (default 2 in both packaged profiles) with a protocol derivation, not
  a measurement (`config-derivations.md`): one loss of a worker may be the host's (a node fault, an external kill), and
  a second loss of the same job alone on a worker started after the first cannot be, so two is the least that tells the
  host from the item. It is at least 1; 1 charges on the first loss of a lone job.
- **"Fresh worker"** is what the code guarantees, no more: a `WorkerClient` that fails a request is closed
  (`_fail`), so it is not alive, and `WorkerSet`'s next checkout closes it and starts another process; a test
  (`worker_set_test`) shows the next checkout's process id differs. A host or server worker runs in this process or over
  a running server and is not replaced, so for those the retry is a second request to the same worker, not a new
  process. The engine counts losses of a lone job in a row and does not know process identity.
- **Cancellation.** A claim that is lost or halted sets the cancel flag `execute` is given; it is read before each
  request and after any exception, so the `WorkerLost` that `WorkerSet.abandon` causes is the claim stopping, never a
  crash to split or retry. `execute` then raises `Abandoned` carrying the outcomes already final.
- **What did not change:** the fault taxonomy (ADR-009 Decision, the 2026-10-31 typed faults) and the rule that no
  failure is stored.

## Amendment log

- 2026-10-30: the worker runtime is the base of the dependency direction, not a kernel dependent.
- 2026-10-31: backends raise typed faults; the exception table is a gap-marking fallback (R-83).
- 2026-11-01: 1.x lives on `alpha`; `src/` is the 1.x pool; use cases move by `git mv` then modify;
  triage's decision layer is superseded (H-R48, H-R51, H-R52).
- 2026-11-01: `src/` is restored to 1.x as on `alpha` and never edited by 2.x (D-6).
- 2026-11-01: the kernel does no I/O; decoding, stamping and the equivalence table load in the runtime (F3, F4, H-R57).
- 2026-11-01: worker placement. A backend of an isolated or server env declares `worker_sources`: each module of its worker, by dotted name, to its source file, the `worker` module among them. `ensure_env` copies each file into the env's site-packages at its dotted path (namespace packages, no `__init__`), beside the `senselab_worker` overlay, and replaces a copy whose bytes differ from the source; the wheel set is unchanged. `WorkerClient` launches `-m senselab_worker.launch <worker>` unchanged. A placed module imports only stdlib, numpy, `senselab_worker`, a library of its env's lock, or another placed module; never `senselab_core`, never any other `senselab` module. A contract test over every backend with `worker_sources` pins this.
- 2026-11-11: one bisection engine in `execute`, used by `InProcess` and the Pool; a lone job lost `faults.crashes_charged` times on fresh workers is charged to the item as `ItemCrashedWorker`; a load is still not an item (R-288, T24-4).
