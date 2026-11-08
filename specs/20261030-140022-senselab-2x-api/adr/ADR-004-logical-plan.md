# ADR-004: The logical plan is separate from placement

**Status:** accepted, 2026-10-30.

## Context
Architect review §9 (R-45): the Plan mixed what is computed (identity) with where and how it runs
(placement). A restart on different hardware therefore changed the plan's identity.

## Decision
- **`LogicalPlan`** holds the graph (ADR-001) plus every resolution: model weight identities,
  catalogue defaults, profile slices, the input manifest, and the reuse policy.
  - Its hash is the **run identity**.
  - It contains no hardware, env, device, batch size or partition.
- **`Placement`** maps each logical node to a backend instance, stack, env, device class, batch
  plan, resource class and partition.
  - It is recorded in provenance and shown by `plan.explain()`.
  - It is never part of any identity.
- **`plan = sl.plan(...)`** returns both. `plan.run()` executes the placement.
- **Restart and site change.** A restart is `resume(logical_plan_hash)` with a **fresh placement**,
  so resuming on another site or after hardware changes is allowed. The reuse policy (ADR-002)
  governs results from the earlier placement.
- **The cost estimate** belongs to the placement.

## Consequences
- P-10 becomes checkable: laptop and cluster runs share a logical plan hash and differ only in
  placement.
- Batch formation and chunk grid enter identity only when a backend declares them output-affecting
  (ADR-002); otherwise they are placement.

- **Amended 2026-11-07 (batch formation, Phase 3 Task 18).** Batches form across the records of one node and one
  placement (`form_batches`). Each job's formation is its own: `mixed` for a batch-invariant backend, never keyed;
  `bucket:<grid id>:<index>` from its own input length on the packaged grid (`batching@1.0`) for a backend that
  declares `batch_formation` output-affecting, keyed; `single`, one job per request, otherwise and for a job with no
  recording input. The key never depends on batch-mates (I-3); results may vary with padding within a bucket, which
  the Task 18 probe measures. A job's length is measured from its held inputs and, for an upstream input, from the
  span its upstream result recorded (`ResultRef.lengths_ns`, carried as `UpstreamRef.length_ns`), so keying reads no
  stored value. Before a run `plan_keys` keys a bucketed node fed by graph inputs; one fed by an upstream recording,
  and every node below it, stays unkeyed until that upstream runs, so status loses stale-result detection there.
  pydra's checksum is the key with the bucket index left out; events carry the job's true key.

- **Amended 2026-11-08 (sizing at run time, Phase 3 Task 20R; design [`calibration-probe.md`](../calibration-probe.md)
  §6).** The placement carries no plan-time batch: `profiles._batch_size` is removed, a placement records batch 1 and
  `explain()` names the estimate the run will size from. The batch per length bucket, the cuts of a fragment into
  items (by predicted seconds over the run's own input lengths) and the worker count are chosen at run time by one
  function, `size(estimate, grant, config, lengths)`, called by the Pool and the in-process executor with the same
  device selection. The cost estimate this ADR gives the placement is that function's output. None of it reaches a
  key; a bucketed backend's batch never spans two buckets.

## Alternatives rejected
- **A single hashed plan.** A hardware change becomes a new run, and resuming across sites is
  impossible.
