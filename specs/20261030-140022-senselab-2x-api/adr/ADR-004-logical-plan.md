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

## Alternatives rejected
- **A single hashed plan.** A hardware change becomes a new run, and resuming across sites is
  impossible.
