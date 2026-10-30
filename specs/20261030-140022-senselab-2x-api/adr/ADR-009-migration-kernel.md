# ADR-009: Migration by use case, the kernel boundary, and faults typed at the source

**Status:** accepted, 2026-10-30.

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
  - kernel ← runtime ← backends;
  - kernel ← `senselab_worker`;
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
