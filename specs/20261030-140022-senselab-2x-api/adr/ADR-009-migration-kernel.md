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
  - `senselab_worker` ← kernel ← runtime ← backends: the worker runtime is stdlib plus numpy and
    depends on nothing in senselab; core imports its codec and weight hashing (design §4.12, §5);
  - nothing depends on 1.x.
- **Enforced by import-linter** in CI.

**Faults typed at the source.**
- Each fault is classified where it happens, into one of three classes:
  - `content`: bad input, or a model error on this item;
  - `resource`: OOM, timeout;
  - `infrastructure`: environment, handshake, network, lost service, preemption.
- **Backends raise typed faults.** A backend states the class by raising `ContentFault`,
  `ResourceFault` or `InfrastructureFault` (from `senselab_worker.faults`), or a subclass. The
  runtime's own faults are typed the same way (`WireValueError` is content, `ResponseTooLarge`
  resource, `ProtocolError` and `ArrayIntegrityError` infrastructure). A typed fault's record has
  `gap: false`.
- **The exception table is a fallback, and it always marks a gap.** Any other exception is
  classified by type (`ValueError`, `TypeError`, `KeyError` → content; `MemoryError`,
  `TimeoutError`, `*OutOfMemoryError` → resource; anything else → infrastructure), and its record
  has `gap: true` whatever class it gets. The class of an untyped exception is a guess: a `KeyError`
  from a backend bug is not a verdict on the input (R-83).
- A gap is reported by the host. Each backend's contract is zero gaps over its fault fixtures
  (Phase 2), and the executor never caches a fault with a gap (Phase 3).
- Only `content` faults charge an item's retry budget. The others re-place the job
  (`execution.md`).

## Consequences
- Half-migrated time is bounded per use case, not global.
- The kernel is small enough to review whole.

## Alternatives rejected
- **Migrating by phase across all use cases.** That is a long global coexistence period.
- **Classifying faults in the coordinator.** It has no context about the failure.
- **Classifying by exception type alone.** Library exceptions do not say whose fault they are; a
  bug and bad input both raise `KeyError`. The type table survives only as a fallback whose every
  answer is flagged as a gap.
