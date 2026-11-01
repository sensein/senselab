# ADR-009: Migration by use case, the kernel boundary, and faults typed at the source

**Status:** accepted 2026-10-30; amended 2026-10-31, 2026-11-01.

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

## Amendment log

- 2026-10-30: the worker runtime is the base of the dependency direction, not a kernel dependent.
- 2026-10-31: backends raise typed faults; the exception table is a gap-marking fallback (R-83).
- 2026-11-01: 1.x lives on `alpha`; `src/` is the 1.x pool; use cases move by `git mv` then modify;
  triage's decision layer is superseded (H-R48, H-R51, H-R52).
- 2026-11-01: `src/` is restored to 1.x as on `alpha` and never edited by 2.x (D-6).
