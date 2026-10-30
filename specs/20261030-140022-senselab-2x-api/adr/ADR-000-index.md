# Architecture decision records

These records resolve the principal-architect review ([`architect-review.md`](../architect-review.md))
and the items it re-opened in [`review-register.md`](../review-register.md).

**Format.** Each record has a Status, Context, Decision, Consequences and Alternatives rejected. A
record that is superseded keeps its number and names its successor.

| ADR | Decision | Resolves |
|---|---|---|
| [001](ADR-001-graph-format.md) | senselab owns the graph format and the `Executor` protocol; the graph compiles to pydra | R-38, R-50, R-51 |
| [002](ADR-002-identity-v2.md) | Identity v2: a portable logical key, a recorded execution context, and a reuse policy | R-39, R-40, R-41, R-44 |
| [003](ADR-003-storage-leases.md) | Storage and lease protocol on shared filesystems; safety by idempotence | R-3, R-8, R-9, R-12 |
| [004](ADR-004-logical-plan.md) | The logical plan is separate from placement | R-45 |
| [005](ADR-005-signal-time.md) | Signal and time model | R-43 |
| [006](ADR-006-formats.md) | Formats, versioning and evolution | R-42 |
| [007](ADR-007-trust-privacy.md) | Trust and privacy | R-48 |
| [008](ADR-008-observability.md) | Observability | R-47 |
| [009](ADR-009-migration-kernel.md) | Migration by use case; the kernel boundary; faults typed at the source | R-49, R-46, R-52 |

The 2.0 cut list (R-53) is in [design.md §6.4](../design.md). R-54 was a set of factual corrections,
which are applied in design §2.
