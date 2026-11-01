# ADR-008: Observability

**Status:** accepted, 2026-10-30.

## Context
Architect review §11 (R-47): logs, metrics and per-attempt diagnostics were unspecified, so a failed
corpus run could not be debugged except by reading worker stderr.

## Decision
- **Events.** Every attempt emits structured JSON-lines events to the run directory:
  - `queued`, `claimed`, `started`, `finished`, `failed` (with its fault class from ADR-009),
    `yielded`, `cache_hit` (with its reuse policy and context match), `verify` (agree or
    disagree);
  - each event carries the node, key, attempt, placement and timestamps.
- **Metrics are derived from the events,** never maintained separately: counts and durations per
  node, queue depth, hit rate per reuse policy, fault rate per class, GPU utilisation from the
  worker heartbeats.
- **Commands:**
  - `senselab status <run>` summarises a run;
  - `senselab explain <run> <node|key>` shows why a node failed, was skipped, hit the cache or was
    placed where it was: events, fault, stderr tail and placement reasoning.
- **Export.** Optional OpenTelemetry export of the same events.

## Consequences
- Events are the single source for status, the MCP progress tool and the benchmark timings.
- The event schema is a versioned format (ADR-006).

## Alternatives rejected
- **Free-text logs only.** Not machine-readable for agents.
- **A metrics server.** It adds an operated dependency.
