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

## Amendment 2026-11-09: placement is an event of the one log (Phase 3 Task 20R-12)

- A run has one event log. Each class's placement in an epoch is a run-scoped `placement` event of `run-event@2.1`
  (ADR-006), appended through the epoch's `EventLog` by the pool; status, `explain` and reports read it there. The
  closing actuals record of T20R-7b is appended the same way.

## Alternatives rejected
- **Free-text logs only.** Not machine-readable for agents.
- **A metrics server.** It adds an operated dependency.

## Amendment 2026-11-04: OpenTelemetry export deferred (Phase 3 Task 10)

- The optional OpenTelemetry export of the events (Decision, Export) is deferred; it is not built in Phase 3
  (ruling P3-28). The events (`run-event@1.0`, ADR-006) are the single source of status, metrics and `explain`, and
  an exporter reads them, so adding it later changes no event.
- `status_from(events)` derives every metric of a run (counts by state, faults by class, broken envs) from the events
  alone (a run's `done` and `cancelled` included, and `last_event_ns` for one whose process died; no status file
  exists to race on); `senselab status <run>` and `senselab explain <run> <node|key>` read the run directory and nothing else.
