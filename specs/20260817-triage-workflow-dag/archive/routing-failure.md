# Routing execution failure — scoped design note

## Scope

This note covers only a failure to execute the `routing()` node in the Python triage runner. It does
not change ROUTING's normal treatment of missing or uncertain TAXONOMY kinds, which remains an
explicit set of `branch_decision` elements.

## Decision

When ROUTING raises or returns no result, the runner records ROUTING as `errored` and does not run
AIRWAY, SPEECH, VOICE, or REDACT. Those nodes are recorded `skipped`: without the decisions that
authorise them, running them would create unaudited conclusions.

VERDICT must turn that recorded routing failure and the absence of decisions into a file `flag` with
a reason that names the routing failure. It must not discard the file merely because TAXONOMY had
classified every kind absent.

## Checklist

- [x] Identify the unsafe fail-open path in `_drive_branches`.
- [x] Add runner regressions for routing exceptions and missing results.
- [x] Add a verdict regression for a routing error with no decisions.
- [x] Stop dependent branch execution and record the skipped states.
- [x] Fold the routing failure into an explicit file flag.
- [x] Run focused runner and verdict tests, then the triage workflow suite.
