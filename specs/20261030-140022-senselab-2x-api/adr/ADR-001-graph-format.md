# ADR-001: senselab owns the graph format; it compiles to pydra

**Status:** accepted, 2026-10-30 (owner: "senselab owns the graph format but make it compatible with
pydra"). Amended 2026-11-01, 2026-11-05.

## Context
- Architect review §1 (R-38): keys passed through pydra's hashing hooks, and `Executor` was a name
  with no definition behind it.
- §14 (R-50): route and loop predicates were opaque callables.
- §15 (R-51): fan-out inferred from list-typed ports is a typing trap.

## Decision

**The graph format.** A senselab graph is a serialisable document (JSON against a published schema)
of nodes and edges.
- **A node** is a declared task or pipeline reference with its `version`, its params, and typed
  input and output ports.
- **An edge** joins two ports, and the port types must match.
- **Fan-out is explicit.** `map` edges run the node once per element of a list-typed port, and a
  `gather` collects the results. Wiring a list into an item port without `map` is a type error.
  `over=` names a dataset grouping key only.
- **Routing and loops.** A `route` node and a `loop` node take a **declared decide task** as their
  predicate. Its output is a recorded result, so a resumed run takes the same branch.
- **Nesting.** A pipeline is a node with ports.
- **What hashes are computed from.** The plan hash and every result key are computed from this
  document plus the identities in ADR-002, never from Python objects or function code.

**The `Executor` protocol** (in the runtime, not the kernel; ADR-009):
```python
class Executor(Protocol):
    def submit(self, fragment: PlanFragment) -> Handle: ...
    def status(self, handle: Handle) -> Status: ...
    def cancel(self, handle: Handle) -> None: ...
    def results(self, handle: Handle) -> Iterator[ResultRef]: ...
```
- A `PlanFragment` is a set of logical-plan nodes with their placement (ADR-004), serialised.
- The implementations are `InProcess` and `Pool` (local or Slurm substrates), plus `Pydra`.

**pydra compatibility.**
- A compiler turns a graph into a pydra workflow:
  - `map` and `gather` become split and combine;
  - nesting becomes a nested workflow;
  - `route` and `loop` become a nested workflow over the predicate's result.
- senselab registers byte representations for its types with pydra, and gives each node its
  ADR-002 logical key. pydra's checksum then equals senselab's key, and pydra never hashes
  senselab function code (pydra #777, #776).
- The pydra executor is the runtime that senselab extends (proposal 0001). `InProcess` and `Pool`
  stay available, so senselab's identity and correctness never depend on pydra.

## Consequences
- The pydra decision gate (design §4.9) now decides only the default executor, not the format.
- Agents author pipelines as JSON against the graph schema and get errors from `sl.validate()`.
- `map` is one extra word in pipeline code, in exchange for type errors at build time.

## Alternatives rejected
- **pydra's workflow definition as the format.** It ties senselab identity to pydra hashing, and
  its definitions are Python code, not data.
- **Inferred fan-out.** A silent per-element run when a list was meant (R-51).

## Amendment 2026-11-01: predicates are any declared task

Replaces "a declared decide task" in the Decision text on routing and loops. A `route` or `loop` predicate
is any declared task with a `bool` output; there is no `decide` task kind (task kinds are `transform` and
`fit`). Its output is still a recorded result, so a resumed run takes the same branch. Authority: owner,
2026-11-01 ("apply all 13", 1.x leakage audit); P-20 retired.

## Amendment 2026-11-05: under Route A pydra is the graph runner, not an executor

Replaces three Decision bullets: "The implementations are `InProcess` and `Pool` (local or Slurm substrates),
plus `Pydra`"; "senselab registers byte representations for its types with pydra, and gives each node its ADR-002
logical key …"; and "The pydra executor is the runtime that senselab extends (proposal 0001) …". The `nesting`,
`route` and `loop` compile bullets are unchanged (outside `graph@1.0`, P3-12).

- The `Executor` protocol is `admit(plan, values) -> Admission`, `release(admission)`, `submit(fragment) -> Handle`,
  `status`, `cancel` and `results(handle) -> Iterator[ResultRef]`. Its implementations are `InProcess` and `Pool`
  (local or Slurm substrates). pydra is not an executor.
- pydra is the graph runner (`PydraRunner`). A plan compiles to a pydra workflow: `map` is a split over the
  manifest's records, and record lists zip per record, both within one node mapped over several of them and where
  mapped upstreams meet; `gather` is a combine over the records. A gather into a node that runs once per element,
  and a map over a mapped node's output, are nested fan-outs that `validate` refuses (`nested_fan_out`).
- Phase 3 limit, not a format rule: a map over an unmapped node's list output (design §4.4,
  `p.map(seg.out.segments)`) is a valid graph but is not run in Phase 3, because one item of a stored list output
  has no identity yet. `plan` refuses it (`UnmappedOutput`); R-208 tracks the element identity and the executor's
  per-item read.
- pydra's worker (`ExecutorWorker`) sends each node's ready elements to an `Executor` as plan fragments of the
  placement's batch size; a failed element fails only its descendants.
- Each node's job checksum is supplied by the node and equals the ADR-002 key digest of the job, so pydra hashes
  neither senselab values nor code, and no byte representations are registered.
- pydra's result store holds the run's completion records only; its `lookup` is pass-through to those records.
  Results are stored, and cross-run reuse decided, by the executor from senselab's store under the run's reuse
  policy. `InProcess` and `Pool` stay usable without pydra, so senselab's identity never depends on it.

Authority: Phase 3 Task 9b, the gate's Route A outcome (design §4.9), P3-2.

## Amendment log

- 2026-11-01: route and loop predicates are any declared task with a `bool` output; no `decide` kind (owner, leakage audit).
- 2026-11-05: under Route A pydra is the graph runner above the `Executor`s, not an executor; node-supplied checksums, no byte representations; record lists zip per record; nested fan-outs refused by `validate`, a map over a list output not run in Phase 3 (Task 9b, P3-2).
