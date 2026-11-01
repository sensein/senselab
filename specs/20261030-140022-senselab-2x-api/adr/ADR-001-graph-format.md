# ADR-001: senselab owns the graph format; it compiles to pydra

**Status:** accepted, 2026-10-30 (owner: "senselab owns the graph format but make it compatible with
pydra").

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
