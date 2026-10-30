# Engineering standards for senselab 2.x

These apply to every commit, by humans and agents alike. Principles P-31..P-35 in
[`principles.md`](principles.md); testing tiers in [`testing.md`](testing.md).

## Exemplar and sources

**Algorithmic efficiency: `src/senselab/utils/fastio.py`.** It is the model for *how a performance
problem is solved*. A network-filesystem walk is latency-bound, so fastio changes the algorithm:
a bounded thread pool walks the tree concurrently and does the per-file work inside the walk,
instead of running serial loops and second passes. It stays elegant (one private engine, thin
public functions), testable (pure callbacks, deterministic outputs, injected error handling) and
robust (worker errors carried to the caller, a bounded pool, explicit cancellation).
- The lesson: identify the cost model first (latency, compute, memory, I/O), then choose the
  algorithm and data layout that fit it. Examples: vectorised numpy over Python loops, streaming
  over materialising, batching over per-item calls, concurrency for latency-bound work.
- Write that algorithm once, behind a small interface, with tests that pin its behaviour.

**Everything else follows current practice for clean, robust, reusable and efficient code.**
- **Design:**
  - Small, deep modules: simple interfaces hiding substantial work, with information hiding.
  - A functional core with an imperative shell: pure computation, with I/O and side effects at
    the edges.
  - Composition over inheritance, and protocols for boundaries.
  - Dependency injection at boundaries instead of global state.
  - Immutable data where practical.
- **Correctness:**
  - Explicit types (mypy strict on core).
  - Explicit, typed errors; no silent fallbacks.
  - Idempotent, deterministic operations.
  - Validation at boundaries (pydantic), trusted values inside.
- **Reuse:** one implementation per concept, generic over types where the algorithm is generic
  (P-26), with stable, documented public APIs and private internals.
- **Efficiency:** measured, not assumed (P-33). Know the complexity of each hot path; avoid
  accidental quadratics, repeated I/O and per-item process or model start-up.
- **Quality gates:**
  - static analysis (ruff, mypy, import-linter for layering);
  - property-based and fault-injection tests (P-34);
  - small reviewed changes;
  - continuous profiling at scale.

## Standards

| Standard | Rule | Enforced by |
|---|---|---|
| **Minimal verbosity** (P-31) | Say a thing once, in the fewest words that are exact. A docstring states what and how to call it. A comment exists only where code cannot say it. History, measurements and rejected alternatives go to `specs/` (P-21). The same applies to logs, errors, CLI output and agent-facing text: an error names the cause and the fix in one line. | lint over prose: docstring and comment length budgets; banned history words ("originally", "used to", "no longer"); register and round ids outside `specs/`. Review |
| **Clean, robust, reusable, efficient code** (P-32) | Follow the sources above: deep modules with thin surfaces, a functional core with an imperative shell, one implementation per concept, typed explicit errors, and algorithms chosen for the cost model (the fastio lesson), not naive loops. No duplicated logic: a second copy is factored before merge. No dead code, no flags for removed behaviour (P-14). Dependencies stay out of core (P-13). | ruff (including complexity `C901` and `PL` limits), mypy strict on core, a duplicate-code check, and file and function size budgets reported per PR |
| **One implementation per behaviour** (P-37) | Two functions must not do the "same" thing, and code must not be copied with variants. A variant of a behaviour is a typed parameter, a strategy object or a backend over **one** engine. Consolidation must not create brittleness: the merged function never branches on its caller, a dataset or a model id; it has no flag that only one caller uses; and each variant keeps its own contract test. Prefer deleting a redundant function to wrapping it. | a duplicate-code check (token- and AST-similarity across the tree, with a threshold); the task registry refuses two tasks with the same declared operation and port types; an adversarial "find the duplicates" agent pass each phase; review checks flag count and caller-specific branches |
| **Efficient at every scale** (P-33) | Code is measured at four scale points: 1 item, 10², 10⁴, and a long signal (hours of audio, days of actigraphy). Time, peak memory, files written and bytes moved per item must not grow with scale. Profiles find the hot spots, not guesses. | `benchmarks/scale/` ladder runs per PR on touched modules (small points) and nightly (large points). Scalene profiles attached on regression. Per-item budgets stored as data with derivations (P-17); a regression beyond a band blocks merge |
| **Brittleness checked always** (P-34) | A change must not make the system depend on an accident: timing, ordering, one dataset, one host, one model's quirk, a magic number, a string match on output. Every boundary is exercised with bad input. Failures are typed and isolated. | property tests and fault injection in the logic tier (killed worker, stale lease, NFS rename race, network off, missing license, corrupted cache entry, truncated audio). A brittleness checklist in the PR template. Review rejects special-casing |
| **Adversarial agents set standards** (P-35) | Agents are used to break the system: hostile inputs, races, scale, misuse of the API, and attempts to make a test pass by special-casing. Each break is fixed **by an architectural change**, not a patch for that case: a new invariant, a type, a contract test or a lint rule. The fix is generalised into the standards here or into a principle. | each phase ends with an adversarial round scoped to its new surfaces (`design.md` §6.2). Findings go to [`review-register.md`](review-register.md) with their architectural resolution. A fix that only adds an `if` for the found case is rejected |

## Per-commit checklist

The PR template carries this checklist. Pre-commit and CI check what they can.

- [ ] Prose is minimal: no history, no ids, docstrings say what and how (P-31).
- [ ] No duplicated logic; surfaces are thin over shared engines (P-32).
- [ ] No second function for an existing behaviour, and no copied variant. Variants are
  parameters, strategies or backends, with no caller-specific branches (P-37).
- [ ] The scale ladder passes for touched modules; per-item cost is flat (P-33).
- [ ] New boundaries have bad-input, fault and property tests; there is no special-casing (P-34).
- [ ] No identifiers; tests are scoped and serial (P-28, P-29).
- [ ] Every new task, backend or profile is declared, so contract tests cover it (P-30).

## What "special-casing" means (rejected in review)

- A branch on a dataset, recording, subject, model id or host name.
- A threshold literal tuned to make one case pass (P-17).
- A retry, sleep or timeout added to hide a race instead of removing it.
- A test that asserts the current output of a model, or a measured number, outside `benchmarks/`.
- A try/except that turns an unknown failure into a default value.
