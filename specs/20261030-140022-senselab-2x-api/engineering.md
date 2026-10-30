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

## Process and engineering principles

The authoritative text of P-27..P-35. `principles.md` keeps their ids and stubs.

<a id="p-27"></a>
### P-27 Commit and push early and often
- **Status**: active
- **Statement**: Each unit of work is committed when done and pushed promptly. Work in progress
  stays visible on a branch with a PR to `alpha`. 2.x is built in place on `alpha`.
- **Source**: owner, 2026-10-30: "commit+push early often"; "in place on alpha".
- **Check**: no local-only work older than a session.

<a id="p-28"></a>
### P-28 No identifiers leak
- **Status**: active
- **Statement**: No participant, subject or session identifier, no personal path and no site
  credential enters the repository, a spec, a test fixture, a commit message or a PR. Examples use
  synthetic identifiers.
- **Source**: owner, 2026-10-30: "ensure no identifiers leak in".
- **Check**: the pre-commit identifier hook passes, and reviewers check prose, which the hook may
  not cover.

<a id="p-29"></a>
### P-29 Fast test cycles locally and on the cluster, within shared limits
- **Status**: active
- **Statement**: Tests run locally for fast cycles and on ORCD for GPU and scale. Agents running in
  parallel share both machines: scope test runs to what changed, run serially (never
  `pytest -n auto`), respect the cluster's partitions and QOS caps, and do no compute on login
  nodes.
- **Source**: owner, 2026-10-30: "for testing do local tests and on orcd to have faster cycles, but
  keep load across agents in mind, and orcd partitions in mind".
- **Check**: test jobs name a partition and resource class; concurrent agent jobs stay within the
  QOS cap.

<a id="p-30"></a>
### P-30 Tests are tiered; benchmarks are separate
- **Status**: active
- **Statement**: Tests come in tiers (`testing.md`):
  - **logic**: behaviour, with small synthetic inputs and no models;
  - **contract**: generated from declarations;
  - **limits**: parameter boundaries with real backends;
  - **use case**: realistic end-to-end pipelines;
  - **env**: locked environments build and round-trip.

  Tests are short and named for the behaviour they check. Parametrisation enumerates meaningful
  cases, never encoded history. **Benchmarks** are a separate evaluation on public data, run as
  senselab pipelines with provenance and tracked over releases rather than gated as pass/fail.
- **Source**: owner, 2026-10-30: "many tests in current senselab are overly verbose or strangely
  parameterized ... some tests should just exercise logic, and some tests should exercise the limits
  of parameters and use cases. then there should be benchmarks as a separate evaluation with public
  benchmarks".
- **Check**: every test belongs to exactly one tier (marker). No test docstring narrates a
  measurement history (P-21). No parametrize id is a register number.

<a id="p-31"></a>
### P-31 Minimal verbosity for humans and agents
- **Status**: active
- **Statement**: Code, docstrings, comments, logs, errors and agent-facing text say each thing
  once, in the fewest exact words. Reasons and history live in `specs/`.
- **Source**: owner, 2026-10-30: "we want to make verbosity minimal for the human and any agent".
- **Check**: prose lint budgets; review. See `engineering.md`.

<a id="p-32"></a>
### P-32 Clean, robust, reusable, efficient code
- **Status**: active
- **Statement**: Code follows current best practice for clean, robust, reusable and efficient code:
  deep modules with thin surfaces, a functional core with an imperative shell, one implementation
  per concept, explicit types and errors, and no duplicated logic.
  - Performance problems are solved algorithmically, chosen for the cost model. That means
    vectorisation, streaming, batching, and concurrency for latency-bound work, not naive loops.
  - `utils/fastio.py` is the exemplar of that elegant, testable, robust algorithmic efficiency.
- **Source**: owner, 2026-10-30: "code itself should be clear to read and compact/efficient (think
  fastio code)"; "the fastio example was more about an elegant, testable, robust algorithmic
  efficiency instead of just writing for loops. the rest of the pieces should come from best
  practices in the field today for clean, robust, reusable, and efficient code".
- **Check**: ruff (including complexity), mypy strict on core, import-linter, a duplicate-code
  check, the scale ladder (P-33), and review against `engineering.md`.

<a id="p-33"></a>
### P-33 Profiling detects inefficiency at every scale
- **Status**: active
- **Statement**: Every module is measured at several scale points (one item, many items, a corpus,
  a long signal). Per-item time, memory, files and bytes must stay flat with scale, and profiles
  locate regressions.
- **Source**: owner, 2026-10-30: "profiling should detect inefficiencies at different scales".
- **Check**: the `benchmarks/scale/` ladder with per-item budgets as data; Scalene profiles on
  regression.

<a id="p-34"></a>
### P-34 Brittleness is always checked
- **Status**: active
- **Statement**: No change may rely on timing, ordering, one dataset, one host, one model's quirk, a
  magic number or a string match. Boundaries are exercised with bad input and injected faults, and
  failures are typed and isolated.
- **Source**: owner, 2026-10-30: "brittleness has to be checked for always".
- **Check**: property tests and fault injection in the logic tier; the PR checklist; review rejects
  special-casing.

<a id="p-35"></a>
### P-35 Agents break the system to raise standards, not to add brittleness
- **Status**: active
- **Statement**: Agents are used adversarially to break the system. Each break is resolved by an
  architectural change (an invariant, a type, a contract test, a lint rule, or a principle), never
  by a patch for the found case.
- **Source**: owner, 2026-10-30: "agents should be used to break system to set architectural
  standards not more brittleness".
- **Check**: each phase ends with an adversarial round. Findings are recorded with their
  architectural resolution, and case-only fixes are rejected.
