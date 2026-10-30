# senselab 2.x testing and evaluation design

Status: draft. Principles: P-30 (tiers; benchmarks separate), P-21 (no rationale in code), P-29
(local plus cluster within shared limits), P-28 (no identifiers), P-22 (Python range).

## 1. What the 1.x suite tells us

These figures come from an audit of `src/tests` at alpha `11992c7d`. Counts were rechecked.

**Size**
- 8,068 tests in 423 files, about 151k lines. Triage holds 38% of the suite.
- Docstrings and comments are 14% of all lines, and up to 45% in some files.

**Parametrisation and markers**
- There are 280 uses of `@pytest.mark.parametrize` and no Hypothesis.
- Variations are mostly hand-written as separate near-duplicate test methods, with setup stacks
  copied between them.
- The markers (`slow`, `large_model`, `network`) are not registered and not deselected. CI runs
  everything, so the markers are labels, not gates.

**Where the verbosity comes from**
- **History in prose:** "this assertion is inverted from what it originally claimed"; spec and
  round ids in headers and file names (`*_r37_test.py`).
- **Measured magic numbers in logic tests:** `0.267125` s, an observed bounding box.
- **Config change-detectors:** `triage/config_test.py` asserts packaged defaults such as
  `pitch_floor_divisor == 1.5`.
- **Source-text greps** of worker scripts.
- **Large inline seeding:** a 1,107-line `conftest.py`.
- **About 3,000 `monkeypatch` uses.** `ensure_venv`, `subprocess.run` and `resolve_revision` are
  stubbed again and again.

**What 1.x does well**
- Synthetic signals generated inline.
- Per-test cache isolation.
- Real exception objects instead of doubles where a double had hidden a bug.

The 2.x suite keeps those and removes the rest by construction: the declarations in `design.md`
let most tests be generated rather than written.

## 2. Tiers

Every test belongs to exactly one tier, by marker. Markers are registered and `--strict-markers` is
on. The default run is `logic` only.

| Tier | Marker | Exercises | Inputs | Runs where | Gate |
|---|---|---|---|---|---|
| **Logic** | `logic` (default) | Behaviour of code: data types, identity and keys, params validation, planner decisions, graph semantics, decision rules, codecs | small synthetic signals; the `fake` stack | every PR on the min and max Python plus a core-only install; full Python matrix nightly; worker-runtime tests on every env Python from the built wheel | blocking |
| **Contract** | `logic` (generated) | Every declared task, backend, env and profile meets the framework contract | declarations | with logic | blocking |
| **Limits** | `limits` | Parameter boundaries and edge inputs against real backends | synthetic edge signals + 3–4 shared fixtures | label-triggered on PRs; nightly; GPU subset on ORCD | blocking for touched backends |
| **Use case** | `usecase` | One realistic end-to-end pipeline per use case | small public or synthetic recordings with licenses recorded | nightly; ORCD before release | blocking for release |
| **Env** | `env` | Each locked env builds, imports the worker runtime and round-trips every data type | lock files | changed locks on the PR; full builds weekly on ORCD | blocking for that env |
| **Benchmark** | not pytest | Accuracy and speed on public data, tracked over releases | public benchmark subsets and full sets | CPU smoke (<15 min) on PRs touching a task and nightly; GPU smoke weekly on ORCD; full per release on ORCD | tracked, not pass/fail; regressions beyond a band open an issue |

Within limits and use cases, sub-markers `gpu`, `network` and `slow` select hardware and runtime.
Both tiers are deselected by default.

## 3. Logic tests: write less, generate more

**Generated contract tests**

For each declared task, backend, env and profile, one generic suite checks:
- Params schema round-trip, unknown key rejected with a suggestion, defaults valid;
- the output type matches the declaration, and the output survives JSON and array serialisation;
- the logical key ([ADR-002](adr/ADR-002-identity-v2.md)) is deterministic and portable: the same
  inputs give the same key on every platform in the CI matrix. The key changes when each logical
  component changes. It does not change when an execution-context field or placement field
  changes (env lock, numerics class, decoder, batch size, device, partition);
- early cutoff (I-5): an upstream change that yields identical output leaves downstream keys
  unchanged;
- the reuse policy: `exact` rejects a context mismatch, `equivalent` with no measurement behaves
  as `exact`, and `tolerant` accepts and labels the result;
- graph documents round-trip and validate; a list wired to an item port without `map` is a type
  error (ADR-001);
- a `measure` task's output schema carries no verdict fields (P-20);
- provenance is emitted with the required fields;
- a license record exists, and a missing one raises under the default policy;
- `Absent` and failure are distinguished, and only content-caused `Absent` is read back from the
  cache (I-6);
- `describe()` produces a valid JSON Schema.

A new task gets this suite by being declared. These tests are generated by one parametrised
module over the registry, with ids equal to the task, backend or profile name.

**Property-based tests** (Hypothesis) for generic signal tasks and core algebra:
- resampling preserves duration and band-limited content;
- gain and channel-order invariances;
- silence yields zero or `Absent` where declared;
- span algebra laws (union and intersection associativity, clipping to extent);
- `TimeMap` composition and inversion;
- identity hashes ignore container format and are stable across numpy versions.

**Example tests** for decision logic: small tables, one row per behaviour.
- Ids describe the behaviour (`"short_vowel_routes_to_review"`), never a register or round number.
- Real-world anecdotes that found a bug become a behavioural row plus a benchmark case, not a
  pasted transcript.

**The `fake` stack**
- A registered backend stack that returns typed outputs derived from the input identity.
- It has a **declared fault model** that logic tests parametrise over: dependence on batch-mates,
  nondeterminism, failure on a single item, OOM, a crash mid-batch, and latency. Planner and
  executor tests therefore exercise I-3, the fault taxonomy and scheduling fairness
  (`execution.md` §3), not only the happy path.
- It replaces the monkeypatch stacks: logic tests of pipelines, the planner and executors run on
  the fake stack through the same registry users use.
- `no_hub` and `fake_env` fixtures cover model resolution and env building.
- Monkeypatching senselab internals is forbidden in 2.x tests.

## 4. Limits and use-case tests

**Limits**
- Generated from the declared `Params` ranges and input types, then run on a real backend.
- Each backend runs the generic edge set:
  - empty, one-sample and shorter-than-window input;
  - silence, clipping, DC offset;
  - each supported sample rate and channel layout;
  - long input that forces chunking;
  - each supported device class;
  - each declared stack.
- It also runs boundary values of each `Params` field (min, max, each enum member).
- Assertions are on type, shape, ranges, determinism (same identity gives the same result) and
  agreement across stacks within the measured tolerance profile. Never on exact model scores.
- **Identity contracts on real backends:**
  - **perturbation (I-1):** perturb every undeclared source (env vars, other profile keys, cache
    roots); the output must not change;
  - **batch invariance (I-3):** batch of 1 against mixed-length batches, within the measured
    tolerance; otherwise the backend must declare itself not batch-invariant;
  - **service attestation (I-8):** restarting a service with a new config changes keys;
  - **cross-context equivalence (P-38):** the limits tier runs each backend in every context of the
    CI and ORCD matrix (platform, device class, precision). Agreement within tolerance is written
    to the backend's tolerance profile; this is the data `equivalent` reuse relies on;
  - **verify sampling:** a run with `verify: k` recomputes k% of cross-context hits. Agreements are
    recorded and disagreements are reported against the equivalence class.
- **Version-bump check (CI, not a test):** a change to a task's or backend's source digest without a
  `task_version` or `backend_version` bump fails the build (ADR-002).
- **Format migrations:** every reader has a test that reads the previous major format (ADR-006).
- Non-audio signals use generators (synthetic ECG and accelerometry, for example NeuroKit2's MIT
  simulators) for the same edge set.

**Use cases**
- One per row of `use-cases.md` §1 that 2.0 supports: for example clinical voice screening,
  conversation turn-taking, and a video → audio → text chain.
- Each is a small pipeline over a few recordings with licenses recorded.
- They assert that the pipeline completes, that outputs are schema-valid, that provenance is
  complete, and that plan hashes are stable.
- They double as the 2.x tutorials, so tutorials are tested by construction.

**Parity tests (migration only)**
- `tests/parity/` compares 1.x and 2.x outputs per ported task, within measured tolerance profiles
  (`design.md` §6).
- They are deleted, along with the 1.x code, in Phase 5.

## 5. Where and how tests run (P-29)

**Local**
- `uv run pytest` runs logic tests in under a few minutes on a laptop, serially, scoped to the
  package changed.
- `uv run pytest -m limits -k <backend>` runs one backend's limits.

**ORCD**
- GPU limits, use cases and env builds run as submitted jobs from a pinned checkout, on the
  partition named by the test profile.
- They use typed GPUs, preferring H100. Concurrency is bounded by the QOS cap shared across
  agents. Nothing runs on login or remote-control nodes.

**Cadence** (the same table drives `benchmarks.md` §5)

| When | What |
|---|---|
| every PR | logic + contract on min and max Python; core-only install; env builds for changed locks; CPU smoke benchmarks (<15 min) when a task is touched |
| label on a PR | limits for the touched backends |
| nightly | full Python matrix; limits; scale ladder (`benchmarks/scale`, P-33); CPU smoke benchmarks |
| weekly (ORCD) | GPU smoke benchmarks; core CPU benchmark suite; full env builds; use cases |
| per release (ORCD) | full benchmarks; GPU limits |

**Forbidden in 2.x tests** (enforced by a lint over `src/tests`, not by tests):
- spec, register or round ids in names, files or headers;
- history narratives ("used to", "originally");
- exact model-output floats in logic tests;
- asserting packaged config defaults;
- grepping source text;
- autouse fixtures that assert;
- `getfixturevalue` string parametrisation;
- single-element parametrize;
- unregistered markers;
- network in unmarked tests;
- real identifiers (P-28).

**Policy checks that were AST tests** (layering, revision pinning, network guard) become lint
rules in `scripts/check_*.py` or import-linter config. They are short, with allowlists as data
files, not essays.

## 6. Benchmarks (separate evaluation)

**Structure**
- Benchmarks live in `benchmarks/`, outside pytest, and are senselab pipelines.
- Each benchmark is a dataset adapter for a public dataset, a pipeline under test, and a **metric
  task**. Metrics (WER, DER, EER, F1, MAE, …) are declared, versioned senselab tasks whose
  implementations are checked against the reference harness for that benchmark (for example the
  HF Open ASR normaliser, the pyannote metrics, PhysioNet scorers).
- Results go to a benchmark store with full provenance (model identity, env, stack, hardware,
  plan hash) and are tracked across releases. A regression beyond a measured band opens an issue
  rather than failing CI.
- Every measured number that 1.x asserted in tests becomes a benchmark case with a band.
  Examples: speaker-count ceilings, degradation-suite detections, device parity, ASR WER on AMI,
  classifier level baselines.

**Suites:** smoke and full, on the cadence in section 5.
- Gated or credentialed datasets run only where access is held, and their derived results are
  published only as their terms allow.

The dataset choices per task and modality are in [`benchmarks.md`](benchmarks.md). It records each benchmark's license and access terms, which results may be published, reference scorers, smoke subsets, and cadence.

## Test data

Toy fixtures in the repository versus versioned data pulled from public sources follow
[`docs.md`](docs.md) (data policy). The same manifest and fetch serve use-case tests, benchmarks
and docs.
