# senselab 2.x API — principles register

This register is the source of truth for what the senselab 2.x API must satisfy. A design, a plan
or a review cites principles by id (`P-3`). When a design and this file disagree, this file wins until
it is amended here.

## How to change this register

- **Add** with the next unused id (ids are never reused). **Amend** in place; **retire** by setting
  `Status: retired` and naming what supersedes it.
- Record conflicts under *Resolved tensions*, and log every change, marking changes of meaning.
- Only the owner adds or amends an owner principle's statement. Agents propose changes in PRs, and
  may fix *Check* fields for consistency, logging each fix.
- Fields: **Statement** (normative), **Source** (an owner quote and date, or an inherited path),
  **Check** (how a reviewer decides compliance).

---

## Owner principles

### P-1 One API for humans and agents
- **Status**: active
- **Statement**: The public API is easy to understand and use for a person at a REPL and for an
  agentic system calling tools. Every task, model, backend and pipeline can be discovered at runtime
  and described in a machine-readable schema (names, typed parameters, defaults, return type,
  supported hardware). That schema is generated from the same code humans call, never maintained
  separately.
- **Source**: owner, 2026-10-30: "design an api that is easy to comprehend and use by both humans and
  agentic systems".
- **Check**: an agent with no repository access can list capabilities, get a schema and run a task
  from a schema alone. No task requires reading source or external docs to call correctly.

### P-2 Separate what is computed from how and where it runs
- **Status**: active
- **Statement**: The API is designed for both *use* (declaring what to compute) and *execution*
  (running it efficiently). These are separate layers. The same declared task or pipeline runs
  unchanged in-process, in an isolated environment, on a local GPU or CPU, or on a cluster; only
  the execution plan differs.
- **Source**: owner, 2026-10-30: "both use and execution should be kept in mind".
- **Check**: no task or pipeline definition names a device, a venv, a scheduler or a host.
  Changing the execution target does not change result identity, apart from recorded
  hardware-dependent numerics.

### P-3 Versioning and provenance are built in
- **Status**: active
- **Statement**: Every result records what produced it: senselab version, the task and its version,
  the resolved model identity (a content hash or commit, never a moving ref), the effective
  parameters, input content identity, the backend/stack and environment, and the hardware. This is
  recorded by the framework, not by each task.
- **Source**: owner, 2026-10-30: "senselab principles of versioning/provenance", "hashing models".
- **Check**: a result can be traced back to the exact model bytes and parameters that produced it, by
  a tool that knows nothing about the task.

### P-4 Models are identified by hash
- **Status**: active
- **Statement**: A model is identified by the content of what is loaded: a resolved commit SHA for
  hub models, a content hash for weights from any other source. Loading always goes through that
  identity. Cache keys and provenance use it.
- **Source**: owner, 2026-10-30: "hashing models".
- **Check**: no loader accepts a moving ref without first resolving it, and no cache key carries a
  bare model name.

### P-5 Efficient and scalable execution is the default
- **Status**: active
- **Statement**: Execution is efficient by default (batching, model reuse across inputs,
  content-addressed caching, no repeated loads) and scales from one file on a laptop to a dataset on
  a cluster without rewriting the pipeline.
- **Source**: owner, 2026-10-30: "efficient+scalable execution", "working on local and clusters".
- **Check**: the same pipeline object runs on one file and on a whole dataset, and a re-run after
  partial completion recomputes only what is missing.

### P-6 Isolated environments are part of execution
- **Status**: active
- **Statement**: A backend whose dependencies conflict with the core or with each other runs in an
  isolated environment (subprocess venv, persistent worker, container) that the framework creates,
  locks, reuses and records. Isolation is declared by the backend, chosen by the executor, and
  invisible at the call site.
- **Source**: owner, 2026-10-30: "this includes subprocess environments".
- **Check**: calling an isolated backend looks the same as calling an in-process one. The
  environment's resolved lock is part of provenance (P-3).

### P-7 Hardware-aware optimisation
- **Status**: active
- **Statement**: Execution is optimised for the available hardware (GPU or CPU, its memory,
  precision support, Apple silicon). Batch size, precision, attention kernel and placement of
  CPU-bound versus GPU-bound nodes are decided by the executor from a hardware description, not
  hard-coded in tasks.
- **Source**: owner, 2026-10-30: "optimization for GPU versus CPU".
- **Check**: a task declares its resource needs and the knobs it can tune. The values come from an
  execution profile or from catalogue resource tables, which the scale ladder produces (Phase 3).

### P-8 Inference stacks are interchangeable
- **Status**: active
- **Statement**: A model can be served by more than one inference stack (transformers, vLLM,
  vLLM-omni, NeMo, MLX, ONNX, …). The stack is an execution choice behind a common interface.
  Switching stacks changes neither the task call nor the result schema; it is recorded in
  provenance and is part of result identity where it can change outputs.
- **Source**: owner, 2026-10-30: "optimization for inference, changing stacks (vllm, vllm-omni,
  transformers, etc)".
- **Check**: the same task runs on two stacks by changing only the execution profile.

### P-9 Pipelines are clear, composable and nestable
- **Status**: active
- **Statement**: Pipelines are declared graphs of nodes with explicit inputs and outputs. A pipeline
  is itself a node, so pipelines nest. Simple pipelines read simply; complex ones stay legible
  because each level shows only its own graph.
- **Source**: owner, 2026-10-30: "design clear and complex pipelines. where pipelines can be nested".
- **Check**: any pipeline can be used as a node in another without adapters, and its graph can be
  rendered or listed without running it.

### P-10 Pipelines adapt to environment, hardware and dataset
- **Status**: active
- **Statement**: A pipeline is adapted to a concrete setting (the execution environment, available
  hardware and the dataset) by planning, not by editing the pipeline. Adaptation produces an
  inspectable execution plan, which is recorded.
- **Source**: owner, 2026-10-30: "being able to adapt a pipeline for an execution environment and
  hardware availability for a dataset".
- **Check**: the plan can be produced and inspected without executing (a dry run), and the plan for
  a cluster differs from the plan for a laptop only in execution fields.

### P-11 A dataset is a layout or a mapping behind one interface
- **Status**: active
- **Statement**: A dataset is either a layout on disk (BIDS first) or a virtual mapping held in
  memory. Pipelines consume both through one interface that yields records (data plus metadata such
  as subject, session, task and sidecar fields) and can be sliced, filtered and resumed.
- **Source**: owner, 2026-10-30: "where dataset is a layout like bids on disk, or a virtual mapping
  held in memory".
- **Check**: the same pipeline runs on a BIDS tree and on an in-memory list of recordings with no
  code change.

### P-12 Pipelines are generic and configured per dataset
- **Status**: active
- **Statement**: A pipeline is written once and reconfigured for a dataset. It is not written for one
  dataset. Dataset-specific knowledge (paths, naming, sidecar fields, task vocabularies, hints,
  thresholds fitted on that data) lives in a dataset adapter or a configuration profile, never in
  pipeline or task code.
- **Source**: owner, 2026-10-30: "many recent pipelines implemented became dataset specific rather
  than pipelines that can be easily re-configured for a dataset (preferred)".
- **Check**: grep pipeline and task code for dataset names, path patterns and dataset field names
  (e.g. `b2ai`, `stimulus_text`); every hit is a violation.

### P-13 The core stays lightweight
- **Status**: active
- **Statement**: senselab core installs and imports without torch or any ML framework. It holds the
  data structures, the task and pipeline declarations, the registry, provenance, hashing,
  configuration, the dataset interface and the executor protocol. Heavy backends are optional
  packages or isolated environments.
- **Source**: owner, 2026-10-30: "we need to keep senselab core as lightweight as possible".
- **Check**: a CI job installs core alone and imports every public core module; importing core
  pulls no ML framework and has no global side effects.

### P-14 senselab 2.x breaks compatibility outright
- **Status**: active
- **Statement**: 2.x replaces the 1.x API. There are no shims, aliases, deprecation paths or
  parallel fields. Migration is documented, not supported in code.
- **Source**: owner, 2026-10-30: "this will all go into a senselab 2.x release so backwards
  compatibility is not required".
- **Check**: no 1.x name is re-exported from 2.x.

### P-15 Principles are a living register
- **Status**: active
- **Statement**: These principles are kept as an editable register (add, retire, amend) so that
  architectural intent survives across sessions and is reusable by humans and agents. Designs cite
  principle ids.
- **Source**: owner, 2026-10-30: "turn my requests into a set of notes so we don't forget these
  principles as context moves. principles should allow modifying (add/delete/update)".
- **Check**: every design decision for 2.x names the principles it serves or trades off.

### P-22 Core runs on both sides of every environment boundary
- **Status**: active
- **Statement**: senselab core supports only CPython releases not past end-of-life and drops each
  version at EOL. Isolated environments may use older Python versions when a model needs them.
  Every environment, whatever its Python, runs the senselab worker runtime (wire codec, worker
  loop, weight hashing, provenance records; stdlib + numpy only), so host and workers share one
  protocol. Host and worker versions match, verified at handshake.
- **Source**: owner, 2026-10-30: "can senselab core be used across subprocess venvs so that install
  works across python versions"; "we can drop python versions that will have gone into EOL";
  "subprocess envs can use older versions".
- **Check**: core CI covers every non-EOL Python. Worker-runtime CI covers every Python any env
  lock uses. Every env build imports the worker runtime and round-trips each data type over the
  wire.

### P-23 Licenses are explicit and machine-readable
- **Status**: active
- **Statement**: Every model, code package, dataset and therefore every task call and every plan has
  a machine-readable license record (SPDX where possible), with derived flags (commercial use,
  copyleft, gated, unknown).
  - Policies are expressed against named license sets (for example `permissive`, `copyleft-ok`,
    `non-commercial-ok`) that users can define and combine.
  - The planner enforces the policy before execution.
  - A missing or unknown license raises by default. A policy flag can relax that to a warning,
    and the relaxation is recorded.
  - Licenses are recorded in provenance.
- **Source**: owner, 2026-10-30: "one of the challenges of senselab is knowing what license a given
  task uses (GPL/NC/apache, etc) which needs to be very clear and machine accessible"; "missing
  licenses can be a flag that defaults to raising exception and the process could be generalized
  to licensing sets".
- **Check**: `sl.licenses(plan)` answers without running anything. The CI gate equals the run-time
  default: it is applied to every catalogue model and locked package (transitive included) shipped
  at that phase (`design.md` §4.13).

### P-24 Modalities share types; environments share by dependency set
- **Status**: active
- **Statement**: Audio, video and text share one timeline, annotation, embedding, record and
  identity model, so cross-modal pipelines chain without glue. Environments are not owned by a
  modality or a task: backends declare requirements, and backends whose combined requirements
  resolve may share an environment and a worker.
- **Source**: owner, 2026-10-30: "with audio/video/text processing enabled consider what
  unions/intersections are across such processing (and knowing subprocess envs exist)".
- **Check**: a video → audio → text pipeline needs no conversion code between nodes. The env
  grouping is computed, recorded as data, and adopted in Phase 4.

### P-25 Every execution environment is a locked, isolated venv
- **Status**: active
- **Statement**: senselab never runs on unlocked or system state. senselab itself is installed into
  a venv built from a lock, and every backend runs in a locked venv: either the senselab venv,
  when its lock covers the backend, or its own isolated env. Native libraries come from wheels.
  A system dependency that cannot be avoided is declared, probed at start, version-recorded in
  provenance, and its absence is an error, not a fallback.
- **Source**: owner, 2026-10-30: "we should always be using isolated envs for reproducibility, not
  relying on system libraries. i.e. one requirement is to install senselab into a venv".
- **Check**: every activity in provenance names its env in the recorded execution context (ADR-002).
  The env tier audits each env's imports against its lock.

### P-26 Signals generalise; modality-specific tasks stay specific
- **Status**: active
- **Statement**: Audio is one kind of signal. The data model and the task catalogue cover any
  sampled or event-based signal (actigraphy, ECG, PPG, EDA, respiration and others). An algorithm
  that applies to any time series is written once over the general signal type and parameterised
  per modality through profiles. Tasks that are inherently modality-specific are declared on the
  specific type.
- **Source**: owner, 2026-10-30: "future extensions to actigraphy, ecg, and other such signals. in
  many cases certain algorithms can be applied across different timeseries, and some are going to
  be specific to the modality".
- **Check**: adding a new signal modality needs a data type, an adapter and profiles, not copies
  of generic algorithms.

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

### P-36 The flywheel is the roadmap goal; it informs, not forces
- **Status**: active
- **Statement**: senselab's long-term goal is a flywheel in which machines and humans iteratively
  improve the system. Datasets, pipelines, annotations, reviews, fitting and model fine-tuning
  happen as needed, and are evaluated and promoted by identity. 2.x builds none of it, but its
  decisions must keep the hooks in `roadmap.md` possible. Triage's decision layer is rebuilt as a
  flywheel application after 2.x rather than ported.
- **Source**: owner, 2026-10-30: "triage should adopt a flywheel approach of a system that allows
  machines and humans to iteratively improve the system ... make this a future roadmap goal that
  should inform not force current decisions".
- **Check**: design reviews confirm that hooks H-1..H-7 in `roadmap.md` stay possible. No 2.x phase
  takes on flywheel features.

### P-37 One implementation per behaviour, consolidated without brittleness
- **Status**: active
- **Statement**: There are never multiple functions that do the same thing, nor chunks of code
  copied with variants. Variants are expressed as typed parameters, strategies or backends over one
  engine. Consolidation must not add brittleness: no caller-specific branches, no single-caller
  flags, and every variant keeps its own contract test.
- **Source**: owner, 2026-10-30: "eliminate function redundancy and consolidation without making it
  brittle. there should not be multiple functions that do the "same" thing, or chunks of code copied
  with variants".
- **Check**: the duplicate-code check; the task registry refuses duplicate operations; an
  adversarial duplicate-finding pass per phase; the consolidation inventory in design §6.1 is
  emptied by Phase 5.

### P-38 Hashes establish exactness; reuse across contexts is measured
- **Status**: active
- **Statement**:
  - Hashes establish exactness only, and only over facts that hash the same on every machine:
    declared versions, model weight bytes, canonical params, raw input bytes, upstream outputs.
  - Function code, Python objects and environments are never hashed into identity. They differ
    across platforms, and a hash cannot measure distance (pydra #777, #776).
  - Whether a result produced in another context may be reused is a user-chosen reuse policy
    (`exact`, `equivalent`, `tolerant`). Equivalence comes from measured tolerances and verify
    sampling.
- **Source**: owner, 2026-10-30: "i don't see how senselab can do all the hashing ... many
  functions may be environment dependent (same function may hash differently between macos and
  linux). since hashing can't be used to judge distance, only exactness, should caches be
  invalidated when moving between machines? perhaps so, but a user may say i don't care as long as
  the outputs are the same/close for given inputs".
- **Check**: ADR-002 key contract tests; a source change without a version bump fails CI; under
  `equivalent`, unmeasured contexts behave as `exact`.

---

## Process principles

How 2.x is built. These bind the people and agents building it, not the API.

### P-27 Commit and push early and often
- **Status**: active
- **Statement**: Each unit of work is committed when done and pushed promptly. Work in progress
  stays visible on a branch with a PR to `alpha`. 2.x is built in place on `alpha`.
- **Source**: owner, 2026-10-30: "commit+push early often"; "in place on alpha".
- **Check**: no local-only work older than a session.

### P-28 No identifiers leak
- **Status**: active
- **Statement**: No participant, subject or session identifier, no personal path and no site
  credential enters the repository, a spec, a test fixture, a commit message or a PR. Examples use
  synthetic identifiers.
- **Source**: owner, 2026-10-30: "ensure no identifiers leak in".
- **Check**: the pre-commit identifier hook passes, and reviewers check prose, which the hook may
  not cover.

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

### P-31 Minimal verbosity for humans and agents
- **Status**: active
- **Statement**: Code, docstrings, comments, logs, errors and agent-facing text say each thing
  once, in the fewest exact words. Reasons and history live in `specs/`.
- **Source**: owner, 2026-10-30: "we want to make verbosity minimal for the human and any agent".
- **Check**: prose lint budgets; review. See `engineering.md`.

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

### P-33 Profiling detects inefficiency at every scale
- **Status**: active
- **Statement**: Every module is measured at several scale points (one item, many items, a corpus,
  a long signal). Per-item time, memory, files and bytes must stay flat with scale, and profiles
  locate regressions.
- **Source**: owner, 2026-10-30: "profiling should detect inefficiencies at different scales".
- **Check**: the `benchmarks/scale/` ladder with per-item budgets as data; Scalene profiles on
  regression.

### P-34 Brittleness is always checked
- **Status**: active
- **Statement**: No change may rely on timing, ordering, one dataset, one host, one model's quirk, a
  magic number or a string match. Boundaries are exercised with bad input and injected faults, and
  failures are typed and isolated.
- **Source**: owner, 2026-10-30: "brittleness has to be checked for always".
- **Check**: property tests and fault injection in the logic tier; the PR checklist; review rejects
  special-casing.

### P-35 Agents break the system to raise standards, not to add brittleness
- **Status**: active
- **Statement**: Agents are used adversarially to break the system. Each break is resolved by an
  architectural change (an invariant, a type, a contract test, a lint rule, or a principle), never
  by a patch for the found case.
- **Source**: owner, 2026-10-30: "agents should be used to break system to set architectural
  standards not more brittleness".
- **Check**: each phase ends with an adversarial round. Findings are recorded with their
  architectural resolution, and case-only fixes are rejected.

---

## Inherited principles

Existing senselab rules that 2.x keeps. The full text is in `CLAUDE.md`; amending one also means
amending its origin.

### P-16 Load by commit, never by ref
- **Status**: active. **Statement**: every load passes a resolved commit SHA. **Source**:
  `CLAUDE.md`. **Check**: the revision-pinning guard, or its 2.x successor, covers every loader.

### P-17 Thresholds are data with a derivation
- **Status**: active. **Statement**: decision constants live in versioned data profiles with a
  derivation, and are regenerated rather than hand-edited. **Source**: `CLAUDE.md`. **Check**: a lint
  flags numeric literals in `decide` tasks that have no profile key.

### P-18 Configuration, not flags
- **Status**: active. **Statement**: entry points take inputs and configuration, and the merged
  configuration hash is stamped into outputs. **Source**: `CLAUDE.md`. **Check**: entry points take
  inputs plus profiles (config, dataset profile, execution), with no per-knob flags.

### P-19 Cache invalidation is cheap
- **Status**: active. **Statement**: keys are content- and commit-aware; bump the schema rather than
  reason about survivors. **Source**: `CLAUDE.md`. **Check**: the logical key in ADR-002 (`design.md` §4.8).

### P-20 Measurements and decisions are separate layers
- **Status**: active. **Statement**: measuring nodes emit measurements; decisions are separate,
  declared nodes. **Source**: `audio_analysis/doc.md`. **Check**: the generated contract suite
  asserts that `measure` outputs carry no verdict fields.

### P-21 Rationale lives in specs
- **Status**: active. **Statement**: code says what and how; the reasons go to `specs/`.
  **Source**: `CLAUDE.md`. **Check**: the prose lint (`engineering.md`) and review.

---

## Resolved tensions

- **P-22, core on non-EOL Python vs old models needing old Python** (2026-10-30). Both hold by
  splitting the code that crosses the boundary into a worker runtime (stdlib + numpy) with no Python
  floor beyond what a lock needs, while full core follows EOL. Typed validation happens on the host.
- **P-25 isolated envs vs P-5 efficiency** (2026-10-30). In-process execution is kept as an
  optimisation only inside a locked senselab venv whose lock covers the backend, so it costs no
  reproducibility. Otherwise the backend runs in its own env behind a persistent worker.

## Change log

- 2026-10-30: register created. P-1..P-15 from the owner's requests in the API-refactor session;
  P-16..P-21 carried over from `CLAUDE.md` and the audio_analysis design.
- 2026-10-30: added P-22 (core in every environment), P-23 (licenses), P-24 (cross-modal sharing)
  from the owner's follow-up requests. Inherited principles keep ids P-16..P-21; new ids continue
  from P-22 because ids are never renumbered.
- 2026-10-30: amended P-22 (meaning change). Its Python range now follows the CPython EOL schedule
  instead of covering every version an env happens to use; this drops 3.10, which reaches EOL in
  2026-10 (PEP 619).
- 2026-10-30: amended P-22 again (meaning change). Envs may keep EOL Python; what crosses the
  boundary is a stdlib+numpy worker runtime, not full core. See *Resolved tensions*.
- 2026-10-30: added P-25 (locked isolated envs) and P-26 (signals generalise); added the
  *Process principles* section with P-27..P-29. Amended P-23 (meaning change): missing licenses
  raise at run time by default, and policies use named license sets.
- 2026-10-30: added P-30 (test tiers) and P-31..P-35 (engineering standards: verbosity, compact
  code, profiling at scale, brittleness, adversarial agents). Their per-commit rules are in
  `engineering.md`.
- 2026-10-30: amended P-32 (meaning change). fastio is the exemplar of algorithmic efficiency, not
  of brevity; the remaining standards come from current best practice.
- 2026-10-30: consistency revision after the five reviews (`review-register.md`).
  - Shortened the change rules.
  - Condensed P-16..P-21 to one-line statements with links.
  - Fixed Check fields: P-7 (catalogue tables from the scale ladder), P-17 (decide-literal lint),
    P-18 (inputs plus profiles), P-19 (canonical key), P-20 (contract check), P-24 (adoption in
    Phase 4), P-25 (numerics identity; env-tier import audit), P-21 (prose lint).
  - Amended P-30 to name the tiers in `testing.md`. This is a wording alignment, not a change of
    meaning.
- 2026-10-30: fixed the P-23 Check field. The CI gate equals the run-time default, including
  transitive packages; the previous Check contradicted the statement.
- 2026-10-30: added P-36 (flywheel as a roadmap goal) and `roadmap.md`. Phase 4 no longer ports
  triage as a pipeline: its measurements become tasks, and its decision layer waits for the
  flywheel.
- 2026-10-30: added P-37 (one implementation per behaviour, consolidated without brittleness), with
  its standard in `engineering.md` and the inventory in design §6.1.
- 2026-10-30: added P-38 (exactness by portable hashes, reuse by measured equivalence) after the
  principal-architect review. The ADRs in `adr/` resolve R-3, R-8, R-9, R-12 and R-38..R-54.
  P-19's check now refers to ADR-002's logical key.
