# senselab 2.x API — principles register

This register is the source of truth for what the senselab 2.x API must satisfy. A design, a plan
or a review cites principles by id (`P-3`). When a design and this file disagree, this file wins until
it is amended here.

## How to change this register

Principles are edited, not accumulated. The register stays short enough to hold in mind.

- **Add**: append with the next unused id. Ids are never reused or renumbered. Give the statement,
  its source (the owner's words, quoted, or the file it is inherited from) and how a reviewer checks
  compliance.
- **Amend**: edit the statement in place and add a change-log line saying what changed and why. If
  the meaning changes, not just the wording, say so in the log.
- **Retire**: set `Status: retired`, keep the entry, and give the reason and what supersedes it.
  A retired principle is not cited by new work.
- **Merge or split**: retire the old ids and point them at the new ones.
- **Conflicts**: when two principles pull against each other in a concrete decision, record the
  resolution under *Resolved tensions* rather than weakening either statement silently.
- Only the owner adds or amends an `owner` principle. An agent may propose a change in a PR, but
  must not apply it unasked.
- Every change gets a line in the change log at the bottom.

Fields: **Statement** is normative. **Source** is either `owner` (with a quote and date) or
`inherited` (with a path). **Check** is how a reviewer, human or agent, decides whether a design
complies.

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
- **Check**: a task declares its resource needs and the knobs it can tune; the values come from an
  execution profile.

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
- **Statement**: senselab core installs and imports in the host and inside every isolated
  environment, across every Python version those environments use (3.10 and up today) and across
  their numpy and pydantic ranges. Host and workers use the same core version, verified at
  handshake.
- **Source**: owner, 2026-10-30: "can senselab core be used across subprocess venvs so that install
  works across python versions".
- **Check**: core CI runs on the full Python range at the minimum and maximum numpy. Every env build
  imports core and round-trips each data type over the wire.

### P-23 Licenses are explicit and machine-readable
- **Status**: active
- **Statement**: Every model, code package, dataset and therefore every task call and every plan has
  a machine-readable license record (SPDX where possible), with derived flags (commercial use,
  copyleft, gated, unknown). A run can carry a license policy that the planner enforces before
  execution. Licenses are recorded in provenance.
- **Source**: owner, 2026-10-30: "one of the challenges of senselab is knowing what license a given
  task uses (GPL/NC/apache, etc) which needs to be very clear and machine accessible".
- **Check**: `sl.licenses(plan)` answers without running anything; CI fails on a catalogue model or
  locked package without a license record.

### P-24 Modalities share types; environments share by dependency set
- **Status**: active
- **Statement**: Audio, video and text share one timeline, annotation, embedding, record and
  identity model, so cross-modal pipelines chain without glue. Environments are not owned by a
  modality or a task: backends declare requirements, and backends whose combined requirements
  resolve may share an environment and a worker.
- **Source**: owner, 2026-10-30: "with audio/video/text processing enabled consider what
  unions/intersections are across such processing (and knowing subprocess envs exist)".
- **Check**: a video → audio → text pipeline needs no conversion code between nodes. The env
  grouping is computed and recorded as data, not hand-assigned per module.

---

## Inherited principles

These are existing senselab rules that the 2.x design must keep. They are recorded here so the API
design does not lose them. Amending one also means amending its origin.

### P-16 Load by commit, never by ref
- **Status**: active
- **Statement**: Resolving a ref to a SHA binds nothing by itself. Every load passes the resolved
  SHA.
- **Source**: inherited, `CLAUDE.md` ("A model load must pass a commit SHA, never a ref").
- **Check**: `src/tests/utils/revision_pinning_guard_test.py`, or its 2.x successor, covers every
  loader.

### P-17 Thresholds are data with a derivation
- **Status**: active
- **Statement**: Thresholds and fitted constants live in versioned data profiles with a written
  derivation, never as code literals. Profiles are regenerated from measurements, not hand-edited.
- **Source**: inherited, `CLAUDE.md` ("Thresholds belong in `data/`").
- **Check**: no numeric decision constant appears in task or pipeline code without a profile key.

### P-18 Configuration, not flags
- **Status**: active
- **Statement**: Entry points take inputs and a configuration. Knobs are config keys with a
  derivation, deep-merged over packaged defaults, and the merged config hash is stamped into every
  output. Entry points do not grow per-knob flags.
- **Source**: inherited, `CLAUDE.md` ("The CLI is two arguments").
- **Check**: CLI entry points take an input, an output and an optional config only.

### P-19 Cache invalidation is cheap
- **Status**: active
- **Statement**: Cache keys are content- and commit-aware. When cached semantics change, bump the
  schema version rather than reasoning about which entries survive.
- **Source**: inherited, `CLAUDE.md` ("Cache invalidation is free", "Cache keys are commit-aware").
- **Check**: the cache key includes the schema version, task version, model identity (P-4),
  parameters and input identity.

### P-20 Measurements and decisions are separate layers
- **Status**: active
- **Statement**: Nodes that measure emit measurements. Decisions over measurements are separate,
  declared nodes.
- **Source**: inherited, `src/senselab/audio/workflows/audio_analysis/doc.md` (L1 measures, L2
  decides).
- **Check**: a measurement node's output schema carries no verdict fields.

### P-21 Rationale lives in specs
- **Status**: active
- **Statement**: Code and docstrings say what a thing is and how to call it. The reasons behind a
  choice live in `specs/`.
- **Source**: inherited, `CLAUDE.md` ("Rationale does not go in code").
- **Check**: review.

---

## Resolved tensions

None yet.

## Change log

- 2026-10-30: register created. P-1..P-15 from the owner's requests in the API-refactor session;
  P-16..P-21 carried over from `CLAUDE.md` and the audio_analysis design.
- 2026-10-30: added P-22 (core in every environment), P-23 (licenses), P-24 (cross-modal sharing)
  from the owner's follow-up requests. Inherited principles keep ids P-16..P-21; new ids continue
  from P-22 because ids are never renumbered.
