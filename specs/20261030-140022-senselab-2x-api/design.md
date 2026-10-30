# senselab 2.x API — design

Status: draft for owner review. Governing principles: [`principles.md`](principles.md), cited here as
`P-n`. Surveyed against `origin/alpha` at `19834fdd` (2026-10-30).

## 1. Intent

senselab 1.x grew a large amount of hard-won machinery:
- commit-pinned model loading;
- content-addressed caches;
- W3C PROV stores with a BEP028 export;
- 29 locked, CUDA-routed isolated environments with persistent workers;
- a vLLM server manager;
- a shared-filesystem work pool that drives Slurm;
- a pinned, sandboxed step DAG (`reproduce`);
- two large pipelines (`audio_analysis`, `triage`).

Each piece was built for the pipeline that needed it first. The result is that the machinery cannot
be reached through one API, cannot be described to an agent, and has drifted toward one dataset.

2.x keeps the machinery and gives it one shape. A user or an agent declares *what* to compute
(tasks, pipelines, a dataset, a configuration). A planner turns that, for a given site and hardware,
into an inspectable execution plan. An executor runs the plan, and the framework records
provenance and caches results the same way for every task.

## 2. Current state (what the design must absorb)

Surveyed facts. They drive the design, so they are stated plainly.

**Public API**
- There are 37 task `api.py` modules with no common signature:
  - backend selection works four ways: wrapper-class `isinstance` checks plus id prefixes, bare
    strings, backend objects, or a flag per toolkit;
  - knobs are passed five ways: unvalidated `**kwargs`, a validated `parameters` mapping
    (`utils/backend_parameters.py`, used by 4 tasks), named args, a `config` mapping, or
    keyword-only args with no defaults;
  - return shapes vary by task and sometimes by argument.
- Nothing is exported at the top level. Imports are deep, every call takes a list, and the README
  quick start fails because `transcribe_audios` requires `model`.
- `model_registry.yaml` is a curated catalogue that runtime code barely reads.
- The agent interface writes notebook code and looks symbols up on DeepWiki. There is no task
  schema and no senselab MCP surface.

**Weight**
- `import senselab` is cheap, but the install is not: torch, transformers, speechbrain, pyannote,
  datasets and accelerate are all required dependencies.
- `Audio` imports torch (`audio/data_structures/audio.py:10`), and
  `utils/data_structures/__init__.py` eagerly imports every model class.
- `senselab/__init__.py` has global side effects: `nest_asyncio`, and
  `set_start_method("spawn")`.
- CI only ever installs `--all-extras`.

**Execution**
- 29 isolated environments, each declared by module constants plus `ensure_venv(...)`, discovered
  by an AST scan (`venv_lock.discover_backends`). Each has a committed hashed lock and CUDA wheel
  routing.
- Three-to-four worker protocols coexist:
  - `call_in_venv` (unused);
  - about 20 one-shot `subprocess.run` scripts;
  - `serve_in_venv` persistent workers;
  - four bespoke Popen workers.
- There are two vLLM server implementations.
- Every in-process backend keeps its own class-level model dict. There is no shared model cache,
  eviction or memory budget.
- Batch size and dtype are chosen ad hoc per backend.
- "Where it runs" is encoded three separate ways: a `device` argument, venv identity, and
  `ResourceClass`.
- `work_pool` items are argv only, so a task call cannot be shipped to the cluster without a CLI.

**Provenance and caching**
- Three subsystems, each with its own key scheme:
  - the `cached_inference` stage cache plus a result cache;
  - `ProvStore` with its BEP028 export;
  - `reproduce` step keys.
- Model identity is mostly solid. `resolve_revision` resolves once and records the run manifest.
  Hub subclasses get commits. Non-hub weights are checked against literal sha256 constants, and
  models that cannot be pinned carry an `unresolved_reason`. Host and hardware recording is rich
  only in `reproduce`.

**Pipelines**
- **triage** names its nodes (`GRAPH_ORDER`) but runs them from a hand-written runner over a
  shared `ProvStore`, with no declared inputs.
- **audio_analysis** declares `StageContract{reads, writes}` and derives edges from them, but uses
  them only for checking; execution is an `if plan.x:` sequence.
- **reproduce** is the only real declared DAG: `needs`, resources, shards, resumability, engine.
  Its steps are whole processes.
- Corpus iteration in triage is a JSONL manifest plus `--slice-index/--slice-count` plus 16
  `extend_*.py` drivers.
- `SenselabDataset` is unused by any pipeline, and `create_bids_dataset` is a stub.

**Dataset coupling** (violates P-12)
- BIDS entity regexes are duplicated in six places.
- B2AI sidecar field names are hard-coded (`b2ai_hints.py`).
- About 28 task-family literals, plus per-task expectations keyed by B2AI task names
  (`nodes/branches.py:830-1130`).
- B2AI cohort and lexicon policy sits inside the packaged `default.yaml`.
- Thresholds were fitted on one corpus without naming the population.
- ORCD paths appear in scripts.
- `audio_analysis`, `explore_conversation` and `health_measurements` are clean.

## 3. Goals, non-goals, anti-goals

**Goals:**
- the principles P-1..P-21;
- a small vocabulary of concepts (section 4) that covers everything in section 2;
- every 1.x capability reachable through it.

**Non-goals for the first 2.0 release:**
- a GUI;
- a general-purpose workflow engine for non-senselab work;
- new models or new analyses (porting only);
- multi-node model parallelism (one model instance fits on one device);
- remote/cloud executors beyond Slurm.

**Anti-goals.** Any of these is a failure even if everything else works:
- core that installs torch;
- a task whose result cannot be traced to model bytes and parameters;
- a pipeline that needs code edits to run on a second dataset;
- two ways to declare the same thing (two config loaders, two worker protocols, two caches);
- a schema for agents that is maintained by hand next to the code;
- a planner whose decisions cannot be printed before running.

## 4. Concepts

Nine nouns. Everything public is one of these.

| Concept | What it is | Layer |
|---|---|---|
| **Data** | `Audio`, `Video`, `Text`, `Transcript` (ScriptLine tree), `Image`; numpy-backed, lazy, content-identified | core |
| **Record** | one item of a dataset: data handles + entities (`participant`, `session`, `task`, `run`, …) + hints | core |
| **Dataset** | an iterable, sliceable, filterable collection of Records; BIDS layout or in-memory mapping | core |
| **Task** | a declared operation with typed inputs, outputs, a parameter schema, a version and a kind (`measure` / `decide` / `transform`) | core |
| **Model** | a reference resolved to a content identity | core |
| **Backend** | an implementation of a Task for some Models on one inference **stack**, with an environment spec and tunables | core declares, backend code implements |
| **Pipeline** | a graph of nodes (Task calls or Pipelines) with named ports; itself a Task | core |
| **Profile** | versioned, hashed configuration in three kinds: run config, dataset profile, execution profile | core |
| **Plan / Executor** | a Plan binds every node to backend, stack, env, device, batch, shard; an Executor runs it | core (protocol) + executors |

Provenance and the cache are not concepts a user handles. They are produced by the framework
around every Task execution (P-3, P-19).

### 4.1 Data (P-13)

- `Audio` holds a numpy `float32` array shaped `(channels, samples)` plus `sampling_rate`. It loads
  lazily from a path, a byte range, or a stream through the existing `portable_audio_io`.
- Tensors appear only inside backends; conversion is the backend's job.
- `Audio.identity` is the existing `audio_signature`, taken after the canonical decode, so a
  container change does not change identity.
- `Video` follows the same pattern, and frames can be shared-memory backed for worker transfer.
- Results are pydantic models with a stable JSON form, so an agent can read any result.
- Arrays serialise as `.npy` or safetensors beside the JSON, as the result cache already does.
- Data objects are immutable: a transform returns a new object whose identity records its parent
  (`wasDerivedFrom`).

### 4.2 Task (P-1, P-20)

A Task is declared once, in core, without importing any backend:

```python
@task(name="speech_to_text", version=3, kind="measure")
class SpeechToText:
    class Params(TaskParams):
        language: Language | None = None
        timestamps: Literal["word", "segment", "none"] = "word"
    inputs  = {"audio": Audio}
    outputs = {"transcript": Transcript}
```

The primary form is a **configured, immutable task object**: hyperparameters at construction, data
at call time (section 4.13):

```python
asr = sl.audio.SpeechToText(model="openai/whisper-large-v3-turbo", language="en")
t   = asr(sl.Audio("a.wav"))           # one item
ts  = asr.map(dataset)                 # many, through the planner
asr.params                             # pydantic Params, = JSON Schema instance
asr2 = asr.with_params(language="es")  # new object; objects never mutate
```

The object is what a pipeline node holds, what the cache key and provenance record, and what an
agent's tool call serialises to.

From that one declaration senselab also generates:

- **a typed function for humans:**
  `sl.audio.speech_to_text(audio, model="openai/whisper-large-v3-turbo", language="en")`. It takes
  **one item and returns one result**, and is exactly `SpeechToText(**params)(audio)`; it is sugar,
  not a second implementation.
- **a batched form:** `sl.audio.speech_to_text.map(dataset_or_items, ...)`, which goes through the
  planner so batching, model reuse and caching happen in one place (P-5).
- **a JSON Schema** of inputs, params, outputs and the models and backends that can serve it, for
  `sl.describe("speech_to_text")`, the CLI and the MCP server.

The same task can be called four equivalent ways:

```python
import senselab as sl

# human, one file
t = sl.audio.speech_to_text(sl.Audio("a.wav"), model="openai/whisper-large-v3-turbo")

# human, many files: same call shape, batched by the planner
ts = sl.audio.speech_to_text.map(sl.Dataset.bids("/data/ds"), model="openai/whisper-large-v3-turbo")

# agent, by name, from a schema it fetched
sl.run("speech_to_text", inputs={"audio": "a.wav"}, model="openai/whisper-large-v3-turbo",
       params={"language": "en"})

# discovery
sl.tasks()                         # names, kinds, one-line summaries
sl.describe("speech_to_text")      # JSON Schema + models + backends + stacks
sl.models(task="speech_to_text")   # runtime catalogue (was model_registry.yaml)
```

Rules that remove the 1.x inconsistencies:
- One parameter-passing style: a pydantic `Params` model per task, with extra keys forbidden and a
  close-match suggestion on a typo. This generalises `backend_parameters.py`.
- Backend-specific knobs go in `backend_params`, validated against the chosen backend's own
  schema, never through a silent `**kwargs`.
- `model` is always a string reference or a `Model`, never a wrapper class chosen by the caller.
- Every task names a default model in the catalogue, so `model` can be omitted. Omitting it is
  recorded in provenance as "catalogue default".
- `device` is not a task argument (P-2).
- Measurement tasks carry no verdict fields (P-20). Decision tasks consume measurements and a
  config profile with derivations (P-17).
- The DSP primitives that today take 6–10 keyword-only arguments with no defaults become tasks
  whose `Params` default from a packaged profile key. The profile is the derivation (P-17).

### 4.3 Model and identity (P-4, P-16)

- `Model("openai/whisper-large-v3-turbo@main")` is a reference.
- `model.resolve()` returns `ResolvedModel{source, id, requested_ref, identity}`. The identity is
  the commit SHA for Hub models, or the sha256 of the weight set for URL, GitHub release, TF-Hub
  and local-file models.
- Loading only ever takes a `ResolvedModel`. This puts the existing
  `resolve_revision`-then-load-by-SHA discipline into the type system instead of leaving it to an
  AST guard.
- Sources that cannot be pinned (Ollama, torchaudio-bundled) resolve to an identity with
  `unresolved_reason`. Their results are flagged `unpinned` in provenance and the cache.
- The catalogue (today's `model_registry.yaml`) becomes runtime data. Per entry it holds the
  tasks served, the backends and stacks that can serve it, a default ref, license, size, and the
  measured resource needs per hardware class. That last field is data with a derivation (P-17),
  and the planner reads it (P-7).

### 4.4 Backend, stack and environment (P-6, P-7, P-8)

A Backend is registered against a Task:

```python
@backend(task="speech_to_text", stack="vllm", models=["openai/whisper-*"],
         env=Env.isolated("vllm-server"), devices=["cuda"],
         tunables={"max_num_seqs": Range(1, 256), "dtype": ["bf16", "fp16"]})
class WhisperVllm: ...
```

- **Stack** is one of `transformers`, `vllm`, `vllm-omni`, `nemo`, `ct2`, `mlx`, `onnx`,
  `speechbrain`, `pyannote`, `dsp`. A Task can have several backends for one model, one per stack.
  The stack goes into the result identity whenever outputs can differ, as the vLLM reviewer
  already does with engine identity.
- **Env** is one of three kinds:
  - `Env.host()`: in-process, using whatever the host has installed;
  - `Env.isolated(name)`: a locked venv. This is today's `ensure_venv` plus lock plus CUDA routing,
    declared as data instead of module constants and an AST scan;
  - `Env.server(name)`: a long-lived engine such as vLLM or vLLM-omni, managed by the one
    `VllmServer` implementation.
- **Torch backends** declare both `host` (the notebook fast path, needs `senselab[torch]`) and
  `isolated`. The planner picks one; the call site never knows (P-6).
  - **`host` means the senselab venv, which is itself locked (P-25).** senselab is always
    installed into a venv built from its lock, for example by `senselab env create` or
    `uv sync --locked`. A notebook kernel is that venv.
  - The planner runs a backend in-process only when the senselab venv's lock covers that
    backend's requirements, verified by digest. Otherwise it runs in its own isolated env.
  - A system Python, an unlocked environment, or a package the lock does not list is never used.
    `env identity` in the cache key (section 4.8) is the senselab venv's lock sha for host runs.
  - The choice is shown in `plan.explain()`.
- **Native libraries come from wheels.** soundfile bundles libsndfile, PyAV bundles FFmpeg, and
  soxr is a wheel.
  - A dependency that still needs a system library declares it as a `SystemRequirement`. Today
    that is torchcodec's FFmpeg shared libraries.
  - A `SystemRequirement` is probed when the env starts, and its version is recorded in
    provenance. A missing one is an error naming what to install.
  - Each such requirement is tracked as debt, to be replaced by a wheel-bundled alternative.
- **One worker protocol:** `serve_in_venv` (persistent, JSON control line, arrays passed by
  file/shared memory, typed errors). The one-shot scripts, the bespoke workers and `call_in_venv`
  are removed. Errors keep their type across the boundary, because a registered exception table
  replaces the ValueError/TypeError/RuntimeError collapse.
- **One model cache per process,** owned by the runtime and not by each backend. It is keyed by
  `(backend, ResolvedModel.identity, device, dtype)`, has a memory budget per device and LRU
  eviction, and can tell the planner which models are already resident. Workers carry the same
  cache.
- **Tunables** are declared ranges. Their values come from the execution profile or from the
  catalogue's measured table for the hardware class. Qwen's "MPS ⇒ batch 1" becomes a measured
  table entry, not a branch in code.

### 4.5 Pipeline (P-9, P-10, P-20)

A pipeline is a graph whose nodes are Task calls or other Pipelines, connected by named ports:

```python
p = sl.Pipeline("conversation")
seg  = p.add(sl.audio.diarize, audio=p.input("audio"))
asr  = p.add(sl.audio.speech_to_text, audio=seg.out.segments, over="segment")
spk  = p.add(speaker_identity, audio=p.input("audio"), turns=seg.out.turns)   # a nested Pipeline
p.output(transcript=asr.out.transcript, speakers=spk.out.speakers)
```

- **Nesting.** A `Pipeline` satisfies the Task protocol (typed ports, version, params schema), so
  it is used as a node unchanged. Rendering shows one level at a time and expands on request.
- **Fan-out and grouping.** `over=` names the level a node runs at: `item` (default), `segment`,
  `speaker`, or a dataset grouping key such as `session`, `participant` or `corpus`.
  - Grouping keys come from the dataset (section 4.6), never from parsing paths in a node.
  - This replaces triage's per-file stubs plus separate `extend_session_floor` and `extend_cohort`
    drivers.
- **Routing.** `p.route(selector, branches={...})` selects a sub-graph at run time from a decision
  node's output. Every branch is declared, so the graph is still fully listable. This is triage's
  `routing`.
- **Iteration.** `p.loop(body, until=decision, max_rounds=n)` runs a declared body for rounds,
  which is `audio_analysis`'s round DAG and `adaptive_loop`. Unrolled rounds get distinct node ids,
  as `unrolled_contracts` already does.
- **Absence is typed.** A node can emit `Absent(reason)`. Downstream nodes declare whether they
  accept absence or are skipped, which generalises triage's gate skipping. A node failure is not
  an absence.
- **Ports carry schemas.** `StageContract` and `StageIO` (D-17/D-18) become the port model: the
  schema travels with the artifact, and edges are checked when the graph is built, not by AST
  guards afterwards.
- **Serialisation.** Pipelines are written in Python and serialise losslessly to YAML/JSON (graph +
  task refs + params). An agent can read, edit and submit a pipeline as data, and a run records the
  exact graph it executed.

### 4.6 Dataset (P-11, P-12)

```python
ds = sl.Dataset.bids("/data/b2ai", profile="b2ai-voice@2026-09")   # layout on disk
ds = sl.Dataset.from_records([{"audio": "a.wav", "participant": "p1"}, ...])  # virtual mapping
ds = ds.filter(task="reading-passage").slice(i, n)                  # lazy, deterministic
```

- One `Dataset` protocol yields `Record`s. It provides `entities()` (the available grouping keys),
  deterministic slicing (it replaces `take_slice`), filtering, a stable `Record.id`, and a manifest
  form.
- Resume is not a dataset feature. It falls out of the content cache (section 4.8): a re-run skips
  records whose results exist.
- **Adapters.** A BIDS adapter parses entities once, through one implementation; this removes the
  six regex copies. An in-memory adapter wraps lists, dicts, pandas frames, or HF datasets.
- **Dataset profiles** hold everything that is true of a dataset rather than of an analysis:
  - the entity mapping;
  - the sidecar → hint mapping (`stimulus_text`, `speech_type`, …);
  - the **task protocol** (task id → family → expectations: pattern, token source, counts,
    durations);
  - lexicons and cue families;
  - cohort conditions;
  - fitted threshold profiles that name the population and release they were fitted on.

  Profiles are versioned and hashed, and the hash goes into provenance.
- **Moving B2AI out.** Everything listed under *Dataset coupling* in section 2 moves into a
  `b2ai-voice` dataset profile. Triage then becomes a generic screening pipeline that reads its
  task vocabulary from whichever profile it is given (P-12).

### 4.7 Profiles and the plan (P-2, P-5, P-7, P-10, P-18)

There are three profile kinds. Each is deep-merged over packaged defaults and hashed. Derivations
live in a sibling file, never inside the hashed mapping, which is what triage already learned.

| Profile | Holds | Example keys |
|---|---|---|
| **run config** | analysis parameters | model choices per task, thresholds (with derivations), aggregators |
| **dataset profile** | facts about a dataset | entities, hints, task protocol, fitted populations |
| **execution profile** | facts about a site | hardware classes, scheduler (none/slurm), partitions, env and cache roots, stack preferences, network policy, concurrency caps |

`sl.plan(pipeline, dataset, config=..., execution=...)` returns a **Plan** that can be inspected,
diffed and hashed. For every node and shard it states:
- the backend and stack;
- the env (host, isolated or server);
- device class and count;
- batch and other tunables;
- the resource class and partition;
- the expected cache hits.

`plan.explain()` prints why each choice was made. Choices come from rules over the hardware probe,
the catalogue's measured tables, and the execution profile, with no hidden fallbacks.

Two plans for the same pipeline, dataset and config, one for a laptop and one for a cluster,
differ only in execution fields (P-10). A **hardware probe** without torch (`nvidia-smi`, `nvcc`,
Apple-silicon detection, memory) feeds the planner. It extends `cuda_probe` and `reproduce/host.py`.

Device placement and data movement:
- **Placement.** GPU-bound and CPU-bound nodes are placed separately. The planner groups GPU nodes
  that share a resident model, and runs DSP and decision nodes on CPU workers.
- **Batching.** Batching happens across records. The executor collects a node's inputs across
  records up to the tunable batch size, which is what turns `.map` into efficient inference.
- **Transfer.** Large arrays move by file or shared memory, never through the JSON control line.

### 4.8 Execution, cache and provenance (P-3, P-5, P-19)

**Executors and the graph runtime** (from [`workflow-engines.md`](workflow-engines.md))

*Assessment.* 13 engines were evaluated critically against nine requirements (R1–R9 in
`workflow-engines.md`). No engine is adopted as the execution layer. The three hardest
requirements are:
- R3: mixed Python versions per node without containers;
- R4: a resident GPU model batched across records;
- R5: Slurm pilots on NFS with preemption.

No candidate is native on more than one of them. senselab already has measured answers to all
three: `venv_worker`, `vllm_server`, and `work_pool` (0 double-claims in the NFS race tests).

**What senselab owns**
- the result store;
- the env manager and worker runtime;
- the model cache with batching across records;
- the *placement* half of the planner (backend, env, device class, batch, partition, `explain()`).

**Executors**
- `InProcess`: the notebook and test default.
- `Pool`: `work_pool`, with items extended from argv to serialised plan fragments, so a task call
  can be shipped without a CLI. A local controller makes it the local multi-process executor too,
  so there is no separate `Local` executor.
- `Slurm`: the `work_pool` `Controller` plus `reproduce`'s held `afterok` chaining.
- A worker on any host runs the same `senselab execute <fragment>` entry point.

**The graph half**
- This covers nesting, typed ports, `over=` fan-out, `route`, bounded `loop` and the ready set.
  It is generic, and is what pydra 1.0 implements.
- *Decision:* use pydra for the graph layer only, behind senselab's `Executor` protocol and result
  store. Do not use pydra's workers, environments or pickle cache.
- This is conditional on the pydra changes that gate Route A in [`execution.md`](execution.md) §4–5
  (per-item dataflow, cheap submission, atomic cache with leases, pickle-free `ResultStore`, worker
  routing).
  - They land in a pydra 1.0 alpha that senselab pins; a non-alpha release is not required.
  - During development senselab depends on the pydra branch through `[tool.uv.sources]`, with the
    commit pinned in `uv.lock`.
  - The owner leads pydra and welcomes the PRs.
- How pools, GPU-only tasks, services and substrates (Slurm, local, other engines) mix, and the
  pydra/senselab boundary, are in [`execution.md`](execution.md).
  1. per-item dataflow, with keep-going on failure;
  2. a cheap submission path;
  3. atomic rename into the cache, with leases and stale-lock recovery;
  4. a pluggable, pickle-free result format.
- **Fallback:** if those PRs have not landed when Phase 4 starts, write the graph layer in
  senselab, estimated at 1–2k lines. Reuse `reproduce/dag.py`'s toposort and `unrolled_contracts`
  for loops. Do not write a third graph runner beyond that: triage's runner and `reproduce/dag.py`
  fold into it.
- Pydra PRs 5–9 (env identity and venv tasks, resources, Slurm fixes, provenance hooks, plan
  printing) are worth upstreaming, but senselab does not depend on them.

**Verified pydra findings** (rechecked 2026-10-30)
- PyPI's stable release is still 0.25; 1.0 is at `1.0a12`.
- In an 8-item fan-out with one failure, none of the 7 healthy items runs its downstream node.
  This was re-run and confirmed.
- The Slurm worker crashes when `--error` is passed: `workers/slurm.py:118` calls `.replace` on
  `None`.
- Issues #778 (stray locks), #826 (environment not in the checksum) and #840 (Python tasks in
  environments) are open.
- The per-item overhead (about 4–5 ms cold, 2–3 ms cached; about 34 KB of pickles per item) was
  measured on a loaded laptop and is an upper bound.

**What carries over from the Nextflow prototype:** changing one model's revision recomputed 14 of
18 tasks, because keys covered whole files. Keys must cover what each node reads, which the
result-store key below does.

**One content-addressed result store**
- It replaces the stage cache, the result cache and the separate key schemes. *Design call:* the key
  is `{schema, task, task_version, backend, stack, env identity, identity tunables, model identity,
  params, input identities, dataset-profile hash if read}`.
  - **env identity** is the env lock sha for an isolated backend, or the digest of the relevant
    installed packages for a host backend. A host run and an isolated run with different torch
    builds are therefore different results.
  - **identity tunables** are the tunables a backend declares as output-affecting. The default is
    `dtype`; vLLM adds concurrency. Batch size, device index and partition are recorded in
    provenance but are not in the key.
  - **device class** (`cpu`/`cuda`/`mps`) is in the key only when the backend declares it
    output-affecting.
  - This favours correctness over cache sharing. A laptop run and a cluster run share hits only
    when env, stack and identity tunables match. Adopting a computed env grouping changes env
    identity and so recomputes, which is correct (section 4.10).
- The layout is `schema-<N>/<k[:2]>/<k>/`, written by atomic rename, first writer wins (the result
  cache design). Bumping the schema opens a new directory.
- Failures are not cached.
- The senselab version is recorded but is not part of the key. This is the lesson of cache bump 24.

**One provenance store**
- `ProvStore`, append-only and with content-derived ids, is written by the executor around every
  node: activity, used, generated, model agent, environment (including the env lock sha), host and
  hardware record. The richer `reproduce/host.py` record is used everywhere.
- BEP028 export, unchanged, is how a run is published.
- `reproduce` becomes a consumer of this store rather than a third scheme. Its pinned-commit,
  sandboxed, weights-stamped step stays as the way to run *external* analyses, expressed as a
  Backend with `Env.isolated` and a sandbox policy.

**Run directory**
- `run/plan.json` (the plan and its hash), `run/store.jsonl` (provenance), `results/` (links into
  the result store, or copies in confidential mode), `config/` (all three merged profiles and their
  hashes).

### 4.9 Surfaces for humans and agents (P-1, P-18)

- **Python:**
  - `import senselab as sl`, with lazy submodules `sl.audio`, `sl.video` and `sl.text`, generated
    from task declarations;
  - top-level `sl.run`, `sl.plan`, `sl.describe`, `sl.tasks`, `sl.models`, `sl.Dataset`,
    `sl.Pipeline`;
  - no import-time side effects (spawn start method and nest_asyncio are set by executors when
    they start).
- **CLI:**
  - `senselab tasks | describe <task> | models [--task]`;
  - `senselab plan <pipeline> <dataset> [--config f] [--execution f]`;
  - `senselab run …` (same arguments);
  - `senselab status <run>`, `senselab prov <run> [query]`, `senselab execute <fragment>` (worker).

  Pipelines and datasets are named or given as paths. There are no per-knob flags (P-18).
- **MCP server (`senselab mcp`):** the same operations as tools whose schemas are the generated
  JSON Schemas, plus `get_result`, `explain_plan` and `query_provenance`. It replaces DeepWiki
  lookups in the agentic interface. Long runs return a run handle and do not block.
- **Docs:** pdoc pages for each task are rendered from the same declaration (schema, default model,
  backends, stacks, measured resource table), so docs cannot drift from code.

### 4.10 Across modalities: what audio, video and text share and where they differ

Audio, video and text tasks are declared the same way, but the design has to state what they
share, so the shared parts exist once and cross-modal work composes.

#### What all modalities share (core, once)

- **One timeline.** Every time-based result uses one `Span{start, end, source}` on the source
  media's clock: ASR words, diarization turns, VAD regions, face tracks, pose sequences, scene
  descriptions, and PII spans in text derived from audio. Today `ScriptLine`, pose frames and
  triage spans each keep their own time convention.
- **One annotation shape.** An `Annotation{span, label|value, score, producer}` and
  `Track[T]` (a time-indexed series) cover classification over windows, f0/HNR tracks, facial
  affect per frame and sentiment per utterance. `Transcript` is a tree of annotations, as
  `ScriptLine.chunks` is today.
- **One embedding type.** `Embedding{vector, space}` with `space` = the resolved model identity, so
  vectors from different models are never compared by accident. This covers speaker, SSL, HeAR,
  text and face-identity embeddings.
- **One Record with several views.** A recording with video carries `video`, `audio` (the
  demuxed track, a lazy derived view with `wasDerivedFrom`) and, once produced, `transcript`. All
  views share the record's entities and timeline.
- **One identity, cache, provenance, Params, Plan and Executor path.** None of these is
  modality-specific.

#### Where the modalities meet (bridges)

Bridges are declared tasks whose input and output modalities differ:

| Bridge | Example tasks |
|---|---|
| video → audio | demux; audio-visual target speaker extraction (`video/tasks/target_speaker_extraction`) |
| audio → text | ASR, forced alignment (text + audio → aligned transcript) |
| text → audio | TTS, voice cloning |
| audio + text → decision | PII in speech, task-protocol checks (expected text vs heard words) |
| video + audio → joint | active-speaker / face–voice association, dyadic turn-taking, conversation analysis |
| any → text | audio and video understanding via VLM/ALM (`describe_audios`, scene description) |

Because the port types are the shared core types, a pipeline chains across modalities with no
glue: `video → demux → diarize → speech_to_text → text.sentiment`, all on one timeline. Grouping by
`over="speaker"` means the same thing whether the speaker came from diarization or face tracks,
once an association task has merged them.

#### What stays per modality

- Decoding and IO: audio via `portable_audio_io`; video frames via av/opencv, shared-memory backed.
- DSP primitives: audio band/envelope/phonation; image ops.
- Domain decision pipelines: triage.
- The packaged task catalogue sections: `sl.audio`, `sl.video`, `sl.text`.

`sl.multimodal` holds the bridge tasks.

#### Environments across modalities

An environment is not owned by a modality. Today there are 29 isolated environments, grouped by
their torch pin:

| Group | Envs | Modalities |
|---|---|---|
| torch 2.8 | coqui, diarizen, emotion2vec, moss, nemo-canary-qwen, nemo-diarization, pii-detection, pii-redaction-review, ppgs, qwen-asr, sparc | audio, text |
| torch 2.11 | clearvoice, clearvoice-speechscore, continuous-ser, crisperwhisper, driftse, maai, pyfeat, qwen-tts | audio, video (pyfeat) |
| torch 2.13 (vLLM) | vllm-server, pii-review-vllm | text, audio (ALM), video (VLM) |
| no torch | hear (TF), mediapipe-face, mlx-vlm, yamnet (TF) | audio, video |
| old one-off pins | brouhaha (2.2), child-adult (2.3), s3prl (2.4), unasdiff (2.6) | audio |

This argues for three things.

1. **Environments are declared by dependency set, not by task.** A backend declares
   *requirements*, and the env layer maps backends to environments. Two backends may share an
   environment when the union of their requirements resolves under one lock. This keeps the same
   `uv pip compile` / requirements-sha machinery.
   - **Grouping is computed.** A lock-time tool tries to merge backends into the fewest
     environments whose union resolves, and records the grouping as data. Merged groups shrink
     disk, install time and, most importantly, the number of resident workers (each about 0.5 GB
     before weights).
   - **Isolation is kept where the union does not resolve.** That covers the old one-off pins, TF
     vs torch, and NeMo's git pin; the plan records why each was kept separate.
   - **Merging changes identity.** A backend's result identity includes its environment lock sha.
     Merging changes the lock, so it changes identity, which is correct: dependency versions can
     change numerics.
2. **One worker can serve several tasks across modalities.** A persistent worker in a merged
   environment loads several backends and keeps their models in its own model cache (section
   4.4). The planner co-locates nodes that share an environment and device, so a `video → audio →
   text` chain on one GPU does not spin up three workers when one environment covers all three.
3. **Engine servers are the strongest point of shared use.** One vLLM (or vLLM-omni) server
   serving a multimodal model can answer the ALM, VLM and text-LLM tasks of a pipeline. The
   planner treats a server as a shared resource:
   - the servers-per-card logic that already exists;
   - one server per (model identity, engine config) per device;
   - nodes routed to it by declared stack.
   vLLM-omni enters as a stack whose models take audio, image and video in one request. A
   multimodal model served once replaces separate ASR, captioning and text calls only where the
   run config chooses it, and that choice is recorded like any other backend choice (P-8).

What the host process holds is then the intersection of all of this: core plus the numpy DSP. With
`senselab[torch]` it also holds whichever torch backends the execution profile allows in-process.
Everything else is the union, held in the fewest environments that resolve.

### 4.11 Core and the worker runtime across environments (P-22)

Two installable pieces, with different Python policies:

| Piece | Where it runs | Python | Dependencies |
|---|---|---|---|
| **senselab-core** | the host | non-EOL CPython only: `>=3.11` at 2.0, raised as versions reach EOL | numpy, pydantic v2, … (section 5) |
| **senselab-worker** (worker runtime) | inside every isolated env | whatever the env's lock needs: today 3.10–3.13, including EOL versions kept for old models | stdlib + numpy `>=1.23` only; syntax at the oldest Python any lock uses |

Today workers get copies of single modules, such as `stage_portable_audio_io` staging
`portable_audio_io.py` onto `sys.path`, or reimplement them in script strings. The worker runtime
replaces both.

**What the worker runtime holds**
- the wire codec (arrays as `.npy`/safetensors or shared memory, JSON control line);
- the `serve_in_venv` loop and handshake;
- plain-dict encoders for the core data types;
- weight hashing, so hashes are computed where the weights are loaded;
- the per-worker model cache;
- the typed error table;
- provenance records emitted as plain JSON.

**What stays in core:** pydantic models, validation, the planner, executors, catalogue and the
provenance store. Everything arriving from a worker is validated into core types on the host
side, so a worker never needs pydantic.

**Constraints from the current locks**
- 2 envs are on Python 3.10, which reaches EOL in 2026-10 (child-adult-diarization: torch 2.3.0;
  unasdiff: torch 2.6.0, numpy 1.23.5). They stay on 3.10: the worker runtime supports them and
  core does not need to.
- Six envs pin numpy 1.x, so the worker runtime uses no numpy-2-only API.

**Rules**
- **Same version on both sides.** The env installer adds the worker runtime at the host core's
  exact version. The handshake checks that version and the wire-protocol version, and refuses on a
  mismatch rather than mis-decoding. Each env lock's requirements-sha includes the worker-runtime
  version, so a change rebuilds envs.
- **Small and slow-moving.** The worker runtime has its own floor test: it must import on the
  oldest Python and numpy in any lock. Raising that floor is allowed only when no lock needs the old
  version.
- **CI:**
  - core tests on every non-EOL Python (3.11–3.14 today) at minimum and maximum numpy;
  - worker-runtime tests on every Python any lock uses;
  - one test per env that imports the worker runtime inside the built env and round-trips each data
    type over the wire.

### 4.12 Licenses are explicit and machine-readable (P-23)

What a task's result may be used for depends on the licenses of everything that produced it.
Several are restrictive:
- non-commercial model weights (the registry already lists CC-BY-NC-SA-4.0 and CC BY-NC 4.0);
- custom model licenses (FunASR);
- copyleft libraries (praat-parselmouth is GPL-3.0);
- toolkits with custom research licenses (openSMILE);
- gated models whose terms must be accepted.

Today 17 of 50 registry entries carry a license, as free text. Package licenses in env locks are
not recorded at all.

**The license record**
```python
License(
    expression="CC-BY-NC-4.0",          # SPDX expression; LicenseRef-<name> for custom licenses
    commercial_use="no",                # yes | no | unknown
    copyleft="none",                    # none | weak | strong | network
    attribution=True,
    gated=False,                        # terms must be accepted before download
    source="https://huggingface.co/<id>/blob/<sha>/README.md",
    verified={"date": "2026-10-30", "how": "hf-card"|"manual"|"pep639"},
)
```

**Where licenses attach**
- **Model weights**, in the catalogue. Read from the hub card at the *resolved commit*, so a
  license change upstream is caught by the same commit pinning.
- **Code packages**, harvested at lock time for every package in every env lock and in the host
  extras. Sources are PEP 639 `License-Expression`, then the `License` field, then trove
  classifiers; unknowns are flagged for manual entry. Stored next to the lock as data.
- **Datasets**, in the dataset profile.
- **senselab itself:** Apache-2.0.

**Machine access**
- `sl.describe(task)` lists, per backend and model, the licenses of the model and of the backend's
  direct code dependencies.
- `sl.licenses(plan)` returns the union across a whole plan, with the derived flags (any
  non-commercial, any strong copyleft, any gated, any unknown) and which node contributes each.
- The **run config** can carry a `license_policy`, for example `{commercial_use: required,
  copyleft: [none, weak], unknown: deny}`.
  - The planner then picks only compliant backends and models, or fails before anything runs,
    naming the offending node and license.
  - `plan.explain()` shows the license reasoning next to the hardware reasoning.
- **Provenance** records the license of every model and software agent. The BEP028 export carries
  them, so a published derivative states what it was made with.

**License sets and run-time policy**
- A **license set** is a named, versioned list of SPDX expressions and flag conditions, shipped as
  data. Users can define their own and combine them by union or intersection. Packaged sets:
  - `permissive`: Apache-2.0, MIT, BSD-*, …;
  - `weak-copyleft-ok`: adds LGPL, MPL;
  - `copyleft-ok`: adds GPL, AGPL;
  - `non-commercial-ok`: adds CC-BY-NC-*;
  - `gated-ok`.
- A `license_policy` names the allowed set or sets and has `on_missing: raise | warn`, **default
  `raise`**.
- The planner checks every node against the policy before anything runs.
  - A missing or unknown license with `raise` stops planning with an error naming the node, the
    model or package, and how to supply a record.
  - With `warn`, the run proceeds, and the relaxation and the affected nodes are recorded in
    provenance.
- The default run config uses the broadest packaged set together with `on_missing: raise`. A
  research run is therefore never blocked by license *type*, only by an unknown license.

**Enforcement**
- A CI test fails when a catalogue model or a backend's direct dependency has no license record, or
  has `unknown` without a dated reason.
  - A harvested record counts.
  - Transitive packages are harvested and reported, but do not fail CI (section 7).
- senselab reports licenses and enforces the policy the user sets. It does not give legal
  interpretation.

### 4.13 Relation to the scikit-learn API (design call, owner asked)

senselab borrows sklearn's conventions where they fit and declines its data and execution model.

**Adopted**
- **Hyperparameters at construction, data at call time.** The configured task object of
  section 4.2. sklearn's `get_params` becomes the typed `Params` model, which is also the JSON
  Schema (P-1).
- **One uniform interface** for every task. This replaces the 37 inconsistent 1.x signatures.
- **`fit` for tasks that learn state.**
  - Covers threshold profiles from measured verdicts (P-17), cohort fences, speaker enrollment,
    dimensionality reduction and classifiers on embeddings.
  - These declare `kind="fit"`.
  - `est.fit(dataset)` returns a **new** fitted object whose state is a content-hashed profile
    artifact with provenance (fitted on which records, which config).
  - Fitting never mutates `self`; `set_params`-style mutation is replaced by `.with_params()`.
- **Interop through `sl.sklearn`.**
  - `SenselabTransformer(task)` wraps any task with an array-valued output (embeddings, features)
    as a sklearn transformer over a sequence of items, for featurisation inside sklearn pipelines.
  - `sl.sklearn.to_frame(results)` returns arrays or frames with record entities as the index.
  - It runs in-process and gives up the planner's batching across envs. It is the interop path,
    not the main one.

**Not adopted**

| sklearn | Why not |
|---|---|
| fit/transform/predict as the core verbs | most tasks are pretrained inference that change the data type (audio → transcript); `__call__` and `.map` are the verbs |
| `X` as an `(n_samples, n_features)` array | records are heterogeneous, variable length, multi-modal, with entities and a timeline |
| linear `Pipeline` / `ColumnTransformer` | senselab needs DAGs with fan-out, routing, loops and cross-modal ports (section 4.5) |
| mutable estimators | conflicts with content identity, caching and provenance (P-3, P-19) |
| eager in-process execution | conflicts with the plan/executor split and isolated envs (P-2, P-6) |

### 4.14 Signals beyond audio (P-26)

The use cases, models, licenses and standards behind this section are in
[`use-cases.md`](use-cases.md): 12 use cases, models per modality with licenses checked
2026-10-30, BIDS physio/motion/eyetrack/EEG, EDF/WFDB/XDF, and a three-tier taxonomy.

**Data model**
- **`Signal`** is the base type:
  - `data` `(channels, samples)`, with a dtype per channel (int ADC data with gain; float64 where
    precision matters);
  - `sampling_rate`;
  - `channels: [ChannelInfo{name, type, units, placement, reference}]`;
  - `start: TimeAnchor`;
  - `device: DeviceInfo`;
  - `identity`.

  `Audio` is a `Signal` with `type="audio"`. The `Track[T]` of section 4.10 (f0, heart rate, ENMO,
  facial-AU intensity) is a `Signal` too, so tracks feed the generic tasks.
- **`IrregularSignal`** carries per-sample timestamps: phone sensing, LSL streams with no nominal
  rate, RR series, eye trackers with drops.
  - Regularising it is an explicit task, never a side effect of reading.
  - Gaps are an `Intervals` value and are never zero-filled.
- **Units are first-class.** Readers apply ADC gain and physical ranges. Tasks declare the units
  they accept, and units enter identity.
- **Clock model.**
  - Each source has a `TimeAnchor{absolute_start, clock_id}`.
  - A synchronisation task produces a `TimeMap` (offset plus drift, or piecewise, with
    uncertainty) from each clock to a reference.
  - Spans stay on their source clock and are projected through a recorded `TimeMap` when sources
    are compared. This extends the single timeline of section 4.10 to multi-device recordings
    (BIDS `StartTime`, XDF clock offsets, audio/video drift, wearable UTC/DST).
- **Columnar `Events` and `Intervals`** (onset, duration, label, value, score arrays) for detector
  outputs at scale. They serialise as Parquet or as BIDS `events.tsv`/`physioevents`.
  `Annotation` remains the per-item view.
- **Time-range loading and time sharding.**
  - A `Signal` loads lazily by time range; derived data is chunked (Zarr or HDF5).
  - A task may declare itself *chunkable* (streaming filter, windowed features), so the planner
    shards along time as well as across records. A week of 100 Hz actigraphy is about 60M samples
    per channel.
- **Record views** are named by modality and device (`audio`, `video`, `ecg`, `ppg`, `acc@wrist`,
  `eyetrack`). The BIDS `recording-`/`tracksys-` entities map to view names.

**Task tiers** (the tables are in `use-cases.md` §4)
1. **Generic over `Signal`,** in `sl.signal`. Covers resample, filter, unit conversion, windowing,
   spectral and envelope features, activity and change-point detection, span algebra, signal QC,
   synchronisation, foundation-model embeddings, and imputation. The 1.x audio DSP that is really
   generic moves here: preprocessing, spans, envelope, band_profile, clipping, spectral continuity.
2. **Families with modality profiles:** one task per family, for example
   `sl.signal.detect_peaks(x, profile="ecg_rpeak")`.
   - Families: peak detection, inter-event intervals, periodicity, signal-quality index,
     decomposition, window classification, state sequencing, circadian measures, coupling.
   - Profiles live in `data/` with derivations (P-17), and the default is chosen by channel `type`.
   - Modality wrappers such as `sl.physio.ecg_rpeaks` are sugar over the family task.
3. **Modality-specific,** in `sl.audio`, `sl.video`, `sl.text`, `sl.physio` (ECG, PPG, EDA,
   respiration), `sl.motion` (actigraphy/IMU), `sl.eeg` (thin, delegating to MNE/YASA) and
   `sl.eyetrack`.

Inputs are declared as `Signal[type ∈ {...}, units]`, so `sl.describe` can answer "which tasks
accept my accelerometer channel".

**Bridges** gain physiological entries:
- ECG/PPG → respiration;
- accelerometer + PPG → motion-corrected pulse;
- video → respiration (rPPG);
- audio + actigraphy → speaking-while-moving.

**Measure vs decide holds** (P-20). QC tasks emit SQI tracks and gap intervals. Non-wear and
unusable-epoch verdicts are decision tasks with fitted profiles.

**Licenses** (P-23)
- Oxford's wearables tools (actipy, asleep, stepcount, ssl-wearables) are under an Academic Use
  Licence, so non-commercial. pyActigraphy, hrv-analysis and pyPPG are GPL.
- Permissive paths exist for most chains: NeuroKit2 (MIT), SKDH (MIT), GGIR (Apache-2.0), YASA
  (BSD-3-Clause), MOMENT (MIT). The permissive path for actigraphy sleep and steps is SKDH or
  GGIR.
- The license sets need `LicenseRef` entries for Oxford Academic Use, audEERING, OpenFace and
  OpenPose academic, and HAI-DEF.
- Items still unverified are listed in `use-cases.md` and stay out of the catalogue until they are
  checked.

**Non-Python runtimes** (P-25). GGIR (R) and actipy (Java) need environments that pip cannot
lock. *Design call:* `Env.isolated` gains a conda-forge kind, locked with pixi, for R, Java and
native-library stacks. It has the same lock-sha identity, probe and provenance as venvs. A Python
worker runtime inside that env drives the R or Java process, so the wire protocol is unchanged.

## 5. Packaging (P-13)

- **`senselab-worker`:** stdlib + numpy only, installed into every isolated env (section 4.11).
  Core depends on it, so the host and the workers run the same codec.
- **`senselab-core` dependencies:** numpy, pydantic, pyyaml, filelock, xxhash, soundfile,
  huggingface-hub (resolution only), packaging. pyarrow becomes an optional `tables` extra.
- **What core holds:**
  - data structures;
  - task and pipeline declarations;
  - the registry and catalogue;
  - identity, cache and provenance (`prov_store`, `prov_bep028`, `file_lock`);
  - profiles and the planner;
  - executors (`work_pool`, `fastio`, `vllm_server`, worker protocol, env locks and `uv_tools`);
  - torch-free DSP (resample via soxr, downmix, chunk, the band, envelope and phonation
    primitives that are numpy-only).
- **Backend code** lives in the same repository, but each backend module imports its heavy
  dependencies only inside its implementation. Declarations import nothing heavy, which is checked
  by a test that imports every declaration with only core installed.
- **Distributions from one repo.** A uv workspace with one `pyproject.toml` per member:
  `packages/senselab-worker`, `packages/senselab-core`, and the root `senselab` (backends, task
  catalogue, pipelines). Each is built and versioned on its own (hatch-vcs with per-package tag
  prefixes), and all three are released together at the same version. P-22 requires the worker
  runtime to match core's version exactly.
- **Install.** senselab is always installed into a venv from its lock (P-25), for example
  `uv sync --locked`, or `senselab env create` for users who do not use uv directly. Optional
  in-process groups (`torch`, `video`, `nlp`) are lock-covered extras of that venv. Isolated envs
  need no extra; their locks ship as package data.
- **CI** gains a core-only job (install core, import every public module, run the
  declaration/schema/planner/provenance tests) next to the existing full jobs.

How far 2.x can realistically get from torch:
- **core: torch-free;**
- **senselab venv: torch-free if every torch backend runs isolated.** That is a valid execution
  profile, at the cost of about 0.5 GB resident per worker and file or shared-memory transfer;
- **senselab overall: not torch-free.** Almost every model is a torch model.

## 6. Migration plan

Phases are ordered so that each one is usable on its own and validates the next. Exit criteria
cite principles.

**Phase 1: core foundations.**
- Build:
  - `senselab-core` with numpy `Audio`/`Video` and result models;
  - `Model`/`ResolvedModel`, with the hashing of non-hub weights generalised from the existing
    constants;
  - one config/profile loader (merging the audio_analysis and triage loaders);
  - the unified result store and the ProvStore writer;
  - the catalogue loaded at runtime;
  - the torch-free hardware probe;
  - no import side effects;
  - the shared timeline, annotation and embedding types (section 4.10);
  - core at the non-EOL Python floor (3.11), and the stdlib+numpy worker runtime at the oldest
    Python any lock uses (3.10 today) (section 4.11);
  - the `License` record, with licenses backfilled for all 50 catalogue entries and harvested from
    every env lock (section 4.12).
- Exit:
  - the core-only CI job is green on every non-EOL Python, 3.11–3.14 today (P-13, P-22);
  - the identity/cache/provenance tests pass with no torch installed (P-3, P-4, P-19);
  - the license-coverage test passes (P-23).

**Phase 2: tasks and backends on a vertical slice.**
- Build the Task and Backend declarations, Params validation, generated functions, `.map`, and
  generated JSON Schemas.
- Port five tasks that between them exercise every Env and stack kind. *Design call:* only
  backends that exist in 1.x, so every port has a 1.x reference.
  1. `speech_to_text`: HF Whisper on transformers (host), and CrisperWhisper on CTranslate2 and on
     transformers (isolated persistent worker). CrisperWhisper already has both stacks.
  2. `diarize`: pyannote (host or isolated).
  3. `speaker_embeddings`: speechbrain (host).
  4. text PII review on vLLM (server) and transformers. The engine switch already exists.
  5. one numpy DSP primitive (`phonation` or `envelope`).
- Collapse the worker protocols into `serve_in_venv` for the backends touched, and build the
  runtime model cache.
- Install the worker runtime into the env of each touched backend, replacing module staging.
- Build the computed env-grouping tool, run it over all 29 locks, and record the grouping it
  proposes without adopting it yet.
- Exit:
  - the same CrisperWhisper call runs on CTranslate2 and transformers, and the same PII-review call
    runs on vLLM and transformers, each by changing only the execution profile (P-8);
  - an agent given only `sl.describe` output runs each task (P-1);
  - outputs match 1.x within each task's parity tolerance (see below).

*Parity tolerances (design call).*
- Measured, not asserted. For each ported task, run 1.x and 2.x on a fixed reference set and
  record the tolerance as a data profile with its derivation (P-17). The reference set is the
  tutorial audio plus a small committed set.
- Measured on CPU, plus one CUDA host.
- A pipeline port additionally must not change any verdict on the reference set; a changed verdict
  blocks the phase.

**New integrations.**
- 2.0 is otherwise porting only. The exception is the vLLM-omni stack, which the owner named
  (P-8). It is added after Phase 3 as the first new integration, together with a vLLM ASR backend,
  because both depend on the plan and server machinery.
- Neither has a 1.x reference. Their acceptance is agreement with the transformers stack on the
  same model, within a measured tolerance.

**Phase 3: plans and executors.**
- Build:
  - the Plan with `explain` and hashing;
  - `InProcess` and `Pool`/`Slurm` executors;
  - open pydra PRs 1–4 upstream, in parallel with Phases 1–3;
  - `work_pool` items as plan fragments;
  - `senselab execute`;
  - cross-record batching;
  - GPU/CPU placement;
  - tunables from measured tables.
- Exit:
  - the same `.map` over a 1,000-record BIDS dataset runs on a laptop and on ORCD, with plans
    differing only in execution fields (P-10);
  - an interrupted run resumes by recomputing only missing results (P-5).

**Phase 4: pipelines and datasets.**
- Build `Pipeline` with ports, nesting, `over=`, `route`, `loop` and `Absent`; the BIDS and
  in-memory adapters; and dataset profiles.
- Port `audio_analysis` first, because it is free of dataset coupling. Then port `triage`, moving
  every coupling item from section 2 into a `b2ai-voice` profile. Port `explore_conversation` and
  `health_measurements` as small pipelines.
- Add one cross-modal pipeline (video → demux → diarize → ASR → text sentiment) as the P-24 test.
- Exit:
  - triage runs on B2AI and on a **constructed** second dataset by changing only the dataset
    profile (P-12). No second labelled dataset exists, so genericity is shown by design:
    - the constructed dataset is a small synthetic in-memory dataset whose profile uses a
      different entity scheme, task ids and task protocol (for example an invented "reading +
      sustained vowel + counting" protocol);
    - its thresholds are declared `unfitted`, so they route to review rather than decide (P-17);
    - the profile schema validates;
    - triage completes with every node running or typed-absent;
    - a static check finds no dataset names, task ids or sidecar field names in pipeline code;
  - grepping pipeline code for dataset names and fields finds nothing;
  - both pipelines nest inside a parent pipeline unchanged (P-9).

**Phase 5: remaining tasks, surfaces, removal.**
- Port the remaining tasks; ship the MCP server and CLI; generate docs; rewrite the tutorials
  against 2.x.
- Delete 1.x modules: the `api.py` dispatch, model wrapper classes, `SenselabDataset`,
  `call_in_venv`, bespoke workers, the second vLLM server, the stage cache, and the 16 `extend_*`
  drivers.
- Exit: no 1.x name exists (P-14), and the README quick start runs in CI.

`reproduce` can migrate at any point after Phase 3. Until then it keeps working unchanged, because
it already sits on `work_pool`, `vllm_server` and `ProvStore`.

## 7. Decisions left to the builder

- Internal module layout within core, and naming of private helpers.
- How generated functions are materialised: code generation at build time or dynamic at import.
  Either way, IDE completion and pdoc must see real signatures.
- The shared-memory mechanism for large arrays.
- The exact JSON Schema dialect and the mapping of numpy and data types into it.
- The order of task ports within Phase 5.

Calls already made in this document, marked *design call* where they appear, for the owner to
check:
- **In-process only inside the locked senselab venv when its lock covers the backend** (section 4.4,
  P-25).
- **Result-key composition** (section 4.8).
- **Phase 2 slice** from existing backends only (section 6).
- **Parity tolerances** measured as data (section 6).
- **vLLM-omni and vLLM ASR** as the only new integrations, after Phase 3 (section 6).
- **sklearn-style configured objects as the primary form, with immutable `fit`** (section 4.13).
- **No separate `Local` executor; pydra for the graph layer only, with a fallback** (section 4.8).
- **License enforcement** (section 4.12).
  - A license harvested from PEP 639, `License` or classifiers counts as a record.
  - At run time a missing or unknown license raises by default (`on_missing: raise`) for every
    model and package a plan touches, transitive packages included.
  - CI checks catalogue models and backends' direct dependencies, so the common path never raises
    for a user.

## 8. Owner answers (2026-10-30)

1. **Graph runtime.**
   - Pydra would need to be stable, debuggable and optimised for execution. PRs to Pydra to get
     there are welcome.
   - A ready layer (joblib or another engine) is an acceptable starting point if it is available.
   - Engine capabilities are to be assessed critically.
   - **Resolved** in section 4.8 from `workflow-engines.md`: no engine is adopted as the execution
     layer. pydra is used for the graph layer only, if PRs 1–4 land; otherwise senselab writes its
     own. joblib was assessed and is not a fit beyond in-process memoisation.
2. **In place on `alpha`.** Core is its own distribution; one repo builds several (section 5).
   1.x modules are deleted as their 2.x replacements land (P-14, P-27).
3. **No second labelled dataset exists.** Genericity is shown by design with a constructed
   dataset and profile (Phase 4 exit).
4. **Keep, port or drop: accepted as proposed.**
   - port: `adaptive_loop` (as `p.loop`), the tutorials (rewritten);
   - replace: `agentic_interface` (by the MCP server);
   - keep outside the package, as tools over run directories: the review-page and VLM tooling;
   - move out of the repo into site profiles: the ORCD campaign scripts and sbatch files;
   - keep as dev scripts: benchmarks.

Also from the owner on the same date:
- isolated, locked envs always (P-25);
- the signal generalisation (P-26, section 4.14, pending `use-cases.md`);
- license sets with raise-by-default (P-23);
- process principles P-27..P-29.
