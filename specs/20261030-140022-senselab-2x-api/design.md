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

From that one declaration senselab generates:

- **a typed function for humans:**
  `sl.audio.speech_to_text(audio, model="openai/whisper-large-v3-turbo", language="en")`. It takes
  **one item and returns one result**.
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

**Executors**
- Executors implement one protocol, `submit(plan_fragment) -> handles`:
  - `InProcess`: the notebook and test default;
  - `Local`: process pool with GPU slots;
  - `Pool`: today's `work_pool`, with items extended from argv to serialised plan fragments, so a
    task call can be shipped without a CLI;
  - `Slurm`: the `work_pool` `Controller` plus held `afterok` chaining from `reproduce`.
- A worker on any host runs the same `senselab execute <fragment>` entry point.

**One content-addressed result store**
- It replaces the stage cache, the result cache and the separate key schemes. The key is
  `{schema, task, task_version, backend, stack-if-output-affecting, model identity, params,
  input identities, dataset-profile hash if read}`.
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

### 4.11 Core inside every environment (P-22)

senselab-core is installed into every isolated environment, so host and workers share one set of
types, one wire format, one identity hash and one provenance writer. Today workers instead get
copies of single modules, such as `stage_portable_audio_io` staging `portable_audio_io.py` onto
`sys.path`, or reimplement them in script strings.

Constraints from the current locks:

| Constraint | Today | Requirement for core |
|---|---|---|
| Python | envs use 3.10 (2), 3.11 (10), 3.12 (16), 3.13 (1); the host is `>=3.11` | only non-EOL CPython (P-22): `requires-python >=3.11` at 2.0, raised as versions reach EOL. Re-lock the two 3.10 envs (child-adult-diarization: torch 2.3.0; unasdiff: torch 2.6.0, numpy 1.23.5) on 3.11, since both pins publish 3.11 wheels. Ruff `target-version` for core = the floor |
| numpy | six envs pin 1.x, oldest `1.23.5` (unasdiff) | `numpy >=1.23`; no numpy-2-only API; tested at both ends |
| pydantic | absent or v2 in every lock; no v1 pin | `pydantic >=2,<3`. A future env that needs v1 is a lock-time error, not a silent downgrade |
| other deps | — | only pure-python or wheel-everywhere dependencies with wide ranges |

- **Same version on both sides.** The env installer adds the host's exact core version (the same
  wheel) on top of the env lock. The worker handshake checks the core version and the
  wire-protocol version, and refuses on a mismatch rather than mis-decoding.
- **The core lock is separate from each env lock.** Each env lock's requirements-sha includes the
  core version, so a core change rebuilds envs. This is cheap and correct, by the same logic as
  cache invalidation.
- **What a worker imports from core:**
  - data types: `Audio`, `Video`, `Span`, `Annotation`, `Embedding`, result models;
  - the wire codec;
  - the `ResolvedModel` identity and weight hashing, so hashes are computed where the weights are
    loaded;
  - the provenance writer;
  - the model-cache runtime;
  - `serve_in_venv`.

  The planner, executors and catalogue are importable but unused there.
- **CI:** core tests on every non-EOL Python (3.11–3.14 today) at minimum and maximum numpy, plus one test per env that imports
  core inside the built env and round-trips each data type over the wire.

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

**Enforcement**
- A CI test fails when a catalogue model or a locked package has no license record, or when a
  record has `unknown` without a dated reason.
- senselab reports licenses and enforces the policy the user sets. It does not give legal
  interpretation.

## 5. Packaging (P-13)

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
- **Extras:**
  - `senselab[torch]` installs torch, transformers, speechbrain and pyannote for in-process
    (`Env.host`) execution;
  - other in-process groups (`video`, `nlp`) as today;
  - isolated environments need no extra; their locks ship as package data.
- **CI** gains a core-only job (install core, import every public module, run the
  declaration/schema/planner/provenance tests) next to the existing full jobs.

How far 2.x can realistically get from torch:
- **core: torch-free;**
- **host process: torch-free if every torch backend runs isolated.** That is a valid execution
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
  - core at the Python 3.11 / numpy 1.23 floor, with the two 3.10 envs re-locked on 3.11 (section 4.11);
  - the `License` record, with licenses backfilled for all 50 catalogue entries and harvested from
    every env lock (section 4.12).
- Exit:
  - the core-only CI job is green on every non-EOL Python, 3.11–3.14 today (P-13, P-22);
  - the identity/cache/provenance tests pass with no torch installed (P-3, P-4, P-19);
  - the license-coverage test passes (P-23).

**Phase 2: tasks and backends on a vertical slice.**
- Build the Task and Backend declarations, Params validation, generated functions, `.map`, and
  generated JSON Schemas. Port four tasks that between them exercise every Env and stack kind:
  1. `speech_to_text`: transformers (host), vllm (server), mlx or ct2 (isolated);
  2. `diarize`: pyannote (host or isolated);
  3. `speaker_embeddings`: speechbrain (host);
  4. one numpy DSP primitive (`phonation` or `envelope`).
- Collapse the worker protocols into `serve_in_venv` for the backends touched, and build the
  runtime model cache.
- Install core into the env of each touched backend, replacing module staging.
- Build the computed env-grouping tool, run it over all 29 locks, and record the grouping it
  proposes without adopting it yet.
- Exit:
  - the same `speech_to_text` call runs on transformers and vLLM by changing only the execution
    profile (P-8);
  - an agent given only `sl.describe` output runs each task (P-1);
  - outputs match 1.x within each task's documented tolerance.

**Phase 3: plans and executors.**
- Build:
  - the Plan with `explain` and hashing;
  - `InProcess`, `Local` and `Pool`/`Slurm` executors;
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
  - triage runs on B2AI and on one second dataset by changing only the dataset profile (P-12);
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

## 8. Open questions for the owner

See the end of this section in the conversation; answers are folded in here once given.
