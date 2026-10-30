# senselab 2.x API: design

Status: draft for owner review, revised after five reviews ([`review-register.md`](review-register.md)).
Principles are cited as `P-n` ([`principles.md`](principles.md)). Surveyed against `origin/alpha`
`19834fdd` (2026-10-30). Companion documents:
- [`execution.md`](execution.md): pools, GPUs, substrates, the pydra boundary;
- [`engine-choice.md`](engine-choice.md): requirements against workflow engines;
- [`testing.md`](testing.md), [`engineering.md`](engineering.md);
- [`use-cases.md`](use-cases.md), [`benchmarks.md`](benchmarks.md).

## 1. Intent

senselab 1.x built hard-won machinery: commit-pinned model loading, content-addressed caches, PROV
stores with a BEP028 export, 29 locked CUDA-routed isolated environments with persistent workers, a
vLLM server manager, a shared-filesystem work pool that drives Slurm, a pinned step DAG
(`reproduce`), and two large pipelines (`audio_analysis`, `triage`). Each piece was built for the
pipeline that needed it first, so it cannot be reached through one API, cannot be described to an
agent, and has drifted toward one dataset.

2.x keeps the machinery and gives it one shape. A user or agent declares *what* to compute. A
planner turns it, for a site and its hardware, into an inspectable plan. An executor runs the plan,
and the framework records provenance and caches results the same way for every task.

## 2. Current state

**Public API.**
- 36 task `api.py` modules with no common signature:
  - backends are selected four ways;
  - knobs are passed five ways (`utils/backend_parameters.py` validates them for 4 tasks only);
  - return shapes vary by task and sometimes by argument.
- Nothing is exported at the top level, every call takes a list, and the README quick start fails
  (`transcribe_audios` requires `model`).
- `model_registry.yaml` is barely read at run time.
- The agent interface writes notebook code and looks up symbols on DeepWiki. There is no task
  schema.

**Weight.**
- The required dependencies include torch, transformers, speechbrain, pyannote, datasets and
  accelerate.
- `Audio` imports torch (`audio/data_structures/audio.py:10`).
- `senselab/__init__.py` applies `nest_asyncio` and `set_start_method("spawn")` at import.
- CI only installs `--all-extras`.

**Execution.**
- 29 isolated envs, declared by module constants and found by an AST scan
  (`venv_lock.discover_backends`).
- Three to four worker protocols: `call_in_venv` (unused), about 20 one-shot scripts,
  `serve_in_venv`, and four bespoke workers.
- Two vLLM server implementations.
- A model dict per backend, with no shared model cache or memory budget.
- "Where it runs" is encoded three ways: a `device` argument, venv identity, and `ResourceClass`.
- `work_pool` items are argv only.

The 29 envs by torch pin:

| Group | Envs |
|---|---|
| torch 2.8 | coqui, diarizen, emotion2vec, moss, nemo-canary-qwen, nemo-diarization, pii-detection, pii-redaction-review, ppgs, qwen-asr, sparc |
| torch 2.11 | clearvoice, clearvoice-speechscore, continuous-ser, crisperwhisper, driftse, maai, pyfeat, qwen-tts |
| torch 2.13 (vLLM) | vllm-server, pii-review-vllm |
| no torch | hear (TF), mediapipe-face, mlx-vlm, yamnet (TF) |
| old one-off pins | brouhaha (2.2), child-adult-diarization (2.3, Python 3.10), s3prl (2.4), unasdiff (2.6, Python 3.10) |

**Provenance and caching.**
- Three key schemes: the `cached_inference` stage and result caches, `ProvStore` with its BEP028
  export, and `reproduce` step keys.
- Model identity is mostly sound: resolve once, load by SHA, check non-hub weights against sha256
  constants, and record an `unresolved_reason` where pinning is impossible.

**Pipelines.**
- triage names its nodes but runs them from a hand-written runner with no declared inputs.
- audio_analysis declares `StageContract{reads, writes}` but only checks it; execution is a
  sequence of `if` blocks.
- `reproduce` is the only real declared DAG, and its steps are whole processes.
- Corpus runs use a JSONL manifest, `--slice-index/--slice-count`, and 15 `extend_*.py` drivers.
- `SenselabDataset` is unused, and `create_bids_dataset` is a stub.

**Dataset coupling (violates P-12).**
- BIDS entity regexes appear in six places.
- B2AI sidecar fields are hard-coded (`b2ai_hints.py`).
- About 28 task-family literals, plus per-task expectations keyed by B2AI task names
  (`nodes/branches.py:830-1130`).
- B2AI policy sits in the packaged `default.yaml`.
- Thresholds were fitted on one corpus without naming the population.
- ORCD paths appear in scripts.
- audio_analysis, explore_conversation and health_measurements are clean.

## 3. Goals, non-goals, anti-goals

**Goals:** satisfy the principle register; provide a small vocabulary (section 4) that covers
section 2; make every kept 1.x capability reachable through it.

**Non-goals for 2.0:**
- a GUI;
- a general-purpose workflow engine;
- new models or analyses (porting only, except vLLM-omni and a vLLM ASR backend, section 6);
- multi-node model parallelism;
- substrates other than local processes and Slurm.

**Anti-goals:**
- a core that installs torch;
- a result that cannot be traced to model bytes, env and inputs;
- a pipeline that needs code edits for a second dataset;
- two ways to declare one thing;
- an agent schema maintained by hand;
- a plan that cannot be printed before it runs.

## 4. Design

### 4.0 Vocabulary

**Core nouns** (what a user composes):
- `Signal`, with its specialisations;
- `Record`;
- `Dataset`;
- `Task`;
- `Model`;
- `Pipeline`;
- `Profile`;
- `Plan`;
- `Run`.

**Supporting types** (what results and declarations are made of):
- `Span`, `Annotation`, `Events`, `Intervals`, `Embedding`, `Transcript`, `TimeMap`, `Absent`;
- `Params`, `Backend`, `Env`, `License`, `ResultSet`.

Provenance and the cache are produced around every task execution; users never handle them
directly.

### 4.1 Data and signals (P-13, P-24, P-26)

- **`Signal`** is the base type:
  - `data` with shape `(channels, samples)`, a numpy array with a dtype per channel;
  - `sampling_rate`;
  - `channels: [ChannelInfo{name, type, units, placement, reference}]`;
  - `start: TimeAnchor{absolute_start, clock_id}`;
  - `device`;
  - `identity`.

  `Audio`, `ECG`, `Accelerometry`, `PPG`, `EDA` and so on are specialisations that validate units
  and layout. Tracks (f0, heart rate, ENMO, AU intensity) are Signals too.
- **`IrregularSignal`** carries per-sample timestamps. Regularising one is an explicit task. Gaps
  are `Intervals` and are never zero-filled.
- **`Video`, `Image` and `Text`** follow the same lazy, content-identified pattern.
- **Loading** is lazy, by path, byte range, stream or time range. Derived data is chunked. Tensors
  exist only inside backends.
- **Identity** is computed after the canonical decode, so changing the container does not change
  identity. Units enter identity. Data objects are immutable, and a transform records its parent
  (`wasDerivedFrom`).
- **One timeline.** Every timed result is a `Span{start, end, source}` on its source clock. A
  synchronisation task produces a `TimeMap` (offset plus drift, with uncertainty) that projects
  between clocks.
- **Results.** An `Annotation{span, label|value, score, producer}` is the per-item view;
  `Events`/`Intervals` are the columnar form at scale (Parquet or BIDS `events.tsv`). A
  `Transcript` is a tree of annotations (1.x `ScriptLine.chunks`). An `Embedding{vector, space}`
  has `space` equal to the resolved model identity.
- **Serialisation.** Results are pydantic models with a stable JSON form. Arrays go to `.npy` or
  safetensors.
- **Records.** A `Record` holds named views (`audio`, `video`, `ecg`, `acc@wrist`, `transcript`)
  that share its entities and timeline.

### 4.2 Task (P-1, P-17, P-20)

A Task is declared once, in core, without importing a backend:

```python
@task(name="speech_to_text", version=3, kind="measure")
class SpeechToText:
    class Params(TaskParams):
        language: Language | None = None
        timestamps: Literal["word", "segment", "none"] = "word"
    inputs  = {"audio": Audio}
    outputs = {"transcript": Transcript}
```

- **Kinds:**
  - `measure`: no verdict fields (P-20);
  - `transform`;
  - `decide`: consumes measurements plus a profile;
  - `fit`: learns state, see section 4.14.
- **Configured, immutable objects are the primary form:**
  ```python
  asr = sl.audio.SpeechToText(model="openai/whisper-large-v3-turbo@main", language="en")
  t   = asr("a.wav")                  # blocking convenience: one item, one result
  rs  = asr.map(dataset)              # lazy ResultSet backed by a Run (section 4.7)
  asr.with_params(language="es")      # a new object; objects never mutate
  ```
  - The function `sl.audio.speech_to_text(x, **params)` is exactly
    `SpeechToText(**params)(x)`.
  - Data arguments accept objects, paths or URIs.
- **Generated from the declaration:** the function, `.map`, a JSON Schema (inputs, params,
  outputs, models, backends, stacks), the pdoc page, and the contract tests (`testing.md`).
- **One call shape** everywhere: `{task | pipeline, inputs, params, model}`. Python keyword
  arguments are that `params` mapping flattened. The CLI and JSON use it as is.
- **Rules:**
  - `Params` forbid unknown keys and suggest close matches.
  - Backend-specific knobs go in `backend_params`, validated against the chosen backend.
  - `model` is a reference string or a `Model`.
  - A model-backed task names a catalogue default model; model-less tasks (DSP) have none.
  - `device` is never a task argument (P-2).
  - Defaults that encode a decision come from a profile key with a derivation (P-17).

### 4.3 Model (P-4, P-16)

- **References.** `Model("org/name@rev")` is a reference (`@` only ever means a model revision).
  `resolve()` returns `ResolvedModel{source, id, requested_ref, identity}`. The identity is the
  commit SHA for hub models and the sha256 of the weight set otherwise.
- **Loading** takes only a `ResolvedModel`. Sources that cannot be pinned resolve with an
  `unresolved_reason` and are labelled `unpinned`.
- **The catalogue** (from `model_registry.yaml`) is runtime data. Each entry holds the tasks it
  serves, its backends and stacks, the default ref, its license, and a measured resource table per
  hardware class. The scale ladder produces those tables (P-7, P-17).

### 4.4 Backend, stack and environment (P-6, P-7, P-8, P-25)

```python
@backend(task="speech_to_text", stack="ct2", models=["nyrahealth/CrisperWhisper"],
         env=Env.isolated("crisperwhisper"), devices=["cuda", "cpu"],
         batch_invariant=False, tunables={"dtype": ["float16", "int8"]})
class CrisperWhisperCt2: ...
```

- **Stacks:** `transformers`, `vllm`, `vllm-omni`, `nemo`, `ct2`, `mlx`, `onnx`, `speechbrain`,
  `pyannote`, `dsp`. A task can have one backend per stack for the same model.
- **Env kinds, all locked (P-25):**
  - `Env.host()`: in-process, in the locked senselab venv. It is allowed only when that lock
    covers the backend.
  - `Env.isolated(name)`: a locked uv venv, or a pixi env for R, Java and native stacks.
  - `Env.server(name)`: the env of a long-lived engine process such as vLLM. A service is a
    resident resource in a pool worker of class `svc` (`execution.md`).
- **Env identity** follows I-2 (section 4.8).
- **Native libraries** come from wheels. Any remaining system library is a declared
  `SystemRequirement`: probed at start, recorded, and tracked as debt. The planner never falls back
  to unlocked or system packages.
- **One worker protocol:** `serve_in_venv`, persistent, with a JSON control line, arrays by file
  or shared memory, and typed errors.
- **One model cache per process,** keyed by `(backend, model identity, device, dtype)`, with an
  LRU under a memory budget.
- **Tunables** are declared ranges. Their values come from the execution profile or the
  catalogue's measured tables, not from branches in code.

### 4.5 Pipeline (P-9, P-10)

```python
p = sl.Pipeline("conversation")
seg = p.add(sl.audio.Diarize(), audio=p.input("audio"))
asr = p.add(sl.audio.SpeechToText(language="en"), audio=seg.out.segments)   # fan-out from list port
spk = p.add(speaker_identity, audio=p.input("audio"), turns=seg.out.turns)  # nested Pipeline
p.output(transcript=asr.out.transcript, speakers=spk.out.speakers)
```

- **Nesting.** A `Pipeline` satisfies the Task protocol, so it can be used as a node.
- **Fan-out** is inferred: wiring a list-typed output into an item-typed input runs the node per
  element. `over=` is used only to name a dataset grouping key (`session`, `participant`,
  `corpus`) that comes from the dataset's entities (section 4.6).
- **Routing.** `p.route(selector, branches={...})` chooses among declared sub-graphs at run time.
  **Loops:** `p.loop(body, until=..., max_rounds=n)` runs a body for bounded rounds, and unrolled
  rounds get distinct ids.
- **Absence.** `Absent(cause)` is typed: content-caused or environment-caused (I-6). Downstream
  nodes declare whether they accept it. A failure is not an absence.
- **Ports** carry schemas: `StageContract`/`StageIO` become the port model, and edges are checked
  when the graph is built.
- **Serialisation.** A pipeline serialises losslessly to JSON/YAML against a published graph
  schema, and `sl.validate()` checks it without planning.

### 4.6 Dataset and profiles (P-11, P-12, P-18)

```python
ds = sl.Dataset.bids("/data/ds", dataset_profile="b2ai-voice:2026-09")
ds = sl.Dataset.from_records([{"audio": "a.wav", "participant": "p1"}])
ds = ds.filter(task="reading-passage")
```

- **The `Dataset` protocol** yields `Record`s. It provides `entities()`, filtering, a stable
  `Record.id`, a manifest form, and slicing that is closed under grouping (I-7).
- **Binding.** A record view binds to the pipeline input of the same name. `bind={}` overrides
  that.
- **Adapters.** The BIDS adapter parses entities once, with one implementation. The in-memory
  adapter wraps lists, dicts, frames, or HF datasets.
- **Three profile kinds**, versioned as `name:version`, deep-merged over packaged defaults, and
  with their derivations in a sibling file:

  | Profile | Holds |
  |---|---|
  | run config | model choices, thresholds, aggregators, `license_policy` |
  | dataset profile | entities, sidecar→hint mapping, task protocol, lexicons, cohort conditions, references to fitted-profile artifacts |
  | execution profile | hardware classes, substrate, partitions, roots, stack preferences, caps, policy limits, network policy |

- **Profile slices.** A node reads only the profile slices it declares, and those slices are
  among its keyed inputs (I-1).
- **Moving B2AI out.** All of section 2's dataset coupling moves into a `b2ai-voice` dataset
  profile.

### 4.7 Plan and run lifecycle (P-2, P-5, P-10)

```python
plan = sl.plan(p, ds, config="cfg.yaml", dataset_profile=..., execution="orcd.yaml")
plan.explain()             # backend, stack, env, device, batch, partition, licenses, cost, why
run  = plan.run()          # -> Run
run.status(); run.wait(); rs = run.results(); run.cancel()
rs.to_frame()              # entities as index
sl.run(p, ds, ...)         # = sl.plan(...).run(); task or pipeline
sl.use(execution="laptop") # session default
```

- **A Plan is immutable and hashed.** It includes every resolution (model SHAs, catalogue
  defaults, hardware probe, env identities, chunk grid), so resuming means `resume(plan_hash)`
  and never re-planning (`execution.md`).
- **Before running, a plan announces** the env builds it needs, its cost (wall time, GPU-hours,
  downloads) and its license set.
- **Laptop and cluster plans** for the same inputs differ only in execution fields (P-10).
- **The hardware probe needs no torch.** It extends `cuda_probe` and `reproduce/host.py`.
- **Placement and batching:**
  - GPU and CPU nodes are placed separately.
  - Nodes sharing a resident model are grouped.
  - Batching happens across records and obeys I-3.
  - Large arrays move by file or shared memory.

### 4.8 Identity (P-3, P-4, P-19)

**The canonical result key.** Every other document references this one:

```
{schema, task, task_version, backend, stack,
 env numerics identity, numerics class (unless a measured tolerance covers both classes),
 identity tunables (dtype, … ; batch formation when the backend is not batch-invariant),
 model identity, params,
 identity of every declared input (data, profile slices, fitted-profile artifacts, hints, TimeMaps),
 chunk grid (when chunked)}
```

The senselab version, batch size (for batch-invariant backends), device index and partition are
recorded but not keyed.

**Invariants.** These came from the adversarial review under P-35, and the owner may promote any of
them to a principle.

| Id | Invariant | Check |
|---|---|---|
| I-1 | **No ambient reads.** Everything a node reads reaches it through a declared, identified input. | perturbation contract: perturb every undeclared source (env vars, other profile keys, cache roots); output unchanged |
| I-2 | **Numerics identity is separate from protocol identity.** Numerics identity = lock sha + platform tag + installed wheel-set hash (the senselab venv uses the same formula). Protocol identity = worker-runtime source hash + wire version, checked only at handshake. The worker runtime is an overlay, not part of the requirements sha. | env tier: a release that changes only the worker runtime leaves numerics identities and cache keys unchanged |
| I-3 | **Batch invariance is declared and certified.** Otherwise batches form from canonical length buckets (bucket in the key) or batch size 1. Nondeterministic backends are labelled. | limits contract: batch of 1 against mixed batches, within a measured tolerance |
| I-4 | **The numerics class** (GPU architecture, precision mode) is keyed unless a measured tolerance covers both classes. | tolerance profiles exist for every class pair treated as equal |
| I-5 | **A composite key** is the Merkle hash of its inner node keys. | contract: changing an inner node changes the composite key |
| I-6 | **`Absent` carries a cause class.** Only content-caused absence is cached. | contract: environment-caused absence is never read back from the cache |
| I-7 | **Shards are closed under the coarsest grouping level.** A group node's key includes a hash of its member manifest and it checks completeness. Chunkable tasks declare a chunk grid and receptive field as identity, and shards align to the grid. | logic: a split group fails planning; misaligned shards are refused |
| I-8 | **Services attest their identity** (engine version, model SHA, config) on every response, and the key uses the attested value. | limits: restarting a service with a new config changes keys |
| I-9 | **License taint propagates** along `wasDerivedFrom`. The policy is checked on everything consumed, including cache hits and fitted profiles. | contract: a non-commercial-derived artifact fails a permissive-only plan |

**One result store** replaces the stage cache, the result cache and the separate key schemes:
- layout `schema-<N>/<k[:2]>/<k>/`, plus a packed tier for small results (`execution.md`);
- atomic rename, first writer wins;
- failures are not cached;
- a new schema directory per bump.

**One provenance store** (`ProvStore`, append-only, content-derived ids) is written around every
node with the attested identities. BEP028 export is how runs are published. `reproduce` becomes a
consumer of this store.

### 4.9 Execution (P-2, P-5, P-6, P-7)

`execution.md` holds the full design; this is the summary.

- **Executors:** `InProcess` (notebooks, tests) and `Pool` (`work_pool` with plan-fragment items).
- **Substrates in 2.0:** local processes and Slurm. They start pool workers; they do not run the
  graph.
- **Coordination:** a restartable coordinator with a fenced lease releases ready jobs to queues
  per resource class and resident identity.
- **Robustness:** typed faults decide what is charged to an item, scheduling is fair, a memory
  admission arbiter runs on each device, and services are registered endpoints.

**pydra's role,** stated once here and used consistently elsewhere:
- **Route A.** A pydra Submitter runs the graph layer (nesting, ports, fan-out, routing, loops),
  with a pool worker plugin sending jobs to senselab's queues. pydra's `Environment.checksum` (G1)
  is used, and pydra's pickle cache is not used once D1 and D2 land.
- **Version pinning.** pydra changes are used from the branch through `[tool.uv.sources]`, with
  the commit pinned in `uv.lock`, and released in 1.0 alphas.
- **Decision gate at the start of Phase 3.** If E1, D1 and F2 are not working on the branch,
  senselab builds its own graph layer behind the same `Executor` protocol: 2–4k lines, 3–5
  person-weeks, reusing `reproduce/dag.py` and `audio_analysis/contracts.py:991`
  `unrolled_contracts`.
- **Citations.** pydra changes are cited by proposal id (A0…J2), from
  [proposal 0001](https://github.com/nipype/pydra/blob/enh/execution-extensions/proposals/0001-execution-extensions.md).
- **Why pydra:** `engine-choice.md`.

### 4.10 Surfaces for humans and agents (P-1, P-18, P-31)

- **Python:**
  - `import senselab as sl`, with lazy `sl.audio`, `sl.video`, `sl.text`, `sl.signal`,
    `sl.physio`, `sl.motion`, `sl.eeg`, `sl.eyetrack` and `sl.multimodal`;
  - top level: `plan`, `run`, `use`, `validate`, `describe`, `tasks`, `pipelines`, `models`,
    `Dataset`, `Pipeline`;
  - no side effects at import.
- **CLI:** inputs plus profiles, with no per-knob flags.
  - `senselab tasks | pipelines | describe <name> | models`;
  - `senselab plan|run <task|pipeline> <data> [--config] [--dataset-profile] [--execution]`;
  - `senselab runs | status | cancel <run>`;
  - `senselab prov <run> <query>`;
  - `senselab execute --pool <dir> --classes <list>` (worker).
- **MCP server (`senselab mcp`).** The same operations as JSON tools, with schemas generated
  from the declarations. It replaces DeepWiki lookups.
- **Discovery:** `describe` works on tasks, pipelines, datasets (`describe_dataset`: entities,
  counts, views, hints), profiles and executors. Port-type queries such as
  `tasks(accepts="Transcript")` are supported, and the graph format has a JSON Schema.
- **Errors:** one envelope, `{code, message, fix, field, suggestions}`.
- **Results:** inline up to a size cap; above it, a reference `{ref, shape, dtype, summary}`.
  Dataset-wide results are paginated.
- **Runs:** `run` returns a handle immediately. The tools `runs`, `status`, `progress`
  (done/failed/absent counts) and `cancel` manage it.
- **Provenance:** `query_provenance(run, select)` uses the ProvStore accessors (`generated_by`,
  `derived_from`, `uses_of`).
- **Safety:**
  - `run` requires a plan handle whose cost estimate the agent has acknowledged.
  - Policy and budget caps live in the execution profile, and the caller cannot override them.
  - Data roots are allowlisted.
  - Relaxing a policy (for example `on_missing: warn`) is a capability of a role, not a parameter.

### 4.11 Across modalities (P-24)

- **Shared types and paths.** All modalities share the timeline, annotation, embedding, record,
  identity, cache, provenance, `Params`, plan and executor.
- **What stays per modality:** decoding, modality-specific DSP, domain decision pipelines, and
  the catalogue sections.
- **Bridges** are declared tasks whose input and output modalities differ:

  | Bridge | Examples |
  |---|---|
  | video → audio | demux; audio-visual target speaker extraction |
  | audio → text | ASR, forced alignment |
  | text → audio | TTS, voice cloning |
  | audio + text → decision | PII in speech, task-protocol checks |
  | video + audio → joint | face–voice association, turn-taking, conversation analysis |
  | any → text | ALM/VLM description |
  | physiological | ECG/PPG → respiration, accelerometer + PPG → motion-corrected pulse, video → rPPG |

  Bridge tasks live in `sl.multimodal`.
- **Environments are grouped by dependency set,** not by modality.
  - A lock-time tool merges backends into the fewest envs whose combined requirements resolve, and
    records the grouping as data.
  - Merging changes numerics identity, which is correct.
  - A persistent worker in a merged env serves several modalities.
  - A single vLLM or vLLM-omni service can answer ALM, VLM and text-LLM nodes, where the run config
    chooses it.

### 4.12 Core and the worker runtime (P-22)

| Piece | Runs | Python | Dependencies |
|---|---|---|---|
| `senselab_core` | host | non-EOL CPython (`>=3.11` at 2.0) | numpy, pydantic v2, … (section 5) |
| `senselab_worker` | inside every isolated env, as an overlay | whatever the lock needs (3.10–3.13 today) | stdlib + numpy `>=1.23` |

- **The worker runtime holds:** the wire codec, the `serve_in_venv` loop and handshake, plain-dict
  encoders, weight hashing, the per-worker model cache, typed errors, and provenance records as
  JSON.
- **Core holds:** validation, the planner, executors, the catalogue and the ProvStore.
- **Handshake.** It exchanges a wire-protocol version and the worker-runtime source hash (I-2). A
  mismatch refuses the connection; nothing is decoded.
- **Two implementations of weight hashing.** Core and worker both hash weights, so a
  cross-implementation golden test runs in the env tier.
- **Floor test.** The worker runtime has a floor test against the oldest Python and numpy in any
  lock, run from the built wheel.

### 4.13 Licenses (P-23)

- **The record:**
  ```python
  License(expression="CC-BY-NC-4.0", commercial_use="no", copyleft="none", attribution=True,
          gated=False, source="<card at resolved commit>",
          verified={"date": "2026-10-30", "how": "hf-card|pep639|manual"})
  ```
- **What carries a record:** models (from the card at the resolved commit), every locked package,
  transitive ones included (from PEP 639, the `License` field or classifiers; a harvested record
  counts), datasets (in the dataset profile), and senselab itself (Apache-2.0).
- **Derived artifacts** inherit the join of their sources' licenses (I-9).
- **Policy.** One schema: `license_policy: {allow: [license sets], on_missing: raise | warn}`.
  - License sets are named, versioned data: `permissive`, `weak-copyleft-ok`, `copyleft-ok`,
    `non-commercial-ok`, `gated-ok`, plus user-defined sets combined by union or intersection.
  - The packaged default is `allow` = the union of the packaged sets, with `on_missing: raise`.
    Research runs are therefore never blocked by license *type*, only by a missing record.
- **Enforcement.**
  - The planner checks every node and every consumed artifact before running.
  - A `warn` relaxation is recorded in provenance, and results made under it are labelled.
  - `sl.licenses(plan)` reports the union and its flags, per node.
- **CI gate.** CI applies the run-time default to everything packaged at that phase: catalogue
  models and locks, transitive packages included. An unknown needs a dated manual record.
  Backfill happens per ported backend, from Phase 2.
- **Model-specific license findings** for new modalities are in `use-cases.md`.
- senselab reports licenses and enforces the user's policy. It does not interpret them legally.

### 4.14 Relation to scikit-learn, and `fit` (P-17)

**Adopted:**
- hyperparameters at construction;
- one uniform interface;
- immutability: `.with_params()` instead of `set_params`;
- `fit` for tasks that learn state.

**`fit` end to end:**
- `est.fit(inputs, labels=<declared input>)`.
  - `inputs` are records or upstream results.
  - `labels` are another task's results, or annotations in the dataset.
- It returns a new fitted object whose state is a **Profile artifact**: content-hashed, with an
  emitted derivation (inputs, labels, code version, config) and provenance.
- Decide tasks take `profile=<artifact>`. Dataset profiles refer to fitted artifacts by identity,
  so there is a single path, and the artifacts carry inherited licenses (I-9).

**Interop:**
- `sl.sklearn.SenselabTransformer(task)` wraps an array-valued task as a sklearn transformer. It
  runs in-process.
- `ResultSet.to_frame()` returns a frame indexed by record entities.

**Not adopted:**
- fit/transform/predict as the core verbs (most tasks are pretrained inference that change the
  type);
- `X` as an `(n_samples, n_features)` array;
- a linear `Pipeline`;
- mutable estimators;
- eager in-process execution.

### 4.15 Signal task tiers (P-26)

The taxonomy and tables are in `use-cases.md` §4.

1. **Generic, in `sl.signal`:** resample, filter, units, windowing, spectral and envelope
   features, activity and change-point detection, span algebra, QC, synchronisation (produces a
   `TimeMap`), embeddings, imputation. The 1.x audio DSP that is really generic moves here.
2. **Families with modality profiles:** one task per family, for example
   `sl.signal.detect_peaks(x, profile="ecg_rpeak")`. Profiles live in `data/` with derivations, and
   the default is chosen by channel `type`. `sl.physio.ecg_rpeaks` is sugar over the family task.
3. **Modality-specific:** `sl.audio`, `sl.video`, `sl.text`, `sl.physio`, `sl.motion`, `sl.eeg`
   (delegating to MNE/YASA), `sl.eyetrack`.

Inputs are typed `Signal[type ∈ {...}, units]`, so `tasks(accepts=...)` works for channels. QC
tasks emit SQI tracks and gap intervals, while non-wear and unusable-epoch verdicts are `decide`
tasks with fitted profiles.

## 5. Packaging (P-13, P-22, P-25)

**Distributions.** One repo builds three distributions as a uv workspace:
- `packages/senselab-worker` (import `senselab_worker`);
- `packages/senselab-core` (import `senselab_core`);
- the root `senselab`, which holds backends, the catalogue and pipelines, and whose facade
  (`import senselab as sl`) imports from core.

**Versioning.**
- A single git tag and hatch-vcs with the repository root.
- An exact same-version pin between the three, written by a metadata hook. Workspace sources do
  not appear in published wheels.

**Core** depends on numpy, pydantic, pyyaml, filelock, xxhash, soundfile, huggingface-hub
(resolution only) and packaging; pyarrow is an extra. It holds:
- data types;
- declarations;
- catalogue;
- identity, store and provenance (`prov_store`, `prov_bep028`, `file_lock`);
- profiles and the planner;
- executors (`work_pool`, `fastio`, `vllm_server`, `uv_tools`);
- torch-free DSP.

**Backend modules** import heavy dependencies only inside their implementation. A test imports
every declaration with only core installed.

**Install.** senselab is installed into a venv from its lock (`uv sync --locked` or
`senselab env create`). In-process groups (`torch`, `video`, `nlp`) are lock-covered extras. The
locks for isolated envs ship as package data.

**Torch reach:** core is torch-free; the senselab venv is torch-free when every torch backend runs
isolated; senselab as a whole is not torch-free.

**Python floors.** The uv workspace locks one `requires-python`, so the worker's floor is tested
from the built wheel in a separate env.

## 6. Migration

### 6.1 Strategy (in place on `alpha`, P-14, P-27)

- **Phase 0:** tag a frozen 1.x commit. Running campaigns use it from a pinned checkout until
  their pipeline is ported.
- **One-way imports:** 1.x code may import core, and core never imports 1.x. import-linter
  enforces this.
- **Moves are rewires.** When a module moves into core (`prov_store`, `file_lock`,
  `model_revision`), every caller is rewired in the same PR. There are no re-exports and no aliases.
- **Separate store.** New types live in `senselab_core`, and the 2.x result store uses a new
  schema directory. The full recompute is announced in advance.
- **Ported backends.** When a backend is ported, triage's call to it is rewired in the same PR,
  so no second protocol survives.
- **1.x import side effects** stay until Phase 5.

### 6.2 Phases

Every phase ends with an adversarial round scoped to its new surfaces (P-35). Findings go into
`review-register.md`.

**Phase 1: core foundations.**
- Build:
  - the workspace and the three packages;
  - the worker runtime: codec, handshake per I-2, wire-protocol spec, and a floor test from the
    wheel;
  - `Signal` (with `Audio` as a Signal), the timeline, annotation and embedding types;
  - `Model`/`ResolvedModel`, rewiring `model_revision` and the sha constants;
  - the result store with the canonical key fields fixed (section 4.8), and the ProvStore writer;
  - the torch-free hardware probe;
  - the `License` record type, with no backfill yet;
  - the `fake` stack with its declared fault model, and the generated contract-test module.
- Exit:
  - the core-only job is green on Python 3.11 and 3.14 (P-13, P-22);
  - the identity, store and provenance tests pass without torch (P-3, P-4, P-19);
  - the worker floor test passes;
  - 1.x tests pass after the rewiring, and triage outputs are unchanged.

**Phase 2: tasks and backends.**
- Build:
  - task and backend declarations, `Params`, generated functions and schemas;
  - a minimal execution-profile resolver and `Env.server`;
  - the runtime catalogue;
  - one worker protocol and the model cache for the touched backends.
- Port, from existing backends only:
  - `speech_to_text`: HF Whisper on transformers (host), and CrisperWhisper on ct2 and
    transformers (isolated);
  - `diarize` (pyannote);
  - `speaker_embeddings` (speechbrain);
  - text PII review on vLLM and transformers;
  - one DSP primitive.
- Backfill licenses for the ported backends. Run the env-grouping tool and record its proposal.
- Exit:
  - each two-stack call switches by execution profile alone (P-8);
  - an agent runs each task from `describe` output alone (P-1);
  - parity holds within measured tolerance profiles (P-17), on CPU and on one CUDA host.

**Phase 3: plans and executors.**
- Build:
  - the immutable `Plan`, `explain`, run lifecycle and `resume`;
  - `InProcess` and `Pool` on local and Slurm substrates;
  - plan-fragment items, the coordinator with a fenced lease, and the fault taxonomy;
  - cross-record batching under I-3;
  - a minimal BIDS adapter;
  - the `benchmarks/scale` ladder and the catalogue resource tables (P-7, P-33).
- The pydra gate is decided at the start of this phase (section 4.9).
- Exit:
  - a 1,000-record BIDS `.map` runs on a laptop and on ORCD, with plans differing only in
    execution fields (P-10);
  - an interrupted run resumes from its plan hash and recomputes only what is missing (P-5).
- New integrations follow this phase: vLLM-omni and a vLLM ASR backend. They are accepted against
  the transformers stack on the same model, within a measured tolerance.

**Phase 4: pipelines, datasets, signals.**
- Build:
  - `Pipeline` with ports, nesting, routing, loops and `Absent`;
  - full dataset adapters and profiles;
  - the merged config loader;
  - the generic `sl.signal` tier;
  - adoption of the env grouping (P-24).
- Port audio_analysis, then triage (all B2AI coupling into `b2ai-voice`), then
  explore_conversation and health_measurements. Add one cross-modal pipeline.
- The triage gate is **adjudicated verdicts**: every changed verdict on the reference set is
  explained with its cause, and an unexplained change blocks the phase.
- Exit:
  - triage runs on B2AI and on a constructed synthetic dataset by changing only the dataset
    profile (P-12):
    - the synthetic dataset has a different entity scheme and task protocol;
    - its thresholds are declared `unfitted`, so they route to review;
    - every node runs or is typed-absent;
    - a static check finds no dataset names, task ids or sidecar fields in pipeline code;
  - pipelines nest unchanged (P-9);
  - one generic task runs unchanged on audio and on accelerometry (P-26).

**Phase 5: remaining tasks, surfaces, removal.**
- Port the remaining tasks. Build the MCP server, the CLI and the docs. Rewrite the tutorials as
  use-case tests.
- Delete 1.x: the `api.py` dispatch, wrapper classes, `SenselabDataset`, `call_in_venv`, bespoke
  workers, the second vLLM server, the stage cache, and the 15 `extend_*` drivers.
- Exit: no 1.x name remains (P-14), and the README quick start runs in CI.

`reproduce` migrates any time after Phase 3.

### 6.3 Effort (person-weeks with agents, from the feasibility review)

| Phase | Effort | Main risk |
|---|---|---|
| 1 | 3–4 | the worker floor and the handshake |
| 2 | 6–8 | CUDA parity, vLLM |
| 3 | 6–10 | plan-fragment pool, resume on NFS |
| 4 | 12–20 | the triage port and verdict adjudication |
| 5 | 10–16 | about 30 tasks, MCP, tutorials |
| pydra PRs | 4–8, in parallel | E1/E2 (the State refactor) |
| fallback graph layer | 3–5, only if triggered | — |

## 7. Builder decisions and design calls

**Left to the builder:**
- core's internal layout;
- generating functions at build time or at import (IDE completion and pdoc must see real
  signatures);
- the shared-memory mechanism;
- the JSON Schema dialect;
- the order of task ports in Phase 5.

**Design calls for the owner to check:**
- the canonical key and invariants I-1..I-9 (section 4.8);
- in-process execution only inside a locked senselab venv that covers the backend (section 4.4);
- the run lifecycle and the single call shape (sections 4.2 and 4.7);
- `fit` producing Profile artifacts (section 4.14);
- the Phase 2 slice drawn from existing backends;
- measured parity tolerances and the adjudicated triage gate;
- vLLM-omni and vLLM ASR as the only new integrations;
- the pydra gate at the start of Phase 3, with its fallback (section 4.9);
- the license CI gate equal to the run-time default (section 4.13);
- pixi envs for R and Java.

## 8. Owner answers (2026-10-30)

- **Graph runtime:** pydra, extended, with a fallback (section 4.9; `engine-choice.md`).
- **Where the work happens:** in place on `alpha`. Core is its own distribution (section 5).
- **Second dataset:** none exists, so genericity is shown by design (Phase 4 exit).
- **The keep, port and drop list** is accepted:
  - port `adaptive_loop` (as `p.loop`), and the tutorials;
  - replace `agentic_interface` with MCP;
  - keep the review-page and VLM tooling outside the package;
  - move ORCD scripts into site profiles;
  - keep benchmarks as dev scripts.
- **Also:** locked isolated envs (P-25), signals (P-26), license sets with raise by default
  (P-23), process principles (P-27..P-29), testing tiers (P-30) and engineering standards
  (P-31..P-35).
