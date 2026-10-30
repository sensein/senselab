# pydra extension points for senselab 2.x execution

Source: `nipype/pydra` main at **c5081a6** (2026-10-20, tag v1.0a12; re-fetched 2026-10-30, unchanged).
Paths are relative to `pydra/`. Companion to `workflow-engines.md`.

Short answer:
- pydra has two real seams today:
  - the **Worker** plugin, which is async, receives an in-memory `Job`, and has completion read
    back from the cache;
  - the **Environment** plugin, which covers shell tasks only.
- It has hooks, but no resource, actor or result-store abstractions.
- Graph building is separable in principle. In practice, expanding state at run time reads upstream
  results through `Job.result()` from pydra's cache.
- So an external executor must either *be* a Worker, or pydra needs a pluggable result store.

## 1. Worker plugin API

**What a worker receives.**
- `Worker.run(job, rerun)` is an abstract method (`workers/base.py`, class `Worker`). It may be
  sync or `async`, and `is_async` is `inspect.iscoroutinefunction(self.run)`.
- It receives a live `Job` object, not a pickle.
- How the job moves depends on the worker:
  - Debug runs it inline (`workers/debug.py:16`, `job.run(rerun)`).
  - `cf` cloudpickles it and runs it in a `ProcessPoolExecutor` (`workers/cf.py`, `run` and
    `uncloudpickle_and_run`).
  - Slurm writes `_job.pklz` and an sbatch script that runs
    `sys.executable -c "load_and_run(pkl)"` (`workers/slurm.py:44-80`; `engine/job.py:637`).
- `Job.__getstate__` cloudpickles the task (`engine/job.py:184-190`).
- A workflow job is not sent to the worker. `Worker.submit` runs `job.run_async` in the main
  process, which re-enters `Submitter.expand_workflow_async` (`workers/base.py`, `submit`).

**How completion and errors are reported.**
- The worker's return value is mostly ignored: Slurm returns `True`.
- The submitter learns the state by polling `job.done`, which calls `job.result()`. That reads
  `cache_root/<checksum>/_result.pklz` (`engine/job.py:441-453`, `engine/result.py:132`).
- `NodeExecution.update_status` moves jobs into `successful`, `errored` or `running` from that
  read (`engine/submitter.py:725-760`).
- Exceptions raised by the worker coroutine are collected as strings into `errors`
  (`submitter.py:445-466`).
- **The contract is therefore "write a pydra `Result` pickle into `job.cache_dir`"**, and
  `job.run()` does that itself. Any remote worker has to run `job.run()` in a pydra-capable
  interpreter, or reproduce the file format.

**How a worker is selected.**
- `Submitter(worker=...)` accepts a name, a class or an instance. Names resolve through
  `Worker.plugin(name)`, which uses entry-point-style discovery of `pydra.workers.*` and
  `pydra-workers-<name>` packages (`submitter.py:143-160`, `workers/base.py`, `plugin`).
- There is **one worker per Submitter**: `self.worker` at `submitter.py:160`.
- Every node, including nodes of nested workflows (same submitter, `submitter.py:433-441`), goes
  to that one worker.

**Per-node routing.**
- Precedent: SGE already reads an ad-hoc per-job `job.qsub_args` (`workers/sge.py:110-119`), but
  nothing sets it.
- Routing is **easy (low risk)**:
  - replace `self.worker` with a `WorkerRouter` (`worker_for(job) -> Worker`) and call it at the
    two dispatch sites (`submitter.py:328` for sync, `433-441` for async);
  - let `workflow.add(..., worker=...)` or a resource class on the node (section 4) choose;
  - the event loop is already shared (`submitter.py:166`), so several async workers coexist.
- *Upstream:* the router, `Node._worker` and `add(worker=)`. *senselab:* the rule that maps a
  node's resource class or env to a pool, Slurm or in-process worker.

## 2. Environment API

**Interface.** `Environment` (`environments/base.py:21`) has three methods:
- `setup()` at line 29;
- `execute(job) -> {"return_code", "stdout", "stderr"}` at line 32;
- `teardown()` at line 48.

The implementations:
- `Native` runs `job.task._command_args()` through `base.execute`, a subprocess
  (`environments/native.py`).
- `Container` is the parent of Docker and Singularity. It computes `get_bindings` to map
  `FileSet` inputs and `cache_root` into the container (`environments/base.py`, roughly
  lines 80-121).
- `lmod` loads modules.

**How tasks use it.**
- Only `ShellTask._run` uses the environment: `job.return_values = job.environment.execute(job)`
  (`compose/shell/task.py:257-259`).
- `PythonTask._run` calls `self.function(**inputs)` in-process and **ignores the environment**
  (`compose/python.py:228-250`).
- An environment reaches a job from `workflow.add(..., environment=)` → `Node._environment`
  (`engine/node.py:39`) → `Job(environment=)` (`submitter.py:686,695`). The default is
  `Submitter.environment`, which is `Native` (`submitter.py:140`).

**A Venv environment for Python tasks (#840) needs four things.**
1. **A dispatch point.** `PythonTask._run` should call `job.environment.execute_python(job)` when
   the environment is not `Native`. Alternatively, `Environment.execute` could be generalised to
   take a callable payload.
2. **A payload that is not a cloudpickle of the function.** Interpreters differ (3.10 vs 3.13),
   and cloudpickle by-value is not portable across Python versions. Use an importable reference
   `module:qualname` plus inputs encoded as JSON or `.npy`, run by a small stdlib runner inside the
   venv. That runner is exactly senselab's worker runtime (design §4.12).
3. **Output decoding back into `Outputs`.** The values come back as files or JSON and are validated
   on the host.
4. **Optional persistence.** A one-shot subprocess per job is simple. A resident process is
   section 6.

**Environment identity in the checksum (#826).**
- `Job.checksum` is `task._checksum` only (`engine/job.py:213-222`, `compose/base/task.py:476`).
  The environment is not part of it, and neither is a Docker image tag.
- Insertion point: add `Environment.checksum() -> str` (default `""` for `Native`; image digest
  for containers; lock sha for a venv). Fold it in at `Job.checksum` as
  `f"{task._checksum}-{env.checksum()}"` or into the hash input. This changes no task hashes for
  `Native` users if the empty case is skipped.
- *Upstream:* `execute_python`, `Environment.checksum`, and a `Venv` environment given a python
  path and a lock sha. *senselab:* how venvs are built (`ensure_venv`, CUDA index routing, lock
  grouping) and the worker-runtime wire codec. pydra could accept a runner as a parameter.

## 3. Hooks and audit (per-job PROV)

- **Hooks.** `TaskHooks` (`engine/hooks.py`) has `pre_run`, `pre_run_task`, `post_run_task` and
  `post_run`. They are called in `Job.run` at `job.py:327, 346, 361, 367` and in `run_async` at
  `388, 406, 420, 426`.
  - They are set per node through `workflow.add(..., hooks=)` (`compose/workflow.py:239`) or per
    call through `Submitter.__call__(hooks=)`.
  - They run **in the process that runs the job**, so on Slurm they run remotely, are cloudpickled
    with the job, and must be picklable.
  - `post_run_task(job, result)` runs in a `finally`, so it fires on errors too. It is the right
    place for one PROV activity per job.
- **Audit.** `Audit` (`engine/audit.py`):
  - `start_audit` (line 48), `audit_task` (185) and `finalize_audit` (97) emit JSON-LD
    `prov:Activity` and `prov:Generation` messages through `Messenger`s (Print, File, RemoteREST;
    `utils/messenger.py:47-142`);
  - `ResourceMonitor` (psutil) records the runtime;
  - it fetches `context.jsonld` from GitHub unless `develop` is set; this is self-declared
    unfinished (#798, #566).
- **Proposal.**
  - *Upstream:* a `ProvSink` protocol called from `post_run_task`, writing one JSON line per job:
    used (input field hashes from `task._hashes`), generated (output ids), associated with
    (environment checksum, host), times, errored. Bundle the context, with no network fetch.
  - *senselab:* the `ProvStore` record schema (model agent, weights hash, env lock sha, host
    hardware record from `reproduce/host.py`) and the BEP028 export.

## 4. Resource requests

- **What exists.**
  - The `Submitter` takes `max_concurrent` (`submitter.py:105`).
  - The `cf` worker has `n_procs`.
  - Slurm has worker-global `sbatch_args` (`workers/slurm.py:31`).
  - SGE has `qsub_args` and a per-job `job.qsub_args` lookup (`workers/sge.py:110-119`).
  - There are **no CPU, memory or GPU fields on tasks**; the proposals are in #227 and #830.
  - `utils/profiler.py` measures resource use, which is not the same as requesting it.
- **Where `ComputeResources` should attach.** Three layers, with the most specific winning:
  1. **The task definition** (`python.define(resources=...)`, `shell.define(...)`, stored on the
     task class like `_executor_name`) is the default for that task;
  2. **The node** (`workflow.add(..., resources=)` → `Node._resources`, next to `_environment` and
     `_hooks` in `engine/node.py:39-40`) overrides it per use;
  3. **The `Submitter` or worker** maps resources to a backend through a site profile: sbatch
     args, partition, GPU type, or a local GPU slot semaphore.
  - The `Job` gets `job.resources` at construction (`submitter.py:684-698`).
  - Resources must **not** enter the checksum, just as batch size and partition stay out of
    senselab's key.
- *Upstream:* the dataclass (cpus, mem, gpus, gpu_class, walltime, with min and max ranges per
  #830), the attachment points, and `cf` GPU slots via `CUDA_VISIBLE_DEVICES` assignment.
  *senselab:* resource classes, partition and QOS tables, and the hardware probe. pydra should
  take a mapping callable, not a site schema.

## 5. Cache and result store: where atomic rename and pickle-free outputs go

All cache I/O is concentrated in four places:

| concern | location | today |
|---|---|---|
| key | `Job.checksum` → `task._checksum` (`job.py:213`, `task.py:476-503`) | `<tasktype>-<blake2b>`; fields with `container_path` are skipped (`task.py:494`) |
| directory | `Job.cache_dir = cache_root/checksum` (`job.py:247`); `lockfile` sibling (`job.py:225`) | the task runs **in** the final directory: `os.chdir(cache_dir)` (`job.py:338`) |
| write | `result.save()` (`result.py:182`) under `SoftFileLock(<dir>_save.lock)` | cloudpickle `_result.pklz` and `_job.pklz`, written in place |
| read | `load_result(checksum, caches)` (`result.py:132`) | `exists` + `stat`, then `cp.load` with 10 retries on `UnpicklingError` |
| errors | `record_error` (`result.py:305`) | `_error.pklz` |

**Insertion point.** Introduce `ResultStore` with three methods:
- `begin(job) -> workdir`, which creates `cache_root/tmp/<uuid>/`;
- `commit(job, workdir, result)`, which serialises the outputs, then renames `workdir` to
  `<checksum>/`. On `EEXIST` the first writer wins and the loser's copy is discarded;
- `load(checksum) -> Result | None`.

Changes this requires:
- `Job.run` and `run_async` (`job.py:313-375`, `377-430`) call `begin` and `commit` instead of
  `_populate_filesystem` and `save`.
- `load_result` delegates to `store.load`.
- The run-long `SoftFileLock` (`job.py:331`, `PydraFileLock` at `612`) becomes optional. With
  first-writer-wins, duplicate work is the only cost of not locking. A lease with a heartbeat
  (senselab `work_pool` protocol) can avoid duplicates on clusters.

**Risk.** File outputs are produced under the working directory and referenced by absolute path.
After the rename, the paths must be rewritten. `copyfile_workflow` (`result.py`) already rewrites
paths for workflows, which gives a precedent, but shell `Out` path templates and `fileformats`
copies need care.

**Pickle-free.** Add a serializer registry keyed by output type, mirroring
`register_serializer` for hashing (`utils/hash.py:368`):
- JSON for scalars and attrs, `.npy` or safetensors for arrays, files as files;
- cloudpickle as the fallback.
- Keep the cache layout versioned (`tasks-v<N>/`, #825).

*Upstream:* all of section 5. *senselab:* its key policy (model commit, env lock, identity
tunables), which it supplies through `__bytes_repr__` on its own types plus the environment
checksum (section 2). No pydra change is needed for that part.

## 6. Stateful and persistent tasks

- **Nothing exists.** There is no concept of a long-lived resource:
  - `Job.run` creates and tears down per job;
  - Slurm is one process per job;
  - the `cf` pool's processes do live across jobs, so a module-level `functools.cache`'d model
    loader is reused *by accident* inside a cf process. That reuse is unsound: placement is
    uncontrolled, GPUs are not partitioned, and `os.chdir` is process-global
    (`job.py:338`).
- **Where a resident-resource concept fits.**
  1. **Declaration on the task.** An input field marked `resource=True` (for example
     `model: Resident["asr", WhisperSpec]`), excluded from the hash in the same way
     `container_path` is (`task.py:494`). Its *identity* (model commit, env lock) is hashed through
     a sibling field, so the key stays correct.
  2. **Provision by the worker.** A `ResourceProvider` registry on the worker:
     `acquire(spec) -> handle`, LRU and memory budget per device, `release` at `Worker.close()`.
     `PythonTask._run` (`compose/python.py:228`) injects `handle` for resource fields before
     calling the function.
     - For a venv environment the handle is a client to a resident `serve_in_venv` process, not
       an in-process object.
  3. **Batching.** `Worker.run` is a coroutine, so a `BatchingWorker` can hold incoming jobs for
     up to *k* items or *t* ms. It groups them by `(task type, resource identity)`, makes one
     batched call (the task declares `batch_function`), then completes each `Job` separately
     through `ResultStore.commit`. **This needs no submitter change**; it does need
     per-element dataflow (proposal 0001 E1/E2), or batches only form within one
     node's split.
- *Upstream:* the `Resident` field marker, the provider protocol on `Worker`, the batching worker,
  and `chdir` removal so that several jobs share one process safely. *senselab:* the providers
  themselves (model cache keyed by backend, identity, device and dtype; `serve_in_venv`;
  `VllmServer` and servers per card) and the co-location rule. pydra stays generic: "a worker
  holds resources; tasks declare which ones."

## 7. Graph-only use: pydra as a graph compiler

**What is available without the Submitter.**
- `Workflow.construct(task)` (`engine/workflow.py:67`) builds the static `Workflow`. It runs the
  constructor with lazy placeholders and caches the result by non-lazy input hash.
- `wf.nodes` (`workflow.py:280`) gives each `Node` with its `_task`, `_environment`, `_hooks` and
  `state` (splitter RPN, combiner).
- `wf.graph(detailed=True)` (`workflow.py:303`) is a `DiGraph` with edge descriptions, and can be
  written as dot (`engine/graph.py:403-520`).
- This is enough to **print and type-check the static plan**.

**What is missing.**
- Expanding a node into concrete jobs needs values: `NodeExecution.start` → `State.prepare_states`
  → `_split_task` (`submitter.py:665-700`, `795-822`; `state.py:849`).
- Lazy fields resolve through `LazyOutField._get_value` → `job.result()`
  (`engine/lazy.py:160-200`), which reads pydra's cache.
- `NodeExecution` needs a `Submitter` instance: every `Job.__init__` reads `submitter.audit`,
  `cache_root` and `environment` (`job.py:91-155`).
- So there is no public "compile to jobs" API, and expansion is interleaved with execution.
- Nested workflows whose shape depends on upstream values are by design unexpandable until those
  values exist.

**Two routes.**
- **A. senselab executor as a pydra Worker.** Possible today, with the lowest risk.
  - `run(job)`: serialise `job.task` (type ref plus inputs), publish it to `work_pool`, and await
    completion.
  - The remote side runs `job.run()` (requires pydra in that interpreter) or writes a
    pydra-compatible result.
  - pydra keeps doing expansion, caching and dependency order; senselab owns placement and pools.
  - Limits: one worker per submitter (section 1), the pickle cache (section 5), and the barrier
    semantics (proposal E1/E2).
- **B. A compile API upstream.** `Submitter` is split in two:
  - a `GraphRunner`, which tracks ready jobs, does lazy resolution and state expansion, and reads
    results from a `ResultStore`;
  - an `Executor` interface (`submit(jobs) -> futures`).
  - Expose `GraphRunner.ready() -> list[JobSpec]` and `GraphRunner.complete(job_id, result_ref)`,
    plus `plan()`, which expands every node whose inputs are known and marks the rest deferred.
  - This makes pydra a pure dataflow compiler that senselab drives step by step.
  - It touches `engine/submitter.py` (the `expand_workflow_async` loop, `NodeExecution`),
    `engine/job.py` (decouple from `Submitter`) and `engine/lazy.py` (resolve through the store).
    Medium-high risk, but it is mostly a refactor along existing seams.

**Recommendation.** Start with route A to get running. Land sections 1, 4 and 5 upstream, and move
to route B once `ResultStore` exists. Route B is the generic piece most worth upstreaming, since
it lets any external scheduler (work pools, Ray, Parsl) drive pydra graphs.

## Summary: upstream vs senselab

| area | generic, so upstream in pydra | senselab-specific |
|---|---|---|
| workers | per-node worker routing; Slurm fixes, arrays; a shared-FS lease-queue worker as `pydra-workers-pool` | partition and QOS policy, controller tuning |
| environments | `Environment.checksum`; Python tasks in environments; a `Venv` env with a pluggable runner | `ensure_venv`, CUDA routing, lock grouping, worker-runtime codec |
| provenance | `ProvSink` from `post_run_task`; offline context | ProvStore schema, model and weights agents, BEP028 |
| resources | `ComputeResources` on task and node; worker mapping hook; cf GPU slots | resource classes, site profiles, hardware probe |
| cache | `ResultStore` (atomic rename, first writer wins); typed serializers; versioned layout | key policy via `__bytes_repr__` and the environment checksum |
| state | `Resident` field marker; `ResourceProvider` on workers; batching worker; no `chdir` | model cache, `serve_in_venv`, `VllmServer`, co-location |
| graph | `GraphRunner` and `Executor` split; `plan()`; per-element dataflow | the planner's placement decisions and `explain()` |
