# Workflow engines for senselab 2.x: an evaluation

Date 2026-10-30. Scope: design §4.4, 4.5, 4.7, 4.9, 4.11 and 4.12 of
`specs/20261030-140022-senselab-2x-api/design.md`, measured against requirements R1–R9.
Legend: **N** native, **P** plugin or extension, **W** feasible with work, **–** no.
Facts about pydra come from source at `nipype/pydra@c5081a6` (tag v1.0a12) and from a local
measurement. Other facts come from docs and PyPI, cited inline. Rows marked † were not re-checked
against a primary source in this pass.

## 1. Matrix

| | R1 cache/keys/NFS | R2 nest/dyn/route/loop/ports | R3 per-node venv, mixed Python | R4 resident GPU actor + batching | R5 Slurm + macOS | R6 overhead | R7 debug | R8 prov / light / lic | R9 maturity |
|---|---|---|---|---|---|---|---|---|---|
| **pydra 1.0a12** | N keys via `__bytes_repr__`; W NFS (pickles, soft locks) | N nest, split, lazy; routing and loops via nested workflows; typed ports N | – for Python tasks (#840); shell via Native/Docker/Singularity/lmod | – | W (one sbatch per job, polling, no arrays); macOS N | measured 4–5 ms/item cold, 2–3 ms/item cached check | N debug worker | P (audit/JSON-LD, unfinished #798); light (7 deps); Apache-2.0 | alpha; one main committer |
| **joblib** 1.6.0 | N (function code + args; no caller key; no lock) | – | – | – | – (local only) | low per batch † | N `n_jobs=1` | – / tiny / BSD-3 | stable, slow cadence |
| **Dask** 2026.8 + jobqueue 0.9.0 | – | W (futures, `submit` from tasks) | – (workers must match client versions †) | W (Actors, no fault tolerance) | P dask-jobqueue (no release since 2024-08); macOS N | 200 µs–1 ms/task ([docs](https://docs.dask.org/en/stable/best-practices.html)) | N `scheduler="synchronous"` | – / medium / BSD-3 | strong core; jobqueue stale |
| **Ray** 2.59 | – | W (dynamic tasks) | P `runtime_env` (uv alpha, `py_executable` experimental); Python must match the cluster | **N** actors, Serve `@serve.batch` | W community Slurm guide; macOS single-node only | sub-ms † | weak (`local_mode` discouraged) | – / heavy / Apache-2.0 | Anyscale, fast cadence |
| **Parsl** 2026.10.26 | N memo + checkpoint; `id_for_memo` per type; NFS safety undocumented | N futures (dynamic, nesting via join apps) | W (`worker_init` per executor) | W (module-global model in an HTEX worker) | **N** SlurmProvider + HTEX pilots; macOS N | ~1k+ tasks/s † | N ThreadPool | P monitoring DB / medium / Apache-2.0 | NSF-funded, weekly |
| **Prefect 3** 3.9.1 | N `cache_key_fn`, `cache_policy`; FS/Redis locks | N (plain Python) | – | – | – (only via Dask/Ray runners) | per-task API state write | P | P events / heavy (server, DB) / Apache-2.0 | 2→3 was a hard break |
| **Hamilton** 1.90 (Apache, incubating) | P caching (SQLite metadata †) | P Parallelizable/Collect, `config.when`; no loops | – | – | – (via Dask/Ray) | low (in-process) | N | P / light / Apache-2.0 | package renamed in Apache move |
| **submitit** 1.5.4 | – | – | P (`python=` per executor †) | – | **N** arrays, `checkpoint()` requeue †; local executor | one job per item | N DebugExecutor † | – / tiny / MIT † | one release in 12 months |
| Dagster, Snakemake, Nextflow, Flyte, Metaflow | dropped, see §2 | | | | | | | | |

## 2. Per-engine notes and failure modes

**Dropped after a quick check:**
- *Dagster* is an asset and partition platform with instance storage and a daemon. It has no
  Slurm support and no per-node envs.
- *Snakemake* 9.x is file-centric and builds its DAG slowly at 10^5+ jobs. Per-rule envs are
  conda/apptainer, not uv venvs, and it cannot run inline in Python.
- *Nextflow* is the JVM plus a work directory per task. senselab already ran it (below).
- *Flyte*: containers are effectively mandatory, flytekit lacks 3.13, and the open-source Flyte 2
  backend is "not yet ready for production"
  ([release notes](https://www.union.ai/docs/v2/byoc/release-notes/)).
- *Metaflow* has a step-granular, process-per-task model. It has no content-addressed cache, and
  its `@slurm` is a thin SSH extension ([pypi](https://pypi.org/project/metaflow-slurm)).

**Nextflow: senselab already measured it.** `specs/20260817-triage-workflow-dag/nextflow/README.md`
ran Nextflow 26.04.6 over the v1 triage graph.
- What worked: stub runs, `-resume` (`cached=18`), single-node reruns and a 40-hex revision gate.
- Over-invalidation: changing HeAR's revision recomputed **14 of 18** tasks. Three of those loaded
  no HeAR, because the task hash covers whole staged files rather than the elements a node reads.
- Parsing and concurrency: the strict parser dropped `-entry`, and the read-anything store
  serialised the branches.
- Why it was superseded: the v2 node design changed under it, not because of a Nextflow defect.
  The lessons carry over to every engine. Key on what a node actually reads, keep model identity
  in the key, and let the runner stay Python so that a node is debuggable.

**joblib.** `Memory` hashes the function code plus its arguments. The caller cannot supply a key,
and there is no lock, only atomic-ish writes. Parallel use has produced a corrupted cache
([#490](https://github.com/joblib/joblib/issues/490)). It is useful as a hashing idea and as
`loky`, not as an engine.

**Dask.**
- Fine-grained scheduling is good up to about 10^5 tasks. The docs warn that "millions of tasks"
  overwhelm the scheduler ([best practices](https://docs.dask.org/en/stable/best-practices.html)).
- Actors: "No effort is made to make actor workloads resilient to worker failure". They have no
  diagnostics, and a worker holding an actor cannot be retired
  ([actors](https://distributed.dask.org/en/stable/actors.html)). That rules them out for
  preemptible GPU nodes.
- dask-jobqueue's last release was 0.9.0 in Aug 2024, so it is a bus-factor risk.
- Environments: every worker must import the client's pickled code, so mixed-Python envs are
  impossible.

**Ray.** Ray has the best actor model (resident model, Serve batching) and fits R4 exactly. It
fails R3, R5 and R8:
- **R3:** `runtime_env` requires "the Ray and Python versions of your uv environment must be the
  same as the Ray and Python versions of your cluster". `uv` is alpha and `py_executable` is
  experimental ([deps](https://docs.ray.io/en/latest/ray-core/handling-dependencies.html)).
  senselab's 3.10 envs (child-adult, unasdiff) and its TF envs cannot join one Ray cluster.
- **R5:** Slurm is a community guide; symmetric-run "does not currently work in multi-tenant
  environments" and ports are set by hand
  ([guide](https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html)). Multi-node
  on macOS is untested.
- **R8:** a large wheel plus a GCS process. It cannot be an optional dependency of a torch-free core
  without dominating it.

**Parsl.** This is the closest ready Slurm layer. HTEX pilot jobs give the queue a worker count
instead of an item count, which is what `work_pool` was built for. Weaknesses:
- Memoization keys on the function *name* plus a hash of the arguments. The function body is not
  in the key, and unregistered types raise unless `id_for_memo` is registered for them.
- Checkpoints live in `runinfo/RUN_ID/checkpoint/` with no statement about shared filesystems.
  Concurrent identical calls "will be launched" many times
  ([checkpoints](https://parsl.readthedocs.io/en/stable/userguide/workflows/checkpoints.html)).
- It does no requeue-aware lease reaping comparable to `work_pool`'s, has no mixed-Python workers
  within one executor, and its interchange process is a central point (throughput †).

**Prefect 3.**
- Caching: the cache design is the best-documented of the field. `cache_key_fn`, composable
  `cache_policy`, `SERIALIZABLE` isolation with a `FileSystemLockManager` or `RedisLockManager`
  ([caching](https://www.prefect.io/docs/v3/concepts/caching)).
- Bookkeeping: every task run is a state write to an API server, which is wrong at 10^6 items and
  heavy for a lean core.
- Slurm: none native.

**Hamilton.** A light in-process function DAG. It has no isolation and no scheduler, and its
loops and runtime routing are limited. It brings nothing that senselab's own `Pipeline` would not
already define.

**submitit.** A clean way to submit arrays with requeue-on-preemption. It is in maintenance mode
(one release in 12 months). An item becomes an array element: arrays are better than one job
per item but still hit MaxSubmitPU, which is exactly the r18d problem `slice_pool` solved.

**pydra 1.0a12 (inspected and measured).**

*Stability:*
- 1.0 is still alpha: v1.0a0 through v1.0a12 between Apr 2025 and Oct 2026. `pip install pydra`
  still resolves to 0.25 from 2024.
- Since 2025-10-30, 95 of about 135 human commits are by one author (tclose). The bus factor is
  one active developer plus reviewers.
- 94 open issues and PRs. Several are directly relevant:
  - environment not in checksum, #826;
  - cache dirs versioned with pydra rather than independently, #825;
  - Python tasks cannot run in environments, #840;
  - per-task Slurm resources, #227 and #830;
  - provenance unfinished, #798;
  - stray lock files, #778;
  - start-up latency, #802;
  - a cf deadlock under forkserver on 3.14, #876.

*Hashing and caching:*
- A job's directory is `cache_root/<tasktype>-<blake2b-128>` over every input field's
  `bytes_repr`, the Outputs class, and the function's AST.
- Caller-controlled keys are native: a type with `__bytes_repr__` or a registered serializer (for
  example a `ResolvedModel` yielding its commit SHA) controls its contribution.
- Weaknesses:
  - The task runs *inside* its final cache dir under a `SoftFileLock` held for the whole run. A
    SIGKILLed or preempted job leaves a lock that blocks every later run until it is removed by
    hand. The Slurm worker deletes it only on its own requeue path.
  - Results are written non-atomically and readers retry on `UnpicklingError`
    (`engine/result.py`).
  - Results are **cloudpickles**, which ties them to one Python and library set and makes them
    unreadable from a 3.10 worker env. On a cache shared across users, `pickle.load` of another
    user's file is code execution.

*Execution, measured (§ script):*

| scenario | cold | warm, same inputs | +10% new items | fs entries/item |
|---|---|---|---|---|
| split of a trivial task, debug worker, n=1000 | 4.2 ms/item | 0.01 ms/item (whole-split hit) | 2.6 s (100 new + 1000 cached) | 3.3 |
| same, n=5000 | 5.4 ms/item | ~0 | 18.4 s | 3.3 |
| same, cf worker 4 procs, n=1000 | 4.1 ms/item | ~0 | 2.8 s | 3.3 |
| 2-node workflow split per item, debug, n=200 | 23 ms/item | ~0 | 1.2 s | 9.9 |
| same, cf 4 procs | 19 ms/item | ~0 | 1.3 s | 9.9 |

What the measurements say:
- **Resume cost.** Resuming after adding items costs about 2–3 ms per *already cached* item. At
  10^6 items that is about 50 min of main-process work before anything runs.
- **The submitter is the bottleneck.** cf with 4 processes is no faster than inline. cProfile
  attributes about 45% of wall time to cloudpickle: `Job.__getstate__` re-pickles the task, and
  every job writes a ~14 KB `_job.pklz` and a ~20 KB `_result.pklz` for an int.
- **Storage.** That is ~34 KB and 3.3 inodes per trivial item, so 10^6 items is ~34 GB and
  3.3 M inodes on NFS.
- **Machine load.** Load average was 3.5–5 on 10 cores during the runs, so treat these numbers as
  upper bounds.

*Scheduling semantics, verified by experiment:*
- A downstream node waits for **all** states of its predecessors (`submitter.py:853`,
  `if True:  # self.node.splitter:`). One failed element makes the whole downstream node
  unrunnable.
- With cf, an 8-element split with one failure ran all 8 upstream jobs and **zero** of the 7
  healthy downstream jobs. The debug worker raises on the first failure.
- For a corpus of 10^4 recordings this is a blocking defect: one bad file stops every record's
  next stage.

*Slurm worker (`workers/slurm.py`, 180 lines):*
- Submission: one `sbatch` per job, then `squeue -j` every `poll_delay=1` s per job. There are no
  arrays (the SGE worker has them) and no pilots, and `sbatch_args` is worker-global, so there are
  no per-task partition or GPU settings.
- Interpreter: it runs `sys.executable` from the submitting host.
- Requeue: it calls `scontrol requeue` on CANCELLED/TIMEOUT/PREEMPTED **unconditionally and without
  a budget** unless `--no-requeue` is passed. A user's `scancel` is requeued, and a task that always
  hits its time limit loops forever.
- Bug: passing `-e/--error` crashes at line 118 (`error_file` is `None`, then `.replace` is called
  on it).

*Debuggability:* the debug worker gives real stack traces. `pydracli crash --rerun` reruns a node.
Lazy outputs make dynamic parts unplannable before execution: routing is by nested workflow
constructed at run time, so "plan printable before execution" holds only for the static shell.

*Provenance:* `Audit` emits JSON-LD PROV messages, and `TaskHooks` gives
`pre_run`/`post_run`/`pre_run_task`/`post_run_task`. The hooks are a usable seam; the audit is
self-declared unfinished (#798).

## 3. What senselab already has, and whether to keep it

| senselab piece | what no engine above gives | keep? |
|---|---|---|
| `utils/work_pool.py` (1,725 lines, stdlib) | race-free leases on NFS v4.2, measured: 400/400 single winners, 2000/2000 done with 0 double, 6 SIGKILLed lease holders recovered. Heartbeats, a stale reaper, release-on-SIGTERM with an interrupt budget, a Controller that sizes workers per partition and skips QOS-pending partitions | **Keep.** Only Parsl HTEX approaches it, without NFS-proven semantics. It is the right substrate for `Pool`/`Slurm` executors |
| `venv_worker.serve_in_venv` | persistent worker in a different-Python venv, JSON control line, one worker per identity, replace-on-death | **Keep and grow.** No engine runs a 3.10 TF env and a 3.13 torch env in one graph. This is R3 and R4 |
| `vllm_server` | servers per card, start lock, engine identity, refused content-logging flags | **Keep.** A shared resource no engine models |
| `reproduce/` (`dag.py` toposort with deterministic tie-break, `slurm.py` held `afterok` chains and `choose_partition`, spec, locks, weights, provenance) | pinned-commit, sandboxed, weights-stamped external runs; lock-sha env identity | **Keep as a consumer** of the store (design §4.8). Merge `dag.py` into the planner rather than keep a second DAG |
| `ensure_venv` + CUDA routing + `uv_tools` | CUDA-aware wheel-index routing, requirements-sha | **Keep.** Ray `runtime_env` and Snakemake conda do not do this |
| `CACHE_SCHEMA_VERSION`, commit-aware keys | caller-defined identity incl. commit and env lock | **Keep the policy.** Every engine needs it bolted on |

## 4. Recommendation

**No engine should be adopted as the execution layer.** The three hardest requirements are mixed
Python versions per node (R3), a resident model batched across records (R4), and pilots on NFS
with preemption (R5). Each engine is native on at most one of them:
- Ray: R4 only.
- Parsl: R5 only.
- Every candidate: none of them does R3 without containers.

senselab already owns working, measured answers to R3, R4 and R5 (`venv_worker`, `vllm_server`,
`work_pool`). Adopting an engine would mean wrapping those anyway, while inheriting its cache
format, its pickling and its failure semantics.

**What senselab must own regardless.** The candidates were planner, model cache and env manager;
each is tested here.

1. **Content-addressed result store** (§4.8). Own it. No engine lets the key be the canonical key
   (`design.md` §4.8) *and* writes
   atomically by rename on NFS *and* stores Python-version-neutral data (`.npy`/JSON/parquet rather
   than pickles).
   - Prefect has the key model but needs a server.
   - pydra has the key model but stores pickles non-atomically.
2. **Env manager and worker runtime** (§4.4, §4.12). Own it. This is senselab's differentiator.
3. **Model cache and resident workers** (§4.4). Own them. Batching across records is a
   planner-to-worker contract (collect inputs for one resident identity), not a scheduler feature.
4. **Planner.** Own the *placement* part: backend, env, device class, batch, partition and
   `explain()`. These are senselab facts that no engine models.
   - *Challenge:* the *graph* part (nesting, typed ports, `over=` fan-out, route, bounded loop, a
     dependency-ordered ready set) is generic and is exactly what pydra already implements.
   - Writing it a third time, after triage's runner and `reproduce/dag.py`, is the risk.
5. **Executors.** `InProcess` and `Pool`/`Slurm` on `work_pool`, as the design says. A `Local`
   process pool adds little over `work_pool` with a local controller; consider dropping it.

**Which ready layer to start on.** If one is used at all, use **pydra for the graph layer only.**
- Use its compose/lazy/state/split-combine and typed ports for `sl.Pipeline`, behind senselab's
  own `Executor` protocol and result store.
- Do not use its pickle cache once the store interface lands. Route A (`execution.md` §4) uses a
  pool worker plugin and pydra's environment identity.
- This is viable because the owner leads pydra. It is conditional on the proposal's E1, D1 and F2
  working on the branch, which senselab uses through `uv` git sources; a non-alpha release is not
  required. Without them, pydra's barrier semantics and per-item cost are disqualifying at 10^4+
  records.
- **Fallback** (decided at the start of senselab Phase 3): write the graph layer in senselab,
  2–4k lines. Reuse `reproduce/dag.py`'s deterministic toposort and
  `audio_analysis/contracts.py:991` `unrolled_contracts` for loops.

Do not adopt Ray, even for R4. A Ray actor cannot host a 3.10 or TF env, and `serve_in_venv` plus
the planner's co-location already gives a resident model per identity. Revisit Ray only if
multi-node tensor-parallel inference becomes a requirement (vLLM already covers single-node).

## 5. pydra changes that would close the gaps

This evaluation first listed nine changes. A maintainer review then revised and re-split them in
[proposal 0001](https://github.com/nipype/pydra/blob/enh/execution-extensions/proposals/0001-execution-extensions.md),
which is now the only numbering. The original items map to it as follows:

| Original item | Proposal ids |
|---|---|
| per-element dataflow, keep-going | E1, E2 |
| cheap submission | B1, B2 |
| atomic, lock-light cache | D1, C |
| pluggable, pickle-free results | D1, D2 |
| environment identity, Python tasks in environments | G1, G2 |
| resources, per-task Slurm settings | F1 (+ F2 routing) |
| Slurm fixes, arrays, pool | A0, A1, A2, I |
| provenance | J1 |
| plan printing | J2 |

The gate for senselab is E1, D1 and F2 working on the branch (`design.md` §4.9).

## Appendix: measurement scripts

Laptop: 10 cores, macOS, Python 3.12.11. The venv was created with `uv venv -p 3.12 bench-venv`
and `uv pip install ./pydra psutil` from pydra `c5081a6`. Scripts are under
`bench/` (next to this file):
- `pydra_overhead.py`: run as `split:debug:100 split:debug:1000 chain:debug:200 split:cf:1000
  split:debug:5000 chain:cf:200`. It times a cold run, a fresh identical task object (warm), and a
  110%-length split.
- `pydra_partial_failure.py`: one failing element of eight, then a downstream node.
