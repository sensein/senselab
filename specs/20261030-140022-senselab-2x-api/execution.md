# senselab 2.x execution design: pools, GPUs, engines, and the pydra boundary

Status: draft. Refines `design.md` §4.4, §4.7 and §4.8. Inputs:
- [`workflow-engines.md`](workflow-engines.md): which engines meet which requirements;
- [`pydra-extension-points.md`](pydra-extension-points.md): where pydra can be extended, at
  pydra `c5081a6`, v1.0a12.

Principles: P-2, P-5, P-6, P-7, P-8, P-10, P-22, P-25.

## 1. The question

A single run can mix any of these:
- CPU-only nodes (DSP, decisions);
- GPU-only nodes (large ASR, VLMs);
- nodes served by a long-lived engine (vLLM, vLLM-omni);
- nodes in isolated environments, some on end-of-life Python;
- several execution substrates at once (a laptop, Slurm partitions, possibly Kubernetes or Ray).

The design has to say how these mix. It also has to say which parts are general workflow features,
which belong in pydra, and which are senselab's own.

## 2. Layers

Five layers, bottom up. Each layer only talks to the one above it.

| Layer | What it is | Generic? | Home |
|---|---|---|---|
| **L0 Environment** | A locked env (uv venv, or pixi for R/Java/native stacks) plus the worker runtime inside it. Identity is the lock sha. | yes | pydra: `Environment.checksum()` and a `Venv` environment with a pluggable runner. senselab: building envs, CUDA routing, the stdlib+numpy runner (`senselab-worker`) |
| **L1 Resident resource** | Expensive state that a worker process loads once and serves to many jobs: a model on a device, a vLLM server. Keyed by identity; LRU under a memory budget. | the concept is generic | pydra: a resource marker kept out of the hash, a provider registry on the worker, and batched dispatch. senselab: the model-cache provider, `serve_in_venv`, `VllmServer` |
| **L2 Pool** | A set of worker processes of one **resource class** pulling from a lease queue on shared storage. Workers run on any **substrate**. | yes | pydra: a pool worker plugin (`work_pool`'s lease protocol, contributed upstream or as `pydra-workers-pool`). senselab: substrate profiles for ORCD and others |
| **L3 Placement** | Assigns each job a resource class, env, device, batch and stack, with `explain()`. | the mechanism is generic; the policy is not | pydra: `ComputeResources` on a task or node, and routing a node to a worker. senselab: the placement policy (catalogue resource tables, license policy, stack choice) |
| **L4 Graph** | Nesting, typed ports, fan-out, routing, loops, the ready set. | yes | pydra (`design.md` §4.8), with senselab's fallback |

The rule for the boundary: **a feature goes upstream to pydra when it is meaningful for a task that
is not senselab's.** A loaded model is senselab's. "A worker keeps expensive state across jobs" is
not, so the hook goes to pydra and the model cache is senselab's provider for it.

## 3. How a mixed run works

```
                 plan (L3)                       shared store (result store + queues)
 Pipeline ──► jobs tagged {class, env,  ──►  queue/cpu        queue/gpu:a100#whisper@<sha>
 (L4)          resident identity, batch}      queue/gpu:any    queue/svc:vllm#<model>@<sha>
                                                 ▲                 ▲
          coordinator (small CPU job) ──releases ready jobs; collects results
                                                 │                 │
                    pool workers on substrates:  local procs   Slurm gpu partition   …
```

**Coordinator**
- The coordinator holds the graph (L4) and the plan (L3).
- It releases a job to its queue once the job's inputs exist in the result store, and marks
  descendants complete or typed-absent as results arrive.
- It is light, and on a cluster it runs as a small CPU job, never on a login node, as
  `reproduce` already does with its held pool job.
- It is restartable: all of its state is the result store plus the queue directories.

**Queues**
- There is one queue per **(resource class, resident identity)**:
  - `cpu`;
  - `gpu:a100` or `gpu:any`;
  - `mps`;
  - `svc:<engine>#<model>`, for a vLLM or vLLM-omni service.

  Splitting by resident identity is what makes affinity possible: a worker already holding
  Whisper claims Whisper items first.
- Items are plan fragments (job, input refs, env, tunables), never pickles of Python objects.

**Pool workers**
- A worker is the process `senselab execute --pool <dir> --classes <list>`, started inside the
  env the queue needs.
- It claims a **batch** of items from one queue (up to the planned batch size), runs them through
  its resident resource, writes results to the store by atomic rename, and renews its lease while
  it works.
- When a worker is preempted, it releases its claims. A dead worker's lease goes stale and is
  reaped.
- Items that keep failing go to `dead/` after their retry budget is used, and their descendants
  become typed failures.
- Healthy items keep going, with no barrier on sibling failures (pydra PR 1).

**Substrates**
- A **substrate** is anything that can start a worker process with a GPU, CPU or memory grant:
  - a local process (laptop, workstation);
  - a Slurm job, partition chosen by resource class, held or requeueable (the `work_pool`
    `Controller` does this today);
  - a Kubernetes Job, or a Ray task that starts the worker command.
- **"Other engines" therefore plug in as substrates, not as graph executors.** That is why mixing
  them is safe: the graph, the store and the queues are the same whichever engine starts the
  workers.

**Elastic sizing.** The controller scales each pool from its queue depth within the profile's caps,
including the QOS CPU and GPU caps (P-29), as `work_pool.Controller` does now.

### GPU-only and GPU-preferred tasks
- A backend declares `devices=["cuda"]` or `["cuda", "mps", "cpu"]`, plus its measured resource
  table.
- **GPU-only with no GPU class in the execution profile:** planning fails and names the node. It
  never silently falls back to CPU.
- **GPU-preferred:** the plan picks the best class available and records the choice, and the
  choice enters identity when the backend declares device class as output-affecting
  (`design.md` §4.8).
- **Several models on one GPU:** packing is planned from the measured memory table. The vLLM
  servers-per-card logic is the existing case.

### Services (vLLM, vLLM-omni)
- A service is a resident resource with its own lifecycle. It is started inside a pool worker of
  class `svc:<engine>#<model>` that serves requests from other workers over loopback or the
  cluster network.
- Its identity (engine version, model sha, engine config) is part of the identity of every
  result it serves.

### Mixing local and cluster
- Every pool must reach the same result store and queues, which means shared storage.
- A laptop together with ORCD therefore means the laptop runs CPU pools against a store on
  storage both can reach, or the laptop only plans and submits.
- Mixing without shared storage, for example through an object store, is a non-goal for 2.0.

## 4. Using pydra for the graph: two routes

The extension-point report establishes three facts:
- a pydra Submitter has exactly one worker (`engine/submitter.py:160`);
- the submitter learns a job finished only by reading `_result.pklz` from pydra's cache;
- expanding a graph into concrete jobs requires a Submitter.

That leaves two routes:

- **Route A (works today, with patches):**
  - senselab's coordinator *is* a pydra Submitter whose worker is a **pool worker plugin**. The
    plugin publishes each ready job to the queue for its class and identity, and completes it when
    the pool writes the result.
  - Per-node routing (`add(worker=...)`, or a router at the two dispatch sites) lets one
    workflow use several queues.
  - Requires: pydra PRs 1–4 plus worker routing.
  - Workers must either run pydra's job runner or write pydra's result format. Under PR 4
    (pluggable `ResultStore`) that format becomes senselab's store, so isolated envs need only the
    worker runtime, not pydra.
- **Route B (the clean end state, an upstream refactor):**
  - pydra's Submitter is split into a **graph runner** (`ready()`, `complete(job, result)`,
    `plan()`) and an **executor interface**.
  - senselab drives the graph runner from its own coordinator and executes through its pools.
  - pydra gains the same pluggability for every user.

*Decision:* build Route A first, then propose Route B upstream once `ResultStore` exists. Both
routes keep the coordinator, the queues and the pools senselab's (or the pool plugin's), so moving
from A to B changes no senselab user code.

Pydra changes are used from a pydra **branch** during development: a
`[tool.uv.sources] pydra = { git = "https://github.com/nipype/pydra", branch = "<branch>" }`
entry, with the resolved commit pinned in `uv.lock` (P-25). They are released by landing in a
pydra 1.0 alpha that senselab pins. They do not have to wait for a non-alpha release.

## 5. What goes into pydra, in order

| # | Upstream change | Needed for |
|---|---|---|
| 1 | Per-item dataflow, keep-going, error summary | Route A at scale |
| 2 | Cheap submission path (no re-pickle, memoised checksum, batched cache checks) | 10⁴–10⁶ items |
| 3 | Atomic rename into the cache, lease-based locks with stale recovery | shared NFS |
| 4 | `ResultStore` protocol with pickle-free serialisers; a cache layout versioned apart from pydra | cross-Python-version envs, shared caches |
| 5 | Worker routing per node (`add(worker=)` and a router) | mixed pools |
| 6 | `ComputeResources` on task and node, kept out of the hash | placement |
| 7 | `Environment.checksum()`; a `Venv` environment with a pluggable runner for Python tasks (#826, #840) | L0 |
| 8 | Resident-resource marker, worker provider registry, batched dispatch | L1 |
| 9 | Pool worker over a shared-filesystem lease queue; Slurm worker fixes (the `--error` crash, a requeue budget, polling all jobs in one call) | L2 |
| 10 | Provenance sink on `post_run_task` | P-3 |
| 11 | `plan()` printing (static graph, splits, expected hits) | P-10 |
| 12 | Graph-runner and executor split (Route B) | end state |

Items 1–5 gate Route A. Items 6–11 can be developed in senselab first and upstreamed when stable.
The design for these changes, written for pydra's users rather than senselab's, lives on the pydra
branch: [`proposals/0001-execution-extensions.md`](https://github.com/nipype/pydra/blob/enh/execution-extensions/proposals/0001-execution-extensions.md) on `nipype/pydra` branch `enh/execution-extensions`.

## 6. What stays senselab-only

- Env *building*: uv/pixi locks, CUDA wheel routing, `SystemRequirement` probes.
- The worker runtime's data codec for senselab types.
- The model-cache provider, `serve_in_venv` and `VllmServer`.
- The cache-key *policy* (what enters identity), supplied to pydra through its hashing hooks.
- The `ProvStore` schema and BEP028 export.
- Placement *policy*: catalogue resource tables, license policy, stack preferences.
- Site profiles (partitions, QOS caps, roots).
