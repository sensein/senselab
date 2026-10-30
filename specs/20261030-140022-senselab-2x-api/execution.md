# senselab 2.x execution design: pools, GPUs, substrates, and the pydra boundary

Status: draft, revised after review. This document refines `design.md` §4.4, §4.7, §4.8 and §4.9.
Identity rules (the canonical key, I-1..I-9) are in `design.md` §4.8 and are not repeated here.
The case for pydra is in [`engine-choice.md`](engine-choice.md). pydra changes are cited by their
ids in
[proposal 0001](https://github.com/nipype/pydra/blob/enh/execution-extensions/proposals/0001-execution-extensions.md).
Extension points: [`pydra-extension-points.md`](pydra-extension-points.md).

Principles: P-2, P-5, P-6, P-7, P-8, P-10, P-22, P-25, P-34.

## 1. The question

A single run can mix:
- CPU nodes and GPU-only nodes;
- nodes served by a long-lived engine (vLLM, vLLM-omni);
- nodes in isolated envs, some on end-of-life Python;
- local and Slurm substrates.

This document says how they mix, and which parts are generic (pydra) and which are senselab's own.

## 2. Layers

| Layer | What it is | Home |
|---|---|---|
| **L0 Environment** | A locked env (uv venv, or pixi for R/Java/native stacks) with the worker runtime as an overlay. Numerics and protocol identities follow I-2. | pydra: `Environment.checksum()` (G1), `Venv` + runner (G2). senselab: env building, CUDA routing, `senselab_worker` |
| **L1 Resident resource** | State a worker loads once and serves to many jobs: a model on a device, or an engine service. Keyed by identity, with LRU under a budget. | pydra: `provided=` objects + batching (H0, H). senselab: the model-cache provider, `serve_in_venv`, `VllmServer` |
| **L2 Pool** | Worker processes of one resource class, pulling from a lease queue on shared storage, started by a substrate. | pydra plugin `pydra-workers-pool` (I). senselab: site profiles |
| **L3 Placement** | Assigns each job a resource class, env, device, batch and stack, with `explain()`. | pydra: `ComputeResources` (F1), routing (F2). senselab: placement policy |
| **L4 Graph** | Nesting, ports, fan-out, routing, loops, the ready set. | pydra, Route A (section 4), with a fallback (`design.md` §4.9) |

Boundary rule: a feature goes to pydra when it means something for a task that is not senselab's.

## 3. A mixed run

```
 Plan (immutable, hashed) ──► coordinator ──► queue/cpu   queue/gpu:a100#<model id>   queue/svc:<service id>
                                 ▲  lease + fencing token        │                          │
                                 └──── results (store) ◄── pool workers on local / Slurm substrates
```

### 3.1 Coordinator
- **What it holds:** the plan and the graph. It releases a job to its queue when the job's inputs
  exist in the store, and marks descendants complete, typed-absent or failed.
- **The Plan is immutable and hashed** and includes every resolution (model SHAs, catalogue
  defaults, hardware probe, env identities, chunk grid). A restart is `resume(plan_hash)`. A
  restart never re-resolves `@main`, re-probes hardware or re-plans.
- **A single coordinator** holds a lease with a **fencing token**. The store and queues reject
  releases that carry a stale token, so a preempted coordinator that is still alive cannot act
  next to its replacement.
- **Route choices and loop rounds are stored as results,** so a restart replays them rather than
  recomputing them.
- **It runs as a small CPU job,** never on a login node.

### 3.2 Queues and workers
- **Queues:** one per (resource class, resident identity): `cpu`, `gpu:<type>`, `mps`, `svc:<id>`.
  Items are plan fragments (job, input refs, env, tunables), never pickled objects.
- **Workers:** `senselab execute --pool <dir> --classes <list>` runs inside the env the queue
  needs. A worker claims a batch from one queue, runs it, commits per-item results by atomic
  rename, and renews its lease.
- **Batches return a result per item.** A failing batch is bisected so one item cannot fail its
  batch-mates.

### 3.3 Fault taxonomy
| Class | Examples | Effect |
|---|---|---|
| content | decoder error on this input, invalid params | charges the item's retry budget; after the budget, the item goes to `dead/` and its descendants become typed failures |
| infrastructure | handshake refusal (version skew), lost service, preemption, storage error | the worker stops claiming; the item is re-placed without charging its budget; the plan reports the fault |
| resource | OOM, timeout from contention | re-placed with a larger grant or a smaller batch; not charged |

When one env or one worker is broken, the fault is reported against the env or worker, never
spread across a million items.

### 3.4 Scheduling and memory
- **A resident model is a cache, not a reservation.** A worker yields after a bounded quantum
  when other queues are waiting. Every non-empty queue is served within a bound, so starvation is
  impossible.
- **The plan checks** that the caps cover the models that must be resident at the same time.
- **Device memory goes through an admission arbiter** on each device. The arbiter uses
  peak-memory-against-input-length tables from the catalogue. Packing several models on one card,
  including vLLM servers' utilisation, is admitted by the arbiter and not assumed from a static
  table.
- **A GPU-only task with no GPU class** in the execution profile fails planning and names the
  node. It never falls back to CPU silently.

### 3.5 Services
- **What a service is:** a resident resource in a pool worker of class `svc`. Its process runs in
  an `Env.server` env.
- **Registration:** services are endpoints registered in the store with leases, and addressed by
  identity.
- **Lifecycle:** a service stays alive while the plan has pending consumers for it; queue depth
  does not decide that.
- **Identity:** every response carries the service's attested identity (I-8).

### 3.6 Substrates
- A substrate starts pool workers with a CPU, GPU or memory grant: local processes, or Slurm jobs
  (`work_pool.Controller`, with the partition chosen by resource class).
- The controller scales pools from queue depth within the profile's caps, including the QOS caps
  (P-29).
- Kubernetes and Ray launchers are post-2.0. The design admits them as substrates, never as graph
  executors.
- **Storage capability probe.** The lease protocol needs atomic rename, exclusive `link(2)` and
  the server's clock. These are probed as a `SystemRequirement` at plan time, and a store that
  fails the probe is refused (sshfs, SMB and some object-store mounts).
- **Mixed placement** of laptop and cluster pools requires a store both can reach; otherwise the
  laptop plans and submits only.

### 3.7 Result store at scale
- **Small results go to a packed tier:** columnar files per shard plus a key index. This avoids
  10⁸ inodes at `over="segment"` scale.
- **Expected cache hits** in a plan are estimated from the index, not by stat-ing every key.
- The store has its own per-item budgets for files and stats (P-33).

## 4. pydra for the graph

What the extension-point report established:
- a Submitter has one worker (`engine/submitter.py:160`);
- completion is read from the cache;
- expanding a graph into jobs needs a Submitter.

**Route A, built first:**
- The coordinator is a pydra Submitter.
- A pool worker plugin (I) publishes ready jobs to the queues and completes them from the store.
  Routing (F2) and resources (F1) map nodes to queues.
- The prerequisites are E1 (per-element readiness), D1 (`ResultStore` interface) and F2.
- After D1/D2, completion reads senselab's store, and pydra's pickle cache is not used.
- Isolated envs need only the worker runtime, not pydra.

**Route B, the end state:** pydra's graph runner (`ready`, `complete`, `plan`) separated from
execution, proposed as RFC 3.10 once E and D land. Moving from A to B changes no senselab user
code.

**Delivery:** used from the branch through `[tool.uv.sources]`, with the commit pinned in
`uv.lock`, and released in 1.0 alphas.

**Decision gate at the start of Phase 3:** if E1, D1 and F2 are not working on the branch, build
the fallback graph layer behind the same `Executor` protocol (`design.md` §4.9).

| pydra id | Change | Needed for |
|---|---|---|
| E1, E2 | per-element dataflow, keep-going | Route A at scale |
| B1, B2 | benchmarks, cheap submission | 10⁴–10⁶ items |
| D1, C, D2 | store interface, leases + atomic commit, `PortableStore` | shared NFS; envs on different Python versions |
| F1, F2 | `ComputeResources`, routing | mixed pools |
| G1, G2 | env checksum, `Venv` + runner | L0 |
| H0, H | no-`chdir` option, provided objects + batching | L1 |
| I | `pydra-workers-pool` | L2 |
| A0, A1, A2 | Slurm fixes, budget, arrays | Slurm |
| J1, J2 | provenance via messengers, `plan()` | P-3, P-10 |

## 5. What stays senselab-only

- env building (uv/pixi locks, CUDA routing, `SystemRequirement` probes);
- the worker runtime codec for senselab types;
- the model-cache provider, `serve_in_venv` and `VllmServer`;
- the cache-key policy (`design.md` §4.8), supplied through pydra's hashing hooks;
- the ProvStore schema and BEP028 export;
- the placement policy and the memory admission tables;
- site profiles.
