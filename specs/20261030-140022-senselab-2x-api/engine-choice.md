# Why senselab extends pydra: requirements against workflow engines

Status: draft, 2026-10-30. Evidence and citations:
- [`workflow-engines.md`](workflow-engines.md): the R1–R9 matrix, per-engine notes, and the
  Nextflow prototype measurement;
- [`pydra-extension-points.md`](pydra-extension-points.md);
- pydra [proposal 0001](https://github.com/nipype/pydra/blob/enh/execution-extensions/proposals/0001-execution-extensions.md),
  cited as §3.n.

The pydra facts were measured or read at `nipype/pydra@c5081a6` (v1.0a12). Facts about other
engines come from their documentation; cells carried over with a † in `workflow-engines.md` were
not re-checked against a primary source.

## 1. Requirements

Each requirement traces to a principle in [`principles.md`](principles.md).

| # | Requirement | Why senselab needs it | Principles |
|---|---|---|---|
| G1 | **Declarative graph**, inspectable and printable before running, serialisable as data | plans are printed and diffed before execution; agents read and edit pipelines as data | P-1, P-9, P-10 |
| G2 | Nesting, **typed ports**, runtime fan-out (split/combine), conditional routing, bounded loops | nested pipelines; fan-out over segments and speakers; triage routing; adaptive rounds | P-9 |
| G3 | Content-addressed cache with **caller-controlled keys** (model sha, env lock sha, params) | results are traceable to model bytes; cheap invalidation | P-3, P-4, P-19 |
| G4 | Cache that is safe on **shared NFS** and holds results in a **Python-version-neutral** format | multi-user cluster caches; envs on different Pythons | P-5, P-22 |
| G5 | **Per-node isolated envs with mixed Python**, containers optional | 29 locked envs on Python 3.10–3.13 | P-6, P-22, P-25 |
| G6 | **Resident model** per worker plus **batching** across items | GPU models loaded once and fed many recordings | P-5, P-7 |
| G7 | **Heterogeneous resources and routing** within one run (GPU, CPU, service nodes) | GPU ASR, CPU DSP and a vLLM service in one pipeline | P-7, P-10 |
| G8 | **Slurm pools** with preemption and requeue, and the **same definition on a laptop** (macOS) | ORCD at scale; local development | P-5, P-10 |
| G9 | **Low per-item overhead** at 10⁴–10⁶ items | corpus runs | P-5 |
| G10 | **Partial-failure isolation** (keep going) | one bad recording must not stall a corpus | P-5 |
| G11 | **Inline debugging**: one process, real stack traces, rerun one node | development and diagnosis | P-1 |
| G12 | **Provenance hook** per task | PROV/BEP028 records | P-3 |
| G13 | **Light, torch-free dependency**, Python ≥3.11, permissive license | the graph layer must sit inside core | P-13, P-23 |
| G14 | **Governance**: senselab can land changes and release on its own schedule | the gaps below have to be closed somewhere | P-27 |

## 2. Comparison

Legend:
- **●** meets the requirement today.
- **◐** partial, or needs work. For pydra the cell names the proposal section that closes the
  gap, which senselab can land itself.
- **○** does not meet it, and the gap follows from the engine's architecture or lies outside
  senselab's control.

| | pydra 1.0a12 | Parsl | Ray | Dask | Prefect 3 | Snakemake | Nextflow | Hamilton |
|---|---|---|---|---|---|---|---|---|
| G1 declarative graph | **●** static workflow, typed definitions | ○ futures; the graph unfolds as code runs | ◐ `.bind()` DAG API; usual use is imperative | ◐ `delayed` graphs are static; futures are not | ○ plain Python, no static DAG | ● rule DAG, dry-run | ◐ channels; stub runs | ● function DAG |
| G2 nest/ports/fan-out/route/loop | **●** nesting, split/combine, lazy typed fields; loops via nested workflows | ◐ dynamic, untyped | ◐ dynamic, untyped | ◐ untyped | ◐ untyped | ◐ file-typed; no loops | ◐ channel-typed | ◐ no loops |
| G3 caller-controlled keys | **●** `__bytes_repr__` hashing | ● `id_for_memo` | ○ | ○ | ● `cache_key_fn` | ◐ files and params | ◐ staged-file hash; changing one model recomputed 14 of 18 tasks (measured in senselab) | ◐ |
| G4 NFS-safe, portable results | ◐ pickles and soft locks → §3.3, §3.4 | ◐ NFS safety undocumented | ○ | ○ | ◐ FS/Redis locks | ● files | ● work directories | ◐ |
| G5 per-node venv, mixed Python | ◐ shell tasks have envs; Python tasks do not → §3.7 | ◐ `worker_init` per executor | ○ Python and Ray versions must match the cluster | ○ workers match client versions | ○ | ◐ conda/apptainer per rule, not uv | ◐ conda/containers per process | ○ |
| G6 resident model + batching | ◐ → §3.8 | ◐ module-global state in a worker | **●** actors, `@serve.batch` | ◐ actors without fault tolerance | ○ | ○ process per job | ○ process per task | ○ |
| G7 heterogeneous routing | ◐ one worker per Submitter → §3.5, §3.6 | ● per-app executors | ● per-task resources | ◐ worker resources | ◐ via runners | ● per-rule resources | ● per-process executors | ○ |
| G8 Slurm pools + laptop | ◐ one sbatch per job; crashes on `--error` → §3.9 | **●** HTEX pilots on SlurmProvider | ◐ community Slurm guide; macOS single-node | ◐ dask-jobqueue, no release since 2024-08 | ○ | ● executor plugin | ● | ○ |
| G9 per-item overhead | ◐ 4–5 ms cold, 2–3 ms cached (measured upper bound) → §3.2 targets <0.5 ms | ● | ● | ● 0.2–1 ms | ○ API state write per task | ○ DAG build slows at 10⁵+ jobs | ○ work directory per task | ● in-process |
| G10 keep-going | ○ today: 7 of 8 healthy items blocked (measured) → §3.1 | ● | ● | ● | ● | ● `--keep-going` | ◐ `errorStrategy` | ◐ |
| G11 inline debugging | **●** debug worker | ● ThreadPool | ○ `local_mode` discouraged | ● synchronous scheduler | ◐ | ○ not inline Python | ○ JVM | ● |
| G12 provenance hook | ◐ `post_run_task` hook exists; audit unfinished → §3.10 | ◐ monitoring DB | ○ | ○ | ◐ events | ◐ reports | ◐ trace and lineage | ◐ |
| G13 light, permissive | **●** 7 dependencies, Apache-2.0 | ◐ medium | ○ heavy | ◐ medium | ○ server and database | ◐ | ○ JVM | ● |
| G14 governance | **●** the owner leads pydra; changes ship in 1.0 alphas | ○ external | ○ external | ○ external | ○ external | ○ external | ○ external | ○ external |

Not shown: joblib (in-process memoisation, local only), submitit (a Slurm submission library),
and Dagster, Flyte and Metaflow (dropped; reasons in `workflow-engines.md` §2).

## 3. Reading the table

**Where pydra is strong.** pydra meets the requirements that are hardest to build and hardest to
change after the fact: G1 (a declarative, typed graph), G2 (nesting with split/combine), G3
(caller-controlled hashing), G11 (inline debugging), G13 (light and permissive) and G14
(governance). These are semantics. Rebuilding them means writing a third graph runner, after
triage's runner and `reproduce/dag.py`.

**pydra's gaps are all in execution, which senselab owns anyway.** Every ◐ or ○ in the pydra
column (G4–G10, G12) is an execution-layer gap: cache storage, environments, resident workers,
routing, Slurm, overhead, failure handling. `execution.md` has senselab building exactly these
layers regardless of the engine (L0 env, L1 resident resource, L2 pool). Extending pydra lets
that work land once, where every pydra user benefits, instead of wrapping another engine's
execution model.

**Other engines' gaps are mostly architectural, so senselab could not fix them.**
- **Parsl** is the strongest alternative on execution (G7–G10). Its graph, though, is a set of
  futures that unfolds as code runs (G1 ○). The planner (P-10) and the agent surface (P-1) need a
  graph that can be inspected, planned and edited before anything runs. That is Parsl's core
  model, not a missing feature.
- **Ray** is best at G6 (actors and batching). It requires the Python and Ray versions to match
  across the cluster (G5 ○), which excludes envs on different Pythons, and it is heavy (G13 ○).
- **Dask** cannot host mixed-Python workers (G5 ○), and its Slurm adapter is stale.
- **Prefect** needs a server and database, and writes API state per task (G9, G13 ○).
- **Snakemake and Nextflow** are file- and process-centric. That gives them process-per-task
  overhead, no resident models, and no inline Python debugging. senselab's own Nextflow prototype
  measured the over-invalidation that file-level keys cause.
- **G14 applies to all of them.** Any gap would be closed on another project's schedule, if at
  all.

**The count is close, and the choice rests on which gaps can be fixed.** pydra and Parsl each
meet six requirements today. pydra has one ○ (G10), which §3.1 closes. Parsl has two: G1, which
is structural, and G14. The choice is pydra because its gaps are fixable by senselab and are work
senselab must do anyway, while Parsl's G1 gap is structural.

## 4. Caveats

- **Maturity.** pydra 1.0 is alpha (1.0a12), and `pip install pydra` still gives 0.25. One
  developer wrote about 70% of the last year's non-bot commits. senselab's dependence on pydra
  increases the maintenance load on the pydra project.
- **Risk.** §3.1 (per-element dataflow) is high-risk: it needs the State refactor, and it gates
  corpus-scale use.
- **Mitigation.** senselab depends only on pydra's graph layer (Route A in `execution.md` §4),
  behind its own `Executor` protocol and result store. If the gating changes stall, the fallback
  is a 1–2k-line graph layer in senselab (`design.md` §4.8). No senselab user code would change.
- **Where others stay better.** For multi-node tensor-parallel inference, Ray would be revisited.
  Parsl's HTEX remains a reference design for the pool worker (§3.9).
