# Cluster execution utilities: `vllm_server`, `work_pool`, `venv_worker`, `uv_tools`

**Date:** 2026-10-12. **Owner's ask:** senselab should hold the tasks and utilities that make execution on a
cluster efficient, not each project that runs on one.

## Why

Two task-specific designs, now on `alpha` from `feat/task-events` (PR #584, then #559), proved the pieces this
branch generalises:

- the PII reviewer's vLLM server (`specs/20261010-vllm-reviewer/design.md`, server half of
  `src/senselab/text/tasks/pii_detection/redaction_review_vllm.py`; `d74d1083`, `bd947c9c`);
- the slice pool (`specs/20261011-slice-pool/design.md`, `specs/20261011-slice-pool/pool/slice_pool.py`;
  `93f24382`), a worker pool over a stage's slices on shared storage.

One lives inside a task package and one is a run-tree tool with site paths in it. This branch lifts the general
parts into `senselab.utils`. All four modules are standard library only at
import, so they load in a lean environment on Python 3.11–3.14. They hold no site knowledge: partitions, paths,
GPU types and resource-class names all come from the caller.

Out of scope here:

- Adopting the utilities in tasks. The reviewer keeps its own server code until a follow-up switches it to
  `vllm_server`. The `venv_worker` callers (qwen-asr, crisperwhisper, clearvoice) are unchanged.
- Any dataset or project code.

## Measurements this design rests on

### The vLLM reviewer (`specs/20261010-vllm-reviewer/design.md`)

On one H100, over 246 reviewer requests (greedy, 1,024 new tokens):

| arm | requests | calls/s | output tok/s | latency p50 | reproducible |
|---|---|---|---|---|---|
| transformers worker, concurrency 1 | 242 of 246 | 0.049 | 17.6 | 15.4 s | -- |
| vLLM 0.31.0, concurrency 1 | 246 of 246 | 0.179 | 66.7 | 4.1 s | 246/246 identical token ids across two runs |
| vLLM 0.31.0, concurrency 32 | 246 | higher | higher | -- | no, with or without `VLLM_BATCH_INVARIANT` |

What `vllm_server` takes from these numbers:

- **Concurrency and identity.** vLLM at concurrency 1 is 3.7x the transformers worker and exactly reproducible,
  but batching is not. So concurrency, batch size and servers per card are part of `engine_identity`. vLLM is
  also a different engine from transformers: 33% of completions were exact, and scalar labels agreed on 92.6%.
  So the engine's version and ordered server arguments are in the identity too.
- **Sizing on one card.** At `gpu_memory_utilization` 0.90, a 31B w4a16 checkpoint loaded 19.78 GiB of weights and
  kept a 56,753-token KV cache.
  - The smoke test at `bd947c9c` ran three servers at 0.30. Every one refused to start: 10.32 GiB of KV cache was
    needed and 0.18 GiB was available.
  - Two servers at 0.45 fit, with 14,345 tokens each and a maximum concurrency of 1.17x at 12,288 tokens.

  Hence `utilisation(n) = floor(90 / n) / 100`. Hence also the start lock: servers that share a card start one at
  a time, so each one sizes its KV cache after its neighbours. And a server that exits before `/health` is
  reported as `VllmStartError("exited")`, which is the fit failure.
- **Lock and CUDA.** vLLM 0.31.0 needs torch 2.13 on the CUDA 13.0 index. `alpha` has the opt-in `cu130` row
  (`d3d86439`) while keeping the default ceiling at (12, 8), so every other venv still routes to `cu128` on a
  CUDA-13 host, and the rule that leaves triton out of a lock wherever torch pulls it in (`d74d1083`). This
  branch adds a test and a design-note entry for both.
  - The committed `venv_locks/vllm-server.txt` has the same body as `pii-review-vllm.txt`
    (requirements-sha256 `5b1a68e3…d8a4`, Python 3.12, exclude-newer `2026-10-10T00:00:00Z`).
- **Logging.** vLLM logs no request content unless `--enable-log-requests` is passed. `VllmServer` refuses any
  argument starting with `--enable-log-req` or `--enable-log-out` (argparse accepts abbreviations), and
  `--config` (a config file could turn them on). The clients log nothing.

### The slice pool (`specs/20261011-slice-pool/`)

- **The problem it solved.** One array task per slice counts against every partition's per-user submit limit, so a
  chain submitted up front capped at about 256 tasks, and a slice whose task died stayed missing until it was
  resubmitted by hand. With a pool, the queue holds the worker count, not the slice count.
- **Atomic primitives on NFS v4.2, measured.** `harness/race.py` ran 32 processes on 4 nodes, racing for the same
  400 names in the same second. `mkdir` gave exactly one winner for 400/400 names, and so did a staged dir rename.
  NFS 4.1+ sessions give exactly-once semantics, so a retransmitted create does not report a false `EEXIST` to
  the process that won it.
- **Liveness reads.** A bare `stat` may answer from an attribute cache up to `acregmax` (60 s) old. An `open()`
  followed by `fstat()` forces close-to-open revalidation.
- **Protocol tests.** All four ended with 0 overlapping attempts:
  - t1: 40 slices, 6 workers, one worker cancelled mid-slice. 40/40 done, and the released slice was rerun.
  - t2: both workers cancelled. The gate released the claims and resubmitted the workers once, then 6/6 done.
  - t3: a worker that SIGKILLed itself. Its claim went stale and was taken over, then 12/12 done.
- **Interruption budget.** `max_interrupts` 6 stops a slice that always outlives the time limit from looping
  forever.

### This work

The runs below were made on `feat/cluster-exec`, whose commits this branch squashes; the commit ids name that
branch.

- **Primitive and protocol races** (`harness/race.sbatch`): 16 workers on 2 nodes. Legs 1 and 2 ran at `74d24359`
  and again at `6cbe4e80`, which added leg 3. The figures below are from the `6cbe4e80` run.
  - Primitives: `mkdir`, staged dir rename and the file-out-of-queue rename each gave exactly one winner for
    400/400 names.
  - Leg 1: 2000/2000 items done, `double=0 lost=0`.
  - Leg 2: `lease_s=5`, with 6 workers SIGKILLed while holding leases. 2006 claims, 6 interrupted, 2000/2000 done,
    0 dead, `double=0 lost=0`.
  - Leg 3: 96 items held for 5.5–8 s under real 1 s heartbeats with `lease_s=5`. 96/96 done, `overlap=0`.
  - The first two-node run, before `74d24359`, found NFS reporting `ESTALE` for a lease another node had renamed
    away mid-lookup. `ESTALE`, `ENOENT` and `ENOTDIR` now all read as "gone".
- **Controller smoke** (`harness/smoke.sbatch`, at `4bccbc5b`): 20 items, one CPU class on two partitions.
  - The worker holding the first item was cancelled. The item came back through `release`, not the reap, and
    finished at attempt 2.
  - Result: `done=20 dead=0`. Workers started on both partitions.
- **Lean matrix.** The five test files of these modules pass on Python 3.11, 3.12, 3.13 and 3.14 with the same
  count (59 at `4bccbc5b`).

## API

### `senselab.utils.uv_tools`

The primitives shared by both hashed-lock formats. The header formats themselves stay with their owners.

- `find_uv(*, install=False)` searches `PATH`, then `~/.local/bin`, then `~/.cargo/bin`. It installs uv with pip
  only when `install=True`, and otherwise raises `FileNotFoundError`. It replaces `subprocess_venv._find_uv` and
  `venv_lock._uv`, and `venv_lock` still raises `VenvLockError` when uv is missing.
- `normalize_name(name)` gives the PEP 503 form of a name.
- `requirements_sha256(lines, *, header_fields)` digests the sorted requirements with the header fields. It is
  order-insensitive and header-sensitive, and the committed digests are unchanged.
- `is_torch_family(name, *, roots=TORCH_NAMES)` covers the roots and their `nvidia-`, `triton`, `pytorch-triton`
  and `cuda-` variants.
- Also on this branch: `dependencies.is_hf_model_cached(..., *, cache_dir=None)`, so a caller can check a cache other
  than the process-global one.

### `senselab.utils.venv_worker` (on `alpha` since `372c7ae1`, `112f15f0`)

- `serve_in_venv(identity, *, python, script, init, request, env, label, load_timeout_s, request_timeout_s)`
  answers a request from a long-lived subprocess-venv worker.
  - The backend script defines `load(init)` and `handle(state, request)`. There is one worker per identity per
    process.
  - The identity is the caller's tuple plus the interpreter and digests of the script, the init payload and the
    env.
  - The wire format is one JSON line each way. The worker's fd 1 is pointed at stderr so that library prints
    cannot interleave with replies.
- A worker found dead between calls is replaced once: under the pool lock, only the caller whose entry is still
  that worker replaces it. A caller queued behind it is served by the replacement. (This branch's fix: on
  `alpha`, each queued caller replaced the entry.)
- `venv_worker_events(clear=False)`, `venv_worker_stats()` and `shutdown_venv_workers()` (registered at exit)
  report and stop the workers. `VenvWorkerTimeout` is raised on a timeout.
- This branch also takes the logger from `logging.getLogger("senselab")` in the standard library (the same named
  logger) instead of `senselab.utils.data_structures.logging`, so the module imports without torch.

### `senselab.utils.vllm_server`

- **`VllmServer(executable, model_dir, *, server_args, devices=None, start_lock_dir, expected_version,
  log_dir=None, startup_timeout_s=1800)`** is a context manager.
  - It takes `fcntl.flock` on `vllm-start-<host>-<devices>.lock`, picks a free loopback port, and starts
    `<executable> serve <model_dir> --host 127.0.0.1 --port <p> *server_args` in its own session.
  - It waits for `/health`, then releases the lock and checks `/version` and the served `snapshots/<commit>` name.
  - A watchdog in its own session kills the server's process group if the owner dies, whether the owner is killed
    alone or with its whole group.
  - On exit the group gets SIGTERM, then SIGKILL.
  - `devices` sets `CUDA_VISIBLE_DEVICES` only when it is given.
  - Server output goes to `log_dir` with mode 600.
  - `VllmStartError.kind` is one of `exited`, `timeout`, `version` or `snapshot`. HTTP errors and malformed JSON
    from the probes are `version`.
- `server_args(settings)` gives the ordered arguments. It refuses the content-logging flags.
- `engine_identity(version, server_args, *, concurrency, batch_size, servers_per_card)` and
  `utilisation(servers_per_card)` give the identity and the per-server memory share.
- The clients are `chat(url, model, messages, *, sampling)` (with `content_part` for text, image, video and audio
  parts), `transcribe(url, model, audio_path, *, language, prompt, sampling)` (multipart, under a fixed file
  name) and `complete(url, model, prompt_token_ids, *, sampling)`.
  - They use a proxy-free opener on loopback and return the parsed JSON.
  - An HTTP error keeps the status, the reason and the server's error message, cut to 300 characters. Fields
    that can echo the prompt (`input`, `param`) are dropped, and so is a body that is not JSON.
  - An error message names a local media path only by its suffix and a short sha256 of the path.
  - The request shapes were checked against the vLLM v0.31.0 source.
- `ensure_vllm_venv()` builds the `vllm-server` venv from the committed lock (vLLM 0.31.0, Python 3.12, CUDA
  ceiling (13, 0)) and returns `<venv>/bin/vllm`.

### `senselab.utils.work_pool`

- **`WorkPool.create(path, *, lease_s, heartbeat_s, max_attempts=2, max_interrupts=6, success_codes=(0,))`**.
  `create` requires `lease_s >= 3 * heartbeat_s`. The layout:
  - `pool.json`, `queue/`, `leases/` and `tmp/`;
  - `interrupted/`, `done/`, `failed/` and `dead/`;
  - `conflicts/`, `workers/`, `hostcheck/` and `status.json`.
- An `Item` has `id`, `resource_class`, `command` (argv), `env`, `attempt`, `max_attempts` and `payload_sha256`.
  The pool attaches no meaning to the class name.
- `publish`, `claim(resource_classes, worker)`, `heartbeat`, `complete`, `fail`, `release`, `reap` and `status`:
  - `release` sends the item back to the queue with attempt+1 and counts toward `max_interrupts`, not toward the
    failure budget.
  - `reap` re-publishes leases whose heartbeat is older than `lease_s`, and abandoned `tmp/` entries.
  - `state(item_id)` names where one item is (`done`, `dead`, `leased`, `queued`, `staged` or None), so a caller
    that publishes a fixed set of ids can skip ids already published and read each one's outcome.
- **Worker:** `python -m senselab.utils.work_pool work POOL --classes a,b [--linger N] [--poll S]
  [--host-check CMD]`.
  - It runs each item's command with `POOL_ITEM_ID` and `POOL_ATTEMPT` set, under a heartbeat thread.
  - SIGTERM releases the item.
  - A failed host check releases the item and exits with code 3.
- **Controller:** `Controller(pool, classes, *, runner, worker_command, poll_s, pending_s=600, release_cap=3)`, with
  `ResourceClass` / `PartitionSpec` from `load_classes(yaml)`, and the CLI
  `python -m senselab.utils.work_pool control POOL --classes classes.yaml [--once]`. Each tick:
  - reaps, then reads `status()` and this pool's own workers from `squeue`;
  - submits `sbatch` workers where queued items outnumber live workers, preferring the partition with the fewest
    and youngest pending workers;
  - skips a partition whose workers stay pending past `pending_s` for `Priority`, `QOS` or `ReqNodeNotAvail`. A
    partition pending on `Resources` is not skipped;
  - pauses a partition after `release_cap` host-check releases within an hour;
  - caps workers at `max_workers` per partition, and cancels idle pending workers once the queue drains.

  `submit_controller(...)` renders a small controller job.

## Why `work_pool` is race-free

**What the storage must give.** The argument assumes three things, all of which the slice-pool and branch race
runs measured on NFS v4.2:

- `rename(2)` of a file or a directory onto an absent name is one atomic server-side operation. Of several renames
  of the same source, exactly one succeeds.
- A rename of a directory onto a non-empty directory fails with `ENOTEMPTY` or `EEXIST`.
- `link(2)` fails with `EEXIST` when its target exists, so it never overwrites.

Nothing assumes that node clocks agree, or that `stat` is fresh.

**Invariant: an item has exactly one carrier.** At any instant it is in exactly one of these places:

- `queue/<i>.json`;
- `tmp/<entry>/item.json`, a staged claim or a lease renamed aside;
- `leases/<i>/item.json`;
- `done/<i>.json` or `dead/<i>.json`;
- a `conflicts/` record, which never runs.

Every transition is either one rename of the carrier (or of its directory), or an exclusive link of a new file
that is the commit point, after which the old copy is removed. The transitions, in turn:

1. **Publish.** The item is written in `tmp/` with a fresh `placed` token and linked to `queue/<i>.json`. Before
   that, the id is checked against every state, `tmp/` included. If the link finds the target present, the item
   is refused, never overwritten.
2. **Claim.** The worker first stages a complete directory `tmp/claim-<T>/` holding `owner.json` (token T) and a
   fresh `heartbeat`. Then:
   - It takes the item with `rename(queue/<i>.json, tmp/claim-<T>/item.json)`. Only one rename of that source can
     succeed, and a loser finds the source gone and tries the next item.
   - It publishes the claim with `rename(tmp/claim-<T>, leases/<i>)`. So a lease is never visible without its
     owner and heartbeat. A watcher over 200 claims never saw one, and a mutation that publishes file by file
     fails that test.
   - A claimed item that already has a `done/` or `dead/` record, or whose `leases/<i>` already exists, is a
     duplicate. It goes to `conflicts/` and does not run.
3. **One actor per lease.** `complete`, `fail`, `release` and `reap` all begin with one rename of `leases/<i>` to a
   fresh `tmp/` name. Of concurrent finishers and reapers, exactly one wins, and the others see the lease gone.
   - The token is checked before the rename and read back after it. If the aside directory holds another token
     (the lease was reaped and claimed again in between), it is renamed back untouched.
4. **Next state by exclusive link.** The next state is a new file inside the aside directory: `done.json`, a dead
   record, or the item with attempt+1. It is linked into `done/`, `dead/` or `queue/`, and the aside is removed
   only after that.
   - A crash before the link leaves `item.json` unchanged, so the next reaper derives the same attempt number and
     no attempt is skipped.
   - A leftover aside whose item has moved on is dropped, not requeued. It has moved on if a newer attempt is in
     `queue/`, `leases/` or `tmp/`, or the item is in `done/` or `dead/`.
5. **NFS read-back.** A retransmitted rename or link can land and still report an error to the process that made
   it. After every error the destination is read back for the caller's own token (`owner.json` or `placed`). A
   winner whose call errored therefore still knows it won. A loser never finds its token.
6. **A lost lease stops its child.** Before each touch, the heartbeat thread re-reads `owner.json`:
   - On a token mismatch, it kills the child's whole process group and the worker writes nothing.
   - On absence, it checks again after `PUTBACK_WAIT_S`, because a reaper may have renamed the lease aside and be
     putting it back.

   Heartbeat state is per run, so a stalled thread cannot touch the next item.
7. **The reaper does not take live leases.** It reads the owner's token and the heartbeat age, renames the lease
   aside, then checks both again. If the token differs or the heartbeat is fresh, it puts the lease back.
8. **Clocks.** Ages are measured on the storage's clock. Heartbeat mtimes are compared with the mtime of a file
   just touched on the same storage, and `tmp/` entries are aged by their directory's mtime and ctime. A fresh
   marker file bumps that mtime on each move. All of these are read through `open()` and `fstat()`, which forces
   NFS revalidation.

**What it guarantees.** Each item is completed at most once, because there is exactly one exclusive link into
`done/`. No item is lost: every `tmp/` state is reaped back after `lease_s`. Every attempt number is recorded once.

**What it does not.** Execution is at least once, as in the slice pool. An owner that is alive but cannot
heartbeat for `lease_s` may still be running when its item is re-run elsewhere. It is stopped within one
heartbeat of resuming, and its late `complete` finds the lease lost and writes nothing. So item commands should
be idempotent, or resumable per unit of work. `max_interrupts` (6) bounds how often an item can be reaped or
released before it goes to `dead/` with reason `interrupts`.

**The evidence.** `src/tests/utils/work_pool_test.py` covers each transition, including fault injection for:

- renames and links that land and then raise;
- `ESTALE`;
- put-back windows;
- a stalled finisher racing the reaper;
- a re-reap after a crash.

Each fix is guarded by a mutation that fails its test. `harness/race.sbatch` runs the three legs above across two
nodes.
