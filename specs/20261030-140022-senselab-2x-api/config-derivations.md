# Config derivations

Why each key of a config file has the value it has (P-17). The packaged execution profiles (`packages/senselab-core/src/senselab_core/runtime/data/profiles/`) come first, by profile and top-level key; every other run config or data file gets a `##` section of its own. The YAML files carry a one-line `#` comment per key; the measurement and the reasoning are here.

## laptop

### laptop.stack_preferences

Empty: no task has two stacks yet, so the backend order a task declares decides. A profile lists a task's stacks best first once a second stack exists.

### laptop.devices

Fastest accelerator first, CPU last. A backend takes the first listed device that the host has and the backend declares.

### laptop.env_roots

The per-user cache directories 1.x built its subprocess venvs under, one root per env kind.

### laptop.envs

No torch index override: the CUDA index is probed from the host. A site whose probe is wrong sets torch_index. server_start_margin_s 60: a server start lease is taken before the server is launched and its health deadline begins, so a waiter for it waits the start-up timeout plus this margin, plus the lease's stale bound and the lease timeout; the lease itself beats under `lease`. Not measured: the launch (two process spawns) is sub-second, and the margin is padded for the storage clock's mtime granularity and for a loaded host.

### laptop.model_sources

Null: the hub organisations of the catalogue entries. A site that mirrors or restricts models lists organisations here.

### laptop.model_cache_budget_bytes

Null: measured where the process runs (`hardware.measured_budget`): the total memory of the smallest GPU the process sees, else the host's physical memory. A host backend's state counts against it by its module's `state_bytes`; one declaring `models` must define it, and one declaring none without it counts 0. Measured rather than a number because the memory the cache can hold is the device's or the host's, which the site, not the profile, decides; an unbounded cache kept every earlier model resident in a long-lived process (the batch invariance probe, Phase 3 Task 20c).

### laptop.retry

Three tries and a one second first wait, doubling: the 1.x schedule (1 s, 2 s, 4 s) of SENSELAB_HF_MAX_RETRIES. Not measured: no failure-rate data for the Hub or a mirror exists in this repository. A fault that has not cleared after three seconds of waiting surfaces to the caller rather than stalling the run.

### laptop.network

Online: a laptop reaches the Hub and the data mirrors directly.

### laptop.resource_classes

One class per device the laptop may have, each the host's own CPUs and memory with no time limit and no partition: a laptop runs its workers as local processes. max_workers 1: one worker per class, since a second worker on one host competes for the same memory and device; the plan places a node on the first class of the first listed device the host has and the backend declares. memory_cap_bytes null: no table measures a model yet, and a laptop's free memory is not fixed, so no static cap is checked. No batch size: a placement records batch 1 (T20R-6); `size` sizes the batch where the work runs, from the unit's estimate and the length of each job, filling sizing.memory_headroom of one worker's share of the class's memory beside the resident model, against the device fit on a `cuda` class (its memory_cap_bytes, else the memory of the device seen) and the host fit otherwise, never the other (calibration-probe.md §6.4); batch 1 is the batching that needs no measurement.

### laptop.substrate

local: a laptop starts its pool workers as processes on itself. start_backoff_first_s 0.05 and start_backoff_max_s 30: the wait after a class's first failed start (a worker that exits before it marks itself ready), doubling per failure in a row up to the maximum; the same wait follows each failure of the substrate's own commands, during which the scaler does not use the substrate. Not measured: the values the pool used before the keys existed, when it took this wait from pool.first_poll_s and pool.poll_s; the basis to measure is a worker's start-to-ready latency on the host. commands: none, a local substrate runs no scheduler command. placement_hook null and placement_hook_s 0.05: as orcd, with the bound equal to this profile's start_backoff_first_s; a laptop has no site, so no hook is named.

### laptop.pool

quantum_items 100 and quantum_s 300 are ceilings: `size` derives each unit's quantum from its measure (sizing.load_share) and holds it to at most a hundred jobs and to the jobs that fit in five minutes after the load (and in the class's time_s), and a pool worker stays on a class for at most that many jobs or seconds, so an older queue waits at most a ceiling quantum. Not measured: the fixed quantum the pool used before sizing; the ladder's per-item cost (Task 20 Step 4) shows whether a ceiling binds. poll_s 30: the 1.x reproduce controller's poll interval; a worker whose queues stay empty for three polls exits. first_poll_s (0.05 on laptop, 1 on orcd; Phase 3 Task 14 fix round): a coordinator's first wait for an outcome, doubled up to poll_s, so a local run does not wait 30 s per fragment; orcd starts at a second to keep stats off the NFS server. Not measured. segment_items 512 (Phase 3 Task 13 review, owner ruling): a claim lists its oldest live segment, so its cost grows with the segment, not with the queue. The Task 13 review measured claim plus commit on local SSD at 10^4 items queued: 22.4 ms with 10^4-item segments (about 1 MB of index read per claim) against 3.4 ms with 500-item segments. 512 keeps a segment's listing to about a thousand names (slots and their marks) while 10^4 items still cross twenty segments; the Step 5 run on ORCD revises it. presence_refresh_s (1 on laptop, 30 on orcd; Phase 3 Task 20R-5 fix round): the longest a pool worker reuses what it read of the ready marks of the workers on its host, the compute processes on its devices (`nvidia-smi`) and the probe queues of its run. A read lists the run's `workers` directory (each mark is read once), checks each peer's liveness and, on a `cuda` class, runs one `nvidia-smi` query, so a claim costs none of them; at 1 worker that is one listing and no check per interval, at 10^2 workers 10^2 liveness checks per interval instead of 10^2 mark reads per claim. Peers change on the scale of worker starts and exits, which `poll_s` paces: orcd takes `poll_s` (30 s); laptop takes one second, for a handful of local workers. Not measured: the interval's effect on the co-tenancy a short run reports, and a probe queue made inside the interval is served from the next one. admit_threads 8 (Task 20R-7a-0 fix round): the threads that ask the result store which of a plan's jobs it would serve, one `admits` (one `context.json` read) per job. The reads are latency-bound, so a thread count bounds open files and not CPU. Measured on local SSD only: 6.9 ms for 1 job and 0.47 ms per job at 10^2 (empty store, 8 threads); not measured on a network file system, where the ORCD run (T20R-10) sets it.

### laptop.sizing

load_share 0.05: a unit's quantum is the fewest jobs whose time keeps its model load and its claim within a twentieth of the quantum's, so loading and claiming cost a worker at most 5 % of its time (the calibrate-then-size design); with neither measured above zero the quantum is the ceiling. A target, chosen, not measured. memory_headroom 0.9: a batch fills at most nine tenths of the worker's memory beside the resident model: on a `cuda` class with a device measure, its memory_cap_bytes against the device pair; otherwise its mem_bytes, else on a local substrate the host's, against the host pair; the tenth left covers allocator fragmentation and a peak above the measured one. Not measured; the spread of peak memory over the 10⁴-item ladder point (Task 20 Step 4) replaces it. target_wall_s 1800: the wall time the workers of one queue aim to end its waiting work in, so a queue starts no more workers than end it in half an hour and none for work one worker ends sooner; a target, chosen so a 10⁴-item ladder point ends within one session; laptop's classes have max_workers 1, so it never binds there. calibration_items 4: a unit with no resource-table row first runs four one-job items on one worker; the first carries the model load and the median of the other three, robust to one outlier, is the seconds per item. The least count whose warm items have a median that tolerates one outlier. Batch against throughput (Phase 3 Task 20e, 20g): a batch sized by memory is kept only where items sent in a batch were measured no slower per item than items sent alone; otherwise, and where the batched time is unmeasured, the batch is one (`size` gives both `memory_batch` and the `batch_size` kept). A calibration round whose items include no warm one-job item measures nothing and writes nothing: its load cannot be told from its work. A calibration round measures both: its calibration_items one-job items first, for the load, the claim and the memory; then calibration_items − 1 pairs on the worker the first round warmed, each two items of the memory-sized batch's jobs, one sent as one request and one sent a job per request, so each side has a median of three warm figures (robust to one outlier on each side), each figure one request call over as many jobs and inputs; the batch is kept only where the batched median per-item seconds of that second round is not above the one-per-request median, and at most at the batch that round measured, never the larger memory batch (P-34; Task 20k: speech_to_text~transformers and speechbrain were measured at 16 against a memory batch of 100, the quantum); a larger batch needs a ladder point with jobs enough to measure it, a 10³ GPU point for a batch of 100 (R-237) (Task 20h: one-job items sampled straight after the load read 2 to 5 times the steady cost at 9380b530; Task 20j: one-job items are gathered into one claim and share its wall, so they were one figure carrying a third of a call's fixed cost, R-234). A ladder's tables carry both, since the ladder's pool reads no packaged table and so calibrates every unit in the run. No slack term: the two medians are of one backend on one warm worker, the median absorbs one outlier per side, and a wrong call errs to batch 1, which is never over memory. Cost bound: once per unit per pool (a written row skips it), 2 × (calibration_items − 1) × batch jobs, all real work whose results are kept, and 2 × (calibration_items − 1) claims. Device memory per item is the backend's own request peak over what stays allocated after the request (`torch.cuda.max_memory_allocated` against `memory_allocated`, reported in `done`), divided by the request's items; a server backend, which reports neither, keeps the device read of the worker's processes after each claim (its fixed reservation); a model backend that reports neither (no CUDA initialised in torch) leaves its device memory unmeasured, so its batch is 1, and the pool logs it once per unit. calibration_workers 1: a queue whose units are all unmeasured is given one worker, so the warm items follow the cold one on the same worker and the load is read once; a queue mixing a measured unit with an unmeasured one is sized by the measured one and may run the round on several workers, each giving a cold sample, which `calibrated` reads as well. Workers of a `cuda` class whose substrate shares devices (local: the GPUs the coordinator sees, one per worker as `CUDA_VISIBLE_DEVICES`) are at most devices × floor(memory_cap_bytes / measured device peak), at least one per device and one while the device peak is unmeasured: the measured peak is what one worker holds, so the cap divided by it is how many fit a card. No new key; Slurm workers each hold their own card, so the bound is max_workers there. device_query_s 30: a pool worker reads its processes' device memory once per claim on a `cuda` class with `nvidia-smi --query-compute-apps`, bounded by the hardware probe's `nvidia-smi` bound; not measured, the basis is the command's p99 latency on a loaded GPU node. T20R-6: `size(estimate, grant, config, work, calibration=)` reads these keys as follows. load_share and the item's per-claim cost `a` close an item once its median seconds reach (load_s + a)/load_share; an item also closes before its upper-predicted seconds t·(1 + z·spread) plus `a` pass the grant's quantum_s, with z = calibration.confidence, so about one item in twenty overruns at the configured one-sided 5 %, and at quantum_items jobs; with the slope `u` or the spread unmeasured an item is cut at quantum_items alone and an unmeasured figure is never read as 0. target_wall_s gives the workers: ceil(sum of median predicted seconds / target_wall_s), at most the item count, max_workers and, on shared devices, devices × w_d, where w_d = floor(memory_headroom × cap / P) for the batch peak P = resident + m0 + m1·B*·L_max at the run's longest job, lowered where a measured regime factor says k/ρ_k stops increasing; each worker's memory is cap / w_d, so the workers of one card together stay within memory_headroom of it. calibration.unmeasured_workers gives the workers of a unit with no estimate, which runs the safe values (batch 1, items of quantum_items jobs). No new key. The Pool still reads `size_row` until T20R-7a; calibration_items, calibration_workers and the `Measure`-based derivation above go with T20R-7a′.

### laptop.calibration

Task 20R-1, design `calibration-probe.md` §11; every value is the design's. Values marked provisional are replaced by Task 20R-10's measurement under the stated rule (ruling O-6).

measures_roots: `${cache_root}/2x/measures`, where `${cache_root}` is senselab's cache root (`SENSELAB_CACHE`, else `~/.cache/senselab/hf`; `revision.cache_root()`), resolved by `resolve_roots`, so the default is written once. A root must be absolute after expansion: a blank root, a repeat and a relative entry are refused at load, and a variable that is unset or empty is no root (logged, never raised), the own root only. Read in order and pooled; writes and leases go to the first writable root, so a read-only member reuses the group's measures and writes its own (§3.3). trust own: a laptop has one user, and another user's records are not read. A run overrides it with the run config's `calibration.trust` (ruling O-3, §3.5). warmup_requests 2: the least w such that, on the table probe, the median of warm requests w+1..w+3 of one stratum lies within the steady band of the requests after them, the maximum over GPU units; the basis until measured is R-230 (CUDA initialised inside the first request) plus one request for shape-dependent kernel selection, R-231's decay being confounded by gathering (R-234). length_strata 2: two points fit a line in length (must-fix 8, R-221); the table profile uses 3 to test linearity. min_requests 3: the least count whose median tolerates one outlier. single_item_jobs 4: items of 1 and k jobs separate the per-item cost `a` from the per-job cost; k is such that `a` is at least a fifth of a k-job item's wall at the measured CPU fit (R-236: `a` 0.040 s against 0.0366 s per job). batched_sizes 1: run time measures B₁ only; the table profile measures 2 (R-237). probe_items 50: the most jobs a probe plans, a bound; chosen as `warmup_requests + length_strata × min_requests × (1 + B) + min_requests × single_item_jobs` at a target batch B = 5; one CPU worker; the table profile sizes it for B = 100. confidence 1.645: one-sided 5 %, the false-batch rate of the batch decision (§5.4 step 6) and the share of items cut past `quantum_s` (§6.2); a wrong batch decision errs to batch 1. drift_confidence 2.576: two-sided 1 %, at most one spurious confirming probe per hundred windows of a stable unit. observations_kept 3: superseded probes kept for audit; three give a median of medians that tolerates one bad probe. unmeasured_workers 1: unmeasured work is not scaled (S5); one worker runs it under the safe values. resolve_s 1800 (provisional; Task 20R-7a design amendment, 2026-11-08): the longest a unit waits in `Resolving` after a worker of its queue is live and not pending, i.e. the longest a live worker takes from its start to its ready mark, which follows the env builds it does first (§7.1, §8). The basis until measured is two env builds, each at the build waiter's bound of `lease.stale_after_s` plus the lock timeout (300 + 600 s, itself 1.5 times the slowest measured cold build, 590 s); a queue that needs more builds goes unmeasured, which costs efficiency and changes no result. Task 20R-10 replaces it with the measured start-to-mark wall, the maximum over the profile's classes. Not keys, derived from other keys: `probe_s` = `sizing.load_share × sizing.target_wall_s` (0.05 × 1800 = 90; probe work, like loading, may cost a queue at most a twentieth of its target wall; it counts warm probe requests only, §4.5), `peer_wait_s` = `probe_s + lease.stale_after_s` (90 + 300 = 390; a peer's probe ends within its budget or its lease goes stale) and `drift_window` = `length_strata × min_requests` (6; drift is judged at the precision of the probe's single phase). They are properties, so no profile can disagree with them.

### laptop.lease

Not measured until Task 16, which measures the store root's attribute delay that bounds both values. stale_after_s 300: five times the NFS client's default maximum attribute-cache time (acregmax and acdirmax, 60 s), so a beat directory a live holder made is seen well inside the bound on a shared root. renew_fraction 0.33: the 1.x work pool's rule that a heartbeat is at most a third of the lease, so two missed renewals still leave the lease live. The one place lease settings live: `pool.lease_renew_fraction` and `envs.lease_stale_after_s` were merged here (Task 12), and every lease holder (env build, vLLM start, run manifest, label map) beats under `Lease.keep_alive` every `renew_fraction × stale_after_s`. The env build lease used to be stale after 1200 s, about twice the slowest measured cold build of the crisperwhisper env on ORCD (362.6 s GPU node, 579.7 s and 590 s CPU node, n=2 per host; specs/20260907-shared-lock-heartbeat-inf/ stampede-timeout-and-identity-file.md); a beating build no longer needs a bound that covers it. A waiter on a build waits stale_after_s plus the lease timeout, 300 + 600 = 900 s, which is about 1.5 times the slowest measured build (590 s, on a loaded CPU node); the padding is less than the old 1200 s carried because the waiter no longer also waits out a stale bound sized to the build. `ExecutionProfile.load` refuses a profile where 2 × renew_fraction × stale_after_s + attribute_delay_s ≥ stale_after_s: the factor 2 allows one missed beat, and a client may see a beat up to attribute_delay_s late. Read by `PydraRunner` for pydra's job leases (P3-53); the pin at c8541fc5 has no `LeaseSettings` yet (Task 45), so the values are held and not yet passed. attribute_delay_s 0: the laptop profile's roots are on local disk, where a `stat` reads the inode itself and no client attribute cache stands between a beat and a breaker; to be replaced by Task 16's measured value.

### laptop.faults

content_attempts 2: the 1.x work pool's DEFAULT_MAX_ATTEMPTS, an item failing for its content twice goes to dead/. infrastructure_replacements 6: the 1.x work pool's DEFAULT_MAX_INTERRUPTS, releases and reaps allowed before an env is reported broken. resource_replacements 0: one per larger memory class the profile lists, and a laptop lists none, so a resource fault fails at once.

### laptop.timeouts

Null, no bound: no measurement of the slowest model load or the slowest single request exists. A site's own ladder run (`--measures-root`) gives load_s and the per-input time from which it sets both; a timeout is a resource fault, so a bound set too low re-places work rather than failing it.

### laptop.supervisor

start_time_probe_s 5: the longest `ps -o lstart= -p <pid>` may take when a supervisor reads a launched process's start time off Linux (Linux reads /proc/<pid>/stat field 22 and needs no bound). Over 200 calls on a 10-core laptop at load average 4.4 it took a median 2.9 ms and at most 4.8 ms (2026-11-07), so 5 s is about a thousand times the slowest call and only a hung or starved `ps` reaches it. A read that does is an unknown start time; the owner records it as unknown, and the keeper never kills a process whose start time is unknown at launch or at the end, so a bound set too low leaks a process at its owner's death rather than killing one that reused its pid. `lstart` has one-second resolution: a pid reused within the same second as the original start would match, which needs the pid space to wrap within a second. The process-wide supervisor, which no run configures, takes the packaged laptop profile's value.

### laptop.services

idle_exit_s 600: ten minutes, twice the pool quantum, so a service outlives a consumer's yield to another queue and the next quantum's requests find it up. Not measured: no measurement of how long consumers leave a service idle exists.

## orcd

### orcd.stack_preferences

Empty: no task has two stacks yet, so the backend order a task declares decides. A profile lists a task's stacks best first once a second stack exists.

### orcd.devices

CUDA first, CPU last: the cluster has no Apple silicon. A backend takes the first listed device that the host has and the backend declares.

### orcd.env_roots

Env trees live under the user's scratch, never the home directory, which is quota-limited and slow for many small files. ORCD sets no `$SCRATCH`; each user's `~/orcd/scratch` links to their scratch directory, so the root is reached through that link. `$SENSELAB_ENV_ROOT`, when set, replaces that root: personal scratch has a file-count quota (1M inodes, shared with the user's other work) that one set of backend envs nearly fills (job 27609929 failed at 950K files), so a job may put envs on group scratch.

### orcd.envs

No torch index override: the CUDA index is probed from the host. A site whose probe is wrong sets torch_index. server_start_margin_s 60: a server start lease is taken before the server is launched and its health deadline begins, so a waiter for it waits the start-up timeout plus this margin, plus the lease's stale bound and the lease timeout; the lease itself beats under `lease`. Not measured: the launch (two process spawns) is sub-second, and the margin is padded for the storage clock's mtime granularity and for a loaded host.

### orcd.model_sources

Null: the hub organisations of the catalogue entries. A site that mirrors or restricts models lists organisations here.

### orcd.model_cache_budget_bytes

Null: measured, as laptop.model_cache_budget_bytes: on a GPU node the smallest visible GPU's memory (an H100: 79.18 GiB as vLLM reads it), else the node's physical memory.

### orcd.retry

Three tries and a one second first wait, doubling: the 1.x schedule (1 s, 2 s, 4 s) of SENSELAB_HF_MAX_RETRIES. Not measured: no failure-rate data for the Hub or a mirror exists in this repository. A fault that has not cleared after three seconds of waiting surfaces to the caller rather than stalling the run.

### orcd.network

Online: compute nodes reach the Hub and the data mirrors.

### orcd.resource_classes

Partitions are site placeholders (${CPU_PARTITION}, ${GPU_PARTITION}), expanded where a worker is submitted; the site's values stay local. cpu: 4 CPUs and 16 GiB for 6 h, the CPU grant of the 1.x diarization jobs (specs/20260915-preprocess-diarization/design.md). cpu:large: 8 CPUs and 64 GiB, the host grant of scripts/speaker_ceiling/evaluate.sbatch, as the larger memory class a resource fault moves to. gpu:h100: one H100 with 8 CPUs and 96 GiB of host memory, the grant of specs/20260820-vlm-spectrogram/ job.sh; H100 because review arrays run there first. time_s 21600 on every class: not measured, the diarization grant's limit. memory_cap_bytes 85899345920 on gpu:h100: the H100's 80 GiB of device memory, which the models resident at once on one worker share. CPU classes have no static cap: their memory is the grant, and an OOM there is a resource fault that moves the item to cpu:large. gpu: any GPU type, with the gpu:h100 host grant and an untyped gres, listed after gpu:h100 so a plan never chooses it first; it is the class a call running on a GPU node of another type takes. Its device memory depends on the type, so it has no static cap. svc:h100: the gpu:h100 grant and cap, as a service class; a server backend (vLLM) is placed on it before any other class, so a `svc` pool worker starts and registers the server and every other worker of that identity on the host consumes it. The ladder's pii_spans/vllm entry ran on gpu:h100 because no service class existed and placement took the first class of the device: four gpu:h100 workers of one card each started an unregistered server at gpu_memory_utilization 0.9, and every start after the first failed for free memory. max_workers 4 as gpu:h100; on a shared device sizing bounds it further by device memory (orcd.sizing). max_workers (32, 8, 4, 4): not measured; the substrate's QOS cap bounds them further. No batch size: a placement records batch 1 (T20R-6); `size` sizes the batch where the work runs, from the unit's estimate and the length of each job, filling sizing.memory_headroom of one worker's share of the class's memory beside the resident model, against the device fit on a `cuda` class (its memory_cap_bytes, else the memory of the device seen) and the host fit otherwise, never the other (calibration-probe.md §6.4); batch 1 is the batching that needs no measurement.

### orcd.substrate

slurm: workers are Slurm jobs submitted per resource class. qos_cpu_cap null: the cap is the group's QOS MaxTRESPU CPU limit, which is account-specific and set in a site override; exclude_nodes empty, a site lists nodes it has found broken. start_backoff_first_s 1 and start_backoff_max_s 30: the wait after a class's first failed start, or after a failure of a Slurm command, doubling per failure in a row up to the maximum; the scaler does not use the substrate until it has passed, and past faults.infrastructure_replacements failures in a row the substrate is broken for the epoch. Not measured: the values the pool used before the keys existed, when it took this wait from pool.first_poll_s and pool.poll_s; the basis to measure is the site's submit-to-ready latency and how long a controller outage lasts. commands.timeout_s 60, commands.attempts 3, commands.retry_first_s 0.5: one Slurm command may take a minute, is tried three times, and waits half a second, then a second, between tries. Not measured: the constants the substrate carried before (SLURM_TIMEOUT_S, SLURM_TRIES, retry_s); the basis to measure is the p99 latency of sbatch and squeue on the site and the controller's failure rate. A submission is idempotent under these settings: each intended worker's job carries a name of its own, and after a try that fails or times out the name is looked up with squeue before the next try, so a try whose job was queued is never submitted again. The lookup reads `squeue --states=all`, which lists a job until slurmctld purges it MinJobAge after it ends (default 300 s), so the bound is: commands.timeout_s + commands.retry_first_s × 2^(commands.attempts − 1) must stay below the site's MinJobAge (`scontrol show config | grep MinJobAge`). With these values that is 60 + 0.5 × 4 = 62 s against 300 s. A job that started, ended and was purged inside that window never reached its run, so its duplicate is surplus that scale stops. A duplicate the controller registers after the lookup has the same name, is listed by `live` with the run's other jobs, and is cancelled when the fleet closes. One submission holds the scaler thread at most commands.attempts × (commands.timeout_s + one lookup), and a lookup is itself at most commands.attempts × commands.timeout_s plus its waits: (3 × 60 + 1.5) for the lookup, so about 3 × (60 + 181.5) = 725 s in the worst case; the first submission that fails ends that start call, so no further worker is submitted that round. While the substrate backs off, the scaler wakes when the wait ends rather than at the next poll_s. placement_hook null and placement_hook_s 1: senselab ships no site hook, a site names its own as `module:callable` (or a run gives `placement_hook`); the hook is called once per class per epoch, and a call that has not answered within placement_hook_s is a failed hook, which costs a run no more than one failed start because the bound equals start_backoff_first_s, the first step of the wait after a failed start. Provisional: the basis to measure is a real adapter's call time (on ORCD one reading live idle resources), after which the bound is its p99 with margin.

### orcd.pool

quantum_items 100 and quantum_s 300 are ceilings: `size` derives each unit's quantum from its measure (sizing.load_share) and holds it to at most a hundred jobs and to the jobs that fit in five minutes after the load (and in the class's time_s), and a pool worker stays on a class for at most that many jobs or seconds, so an older queue waits at most a ceiling quantum. Not measured: the fixed quantum the pool used before sizing; the ladder's per-item cost (Task 20 Step 4) shows whether a ceiling binds. poll_s 30: the 1.x reproduce controller's poll interval; a worker whose queues stay empty for three polls exits. first_poll_s (0.05 on laptop, 1 on orcd; Phase 3 Task 14 fix round): a coordinator's first wait for an outcome, doubled up to poll_s, so a local run does not wait 30 s per fragment; orcd starts at a second to keep stats off the NFS server. Not measured. segment_items 512 (Phase 3 Task 13 review, owner ruling): a claim lists its oldest live segment, so its cost grows with the segment, not with the queue. The Task 13 review measured claim plus commit on local SSD at 10^4 items queued: 22.4 ms with 10^4-item segments (about 1 MB of index read per claim) against 3.4 ms with 500-item segments. 512 keeps a segment's listing to about a thousand names (slots and their marks) while 10^4 items still cross twenty segments; the Step 5 run on ORCD revises it. presence_refresh_s (1 on laptop, 30 on orcd; Phase 3 Task 20R-5 fix round): the longest a pool worker reuses what it read of the ready marks of the workers on its host, the compute processes on its devices (`nvidia-smi`) and the probe queues of its run. A read lists the run's `workers` directory (each mark is read once), checks each peer's liveness and, on a `cuda` class, runs one `nvidia-smi` query, so a claim costs none of them; at 1 worker that is one listing and no check per interval, at 10^2 workers 10^2 liveness checks per interval instead of 10^2 mark reads per claim. Peers change on the scale of worker starts and exits, which `poll_s` paces: orcd takes `poll_s` (30 s); laptop takes one second, for a handful of local workers. Not measured: the interval's effect on the co-tenancy a short run reports, and a probe queue made inside the interval is served from the next one. admit_threads 8 (Task 20R-7a-0 fix round): the threads that ask the result store which of a plan's jobs it would serve, one `admits` (one `context.json` read) per job. The reads are latency-bound, so a thread count bounds open files and not CPU. Measured on local SSD only: 6.9 ms for 1 job and 0.47 ms per job at 10^2 (empty store, 8 threads); not measured on a network file system, where the ORCD run (T20R-10) sets it.

### orcd.sizing

load_share 0.05: a unit's quantum is the fewest jobs whose time keeps its model load and its claim within a twentieth of the quantum's, so loading and claiming cost a worker at most 5 % of its time (the calibrate-then-size design); with neither measured above zero the quantum is the ceiling. A target, chosen, not measured. memory_headroom 0.9: a batch fills at most nine tenths of the worker's memory beside the resident model: on a `cuda` class with a device measure, its memory_cap_bytes against the device pair; otherwise its mem_bytes, else on a local substrate the host's, against the host pair; the tenth left covers allocator fragmentation and a peak above the measured one. Not measured; the spread of peak memory over the 10⁴-item ladder point (Task 20 Step 4) replaces it. target_wall_s 1800: the wall time the workers of one queue aim to end its waiting work in, so a queue starts no more workers than end it in half an hour and none for work one worker ends sooner; a target, chosen so a 10⁴-item ladder point ends within one session. calibration_items 4: a unit with no resource-table row first runs four one-job items on one worker; the first carries the model load and the median of the other three, robust to one outlier, is the seconds per item. The least count whose warm items have a median that tolerates one outlier. Batch against throughput (Phase 3 Task 20e, 20g): a batch sized by memory is kept only where items sent in a batch were measured no slower per item than items sent alone; otherwise, and where the batched time is unmeasured, the batch is one (`size` gives both `memory_batch` and the `batch_size` kept). A calibration round whose items include no warm one-job item measures nothing and writes nothing: its load cannot be told from its work. A calibration round measures both: its calibration_items one-job items first, for the load, the claim and the memory; then calibration_items − 1 pairs on the worker the first round warmed, each two items of the memory-sized batch's jobs, one sent as one request and one sent a job per request, so each side has a median of three warm figures (robust to one outlier on each side), each figure one request call over as many jobs and inputs; the batch is kept only where the batched median per-item seconds of that second round is not above the one-per-request median, and at most at the batch that round measured, never the larger memory batch (P-34; Task 20k: speech_to_text~transformers and speechbrain were measured at 16 against a memory batch of 100, the quantum); a larger batch needs a ladder point with jobs enough to measure it, a 10³ GPU point for a batch of 100 (R-237) (Task 20h: one-job items sampled straight after the load read 2 to 5 times the steady cost at 9380b530; Task 20j: one-job items are gathered into one claim and share its wall, so they were one figure carrying a third of a call's fixed cost, R-234). A ladder's tables carry both, since the ladder's pool reads no packaged table and so calibrates every unit in the run. No slack term: the two medians are of one backend on one warm worker, the median absorbs one outlier per side, and a wrong call errs to batch 1, which is never over memory. Cost bound: once per unit per pool (a written row skips it), 2 × (calibration_items − 1) × batch jobs, all real work whose results are kept, and 2 × (calibration_items − 1) claims. Device memory per item is the backend's own request peak over what stays allocated after the request (`torch.cuda.max_memory_allocated` against `memory_allocated`, reported in `done`), divided by the request's items; a server backend, which reports neither, keeps the device read of the worker's processes after each claim (its fixed reservation); a model backend that reports neither (no CUDA initialised in torch) leaves its device memory unmeasured, so its batch is 1, and the pool logs it once per unit. calibration_workers 1: a queue whose units are all unmeasured is given one worker, so the warm items follow the cold one on the same worker and the load is read once; a queue mixing a measured unit with an unmeasured one is sized by the measured one and may run the round on several workers, each giving a cold sample, which `calibrated` reads as well. Workers of a `cuda` class whose substrate shares devices (local: the GPUs the coordinator sees, one per worker as `CUDA_VISIBLE_DEVICES`) are at most devices × floor(memory_cap_bytes / measured device peak), at least one per device and one while the device peak is unmeasured: the measured peak is what one worker holds, so the cap divided by it is how many fit a card. No new key; Slurm workers each hold their own card, so the bound is max_workers there. device_query_s 30: a pool worker reads its processes' device memory once per claim on a `cuda` class with `nvidia-smi --query-compute-apps`, bounded by the hardware probe's `nvidia-smi` bound; not measured, the basis is the command's p99 latency on a loaded GPU node. T20R-6: `size(estimate, grant, config, work, calibration=)` reads these keys as follows. load_share and the item's per-claim cost `a` close an item once its median seconds reach (load_s + a)/load_share; an item also closes before its upper-predicted seconds t·(1 + z·spread) plus `a` pass the grant's quantum_s, with z = calibration.confidence, so about one item in twenty overruns at the configured one-sided 5 %, and at quantum_items jobs; with the slope `u` or the spread unmeasured an item is cut at quantum_items alone and an unmeasured figure is never read as 0. target_wall_s gives the workers: ceil(sum of median predicted seconds / target_wall_s), at most the item count, max_workers and, on shared devices, devices × w_d, where w_d = floor(memory_headroom × cap / P) for the batch peak P = resident + m0 + m1·B*·L_max at the run's longest job, lowered where a measured regime factor says k/ρ_k stops increasing; each worker's memory is cap / w_d, so the workers of one card together stay within memory_headroom of it. calibration.unmeasured_workers gives the workers of a unit with no estimate, which runs the safe values (batch 1, items of quantum_items jobs). No new key. The Pool still reads `size_row` until T20R-7a; calibration_items, calibration_workers and the `Measure`-based derivation above go with T20R-7a′.

### orcd.calibration

Task 20R-1, design `calibration-probe.md` §11; every value is the design's. Values marked provisional are replaced by Task 20R-10's measurement under the stated rule (ruling O-6).

measures_roots: the group's shared root, `${SENSELAB_MEASURES_ROOT}` with no fallback (a fallback under `~` would give each member a private root and `trust: group` nothing to pool; an unset variable is no root, logged once per run and never raised), then the user's own `${cache_root}/2x/measures`, where `${cache_root}` is senselab's cache root (`SENSELAB_CACHE`, else `~/.cache/senselab/hf`; `revision.cache_root()`), resolved by `resolve_roots`, so the default is written once. A root must be absolute after expansion: a blank root, a repeat and a relative entry are refused at load, and a variable that is unset or empty is no root (logged, never raised). Read in order and pooled; writes and leases go to the first writable root, so a read-only member reuses the group's measures and writes its own (§3.3). trust group: a cluster group reuses its members' measures; a record kind that supersedes or caps (`drift`, `supersede`, `regime`, `oom`) acts only for readers whose filter admits it. A run overrides it with the run config's `calibration.trust` (ruling O-3, §3.5). warmup_requests 2 (provisional): the least w such that, on the table probe, the median of warm requests w+1..w+3 of one stratum lies within the steady band of the requests after them, the maximum over GPU units; the basis until measured is R-230 (CUDA initialised inside the first request) plus one request for shape-dependent kernel selection, R-231's decay being confounded by gathering (R-234). length_strata 2: two points fit a line in length (must-fix 8, R-221); the table profile uses 3 to test linearity. min_requests 3: the least count whose median tolerates one outlier. single_item_jobs 4 (provisional): items of 1 and k jobs separate the per-item cost `a` from the per-job cost; k is such that `a` is at least a fifth of a k-job item's wall at the measured CPU fit (R-236: `a` 0.040 s against 0.0366 s per job). batched_sizes 1: run time measures B₁ only; the table profile measures 2 (R-237). probe_items 200 (provisional): the most jobs a probe plans, a bound; chosen as `warmup_requests + length_strata × min_requests × (1 + B) + min_requests × single_item_jobs` at a target batch B = 30, a target until Task 20R-10 re-derives B from the measured memory batches (N4); the table profile sizes it for B = 100. confidence 1.645: one-sided 5 %, the false-batch rate of the batch decision (§5.4 step 6) and the share of items cut past `quantum_s` (§6.2); a wrong batch decision errs to batch 1. drift_confidence 2.576: two-sided 1 %, at most one spurious confirming probe per hundred windows of a stable unit. observations_kept 3: superseded probes kept for audit; three give a median of medians that tolerates one bad probe. unmeasured_workers 1: unmeasured work is not scaled (S5); one worker runs it under the safe values. resolve_s 1800 (provisional; Task 20R-7a design amendment, 2026-11-08): the longest a unit waits in `Resolving` after a worker of its queue is live and not pending, i.e. the longest a live worker takes from its start to its ready mark, which follows the env builds it does first (§7.1, §8). The basis until measured is two env builds, each at the build waiter's bound of `lease.stale_after_s` plus the lock timeout (300 + 600 s, itself 1.5 times the slowest measured cold build, 590 s); a queue that needs more builds goes unmeasured, which costs efficiency and changes no result. Task 20R-10 replaces it with the measured start-to-mark wall, the maximum over the profile's classes. Not keys, derived from other keys: `probe_s` = `sizing.load_share × sizing.target_wall_s` (0.05 × 1800 = 90; probe work, like loading, may cost a queue at most a twentieth of its target wall; it counts warm probe requests only, §4.5), `peer_wait_s` = `probe_s + lease.stale_after_s` (90 + 300 = 390; a peer's probe ends within its budget or its lease goes stale) and `drift_window` = `length_strata × min_requests` (6; drift is judged at the precision of the probe's single phase). They are properties, so no profile can disagree with them.

### orcd.lease

Not measured until Task 16, which measures the store root's attribute delay that bounds both values. stale_after_s 300: five times the NFS client's default maximum attribute-cache time (acregmax and acdirmax, 60 s), so a beat directory a live holder made is seen well inside the bound on a shared root. renew_fraction 0.33: the 1.x work pool's rule that a heartbeat is at most a third of the lease, so two missed renewals still leave the lease live. The one place lease settings live: `pool.lease_renew_fraction` and `envs.lease_stale_after_s` were merged here (Task 12), and every lease holder (env build, vLLM start, run manifest, label map) beats under `Lease.keep_alive` every `renew_fraction × stale_after_s`. The env build lease used to be stale after 1200 s, about twice the slowest measured cold build of the crisperwhisper env on ORCD (362.6 s GPU node, 579.7 s and 590 s CPU node, n=2 per host; specs/20260907-shared-lock-heartbeat-inf/ stampede-timeout-and-identity-file.md); a beating build no longer needs a bound that covers it. A waiter on a build waits stale_after_s plus the lease timeout, 300 + 600 = 900 s, which is about 1.5 times the slowest measured build (590 s, on a loaded CPU node); the padding is less than the old 1200 s carried because the waiter no longer also waits out a stale bound sized to the build. `ExecutionProfile.load` refuses a profile where 2 × renew_fraction × stale_after_s + attribute_delay_s ≥ stale_after_s: the factor 2 allows one missed beat, and a client may see a beat up to attribute_delay_s late. Read by `PydraRunner` for pydra's job leases (P3-53); the pin at c8541fc5 has no `LeaseSettings` yet (Task 45), so the values are held and not yet passed. attribute_delay_s 60: the NFS client default acdirmax; to be replaced by Task 16's measured value. With the defaults, 2 × 0.33 × 300 + 60 = 258 s < 300 s.

### orcd.faults

content_attempts 2: the 1.x work pool's DEFAULT_MAX_ATTEMPTS, an item failing for its content twice goes to dead/. infrastructure_replacements 6: the 1.x work pool's DEFAULT_MAX_INTERRUPTS, releases and reaps allowed before an env is reported broken. resource_replacements 1: one per larger memory class the profile lists (cpu:large), so a resource fault walks the classes once and then fails.

### orcd.timeouts

Null, no bound: no measurement of the slowest model load or the slowest single request exists. A site's own ladder run (`--measures-root`) gives load_s and the per-input time from which it sets both; a timeout is a resource fault, so a bound set too low re-places work rather than failing it.

### orcd.services

idle_exit_s 600: ten minutes, twice the pool quantum, so a service outlives a consumer's yield to another queue and the next quantum's requests find it up. Not measured: no measurement of how long consumers leave a service idle exists.

### orcd.supervisor

start_time_probe_s 5: as laptop; the longest `ps -o lstart= -p <pid>` may take when a supervisor reads a launched process's start time off Linux (Linux reads /proc/<pid>/stat field 22 and needs no bound). Over 200 calls on a 10-core laptop at load average 4.4 it took a median 2.9 ms and at most 4.8 ms (2026-11-07), so 5 s is about a thousand times the slowest call and only a hung or starved `ps` reaches it. A read that does is an unknown start time; the owner records it as unknown, and the keeper never kills a process whose start time is unknown at launch or at the end, so a bound set too low leaks a process at its owner's death rather than killing one that reused its pid. `lstart` has one-second resolution: a pid reused within the same second as the original start would match, which needs the pid space to wrap within a second.

## test_slack

`packages/senselab-core/tests/data/test_slack.yaml`, layered over the packaged orcd profile by the execution tests.

The `slack` fixture of `packages/conftest.py` returns the requesting package's `tests/data/test_slack.yaml`.

### test_slack.substrate

commands.timeout_s 1: the time the hang test gives a fake `sbatch` that queues its job and never answers. The fake is one Python process that writes its job with a single rename before it waits, so every positive value leaves exactly one job: killed before the rename it queued nothing and the retry queues one; killed after, the lookup finds its job. The value only decides whether the test exercises the second case, which needs the interpreter to start and rename within it. Starting the interpreter with the fake's imports took a median 24 ms and at most 28 ms over 50 runs on a 10-core laptop at load average 4.6 (2026-11-07); 1 s is about 35 times the slowest, and the test takes about that long. The same timeout bounds the fake `squeue` and `scancel`, which are the same interpreter start with no wait, so the same measurement covers them; a timed-out call there raises `SlurmError` and fails the test, which is the signal that the margin was wrong.

### test_slack.pool

first_poll_s 0.01 and poll_s 10: the device-pinning test reads what its local worker processes wrote every hundredth of a second and fails after ten seconds. Each worker is one interpreter start and one file write, about 30 ms on a loaded laptop (the measurement in test_slack.substrate); 0.01 s keeps the poll below that start, and 10 s is over 300 times it, so only a worker that never starts fails the bound. Not measured on CI runners.

### test_slack.tests

Values only tests read; the `slack` fixture leaves them out of the profile override and `slack_tests` returns them.

probe.fit_ops_ratio 25: the most the pooled-fit scaling test lets the slope fit's step count at 10^4 requests reach over its count at 10^3. The test counts the steps the code charges itself (`Work`): each merge level of an inversion count charges its n values, each merge of the final pair listing charges its run, and each pair listed and each tie-path value charges one. The count does not depend on the host. The fit bisects over the doubles, so the number of inversion counts depends on n only through the stop rule (at most n slopes left between the bounds). On the test's seeded data the counts are exact: 461,740 steps at n = 10^3 and 7,558,156 at n = 10^4, a ratio of 16.4 (2026-11-09). An n log n fit is about 13 and an n log² n fit about 18 (10·(log 10⁴ / log 10³)²); a quadratic inversion count or a stop rule that lists a constant fraction of the n(n−1)/2 slopes multiplies by 100. 25 is 1.5 times the measured ratio and a quarter of the quadratic one. Checked by mutation (2026-11-09): an O(n²) inversion count and a stop rule that lists a constant fraction of all slopes each make the test fail.

sizing.relative_tolerance 1.0e-9: the relative slack the sizing-invariants test allows its float comparisons (a cut's upper seconds plus the per-item cost against the quantum, a card's workers' budgets against the headroom of its memory). Those are sums of at most forty jobs' predictions (the test draws up to forty lengths) and one product, each rounding by one part in 10^16 at most; 10^-9 is seven orders above that accumulation, and 10^-9 of a 300 s quantum is 3·10^-7 s, orders below any excess the invariant is there to catch. Not measured.

lease_clock.poll_s 0.0005: the real seconds a test thread parked on an injected `ManualClock` waits between looks at its event. It sets no bound and asserts no time: the clock advances only when a test advances it, and this poll is how a parked thread notices. Small enough that a test of a few hundred lease steps finishes in well under a second; large enough that a parked thread does not spin a core. The hang guard that turns a never-released lease into a failure is `tier_bound`, not this value (testing.md, time in tests).

## pyproject.toml

The root `[tool.ruff.lint]` limits, which `packages/ruff.toml` extends. A site over one carries a `noqa` with its reason;
a script that needs a split carries a per-file ignore naming its register entry.

### tool.ruff.lint.mccabe

max-complexity 10: McCabe's own recommended ceiling (McCabe 1976, IEEE Trans. Software Eng. SE-2(4), 308-320, doi:10.1109/TSE.1976.233837) and
ruff's default. The packages met it before it was enforced from the root, except the sites that carry a `noqa`.

### tool.ruff.lint.pylint

max-statements 50: the size budget's function limit of 100 lines at two lines a statement (engineering.md, size
budgets). max-branches 12, max-returns 6 and max-args 5: pylint's defaults, which the packages met before they were
enforced from the root, except the sites that carry a `noqa`.

## scripts/data/size_budget.yaml

### src_file_max_lines

A reviewer reads a module whole in one pass (P-32). The read tool returns 2000 lines at a time; 1200 leaves the reviewer's own notes and the module's test beside it inside one window. Derivation: engineering.md, size budgets.

### test_file_max_lines

A test file is read beside its subject, so it may exceed the source limit by one function limit: 1200 + 100. Derivation: engineering.md, size budgets.

### src_function_max_lines

ruff caps a function at 50 statements (PLR0915); at two lines a statement, for a wrapped call or a literal, that is 100 lines. Derivation: engineering.md, size budgets.

### test_function_max_lines

The same 100 as the source limit: a test function past it is a scenario that wants splitting. Derivation: engineering.md, size budgets.

### exempt

A function or file over its limit on purpose, each with its reason in the file: a literal table, or vendored code kept diffable (engineering.md, size budgets). Each exemption is re-read at each phase gate.

## scripts/data/fixture_budget.yaml

### file_max_bytes

docs.md, "Data policy: toy fixtures versus pulled data", budget (design call): at most 256 KB per file. KB is read as 1000 bytes. It applies to every tracked file under packages/, code included, since no source file of ours approaches it and a file that does is data.

### data_total_max_bytes

docs.md, same section and design call: at most 2 MB in total. MB is read as 1000 * 1000 bytes. It totals the data files, every file that is not .py, .pyi or .ipynb, because the policy budgets fixtures, and code is bounded by the per-file limit.

### exempt_prefix

The budget covers test fixtures. The env locks are package data, generated by scripts/lock_envs.py from pinned, hashed requirements; the vllm-server lock is 408 KB and cannot be made smaller without dropping pins. Files under this prefix count against neither limit.

## scripts/data/license_policy.yaml

### on_missing

`raise`: a package with no license record fails the gate (P-23). `warn` is a relaxation, made only as a visible edit of the file.

### sets

permissive: OSI-approved licenses with no copyleft condition on the code that links or imports them: attribution and notice only. CC0-1.0 (a public-domain dedication) and 0BSD and Zlib appear because numpy's wheel lists them for vendored sources.

permissive-other: Notice-only licenses outside `permissive` that the locked env packages carry: MIT-CMU (pillow), CNRI-Python (regex, with Apache-2.0), the LLVM exception to Apache-2.0 (llvmlite, nvtx) and the Repoze BSD-derived license (supervisor). None carries a copyleft condition.

vendor-eula: NVIDIA's end-user agreements for the CUDA runtime libraries and the CUTLASS DSL, which a backend env installs at run time as dependencies of torch and vllm and which senselab never redistributes. Proprietary and not open source; named so only the env locks, never a shipped distribution, accept it.

runtime-exception: GPL-3.0-or-later with the GCC Runtime Library Exception, which lets code that is not GPL link the compiler's runtime libraries (libgfortran, libgcc_s) that numpy's Linux wheels bundle. The exception removes the copyleft condition for the program that uses them.

weak-copyleft-ok: File-level or library-level copyleft: changes to the licensed files, or to the library itself, are shared under the same license, and linking from other code is permitted. MPL-2.0 is file-level, LGPL is library-level (libsndfile, libquadmath).

copyleft-ok: Strong copyleft: the GPL and AGPL families. No shipped package may need these today; the set is named so a policy can opt in to it.

non-commercial-ok: Licenses that forbid commercial use. Not allowed for any shipped distribution; named so a policy can opt in.

### distributions

The sets each 2.x distribution allows for its locked runtime dependencies, and `allow_bundled` for the native libraries its wheels bundle. senselab-worker, senselab-core and senselab allow the permissive sets and weak copyleft; the env locks also allow the vendor EULAs, which senselab never redistributes (P-23).

### aliases

Spellings of a license in a package's `License` field mapped to SPDX ids, as found in the locked packages; the field is used only when it is one line and parses as an expression of known ids or aliases.

### classifiers

Trove license classifiers mapped to SPDX ids, for packages whose metadata carries no expression.

### manual

Dated records for a package whose installed metadata names no license, or that is not installed on every platform; each names its source and how it was verified. A manual record is the fallback and never overrides metadata.

### bundled_libraries

The license of each native library a wheel bundles, found from the wheel's RECORD, keyed by the bundling package; each names its source and how it was verified.

## scripts/data/default_python.yaml

### default

The owner's ruling: senselab has a single default Python, the Google Colab default runtime Python. Colab runtime 2026.10 ships Python 3.13.18; the upgrade went live on 2026-08-25 (googlecolab/colabtools#6081). Source: https://research.google.com/colaboratory/runtime-version-faq.html (runtime 2026.10, Python 3.13.18).

## packages/senselab-core/src/senselab_core/runtime/data/store_policy.yaml

### world_readable

Store entries hold participant-derived outputs, and fetch blobs may hold participant recordings. Neither is shared outside the group that runs the analyses by default, so committed objects and the directories above them carry no world bits. A site that publishes a public store sets this to true; that adds read (and, on directories, search) for others and nothing else.

### quarantine_retention_s

Seven days. Quarantine holds only entries that failed a completed verification; nothing reads them, so the age bounds disk use, not correctness. One week lets someone inspect what was quarantined after a weekly batch before `collect` removes it.

### staging_stale_s

One day, as before this policy existed. A staging directory or download lives for one `put` or one transfer. A transfer of the largest dataset in a manifest is the longest of these; a day is above any single transfer we run, so an older staging entry belongs to a writer that crashed. Ages are judged against the storage's own clock, so clock skew between nodes does not shorten it.

### collect_interval_s

One hour. A collection makes and removes a probe directory to read the storage clock and lists quarantine/ and tmp/, which costs metadata round trips on NFS; running it on every put multiplied them by the put rate. What it removes is at least a day old (staging_stale_s) or a week old (quarantine_retention_s), so collecting hourly delays a removal by at most an hour against ages of a day or more.

### measures_file_bytes_max

4 MiB. The most one measures file a reader takes; a larger file is malformed (quarantined by its owner, skipped by anyone else), so a group member's oversized file cannot exhaust a coordinator's memory. The largest legitimate file is one probe's observations: at most `calibration.probe_items` observations (200 on orcd) of under 0.5 KiB each (429 bytes serialised with a cuda `HardwareClass`), about 0.1 MB. The bound is that times forty, headroom for a larger `probe_items`, longer hardware strings and a later minor's added fields, still far below a size that matters to a process's memory.

## packages/senselab-core/src/senselab_core/runtime/data/launch_environment.yaml

### linker

Cluster CUDA, cuDNN and MPI libraries are found through the dynamic linker's search path, set by `module load`; without it a torch or vLLM import fails to load its shared objects. It selects which build of a library is loaded, not what a pinned library computes.

### tls

Sites behind an inspecting proxy install their own certificate authority; a model or data download without it fails verification. Trust roots change whether a connection opens, not the bytes a content-addressed fetch keeps.

### proxy

Compute nodes without direct egress reach the Hub only through the site proxy. Both spellings are read by different HTTP clients. A proxy changes the route, not the pinned bytes.

### hugging_face

The Hub cache location, token and offline switch. Every model load passes a commit, so these change where the weights are read from and whether they may be fetched, never which weights are loaded.

### gpu

The scheduler hands a job its devices as `CUDA_VISIBLE_DEVICES`; a worker that drops it sees every device or none. These select devices and CUDA behaviour (`CUDA_LAUNCH_BLOCKING`, `CUBLAS` workspace via `CUDA_`), which can change numerics, so they are recorded. They are not `measured`: which device a worker is given changes no cost, and the hardware class names the device (calibration-probe.md §3.1).

### threads

Slurm jobs set thread counts to their CPU allocation; without them BLAS oversubscribes the node. Thread counts can change the order of floating-point reductions, so they are recorded. The three counts (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`) also set per-item time, so they are `measured`: a measure unit holds them (calibration-probe.md §3.1). The family's other variables are not: they name paths (`OMP_TOOL_LIBRARIES`), CPU placements (`OMP_PLACES`) that would split a unit by slot, or diagnostics.

### locale

Text decoding of tool output and file names follows the locale. Results are computed on decoded values, not on locale-formatted text.

### process

The home directory holds caches and credentials, the temporary directory holds scratch files, and the path finds tools such as ffmpeg. None of them reaches a value a backend computes.

### never

These choose which interpreter, packages and startup code run, so a caller's own would shadow the locked env and make its recorded `env_lock` and `wheel_set` false. No family may pass them.

## packages/senselab-core/src/senselab_core/runtime/data/run_config.yaml

### thresholds

time_ns: One millisecond, in the nanoseconds every kernel time is held in. The ORCD runs behind the retired tolerance profiles found no difference in turn and segment times across repeat, batch, CPU and CUDA (diarize pyannote, jobs 27661203 and 27661204; speech_to_text transformers, 27661289 and 27661304; crisperwhisper, 27661305, 27661306 and 27661308). A boundary that moves by a millisecond or more is a different boundary.

probability: A confidence in [0, 1]. The largest CPU-against-CUDA difference measured in a segment confidence was 0.00385 (speech_to_text transformers, jobs 27661289 and 27661304) and 0.00079 (crisperwhisper, jobs 27661305, 27661306 and 27661308); 0.01 is above both and below any change a reader of a confidence acts on.

score: A span or annotation score. PII span scores did not differ across repeat and batch (pii_spans transformers, job 27661205; vllm, job 27661206), so the bound is float32 rounding: six significant digits.

embedding: Each component of an embedding vector or trial row. The largest CPU-against-CUDA difference measured in a speechbrain speaker embedding was 0.0858 (jobs 27663219 and 27663927, at 9d4a3304); repeat and batch differed by 0. 0.1 is the next written value above it.

samples: A signal sample. Resampled series did not differ across repeat and batch (signal.resample numpy-sinc, job 27663928). 1e-6 is about eight float32 steps at full scale (float32 epsilon is 1.19e-7).

value: Any other number a kernel type holds: a metric or annotation value, an uncertainty or a clock drift. Nothing was measured for these; the bound is agreement to six significant digits.

## packages/senselab-core/src/senselab_core/runtime/data/equivalence.yaml

### backends

The lines `scripts/regenerate_equivalence.py` prints beside the table it writes, one per class: the verify evidence that joined its members.

pyannote.class-0: verify sampling: 2 agreements and 0 disagreements over 2 keys, runs p2cmd-27735059, thresholds 674c9fc33cbd23871f5e182c9fc244d562a2fbdff4b66d31d201cc5cc72045c7

transformers.class-0: verify sampling: 6 agreements and 0 disagreements over 6 keys, runs p2cmd-27735057, p2cmd-27735059, thresholds 674c9fc33cbd23871f5e182c9fc244d562a2fbdff4b66d31d201cc5cc72045c7

## benchmarks/scale/budgets.yaml

### flat_ratio

1.25, still: one plus the largest spread of a per-item wall time between full runs of the same point under the same sizing, rounded up to a quarter. Spread is (max − min) / min per (task, backend, point) over the runs that hold its result file: d0c1203e (CPU ladder 28257809, fake stack 28257810, H100 28257811), e801bc6c (28262078, 28262079, 28262080) and 11e55ebd (28266570, 28266571, 28266572). The first two sized their pool by calibration rounds; 11e55ebd sized it from the packaged tables generated at e801bc6c, so for the packaged backends the third run is a change of sizing, not a repeat. The fake stack has no packaged table and was sized by calibration in all three: fake.echo~fake reads 0.02653, 0.02840 and 0.02673 s at 10² (7.06 %) and 0.01535, 0.01817 and 0.01565 s at 10⁴ (18.36 %). Between the two calibration-sized runs the packaged backends read at most 2.49 % at 10² (pii_spans~transformers; signal.resample 0.32 %, diarize 0.50 %, speaker_embeddings 0.42 %, speech_to_text~crisperwhisper 0.07 %) and 1.32 % at 10⁴ (signal.resample). The largest, fake.echo~fake at 10⁴ (18.36 %), rounds up to 25 %. Excluded: speech_to_text~transformers and pii_spans~vllm at d0c1203e (files lost to the result-name collision, fixed at e801bc6c); the one-item points, which carry a worker's start; the `long` point (R-225). Under the table sizing of 11e55ebd, signal.resample ran faster (0.0138 against 0.0175 s at 10⁴) and two GPU entries slower (speech_to_text~crisperwhisper 2.141 against 1.365 s, speaker_embeddings~speechbrain 0.150 against 0.104 s at 10²): a change of batch, registered as R-226, not spread. Peak-memory flatness stays over the ratio in all three runs (signal.resample 1.487, 1.662, 1.442; fake.echo 2.864, 2.896, 2.952): R-224's growth with plan size. The files and bytes ratios agree between runs to three places but for signal.resample files at 11e55ebd (0.151 against 0.156), where the table's quantum changed the item count. The regression band measured from repeated idle-node runs replaces this ratio once the nightly ladder repeats a point.

### synthetic tree

The `signal.resample` ceilings are measured on `bids_tree.synthetic` at seed 0: 16 kHz triangle tones, lengths drawn in samples between 1 and 4 s without replacement (48 001 lengths, so every tone's content is its own; R-232) and periods of 16 to 400 samples, pinned by the digest of `synthetic(root, 3, 0)`, `d9fabd59a9907c1ffd7afea617a5327ba9deb56d5be2947e7a80d49a9e63ebb9` (`TREE_DIGEST`, Task 20i); ceilings measured on the earlier tree (1 540 distinct tones) are not of 10⁴ items.

### regression_band

Empty: no module has repeated idle-node runs under one sizing yet; each is set from the spread of repeated runs once the per-PR ladder runs.

### per_item

Basis (Task 20R-8): the wall seconds per item are `per_item_s`, the throughput wall per production job, which `scale-result@2.0` computes as the production span (first production claim to last production commit, both on the workers' clock from the items' samples) over its production jobs. It is the quantity the ceilings below were measured as (wall across all workers per job), with the idle time between items in it; `worker_s_per_item`, worker seconds per job, is a separate figure that no ceiling reads. A ladder re-derives a ceiling it re-measures.

Per suite entry label, the per-item ceilings: wall seconds, peak memory bytes, files and bytes written per item, each a measured value times 1.25 (flat_ratio's spread), rounded up to three significant figures (Task 20k). The ladder judges each ceiling at the basis it was set from and names it on the line (`flatness`).

- CPU and fake entries, from their 10⁴ point at a9fc9adc (jobs 28300055, 28300056): the wall seconds per item after calibration, the peak memory, files and bytes written per item. The 10² point's per-item seconds carry the pool worker's idle poll (R-233) and are not a basis. signal.resample~numpy-sinc 0.04151 s, 410 476 544 B, 3.10, 84 864 -> 0.0519, 514 000 000, 3.88, 107 000; fake.echo~fake 0.01190 s, 287 432 704 B, 2.17, 3 681 -> 0.0149, 360 000 000, 2.72, 4 610.
- GPU entries, from their 10² point at b7c25c4c (job 28305384, H100): the wall seconds are the calibration's second-round median in the regime it chose (`calibration_per_item_s`), since its round took every new item at 10² (four one-job items and three pairs of two 16-job items), or all but 24 (pii_spans transformers, pairs of 12), and the other items after it are retries of failed documents; the other three are the 10² point's. speech_to_text~transformers 0.1967 s, 2 691 407 872 B, 2.83, 5 640 -> 0.246, 3 370 000 000, 3.54, 7 050; crisperwhisper 1.1079 s, 372 764 672 B, 2.83, 5 958 -> 1.39, 466 000 000, 3.54, 7 450; speaker_embeddings~speechbrain 0.03286 s, 376 365 056 B, 3.83, 6 468 -> 0.0411, 471 000 000, 4.79, 8 090; diarize~pyannote 0.04479 s, 380 145 664 B, 5.83, 6 284 -> 0.056, 476 000 000, 7.29, 7 860; pii_spans~transformers 9.791 s, 422 572 032 B, 6.84, 11 627 -> 12.3, 529 000 000, 8.55, 14 600; pii_spans~vllm 2.893 s, 401 915 904 B, 6.76, 12 488 -> 3.62, 503 000 000, 8.45, 15 700.

Flatness at these results: per-item seconds at 10⁴ over 10² read 0.118 (resample) and 0.037 (fake), not a flatness figure while the 10² point carries the poll wait (R-233); peak memory reads 1.73 and 2.80, above flat_ratio, as in every ladder since e801bc6c (fake 2.80 to 2.95; R-235); files and bytes per item read 0.83 to 0.98. Every ceiling holds at its basis. The CPU peak-memory ceilings, set at 10⁴, carry R-235's growth (fake 2.80 times its 10² figure) and are derived again once R-235 is fixed.

## Measured resources

There is no `packages/senselab/src/senselab/data/resources` directory: senselab ships no measured rows (§3.7 of `calibration-probe.md`, owner ruling 2026-11-09). A measure is read only on a unit's whole identity, so a row shipped with the package answered for one configuration and every other machine probed anyway. A group that wants its resource measured runs the scale ladder on it with `--measures-root` naming a shared measures root and lists that root in its profile's `calibration.measures_roots`; the root is ordinary measures data, not config, and carries no derivation section. The seven `@1.0` tables written at Task 20k and the `@2.0` format that replaced them are deleted.

## benchmarks/scale/suites/table.yaml

Also `benchmarks/scale/suites/gpu-table.yaml`, whose `calibration` and `point_wall_s` are these, over the Phase 2 backends at 10³ items, the point at which the table profile's memory batch and second batch size are measured (R-237, R-221). `suites/ladder.yaml` and `suites/gpu.yaml` carry no table profile: they are the flatness ladders, whose points keep production items.

### calibration

The suite's table profile; it overrides `length_strata`, `batched_sizes` and `probe_items` and no seconds key. `length_strata` 3: the run-time probe fits a line through two strata, so a third tests its linearity in length (§11). `batched_sizes` 2: the batched phase measures two batch sizes (R-237), so the chosen batch rests on a measured pair. `probe_items` 923: the jobs of the minimal protocol at a target batch B = 100, `warmup_requests + length_strata·min_requests·(1 + B) + min_requests·single_item_jobs`, with the profiles' `warmup_requests` 2, `min_requests` 3 and `single_item_jobs` 4: 2 + 3·3·101 + 3·4 = 923. A test pins the arithmetic against the profile's values.

### point_wall_s

900 *provisional*: the seconds of warm probe work a point's probe may take, `Probe.plan`'s seconds budget in place of the profile's `probe_s` (`load_share × target_wall_s` = 90). The table probe runs up to 923 jobs at two batch sizes over three strata, which 90 s would end early; 900 is half of `sizing.target_wall_s` (1800), the most of a worker's target wall one unit's probe should take. Task 20R-10 replaces it with the maximum measured warm probe seconds over the ORCD ladder's points.

## packages/senselab-core/src/senselab_core/runtime/data/batching.yaml

### id

The grid's name, which every bucketed formation `bucket:<id>:<index>` carries into its key. It names the edges below, so a change of edges takes a new id and never reuses keys made under the old buckets.

### edges_ns

1, 2, 4, 8, 16 and 30 s. Not measured. Doubling edges keep, in each interior bucket ([1, 2) s to [16, 30) s), a batch's shortest item at least half its longest, so padding stays under half of a padded batch's samples there; bucket 0 (under 1 s) and the last, unbounded bucket (30 s and over) have no such bound. The top edge is the 30 s input window of the Whisper encoder behind the transformers ASR backend, the one Phase 2 backend that keys its batch formation; longer inputs share the last, unbounded bucket. The basis to measure is the Phase 3 Task 18 ORCD comparison: batch of 1 against bucketed batches on the held-out split, by kernel type under the run-config thresholds. Batch sizes are not here: they are the placement's, sized from the resource tables (`size`, Task 20). The edges are recalibrated from the Task 20 Step 4 ladder: per-item cost by input length on the transformers backend, with padding cost per bucket; a changed grid takes a new id.

## packages/senselab/src/senselab/data/env_locks/inputs.yaml

### quarantine

PyPI withdrew lightning 2.6.2 and 2.6.3 as malicious; a resolution must never pick them, whichever package pulls lightning in. The constraints are part of every lock's digest, so changing them stales every lock.

### base_requirements

The worker runtime (senselab_worker.codec and every backend's arrays) imports numpy, and the worker overlay adds no distribution, so the lock must provide it whether or not the backend requires it.

### envs

Each backend env's Python, requirements, upload cutoff (`exclude_newer`) and optional CUDA ceiling (`max_cuda`). The requirements are the backend's own; `exclude_newer` pins the resolution so a relock reproduces it; vllm-server caps CUDA at 13.0, the newest index its wheel supports. Every field enters the lock digest.

## packages/senselab/src/senselab/signal/data/resample.yaml

### kaiser

Window shape beta and length follow Kaiser's design formulas (Kaiser and Schafer 1980, IEEE Trans. ASSP 28(1)): beta = 0.1102 (A - 8.7) for A > 50, 0.5842 (A - 21)^0.4 + 0.07886 (A - 21) for 21 <= A <= 50, else 0; taps = (A - 7.95) / (14.36 * width) for transition width in cycles per sample.

### stopband_attenuation_db

The Kaiser design target A: the filter's stop band lies at least this far below its pass band, and its pass-band ripple is at most -20*log10(1 - delta) dB with delta = 10**(-A/20), the Kaiser target for the ripple in both bands; the realised ripple is within ripple_margin times it. 80 dB puts delta at 1e-4, a ripple of 0.00087 dB and an aliasing floor below the 16-bit quantisation floor of 98 dB relative to full scale for content within 18 dB of full scale; it is a design choice, not a fit to any recording.

### transition_fraction

The transition band's width as a fraction of the lower of the two Nyquist frequencies. It ends at that Nyquist (the stop-band edge) and starts below it, so a component that aliases lands at or above the pass-band edge and is attenuated by A. 0.1 keeps the pass band flat to 90 % of the lower Nyquist (14.4 kHz of 16 kHz audio resampled to 32 kHz) at about 16 000 taps for 44.1 to 48 kHz.

### ripple_margin

The factor by which the realised filter's worst ripple, in either band, may exceed the Kaiser target delta. Kaiser's length and beta formulas are an approximation and `ceil` rounds the length up; measured on the FFT of the taps of every pair of 8000, 11025, 16000, 22050, 32000, 44100 and 48000 Hz, the worst pass-band deviation is 1.160 delta (16000 to 48000 Hz) and the worst stop-band peak 1.040 delta. The worst over every up and down ratio up to 8, for five further base rates, is the same 1.160. 1.2 is that maximum rounded up to one decimal, a headroom of 3.5 %.

### max_block_elements

The most (channels x outputs x taps) values one gather holds at once, 2**22 float64 values (32 MiB): a memory bound that changes no output.

## benchmarks/exit/exit.yaml

### plan_profiles

`laptop` and `orcd`: the two profiles Phase 3 Task 21 Step 2 plans one scenario under. They differ in substrate, resource classes and env roots, which are all placement, so the check is that nothing but placement differs.

### steps

The scenarios of Task 21 Steps 2 and 3 as the plan states them: `Resample(target_rate=8000)` over `bids_tree.synthetic(n=1000, seed=0)`, and `SpeechToText()` with its default model over the first 1000 utterances of the `librispeech-test-clean` manifest. 8000 Hz is half the tree's 16 kHz, so every tone is resampled; 1000 items is the plan's count and sits between the ladder's 10² and 10⁴ points.

### bounds

Per executor, the wall seconds and resident bytes the coordinator and its workers may use. `in-process` and `pool-local`: Task 21 Step 2's laptop bound, 15 minutes (900 s) and 2 GB (2 × 1024³ B, 2147483648). `pool-slurm`: null, the allocation's: the coordinator's `sbatch --time` and `--mem` on the submit line bound it, and the workers' bounds are their resource classes' `time_s` and `mem_bytes`.

### pool_local_workers

2, Task 21 Step 2's "`Pool` on `local` (2 workers)": the workers on each cpu class of the local profile, which is its `max_workers`.

### watch_poll_s

1 s, the `orcd` profile's `pool.first_poll_s`, the finest cadence the cluster profile reads outcomes at, so the watch costs no more reads than the coordinator's. One `ps -A -o pid=,ppid=,rss=` took 18 ms on the laptop (mean of 20 calls), 1.8 % of one core at this cadence; not measured on a cluster node.

### interrupt_share

1/3, Task 21 Step 4's "a third of the items are committed". The interrupt target is that share of the items rounded up.

### test_slack.observations

wall_slack_s 0.04: the seconds a request may take beyond the sleep its test backend was asked for, over four requests of 0.05 to 0.2 s; one request is a thread hand-off and a file write, a few milliseconds, and 0.04 s is over ten times that. first_bound_s 1.0: with a poll of 3 s the first-poll wait must claim an item within a second, a third of the poll, over the 0.1 s the item is put after and the 0.01 s first poll; a worker without the short wait takes the whole 3 s. put_later_s 0.1 and watch_poll_s 0.005: the item is put a tenth of a second after the worker starts, so the worker is waiting, and the test looks for the worker's progress every 5 ms, below the first poll. first_poll_s 0.01, poll_s 0.05, long_poll_s 3: the worker's polls in the tests that need it to finish fast, and the long poll the first poll must beat. presence_refresh_s 100: longer than any test, so a worker reuses its first read of the run's probe queues. Not measured on CI runners.

### test_slack.held_lease

stale_after_s 60 and timeout_s 0.2: the fault-message test holds a lease that must not go stale while a second
acquirer waits 0.2 s and times out; a minute is longer than any test, and the timeout only needs to end the wait. Not
measured.

### test_slack.probe_queue

stale_after_s 0.4 and renew_fraction 0.25: the serve lease goes stale in 0.4 s and is beaten every 0.1 s, as the queue tests' lease; peer_lifetime_s 60: a stand-in peer process sleeps for a minute and is killed by its test, so it outlives any test of this file. Not measured.
