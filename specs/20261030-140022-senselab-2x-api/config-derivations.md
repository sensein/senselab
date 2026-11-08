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

One class per device the laptop may have, each the host's own CPUs and memory with no time limit and no partition: a laptop runs its workers as local processes. max_workers 1: one worker per class, since a second worker on one host competes for the same memory and device; the plan places a node on the first class of the first listed device the host has and the backend declares. memory_cap_bytes null: no table measures a model yet, and a laptop's free memory is not fixed, so no static cap is checked. No batch size: a placement's batch is sized (`size`) from the backend and model's resource table on the class, filling sizing.memory_headroom of the class's memory beside the resident model, and is 1 while unmeasured (one item per request is the batching that needs no measurement).

### laptop.substrate

local: a laptop starts its pool workers as processes on itself. start_backoff_first_s 0.05 and start_backoff_max_s 30: the wait after a class's first failed start (a worker that exits before it marks itself ready), doubling per failure in a row up to the maximum; the same wait follows each failure of the substrate's own commands, during which the scaler does not use the substrate. Not measured: the values the pool used before the keys existed, when it took this wait from pool.first_poll_s and pool.poll_s; the basis to measure is a worker's start-to-ready latency on the host. commands: none, a local substrate runs no scheduler command.

### laptop.pool

quantum_items 100 and quantum_s 300 are ceilings: `size` derives each unit's quantum from its measure (sizing.load_share) and holds it to at most a hundred jobs and to the jobs that fit in five minutes after the load (and in the class's time_s), and a pool worker stays on a class for at most that many jobs or seconds, so an older queue waits at most a ceiling quantum. Not measured: the fixed quantum the pool used before sizing; the ladder's per-item cost (Task 20 Step 4) shows whether a ceiling binds. poll_s 30: the 1.x reproduce controller's poll interval; a worker whose queues stay empty for three polls exits. first_poll_s (0.05 on laptop, 1 on orcd; Phase 3 Task 14 fix round): a coordinator's first wait for an outcome, doubled up to poll_s, so a local run does not wait 30 s per fragment; orcd starts at a second to keep stats off the NFS server. Not measured. segment_items 512 (Phase 3 Task 13 review, owner ruling): a claim lists its oldest live segment, so its cost grows with the segment, not with the queue. The Task 13 review measured claim plus commit on local SSD at 10^4 items queued: 22.4 ms with 10^4-item segments (about 1 MB of index read per claim) against 3.4 ms with 500-item segments. 512 keeps a segment's listing to about a thousand names (slots and their marks) while 10^4 items still cross twenty segments; the Step 5 run on ORCD revises it.

### laptop.sizing

load_share 0.05: a unit's quantum is the fewest jobs whose time keeps its model load and its claim within a twentieth of the quantum's, so loading and claiming cost a worker at most 5 % of its time (the calibrate-then-size design); with neither measured above zero the quantum is the ceiling. A target, chosen, not measured. memory_headroom 0.9: a batch fills at most nine tenths of the worker's memory beside the resident model: on a `cuda` class with a device measure, its memory_cap_bytes against the device pair; otherwise its mem_bytes, else on a local substrate the host's, against the host pair; the tenth left covers allocator fragmentation and a peak above the measured one. Not measured; the spread of peak memory over the 10⁴-item ladder point (Task 20 Step 4) replaces it. target_wall_s 1800: the wall time the workers of one queue aim to end its waiting work in, so a queue starts no more workers than end it in half an hour and none for work one worker ends sooner; a target, chosen so a 10⁴-item ladder point ends within one session; laptop's classes have max_workers 1, so it never binds there. calibration_items 4: a unit with no resource-table row first runs four one-job items on one worker; the first carries the model load and the median of the other three, robust to one outlier, is the seconds per item. The least count whose warm items have a median that tolerates one outlier. Batch against throughput (Phase 3 Task 20e, 20g): a batch sized by memory is kept only where items sent in a batch were measured no slower per item than items sent alone; otherwise, and where the batched time is unmeasured, the batch is one (`size` gives both `memory_batch` and the `batch_size` kept). A calibration round whose items include no warm one-job item measures nothing and writes nothing: its load cannot be told from its work. A calibration round measures both: its calibration_items one-job items first, for the load, the claim and the memory; then calibration_items − 1 pairs on the worker the first round warmed, each two items of the memory-sized batch's jobs, one sent as one request and one sent a job per request, so each side has a median of three warm figures (robust to one outlier on each side), each figure one request call over as many jobs and inputs; the batch is kept only where the batched median per-item seconds of that second round is not above the one-per-request median (Task 20h: one-job items sampled straight after the load read 2 to 5 times the steady cost at 9380b530; Task 20j: one-job items are gathered into one claim and share its wall, so they were one figure carrying a third of a call's fixed cost, R-234). A ladder's tables carry both, since the ladder's pool reads no packaged table and so calibrates every unit in the run. No slack term: the two medians are of one backend on one warm worker, the median absorbs one outlier per side, and a wrong call errs to batch 1, which is never over memory. Cost bound: once per unit per pool (a written row skips it), 2 × (calibration_items − 1) × batch jobs, all real work whose results are kept, and 2 × (calibration_items − 1) claims. Device memory per item is the backend's own request peak over what stays allocated after the request (`torch.cuda.max_memory_allocated` against `memory_allocated`, reported in `done`), divided by the request's items; a server backend, which reports neither, keeps the device read of the worker's processes after each claim (its fixed reservation); a model backend that reports neither (no CUDA initialised in torch) leaves its device memory unmeasured, so its batch is 1, and the pool logs it once per unit. calibration_workers 1: a queue whose units are all unmeasured is given one worker, so the warm items follow the cold one on the same worker and the load is read once; a queue mixing a measured unit with an unmeasured one is sized by the measured one and may run the round on several workers, each giving a cold sample, which `calibrated` reads as well. Workers of a `cuda` class whose substrate shares devices (local: the GPUs the coordinator sees, one per worker as `CUDA_VISIBLE_DEVICES`) are at most devices × floor(memory_cap_bytes / measured device peak), at least one per device and one while the device peak is unmeasured: the measured peak is what one worker holds, so the cap divided by it is how many fit a card. No new key; Slurm workers each hold their own card, so the bound is max_workers there. device_query_s 30: a pool worker reads its processes' device memory once per claim on a `cuda` class with `nvidia-smi --query-compute-apps`, bounded by the hardware probe's `nvidia-smi` bound; not measured, the basis is the command's p99 latency on a loaded GPU node.

### laptop.lease

Not measured until Task 16, which measures the store root's attribute delay that bounds both values. stale_after_s 300: five times the NFS client's default maximum attribute-cache time (acregmax and acdirmax, 60 s), so a beat directory a live holder made is seen well inside the bound on a shared root. renew_fraction 0.33: the 1.x work pool's rule that a heartbeat is at most a third of the lease, so two missed renewals still leave the lease live. The one place lease settings live: `pool.lease_renew_fraction` and `envs.lease_stale_after_s` were merged here (Task 12), and every lease holder (env build, vLLM start, run manifest, label map) beats under `Lease.keep_alive` every `renew_fraction × stale_after_s`. The env build lease used to be stale after 1200 s, about twice the slowest measured cold build of the crisperwhisper env on ORCD (362.6 s GPU node, 579.7 s and 590 s CPU node, n=2 per host; specs/20260907-shared-lock-heartbeat-inf/ stampede-timeout-and-identity-file.md); a beating build no longer needs a bound that covers it. A waiter on a build waits stale_after_s plus the lease timeout, 300 + 600 = 900 s, which is about 1.5 times the slowest measured build (590 s, on a loaded CPU node); the padding is less than the old 1200 s carried because the waiter no longer also waits out a stale bound sized to the build. `ExecutionProfile.load` refuses a profile where 2 × renew_fraction × stale_after_s + attribute_delay_s ≥ stale_after_s: the factor 2 allows one missed beat, and a client may see a beat up to attribute_delay_s late. Read by `PydraRunner` for pydra's job leases (P3-53); the pin at c8541fc5 has no `LeaseSettings` yet (Task 45), so the values are held and not yet passed. attribute_delay_s 0: the laptop profile's roots are on local disk, where a `stat` reads the inode itself and no client attribute cache stands between a beat and a breaker; to be replaced by Task 16's measured value.

### laptop.faults

content_attempts 2: the 1.x work pool's DEFAULT_MAX_ATTEMPTS, an item failing for its content twice goes to dead/. infrastructure_replacements 6: the 1.x work pool's DEFAULT_MAX_INTERRUPTS, releases and reaps allowed before an env is reported broken. resource_replacements 0: one per larger memory class the profile lists, and a laptop lists none, so a resource fault fails at once.

### laptop.timeouts

Null, no bound: no measurement of the slowest model load or the slowest single request exists. The scale ladder's resource tables give load_s and the per-input time from which a site sets both; a timeout is a resource fault, so a bound set too low re-places work rather than failing it.

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

Partitions are site placeholders (${CPU_PARTITION}, ${GPU_PARTITION}), expanded where a worker is submitted; the site's values stay local. cpu: 4 CPUs and 16 GiB for 6 h, the CPU grant of the 1.x diarization jobs (specs/20260915-preprocess-diarization/design.md). cpu:large: 8 CPUs and 64 GiB, the host grant of scripts/speaker_ceiling/evaluate.sbatch, as the larger memory class a resource fault moves to. gpu:h100: one H100 with 8 CPUs and 96 GiB of host memory, the grant of specs/20260820-vlm-spectrogram/ job.sh; H100 because review arrays run there first. time_s 21600 on every class: not measured, the diarization grant's limit. memory_cap_bytes 85899345920 on gpu:h100: the H100's 80 GiB of device memory, which the models resident at once on one worker share. CPU classes have no static cap: their memory is the grant, and an OOM there is a resource fault that moves the item to cpu:large. gpu: any GPU type, with the gpu:h100 host grant and an untyped gres, listed after gpu:h100 so a plan never chooses it first; it is the class a call running on a GPU node of another type takes. Its device memory depends on the type, so it has no static cap. svc:h100: the gpu:h100 grant and cap, as a service class; a server backend (vLLM) is placed on it before any other class, so a `svc` pool worker starts and registers the server and every other worker of that identity on the host consumes it. The ladder's pii_spans/vllm entry ran on gpu:h100 because no service class existed and placement took the first class of the device: four gpu:h100 workers of one card each started an unregistered server at gpu_memory_utilization 0.9, and every start after the first failed for free memory. max_workers 4 as gpu:h100; on a shared device sizing bounds it further by device memory (orcd.sizing). max_workers (32, 8, 4, 4): not measured; the substrate's QOS cap bounds them further. No batch size: a placement's batch is sized (`size`) from the backend and model's resource table on the class, filling sizing.memory_headroom of the class's memory beside the resident model, and is 1 while unmeasured (one item per request is the batching that needs no measurement).

### orcd.substrate

slurm: workers are Slurm jobs submitted per resource class. qos_cpu_cap null: the cap is the group's QOS MaxTRESPU CPU limit, which is account-specific and set in a site override; exclude_nodes empty, a site lists nodes it has found broken. start_backoff_first_s 1 and start_backoff_max_s 30: the wait after a class's first failed start, or after a failure of a Slurm command, doubling per failure in a row up to the maximum; the scaler does not use the substrate until it has passed, and past faults.infrastructure_replacements failures in a row the substrate is broken for the epoch. Not measured: the values the pool used before the keys existed, when it took this wait from pool.first_poll_s and pool.poll_s; the basis to measure is the site's submit-to-ready latency and how long a controller outage lasts. commands.timeout_s 60, commands.attempts 3, commands.retry_first_s 0.5: one Slurm command may take a minute, is tried three times, and waits half a second, then a second, between tries. Not measured: the constants the substrate carried before (SLURM_TIMEOUT_S, SLURM_TRIES, retry_s); the basis to measure is the p99 latency of sbatch and squeue on the site and the controller's failure rate. A submission is idempotent under these settings: each intended worker's job carries a name of its own, and after a try that fails or times out the name is looked up with squeue before the next try, so a try whose job was queued is never submitted again. The lookup reads `squeue --states=all`, which lists a job until slurmctld purges it MinJobAge after it ends (default 300 s), so the bound is: commands.timeout_s + commands.retry_first_s × 2^(commands.attempts − 1) must stay below the site's MinJobAge (`scontrol show config | grep MinJobAge`). With these values that is 60 + 0.5 × 4 = 62 s against 300 s. A job that started, ended and was purged inside that window never reached its run, so its duplicate is surplus that scale stops. A duplicate the controller registers after the lookup has the same name, is listed by `live` with the run's other jobs, and is cancelled when the fleet closes. One submission holds the scaler thread at most commands.attempts × (commands.timeout_s + one lookup), and a lookup is itself at most commands.attempts × commands.timeout_s plus its waits: (3 × 60 + 1.5) for the lookup, so about 3 × (60 + 181.5) = 725 s in the worst case; the first submission that fails ends that start call, so no further worker is submitted that round. While the substrate backs off, the scaler wakes when the wait ends rather than at the next poll_s.

### orcd.pool

quantum_items 100 and quantum_s 300 are ceilings: `size` derives each unit's quantum from its measure (sizing.load_share) and holds it to at most a hundred jobs and to the jobs that fit in five minutes after the load (and in the class's time_s), and a pool worker stays on a class for at most that many jobs or seconds, so an older queue waits at most a ceiling quantum. Not measured: the fixed quantum the pool used before sizing; the ladder's per-item cost (Task 20 Step 4) shows whether a ceiling binds. poll_s 30: the 1.x reproduce controller's poll interval; a worker whose queues stay empty for three polls exits. first_poll_s (0.05 on laptop, 1 on orcd; Phase 3 Task 14 fix round): a coordinator's first wait for an outcome, doubled up to poll_s, so a local run does not wait 30 s per fragment; orcd starts at a second to keep stats off the NFS server. Not measured. segment_items 512 (Phase 3 Task 13 review, owner ruling): a claim lists its oldest live segment, so its cost grows with the segment, not with the queue. The Task 13 review measured claim plus commit on local SSD at 10^4 items queued: 22.4 ms with 10^4-item segments (about 1 MB of index read per claim) against 3.4 ms with 500-item segments. 512 keeps a segment's listing to about a thousand names (slots and their marks) while 10^4 items still cross twenty segments; the Step 5 run on ORCD revises it.

### orcd.sizing

load_share 0.05: a unit's quantum is the fewest jobs whose time keeps its model load and its claim within a twentieth of the quantum's, so loading and claiming cost a worker at most 5 % of its time (the calibrate-then-size design); with neither measured above zero the quantum is the ceiling. A target, chosen, not measured. memory_headroom 0.9: a batch fills at most nine tenths of the worker's memory beside the resident model: on a `cuda` class with a device measure, its memory_cap_bytes against the device pair; otherwise its mem_bytes, else on a local substrate the host's, against the host pair; the tenth left covers allocator fragmentation and a peak above the measured one. Not measured; the spread of peak memory over the 10⁴-item ladder point (Task 20 Step 4) replaces it. target_wall_s 1800: the wall time the workers of one queue aim to end its waiting work in, so a queue starts no more workers than end it in half an hour and none for work one worker ends sooner; a target, chosen so a 10⁴-item ladder point ends within one session. calibration_items 4: a unit with no resource-table row first runs four one-job items on one worker; the first carries the model load and the median of the other three, robust to one outlier, is the seconds per item. The least count whose warm items have a median that tolerates one outlier. Batch against throughput (Phase 3 Task 20e, 20g): a batch sized by memory is kept only where items sent in a batch were measured no slower per item than items sent alone; otherwise, and where the batched time is unmeasured, the batch is one (`size` gives both `memory_batch` and the `batch_size` kept). A calibration round whose items include no warm one-job item measures nothing and writes nothing: its load cannot be told from its work. A calibration round measures both: its calibration_items one-job items first, for the load, the claim and the memory; then calibration_items − 1 pairs on the worker the first round warmed, each two items of the memory-sized batch's jobs, one sent as one request and one sent a job per request, so each side has a median of three warm figures (robust to one outlier on each side), each figure one request call over as many jobs and inputs; the batch is kept only where the batched median per-item seconds of that second round is not above the one-per-request median (Task 20h: one-job items sampled straight after the load read 2 to 5 times the steady cost at 9380b530; Task 20j: one-job items are gathered into one claim and share its wall, so they were one figure carrying a third of a call's fixed cost, R-234). A ladder's tables carry both, since the ladder's pool reads no packaged table and so calibrates every unit in the run. No slack term: the two medians are of one backend on one warm worker, the median absorbs one outlier per side, and a wrong call errs to batch 1, which is never over memory. Cost bound: once per unit per pool (a written row skips it), 2 × (calibration_items − 1) × batch jobs, all real work whose results are kept, and 2 × (calibration_items − 1) claims. Device memory per item is the backend's own request peak over what stays allocated after the request (`torch.cuda.max_memory_allocated` against `memory_allocated`, reported in `done`), divided by the request's items; a server backend, which reports neither, keeps the device read of the worker's processes after each claim (its fixed reservation); a model backend that reports neither (no CUDA initialised in torch) leaves its device memory unmeasured, so its batch is 1, and the pool logs it once per unit. calibration_workers 1: a queue whose units are all unmeasured is given one worker, so the warm items follow the cold one on the same worker and the load is read once; a queue mixing a measured unit with an unmeasured one is sized by the measured one and may run the round on several workers, each giving a cold sample, which `calibrated` reads as well. Workers of a `cuda` class whose substrate shares devices (local: the GPUs the coordinator sees, one per worker as `CUDA_VISIBLE_DEVICES`) are at most devices × floor(memory_cap_bytes / measured device peak), at least one per device and one while the device peak is unmeasured: the measured peak is what one worker holds, so the cap divided by it is how many fit a card. No new key; Slurm workers each hold their own card, so the bound is max_workers there. device_query_s 30: a pool worker reads its processes' device memory once per claim on a `cuda` class with `nvidia-smi --query-compute-apps`, bounded by the hardware probe's `nvidia-smi` bound; not measured, the basis is the command's p99 latency on a loaded GPU node.

### orcd.lease

Not measured until Task 16, which measures the store root's attribute delay that bounds both values. stale_after_s 300: five times the NFS client's default maximum attribute-cache time (acregmax and acdirmax, 60 s), so a beat directory a live holder made is seen well inside the bound on a shared root. renew_fraction 0.33: the 1.x work pool's rule that a heartbeat is at most a third of the lease, so two missed renewals still leave the lease live. The one place lease settings live: `pool.lease_renew_fraction` and `envs.lease_stale_after_s` were merged here (Task 12), and every lease holder (env build, vLLM start, run manifest, label map) beats under `Lease.keep_alive` every `renew_fraction × stale_after_s`. The env build lease used to be stale after 1200 s, about twice the slowest measured cold build of the crisperwhisper env on ORCD (362.6 s GPU node, 579.7 s and 590 s CPU node, n=2 per host; specs/20260907-shared-lock-heartbeat-inf/ stampede-timeout-and-identity-file.md); a beating build no longer needs a bound that covers it. A waiter on a build waits stale_after_s plus the lease timeout, 300 + 600 = 900 s, which is about 1.5 times the slowest measured build (590 s, on a loaded CPU node); the padding is less than the old 1200 s carried because the waiter no longer also waits out a stale bound sized to the build. `ExecutionProfile.load` refuses a profile where 2 × renew_fraction × stale_after_s + attribute_delay_s ≥ stale_after_s: the factor 2 allows one missed beat, and a client may see a beat up to attribute_delay_s late. Read by `PydraRunner` for pydra's job leases (P3-53); the pin at c8541fc5 has no `LeaseSettings` yet (Task 45), so the values are held and not yet passed. attribute_delay_s 60: the NFS client default acdirmax; to be replaced by Task 16's measured value. With the defaults, 2 × 0.33 × 300 + 60 = 258 s < 300 s.

### orcd.faults

content_attempts 2: the 1.x work pool's DEFAULT_MAX_ATTEMPTS, an item failing for its content twice goes to dead/. infrastructure_replacements 6: the 1.x work pool's DEFAULT_MAX_INTERRUPTS, releases and reaps allowed before an env is reported broken. resource_replacements 1: one per larger memory class the profile lists (cpu:large), so a resource fault walks the classes once and then fails.

### orcd.timeouts

Null, no bound: no measurement of the slowest model load or the slowest single request exists. The scale ladder's resource tables give load_s and the per-input time from which a site sets both; a timeout is a resource fault, so a bound set too low re-places work rather than failing it.

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

One hour. A collection touches the clock sentinel and lists quarantine/ and tmp/, which costs metadata round trips on NFS; running it on every put multiplied them by the put rate. What it removes is at least a day old (staging_stale_s) or a week old (quarantine_retention_s), so collecting hourly delays a removal by at most an hour against ages of a day or more.

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

The scheduler hands a job its devices as `CUDA_VISIBLE_DEVICES`; a worker that drops it sees every device or none. These select devices and CUDA behaviour (`CUDA_LAUNCH_BLOCKING`, `CUBLAS` workspace via `CUDA_`), which can change numerics, so they are recorded.

### threads

Slurm jobs set thread counts to their CPU allocation; without them BLAS oversubscribes the node. Thread counts can change the order of floating-point reductions, so they are recorded.

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

Per suite entry label, the per-item ceilings: wall seconds, peak memory bytes, files and bytes written per item. Each is the entry's value at its 10² point at 11e55ebd (the commit the packaged tables were generated at; jobs 28266570, 28266571, 28266572) times 1.25 (flat_ratio's spread), rounded up to three significant figures. The 10⁴ point is held by flat_ratio, and the ladder judges a ceiling at the 10² and 10⁴ points, never at the one-item point that carries a worker's start. Two per-item times keep the e801bc6c ceiling (value at 10² times 1.25, job 28262080), because their 11e55ebd value rose past flat_ratio under the table's batch and that is R-226, not a looser budget: speech_to_text~crisperwhisper 1.71 s (1.365 s at e801bc6c; 2.141 s at 11e55ebd) and speaker_embeddings~speechbrain 0.13 s (0.104; 0.150). Values at 11e55ebd, 10² (per item s; peak bytes; files; bytes per item) then ceiling: The ladder's per-item figure counts only the items after calibration, from their end times (Task 20h); held parts' queue wait before their first claim is inside it, the run's start and the wait to see it end are not. CPU ceilings are to be taken from that figure at the 10⁴ point; the GPU suites run 1 and 10² only, where calibration takes most or all of the point, so their ceilings are to be taken from the chosen regime's median per-item time in the calibration's second round, times flat_ratio. They stay at e801bc6c's until the ladder reruns at the 20h merge: the 9380b530 figures carry the effects R-231 records.

- signal.resample~numpy-sinc (28266570): 0.0696 s, 178 352 128 B, 3.13, 86 837 -> 0.0871, 223 000 000, 3.92, 109 000.
- fake.echo~fake (28266571): 0.02673 s, 95 195 136 B, 3.2, 4 228 -> 0.0335, 119 000 000, 4.0, 5 290.
- H100, 28266572: speech_to_text~transformers 0.3171 s, 2 633 408 512 B, 2.22, 5 533 -> 0.397, 3 300 000 000, 2.78, 6 920; speech_to_text~crisperwhisper 2.141 s, 357 851 136 B, 2.29, 5 855 -> 1.71 (see above), 448 000 000, 2.87, 7 320; speaker_embeddings~speechbrain 0.1498 s, 361 926 656 B, 3.22, 6 361 -> 0.13 (see above), 453 000 000, 4.03, 7 960; diarize~pyannote 0.1514 s, 365 801 472 B, 5.22, 6 177 -> 0.19, 458 000 000, 6.53, 7 730; pii_spans~vllm 4.546 s, 366 747 648 B, 6.21, 12 388 -> 5.69, 459 000 000, 7.77, 15 500; pii_spans~transformers 11.764 s, 366 747 648 B, 6.36, 11 534 -> 14.8, 459 000 000, 7.95, 14 500.

At 10² the GPU entries wrote about a third of the files of the earlier runs (speech_to_text~transformers 222 against 817; crisperwhisper 229 against 817; speechbrain 322 against 917; diarize 522 against 1 117; pii_spans 621 to 636 against about 1 210): with packaged tables the pool sizes each item's batch and quantum from them, so fewer, larger requests each write their provenance. Peak memory is the pool worker's own high-water (`ru_maxrss`); the tables' host pair is unmeasured on these nodes (no readable cgroup v2).

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
