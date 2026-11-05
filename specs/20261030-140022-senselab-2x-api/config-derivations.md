# Execution-profile derivations

Why each key of the packaged execution profiles (`packages/senselab-core/src/senselab_core/runtime/data/profiles/`) has the value it has, by profile and top-level key. The YAML files carry a one-line `#` comment per key; the measurement and the reasoning are here.

## laptop

### laptop.stack_preferences

Empty: no task has two stacks yet, so the backend order a task declares decides. A profile lists a task's stacks best first once a second stack exists.

### laptop.devices

Fastest accelerator first, CPU last. A backend takes the first listed device that the host has and the backend declares.

### laptop.env_roots

The per-user cache directories 1.x built its subprocess venvs under, one root per env kind.

### laptop.envs

lease_stale_after_s 1200: about twice the slowest measured cold build of the crisperwhisper env on ORCD (362.6 s GPU node, 579.7 s and 590 s CPU node, n=2 per host; specs/20260907-shared-lock-heartbeat-inf/ stampede-timeout-and-identity-file.md), padded because two samples bound no population. A lease younger than that is a build in progress and is waited on. No torch index override: the CUDA index is probed from the host. A site whose probe is wrong sets torch_index. server_start_margin_s 60: a server start lease is taken before the server is launched and its health deadline begins, so its stale bound is the start-up timeout plus this margin. Not measured: the launch (two process spawns) is sub-second, and the margin is padded for the storage clock's mtime granularity and for a loaded host.

### laptop.model_sources

Null: the hub organisations of the catalogue entries. A site that mirrors or restricts models lists organisations here.

### laptop.model_cache_budget_bytes

Null: no cap. No measurement of a model-cache size exists to derive a cap from.

### laptop.retry

Three tries and a one second first wait, doubling: the 1.x schedule (1 s, 2 s, 4 s) of SENSELAB_HF_MAX_RETRIES. Not measured: no failure-rate data for the Hub or a mirror exists in this repository. A fault that has not cleared after three seconds of waiting surfaces to the caller rather than stalling the run.

### laptop.network

Online: a laptop reaches the Hub and the data mirrors directly.

### laptop.resource_classes

One class per device the laptop may have, each the host's own CPUs and memory with no time limit and no partition: a laptop runs its workers as local processes. max_workers 1: one worker per class, since a second worker on one host competes for the same memory and device; the plan places a node on the first class of the first listed device the host has and the backend declares. memory_cap_bytes null: no table measures a model yet, and a laptop's free memory is not fixed, so no static cap is checked.

### laptop.substrate

local: a laptop starts its pool workers as processes on itself.

### laptop.pool

Not measured; each value is the starting point the scale ladder revises. quantum_items 100 and quantum_s 300: a worker yields to an older queue after a hundred items or five minutes, whichever comes first, so a model load is paid at most once per quantum of work. poll_s 30: the 1.x reproduce controller's poll interval; a worker whose queues stay empty for three polls exits. lease_renew_fraction 0.33: the 1.x work pool's rule that a heartbeat is at most a third of the lease, so two missed renewals still leave the lease live. segment_items 10000: an index segment rolls at the largest scale point the ladder runs (10^4 items), so a plan of that size reads one segment.

### laptop.lease

Not measured until Task 16, which measures the store root's attribute delay that bounds both values. stale_after_s 300: five times the NFS client's default maximum attribute-cache time (acregmax and acdirmax, 60 s), so a beat directory a live holder made is seen well inside the bound on a shared root. renew_fraction 0.33: the 1.x work pool's rule that a heartbeat is at most a third of the lease, so two missed renewals still leave the lease live. Read by `PydraRunner` for pydra's job leases (P3-53); the pin at c8541fc5 has no `LeaseSettings` yet (Task 45), so the values are held and not yet passed.

### laptop.faults

content_attempts 2: the 1.x work pool's DEFAULT_MAX_ATTEMPTS, an item failing for its content twice goes to dead/. infrastructure_replacements 6: the 1.x work pool's DEFAULT_MAX_INTERRUPTS, releases and reaps allowed before an env is reported broken. resource_replacements 0: one per larger memory class the profile lists, and a laptop lists none, so a resource fault fails at once.

### laptop.timeouts

Null, no bound: no measurement of the slowest model load or the slowest single request exists. The scale ladder's resource tables give load_s and the per-input time from which a site sets both; a timeout is a resource fault, so a bound set too low re-places work rather than failing it.

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

lease_stale_after_s 1200: about twice the slowest measured cold build of the crisperwhisper env on ORCD (362.6 s GPU node, 579.7 s and 590 s CPU node, n=2 per host; specs/20260907-shared-lock-heartbeat-inf/ stampede-timeout-and-identity-file.md), padded because two samples bound no population. A lease younger than that is a build in progress and is waited on. No torch index override: the CUDA index is probed from the host. A site whose probe is wrong sets torch_index. server_start_margin_s 60: a server start lease is taken before the server is launched and its health deadline begins, so its stale bound is the start-up timeout plus this margin. Not measured: the launch (two process spawns) is sub-second, and the margin is padded for the storage clock's mtime granularity and for a loaded host.

### orcd.model_sources

Null: the hub organisations of the catalogue entries. A site that mirrors or restricts models lists organisations here.

### orcd.model_cache_budget_bytes

Null: no cap. No measurement of a model-cache size exists to derive a cap from.

### orcd.retry

Three tries and a one second first wait, doubling: the 1.x schedule (1 s, 2 s, 4 s) of SENSELAB_HF_MAX_RETRIES. Not measured: no failure-rate data for the Hub or a mirror exists in this repository. A fault that has not cleared after three seconds of waiting surfaces to the caller rather than stalling the run.

### orcd.network

Online: compute nodes reach the Hub and the data mirrors.

### orcd.resource_classes

Partitions are site placeholders (${CPU_PARTITION}, ${GPU_PARTITION}), expanded where a worker is submitted; the site's values stay local. cpu: 4 CPUs and 16 GiB for 6 h, the CPU grant of the 1.x diarization jobs (specs/20260915-preprocess-diarization/design.md). cpu:large: 8 CPUs and 64 GiB, the host grant of scripts/speaker_ceiling/evaluate.sbatch, as the larger memory class a resource fault moves to. gpu:h100: one H100 with 8 CPUs and 96 GiB of host memory, the grant of specs/20260820-vlm-spectrogram/ job.sh; H100 because review arrays run there first. time_s 21600 on every class: not measured, the diarization grant's limit. memory_cap_bytes 85899345920 on gpu:h100: the H100's 80 GiB of device memory, which the models resident at once on one worker share. CPU classes have no static cap: their memory is the grant, and an OOM there is a resource fault that moves the item to cpu:large. gpu: any GPU type, with the gpu:h100 host grant and an untyped gres, listed after gpu:h100 so a plan never chooses it first; it is the class a call running on a GPU node of another type takes. Its device memory depends on the type, so it has no static cap. max_workers (32, 8, 4, 4): not measured; the substrate's QOS cap bounds them further.

### orcd.substrate

slurm: workers are Slurm jobs submitted per resource class. qos_cpu_cap null: the cap is the group's QOS MaxTRESPU CPU limit, which is account-specific and set in a site override; exclude_nodes empty, a site lists nodes it has found broken.

### orcd.pool

Not measured; each value is the starting point the scale ladder revises. quantum_items 100 and quantum_s 300: a worker yields to an older queue after a hundred items or five minutes, whichever comes first, so a model load is paid at most once per quantum of work. poll_s 30: the 1.x reproduce controller's poll interval; a worker whose queues stay empty for three polls exits. lease_renew_fraction 0.33: the 1.x work pool's rule that a heartbeat is at most a third of the lease, so two missed renewals still leave the lease live. segment_items 10000: an index segment rolls at the largest scale point the ladder runs (10^4 items), so a plan of that size reads one segment.

### orcd.lease

Not measured until Task 16, which measures the store root's attribute delay that bounds both values. stale_after_s 300: five times the NFS client's default maximum attribute-cache time (acregmax and acdirmax, 60 s), so a beat directory a live holder made is seen well inside the bound on a shared root. renew_fraction 0.33: the 1.x work pool's rule that a heartbeat is at most a third of the lease, so two missed renewals still leave the lease live. Read by `PydraRunner` for pydra's job leases (P3-53); the pin at c8541fc5 has no `LeaseSettings` yet (Task 45), so the values are held and not yet passed.

### orcd.faults

content_attempts 2: the 1.x work pool's DEFAULT_MAX_ATTEMPTS, an item failing for its content twice goes to dead/. infrastructure_replacements 6: the 1.x work pool's DEFAULT_MAX_INTERRUPTS, releases and reaps allowed before an env is reported broken. resource_replacements 1: one per larger memory class the profile lists (cpu:large), so a resource fault walks the classes once and then fails.

### orcd.timeouts

Null, no bound: no measurement of the slowest model load or the slowest single request exists. The scale ladder's resource tables give load_s and the per-input time from which a site sets both; a timeout is a resource fault, so a bound set too low re-places work rather than failing it.

### orcd.services

idle_exit_s 600: ten minutes, twice the pool quantum, so a service outlives a consumer's yield to another queue and the next quantum's requests find it up. Not measured: no measurement of how long consumers leave a service idle exists.
