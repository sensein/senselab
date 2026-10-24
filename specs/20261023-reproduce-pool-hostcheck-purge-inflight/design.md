# Reproduce driver: the restricted purge spares in-flight results; work-pool workers run a host probe

Two gaps found while running a public sharded step through the work pool on a Slurm cluster whose results root is
on NFSv4.2.

## 1. The restricted purge removed two finished shard results

### The failure

`verify-inputs` of a restricted id runs `restricted.purge_closure`, which removes every public result whose
summary's `input_closure` holds the id. A result whose summary cannot say (none readable, or one written before
closures were recorded) is taken under its claim and read again there; before this change it was removed unless
that second read showed a closure without the id.

A public sharded step (48 shards) ran while a `verify-inputs --all` array verified restricted ids. Every shard ran
to `executed 0` (worker logs), yet two shard result dirs and their run links were gone when the pool settled, and
the step was `failed E_UPSTREAM_NOT_DONE`. One restricted verify ran from 21:50:58 to 22:16:04; one of the lost
shards ran from 21:49:02 to 22:15:45, i.e. it finished 19 s before the verify ended. That verify ended
`E_CLAIM_HELD` (another shard held its claim past the 600 s timeout), so it printed no purge count.

### The mechanism

On a local filesystem no interleaving of claim, summary write and release makes the purge remove a finished
result: the claim is an `flock` on the server (`local_lock=none`), the shard writes its summary before releasing
it, and the purge reads the summary again after acquiring it. What a local filesystem does not show is the NFS
client's lookup cache. The purging host looked up `summary.json` before the shard wrote it (no such file), waited
for the claim, and looked it up again within the client's cache time; the second lookup returned the cached
"no such file", so the purge removed a result that had just recorded a closure without the id.

Measured on the cluster (two nodes, NFSv4.2, default mount options, so `lookupcache=all`): node A looks up
`summary.json` in 16 directories (no such file); node B creates all 16 at time W; node A then looks each up once
at W + d. Plain lookup: not seen at d = 0.5, 1, 2 and 5 s, seen at 10 s. Lookup after listing the directory:
seen at every d from 0.5 s.

### The change

`purge_closure` returns `Purge(purged, skipped_inflight)`. Each candidate is read again under its claim after
listing its dir (`_fresh_closure_state`), which the measurement above shows makes a just-written summary visible. Then:

- a closure that holds the id, in the first read or the second, removes the result (a failed second read, such as
  a stale handle, does not save a result already seen to hold the id);
- a closure without the id keeps it;
- a summary with no `input_closure` (written before closures were recorded) is finished and cannot say what it
  read, so it is removed, as before;
- a result with no readable summary is removed unless a run may still own it; it is left alone and counted in
  `skipped_inflight` when the claim was contended (a live holder released it while the purge waited, more than
  0.5 s), when the summary is listed but unreadable, or while any work pool under the runs root has a running
  controller (`live_pools`: the pool's `controller/lock` is held, tested with a non-blocking shared `lockf`). A shard
  result can be between attempts there, keeping its resume store. A pool whose controller is gone (cancelled, or
  dead) is not live, so it never blocks the purge of dead dirs for good;
- a claim still held after `PURGE_CLAIM_TIMEOUT_S` (unchanged, 600 s) is counted the same way and no longer raises
  `E_CLAIM_HELD`, so a restricted verify ends `ok … purged=<n> skipped_inflight=<n>` while public pools run.

A run that is still in flight checks its own closure near its end (the run's closure recheck withdraws it when an id
of its closure became restricted). Known residual window: a run that passed that recheck before the id was flagged,
and then holds its claim for more than 600 s before writing its summary, ends with a summary whose closure holds the
id. Lint and submit then refuse that id's dependants (`E_INPUT_UNPURGED`) and the next verify of the id removes it.
The purge of finished results whose closure holds the id is unchanged, and a second call removes nothing.
`verify-inputs` prints `purged=<n> skipped_inflight=<n>`.

The liveness rule is global (any running controller under the runs root, not the result's own pool) because an
in-flight shard result carries no link to its pool until it finishes; keeping every summary-less dir while any
controller runs is the conservative choice, and those dirs are counted, not hidden.

Rejected: refusing `verify-inputs` of a restricted id while any public run is in flight. It is simpler but blocks
the scheduled re-verification for as long as public pools run (days), which is the situation the purge has to
handle.

## 2. Work-pool workers of a reproduce pool ran no host check

### The failure

`work_pool` workers support `--host-check`, but the reproduce pool job built the worker command without it, so a
node where the step's sandboxed tool could not start consumed items at seconds per attempt (125 items lost on one
node, 32 attempts on 13 more, in a CPU pool on a preemptable partition family).

### The change

A public sharded step may declare `host_probe` (an argv with the step's placeholders; an absolute program needs a
placeholder, like `argv`) and `host_probe_timeout_s` (default 120); a restricted or plain step may not
(`E_SPEC_HOST_PROBE`). Neither is keyed: they decide where a step runs, not what it computes. The pool job passes
each worker `--host-check` with `run … --step <s> --shard 0 --host-probe`, which keys and prepares the step as its
shard would (its needs resolved, its refusals applied, its stage files skipped) in a private result dir under
`<results root>/.host-probe/`, removed afterwards, so the sandbox layout (binds, the `/tmp` bind, masks) is the
step's own. It starts the sandbox once with `/usr/bin/true`, then runs the probe in it. Exit codes name the reason
(`run.HOST_PROBE_EXIT`): `exec-error` 20 (the probe exits 126/127), `exit-nonzero` 21, `timeout` 22 (the sandbox is
killed), `sandbox-unavailable` 23 (a host-dependent refusal: `E_SANDBOX_UNAVAILABLE`, `E_NO_USERNS`,
`E_GPU_MISSING`, `E_DRIVER_OLD`, or the sandbox check fails). A failure that is the same on every host (keying, an
upstream step not done, an env refusal, preparation, a manifest changed since submission) passes the probe, so the
shard itself fails with its code instead of the pool waiting for a host that can never pass.

The worker records a failure under the pool's `hostcheck/` with its partition and the reason
(`--host-check-reason CODE=NAME`, new in `work_pool`; other exit codes are recorded with `reason: null`) and exits
without claiming, so no item's attempt count changes; the controller's existing per-partition release count and
pause apply. The worker's host check may run 2 × `host_probe_timeout_s` + 600 s (the two sandbox runs plus keying,
which digests the step's inputs once per worker start).

Known limit: a probe that fails on every host for a reason of its own (for example a wrong command) leaves the pool
waiting until its job's time limit; probes are checked on one host before a pool uses them.

## Compatibility

Every step re-keys once, because the driver sources (`reproduce_source_sha256`) change. Manifests, locks and
summaries of steps without `host_probe` are otherwise unchanged; the `verify-inputs` line gains
`skipped_inflight=<n>`; the `hostcheck/` record gains `reason`.
