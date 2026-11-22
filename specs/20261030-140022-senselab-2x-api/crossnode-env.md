# Cross-node envs: the torch build is the worker's device's (R-316)

## What failed

ORCD pool step 3 (`benchmarks/exit/orcd/step-pool.sh`, job 30027801 at `8d020317`): the coordinator ran as a CPU job
on one partition and `SlurmSubstrate` started the `gpu:h100` worker as its own job on a GPU node (granted correctly).
The worker's load failed six times with `AttributeError: module 'torch._C' has no attribute '_cuda_setDevice'`, the
pool typed it an unclassified infrastructure fault each time (a gap), and the residency was held at the budget of 6.
Step 4 (`LOCAL=1`, coordinator and worker in one GPU allocation) passed.

Evidence, read on the cluster:

- Every job payload in the pool root names the env `speechbrain-cpu-9318693165b2ca1e-…` and the index `whl/cpu`.
- The env root (`$WORK_ROOT/envs-<sha>`, from `lib.sh`) holds both `speechbrain-cpu-…` (record `torch_index:
  …/whl/cpu`, `torch-2.11.0+cpu`, no `nvidia-*` wheels) and `speechbrain-cu128-…` (`torch-2.11.0+cu128`).
- The checkout's own `.venv` has CUDA torch; host backends were never the problem.

## Cause

`runner.prepare` runs on the coordinator and calls `open_env(backend, profile)`, which reached `ensure_env` →
`_stem` → `pick_torch_index(detect_host_cuda())` (`runtime/envs.py` at `8d020317`, `_stem` and `pick_torch_index`).
The probe ran `nvidia-smi`/`nvcc` on the coordinator's CPU node, found none, picked the CPU index and named the env
`speechbrain-cpu-…`. That env's interpreter path went into every `JobDoc` and `Loader`, and the GPU worker on another
node ran it. The env was a function of the host that opened it, not of the hardware that ran it.

## Design

Two candidates were weighed:

1. **The worker's device selects the build.** `ensure_env`, `current_env` and `env_built` take the device the env's
   workers run on; `Packaged.open_env`, `env_built` and `env_read` take it from the placement (`Placement.device`).
   `pick_torch_index(device, override, max_cuda)`: `cuda` gets the newest CUDA index at or below the lock's
   `torch-index:` ceiling, every other device the CPU index (whose macOS wheels carry MPS); the profile's
   `envs.torch_index` still wins.
2. **Build or verify inside the worker job.** The worker would open its own env on its node before load.

Option 1 is taken. The env name, the record and the `driver.torch_build` context field become a function of (lock,
device class, profile) alone, so the same plan opens the same env whichever node coordinates, which a content-keyed
store wants anyway. Option 2 moves env building into every worker start, duplicates it per node class, and leaves the
coordinator's `JobDoc` naming an interpreter it never resolved. The host probe is removed rather than kept as a
preference: keeping it for a coordinator that happens to have a GPU would make the env again depend on where the
coordinator runs. The cost is that a node whose driver is older than the ceiling's CUDA is no longer routed to an
older index automatically; CUDA 12 minor-version compatibility covers most such nodes, and the remainder fail typed
at load (below) with the remedy, setting `envs.torch_index`. This supersedes the phase-2 plan's "keeps its semantics"
for the probe routing.

`orcd_driver envs` builds each lock once per profile device (a server env for `cuda` only), since a lock now has one
build per device.

## The typed failure at load

Every worker checks its grant before it loads: `senselab_worker.device_check.require_device(init["device"])`, in
`serve` and in `HostWorker`. For `cuda` it raises `DeviceUnusable` (a `ResourceFault`) when the env's torch is a build
without CUDA, or a CUDA build that sees no GPU; `mps` likewise; `cpu` and torch-free envs are not checked and import no
torch. The message names the build, `CUDA_VISIBLE_DEVICES` and the remedy.

`warm.classify` types a load whose chain carries `DeviceUnusable` (raised in process, or a worker's fatal record of that
type) as a resource fault with no reason. pydra then moves the job up its ladder of same-device classes, or fails it
`Unplaceable` with the cause, and records no out-of-memory unfit load (that happens only for reason `oom`). Before, the
same failure was an unclassified infrastructure fault charged once per load until the residency was held.

A final kind at load does not exist in pydra at `b98cb8a0`: `_load_fault` treats every non-resource kind, `policy`
included, as an infrastructure fault of the residency (R-317). Resource is also the honest class: the granted device is
the resource the env cannot use.
