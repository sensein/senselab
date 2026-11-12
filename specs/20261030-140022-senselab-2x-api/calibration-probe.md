# Task 20R: execution calibration around a dedicated Probe

Status: revised 2026-11-08 after the design review (`t20r-design-review.md`: B1-B3, I1-I12, N1-N5), the owner's
rulings on O-1..O-6 (§13) and the final review (`t20r-design-final-review.md`: F1-F11, N1-N7). Replaces the calibration rounds of Phase 3 Tasks 20e to 20k (`Pool._calibrate`,
`_batch_round`, `_round_measure`, `sizing.calibrated`). Keeps `size`, `Grant` and `Sizing` as the one sizing site,
reshaped as §6 says. Adds a site placement hook and a machine-readable statement of what a run needs (§10, owner 2026-11-08).

Baseline: the Task 20 architecture review at `438cb624` (`.superpowers/sdd/phase-3/t20-architecture-review.md` in the
implementation worktree), its measurement contract M1..M10, its sizing contract S1..S7, its §3 restructuring and its
§4 eight must-fix items. Code paths below are at `origin/2x/phase-3` `5ae1fc3d`. Scenarios: `user-scenarios.md` on
`2x/phase-3-spec`, checked in §14.4. Amendments: ADR-002, ADR-003 and ADR-004 of
2026-11-08 (§12). Principles: P-17, P-28, P-33, P-34 (as amended 2026-11-07), P-37, P-40 (as amended 2026-11-08).

## 1. What the owner asked

1. "probe should be designed around caching while optimizing so that cached results can be reused without running."
   A probe's items are real items of the run. Their outputs are ordinary keyed results in the content-addressed store
   and the run uses them: never computed twice, never discarded. A job the store would serve is not run to probe; the
   probe picks among jobs that will be computed, or uses timings already measured where they exist.
2. "probe results on an hpc should be reusable themselves." A probe's measurement is kept under the full identity
   of what it measured, in a measures store on the cluster's shared storage, so a later run, or another user on the
   same hardware class, reuses it without probing.
3. "do job submissions run a quick availability check before submitting a job to identify best options", refined:
   "they shouldn't be specific in senselab ... whatever provides the most useful info in a setting" and "senselab
   could hook into a site-specific callable for matching" (§10).

Requirements 1 and 2 follow from one rule: **a measurement is an artefact with an identity, like a result.** Results
are keyed by what they compute (ADR-002); measures are keyed by what determines their cost (§3). Neither key holds the
other.

## 2. Shape

```
                 coordinator (Pool)                                  pool worker (pinned for a probe)
 fragment ──► servable? ──yes──► put at once (serving needs no sizing)
                 │ no
                 ▼
     hardware class resolved (index, else first worker hello)
                 ▼
          MeasureUnit ──► MeasureSource.get ──► Estimate? ──yes──► size(estimate, grant, lengths) ──► items
                               │ no                                              ▲
                               ▼                                                 │
                 probe lease (first writable root) ──► Probe.plan ──► probe items ──► Observation per request
                               │ busy                         ▲                      │
                               ▼                              └── Probe.estimate ◄───┘ (+ stored observations)
                 wait for the peer (bounded), then New with its observations as prior
                                                                       │
                                                                       ▼
                                  measures roots (shared, then own): observations, drift, regime, OOM records
```

Six parts, each the only place its decision is made (P-37):

| part | module | decides |
|---|---|---|
| `ResultStore.admits` | `runtime/store.py` | whether a job would be served; `get` and the probe both call it |
| `MeasureUnit`, `HardwareClass` and the measures store | `runtime/measures.py` (new) | what a measure is of, where it lives, who may read it |
| `Observation` | `runtime/calibration.py` (new), filled by `pool_worker.py` | what one request took, whether the process was warm, its co-tenancy |
| `Probe.plan`, `Probe.estimate` | `runtime/calibration.py` | which jobs probe, in which phases; the one classifier |
| `size` | `runtime/sizing.py` | batch per length, item cuts, workers, workers per device |
| `ProbeState` in the Pool | `runtime/execution/pool.py` | probe vs production items, release, drift |

Removed outright (pre-alpha, no shims), each by the task that removes its last caller (§14): `Sample.cold`,
`calibrated`, `_batched`, `measure_of`, `Measures`, `code_version`, `resources.Unit`, `Pool._calibrating`,
`_first_round`, `_round_measure`, `_batch_round`, `_round_digests`, `_note_unmeasured_device`, `_grow`,
`PoolWorker._warm`, `profiles._batch_size`, `sizing.calibration_items`, `sizing.calibration_workers`,
`resource-table@1.0` with its seven packaged rows, and the ladder's `Measures(...).get(unit) or calibrated(samples)`
fallback (`ladder.py:333`).

## 3. Measure identity and the measures store

### 3.1 `MeasureUnit`: what a measure is of

`MeasureUnit` replaces `resources.Unit`. Its digest is `identity_digest("measure-unit@1", fields)` over:

| field | source | why it changes cost |
|---|---|---|
| `task` | the node's task and its version | different work; a task version change is a cost change (UX-61) |
| `backend`, `backend_version` | `BackendSpec` | different code path; the version also enters the logical key |
| `backend_params` | canonical JSON of the placement's `backend_params` (all of them) | engine, dtype, beam width, `max_new_tokens` (R-229: crisperwhisper `engine`) |
| `params` | canonical JSON of the task call's `params` | a param that shapes cost (word timestamps, `num_speakers`, returned embeddings) |
| `profile_slices` | the logical key's `profile_slices` | a slice is part of what the call computes; the key already holds it |
| `chunk_grid` | the runtime `Placement`'s grid, whether or not the backend declares it output-affecting | chunk length sets padding and per-item cost |
| `formation` | the runtime `Placement`'s `batch_formation`: `mixed`, `single`, or `bucket:<grid id>` (ADR-004, 2026-11-07); a placement whose formation is not yet decided has no unit | padding cost; whether a batch can be formed |
| `model` | `(id, identity)`, both None without a model | different weights |
| `env` | the env's fingerprinted context fields: `env_lock`, `wheel_set`, `driver` (only `torch_build`, such as `cu128` or `cpu`, never the index URL, and for a server `engine` and `engine_version`; any other key is refused), `platform`, `decoder` (the job's: a probe over file-backed and in-memory inputs measures two units), `numerics`; and `threads`, the worker launch's values of the variables `launch_environment.yaml` names `measured` (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`), each a positive count and anything else refused; no other variable of the family (a tool path, `OMP_PLACES`) enters; `{}` for a host env | library versions and builds: one lock installs torch from a host-chosen index (`envs.py:449-450`), so cu121 and cu128 builds differ in `wheel_set` and `driver` (R-229: an env rebuild doubled crisperwhisper's per-item time); thread counts set per-item time |
| `service` | the `Attestation`'s `config_digest` for a server backend, else None | server settings (`max_num_seqs`, `gpu_memory_utilization`) |
| `hardware` | the canonical `HardwareClass` (§3.2) | the device and the cores |
| `observation_format` | `measure-observations@1` | an observation of another format cannot be pooled |

A unit is built from the runtime `Placement` and the call's `LogicalKey`: it holds every field of the key but `inputs`
(a measure is per input unit), plus env, hardware and formation.

Every field is a declared value that shapes cost, or a digest whose every input is one; the identity takes no digest
it did not see the inputs of. So the launch digest (`launch_env`) is not in it: it covers device selection
(`CUDA_VISIBLE_DEVICES`, `NVIDIA_*`), which changes no cost, and `hardware` names the device. Two workers of one class
on different GPU slots share a unit. Of the `Attestation`, only `config_digest`, the digest of the asserted server
settings, is held: `engine_version` is in `driver`, `model_identity` in `model`, and `served_model` (for a `local:`
model, its path) is provenance, so one model served from two directories is one unit. ADR-002's result identity is
unchanged: a result's key still holds the whole `Attestation`.

The `env` fields are the ones `ExecutionContext.fingerprint()` takes that the env decides; all are read from the
env's record before a worker starts (`env.context_fields()`). This is ruling O-1 made precise (§13).

Not in the identity:

- **the input lengths.** A measure is per input unit (§5.2) and records the length range it covers. A run whose
  lengths lie outside it probes only the uncovered stratum (§4.2).
- **the senselab source digest.** `code_version()` over all of senselab-core invalidated every cached row on a
  docstring edit and never invalidated a packaged row (review §2, M10). The backend version and the env fields name the
  code that runs. A cost change with neither changed (a performance fix) is caught by drift (§7.3).
- **the concurrency regime.** It is recorded per observation (§5.3); drift compares only like regimes (§7.3).
- **the run, the user, the profile name, the resource class name.** These are provenance (§3.5) or profile-local.

### 3.2 One canonical hardware class

`HardwareClass` is the normalised report of a worker, and it is the only form used to look up and to write:

- `cuda`: the driver's product name as `nvidia-smi --query-gpu=name` gives it, `memory.total` in MiB, compute
  capability;
- `mps`: the chip string (`sysctl machdep.cpu.brand_string`) and unified memory;
- `cpu`: the CPU model string, read from `/proc/cpuinfo` `model name`, else `lscpu` under `LC_ALL=C`, else `sysctl` on
  Darwin; a host that names none has no class, never the architecture;
- every class: the granted `cpus` (the resource class's, else the CPUs the worker's process may run on: its affinity,
  `os.sched_getaffinity(0)`, where the platform has one, else `os.cpu_count()`), since decode and resample run on them.

The profile's `gpu_type` (which may be None, "any GPU") and `memory_cap_bytes` are never compared with a report and
never enter a digest. The coordinator resolves a resource class to the hardware it runs on before any lookup:

1. Each measures root keeps an index `classes/<declared digest>/<seen digest>.json`, where the declared digest
   (`hardware-declared@1`) covers the resource class's hardware fields (`device`, `gpu_type`, `memory_cap_bytes`,
   `cpus`, `partition`) and the seen digest (`hardware-class@1`) is the digest of a `HardwareClass`. The file holds that
   class (`measure-class@1.0`), since a digest alone cannot rebuild the unit's hardware field; a reader requires the
   name to be the class's digest. The coordinator writes the entry (write-once) on a worker's first hello, from the
   `HardwareClass` in its ready mark (`MeasureRoots.note_seen`; `seen_classes` reads it).
2. If the index names exactly one seen class, the coordinator uses it.
3. If the index is empty, or names more than one, the first worker's hello for that class decides. The unit is in
   `Resolving` (§7.1) until then; the queue's first worker is started as it would be anyway, so this costs no extra
   start. Probe and production items wait in that state; servable jobs do not (I4).
4. A class is resolved once per run, by the first hello (step 3), or by step 2 until then. When it maps to several seen classes (a partition of mixed
   cards), the plan says the class is mixed and that its items are sized for the resolved class. Each observation is
   written under the class of the worker that made it; only a worker of the resolved class takes a probe queue's
   serve lease (§4.4), and drift counts only requests of the resolved class (§7.3).
5. (Amended 2026-11-08, T20R-7a review I-3.) Step 2's class is a prediction. When the run's first hello for the class
   names another class, the hello wins: every unit of the class resolved from the prediction is resolved again under
   the hello's class, in whatever state it is (§7.1, "A disagreeing hello"). A later hello of another class is step 4's
   mixed class and re-resolves nothing.

On a local substrate the first worker's hello is this host's, so plan time resolves the class by the one rule: the class is
this host's `HardwareClass` (from its report, as the worker would give it), and it must be in the index; else the plan says
`hardware not yet seen on this host`, or `this host reports no hardware class` where the host names none. Several seen
classes in a shared root (a partition of mixed cards, or two workstations under one declared class) therefore change nothing
here: only this host's class is read. On a Slurm substrate the worker's node, platform and numerics are not known before it
starts, so the unit is unresolved until a hello (T20R-7a) says them.

**Pending and final causes (amended 2026-11-08, T20R-7a review B-1).** A unit that cannot be built has one of two
causes, and the cause, not the substrate, decides what follows:

- **Pending**: a worker's start removes it. The node a Slurm worker runs on is not yet chosen, so its `HardwareClass`,
  `platform`, `numerics` and the env fields it builds (`wheel_set`, `driver`) are unknown; the hardware is not yet
  seen on this host; a backend's env is not yet built. The unit waits in `Resolving` (§7.1) and is built at the first
  hello of its queue whose ready mark (§8) carries every field it lacks: the queue's `HardwareClass`, and per env of
  the queue's backends that env's fields of §3.1 (`env_lock`, `wheel_set`, `driver`, `platform`, `numerics`, and
  `threads`: `{}` for a host env, the served variables for a served one). A server unit's `driver` and `service` are
  completed by the coordinator from `service_of`, as on a local substrate. The unit then goes to `Lookup`, the lease
  and the probe like any other. The hello's values are the worker's own and win over any value the coordinator
  predicted.
- **Final**: nothing at run time removes it. The host or the hello names no hardware class; the jobs read both files
  and in-memory inputs (two decoders); a server's attestation is unreadable; an env record is unreadable; the hello
  says an env's build failed. Only a final cause takes the safe values (§6.5): the unit is `Unmeasured` with that
  reason, which the plan and the run report state.

A hello that resolves a pending unit to a final cause (no class, a failed build) makes it final. A first hello with
no class makes the class final for the run even where a later node would name one, since a class is resolved once per
run (step 4); this errs to the safe values. After a failed build the held parts are put as production, fail, and
raise the env hold (`QueueHeld`): the hold, not `Unmeasured`, is what the user sees. On a local substrate
the plan still resolves from this host's report and env records; only what they cannot yet say is pending.

So a unit is never looked up under a digest that no worker writes (review B3). Plan time (`explain`, §6.6) uses the
index only; with no entry, its figures are "unmeasured: hardware not yet seen".

### 3.3 Where measures live: layered roots

`calibration.measures_roots` is a list of roots, read in order and pooled (a run reads the files of every root it can
read, under the trust filter of §3.5):

- in a group's site profile (a user file, P-42): the group's shared root beside the result store, then the user's own
  `<cache root>/2x/measures`, written `${cache_root}/2x/measures` in the profile (the token is `revision.cache_root()`); a resolved root must be absolute,
  an unset or empty `SENSELAB_MEASURES_ROOT` is no root (logged, never raised), and a blank, repeated or relative root
  is refused at load;
- in `laptop` and the `slurm` template: the own root only.

Writes and leases go to the **first writable root**. Writability is checked once at run start by the storage probe
(ADR-003; `mkdir` of a probe directory), before any work, and the plan states where measures will be written
(UX-10: missing write access is never discovered after computing). Any `OSError` on a measures root, on a read or a
later write (permission, `ENOSPC`, `ESTALE`, an unreachable mount), is logged once per run per root and treated as that
root unreadable or unwritable: reads go on over the other roots, writes move to the next writable root, else the
measure is not persisted (the run goes on with it in memory). It is never raised (UX-37, UX-56). A read-only group member
therefore reuses the group's measures and writes their own probes to their own root. The ladder sets one root, in
its output directory (§9), so a table point measures afresh.

Layout of each root (`measures@1`):

```
<root>/measures@1/
  classes/<declared digest>/<seen digest>.json  hardware index (§3.2); measure-class@1.0, written once
  units/<u[:2]>/<u>/                          u = MeasureUnit digest
    unit.json                                 measure-unit@1.0: the identity fields; first writer wins
    obs/<probe id>.json                       measure-observations@1.1: one probe's observations; written once
    drift/<record id>.json                    measure-drift@1.0: production evidence of a suspect estimate
    supersede/<record id>.json                measure-supersede@1.0: a confirming probe disagreed; earlier obs stop pooling
    regime/<record id>.json                   measure-regime@1.0: a co-tenancy factor (§7.3)
    oom/<record id>.json                      measure-oom@1.1: a resource fault (§7.4)
    unfit/<record id>.json                    measure-unfit@1.0: a load that ran out of memory in a grant (§7.5)
    probe.lease/                              the ADR-003 lease held while a coordinator probes u
  tmp/<uuid>/                                 staging
  quarantine/<uuid>                           misfiled or malformed files, moved aside
```

There is no estimate file. `Probe.estimate` runs over the unit's pooled observations, newer than the newest trusted
`supersede` record names in `outdates`, when the coordinator first needs the unit, and the result is held for the run.

### 3.4 Concurrency (ADR-003)

- **Writes are write-once.** Every file is staged in `tmp/<uuid>/` with the store policy's modes and renamed onto a
  name no other writer uses (uuids). Nothing is modified in place, so concurrent writers never conflict and no lock
  guards an append. `unit.json` is first writer wins; a reader recomputes the digest from it and requires it to equal
  the directory name, else the directory is misfiled (the store's binding rule, ADR-003 amendment 2026-10-31) and
  handled by §3.5's malformed-file rule; a skipped unit directory takes no writes, which go to the next writable root.
- **One probe per unit at a time, for efficiency only.** A coordinator that needs to probe `u` reads every root for a
  live `<u>/probe.lease` and, finding none, takes the lease in its write root (the 2026-11-03 directory lease, renewed
  by `keep_alive`, proven before it writes its observations). "A peer" is defined once, by the trust filter: a lease
  whose owner uid the filter admits. `MeasureRoots.probe_lease` returns `Lease | PeerProbing | Unleased`. A coordinator
  finding a live lease of an admitted owner (`PeerProbing`) waits for the
  holder's observations up to `peer_wait_s`, re-reading once per `pool.poll_s`, then returns to `New` with
  whatever it found as prior (§7.1). The wait holds only the unit's computed jobs; its servable jobs and other units
  proceed, and the plan states it (§6.6). Under `calibration.trust: own`, a live lease of another uid is no peer: the
  coordinator probes without a lease and does not wait (`Unleased`, as when no root is writable); a stale lease its
  uid cannot break raises `LeaseUnavailable` on taking it. Liveness is judged on the storage clock, and on the local
  clock in a root that cannot be written. A broken lease, or two members writing to different roots, costs a duplicate
  probe, never a wrong measure.
- **Walks are by descriptor** below each root; ids that become path components are one component.

### 3.5 Trust and provenance, for every record kind

Measures size work; they never reach a key, a context fingerprint or an output. A wrong or hostile record can only
mis-size: a batch too large is caught by bisection and the OOM record (§7.4); a cut too long overruns `quantum_s` and
is caught by drift (§7.3). So shared records need provenance and a reader's choice, not signatures.

- **Every record kind** (`obs`, `drift`, `supersede`, `regime`, `oom`) carries the same provenance: `job` (the
  scheduler job id, when there is one), `run`, `senselab` (version and 40-hex commit when known), the profile's
  canonical digest and the worker's `HardwareClass`. **No record carries an account name or a hostname**: a measures
  root is a location other users read, so its files follow the widest audience's rule (UX-57). A writer's identity is
  its file's owner uid, read from the file system, never from content; index entries hold a `HardwareClass`, which carries no account or host. A
  host is identified, where diagnosis needs it, by its `HardwareClass` and the scheduler job id.
- **One reader filter over all of them.** `calibration.trust` is `group` (read every file of every root) or `own`
  (only files whose owner uid is the current user's). A `drift`, `supersede`, `regime` or `oom` record acts only for readers whose filter
  admits it; under `own`, another user's records never supersede or cap your measures.
- **Who sets it.** The profile sets the default (`laptop` and the `slurm` template `own`; a user's site profile may set `group`). A user opts out per run with the
  run config key `calibration.trust: own`, which overrides the profile's value (P-34's run-config form; ruling O-3).
- **`collect`** removes only files the collector's uid owns, or files a record it trusts supersedes and that its uid
  owns. It never removes another user's file.
- **A file of a format tag the reader does not know** (a newer release's) is skipped, never moved. **A malformed or
  misfiled file** is quarantined only when the reader's uid owns it and its root is writable; any other is skipped.
  Each is logged once with its owner uid, rendered under the reader's audience rule, and never fails a run. One member
  never makes others lose measures (UX-10, UX-61).
- **What leaves the store** (UX-43, UX-45, P-28). Result and run provenance, `plan.explain()` and exports name an
  estimate's source by probe ids and observation counts only. The store's own listing renders owner uids under the
  visibility policy.

### 3.6 Invalidation

1. **Identity change.** Any field of §3.1 changes the digest; the old directory is never read for the new unit.
2. **Measured staleness.** Drift under co-tenancy writes a `regime` record; drift in the probe regime makes the
   estimate suspect, and a confirming probe decides whether it is stale (`supersede`) (§7.3). A `supersede` record names the probe ids it outdates (`outdates`, the ids of the suspect
   estimate's basis); a trusted record's ids do not pool, whatever their files' times, so a copy, restore or clock skew
   changes nothing and the order of writing does not matter.
3. **No age rule.** A wall-clock time-to-live would be a fixed parameter with no measurement behind it.
4. **Collection** (store policy's `collect_interval_s`): observation files the collector wrote that a trusted record
   outdates, keeping
   the newest `calibration.observations_kept` for audit, and staging and quarantine older than the policy's ages.

### 3.7 Lookup, and how a group calibrates its resource

Lookup order for a unit (`MeasureSource.get`):

1. the measures roots, in order, identity match, their observations pooled and estimated once, the estimate used when
   it is `measured`;
2. otherwise the Pool probes, and the probe's observations are written to the first writable root.

`default_measures` is `MeasureSource` over the profile's roots; it is the one source `explain`, the in-process executor
and the Pool read, so the plan and the run size alike. A unit no root holds is unmeasured: `explain` says so and names
the reason, and a run sizes it by the safe values until its probe has run.

Senselab ships no table of measured rows (owner ruling, 2026-11-09). A measure is read only on a unit's whole identity
(§3.1: card class, backend params, env build, hardware class; M10, must-fix 6), so a row shipped with the package was
read only on the one configuration that generated it, and every other machine probed anyway while the package carried a
second lookup path, a generator, a format and a commit-bound provenance for it. Senselab is general purpose and its
users' resources are not ours to measure; what it provides is the means for a group to measure its own.

**How a group calibrates its resource.** The ladder (§9) is the calibration run: it is a probe run that writes into a
measures root the group names.

```
uv run python benchmarks/scale/ladder.py run /path/to/suite.yaml --execution /path/to/profile.yaml \
    --measures-root /path/to/group/measures
```

and the group lists `/path/to/group/measures` in `calibration.measures_roots` of its execution profile (or sets
`SENSELAB_MEASURES_ROOT`). The root is an ordinary `measures@1` root, written by the probe itself, so it keeps what a
copy cannot: the writer's owner uid (the trust filter, §3.5), the order of `supersede` and `regime` records, and the
later-minor rule of `measures@1`. Step 1 reads it on identity match like any other root; there is no second lookup
path, no second format and no tool that copies between roots (a copy was considered and rejected for those reasons).
A unit absent from it is probed, and its probe is written to the first writable root. Roots pool across senselab
commits, as they have since `code_version` was removed (T20R-8); no commit is bound to a measure.

## 4. The Probe protocol (`Probe.plan`, pure)

### 4.1 Inputs

`Probe.plan(unit, candidates, prior, config, formation, seconds, *, bucket_edges) -> ProbePlan` (`seconds` is the
warm-work budget, §4.5; `bucket_edges` are the input-unit edges of a `bucket:` formation's grid, empty otherwise; the plan
returns the `WARMUP` and `SINGLE` items, and `Probe.plan_batched(plan, single_estimate, grant, config, seconds_left, *,
memory_headroom)` adds the `BATCHED` ones once `SINGLE` is estimated, the worker's grant (§6.3) giving their memory
batch; `Probe.replace_served` swaps a `served` item for the next candidates of its stratum):

- `candidates`: for each held job of the unit that the store would **not** serve (§4.2 step 1), its fragment and
  index, its key digest and its input units (§5.1);
- `prior`: `Prior(observations, records)`: the unit's pooled observations (own, peers', or a superseded probe's still
  valid) and its regime and `oom` records, oldest first (the records re-open strata, §4.2 step 5);
- `config`: the `calibration` section (§11);
- `formation`: the backend's formation; `single` plans no batched phase.

### 4.2 Choosing jobs (owner requirement 1)

1. **Servable jobs are not candidates, and they do not wait.** Whether a job would be served is decided by
   `ResultStore.admits(key_digest, context, policy, run)`, the predicate `_serve` itself uses (P-37): an entry under
   the key digest whose context fingerprint is the current one, or any member the run's reuse policy admits (the
   policy decides: equivalence members under `equivalent`, any context, labelled, under `tolerant`), that the license rule admits (`_license_admits`) and that verify sampling (`sampled`, which is
   deterministic in the key and the two contexts) would not recompute. It lists `<d>/` for fingerprints and reads
   `context.json` of non-exact entries only; it decodes no output. The current context is the env record's
   (`env.context_fields()`); where it is not yet known (an env not yet built), every job is a candidate. Servable jobs
   are put at once (I4). An entry under an old context, after an env or hardware change, is therefore not servable,
   and its job is a candidate (review B1).
2. **A probe request that returns `served`** (the predicate's view went stale, or the context was unknown) is
   discarded by the classifier and replaced by the next candidate of the same stratum, within the derived `probe_items` (§11).
3. Drop all but one job of each key digest (M4, R-232).
4. Cut the candidates that have a length into `calibration.length_strata` strata of equal count by input units; a cut
   never splits equal lengths, and bucket edges never move it. On a bucketed formation each batched item draws its jobs
   from one bucket, so its jobs' keys are unchanged; a stratum's items may come from different buckets, and its supply
   at width w is Σ_b ⌊n_b / w⌋. Jobs without a length form one more stratum.
5. Skip each (phase, stratum) that `prior` already covers with `calibration.min_requests` steady observations, and
   that no unlifted trusted `oom` record re-opens (§7.4): that is the stored timing reused instead of a run. A run whose lengths
   lie inside what `prior` covers plans nothing.
6. Within a stratum, `WARMUP` and `SINGLE` take its shortest candidates (bounded seconds, I3); `BATCHED` takes
   candidates spread evenly by rank of length.

If fewer candidates exist than a phase needs, that phase is not planned and its dimension stays unmeasured (§6.5);
the jobs left run as production.

### 4.3 Phases

Each item carries `probe: {id, phase, stratum}` and is one request (§4.4). Phases declare intent; whether an
observation counts is decided by `Probe.estimate` from what the worker reported (§5.3), never from the phase alone.

| phase | items | gives |
|---|---|---|
| `WARMUP` | `calibration.warmup_requests` one-job items from the shortest stratum | `load_s`; warms the process |
| `SINGLE` | per planned stratum, `min_requests` one-job items; in the middle stratum also `min_requests` items of `calibration.single_item_jobs` jobs, each job its own request | per-request cost `r + u·L`; per-item cost `a`; device bytes per input unit |
| `BATCHED` | per planned stratum, `min_requests` items of one request of `B₁` jobs; with `calibration.batched_sizes` 2, also of `B₂` | per-request cost at batch B; device bytes at B |

`BATCHED` is planned after `SINGLE` is estimated:

- `B₁` = the least of: the memory batch at the stratum's longest candidate under the probing worker's grant; the
  largest batch the item budget leaves for `length_strata × min_requests` items; the largest batch whose items the
  `SINGLE` estimate's upper prediction (`t(L)·(1 + z·spread)`, §6.2) fits in the probe seconds left (I3); and
  `pool.quantum_items`. The fragment's size no longer sets it (R-237).
- `B₂` = the integer nearest √B₁ (the geometric midpoint between one and `B₁`), when it is at least 2 and differs from
  `B₁`: it tells a throughput curve that peaks inside from one that rises to the memory bound.
- No `BATCHED` phase when the formation is `single`, the memory batch is below 2, or device memory is unmeasured.

### 4.4 One item per claim, on a pinned worker

- The Pool puts a probe's items on a probe queue, `<queue>~probe-<id>`, under the run's queues. A worker serving the
  queue's class, whose `HardwareClass` is the resolved one (§3.2), takes the probe queue's serve lease before it claims
  from it; a worker that finds the lease live skips
  the probe queue. The lease is the pin: one worker runs the probe, and its beats show it alive.
- A probe item is claimed alone (`max_items=1`), never gathered (M3; R-234, R-236). The pinned worker drains its probe
  queue before production claims, quantum yielding aside, and waits on it with `pool.first_poll_s` (doubling to
  `poll_s`) while its serve lease is held, so a phase change costs no full poll.
- If the pinned worker dies, its leases go stale; another worker takes the serve lease and the remaining items. Its
  first requests report a load (§5.3), so warmness needs no special case.
- With `warmup_requests` 1 and a single candidate the plan is the WARMUP item alone: the probe measures the unit's load
  and no `r` (`plan_batched` returns `single_unmeasured`), the job's result is kept, and the unit stays unmeasured.

### 4.5 Budgets (M8)

- The jobs a probe plans are bounded by `probe_items(config, pool.quantum_items)` (§11), derived, not a key; `B₁` is chosen inside it (§4.3).
- The seconds budget (`probe_s`, §11, an argument of `Probe.plan`) bounds **probe work**: the sum of the walls of the probe's warm requests, counted from the
  first warm request. The model load (already `load_share`'s and `timeouts.load_s`'s) and the waits between phases do
  not count, so a model that loads for minutes is still measured (I3). Past it the Pool puts no further probe item;
  items already claimed finish, because they are real work whose results are kept, and §4.3 bounds each item's
  seconds. A phase below its minimum counts stays unmeasured; a measured phase is kept. `SINGLE` complete and
  `BATCHED` incomplete is a measure with batch 1; `SINGLE` incomplete is unmeasured. Warm observations are written
  either way, so later runs accumulate them (§7.1).
- The budgets count only probe items; production items of other units, and the unit's servable jobs, are not held.

## 5. Observations and the classifier

### 5.1 Input units

A job's input units are its recording seconds (`job_length_ns`), else its held text's characters
(`job_text_chars`), else none. One unit kind per `MeasureUnit`; a unit whose jobs mix kinds is measured per request
only (`u` unmeasured, `r` the per-request median), valid inside the observed request count.

### 5.2 `Estimate`

```
Estimate
  status: measured | unmeasured(reasons)
  unit_kind: audio_s | chars | none
  load_s, load_n
  item_s (a), item_n                       per-item cost beyond its requests: claim, bind, commit
  single:  r, u, n, spread, range (lo, hi in units)
  batched: per B: r_B, u_B, n, spread_B, range
  batch:   per stratum (lo, hi in per-job units): the chosen B* (1 when no batch is faster beyond the margin)
  device:  resident, m0, m1 (bytes per padded unit), n | None (unmeasured, never 0)
  host:    resident, m0, m1, n | None
  regimes: per co-tenancy k: ρ_k, n             from regime records (§7.3)
  oom:     [(resource, B, L, cotenancy)]         from oom records (§7.4)
  source:  probe ids and counts
  estimator: estimate@1
```

### 5.3 `Observation`: what a worker reports per request

```
Observation
  probe: {id, phase, stratum} | None       None for a production request (telemetry, §7.3)
  item, request_index, jobs, batch
  units, padded_units                      Σ input units; batch × the longest job's units
  wall_s                                   request start to end inside the worker (M6)
  item_wall_s                              on the item's last request: claim taken to commit
  since_load                               requests this model state answered before this one; None on a request
                                           that raised
  load_s                                   on a request that loaded: the load's seconds, else None
  device_peak, device_after                torch peak and allocation after, else None
  host_peak, host_after                    the model process's peak and after, else None (T20R-11)
  cotenancy: {device, host}                pool workers with compute processes on its device (the most on any one
                                           of its granted GPUs; a worker holding several processes counts once),
                                           read at most once per pool.presence_refresh_s; None: unread. Live pool
                                           workers on its host (the run's workers directory), likewise
  outcome: ok | failed | bisected | served
  hardware: HardwareClass as the worker reads it
```

`since_load` and `load_s` come from the process that holds the model, not from the queue. `ModelCache` counts each
load; the worker reports, on `done` (wire-protocol amendment, T20R-5), how many requests the cache entry answering
the request has answered since it was loaded, and the load's seconds when this request loaded it. An LRU eviction and
reload, a worker-process restart, a second backend on one queue and a second model-less unit on `cpu` all reset it.
A server backend counts per attested server instance on the consumer side: one count for a server's address and
attestation across the consumers in a process. A request that raised has no `since_load`; an unread device count
is `None` and the observation is kept (§5.4 step 1 treats it as shared). `Sample.cold` goes.

### 5.4 `Probe.estimate(observations, records, config, *, hardware, unit_kind, memory_headroom) -> Estimate`: the only classifier

1. **Keep** an observation only if `outcome` is `ok` (M7: failed, bisected and served requests never enter, nor does
   any request of a bisected item), its `hardware` is the unit's, and its co-tenancy is the probe regime (device 1,
   host as its own probe ran: the least host co-tenancy among that probe's requests) for the `single`/`batched` fits; other regimes feed `regimes` only (§7.3).
2. **Warm** iff `since_load ≥ calibration.warmup_requests` (M2). A request with `load_s` set gives `load_s` and
   nothing else. Cold or reloaded requests never move `r`, `u`, `a` or the batch choice.
3. **Per request, single:** fit `wall_s = r + u·units` over warm batch-1 requests by Theil–Sen (the median of
   pairwise slopes; robust to one outlier per three, no tuning constant). `spread` is the median absolute residual
   over the median prediction, scaled to a standard deviation (× 1.4826, an invariant: MAD to σ under normality).
   Fewer than two distinct strata with `min_requests` each: `u` is unmeasured and `r` is the median, valid within
   `range`.
4. **Per item:** `a` = median of (`item_wall_s` − Σ the item's request walls) over warm probe items, both sizes (M3).
5. **Per request, batched at B:** fit `wall_s = r_B + u_B·padded_units` the same way, with its own `spread_B`.
6. **Batch decision**, per stratum at its median length L: per-job time single `t₁ = r + u·L`; batched
   `t_B = (r_B + u_B·B·L)/B`. B beats one only when `(t₁ − t_B)/t₁ > z·σ_d`, with
   `σ_d = 1.2533·√(spread₁²/n₁ + spread_B²/n_B)` (1.2533 = √(π/2), an invariant: the standard error of a median
   relative to a mean) and `z = calibration.confidence`. `B*` is the best B that passes, else 1. A B never measured is
   never chosen (S2).
7. **Device memory:** over warm requests with device figures, `resident` = min `device_after`; `peak − after` fitted
   as `m0 + m1·padded_units` (Theil–Sen). No device figures: `device` is None, explicitly unmeasured.
8. **OOM records** in force raise the fit (§7.4). A probe measures and lifts in the probe regime only: a record at
   co-tenancy 1 and B ≥ 2 is lifted once a probe outside its `basis` has `min_requests` warm `ok` batched observations
   in the stratum holding its length; every other record stays in force and never re-opens a stratum. `Probe.plan`
   re-opens by the same predicate, and `oom` holds the records in force.
9. **Minimum counts (M9):** each quantity needs `min_requests` warm observations in each stratum it uses; below that
   it is None and the reasons name what is missing. `status` is `measured` when `r` is measured; every other None is
   treated by S5 in `size`. A stratum here is a length range: the kept warm requests' per-job lengths
   (`padded_units / batch`) cut into `length_strata` ranges by §4.2 step 4's engine. A probe's `stratum` label is not
   read, so probes, members and runs pooled together share strata by length.

`hardware`, `unit_kind` and `memory_headroom` are required: a caller never inherits a guessed value. `MeasureSource`
gets `Probe.estimator(config, records_of=, unit_kind_of=, memory_headroom=)`, which resolves each per unit, so
`estimate=`'s `(unit, observations)` signature is unchanged. A median-only fit carries its spread
(`1.4826·MAD/median`). A fit with a negative slope is not trusted (`u` unmeasured, `r` the median); a negative intercept
is held at 0. `reasons` is a declared
type and names no identifier. A regime factor exists only for k ≥ 2.

Properties pinned by tests (T20R-4): a cold or reloaded observation never changes `r`, `u` or `B*`; a failed,
bisected or served observation never enters any figure; an observation of another regime never enters `r` or `u`; a
batch above every measured B is never chosen; one outlier among three moves no median beyond the others' range; the
estimate is invariant to the order of observations.

## 6. Sizing (`size`)

### 6.1 Signature

```
size(estimate: Estimate | None, grant: Grant, config: SizingSettings, work: LengthProfile) -> Sizing
LengthProfile: the input units of each waiting job, in put order (None for a job whose length is not yet known)
Sizing: batch(L) per length bucket, item cuts (job index ranges), workers, workers_per_device, predicted_s per job,
        and per figure whether it is measured
```

`size` stays pure and is the one sizing site (S6), for the Pool, the in-process executor and `explain()`.

`size` also takes `calibration: CalibrationSettings` as a keyword: its `confidence` is the z of the item cut and its
`unmeasured_workers` the workers of an unmeasured unit. Every figure that sizes work at plan time or run time (batch,
cuts, workers, predicted seconds, wall time, GPU hours, peak memory, the flag of a job whose batch-1 peak passes the
cap) is a field of `Sizing`; no other function estimates one. `estimate_of`, which read the `resource-table@1.0` rows
as an `Estimate` over recording seconds with no range, spread or oom, is deleted (T20R-8).

### 6.2 Time, scaled to the run's lengths (S1, must-fix 1)

- Per job at batch 1: `t(L) = r + u·L`; at the chosen batch B for its bucket: `(r_B + u_B·B·L_pad)/B`; under a
  measured regime k, times `ρ_k` (§7.3).
- **Items are cut at the upper prediction** `t(L)·(1 + z·spread)` with `z = calibration.confidence` (I9), so about
  one item in twenty overruns at the configured confidence, whatever the spread; no separate dispersion threshold.
  `split` walks the jobs in put order and closes an item when its upper-predicted seconds plus `a` would pass
  `quantum_s` (within the class's `time_s`), or when its seconds reach `(load_s + a)/load_share`, whichever comes
  first, and at most `pool.quantum_items` jobs.
- **Workers** = ⌈Σ median predicted seconds / `target_wall_s`⌉, at most the item count, `max_workers` and the device
  bound (§6.3).
- Outside the estimate's measured `range`, time is extrapolated linearly for cutting items only, logged once per unit,
  and the batch is 1 (§6.4).
- A job of unknown length is cut at the range's `hi` and its time is unmeasured. An unmeasured spread or slope never
  enters as 0: with `u` or the spread unmeasured, items are cut by `pool.quantum_items` alone, and with `u` unmeasured
  a job outside the range is untimed and its unit takes the no-estimate workers (§6.5). A gap between two strata takes
  the smaller neighbouring batch.

### 6.3 Device memory divided among a card's workers (S3, S4, must-fix 2)

On a substrate whose workers share devices (`Local` with visible GPUs):

1. With `device` unmeasured: one worker per device, batch 1.
2. Otherwise, for the run's longest bucket length `L_max` and its chosen `B*`, the batch peak is
   `P = resident + m0 + m1·B*·L_max`. Workers per device `w_d = max(1, ⌊memory_headroom·memory_cap / P⌋)`, lowered
   where a measured regime factor says more workers do not add throughput (`k/ρ_k` not increasing in k).
3. Each worker's grant is `memory_cap / w_d`; every memory batch (§6.4) is computed against that share, so the
   workers of one card together stay within `memory_headroom` of it.
4. The device bound on workers is `devices × w_d`.

A device grant is the class's `memory_cap_bytes`, else the least memory of the devices seen (`Grant.seen`), as the host
grant falls back to the host's. Regime lowering treats an unmeasured `k` as linear (`ρ_k = 1`) and stops at the first
measured `k` whose `k/ρ_k` does not rise (ruling O-4).

On Slurm each worker holds its own card: `w_d = 1` and the grant is the whole `memory_cap`. Ruling O-4: `w_d` is
bounded by memory, and lowered only by a measured regime factor (§7.3).

### 6.4 Batch per length bucket (S2, S3)

For each length bucket of the run (the formation's grid for a bucketed backend, else the estimate's strata):
`B(L) = min(B*(L), memory batch at L, the item's jobs)`, the memory batch at L being
`⌊(memory_headroom·grant − resident − m0)/(m1·L)⌋` with `m1` as §7.4 raises it. `B(L) = 1` when L lies above the
measured range, or when device memory or batched time is unmeasured.
An executor that does not split fragments asks `size` for its fragment size (`fragment_size`) and cuts a fragment into
runs of equal batch at stratum boundaries; cutting at a formation grid's buckets takes the grid as `size`'s `bucket_edges` keyword, in the estimate's input units
(T20R-7a-0): no item spans two buckets. The item's placement carries its bucket's
batch; a fragment whose jobs fall in several buckets is cut at bucket boundaries.

### 6.5 Missing measure: the safe value per dimension (S5)

| missing | value |
|---|---|
| no estimate, probe not possible or over budget | batch 1; one worker per device; items of `pool.quantum_items` jobs bounded by `quantum_s` with the worker's own yielding; `calibration.unmeasured_workers` workers |
| spread | items cut by count |
| device memory | batch 1; one worker per shared device |
| batched time | batch 1 |
| `u` | per-request median, items cut by count within the measured request range; a job outside the range is untimed and its unit takes the no-estimate row |
| grant memory | batch 1 |

### 6.6 One site for plan time and run time (S6, must-fix 4, UX-31)

`profiles._batch_size` is removed and a placement records batch 1. `plan.explain()` calls the same pure `size` with
`MeasureSource.get(u)` (never a probe, never a download, never a worker start) over the lengths known at plan time
(graph inputs; a node fed by an upstream recording has unknown lengths until it runs). It states wall time, GPU hours
and peak device memory per node, each labelled `measured` (with the estimate's source, §3.5) or `unmeasured` (with
the reason: no estimate, hardware not yet seen, lengths unknown), how many jobs are servable and how many will be
computed (§4.2's predicate), the measures root writes will go to (§3.3), and the waits a run may incur on another
run: the lease takeover delay and `peer_wait_s` (§3.4, UX-36, UX-42). A job whose predicted batch-1 peak at its
length, `resident + m0 + m1·L`, exceeds the class's memory cap is flagged before running with the memory it needs
(UX-2, UX-4). The Pool and the in-process executor call
the same `size` at run time with the same device selection: the device pair on a `cuda` class, the host pair
otherwise. The in-process executor never probes; without an estimate it runs the safe values of §6.5.

`plan_cost`, `check_caps`, `MemoryCapExceeded` (and its `memory_cap_exceeded` envelope code) and `Cost` are deleted
(T20R-8): a plan states no cost of its own. Wall time, GPU hours and peak memory are each node's `Sizing`; the flag for
a job whose batch-1 peak exceeds its class's cap is `explain`'s, as above; `placement@3.0` drops `cost` and `explain()`
prints no `cost:` line. The download size went with the `@1.0` rows' `download_bytes`; the probe measures requests, not
downloads. A reason printed for an
unmeasured figure is true of the state: the unit's reason above, `no estimate for its unit`, `lengths unknown`. One `MeasureUnit` resolver (`UnitResolver`, T20R-7a-0) over the env record and the hardware index is what `explain()` and
the in-process executor use by default, with `MeasureSource` over the profile's roots; the Pool builds every unit it reads
or writes through it (T20R-7a). The env comes from one `EnvStates.get` (`Packaged.env_read`, raising `EnvUnreadable` for any
cause; a built handle is kept under the env's identity, its env, worker overlay and requirements, so backends sharing a name and version across tasks read their own envs; an unbuilt env is read again), which the servable counts use too. The decoder is read from the
plan's recorded sources (a path source of a port with a reader is file-backed, a JSON value is not, a `{"memory": ...}` source
is as the live value, unresolved when this process does not hold it), else the unit is unresolved, as it is when the plan records no sources; a node fed by both has two units. A bucketed backend's formation is the
grid's `bucket:<grid id>`. An unresolved unit says why: `hardware not yet seen on this host`, `this host reports no hardware
class`, `the env of backend <b> is not built` (or `cannot be read`), `the node a slurm worker runs on is chosen at
submission`, an unreadable server attestation, or `its jobs read both files and in-memory inputs, which are two units`.
`explain` and logs name a unit by the first 12 hex digits of its digest, never an identifier. `explain` counts against the
store a run would use (`default_store()`, or the `store` given), binding the plan's inputs as a run does: per resource class
the jobs servable (`ResultStore.admits`, the one predicate, asked on `pool.admit_threads` threads), computed, and
undetermined (an unbuilt, unreadable or Slurm env; a key that waits on an upstream result). Binding is needed only for the counts and lengths: a plan whose inputs cannot be bound (changed, held by another process, or no sources recorded, `SourcesUnrecorded`) still renders, every class's counts undetermined and its lengths unknown, each with the binding error's reason (UX-32). `size` marks
`Sizing.extrapolated` when a known length lies outside the measured range; the executor logs it once per unit per run (a new
log at `bind_epoch`). `regime` and `oom` payloads are typed by one table (`RECORD_MODELS`) that `put_record` validates on
write and `regime_records` reads; a payload change bumps `measure-regime@` or `measure-oom@`.

## 7. The Pool: probe and production items

### 7.1 `ProbeState`, per unit

```
 New ──► Resolving (pending causes, §3.2) ── hello ──► Lookup
 New ── final cause ──► Unmeasured;  Resolving ── final cause at hello, or resolve_s out ──► Unmeasured
 Lookup ── get() measured ──────────────────────────────────────────────► Measured
 Lookup ── none ── lease taken ──► Probing ── estimate measured ────────► Measured
                                      └──── budget out / too few jobs ──► Unmeasured
 Lookup ── none ── lease busy ───► PeerWait (≤ peer_wait_s) ──► New (peer's observations as prior)
 Measured ── drift, k > 1 ──► Measured (regime record, no probe)
 Measured ── drift, k = 1 ──► Confirming (one probe, prior excluded, jobs held) ──► Measured (supersede, or no record)
```

- Servable jobs (§4.2 step 1) bypass the state machine and are put at once, in every state.
- `Probing` holds the unit's computed jobs, puts `WARMUP` and `SINGLE` items on the probe queue, estimates once they
  have outcomes, puts `BATCHED` items when planned, then estimates again.
- At `Measured` or `Unmeasured` the probe's observations are written to the write root (`obs/<probe id>.json`, after
  `prove()` of the probe lease), the lease is released, and the held jobs are put, sized by §6. **Every probe job's
  result is already in the result store and the Pool yields it like any other**: probe items are ordinary items whose
  outcomes carry `ResultRef`s. Owner requirement 1.
- `Unmeasured` writes its warm observations too; a later run pools them and may reach the minimum counts without
  repeating them. A unit probes at most once per run, plus at most one confirming probe (§7.3): no loop.
- `PeerWait` returns to `New`, where `Lookup` either finds the peer's estimate or plans only what the peer's
  observations leave uncovered (I8).
- A retry of a probe job is a production retry (one-job fragment), never a probe item.

Amended 2026-11-08 (T20R-7a review B-1, B-2, I-2, I-3):

- **Order while a unit is pending.** Servable jobs are put at once, as in every state. The pending unit's computed
  jobs are held in the order they arrive and hold no worker. The queue wants one worker while any of its units is
  `Resolving` (§7.2), started as it would be anyway, and other units of the queue proceed by their own states: a
  resolved unit's production and probe items are put as they come and are never delayed by a pending one. At the
  resolving hello every pending unit of the queue is built at once, and each unit's held parts go to `Lookup` in the
  order held, before parts of that unit that arrive later.
- **Held parts are checked like flights.** On every poll, each part a unit holds (in `Resolving`, `PeerWait`,
  `Probing` or `Confirming`) is checked against its queue exactly as a flight is: an env held on the queue raises
  `QueueHeld`, and a broken class or substrate (`Fleet.broken_for`) raises `WorkersBroken`, each once the results
  landed with it are given. A class whose workers never say hello therefore breaks and is reported (UX-36, UX-37).
- **Resolving is bounded.** The bound is `ExecutionProfile.resolve_s` (§11), on the Pool's clock, counted from the first
  poll at which a worker of the unit's queue is live and not pending. A worker waiting in the scheduler's queue
  starts no clock and breaks nothing (`Fleet` counts failed starts only), so parts of a class whose jobs stay pending
  wait as production items do today: no new hang, and no new coverage. When the bound passes with no resolving
  hello, the unit goes `Unmeasured` with the reason `no readable ready mark from a worker of <queue> within resolve_s
  of starting (<n> unreadable)`, and its held parts are put as production under the safe values (§6.5).
- **The probe queue's fallback (I-2).** The stale window (`lease.stale_after_s`) runs from the probe queue's last claim
  or outcome, else from its put. When it passes, the Pool first writes the probe queue's write-once `closed` mark,
  which a worker checks before it takes the serve lease or claims (`_serves`): a closed probe queue is served by no
  one. Only then are the jobs of its unclaimed items put as production; an item under a live claim finishes on the
  closed queue, and its jobs are put as production only if that claim goes stale. The unit goes `Unmeasured` and
  writes the warm observations it has.
- **A disagreeing hello (I-3, §3.2 step 5).** Every unit of the class resolved from the prediction is resolved again
  under the hello's class. A `Probing` or `Confirming` unit's probe is abandoned: its probe queue is closed as above,
  its in-flight items finish and their results are yielded, the observations they made are written under the class of
  the worker that made each (§3.2 step 4) and the probe lease is released; the jobs of its unclaimed items and its held
  parts return to `Resolving` and are placed under the new unit. A `Measured` unit is dropped: items already put keep
  their sizing, which costs efficiency and changes no result, and every later part is placed under the new unit. A
  `PeerWait` or `Lookup` unit is placed again under the new unit.

### 7.2 What replaces what

| old | new |
|---|---|
| `_calibrating`, `_first_round`, `_round_digests`, `calibrated` | `ProbeState` per unit, with its probe id and items |
| `_batch_round` | `BATCHED` phase planned by `Probe.plan` from the `SINGLE` estimate |
| `_round_measure`, `calibrated` | `Probe.estimate` |
| `workers_wanted` for an unmeasured queue (`calibration_workers`) | a probing unit wants one worker for its probe queue; a queue with a unit in `Resolving` wants one worker; other units of the queue are sized by their own estimates |
| `_grow` | `oom` record (§7.4) |
| `Measures` (cache root by code version) | `MeasureSource` over the measures roots; a unit they do not hold is probed |

### 7.3 Drift and regimes, from production telemetry

Production items carry a per-request `Observation` (`probe: None`) in their outcome. Ruling O-5: production requests
are drift evidence only, never pooled into `r`, `u`, `a` or `B*`.

- For each warm, `ok`, unserved production request whose `hardware` equals the estimate's, the Pool computes
  `ρ = wall_s / predicted`, the prediction at its batch and units in the probe regime. A request of another hardware
  class feeds no drift.
- Requests are grouped by regime (device and host co-tenancy). Over the last `drift_window` requests of a
  regime, when the median ρ lies outside `1 ± drift_confidence · 1.2533 · spread_B / √drift_window` (`spread_B` at the
  requests' batch, `spread` at batch 1), the Pool writes a `drift` record (the window's ρ, regime, batch, estimate
  source). The regime that drifted decides what follows:
  - **co-tenancy k > 1**: the drift is a regime effect by construction. The Pool writes a `regime` record
    `{k, ρ_k, n}` from the window (ρ_k its median ρ, n its size) at once, with no probe; `size` scales predictions in
    that regime by `ρ_k` and §6.3 may lower `w_d`. The newest trusted record of a regime wins.
  - **the probe regime (k = 1)**: the estimate may be stale. The unit moves to `Confirming`, which runs one probe of
    its planned strata with the stored observations excluded from `prior` and, as `Probing` does, holds the unit's
    computed jobs while it runs, so no worker of the unit shares the probing device. If its estimate disagrees with
    the suspect one beyond their bands, the Pool writes a `supersede` record and the confirming probe's observations
    are the unit's new basis. If it agrees, or is unmeasured (another process shared the device), nothing more is
    written and the unit returns to `Measured`.
  A unit writes at most one regime record per regime and runs at most one confirming probe per run. A drift record
  alone supersedes nothing, so two runs over different content cannot supersede each other's measures by drift alone
  (I2).
- A device peak above the prediction at its padded units beyond the same band raises `m1` at once (§7.4).

Amended 2026-11-09 (T20R-7b):

- **Regime is the device co-tenancy.** A production request's regime is `k = Observation.cotenancy.device`; a request
  whose device count is unread (None) is no drift evidence. `k = 1` is the probe regime, so a host-only co-tenancy on a
  `cpu` class (device count always 1) is never a regime record.
- **Evidence.** A request is evidence when it is `ok`, warm (`load_s` None and `since_load` at least
  `warmup_requests`), of the unit's hardware, unserved and has a positive wall; its prediction is the single fit at
  batch 1 and the batched fit of its batch above 1, with that fit's `spread`. A request whose batch has no fit, or
  whose fit has no spread, is not evidence. The window's band uses the median of its requests' spreads. A window that
  drifted starts over.
- **Records.** Every kind's payload is a typed model (`RECORD_MODELS`: `DriftRecord`, `SupersedeRecord`, `RegimeFactor`,
  `OomRecord`). A `drift` payload is `{metric, k, rho[], median, band[low, high], batch, source[]}` (`metric` is
  `wall_s`, or `device_peak` with `{m1, batch}`); `source` is the probe ids of the estimate it doubts. A unit writes
  at most one `drift` record of the wall metric per co-tenancy and at most one `regime` record per co-tenancy per run;
  the estimate is read again from the roots at once, so `size` scales by `ρ_k` for the rest of the run. A window at
  `k > 1` is judged against the factor in force: its ratios are `wall_s / (ρ_k · prediction)` where a regime record of
  that `k` is in the estimate that sized the request, so a later run at the same co-tenancy writes no second record.
  The record's `rho` is the window's median of `wall_s / prediction`.
- **Confirming.** At `k = 1` the unit is `confirming` with no probe yet: the next part of the unit to arrive is the
  held part, planned fresh (stored observations out of the prior, the unit's records in), probed under the unit's
  lease, and the unit's later parts are held until it ends. The probe starts only once every item of the unit already
  put has its outcome read, a job's retry on the same class included (it is noted under its parent's unit), so no
  worker of the unit shares the probing device; parts that arrive meanwhile are held. With no further part there is nothing to confirm and
  nothing runs. A part with nothing worth probing, or a peer holding the lease, returns the unit to `Measured`. The
  probe disagrees when its single fit and the suspect's differ, at the middle of the probe's measured range, by more
  than `drift_confidence · 1.2533 · hypot(t₁·spread₁/√n₁, t₂·spread₂/√n₂)` (`t` the fit's seconds there); a fit with no
  spread cannot disagree. A `supersede` record `{outdates, confirmed_by}` names the suspect estimate's probe ids; a
  suspect with no probe ids names none and writes none. The probe's observations are written either way.
- **Device peak.** A warm request whose `device_peak − device_after` exceeds `m0 + m1·padded_units` of the device fit by
  more than the band (at the spread of its batch and `drift_window`) raises the unit's `m1` to `(peak − after − m0) /
  padded_units` for the rest of the run, held across the estimate's re-reads, and writes one `drift` record. It is not
  persisted as an estimate: the next run's probe measures it, or a fault records it.
- Content effects beyond length (two concurrent runs whose content differs in cost at equal length) are not in the
  identity; each run confirms at most once, so such a disagreement costs one probe per run, never a loop.

### 7.4 OOM backstop (S7)

A resource fault on an item at batch B and longest length L writes an `oom` record `{resource: device | host, B, L,
cotenancy, worker, grant_bytes, basis, run}`, where `worker` is `worker_id()`, `grant_bytes` the faulting grant's
memory and `basis` the probe ids of the estimate that sized the faulting batch together with the faulting item's own
probe id when it is a probe item (`Estimate.fault_basis`), so a probe's own successes at another batch never lift its
fault. It acts in two ways:

- **The fitted model is raised, no constant.** At B ≥ 2, for lengths at or above L, `m1` is raised to the value at
  which the memory batch at L becomes B − 1, so the faulting pair is infeasible:
  `m1′ = max(m1, (memory_headroom·grant − resident − m0)/((B − 1)·L))`. The memory batch at every L′ ≥ L follows
  from `m1′` (§6.4). A fault at B = 1 has no batch to shrink: the record marks length L infeasible under this grant,
  `m1` is not raised, and the item goes to the re-placement on a larger class below. A record whose co-tenancy shows other processes on the device
  applies to that regime and above only.
- **The stratum re-opens.** The record makes its stratum uncovered for `BATCHED` (§4.2 step 5), so the next probe
  measures memory there. A probe outside `basis` with `min_requests` warm batched observations in the stratum lifts the record, and its
  observations replace the raised `m1` there;
  if the probe itself faults, a new record raises it again and the bisection proceeds as now. A record whose
  co-tenancy is above 1 never re-opens: a probe runs alone and can neither measure nor lift it.

Any batch ≥ 2 lifts, so a probe at B − 1 lifts a fault at B; the refitted `m1` may permit B again, and the next run
may fault once and probe once more. §7.1's one probe per run bounds this: a cost per run, never a loop.

The bisection of the failing request, the re-placement on a larger class and the jobs' retries are unchanged.

Amended 2026-11-09 (T20R-7b):

- **Where a fault is read.** Each request that failed with a resource fault is a fault at its own batch and longest
  length. A request's fault class is the class of the exception the host raised for it (a worker killed for memory,
  a timeout, a value too large), else, where nothing raised, the one class every job of the request failed with: a
  backend's out-of-memory raises inside the worker and reaches the host as every item's error record, each naming
  `resource` (`call_handle`). A request whose jobs failed with mixed classes names none. `observations()` computes it,
  the one place, into the request's `Observation` (`fault`, `measure-observations@1.1`); the pool writes one `oom`
  record per item from its resource-faulted requests: the largest faulting batch `B` (the shortest `L` among equals),
  `L = padded_units / batch`, and the co-tenancy of the grant that faulted, an unread count filed as 1 (the
  conservative direction: a probe running alone can lift it). The co-tenancy is the device's on a device grant; on a
  host grant it is the request's host count only where the grant is the host's shared memory (a class with no
  `mem_bytes`), and 1 where each worker is granted its own memory, since no other process shares that grant.
  `resource` is `device` on a class whose grant is a device's, else `host`; `grant_bytes` is the
  class's grant; `worker` is the item's `worker_id()`, None where its outcome names none (`measure-oom@1.1`). An item
  whose grant states no memory is logged and not recorded. A batch that faults and then bisects cleanly is recorded
  at its batch; a job that fails alone is recorded at batch 1 and is not liftable as `B ≥ 2`; a bisect caused by a
  lost worker or a protocol error (a fault class other than `resource`) writes nothing.
- **One record per shape.** In a run, a fault at a length no shorter than one already written for the same unit,
  batch, co-tenancy, grant and basis adds nothing: the shorter length is the more restrictive.
- **Reading it back.** The pool reads the unit's estimate again at once, so later parts of the run are sized under the
  raised `m1`. A record a probe can lift (`probe_regime`: co-tenancy 1, batch of two or more) makes `covers` false for the
  unit, so the next run plans a probe whose `BATCHED` phase re-opens the stratum. A record of co-tenancy `k ≥ 2` and
  batch of two or more is read by `size` whenever the card's workers `w_d ≥ k`, and a `host` record likewise on a
  shared host whose workers `w = min(⌊memory_headroom · host / P⌋, max_workers) ≥ k`, each then sized in `host / w`: `m1′ = max(m1, (memory_headroom ·
  grant/k − resident − m0)/((B − 1)·L))` (`probe.raised`, the one formula); a probe never lifts it. The raised `m1` is one
  slope, so lengths just below `L` (within `L/B`) also read a batch below `B`.
- **A backend out-of-memory is bisected per request and recorded.** A call that did not raise but gave a resource
  fault for every job of one of its requests (a backend's out-of-memory raises inside the worker and reaches the host as
  each item's error record) has that request split in halves, found by the request report's tags; the call's other
  requests keep their outcomes and reports. A backend that returns per-item resource errors is re-run about
  log₂ n + 1 times, because on the wire it is indistinguishable from a raised out-of-memory. The record at `B` makes the
  next `size` avoid the batch. A cancel as the halves start keeps the call's other outcomes (`Abandoned.kept`).
- **`since_load` across processes.** A second consumer of a warm server in another process is not covered: the server
  does not report its count, and a `served` request is no evidence for any estimate or drift, so nothing reads it.

Amended 2026-11-12 (T24-7): a load is not a batch. A request that failed while its worker started, loaded or reloaded
its model (`FaultRecord.load`) carries no fault in its `Observation`, so it writes no `oom` record, and a fault at B = 1
that this section marks "infeasible under this grant" is not read from a load. The load's fact is the `unfit` record
of §7.5.

### 7.5 Fit: a unit that does not fit its class (T24-7)

A unit's footprint beside its model is a fact the grant either holds or does not, whatever batch is asked for. Two
things record it, and one rule reads both.

- **Measured.** `estimate.device` (a device grant) or `estimate.host` gives `resident` and `m0`. A class is *too small*
  for the unit when `resident + m0 > sizing.memory_headroom × grant`: no batch of any size fits beside the model.
  `memory_batch` returning 1 there is the safe value for a unit that does fit; it is not a placement. A class is
  judged only when it states its memory (a `cuda` class its `memory_cap_bytes`, or a GPU the coordinator sees).
- **Recorded.** A load that ran out of memory (a `resource` fault with `load` true) writes an `unfit` record
  `{resource: device | host, grant_bytes, source, overlay, worker}` under the unit, whether the load was a probe item's, a
  production item's or a retry re-placed on a larger class: each class that fails writes its own record, against its
  own grant. A class is *unfit* for the unit when a record of the same `resource` names a grant at least as large as
  the class's. One record stands for a grant: a load that fails again in a grant already recorded adds none.
- **What a record binds.** A record is a fact about the unit, so it carries no basis and no probe lifts it. It binds
  the unit (its digest: task and backend versions, params, model revision, env lock, wheel set, platform, numerics,
  hardware class) and the code that failed: `source` is `identity_digest("core-runtime@1", ...)` over the digest of
  every `.py` file of senselab-core and senselab-worker, and `overlay` the digest of the backend's worker overlay
  (`envs.overlay_digest`: the worker runtime and each `worker_sources` module). A change to any of these is another unit or other code, and the record
  no longer binds it; a record is read only under the code it was written by (`fit.current_unfit`).
- **Clearing one.** A record of a load that failed for a reason that was not the unit's (a card another tenant held) is
  removed with `senselab unfit <unit> --clear`, which needs the label or prefix to name exactly one unit (several matches are listed and nothing is removed) and removes that unit's records this user owns; `senselab unfit`
  alone lists every unit with a record. A unit is named by the digest label `explain` prints. No probe lifts a record.
- **The rule.** `choose` walks the class a node is placed on and the classes above it (`larger_class`: the least memory
  above it on the same device and GPU type, never a class the substrate's `qos_cpu_cap` forbids) and takes the first
  that is neither too small nor unfit. A class that states no memory is not judged and is taken. A unit with no measure
  and no record is not judged.
- **Where it is applied.** (1) `Plan.fit`, from `start()`, `coordinate()` and `explain`: each node moves to the class
  `choose` gives; a unit that fits none stops the plan as `Unplaceable` before an attempt is claimed or anything is
  written. The placement's `why` says only that the node moved from a class, never a measured figure: `why` is part of
  the item digest, and a figure re-measured by a later probe must not re-key the items. `explain` prints, per node, each
  class tried with its verdict and the figures (resident, m0, the headroom share of the grant, the grant a load
  failed in) and the class it is placed on or `unplaceable`. On a slurm substrate a unit is `Pending` until a worker's
  first hello, so the plan-level fit judges nothing there. (2) The probe machine's one release gate: every release of
  a unit's jobs, and the start of a probe, is judged against the unit's class, so a unit with an `unfit` record is not
  probed or released on that class and goes above it, or the run stops as `Unplaceable`. A part whose jobs are already
  in a live item stays where it is. (3) The runtime fallback also records: a load OOM no placement predicted writes
  the `unfit` record, holds the environment as `execution.md` §3.4 says, and re-places the job on `larger_class`; the
  re-placed retry keeps its origin under the larger class's queue, so its own load OOM is recorded against that grant.
  The next plan and the next release avoid every class recorded.
- **Unplaceable is a plan outcome.** `Unplaceable` is a resource fault of the plan, not of an environment: its failure
  record (`type` `Unplaceable`, `fault` `resource`, not a gap, not a load) never counts against an env, so it never
  breaks one. `senselab status` prints an `unplaceable:` line naming the unit, each class tried and the figures;
  `senselab explain <node>` prints the same line before the fault; `senselab coordinate` prints it to stderr and exits
  3, with no traceback.
- **Unprobed releases.** A unit can be released with no estimate on five paths (a part not worth probing or unplannable;
  an unresolved unit or a late hello; a probe abandoned or with no estimate; a servable job; an item claimed by a worker
  that has run no queue). All but the unresolved unit's reach a worker through the one release gate above, and an item
  of one that fails its load writes the `unfit` record, so the exposure of an unprobed unit is its first load, once per
  grant. A servable job loads no model.
- **The residual, stated plainly.** A unit that is unresolved, and a unit released on a late hello, has no digest, so no
  record can be keyed and the release gate cannot judge it. It is released unjudged; its load OOM is held as a load
  fault of the environment (`execution.md` §3.4, T24-3) and is visible there, and no `unfit` record is written. This
  is an exception, not part of the rule, accepted by the controller on 2026-11-12 per run-without-stalls, pending the
  owner's confirmation; if refused, a follow-up gates the late-hello path.


## 8. Worker changes

- `PoolWorker._next`: a probe queue whose serve lease this worker holds, or can take, comes first; its items are
  claimed one per claim. Production claims are unchanged: one item, or one-job items gathered up to the bound; their
  per-request walls are reported individually, so gathering no longer matters to any measure.
- `_Spans` records per-request walls (`time.monotonic` around each `send` inside `bisect`), not group spans shared by
  members; device figures stay per request (`device_requests.collected`), never copied over a group.
- `_commit` reports each request's `Observation` with its `outcome` and co-tenancy; a request that `bisect` split is
  `bisected` and its halves are their own requests.
- `ModelCache.get_or_load` returns whether it loaded and keeps a per-entry request count; the worker process reports
  both on `done` (`since_load`, `load_s`), and the in-process host worker reads them directly.
- `worker_id()` is `[SLURM_JOB_ID-]<host_token[:8]>-<pid>-<nonce>`, in or out of Slurm: the job id (when set), an opaque
  host token, the process id and a per-process nonce (`secrets.token_hex(4)`), so workers of one allocation, and a
  process id reused on one host, name distinct workers. It names no host, account or path.
- A worker reports its `HardwareClass` to the coordinator through its ready mark; the coordinator writes the hardware
  index entry (§3.2). The worker writes nothing to a measures root.
- **The ready mark (amended 2026-11-08, T20R-7a review B-1).** The mark at `<run>/workers/<worker id>` is a versioned
  record, `worker-ready@1.0` (ADR-006). A file without that tag, or of another major, is no readable mark: `read_mark`
  logs once, at warning, the tag it found and the one it reads, and the Pool counts it for the `resolve_s` reason
  (§7.1). Before it writes the mark, the worker builds (or reads, if built) the env of each backend placed on its
  queues. It reads that set from the run's placements (`<run>/private/placements/<attempt>.json`, the latest
  attempt), which the coordinator writes when it binds the run, before the first `Fleet.start` of any class. A backend
  missing from them gives a mark without its entry, and its unit stays pending until `resolve_s`. A failed build is
  recorded in the mark, so the mark is written whether or not every build succeeded. The record:

  ```
  {"format": "worker-ready@1.0",
   "host": <host_token()>, "pid": <int>, "started": <str | null>,
   "queues": {<queue>: {
       "hardware": <HardwareClass> | null,
       "envs": {<env name[-tag]>: {"env_lock", "wheel_set", "driver", "platform", "numerics", "threads"}
                                | {"unbuilt": "build-failed" | "unreadable"}}}}}
  ```

  `host`, `pid` and `started` serve liveness only (`Peers`). Each env entry is exactly that env's `env` fields of §3.1,
  produced by the one filter `MeasureEnv.from_fields` applies (hoisted so both call it): `driver` holds `torch_build`
  only, never the index URL or a server's attestation fields. `threads` is per env: `{}` for a host env, the measured
  launch values otherwise, with the served variables for a served backend. `platform` and `numerics` are the node's
  and agree across a queue's envs. The mark is written before any server starts, so it carries no `engine_version`; the
  coordinator completes a server unit's `driver` and `service` from `service_of`, as on a local substrate. `hardware`
  is null where the worker names no class, and `unbuilt` is a cause code; both are final causes (§3.2). The mark
  carries no identifier (I7, UX-57): no account or host name, uid, path, URL or free-text message, only the host token
  (ADR-007), digests, cause codes and the values above. `test_a_ready_mark_carries_no_identifier` builds its mark from
  an env whose index is a mirror URL with userinfo and a backend serving a `local:` model.
- A worker takes a probe queue's serve lease only when its `HardwareClass` and its env fields for the probe's backend
  equal the resolved unit's (the F2 filter over every field the hello carries), and never on a `closed` probe queue
  (§7.1).

## 9. Ladder and generator on the same Probe

- A ladder point runs the Pool with one measures root in the point's output directory. A suite may carry a table
  profile, a `calibration:` section that overrides `length_strata` and `batched_sizes` only (`Suite.load`
  refuses any other key, a seconds key among them), so a 10³ GPU point measures B up to the memory batch (R-237) over
  three length strata (R-221); `probe_s` stays the profile's derived property. The suite's `point_wall_s` is the
  per-point wall budget: the ladder passes it to the Pool (`Pool(probe_s=)`), which gives it to `Probe.plan` as its
  seconds budget. Suites are `scale-suite@2.0` (the reader takes any `@2.x`; `@1` is retired and refused, naming the dropped `probe_items`).
- **Two kinds of suite.** `suites/table.yaml` (CPU, `signal.resample`, points 10² and 10³) and `suites/gpu-table.yaml`
  (the Phase 2 backends, 10³) carry the table profile (`length_strata` 3, `batched_sizes` 2, `point_wall_s` 900) and exist to be generated from. `suites/ladder.yaml` and `suites/gpu.yaml` carry none and are
  the flatness ladders: under the table profile a 10² point's items are all probe items (1006 probe jobs), so no
  production request is left to judge flatness by.
- A point's result (`scale-result@3.0`, replacing `@2.0`) records the derived `probe_items` the point ran under and, per unit its measures root holds, the unit, the
  probe's observations and its `Estimate`, and nothing else as its measure; the calibration settings it ran under; and
  the per-item figures below. The fallback `or calibrated(samples)` is deleted (must-fix 5): a unit whose probe is
  unmeasured has `estimate.status: unmeasured` with reasons. The result's `measure`, `samples`, `calibration_s`,
  `calibration_jobs`, `began_s` and `calibration_per_item_s` are gone: they were read from item end times.
- Per-item flatness (`budgets.yaml`) is judged on `per_item_s`, the **throughput wall per production job**: the
  production span over the production jobs that gave a result. Both ends of the span are the workers' own clock, from
  the production items' `Sample`s (`ended_s`, taken just before the commit; a claim's start is `ended_s - wall_s -
  claim_s`): the latest `ended_s` less the earliest start. Every ladder suite runs on one allocation, so one host's
  clock covers both ends. The coordinator's events do not enter it: a `finished` event is written when the follower next
  reads, up to `pool.poll_s` after the commit. Production items are those committed on a queue that is not a probe's,
  and the job count is read from their outcome records, so no request identity enters it. A worker start, a model load
  before the first claim, the wait after the last commit, the coordinator's lag in settling and shutdown are outside
  the span, and a probe running beside production neither lies in it nor is subtracted from it.
  That is the basis the ceilings and `flat_ratio` were derived on (wall across all workers per job), with the
  idle time between items in it. `worker_s_per_item`, worker seconds per job (the walls of the `ok`
  production requests that did not load the model over their jobs), is a separate field judged against no ceiling. A
  request spanning several items is in each one's observations, so for this field requests are told apart by
  everything they reported and counted once, and each is timed from its own start, never by an item's wall, which runs
  from its claim's start when a claim holds several items. With no production job `per_item_s` is None and flatness and
  ceiling read n/a. The ceiling is judged at the largest flatness point the entry ran, always on `per_item_s`.
- `ladder.py run SUITE --execution PROFILE [--measures-root PATH]`. Without `--measures-root` each point probes into a
  measures root of its own, so each unit is probed in the run and no point reads another's (the flatness ladders).
  With it every point's pool probes into and reads that one root, so a unit already measured there is not probed
  again; this is how a group calibrates its own resource (§3.7). A point's result `units` lists only the units its plan
  resolved, each with `probed` (whether this run probed it) and the `probe_ids` this run wrote, empty when its estimate
  was served from the root's earlier observations; its `observations` are those of this run's probes. A campaign whose
  figures are evidence therefore starts from an empty `--measures-root`. `--execution` has no default: a packaged profile name
  or a path to a whole profile file (a partial file is refused; layering is `load(name, override)` or `--override FILE`),
  so no site's profile is assumed. Nothing leaves the store but probe ids and counts: no
  `MeasureRecord.uid` is in `explain`'s text (T20R-2b N-9).

## 10. What a run needs, and a site placement hook (owner, 2026-11-08)

The owner's ruling: senselab holds no site scheduling policy. It does not query queues, read group usage, compare
partitions or keep headroom rules. Those belong to a site adapter outside senselab (on ORCD, one built on the
orcd-remote skill's scripts), or to an agent using site tools. senselab supplies what such a party needs, accepts
its answer through one narrow hook, checks the answer against the request, and reports a site's refusal so that the
caller can adjust and resume. It never guesses a fix. (Task 21 Step 3 failed this way: a class `time_s` of 21600 was
submitted as `--time=360` to a partition whose MaxTime is 4 h, and every start was refused.)

### 10.1 The request: `placement-request@1.0`

For each resource class a run will use, `plan.explain()` and `senselab plan --json` emit:

- `class`: the profile's class name;
- `device`, `gpu_type`, `device_memory_bytes` (the batch peak per worker from §6.3, or the class's cap when
  unmeasured), `cpus`, `mem_bytes`;
- `workers`: the count `size` wants (§6.2), and `workers_max`, the class's `max_workers`;
- `time_per_worker_s`: the longest a worker needs: its largest item's upper-predicted seconds plus the load, held
  within the class's `time_s`; None when any node's time is unmeasured, or its request fit is;
- `time_floor_s`: given only when `time_per_worker_s` is None: the largest load plus per-item cost of the class's
  nodes whose load is measured, None otherwise (the lower bound §10.3 checks an answer's time against);
- `wall_s` and `gpu_hours` for the class's work, from §6.6;
- for each figure, `measured` or `unmeasured` with the reason (UX-31);
- the current placement values (`partition`, `time_s`, `max_workers`) from the profile and run overrides, each with the
  layer that set it.

It is data, the same figures `explain()` prints. Producing it starts no computation and downloads no model.

### 10.2 Explicit placement values, layered

A run accepts `placement` overrides per resource class (`partition`, `time_s`, `max_workers`, `exclude`) as a
documented input: a profile override file layered over the packaged profile, which is already possible, now named in
the run API (`sl.plan(..., placement=path_or_mapping)`) and the CLI (`--placement FILE`). The order is profile, then
override file, then the hook's answer (§10.3). The plan records the value it used and which layer set it.

### 10.3 The hook: `placement-answer@1.0`

- **Where it is named.** The profile key `substrate.placement_hook` is a `module:callable` import path, and the run's
  `placement_hook` overrides it. With neither, the explicit and profile values apply unchanged. senselab ships no
  implementation.
- **The call.** Before the substrate submits workers of a class, and once per epoch per class, it calls
  `hook(request: placement-request@1.0) -> placement-answer@1.0`, bounded by `substrate.placement_hook_s`. The answer
  holds, per class, any of `partition`, `time_s`, `max_workers` and `exclude`, plus a free-text `why` (for example the
  limits and estimates the adapter read).
- **Validation, before any submission.** senselab checks the answer against the request and refuses it, with a
  typed `PlacementAnswerRefused` naming the field and both values, when:
  - `time_s` is below `time_per_worker_s` (a worker could not finish its longest item); when the time is unmeasured,
    below the load's measured seconds plus one item's, or, with that unmeasured too, any answer is accepted and the
    event says the check was unmeasured;
  - `max_workers` is below 1, or above the request's `workers_max`;
  - a field names a class the request does not hold.
  An answer may give fewer workers, a shorter time that still holds a worker, or another partition; it never makes a
  worker unable to finish an item, and never asks for more than the run needs.
- **A hook that fails or times out** is reported (`PlacementHookFailed`, chaining its error) and the run falls back
  to the explicit and profile values; the event says so. The fallback is logged, not silent.
- **A hook that cannot be imported** (a malformed or unresolvable `module:callable`) is `PlacementHookFailed` when
  the run starts, not a fallback: a misconfigured hook fails loudly. A callable given as an object that no import
  can resolve (a lambda, a nested function) is recorded with `importable: false`.
- **Recorded.** One `placement` event per class per epoch, a run-scoped kind of `run-event@2.2` in the epoch's event
  log, holds the request, the answer, the callable's path and the version of its distribution
  (`importlib.metadata`), the validation verdict, the values used with the layer of each, and the `--exclude` list
  submitted. The request is built per epoch from the input values the pool bound. Its request
  values (partition, account) and `why` render under the reader's audience rule, like any error text (Terms, UX-45).
- **Actuals beside the estimate.** At the epoch's end the event gains a closing record per class: the actual wall
  time, GPU hours (from worker lifetimes) and peak device memory (from the epoch's `Observation`s) of the class's work, beside the request's
  figures. The run report reads both (UX-31 guarantee 3, UX-47).

  Amended 2026-11-09 (T20R-7b). The log is append-only, so the closing record is the class's `placement` event written
  again at the epoch's end (a change of epoch, or `Pool.close`), its `PlacementRecord` carrying `closing:
  {wall_s, gpu_hours, device_peak_bytes}` (`run-event@2.2`, an additive minor). `request` is the request the class
  was placed against: the hook's, else the plan's request built when the class was first placed. `wall_s` runs from
  the class's first put to its last outcome on the pool's clock; `gpu_hours` is the lifetimes of its workers (a
  worker lives from the first listing that shows it started to the first that omits it) on a `cuda` class, None on
  another; `device_peak_bytes` is the largest `device_peak` of any `Observation` of the class, None where none was
  read. `senselab status` prints the closing record on its own line. The epoch's close writes it for every executor:
  the `Pool` with its fleet's lifetimes, and an executor with no workers (`InProcess`) at the end of the run with the
  run's wall time and no GPU hours or peak. A request reused from another epoch is not part of this epoch's actuals.

### 10.4 A site's refusal is actionable

When the scheduler refuses a submission (Slurm's `sbatch` exit with "Requested time limit is invalid", "Invalid
account or account/partition combination", a QOS refusal), the substrate raises `SiteRefused` carrying the class, the
exact request (the `sbatch` line), the site's message, and the placement layer that set each refused value; the
request line and the message name accounts and partitions, so they render under the reader's audience rule like any
error text. The queue
is held as now (`BrokenClass`), and the run's status names the refusal and the inputs that can change it (`--placement`,
the hook). senselab does not rewrite the request. A resumed run with corrected placement continues from the store.

### 10.5 What senselab does not do

It does not call `sinfo`, `squeue`, `sacctmgr`, `scontrol` or `sbatch --test-only` to choose anything. It keeps no
headroom rule for other group members, and does no partition ranking. All of these belong to the hook, because their
best source differs by site: on ORCD, `--test-only` estimates ignore backfill and were weeks off while jobs started in
minutes, so an adapter there uses live idle resources and observed start latency.

## 11. Configuration (no fixed parameters)

A new profile section `calibration`; `sizing.calibration_items` and `sizing.calibration_workers` are removed by
T20R-7a′. Every key gets a section in `config-derivations.md` (T20R-1). Values marked *provisional* are replaced by
T20R-10's measurement under the stated rule (ruling O-6); `substrate.placement_hook_s` (§10.3, T20R-12) is provisional
too, until a real adapter is timed.

| key | `slurm` | `laptop` | derivation rule |
|---|---|---|---|
| `measures_roots` | `<cache root>/2x/measures` | `<cache root>/2x/measures` | §3.3: the own root; a group's site profile (a user file) lists its shared root first, written `${VAR}` with no fallback, so the group reuses and the own root is the write fallback for a read-only member |
| `trust` | `own` | `own` | §3.5; ruling O-3; overridable per run by the run config's `calibration.trust` |
| `warmup_requests` | 1 | 1 | the floor: the first request after a load is cold; `Estimate.warmup_unsettled` is True, False or None (unknown) from the first three warm requests (`since_load` from `warmup_requests`) slower than the baseline `median(later walls) + u·(units − median(later units))` within a probe's planned length stratum (the middle stratum's singles are planned first), beyond the single fit's one-sided spread band `confidence × spread`; `explain` prints it when True |
| `length_strata` | 2 | 2 | two points fit a line in length (must-fix 8, R-221); the table profile uses 3 to test linearity |
| `min_requests` | 3 | 3 | the least count whose median tolerates one outlier |
| `single_item_jobs` | 2 | 2 | the schema minimum: `a` is measured per whole probe item (`Estimate.item_s`), so two jobs separate it from the per-request cost |
| `batched_sizes` | 1 | 1 | run time measures `B₁` only; the table profile measures 2 (R-237) |
| `probe_items` | derived | derived | not a key: `warmup_requests + length_strata·min_requests·(1 + J) + min_requests·single_item_jobs`, J the jobs of one repetition of the batched sizes at B₁ = `pool.quantum_items` (B₁ plus round(√B₁) at `batched_sizes` 2), computed by `probe_items(config, quantum_items)`; the batch is measured per resource from the unit's device fit |
| `confidence` | 1.645 | 1.645 | one-sided 5 %: the false-batch rate of the batch choice (§5.4 step 6) and the share of items cut past `quantum_s` (§6.2); a wrong batch choice errs to batch 1 |
| `drift_confidence` | 2.576 | 2.576 | two-sided 1 %: at most one spurious confirming probe per hundred windows of a stable unit |
| `observations_kept` | 3 | 3 | superseded probes kept for audit: three give a median of medians tolerating one bad probe |
| `unmeasured_workers` | 1 | 1 | S5: unmeasured work is not scaled; one worker runs it under the safe values |
| `resolve_s` | derived | derived | not a key: `ExecutionProfile.resolve_s` = 2 × (`lease.stale_after_s` + the build lock timeout `locks.TIMEOUT_S`) = 1800 s; §7.1: the longest a live worker takes from its start to its ready mark, dominated by the env builds it does first (§8), two at the build waiter's bound. The 600 s lock timeout was fitted to one resource's slowest build (R-254), so this depends on it until that has a rule; a queue needing more builds goes `Unmeasured`, which costs efficiency only |

Three values are derived properties of the profile, not keys, so they cannot drift and no override restates them:
`probe_s` = `sizing.load_share × sizing.target_wall_s` (0.05 × 1800 = 90 on both profiles: probe work, like loading,
may cost a queue at most a twentieth of its target wall; counts warm probe requests only, §4.5), `peer_wait_s` =
`probe_s + lease.stale_after_s` (a peer's probe ends within its budget or its lease goes stale) and `drift_window` =
`length_strata × min_requests` (drift is judged at the precision of the probe's single phase). `probe_s` and
`peer_wait_s` are properties of `ExecutionProfile`, which holds `sizing` and `lease`; `drift_window` is a property of
the `calibration` settings. `Probe.plan` takes its seconds budget as an argument: the pool passes `probe_s`, the
ladder the suite's per-point wall budget (a suite key), so the table profile overrides `length_strata` 3,
`batched_sizes` 2 only.

## 12. ADR changes

Dated amendment text is in the ADR files, 2026-11-08:

- **ADR-002**: measures are not results. `MeasureUnit` is a separate namespace (`measure-unit@1`) whose env part is
  the env's fingerprinted context fields; it never enters a logical key or a context fingerprint. A probe job is keyed
  like any job; its batch never reaches its key; a batched probe item is formed within one bucket. A probe candidate is
  a job the reuse predicate (`ResultStore.admits`, the one `get` uses) would not serve.
- **ADR-003**: the measures roots, layered and read in order, under the store's path, mode and lease rules:
  write-once files, the hardware index, the per-unit probe lease in the first writable root, the probe queue's serve
  lease as the pin, bounded peer waits, a read-only root that falls back and never fails, one trust filter over every
  record kind, `collect` of the collector's own files only.
- **ADR-004**: the placement carries no plan-time batch; the batch, item cuts and workers are sized at run time by
  `size`; `explain()` calls the same `size` with the stored estimate, never probing, so the plan's cost figures are
  measured or labelled unmeasured.
- **ADR-006**: the format list gains `placement-request@1.0`, `placement-answer@1.0`, `measures@1`,
  `measure-unit@1.0`, `measure-observations@1.1`, `measure-drift@1.0`, `measure-supersede@1.0`, `measure-regime@1.0`,
  `measure-oom@1.1`, `measure-unfit@1.0` and `resource-table@2.0`, `worker-ready@1.0` (the ready mark, §8; amended 2026-11-08 after the
  T20R-7a review), and `run-config@2.1` (additive minor: the `calibration` section, whose
  `trust` a run sets). T20R-1 registers them in `kernel/formats.py`; `resource-table@1.0` is retired by T20R-8, and `resource-table@2.0` by T20R-10b (§3.7).
- **Elsewhere, by the implementing tasks**: `wire-protocol.md` (`done` gains `since_load` and `load_s`, T20R-5);
  `execution.md` §3.2 and §3.6 (T20R-7a, T20R-12); `config-derivations.md` (T20R-1,
  T20R-12); `design.md`'s public API (`placement=`, `placement_hook=`, T20R-12).
- **OPEN (2026-11-08, T20R-7a design review; not done in T20R-7a).** `ExecutionContext.driver`, and so the shared
  store's `context.json` and the `context@4` fingerprint, still carry the torch index URL (ruling P2-42). The rule "a
  URL is a location, never a record value; records carry `torch_build`" awaits an owner ruling on a `context@5`
  identity amendment. The measures (`MeasureEnv.from_fields`) and the ready mark (§8) follow the rule now; the env
  record keeps the URL.

## 13. Rulings (owner, 2026-11-08)

- **O-1, code and env in the identity.** Backend version plus the env, with drift as the staleness rule for a cost
  change that moves neither. Made precise by the review (I1): the env part is the env's fingerprinted context fields,
  `env_lock`, `wheel_set`, `driver`, `platform`, `decoder`, `numerics` (§3.1).
- **O-2, host backends.** The whole host env, accepted: any host dependency change re-probes every host backend, and a
  packaged host row matches only the env that generated it.
- **O-3, trust.** `group` by default on a cluster, with a per-user opt-out: the run config's `calibration.trust: own`
  (§3.5). The filter covers every record kind (I5).
- **O-4, workers per card.** Bounded by memory (§6.3); lowered only by a measured regime factor, recorded per
  observation as co-tenancy (I2, §7.3).
- **O-5, production timings.** Drift evidence only, never pooled into an estimate (§7.3).
- **Placement (owner, 2026-11-08).** No site scheduling policy in senselab; a site hook outside it answers a
  versioned request, senselab validates the answer and reports refusals (§10).
- **O-6, the three keys (amended 2026-11-09, R-252, R-253).** `warmup_requests` 1 and `single_item_jobs` 2 are
  stated floors and `probe_items` is derived (§11). T20R-10 did not fit them from one cluster's ladder into packaged
  defaults: each is measured per resource at run time by the probe where feasible, else it is a resource-agnostic
  default whose rule is stated, and the campaign's figures are evidence. `substrate.placement_hook_s` (the
  substrate's start backoff's first step) stands until a real adapter is timed.

## 14. Review items, must-fix items and register rows

### 14.1 The design reviews (2026-11-08)

| item | where | task |
|---|---|---|
| B1 servable, not present | §4.2 steps 1-2, §6.6 | T20R-3, T20R-7a |
| B2 read-only shared root | §3.3, §3.5 `collect` | T20R-2b |
| B3 one hardware class | §3.2, §8 | T20R-2a, T20R-2b, T20R-5, T20R-7a |
| I1 env build | §3.1 | T20R-2a |
| I2 regime | §5.3, §5.4 step 1, §7.3, §6.3 | T20R-5, T20R-4, T20R-7b |
| I3 probe seconds | §4.2 step 6, §4.3 `B₁`, §4.4, §4.5 | T20R-4, T20R-7a |
| I4 servable not held | §4.2 step 1, §7.1 | T20R-7a |
| I5 trust for every record | §3.5 | T20R-2b |
| I6 OOM from the model, re-opened stratum | §7.4, §4.2 step 5 | T20R-4, T20R-7b |
| I7 no account or host names out of the store | §3.5, §9 | T20R-2a, T20R-2b, T20R-8 |
| I8 PeerWait to New | §7.1 | T20R-7a |
| I9 upper prediction | §6.2 | T20R-6 |
| I10 UX-31 | §6.6 | T20R-6 |
| I11 task order and size | §15 | all |
| I12 rulings | §13 | — |
| N1 `spread_B` in drift | §7.3 | T20R-7b |
| N2 `mps` chip | §3.2 | T20R-2a |
| N3 contrasting figures in tests | §15 T20R-4 | T20R-4 |
| N4 `probe_items` target | §11 | T20R-10 |
| N5 trust bullet names all kinds | ADR-003 amendment | — |
| final F1 no account or host name in a record; owner uid as trust identity | §3.5, ADR-003 | T20R-2b |
| final F2 one hardware resolution; drift on the resolved class only | §3.2 step 4, §4.4, §7.3 | T20R-7a, T20R-7b |
| final F3 drift at k > 1 writes a regime; at k = 1 confirms with jobs held | §7.1, §7.3, §3.6 | T20R-7b |
| final F4 OOM raise at (B − 1); B = 1 re-placed | §7.4 | T20R-4, T20R-7b |
| final F5 index write by the coordinator | §3.2 step 1, §8, §15 | T20R-5, T20R-7a |
| final F6 T20R-2 and T20R-7a split | §15 | T20R-2a, T20R-2b, T20R-7a, T20R-7a′ |
| final F7 any `OSError` on a root non-fatal | §3.3, ADR-003 | T20R-2b |
| final F8 unknown formats skipped; quarantine own files only | §3.4, §3.5, ADR-003 | T20R-2b |
| final F9 actuals beside the estimate | §10.3 | T20R-7b |
| final F10 refusals and request values rendered per audience | §10.3, §10.4 | T20R-12 |
| final F11 waits on another run stated | §3.4, §6.6 | T20R-6 |
| final N1-N7 | §4.2, §3.1, §3.2, §6.6, §15 T20R-8, header, §11, §13 | — |

Nothing in either review is declined.

### 14.2 The architecture review's eight must-fix items

| must-fix (review §4) | design element | task |
|---|---|---|
| 1. length-scale time and memory, or batch 1 outside the measured range | `Estimate` per input unit and per padded unit (§5.4); `size` over the run's lengths, cuts at the upper prediction, batch 1 above `range` (§6.2, §6.4) | T20R-4, T20R-6 |
| 2. divide device memory by workers per device; bound workers by the batch peak | §6.3 | T20R-6 |
| 3. cold/warm from model-load events; exclude failed and bisected requests | `since_load`, `load_s` (§5.3, §8); §5.4 steps 1-2 | T20R-5, T20R-4 |
| 4. remove or unify `profiles._batch_size` | removed; one `size` for Pool, in-process and `explain` (§6.6) | T20R-6 |
| 5. drop the ladder's fallback | §9 | T20R-7a |
| 6. key measured rows by backend params and env | `MeasureUnit` (§3.1, §3.2), read on exact identity from the measures roots (§3.7) | T20R-2a, T20R-8 |
| 7. fit the per-call fixed cost; a seconds budget | `r`, `a` (§5.4 steps 3-4); `probe_s` over warm probe work (§4.5) | T20R-4, T20R-7a |
| 8. regenerate with recorded n and spread, two batch sizes, two length strata | table profile, `ladder.py --measures-root` (§3.7, §9) | T20R-10 |

### 14.3 Register rows

Closed when their task merges:

| row | closed by |
|---|---|
| R-236 | T20R-4 (`a` and `r` fitted; test on a fixed-cost stand-in) and T20R-10 (CPU row regenerated) |
| R-237 | T20R-10: the 10³ GPU table point measures `B₁` at the memory batch and `B₂` |
| R-229 | T20R-2a (env build and `backend_params` in the identity) and T20R-10 (crisperwhisper at batch 1 from warm unshared requests, beside e801bc6c) |
| R-233 | budget side by T20R-8 (flatness from request walls); the worker poll wait stays open, harmless to any measure |
| R-230 | T20R-4 records "device unmeasured" in the estimate, batch 1; stays open for a non-torch peak |

To open at merge (ids assigned then):

| id | P | item | closed by |
|---|---|---|---|
| N-1 | P2 | cold/warm is per queue name, so a reload, a second backend on a queue or a second model-less unit is counted warm | T20R-5, T20R-7a |
| N-2 | P2 | time and device memory are scaled to the table's median item, not the run's lengths | T20R-4, T20R-6 |
| N-3 | P2 | workers sharing a card each size their batch against the whole card | T20R-6 |
| N-4 | P2 | plan time sizes a second way (`profiles._batch_size`, host pair against a GPU) | T20R-6 |
| N-5 | P2 | failed and bisected requests enter the medians | T20R-4, T20R-7a |
| N-6 | P2 | the ladder ships `calibrated(samples)` when its pool measured nothing | T20R-7a |
| N-7 | P2 | `Unit` omits backend params, the env build and the device; packaged rows are reused across them | T20R-2a, T20R-8 |
| N-8 | P3 | host memory is not per request and not of the process holding the model | T20R-11 (after exit) |
| N-9 | P3 | `_grow` doubles per-item bytes for the whole unit, unmarked and without decay | T20R-7b |
| N-10 | P2 | placement values are fixed per profile and a site refusal is not actionable: Task 21 Step 3 submitted `--time=360` to a 4 h partition and every start was refused | T20R-12 |
| N-11 | P3 | `explain` lost the model download-size figure (it was the `@1.0` rows' `download_bytes`; the probe measures requests, not downloads), so a user no longer sees the bytes a fetch will take before it | T20R-12 (placement request) |

R-224, R-225, R-235 and R-165 follow exit as the review says; R-221's grid edges stay open, with T20R-10's per-stratum
costs as their first evidence.

### 14.4 Scenarios checked

Against `user-scenarios.md` on `2x/phase-3-spec`, groups A-D, F, J, L, M and Terms. Hold: UX-1, UX-2, UX-3, UX-4
(§6.6's over-cap flag), UX-5 to UX-10 (UX-10 guarantee 4 by §3.5's own-files rule), UX-12, UX-14 to UX-16, UX-18 to
UX-21, UX-31 (guarantee 3 by §10.3's actuals), UX-32, UX-35 to UX-39, UX-42 (§6.6 states `peer_wait_s`), UX-43,
UX-45 to UX-47, UX-56 (§3.3), UX-57 (§3.5: no account or host name in a measures root), UX-58, UX-61 to UX-63, and
group L's preamble (§10.3, §10.4). Not designed here: UX-11 and UX-13 (measures roots on object storage), UX-17, UX-40,
UX-41, UX-59 (beyond the attestation in the identity), UX-60, UX-64.

## 15. Implementation tasks

Each is sized for one sonnet implementer (P-40 as amended 2026-11-08: sonnet implements and tests, opus reviews
first, fable reviews before merge). **Each merges on its own with the tree green and `pool.py` working**: new code
lands beside the old, and every deletion belongs to the task that removes its last caller. Order:

```
T20R-1 ──► T20R-2a ──┬─► T20R-2b ──┐
       │             └─► T20R-5 ───┤
       ├─► T20R-3 ─────────────────┤
       └─► T20R-4 ─────────────────┴─► T20R-6 ──► T20R-7a ──► T20R-7a′ ──► T20R-7b ──► T20R-8 ──► T20R-10
T20R-12 ──────────────────────────────────────────────────────────────────►┘
T20R-11 after the Phase 3 exit.
```

T20R-5 follows T20R-2a for the `HardwareClass` normalisation. T20R-12 needs no other task: until T20R-6 merges every
request figure is `unmeasured`, so the `time_per_worker_s` check accepts any answer and the event says so (§10.3). It
merges before T20R-7b, which closes its `placement` event with actuals (§10.3).

Every "fails now" test is written first and shown failing on `5ae1fc3d` in the PR, or, where it is new behaviour,
states the old mechanism's contrasting result in its assertion (P-34). No task runs a full suite locally; GPU and
multi-node checks are submitted ORCD jobs, never run on login or claude-rc nodes.

### T20R-1: configuration and data types (additive)

- **Scope.** The `calibration` profile section (§11), with `probe_s`, `peer_wait_s` and `drift_window` as derived
  properties and no load-time rule checks; one run config reader, validating `thresholds` and `calibration`, so the run
  config's `calibration.trust` override is read (`run-config@2.1`); `Observation`, `Estimate`, `ProbePhase`, `LengthProfile`, `HardwareClass` types (data only); format
  tags registered (§12); `config-derivations.md` sections. `sizing.calibration_*` stay (T20R-7a′ removes them).
- **Files.** `runtime/execution_settings.py`, `runtime/profiles.py`, `runtime/compare.py` (the run config reader), `runtime/call.py` (run config), `runtime/data/profiles/{laptop,slurm}.yaml`,
  `runtime/calibration.py` (types only), `kernel/formats.py`, `config-derivations.md`.
- **Tests failing now.** `test_probe_s_peer_wait_s_and_drift_window_are_derived_from_the_profile`;
  `test_one_run_config_reader_validates_thresholds_and_calibration`; `test_the_group_measures_root_is_the_variable_alone`;
  `test_an_estimate_refuses_two_entries_for_one_key`.
- **ORCD.** None.

### T20R-2a: `MeasureUnit`, `HardwareClass` and `MeasureSource` (additive)

- **Scope.** §3.1, §3.2's normalisation, §3.7: identity with the env fields; `HardwareClass` normalisation (`cuda`,
  `mps`, `cpu`, `cpus` on every class); `MeasureSource` over its sources (its packaged-table reader was deleted by
  T20R-10b, R-252), with the estimator **passed in** as a callable, so it does not wait on T20R-4. `Measures` and `Unit` stay
  (T20R-7a′ removes them).
- **Files.** `runtime/measures.py` (new), `runtime/resources.py` (`@2.0` schema and reader added beside `@1.0`).
- **Interfaces.** `MeasureUnit.of(placement, model, env_fields, hardware) -> MeasureUnit`; `.digest()`;
  `HardwareClass.from_report(report, grant) -> HardwareClass`; `MeasureSource(sources, packaged, estimate=...).get(u)`.
- **Tests failing now** (each states the `5ae1fc3d` contrast): `test_units_differing_only_in_backend_params_have_different_measures`
  (`Measures.get` returns one row for both); `test_units_differing_only_in_wheel_set_have_different_measures`;
  `test_a_cuda_class_carries_its_granted_cpus`; `test_a_packaged_row_is_read_only_on_exact_identity`;
  `test_packaged_rows_carry_no_user_or_host` (I7).
- **ORCD.** None.

### T20R-2b: the layered measures roots (additive)

- **Scope.** §3.2's index, §3.3-§3.6: layered roots with the writability check and the `OSError` fallback; the
  hardware index (`seen_classes`, `note_seen`; called by T20R-7a); write-once `obs`, `drift`, `supersede`, `regime`,
  `oom` files with provenance and no account or host name; probe lease (taken in the write root, looked for in all);
  trust filter by owner uid over every record kind; unknown formats skipped; quarantine of own files only; `collect`
  of own files only; the roots as a `MeasureSource` source.
- **Files.** `runtime/measures.py`.
- **Interfaces.** `MeasureRoots(roots, policy, trust, uid)` with `.observations(u)`, `.records(u)`, `.write_root`,
  `.put_observations(u, probe_id, obs, provenance)`, `.put_record(u, kind, record)`, `.probe_lease(u)`,
  `.seen_classes(declared)`, `.note_seen(declared, seen)`.
- **Tests failing now.** `test_read_only_shared_root_falls_back_to_own` (B2: no write fails, the observation lands in
  the own root, a log line names the fallback); `test_an_oserror_on_a_root_is_logged_and_never_raised` (`ENOSPC` and
  `ESTALE` injected on read and write); `test_lookup_and_write_use_one_hardware_digest` (B3: a class with `gpu_type`
  None and a worker reporting `NVIDIA H100 80GB HBM3` is read under the digest it was written under);
  `test_no_measures_record_carries_an_account_or_host_name`; `test_another_users_drift_record_does_not_act_under_trust_own`
  (I5); `test_collect_never_removes_another_users_file`; `test_two_writers_of_one_unit_both_keep_their_observations`;
  `test_a_misfiled_unit_directory_is_quarantined_only_by_its_owner`; `test_a_file_of_an_unknown_format_is_skipped_not_moved`.
- **ORCD.** One submitted job on three nodes against a shared root: concurrent `put_observations`, two coordinators
  racing `probe_lease` with one SIGKILLed (adapt `scripts/probe_lease_cross_node.py`), and one process with the shared
  root mounted read-only (a bind of it with `ro`) falling back to its own root.

### T20R-3: the servable predicate (B1)

- **Scope.** `ResultStore.admits(key_digest, context, policy, run) -> StoredContext | None`: fingerprint listing,
  equivalence, license and verify-sampling rules, reading `context.json` of non-exact entries only. `job._serve` and
  `store.get` call it, so serving and probing share one predicate (P-37). No Pool change.
- **Files.** `runtime/store.py`, `runtime/execution/job.py`.
- **Tests failing now.** `test_entries_under_another_context_are_probe_candidates` (B1: an entry under an old
  fingerprint, not an equivalence member, is not admitted; the design's earlier directory-existence rule would have
  called it stored); `test_admits_agrees_with_serve` (property: for any store state and context, `admits` is not None
  exactly when `_serve` serves); `test_an_unlicensed_entry_is_not_admitted_when_hits_require_a_license`;
  `test_a_verify_sampled_entry_is_not_admitted`.
- **ORCD.** None.

### T20R-4: `Probe.plan` and `Probe.estimate` (pure, additive)

- **Scope.** §4.1-4.3, §4.5 arithmetic, §5.4, the OOM raise of §7.4. No I/O.
- **Files.** `runtime/calibration.py`.
- **Interfaces.** `Probe.plan(unit, candidates, prior, grant, config, formation, seconds) -> ProbePlan` (`seconds` is the warm-work
  budget: the pool passes `probe_s`, the ladder its suite's per-point budget);
  `Probe.plan_batched(plan, single_estimate, grant, config, seconds_left) -> ProbePlan`;
  `Probe.replace_served(plan, served_item) -> ProbePlan`; `Probe.estimate(observations, records, config) -> Estimate`.
- **Tests (hypothesis).** Cold/reloaded observations never move `r`, `u`, `B*`; failed, bisected, served never enter;
  another regime never enters `r`, `u`; no B above the measured ones; duplicate-key jobs never chosen; covered strata
  never re-planned unless an `oom` record re-opens them; plan within `probe_items`; `B₁` within the seconds left at the
  upper prediction; Theil–Sen tolerates one outlier in three; order invariance. **Failing now, with the contrasting
  figure asserted (N3):** `test_a_failed_request_never_enters_the_per_item_time` (`calibrated` on the same samples
  gives the failed wall in its median); `test_fixed_cost_is_fitted_not_shared` (stand-in `wall = 0.040 + 0.0366·jobs`:
  the estimate gives `u` 0.0366 per job and `a` 0.040, where `calibrated` gives 0.075; R-236);
  `test_device_bytes_scale_with_length` (`measure_of` gives one constant); `test_an_oom_raises_m1_without_a_constant`.
- **ORCD.** None.

### T20R-5: worker observations, load events and co-tenancy (additive)

- **Scope.** §5.3, §8: `ModelCache` load flag and per-entry count; `done` fields `since_load`, `load_s`
  (wire-protocol amendment text); per-request walls in `_Spans`; per-request `Observation` with `outcome`, co-tenancy
  and `HardwareClass` in each outcome **beside** `Sample` (which keeps `cold` until T20R-7a′); the worker's
  `HardwareClass` in its ready mark, with no store write (the index entry is T20R-7a's); probe queue naming, serve lease, one-per-claim and the short wait of §4.4.
- **Files.** `senselab-worker/model_cache.py`, the worker's `done` writer, `runtime/worker_client.py`,
  `runtime/host_worker.py`, `runtime/execution/pool_worker.py`, `runtime/execution/queue.py` (probe queue naming),
  `wire-protocol.md`.
- **Tests failing now.** `test_a_reloaded_model_reports_a_load` (LRU budget forcing an eviction: at `5ae1fc3d` the
  item's sample says warm); `test_a_second_backend_on_one_queue_reports_its_load`;
  `test_a_second_model_less_unit_on_cpu_reports_a_load`; `test_gathered_items_report_their_own_request_walls` (at
  `5ae1fc3d` every member gets the group mean); `test_a_bisected_request_is_reported_bisected`;
  `test_observations_carry_device_cotenancy`; `test_probe_items_are_claimed_one_per_claim_by_the_serve_lease_holder`.
- **ORCD.** One H100 job: speechbrain and crisperwhisper under a model-cache budget that evicts between them; check
  `since_load` resets on reload and crisperwhisper's first request (CUDA initialised inside it, R-230) reports
  `since_load` 0; two workers on one card report device co-tenancy 2.

### T20R-6: `size` over estimates and lengths; one sizing site

- **Scope.** §6: the new `size`; `split` by upper-predicted seconds; batch per bucket; device memory per card; worker
  bound from the batch peak; regime factors; safe values; `profiles._batch_size` removed (its last caller);
  in-process executor and `explain()` on the new `size` with `MeasureSource` (UX-31 figures, labelled). The old
  function is renamed `size_row` and kept for the Pool's existing calls until T20R-7a.
- **Files.** `runtime/sizing.py`, `runtime/profiles.py`, `runtime/execution/in_process.py`, `runtime/planner.py`
  (`explain`), `runtime/execution/pool.py` (the rename at its call sites only).
- **Interfaces.** `size(estimate, grant, config, work: LengthProfile) -> Sizing`; `Sizing.cuts`, `Sizing.batch_at(L)`,
  `Sizing.workers_per_device`, `Sizing.measured`.
- **Tests failing now.** `test_ten_minute_recordings_keep_items_within_quantum_s` (at `5ae1fc3d` the quantum is sized
  at the row's 6.1 s and overruns about 100×); `test_items_are_cut_at_the_upper_prediction`;
  `test_workers_sharing_a_card_fit_it_together` (four workers × 0.9 of the cap at `5ae1fc3d`);
  `test_workers_per_device_bounded_by_the_batch_peak`; `test_plan_time_never_sizes_a_gpu_batch_against_host_memory`;
  `test_batch_is_one_above_the_measured_length_range`; `test_in_process_and_pool_size_one_unit_alike`;
  `test_explain_labels_each_figure_measured_or_unmeasured_and_never_probes` (UX-31);
  `test_explain_states_the_waits_on_another_run`; `test_explain_flags_a_job_whose_batch_one_peak_exceeds_the_cap`.
- **ORCD.** None.

### T20R-7a: the Pool's probe state machine

- **Scope.** §7.1, §7.2: `Resolving` (hardware index, else the first hello's ready mark, and the coordinator's
  `note_seen` write of the index entry), `Lookup`, probe lease and `PeerWait` → `New`, `Probing` with the probe queue,
  `replace_served`, budgets over warm probe work, servable jobs put at once (T20R-3's predicate), observation writes,
  release under §6 or the safe values; `workers_wanted` for probing units; the ladder's two reads (`pool.calibrated`
  becomes `pool.probes`; the `or calibrated(samples)` fallback is deleted, must-fix 5). The old calibration path is
  left with no caller, for T20R-7a′. Also here: every unit the Pool reads or writes is built through `UnitResolver.unit`, the Pool passes `bucket_edges` from the
  backend's formation whether or not it has an estimate, and holds one `ExtrapolationLog` per run; a probe of a stratum the estimate
  does not cover before production on a unit with `u` unmeasured, and the Pool half of
  `test_the_pool_and_the_in_process_executor_size_one_unit_alike`. `execution.md` §3.2 updated.
- **Files.** `runtime/execution/pool.py`, `benchmarks/scale/ladder.py` (the two reads only), `execution.md`.
- **Tests failing now.** `test_probe_results_are_yielded_and_never_rerun`; `test_servable_jobs_are_put_at_once`
  (at `5ae1fc3d` they wait behind the calibration round, I4); `test_entries_under_another_context_are_probed_not_skipped`
  (B1 at Pool level); `test_a_served_probe_request_is_replaced_from_its_stratum`; `test_held_work_waits_at_most_probe_s_of_warm_work`
  (a stand-in with a long load is still measured; at `5ae1fc3d` round 2 is bounded by items only);
  `test_a_second_run_reuses_the_first_runs_measure_without_probing`;
  `test_a_peer_wait_returns_to_planning_for_uncovered_strata` (I8); `test_a_point_whose_probe_measured_nothing_ships_no_measure`.
- **Amended 2026-11-08 (review B-1, B-2, I-2, I-3).** Scope adds pending and final causes (§3.2), the
  `worker-ready@1.0` mark with the worker's env builds before it (§8), the held-part checks, `calibration.resolve_s`,
  the `closed` probe queue and the disagreeing hello (§7.1). Files add `runtime/units.py`,
  `runtime/execution/substrates.py`, `runtime/execution/pool_worker.py`, `runtime/execution/queue.py`,
  `kernel/formats.py`, the profiles and `config-derivations.md`. Tests failing now:
  `test_a_slurm_unit_resolves_at_the_first_hello_and_probes`; `test_a_worker_builds_an_unbuilt_env_before_its_mark_and_the_unit_probes`;
  `test_a_final_cause_takes_the_safe_values`; `test_a_broken_class_with_no_hello_fails_its_held_parts`;
  `test_resolving_past_resolve_s_runs_unmeasured_with_its_reason`; `test_a_closed_probe_queue_is_served_by_no_one`;
  `test_the_stale_window_runs_from_the_last_claim`; `test_a_disagreeing_hello_resolves_a_probing_unit_again`;
  `test_a_ready_mark_carries_no_identifier`.
- **ORCD.** Two consecutive H100 runs of one 10² suite against one shared root, the second as another group member:
  the first probes each unit once; the second plans no probe item (owner requirement 2) and its results match by key;
  a third run after an env rebuild (new `wheel_set`) probes, and its old-context entries are computed, not skipped.

### T20R-7a′: removal of the old calibration path (mechanical, same day as T20R-7a)

- **Scope.** Deletes what T20R-7a left without a caller: `_calibrating`, `_first_round`, `_round_measure`,
  `_batch_round`, `_round_digests`, `_note_unmeasured_device`, `calibrated`, `_batched`, `Measures`, `code_version`,
  `Unit`, `measure_of`, `size_row`, `RowSizing`, `Grant.worker_bound`, `_held`, `Sample.cold`, `PoolWorker._warm`, `sizing.calibration_*` and their
  `config-derivations.md` sections. No behaviour change. Sonnet at low effort.
- **Files.** `runtime/execution/pool.py`, `runtime/execution/pool_worker.py`, `runtime/sizing.py`,
  `runtime/resources.py`, profiles, `config-derivations.md`.
- **Tests.** None new; the suites of the touched modules pass, and a grep finds none of the names under `packages/`.
- **ORCD.** None.

### T20R-7b: production telemetry, drift, regimes and OOM records

- **Scope.** §7.3, §7.4: ρ per regime with `spread_B`, same-hardware requests only; `drift` records; `regime` records
  from a k > 1 window; `Confirming` with the unit's jobs held, and `supersede` records; immediate `m1` raise on a device
  peak above band; `oom` records replacing `_grow` (removed by 7a) with the re-opened stratum, and a B = 1 fault routed
  to re-placement; the `placement` event's closing record of actuals (§10.3).
- **Files.** `runtime/execution/pool.py`, `runtime/execution/events.py`.
- **Tests failing now.** `test_drift_confirms_before_superseding`; `test_drift_under_cotenancy_records_a_regime_without_a_probe`
  (I2: no probe loop under co-tenancy); `test_a_confirming_probe_holds_the_units_jobs`;
  `test_requests_of_another_hardware_class_feed_no_drift`; `test_drift_in_one_run_does_not_supersede_another_runs_regime`;
  `test_an_oom_caps_only_the_lengths_at_or_above_it` (at `5ae1fc3d` `_grow` halves every length, permanently);
  `test_an_oom_reopens_its_stratum_and_the_next_probe_lifts_it` (I6); `test_an_oom_at_batch_one_is_replaced_not_raised`;
  property `test_after_an_oom_size_never_returns_its_batch_at_or_above_its_length` (after an `oom` record at (B, L),
  `size` returns batch < B at every L′ ≥ L); `test_the_placement_event_closes_with_actuals_beside_the_request`.
- **ORCD.** One H100 run with two workers per card on a Local substrate (an interactive allocation of one node): a
  regime record is written and no confirming probe runs.

### T20R-8: ladder, generator and `resource-table@2.0` (the table and generator amended by T20R-10b)

- **Scope.** §9: table profile in the suite; point results carry observations and `Estimate`; flatness from request
  walls; the table profile overrides no seconds key, the ladder passing its suite's per-point wall budget to
  `Probe.plan`; generator from measures roots, refusing unmeasured, writing no user or host; `@1.0` loader and its seven rows
  deleted (their readers after T20R-7a′ are `plan_cost`, `check_caps`, `estimate_of` and the generator; this task routes the
  first two through `Sizing` or deletes them, deletes `estimate_of`, and rewrites the generator); the `config-derivations.md` resources section.
- **Files.** `benchmarks/scale/ladder.py`, `benchmarks/scale/suites/ladder.yaml`, `scripts/generate_resources.py`,
  `runtime/resources.py`, `packages/senselab/src/senselab/data/resources/` (emptied), `config-derivations.md`.
- **Tests failing now.** `test_generator_refuses_an_unmeasured_estimate`; `test_a_packaged_row_of_another_env_build_is_not_read`;
  `test_flatness_reads_request_walls_not_end_times`; `test_generated_rows_carry_probe_ids_not_users`.
- **ORCD.** A CPU ladder smoke at 10² with the table profile, to check the result shape before T20R-10.
- **Amended 2026-11-09 (T20R-10b).** `resource-table@2.0`, `PackagedTable`, `Packaged.measures`, `runtime/resources.py`,
  `EstimateSource.commit`, `data/resources/` and `scripts/generate_resources.py` are deleted by the owner's ruling (§3.7); the
  ladder, `scale-result@3.0` and the flatness checks stay, and `ladder.py --measures-root` is how a group calibrates its resource.
- **Delivered (2026-11-09).** `plan_cost`, `check_caps`, `MemoryCapExceeded`, `Cost` and `estimate_of` are deleted, with
  `Unit`, `write_row`, `Measures`, `code_version`, `Catalogue.resource_tables` and the unused `sizing.Measure`;
  `placement@3.0` drops `cost`; `Packaged.measures` is the packaged table and `default_measures` its one reader;
  `scale-result@2.0`; the carried items: T20R-6 R-1 (the `sizing.relative_tolerance` test slack), R-2 and R-3 (moot with
  `estimate_of`), R-4 (`HOUR_S` is public), N-7 (rows read on exact identity); T20R-2a N-3 (`MeasureRow` validates its
  derivation); T20R-2b N-9 (above); T20R-7c R-1 to R-4. A run directory's placement of a retired major (`placement@2`) raises
  `PlacementFormatRetired`, naming the run as made by an older build, from resume, the pool and `senselab explain`.

### T20R-10: ORCD measurement and the group measures root

- **Scope.** The table ladder at one pinned SHA (a fresh clone per campaign): GPU suite on H100 at 10³ (six units),
  CPU suite at 10⁴ (signal.resample), run with `--measures-root` into a group measures root that starts empty, so every unit is probed in the campaign and
  its result says so (§3.7, §9); no table is
  regenerated or shipped, and nothing it measures becomes a packaged default. `warmup_requests`, `single_item_jobs`
  and `probe_items`' target batch are not fitted from this one resource's ladder: each is measured per resource at
  run time by the probe where that is feasible, else it is a resource-agnostic default whose rule is stated in
  `config-derivations.md`. The campaign's figures (the post-load decay of unshared warm requests, the fitted `a`, the
  measured memory batches, N4) are evidence for those rules and for each group's own check. Update
  `config-derivations.md`; close R-236, R-237, R-229; add R-221's per-stratum evidence.
- **Measurements per unit.** n and spread per phase and stratum; `r`, `u`, `a`, `load_s`; `B₁`, `B₂` and their
  per-job times at each stratum's median; `B*`; device `resident`, `m0`, `m1`; the post-load wall of requests 0..10 at
  one stratum; crisperwhisper at batch 1 beside e801bc6c (R-229); the probe's warm work against `probe_s`.
- **Tests.** None new; the group root, read through `MeasureSource` on identity match, holds a `measured` estimate
  for each of the seven units.
- **ORCD.** As scoped; typed H100s, submitted by the agent and watched by the controller.

### T20R-11 (after the Phase 3 exit): host memory per request

- **Scope.** The model-holding process resets its peak (`/proc/self/clear_refs` writing 5 on Linux, children
  included) before each request and reports `VmHWM` and `VmRSS` after it on `done`; `Estimate.host` fitted like
  `device`; `budgets.yaml` `peak_rss_bytes` read from it for isolated backends (N-8; informs R-224, R-235).
- **Tests failing now.** `test_an_isolated_backends_host_peak_is_its_model_process`.
- **ORCD.** One CPU job over an isolated-venv backend at 10² and 10⁴.

### T20R-12: placement request, explicit values and the site hook (independent)

- **Scope.** §10: `placement-request@1.0` from `explain()` and `senselab plan --json`; the `placement` override input in
  the run API and CLI with its layer recorded; `substrate.placement_hook` and the run's `placement_hook`; the call per
  class per epoch, bounded by `substrate.placement_hook_s` (derivation: the substrate's start backoff's first step,
  so a slow hook costs no more than one failed start; provisional until a real adapter is timed); answer validation;
  `PlacementAnswerRefused`, `PlacementHookFailed`, `SiteRefused`; the `placement` event; formats registered;
  `execution.md` §3.6 and `design.md`'s API. No scheduler query anywhere in senselab.
- **Files.** `runtime/execution/substrates.py`, `runtime/execution_settings.py`, `runtime/planner.py`,
  `runtime/execution/events.py`, `kernel/formats.py`, the CLI module, `config-derivations.md`, `execution.md`,
  `design.md`.
- **Interfaces.** `PlacementRequest`, `PlacementAnswer` (frozen pydantic, versioned); `PlacementHook =
  Callable[[PlacementRequest], PlacementAnswer]`; `resolve_placement(profile, overrides, hook) -> ResolvedPlacement`.
- **Tests (a fake callable and a stub `sbatch` on `PATH`).** `test_a_partitions_maxtime_refusal_names_the_request_and_layer`
  (the stub replies "Requested time limit is invalid" to `--time=360`: at `5ae1fc3d` the queue is held after
  repeated starts with no named cause); `test_the_hooks_answer_is_used_and_recorded`;
  `test_an_answer_below_time_per_worker_is_refused`; `test_an_answer_above_the_runs_workers_is_refused`;
  `test_a_failing_hook_falls_back_and_says_so`; `test_a_site_refusal_renders_under_the_audience_rule`; `test_explicit_placement_overrides_the_profile_and_is_recorded`;
  `test_the_request_is_produced_without_computing` (UX-31); `test_senselab_runs_no_scheduler_query_to_place`
  (the stub records every call; only `sbatch` submissions appear).
- **ORCD.** None required; the ORCD adapter is outside senselab. A smoke run with a toy hook that returns the
  owner's chosen partition and a 4 h time, submitted from a login-side coordinator, checks the event record.
