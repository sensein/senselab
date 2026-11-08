# Task 20R: execution calibration around a dedicated Probe

Status: draft for owner review, 2026-11-08. Replaces the calibration rounds of Phase 3 Tasks 20e to 20k
(`Pool._calibrate`, `_batch_round`, `_round_measure`, `sizing.calibrated`). Keeps `size`, `Grant` and `Sizing` as
the one sizing site, reshaped as §6 says.

Baseline: the Task 20 architecture review at `438cb624` (`.superpowers/sdd/phase-3/t20-architecture-review.md` in the
implementation worktree), its measurement contract M1..M10, its sizing contract S1..S7, its §3 restructuring and its
§4 eight must-fix items. Code paths below are at `origin/2x/phase-3` `5ae1fc3d`. Amendments: ADR-002, ADR-003 and
ADR-004 of 2026-11-08 (§11). Principles: P-17, P-33, P-34 (as amended 2026-11-07), P-37, P-40 (as amended 2026-11-08).

## 1. What the owner asked

1. "probe should be designed around caching while optimizing so that cached results can be reused without running."
   A probe's items are real items of the run. Their outputs are ordinary keyed results in the content-addressed store
   and the run uses them: never computed twice, never discarded. A job whose result is already in the store is not
   run to probe; the probe picks among uncached jobs, or uses timings already measured where they exist.
2. "probe results on an hpc should be reusable themselves." A probe's measurement is kept under the full identity
   of what it measured, in a measures store on the cluster's shared storage, so a later run, or another user on the
   same hardware class, reuses it without probing.

Both follow from one rule: **a measurement is an artefact with an identity, like a result.** Results are keyed by
what they compute (ADR-002); measures are keyed by what determines their cost (§3). Neither key holds the other.

## 2. Shape

```
                 coordinator (Pool)                                  pool worker (pinned for a probe)
 fragment ──► Unit identity ──► MeasureSource.get ──► Estimate? ──yes──► size(estimate, grant, lengths) ──► items
                                   │ no                                              ▲
                                   ▼                                                 │
                     probe lease (measures store) ──► Probe.plan ──► probe items ──► Observation per request
                                   │ busy                     ▲                      │
                                   ▼                          └── Probe.estimate ◄───┘ (+ stored observations)
                         wait for the peer's estimate (bounded)            │
                                                                           ▼
                                               measures store: observations, drift and OOM records
```

Five parts, each the only place its decision is made (P-37):

| part | module | decides |
|---|---|---|
| `MeasureUnit` and the measures store | `runtime/measures.py` (new) | what a measure is of, where it lives, who may read it |
| `Observation` | `runtime/calibration.py` (new), filled by `pool_worker.py` | what one request took, and whether the process was warm |
| `Probe.plan`, `Probe.estimate` | `runtime/calibration.py` | which jobs probe, in which phases; the one classifier |
| `size` | `runtime/sizing.py` | batch per length, item cuts, workers, workers per device |
| `ProbeState` in the Pool | `runtime/execution/pool.py` | probe vs production items, release, drift |

Removed outright (pre-alpha, no shims): `Sample.cold`, `calibrated`, `_batched`, `measure_of`, `Measures`,
`code_version`, `Pool._calibrating`, `_first_round`, `_round_measure`, `_batch_round`, `_round_digests`,
`_note_unmeasured_device`, `_grow`, `PoolWorker._warm`, `profiles._batch_size`, `sizing.calibration_items`,
`sizing.calibration_workers`, `resource-table@1.0` and the seven packaged rows of that format, and the ladder's
`Measures(...).get(unit) or calibrated(samples)` fallback (`ladder.py:333`).

## 3. Measure identity and the measures store

### 3.1 `MeasureUnit`: what a measure is of

`MeasureUnit` replaces `resources.Unit`. Its digest is `identity_digest("measure-unit@1", fields)` over:

| field | source | why it changes cost |
|---|---|---|
| `task` | the node's task | different work |
| `backend`, `backend_version` | `BackendSpec` | different code path; the version also enters the logical key |
| `backend_params` | canonical JSON of the placement's `backend_params` (all of them) | engine, dtype, beam width, `max_new_tokens` (R-229: crisperwhisper `engine`) |
| `formation` | `mixed`, `single`, or `bucket:<grid id>` (ADR-004, 2026-11-07) | padding cost; whether a batch can be formed |
| `model` | `(id, identity)`, both None without a model | different weights |
| `env_lock` | the env lock digest the declaring package names (`Packaged.env_lock`); for a host backend, the host lock | library versions (R-229: an env rebuild doubled crisperwhisper's per-item time) |
| `service` | the `Attestation` digest for a server backend, else None | server settings (`max_num_seqs`, `gpu_memory_utilization`) |
| `numerics` | the `numerics` the env reports in its context fields | fp16 against fp32 |
| `hardware` | `HardwareClass` (below) | the device and the cores |
| `observation_format` | `measure-observations@1` | an observation of another format cannot be pooled |

`HardwareClass` is what the coordinator can know before a worker starts, checked against what the worker reports:

- `device`: `cpu`, `cuda` or `mps`;
- for `cuda`: the resource class's `gpu_type` and `memory_cap_bytes`; the worker records the device name and total
  memory it sees (`nvidia-smi --query-gpu=name,memory.total`, already read by the hardware probe). An observation
  whose reported device disagrees with the declared one is written under the class the worker saw, never the one
  declared, and the Pool logs the mismatch once per class;
- for `cpu`: the granted `cpus` (the resource class's, else the host's) and the CPU model string;
- never the resource class name, which is profile-local: two profiles naming the same hardware differently share
  measures, and one name on two partitions with different cards does not.

Not in the identity:

- **the input lengths.** A measure is per input unit (§5.2) and records the length range it covers. A run whose
  lengths lie outside it probes only the uncovered stratum (§4.2).
- **the senselab source digest.** `code_version()` over all of senselab-core invalidated every cached row on a
  docstring edit and never invalidated a packaged row (review §2, M10). The backend version and the env lock name the
  code that runs. A cost change with neither changed (a performance fix) is caught by drift (§7.3), which supersedes
  the old observations from measurement. Open question O-1 (§13).
- **the run, the user, the profile name.** These are provenance (§3.4).

### 3.2 Where measures live

A profile key, `calibration.measures_root`, names the root. On `orcd` it is a directory on the group's shared
storage beside the result store, so every user of the group and every node reads one store; on `laptop` it is
`<cache root>/2x/measures`. The ladder points its own root at its output directory (§9), so a table point measures
afresh.

Layout (`measures@1`):

```
<root>/measures@1/
  <u[:2]>/<u>/                 u = MeasureUnit digest
    unit.json                  measure-unit@1.0: the identity fields; first writer wins
    obs/<probe id>.json        measure-observations@1.0: one probe's observations and provenance; written once
    drift/<record id>.json     measure-drift@1.0: production evidence that supersedes earlier observations
    oom/<record id>.json       measure-oom@1.0: a resource fault's batch and length (§7.4)
    probe.lease/               the ADR-003 lease held while a coordinator probes u
  tmp/<uuid>/                  staging
  quarantine/<uuid>            misfiled or malformed files, moved aside
```

There is no estimate file. `Probe.estimate` runs over the unit's observations newer than its newest drift record,
when the coordinator first needs the unit, and the result is held in memory for the run. A stored estimate would be a
second copy that could disagree with its observations; the observation files of one unit are few (§3.5).

### 3.3 Concurrency (ADR-003)

- **Writes are write-once.** Every file is staged in `tmp/<uuid>/` with the store policy's modes and renamed onto a
  name no other writer uses (`<probe id>` and `<record id>` are uuids). Nothing is modified in place, so concurrent
  writers never conflict and no lock guards an append. `unit.json` is first writer wins; a reader recomputes the
  digest from it and requires it to equal the directory name, else the directory's files are quarantined as misfiled
  (the store's binding rule, ADR-003 amendment 2026-10-31).
- **One probe per unit at a time, for efficiency only.** A coordinator that needs to probe `u` takes
  `<u>/probe.lease` (the directory lease of the 2026-11-03 amendment, renewed by `keep_alive`, proven before it
  writes its observations). A coordinator that finds the lease live does not probe: it waits for the holder's
  observations up to `calibration.peer_wait_s`, re-reading `obs/` once per `pool.poll_s`, then sizes from what it
  finds, or under the safe defaults (§6.5) when it finds nothing. A broken lease costs a duplicate probe, never a
  wrong measure: both probes' observations are valid and both are pooled.
- **Walks are by descriptor** below the root, as for the result store; ids that become path components are one
  component.

### 3.4 Trust and provenance of shared rows

Measures size work; they never reach a key, a context fingerprint or an output. The worst a wrong or hostile
observation can do is mis-size: a batch too large is caught by bisection and the OOM record (§7.4); a quantum too
long overruns `quantum_s` and is caught by drift (§7.3). So shared measures need provenance and a reader's choice,
not signatures.

- Each `measure-observations@1.0` file records `user` (the uid's name), `host`, `job` (scheduler job id or pid),
  `run`, `senselab` (version and 40-hex commit when known), the profile's canonical digest and the worker's reported
  `HardwareClass`.
- `calibration.trust` is `group` (read every file in the root) or `own` (read only files whose `user` is the
  current one). `orcd` sets `group`; `laptop` sets `own`. Open question O-3.
- A file that fails its format check is quarantined and logged with its `user`; it never raises in a run.
- `plan.explain()` and the run's provenance name the source of each unit's estimate: the probe ids, users and
  counts it pooled, or the packaged row and its commit.

### 3.5 Invalidation

1. **Identity change.** Any field of §3.1 changes the digest; the old directory is never read for the new unit.
2. **Drift, from measurement.** Production requests that fall outside the estimate's band (§7.3) write a drift
   record; observations older than the newest drift record are not pooled, and the next fragment of the unit probes.
3. **No age rule.** A wall-clock time-to-live would be a fixed parameter with no measurement behind it; drift is the
   measured form of staleness.
4. **Collection.** The store's `collect` (its policy's `collect_interval_s`) removes observation files older than the
   newest drift record of their unit, keeping the newest `calibration.observations_kept` superseded files for
   audit, and staging and quarantine older than the policy's ages.

### 3.6 Packaged tables and cluster measures

`resource-table@2.0` replaces `@1.0`. A row is a `MeasureUnit` (all fields of §3.1) with its `Estimate` (§5.2),
the counts and spreads behind it and its derivation (commit, jobs, probe ids). The generator writes it from
`Probe.estimate` over a ladder point's observations (§9), never by hand.

Lookup order for a unit, the first `measured` estimate wins (`MeasureSource.get`):

1. the measures store (`calibration.measures_root`), identity match;
2. the packaged table, identity match;
3. otherwise the Pool probes.

The packaged row is read only on an exact identity match, so a row is never applied to another env lock, other
backend params or another card (M10, must-fix 6). The cluster's store comes first because it may hold drift records
the package has not seen. A packaged row needs no import into the store: it is a read-only second source with the
same schema.

## 4. The Probe protocol (`Probe.plan`, pure)

### 4.1 Inputs

`Probe.plan(unit, candidates, prior, grant, config, formation) -> ProbePlan`:

- `candidates`: for each job the Pool holds of the unit, its fragment and index, its key digest, its input units
  (§5.1) and `stored`, a hint that the result store holds any entry under that key digest (a directory listing of
  `store@5/<d[:2]>/<d>`; no context is needed to read it);
- `prior`: the unit's observations already pooled (its own store, peers', or a superseded probe's still valid);
- `grant`: the worker's grant (§6.3), for the memory batch;
- `config`: the `calibration` section (§10);
- `formation`: the backend's formation; `single` plans no batched phase.

### 4.2 Choosing jobs (owner requirement 1)

1. Drop `stored` jobs: a stored job would be served by `_serve` in microseconds and its timing would mean nothing.
   They still run as production items and are served from the store.
2. Drop all but one job of each key digest (M4, R-232).
3. Cut the remaining jobs into `calibration.length_strata` strata of equal count by input units (on a bucketed
   formation, strata are cut at bucket edges so a batched item never spans two buckets and its jobs' keys are
   unchanged). Jobs without a length form one stratum.
4. Skip each (phase, stratum) that `prior` already covers with `calibration.min_requests` steady observations: that
   is the stored timing reused instead of a run. A run whose lengths lie inside what `prior` covers plans nothing.
5. Within a stratum take jobs spread evenly by rank of length, never the first ones in fragment order.

If fewer uncached jobs exist than a phase needs, that phase is not planned and its dimension stays unmeasured (§6.5);
the jobs left run as production. If every job is stored, nothing is probed: the unit's items are served, and serving
does not need sizing.

### 4.3 Phases

Each item carries `probe: {id, phase, stratum}` and is one request (§4.4). Phases declare intent; whether an
observation counts is decided by `Probe.estimate` from what the worker reported (§5.3), never from the phase alone.

| phase | items | gives |
|---|---|---|
| `WARMUP` | `calibration.warmup_requests` one-job items from the shortest stratum | `load_s`; warms the process |
| `SINGLE` | per planned stratum, `min_requests` one-job items; and in the middle stratum, `min_requests` items of `calibration.single_item_jobs` jobs, each job its own request | per-request cost `r + u·L`; per-item cost `a`; device bytes per input unit |
| `BATCHED` | per planned stratum, `min_requests` items of one request of `B₁` jobs; with `calibration.batched_sizes` 2, also of `B₂` | per-request cost at batch B; device bytes at B |

`BATCHED` is planned after `SINGLE` is estimated, from its device bytes per input unit:

- `B₁` = min(the memory batch at the stratum's longest job under the probing worker's grant, the largest batch the
  item budget leaves for `length_strata × min_requests` items, `pool.quantum_items`). The fragment's size no longer
  sets it (R-237).
- `B₂` = the integer nearest √B₁, when it is at least 2 and differs from `B₁`: a point between one and `B₁` tells a
  throughput curve that peaks inside from one that rises to the memory bound.
- No `BATCHED` phase when the formation is `single`, the memory batch is below 2, or device memory is unmeasured
  (memory unmeasured means batch 1, S5).

### 4.4 One item per claim, on a pinned worker

- The Pool puts a probe's items on a probe queue, `<queue>~probe-<id>`, under the run's queues. A worker serving the
  queue's class takes the probe queue's serve lease `<queue>~probe-<id>/serve.lease` before it claims from it; a
  worker that finds the lease live skips the probe queue. The lease is the pin: one worker runs the probe, and the
  lease's beats show it alive.
- A worker claims a probe item alone (`max_items=1`) and never gathers it with another (M3; R-234, R-236). The probe
  queue is drained before the worker returns to production claims, quantum yielding aside.
- If the pinned worker dies, its claim and serve leases go stale; another worker takes the serve lease and the
  remaining items. Its first requests report a load (§5.3), so warmness needs no special case.

### 4.5 Budgets (M8)

- `calibration.probe_items` bounds the jobs a probe plans; `B₁` is chosen inside it (§4.3).
- `calibration.probe_s` bounds the seconds from the probe's first claim. Past it the Pool puts no further probe item;
  items already claimed finish, because they are real work whose results are kept. A phase that has not reached its
  minimum counts by then stays unmeasured; a measured phase is kept. Nothing partial is estimated: `SINGLE` complete
  and `BATCHED` incomplete is a measure with batch 1; `SINGLE` incomplete is unmeasured.
- The budgets count only probe items; production items of other units on the same queue are not held.

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
  batched: per B: r_B, u_B, n, spread, range
  batch:   per stratum: the chosen B* (1 when no batch is faster beyond the margin)
  device:  resident, m0, m1 (bytes per padded unit), n | None (unmeasured, never 0)
  host:    resident, m0, m1, n | None
  oom_caps: [(B, L)]                       from oom records (§7.4)
  source: probe ids, users, counts | packaged row and commit
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
  since_load                               requests this model state answered before this one
  load_s                                   on a request that loaded: the load's seconds, else None
  device_peak, device_after                torch peak and allocation after, else None
  host_peak, host_after                    the model process's peak and after, else None (§12, T20R-9)
  outcome: ok | failed | bisected | served
  device_seen: HardwareClass as the worker reads it
```

`since_load` and `load_s` come from the process that holds the model, not from the queue. `ModelCache` counts each
load; the worker reports, on `done` (wire-protocol amendment, T20R-4), how many requests the cache entry answering
the request has answered since it was loaded, and the load's seconds when this request loaded it. An LRU eviction and
reload, a worker-process restart, a second backend on one queue and a second model-less unit on `cpu` all reset it.
A server backend counts per attested server instance on the consumer side (a new server is a new load). This is
`requests_since_load` and `reloaded` of the review's §3 part 2; `Sample.cold` goes.

### 5.4 `Probe.estimate(observations) -> Estimate`: the only classifier

1. **Keep** an observation only if `outcome` is `ok` (M7: failed, bisected and served requests never enter, nor does
   any request of a bisected item) and its `device_seen` matches the unit's hardware class.
2. **Warm** iff `since_load ≥ calibration.warmup_requests` (M2). A request with `load_s` set gives `load_s` and
   nothing else. Cold or reloaded requests never move `r`, `u`, `a` or the batch decision.
3. **Per request, single:** fit `wall_s = r + u·units` over warm batch-1 requests by Theil–Sen (the median of
   pairwise slopes; robust to one outlier per three, needs no tuning constant). `spread` is the median absolute
   residual divided by the median prediction, scaled to a standard deviation (× 1.4826). Fewer than two distinct
   strata with `min_requests` each: `u` is unmeasured and `r` is the median, valid within `range`.
4. **Per item:** `a` = median of (`item_wall_s` − Σ the item's request walls) over warm probe items, both sizes (M3).
   The per-call fixed cost `r` and the per-item cost `a` are never shared out over gathered items, because no probe
   item is gathered.
5. **Per request, batched at B:** fit `wall_s = r_B + u_B·padded_units` the same way.
6. **Batch decision**, per stratum at its median length L: per-job time single `t₁ = r + u·L`; batched
   `t_B = (r_B + u_B·B·L)/B`. B is better than one only when `(t₁ − t_B)/t₁ > z·σ_d`, with
   `σ_d = 1.2533·√(spread₁²/n₁ + spread_B²/n_B)` (the standard error of a difference of medians) and
   `z = calibration.batch_confidence`. `B*` is the best B that passes, else 1. A B never measured is never chosen
   (S2).
7. **Device memory:** over warm requests with device figures, `resident` = min `device_after`; `peak − after`
   fitted as `m0 + m1·padded_units` (Theil–Sen). Memory is per padded input unit, so it scales with length (S3,
   must-fix 1). No device figures: `device` is None, explicitly unmeasured.
8. **Minimum counts (M9):** each estimated quantity needs `min_requests` warm observations in each stratum it uses;
   below that the quantity is None and the reasons name what is missing. `status` is `measured` when `r` is
   measured; everything else may be None and `size` treats each None by S5.

Properties pinned by tests (T20R-3): a cold or reloaded observation never changes `r`, `u` or `B*`; a failed,
bisected or served observation never enters any figure; a batch above every measured B is never chosen; adding
an outlier to three observations moves no median by more than the others' range; the estimate is invariant to the
order of observations.

## 6. Sizing (`size`)

### 6.1 Signature

```
size(estimate: Estimate | None, grant: Grant, config: SizingSettings, work: LengthProfile) -> Sizing
LengthProfile: the input units of each waiting job, in put order
Sizing: batch(L) per length bucket, item cuts (job index ranges), workers, workers_per_device, predicted_s per job
```

`size` stays pure and is the one sizing site (S6).

### 6.2 Time, scaled to the run's lengths (S1, must-fix 1)

- per job at batch 1: `t(L) = r + u·L`; at the chosen batch B for its bucket: `(r_B + u_B·B·L_pad)/B`.
- **Items are cut by predicted seconds**, not by a job count at the table's median: `split` walks the jobs in put
  order and closes an item when its predicted seconds plus `a` would pass `quantum_s` (within the class's `time_s`),
  or when its seconds reach `(load_s + a)/load_share` (the load's share), whichever comes first, and at most
  `pool.quantum_items` jobs. A run of 10-minute recordings gets items of a few jobs; a run of 6 s items gets items
  of many.
- **Workers** = ⌈Σ predicted seconds / `target_wall_s`⌉, at most the item count, `max_workers` and the device bound
  (§6.3).
- Outside the estimate's measured `range`, time is extrapolated linearly for cutting items only, logged once per unit,
  and the batch is 1 (§6.4).

### 6.3 Device memory divided among a card's workers (S3, S4, must-fix 2)

On a substrate whose workers share devices (`Local` with visible GPUs):

1. With `device` unmeasured: one worker per device, batch 1.
2. Otherwise, for the run's longest bucket length `L_max` and its chosen `B*`, the batch peak is
   `P = resident + m0 + m1·B*·L_max`. Workers per device `w_d = max(1, ⌊memory_headroom·memory_cap / P⌋)`.
3. Each worker's grant is `memory_cap / w_d`; every memory batch (§6.4) is computed against that share, so the
   workers of one card together stay within `memory_headroom` of it.
4. The device bound on workers is `devices × w_d`.

On Slurm each worker holds its own card: `w_d = 1` and the grant is the whole `memory_cap`. That more than one worker
on a card gains throughput is not measured by a one-worker probe; `w_d` is bounded by memory only and the gap is a
register row (§12, N-10; open question O-4).

### 6.4 Batch per length bucket (S2, S3)

For each length bucket of the run (the formation's grid for a bucketed backend, else the estimate's strata):
`B(L) = min(B*(L), memory batch at L, the item's jobs)`, where the memory batch at L is
`⌊(memory_headroom·grant − resident − m0)/(m1·L)⌋`. `B(L) = 1` when L lies above the measured range, when device
memory is unmeasured, when batched time is unmeasured, or when an OOM cap covers L (§7.4). The item's placement
carries the batch of its bucket; a fragment whose jobs fall in several buckets is cut at bucket boundaries.

### 6.5 Missing measure: the safe value per dimension (S5)

| missing | value |
|---|---|
| no estimate at all, probe not possible or over budget | batch 1; one worker per device; items of `pool.quantum_items` jobs bounded by `quantum_s` with the worker's own yielding; `calibration.unmeasured_workers` workers |
| device memory | batch 1; one worker per shared device |
| batched time | batch 1 |
| `u` | per-request median, items cut by count within the measured request range |
| grant memory | batch 1 |

### 6.6 One site for plan time and run time (S6, must-fix 4)

`profiles._batch_size` is removed. A placement records `batch_size` 1 at plan time and `explain()` says the batch is
sized at run time from the named estimate source (§3.4). The Pool and the in-process executor both call
`size(MeasureSource.get(unit), grant, config, lengths)` with the same device selection: the device pair on a `cuda`
class, the host pair otherwise, never the host pair against a GPU. The in-process executor never probes; without an
estimate it runs the safe values of §6.5.

## 7. The Pool: probe and production items

### 7.1 `ProbeState`, per unit

```
           get() measured                      lease taken                        estimate measured
 New ───────────────────────► Measured   New ─────────────► Probing ──────────────────────────────► Measured
  │ get() none                               lease busy                  budget out / too few jobs
  └─► (take probe lease) ──────────────────► PeerWait ──► Measured | Unmeasured        └──────────► Unmeasured
                                              (≤ peer_wait_s)
 Measured ── drift record ──► New (the next fragment probes; items already put keep their sizing)
```

- `Probing` holds the unit's fragments (as `_calibration_part` held them), puts `WARMUP` and `SINGLE` items on the
  probe queue, estimates once they all have outcomes, then puts `BATCHED` items when planned, then estimates again.
- At `Measured` or `Unmeasured` the probe's observations are written to the measures store
  (`obs/<probe id>.json`, after `prove()` of the probe lease), the lease is released, and the held jobs are put,
  sized by §6. **Every probe job's result is already in the result store and the Pool yields it like any other**:
  probe items are ordinary items whose outcomes carry `ResultRef`s; only their claim rule and their observations
  differ. Owner requirement 1.
- `Unmeasured` writes its observations too: a later run pools them with its own and may reach the minimum counts
  without repeating them. It never re-probes within the run (no loop).
- A retry of a probe job is a production retry (one-job fragment), never a probe item.

### 7.2 What replaces what

| old | new |
|---|---|
| `_calibrating`, `_first_round`, `_round_digests`, `calibrated` | `ProbeState` per unit, with its probe id and items |
| `_batch_round` | `BATCHED` phase planned by `Probe.plan` from the `SINGLE` estimate |
| `_round_measure`, `calibrated` | `Probe.estimate` |
| `workers_wanted` for an unmeasured queue (`calibration_workers`) | a probing unit wants one worker for its probe queue; other units of the queue are sized by their own estimates |
| `_grow` | `oom` record (§7.4) |
| `Measures` (cache root by code version) | `MeasureSource` over the measures store, then the packaged table |

### 7.3 Drift, from production telemetry

Production items keep a light per-request `Observation` (`probe: None`) in their outcome, for telemetry and drift.
They are never pooled into an estimate: a production request is sized by the estimate it would then confirm, and
folding it in would let one regime's sizing feed its own measure (open question O-5).

- For each warm, `ok`, unserved production request the Pool computes `ρ = wall_s / predicted`, the prediction from
  the estimate at its batch and units.
- Over the last `calibration.drift_window` such requests of the unit, when the median ρ lies outside
  `1 ± calibration.drift_confidence · 1.2533 · spread / √drift_window`, the Pool writes a `drift` record (the
  window's ρ values, the estimate's source, the worker's hardware) and moves the unit to `New`.
- A device peak above the prediction at its padded units beyond the same band writes a drift record too, and caps
  the batch for that length at once (§7.4), because memory errors are not merely slow.

### 7.4 OOM backstop (S7)

A resource fault on an item at batch B and longest length L writes an `oom` record `{B, L, worker, run}`. The
estimate's `oom_caps` hold it, and `size` gives `B(L′) ≤ max(1, B // 2)` for every `L′ ≥ L`. A cap is lifted only by
a later warm `ok` observation at batch ≥ B and length ≥ L, which is measurement, not decay. The bisection of the
failing request, the re-placement on a larger class and the jobs' retries are unchanged.

## 8. Worker changes

- `PoolWorker._next`: a probe queue whose serve lease this worker holds, or can take, comes first; its items are
  claimed one per claim. Production claims are unchanged: one item, or one-job items gathered up to the bound; their
  per-request walls are reported individually, so gathering no longer matters to any measure.
- `_Spans` records per-request walls (`time.monotonic` around each `send` inside `bisect`), not group spans shared by
  members; device figures stay per request (`device_requests.collected`), never copied over a group.
- `_commit` reports each request's `Observation` with its `outcome`; a request that `bisect` split is `bisected` and
  its halves are their own requests.
- `ModelCache.get_or_load` returns whether it loaded and keeps a per-entry request count; the worker process reports
  both on `done` (`since_load`, `load_s`), and the in-process host worker reads them directly.

## 9. Ladder and generator on the same Probe

- A ladder point runs the Pool with its measures root in the point's output directory and `calibration` overridden
  by the suite's table profile (`benchmarks/scale/suites/ladder.yaml`, `calibration:` section): more strata, two batch
  sizes, a larger item budget and seconds budget, so a 10³ GPU point measures B up to the memory batch (R-237) over
  three length strata (R-221).
- A point's result records the probe's observations and its `Estimate`, and nothing else as its measure. The
  fallback `or calibrated(samples)` is deleted (must-fix 5): a point whose probe is unmeasured has `estimate:
  unmeasured` with reasons, and the generator skips it and lists why.
- Per-item flatness (`budgets.yaml`) is read from production requests' walls (`Observation.wall_s` and `item_wall_s`),
  not from item end times across workers, so neither poll waits nor clock skew enter it (R-233, budget side).
- `generate_resources.py` reads measures roots (a ladder run's, or a cluster's store with `--from-measures`), runs
  `Probe.estimate` per unit, and writes `resource-table@2.0` rows with the full `MeasureUnit` and the estimate's
  counts and spreads. It refuses an unmeasured estimate and rows of more than one commit, as now.

## 10. Configuration (no fixed parameters)

A new profile section `calibration`; `sizing.calibration_items` and `sizing.calibration_workers` are removed. Every
key gets a section in `config-derivations.md` (T20R-1). Values marked *provisional* are replaced by T20R-8's
measurement under the stated rule.

| key | `orcd` | `laptop` | derivation rule |
|---|---|---|---|
| `measures_root` | group shared storage beside the result store | `<cache root>/2x/measures` | where every user and node of the hardware class can read; owner requirement 2 |
| `trust` | `group` | `own` | §3.4; O-3 |
| `warmup_requests` | 2 *provisional* | 2 | the least w such that, on the table probe, the median of warm requests w+1..w+3 of one stratum lies within the steady band of requests after them, the maximum over GPU units. Provisional basis: R-230 (CUDA initialised inside the first request) plus one request for shape-dependent kernel selection; R-231's decay was confounded by gathering (R-234), so no measured decay exists yet |
| `length_strata` | 2 | 2 | two points fit a line in length (must-fix 8, R-221); the table profile uses 3 to test linearity |
| `min_requests` | 3 | 3 | the least count whose median tolerates one outlier (the derivation `calibration_items` had) |
| `single_item_jobs` | 4 *provisional* | 4 | items of 1 and k jobs separate the per-item cost `a` from the per-job cost; k such that `a` is at least a fifth of a k-job item's wall at the measured CPU fit (R-236: `a` 0.040 s against 0.0366 s per job) |
| `batched_sizes` | 1 | 1 | run time measures `B₁` only; the table profile measures 2 (R-237) |
| `probe_items` | 200 | 50 | the jobs of the minimal protocol at a target batch B: `warmup_requests + length_strata·min_requests·(1 + B) + min_requests·single_item_jobs`. `orcd` B = 30, above the largest batch any unit has been measured faster at (16, b7c25c4c); `laptop` B = 5, one CPU worker; the table profile sizes it for B = 100 (R-237) |
| `probe_s` | 90 | 90 | `sizing.load_share × sizing.target_wall_s` (0.05 × 1800 on both): probing, like loading, may cost a queue at most a twentieth of its target wall |
| `batch_confidence` | 1.645 | 1.645 | one-sided 5 % false-batch rate; a wrong decision errs to batch 1, never over memory |
| `drift_window` | 6 | 6 | `length_strata × min_requests`: drift is judged at the precision of the probe's single phase |
| `drift_confidence` | 2.576 | 2.576 | two-sided 1 %: at most one spurious re-probe per hundred windows of a stable unit |
| `peer_wait_s` | `probe_s + lease.stale_after_s` | same | a peer's probe ends within its budget or its lease goes stale |
| `observations_kept` | 3 | 3 | superseded probes kept for audit: three give a median of medians tolerating one bad probe |
| `unmeasured_workers` | 1 | 1 | S5: unmeasured work is not scaled; one worker runs it under the safe values |

The ladder's table profile overrides `length_strata` 3, `batched_sizes` 2, `probe_items` (B = 100) and `probe_s`
(the suite's per-point wall budget). Rules that read other keys (`probe_s`, `peer_wait_s`, `drift_window`) are
checked at profile load: a profile whose value disagrees with its rule is refused, so the rule cannot drift from the
value.

## 11. ADR changes

Dated amendment text is in the ADR files, 2026-11-08:

- **ADR-002**: measures are not results. `MeasureUnit` is a separate identity namespace (`measure-unit@1`), never part
  of a logical key or a context fingerprint. A probe job is keyed like any job; its batch size never reaches its key;
  a batched probe item is formed within one bucket, so a bucketed job's key is its own. The probe's `stored` hint reads
  a key digest's directory only.
- **ADR-003**: the measures store, a third kind of root under the same path, mode and lease rules: write-once files by
  staged rename, the per-unit probe lease, the probe queue's serve lease as the pin, bounded peer waits, misfiled
  directories quarantined, `trust` as a reader's filter. Not guaranteed: one probe per unit (a broken lease gives two,
  both valid).
- **ADR-004**: the placement carries no plan-time batch; the batch, item cuts and workers are sized at run time by
  `size` from the estimate, the grant and the run's lengths, the same function for the Pool and the in-process
  executor; the cost estimate the placement owns is that function's output.
- **Elsewhere, by the implementing tasks**: `wire-protocol.md` (`done` gains `since_load` and `load_s`, T20R-4);
  ADR-006's format list (`measures@1`, `measure-unit@1.0`, `measure-observations@1.0`, `measure-drift@1.0`,
  `measure-oom@1.0`, `resource-table@2.0`; `resource-table@1.0` retired, T20R-1); `execution.md` §3.2 (the
  quantum and the `Sample` paragraph, T20R-6); `config-derivations.md` `orcd.sizing` and the new `calibration`
  sections (T20R-1).

## 12. The review's eight must-fix items, and register rows

| must-fix (review §4) | design element | task |
|---|---|---|
| 1. length-scale time and memory, or batch 1 outside the measured range | `Estimate` per input unit and per padded unit (§5.4 steps 3, 5, 7); `size` over the run's `LengthProfile`, items cut by predicted seconds, batch 1 above `range` (§6.2, §6.4) | T20R-3, T20R-5 |
| 2. divide device memory by workers per device; bound workers by the batch peak | §6.3 | T20R-5 |
| 3. cold/warm from model-load events; exclude failed and bisected requests | `since_load`, `load_s` from `ModelCache` (§5.3, §8); `Probe.estimate` steps 1-2 | T20R-4, T20R-3 |
| 4. remove or unify `profiles._batch_size` | removed; one `size` for Pool and in-process (§6.6) | T20R-5 |
| 5. drop the ladder's fallback | §9 | T20R-7 |
| 6. key packaged rows by backend params and env lock | `MeasureUnit` (§3.1), `resource-table@2.0` read on exact identity (§3.6) | T20R-2, T20R-7 |
| 7. fit the per-call fixed cost; a seconds budget | `r`, `a` fitted from unshared requests and items of two sizes (§5.4 steps 3-4); `probe_s` (§4.5) | T20R-3, T20R-6 |
| 8. regenerate with recorded n and spread, two batch sizes, two length strata | ladder table profile, `resource-table@2.0` with counts and spreads (§9) | T20R-8 |

Register rows closed when their task merges (the closing evidence named):

| row | closed by |
|---|---|
| R-236 | T20R-3 (`a` and `r` fitted; test on a fixed-cost stand-in) and T20R-8 (CPU row regenerated) |
| R-237 | T20R-8: the 10³ GPU table point measures `B₁` at the memory batch and `B₂` |
| R-229 | T20R-2 (`env_lock` and `backend_params` in the identity) and T20R-8 (crisperwhisper at batch 1 from warm unshared requests, compared with e801bc6c) |
| R-233 | budget side by T20R-7 (flatness from request walls); the worker-side poll wait stays open, now harmless to any measure |
| R-230 | T20R-3 records "device unmeasured" explicitly in the estimate, batch 1; stays open for a non-torch peak |

Rows to open at merge (ids assigned then; each is review §2's "not in any register"):

| id | P | item | closed by |
|---|---|---|---|
| N-1 | P2 | cold/warm is per queue name, so a reload, a second backend on a queue or a second model-less unit is counted warm | T20R-4 |
| N-2 | P2 | time and device memory are scaled to the table's median item, not the run's lengths | T20R-3, T20R-5 |
| N-3 | P2 | workers sharing a card each size their batch against the whole card | T20R-5 |
| N-4 | P2 | plan time sizes a second way (`profiles._batch_size`, host pair against a GPU) | T20R-5 |
| N-5 | P2 | failed and bisected requests enter the medians | T20R-3 |
| N-6 | P2 | the ladder ships `calibrated(samples)` when its pool measured nothing | T20R-7 |
| N-7 | P2 | `Unit` omits backend params, env lock and device; packaged rows are reused across them | T20R-2, T20R-7 |
| N-8 | P3 | host memory is not per request and not of the process holding the model (`budgets.yaml` `peak_rss_bytes` of isolated backends is the pool worker's) | T20R-9 (after exit) |
| N-9 | P3 | `_grow` doubles per-item bytes for the whole unit, unmarked and without decay | T20R-6 |
| N-10 | P3 | throughput of more than one worker per card is unmeasured; `w_d` is bounded by memory only | open: a two-worker probe on one card, after exit |

R-224, R-225, R-235 and R-165 follow exit as the review says; R-221's grid edges stay open, with T20R-8's per-stratum
costs as their first evidence.

## 13. Open questions for the owner

- **O-1.** Code in the identity: backend version and env lock, with drift as the staleness rule for a cost change
  that moves neither (§3.1). The alternative is a digest of the backend's own source files (narrower than
  `code_version()`, which invalidated everything on any core edit).
- **O-2.** A host backend's `env_lock` is the whole host lock, so any host dependency bump re-probes every host
  backend. Accept, or key on the lock's resolution of the backend's declared dependency closure.
- **O-3.** `trust: group` on `orcd`: every group member's observations size everyone's runs. Alternative: `own`, at
  the cost of one probe per user per unit.
- **O-4.** More than one worker per card is bounded by memory only (§6.3, N-10). Alternative: one worker per card
  until a two-worker probe measures a gain.
- **O-5.** Production requests are drift evidence only (§7.3). Alternative: pool warm production requests as
  observations, so estimates sharpen with use, at the cost of a sizing that measures itself.
- **O-6.** `warmup_requests` 2 and `single_item_jobs` 4 are provisional until T20R-8.

## 14. Implementation tasks

Each is sized for one sonnet implementer (P-40 as amended 2026-11-08: sonnet implements and tests, opus reviews
first, fable reviews before merge). Order: T20R-1 first; T20R-2, T20R-3 and T20R-4 in parallel; then T20R-5, T20R-6,
T20R-7, T20R-8. T20R-9 follows the Phase 3 exit. Every "fails on current code" test is written first and shown
failing on `5ae1fc3d` in the PR (P-34). No task runs a full suite locally; GPU and multi-node checks are submitted
ORCD jobs (never on login or claude-rc nodes).

### T20R-1: configuration and data types

- **Scope.** The `calibration` profile section (§10) with its load-time rule checks; `Observation`, `Estimate`,
  `ProbePhase`, `LengthProfile` types (data only); format tags registered (§11); `sizing.calibration_items` and
  `calibration_workers` removed from schema and profiles; `config-derivations.md` sections for every key.
- **Files.** `runtime/execution_settings.py`, `runtime/data/profiles/{laptop,orcd}.yaml`, `runtime/calibration.py`
  (types only), `kernel/formats.py`, `config-derivations.md`, `scripts/data/invariants.yaml` if a site moves.
- **Interfaces.** `CalibrationSettings` (frozen pydantic); `ExecutionProfile.calibration`.
- **Tests failing now.** `test_profile_refuses_probe_s_not_load_share_of_target_wall`;
  `test_profile_has_no_calibration_items` (the key exists at `5ae1fc3d`); `test_every_calibration_key_has_a_derivation`.
- **ORCD.** None.

### T20R-2: `MeasureUnit` and the measures store

- **Scope.** §3: identity and `HardwareClass`; `measures@1` layout; write-once observation, drift and oom files;
  probe lease; peer wait read; trust filter; quarantine of misfiled and malformed files; `collect`; `MeasureSource`
  (store, then packaged) replacing `Measures` and `resources.Unit`.
- **Files.** `runtime/measures.py` (new), `runtime/resources.py` (`Unit` removed), `runtime/sizing.py` (`Measures`,
  `code_version` removed), callers.
- **Interfaces.** `MeasureUnit.of(placement, model, packaged, resource_class) -> MeasureUnit`; `.digest()`;
  `MeasuresStore(root, policy, trust).observations(u) / put_observations(u, probe_id, obs, provenance) /
  put_drift / put_oom / probe_lease(u)`; `MeasureSource.get(u) -> Estimate | None`.
- **Tests failing now.** `test_units_differing_only_in_backend_params_have_different_measures` (R-229; at
  `5ae1fc3d` `Measures.get` returns the row for both); `test_units_differing_only_in_env_lock_have_different_measures`;
  `test_a_row_written_by_one_core_edit_is_read_after_another` (at `5ae1fc3d` the `code_version` root hides it);
  `test_two_writers_of_one_unit_both_keep_their_observations`; `test_a_misfiled_unit_directory_is_quarantined`;
  `test_trust_own_reads_no_other_users_observations`.
- **ORCD.** One submitted job on three nodes against the shared root: concurrent `put_observations` from each, two
  coordinators racing `probe_lease` with one SIGKILLed (adapt `scripts/probe_lease_cross_node.py`); expected: every
  file readable, one lease holder at a time, the killed holder's lease broken within `stale_after_s`.

### T20R-3: `Probe.plan` and `Probe.estimate`

- **Scope.** §4.1-4.3, §4.5 planning arithmetic, §5.4. Pure functions, no I/O.
- **Files.** `runtime/calibration.py`.
- **Interfaces.** `Probe.plan(unit, candidates, prior, grant, config, formation) -> ProbePlan`;
  `Probe.plan_batched(plan, single_estimate, grant, config) -> ProbePlan`;
  `Probe.estimate(observations, config) -> Estimate`.
- **Tests (property-based, hypothesis).** Cold/reloaded observations never move `r`, `u`, `B*`; failed, bisected,
  served never enter; no B above the measured ones; stored and duplicate-key jobs never chosen; strata covered by
  `prior` never re-planned; plan within `probe_items`; Theil–Sen tolerates one outlier in three. **Failing now**, as
  behaviour of the current classifier they replace: `test_a_failed_request_never_enters_the_per_item_time` (at
  `5ae1fc3d` `calibrated` takes failed samples' walls); `test_fixed_cost_is_fitted_not_shared`
  (a stand-in with `wall = 0.040 + 0.0366·jobs`: `calibrated` gives 0.075, the estimate 0.0366 per job and `a` 0.040;
  R-236); `test_device_bytes_scale_with_length` (`measure_of` gives one constant).
- **ORCD.** None.

### T20R-4: worker observations and load events

- **Scope.** §5.3, §8: `ModelCache` load flag and per-entry count; `done` fields `since_load`, `load_s`
  (wire-protocol amendment text included); per-request walls in `_Spans`; per-request `Observation` with `outcome`;
  probe-queue serve lease and one-per-claim; `Sample.cold` and `_warm` removed.
- **Files.** `senselab-worker/model_cache.py`, worker serve loop and `done` writer, `runtime/worker_client.py`,
  `runtime/host_worker.py`, `runtime/execution/pool_worker.py`, `runtime/execution/queue.py` (probe queue naming
  only), `wire-protocol.md`.
- **Tests failing now.** `test_a_reloaded_model_reports_a_load` (LRU budget forcing an eviction: at `5ae1fc3d` the
  item is labelled warm because the queue was run); `test_a_second_backend_on_one_queue_reports_its_load`;
  `test_a_second_model_less_unit_on_cpu_is_cold`; `test_gathered_items_report_their_own_request_walls` (at `5ae1fc3d`
  every member gets the group mean); `test_a_bisected_request_is_reported_bisected`;
  `test_probe_items_are_claimed_one_per_claim_by_the_serve_lease_holder`.
- **ORCD.** One H100 job: speechbrain and crisperwhisper with a model-cache budget that evicts between them;
  check `since_load` resets on reload and crisperwhisper's first request (CUDA initialised inside it, R-230) reports
  `since_load` 0 and is excluded from every steady figure.

### T20R-5: `size` over estimates and lengths; one sizing site

- **Scope.** §6: new signature; items cut by predicted seconds (`split` takes `Sizing`); batch per bucket; device
  memory divided per card; worker bound from the batch peak; safe values; `measure_of` and `profiles._batch_size`
  removed; the in-process executor calls `size`.
- **Files.** `runtime/sizing.py`, `runtime/execution/pool.py` (`split`, `_grant`, `_sizing` only),
  `runtime/profiles.py`, `runtime/execution/in_process.py`, `runtime/planner.py` (`explain`).
- **Interfaces.** `size(estimate, grant, config, work: LengthProfile) -> Sizing`; `Sizing.cuts`, `Sizing.batch_at(L)`,
  `Sizing.workers_per_device`.
- **Tests failing now.** `test_ten_minute_recordings_keep_items_within_quantum_s` (at `5ae1fc3d` the quantum is sized
  at the row's 6.1 s and overruns by about 100×); `test_workers_sharing_a_card_fit_it_together` (four workers × 0.9 of
  the cap at `5ae1fc3d`); `test_workers_per_device_bounded_by_the_batch_peak`;
  `test_plan_time_never_sizes_a_gpu_batch_against_host_memory`; `test_batch_is_one_above_the_measured_length_range`;
  `test_in_process_and_pool_size_one_unit_alike`.
- **ORCD.** None (pure; the integration is T20R-6's).

### T20R-6: the Pool state machine, drift and OOM records

- **Scope.** §7: `ProbeState`; lookup order; probe lease and peer wait; probe queue puts; budgets (`probe_items`,
  `probe_s`); release under the safe values; observations written; drift from production telemetry; `oom` records
  replacing `_grow`; `workers_wanted` for probing units; removal of `_calibrating` and the rest (§7.2);
  `execution.md` §3.2 updated.
- **Files.** `runtime/execution/pool.py`, `runtime/execution/substrates.py` (scale for probe queues if needed),
  `execution.md`.
- **Tests failing now.** `test_probe_results_are_yielded_and_never_rerun` (every probe job's result yielded once, no
  job executed twice); `test_stored_jobs_are_not_probed` (at `5ae1fc3d` the first k jobs are taken whether stored or
  not, and served ones enter the medians); `test_held_work_waits_at_most_probe_s` (at `5ae1fc3d` round 2 is bounded by
  items only); `test_a_second_run_reuses_the_first_runs_measure_without_probing`;
  `test_a_peer_probing_the_unit_is_waited_for_not_duplicated`; `test_drift_writes_a_record_and_reprobes`;
  `test_an_oom_caps_only_the_lengths_at_or_above_it` (at `5ae1fc3d` `_grow` halves every length, forever).
- **ORCD.** Two consecutive H100 runs of one 10² suite against one shared measures root, the second as a different
  user of the group: the first probes each unit once; the second plans no probe item for any unit
  (owner requirement 2), and its results match by key.

### T20R-7: ladder, generator and `resource-table@2.0`

- **Scope.** §9: table profile in the suite; point results carry observations and `Estimate`; fallback deleted;
  flatness from request walls; generator from measures roots, refusing unmeasured; `resource-table@2.0` schema and
  loader (exact identity); `@1.0` and its seven rows deleted.
- **Files.** `benchmarks/scale/ladder.py`, `benchmarks/scale/suites/ladder.yaml`, `scripts/generate_resources.py`,
  `runtime/resources.py`, `packages/senselab/src/senselab/data/resources/` (emptied), `config-derivations.md`
  (`packages/senselab/.../resources` section).
- **Tests failing now.** `test_a_point_whose_probe_measured_nothing_ships_no_measure` (at `5ae1fc3d` `ladder.py:333`
  ships `calibrated(samples)`); `test_generator_refuses_an_unmeasured_estimate`;
  `test_a_packaged_row_of_another_env_lock_is_not_read`; `test_flatness_reads_request_walls_not_end_times`.
- **ORCD.** A CPU ladder smoke at 10² with the table profile, to check the result shape before T20R-8.

### T20R-8: ORCD measurement and table regeneration

- **Scope.** Run the table ladder at one pinned SHA (a fresh clone per campaign): GPU suite on H100 at 10³ (six
  units), CPU suite at 10⁴ (signal.resample); regenerate the packaged tables; set the provisional keys by their rules
  (`warmup_requests` from the post-load decay of unshared warm requests, `single_item_jobs` from the fitted `a`);
  update `config-derivations.md` with the measured bases; close R-236, R-237, R-229; add R-221's per-stratum evidence.
- **Measurements to record per unit.** n and spread per phase and stratum; `r`, `u`, `a`, `load_s`; `B₁`, `B₂`
  and their per-job times at each stratum's median; `B*`; device `resident`, `m0`, `m1`; the post-load wall of
  requests 0..10 at one stratum (for `warmup_requests`); crisperwhisper at batch 1 beside e801bc6c (R-229); the
  probe's seconds against `probe_s`.
- **Tests.** None new; the regenerated tables load and every row is `measured`.
- **ORCD.** As scoped; jobs on typed H100s, submitted by the agent and watched by the controller.

### T20R-9 (after the Phase 3 exit): host memory per request

- **Scope.** The model-holding process resets its peak (`/proc/self/clear_refs` writing 5 on Linux, its children
  included) before each request and reports `VmHWM` and `VmRSS` after it on `done`; `Estimate.host` fitted like
  `device`; `budgets.yaml` `peak_rss_bytes` read from it for isolated backends (N-8; informs R-224, R-235).
- **Tests failing now.** `test_an_isolated_backends_host_peak_is_its_model_process`.
- **ORCD.** One CPU job over an isolated-venv backend at 10² and 10⁴.
