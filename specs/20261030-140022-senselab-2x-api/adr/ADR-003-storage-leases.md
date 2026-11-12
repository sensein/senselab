# ADR-003: Storage and lease protocol on shared filesystems

**Status:** accepted 2026-10-30; amended 2026-10-31, 2026-11-01.

## Context
Architect review §6 re-opened four items:
- R-3: two coordinators can run at once, and a check-then-act fencing scheme races;
- R-8: starvation, given today's lexicographic claim order and O(N) directory scans;
- R-9: service lifecycle;
- R-12: commits to the packed tier.

None of the recorded resolutions had a mechanism on plain shared files.

## Decision

The protocol uses only primitives that `work_pool` has measured on NFS v4.2: atomic `rename`,
`O_EXCL` hard link, and server-clock timestamps.

- **Safety by idempotence.**
  - Jobs are content-addressed (ADR-002), and results are written to their key by rename, first
    writer wins.
  - A duplicate coordinator, a duplicate claim or a resumed run can only cause **duplicate work,
    never a wrong or mixed result**.
  - The coordinator epoch file (incremented by `O_EXCL` link) is a fence for efficiency only, as to results:
    a stale coordinator sees a newer epoch and stops releasing, and results stay correct without it. Which attempt is
    credited with the work depends on it (amendment 2026-11-08, outcome record epoch).
- **Claim order and fairness.**
  - Each queue keeps append-only, time-ordered index segments, and workers claim oldest-first
    from the head segment. This replaces lexicographic scans; claiming costs O(1) amortised.
  - A worker holding a resident model yields after N items or T seconds (its quantum) whenever
    another queue it can serve has older work.
- **GPU memory admission.**
  - Devices are host-local, so admission is a host-local `flock` token budget per device, not an
    NFS lock.
  - **In 2.0:** static caps per resource class, and an OOM is classified as a resource fault (it
    re-places the job and does not charge the item). The token budget is added when the scale
    ladder shows the need (design §6.4).
- **Services.**
  - A service is a registry entry `{identity, endpoint}` under a renewed lease (amended 2026-11-07).
  - Each consumer holds its own lease on the service. The service exits when no consumer lease has
    been live for `services.idle_exit_s`.
  - The service identity is attested on every response (I-8).
- **Packed tier (deferred, design §6.4).** When added: per-shard append-only segments, written only
  by that shard's lease holder, plus an index committed by rename. Readers trust only indexed
  entries.

**Guaranteed:**
- no wrong, torn or mixed results;
- eventual completion while workers are alive;
- results that are never lost once committed.

**Not guaranteed:**
- exactly-once execution;
- bounded latency when the cluster grants no workers;
- fairness across users of a shared store.

## Consequences
- `work_pool` gains index segments and quantum yielding. Items become plan fragments.
- The packed tier and the token arbiter wait for measurements.

## Alternatives rejected
- **A database or lock service for coordination.** It adds an operated dependency to every site.
- **Strict fencing on NFS.** It cannot be made atomic with check-then-act; idempotence makes it
  unnecessary for correctness.

## Amendment 2026-10-31: paths, kinds and entries under a storage root

Adds to the Decision; replaces nothing in it. Resolves R-100 and R-101
([`review-register.md`](../review-register.md)).

**Why.** The result store, the fetch cache and the run manifests live under shared, group-writable
roots, so anyone in the group can plant a symlink or a file there. The H8c re-attack
([`reviews/phase1-reattack/store.md`](../reviews/phase1-reattack/store.md)) showed what followed: fetch
replaced a file outside the cache through a planted `files/sub -> /victim` (XS-1); a directory at
`blobs/<sha256>` or a file at a store directory wedged its key behind a retryable-looking error (XS-2,
XS-3); a run id `../../escaped` wrote outside `runs/`, and under umask 0 a run manifest was world-writable
(XS-7); and two contexts' entries for one key could be swapped (XS-4). The rules below make those states
unconstructible.

- **Below a root, paths are walked by descriptor.** Every directory below a storage root is reached one
  component at a time with `O_DIRECTORY | O_NOFOLLOW` relative to its parent's descriptor, and every
  create, rename, link, removal, mode change and time change is made relative to a held descriptor. The
  root itself is opened as given and may be a symlink, an operator's choice. No symlink below a root is
  followed by a write, a mode change or a removal, and files the runtime owns are read only if they are
  regular.
- **Nothing under a root is modified in place.** A file that changes is written to a fresh private
  temporary created with `O_EXCL | O_NOFOLLOW`, given its mode through its descriptor, and renamed over its
  name. Replacing a symlink or hard link replaces the name; its target is never written, truncated or
  chmodded.
- **Downloads are staged privately.** Fetch writes and hashes each download through one descriptor in a
  private `0o700` staging directory, then links or renames it into `blobs/<sha256>` by descriptor, first
  writer wins. A blob nobody hashed is never trusted by size.
- **A wrong kind at an owned path is corrupt, never retryable.** A symlink or non-directory where a store or
  cache directory belongs, or anything but a regular file where a blob or marker belongs, is moved aside
  (quarantined) and the key heals. Only an object that cannot be moved raises an error, naming the path,
  its owner and the command that removes it.
- **Ids that become path components are one component:** non-empty, not `.` or `..`, no `/`, `\` or NUL.
- **Modes come from the store policy, never the umask.** Files and directories are created private and
  then set to their policy mode through their descriptor.
- **An entry is bound to its address and to its context.** `result.json` records the key digest and the
  context fingerprint (`result@2.0`, `store@4`, ADR-006). A reader requires both to match the path and
  `context.json`; a mismatch is misfiled and quarantined.
- **Not guaranteed.** The directories above an entry are reached by an earlier walk, so a rename between
  the walk and the read can redirect a read to another entry, which must still verify against its address
  and context. Store trees of an older format are never collected; the operator removes them once no
  process of the release that wrote them still runs.

**Alternatives rejected.**
- **Refusing a wrong kind and leaving it.** In a shared root another user's stray object would wedge a key
  for everyone until its owner acted. Moving it aside heals the key and keeps the object for its owner.
- **Resolving paths with `realpath` and checking the prefix.** It races with a rename between the check
  and the use; descriptor-relative operations do not.

## Amendment 2026-11-01: locks are leases by rename and `O_EXCL` (D-2)

Restores the Decision's lease protocol for every lock under a storage root, including the lock beside a
run manifest. Owner decision D-2 ([`decisions-2026-11-01.md`](../decisions-2026-11-01.md)).

- **Replaces** the `flock` lock with a heartbeat, holder file and takeover that rulings H-R40..H-R42
  ([`rulings.md`](../rulings.md)) introduced during Phase 1 hardening, and the negative-timeout
  (unbounded) and shared modes that only 1.x callers used. Those were a change of lock semantics that no
  amendment recorded; the Decision chose rename and `O_EXCL` only and rejected fencing.
- **A lock is a lease.** The lease is the file `<name>.lease`, taken by an `O_EXCL` link of a private,
  fsynced temporary and never rewritten. It is stale when its age on the storage clock passes
  `stale_after`. A holder owns its lease by the uuid token it wrote; a breaker identifies the lease it
  judged by device, inode and mtime, because inode numbers alone are reused at once. A lease is broken or
  released by rename and removed only if it matches. POSIX has no conditional rename, so a narrow race can
  displace a newer holder; the overlap is duplicate work, which is within "correctness by idempotence".
- **Bounded.** A wait ends in `LeaseTimeout` naming the holder; a directory that refuses the temporary, or
  a lease that cannot be judged or moved, raises `LeaseUnavailable`. Correctness rests on idempotence, not
  on the lock.
- **Kept from the 2026-10-31 amendment**, because they are path safety and not lock semantics: the
  descriptor walk with `O_NOFOLLOW`, no modification in place, quarantine of wrong kinds, and private
  staging.

## Amendment 2026-11-01: one transient classifier; retry takes a policy

F1 and F3 ([`rulings.md`](../rulings.md), H-R57).

- One classifier decides whether a fault is transient (a retry may succeed) or permanent, for OS errors and
  for HTTP errors from any client. Storage faults say "retry" only when their cause is transient.
- Retry takes an explicit policy loaded from data, not an environment variable. The values are the 1.x
  defaults and are not measured (R-127).

## Amendment 2026-11-01: services are staged

Phase 2 Task 9b; rulings P2-14, P2-38.

- **Phase 2: a service is process-scoped.** The calling process starts it (`VllmServer`, an `Env.server` env),
  owns it and stops it: on leaving its context, or through a watchdog when the owner dies. No registry entry,
  no consumer lease and no idle timeout exist; two processes that need one engine start two servers.
- **Why staged:** a lease has no renewal (R-111), and a server outlives any stale bound, so a registry lease
  would be broken under a live service. Only the start is leased: servers sharing a host's devices start one
  at a time, under a `Lease` (bound below).
- **Phase 3:** the leased registry `{identity, endpoint, heartbeat}`, consumer leases, and the `svc` pool worker
  of design §4.4 and `execution.md` §3.5 arrive with lease renewal.
- **Unchanged:** every reply carries the server's attestation (I-8), and the execution context records it.
- **Start lease bound:** the lease is taken before the server is launched and before its health deadline starts,
  so its stale bound is the start-up timeout plus the profile's `envs.server_start_margin_s`; otherwise a holder
  still inside its own deadline could be broken.
- **Owner death:** the server runs in its own process group. A watchdog, a child of the owner in a session of its
  own (so a signal to the owner's group does not reach it), kills that group once the owner is gone: on the
  SIGTERM `PR_SET_PDEATHSIG` sends when the owner dies (Linux), or when its parent is no longer the owner (every
  platform, polled each second). A SIGTERM while the owner lives is ignored.
- **Start failure kinds:** `exited` covers a KV cache that cannot hold `max_model_len` and a port taken between
  the free-port probe releasing it and the server binding it (a process outside the start lease can take it);
  `version` covers an HTTP error or a wrongly shaped body from `/version` or `/v1/models`; `model` is a served
  root other than the requested model.
- **Per-reply check is by model name:** a reply is refused when its server has exited or it names another model.
  A dead server's port reused by another server of the same model would pass; the window is narrow and the
  version and root are not re-probed per reply.

## Amendment 2026-11-03: a lease is a directory, renewed by beats (owner item 58)

Replaces the lease primitive of the D-2 amendment. Everything else in it stands: a lease is stale when its age
on the storage clock passes `stale_after`, a holder owns it by its uuid token, nothing is modified in place,
waits are bounded, and correctness rests on idempotence.

- **Primitive.** `mkdir` is atomic within one filesystem on local disks, every NFS version, SMB, NTFS, Lustre and
  GPFS. `O_EXCL` hard links are not available on SMB, NTFS or some FUSE mounts, and a rename over an open file
  fails on Windows. No operation is atomic across two filesystems, so every path below is inside the lease's own
  directory.
- **Take.** `mkdir <name>.lease`, then `mkdir <name>.lease/<token>`. The token in the path tells a holder that a
  `mkdir` answered `EEXIST` after a lost NFS reply was its own.
- **Renew (the heartbeat).** A renewer owned by the lease, independent of the work it guards, creates
  `<token>/beat-<n+1>` and removes `<token>/beat-<n>` every `renew_fraction × stale_after`. Each entry change sets
  the directory's mtime on the server, so the last beat has a storage-clock time and no file is rewritten. A beat
  whose `mkdir` fails with `ENOENT` means the lease was broken: the holder stops and does not commit as owner.
- **Judge and break.** A breaker reads the lease's age by a fresh `stat` of `<token>/` against the storage clock
  (a probe directory's mtime). Past `stale_after`, it renames `<name>.lease` aside (one atomic rename within the
  parent) and takes its own. Beats have new names each time, so inode reuse needs no `(dev, ino, mtime)` check.
- **Release.** The holder renames the lease aside and removes it.
- **Bounds.** The renew fraction leaves room for clock skew and the measured attribute-cache delay of the root;
  a root that cannot `mkdir` atomically, rename a directory, or give a server-clock mtime is refused with
  `StorageUnsupported` (the storage probe). Object stores have no directories and are refused.
- **Services.** A registry entry `{identity, endpoint}` lives under a lease kept alive by its server's renewer;
  the beats are its liveness, so it carries no separate heartbeat field.

## Amendment 2026-11-06: renewal as built (R-111)

Records how the 2026-11-03 amendment was built (Phase 3 Task 12, ruling P3-30, and the review fix round) and closes
R-111. It replaces nothing in that amendment; it fixes what that amendment left open.

- **Storage clock.** A breaker reads the storage clock from a fresh probe directory: `mkdir` of a uuid-named
  directory beside the lease, its `lstat`, then `rmdir` (`StorageDir.dir_now`). No file is written for the clock.
  A wait reads the clock once and adds the monotonic time since; a lease is broken only when its age, read again by
  `lstat` after a second, fresh probe, passes `stale_after`. The monotonic extrapolation can only make a waiter
  re-probe early, never break early. The storage clock itself stepping forward by more than
  `stale_after − renew interval − attribute_delay_s` would break a live lease; that is outside the protocol.
- **Age.** The newest mtime of `<name>.lease` (read before and again after its entries are listed, so a claim
  renamed inside it meanwhile shows), of its token directories and of its `gone-`/`claim-` entries; a stray
  entry does not keep a lease fresh. An empty `<name>.lease`, left by a taker that died
  between its two `mkdir`s, is judged by its own mtime. A stale lease this user cannot list is removed by `rmdir` when
  it is empty, as a crash between a lease's `mkdir` and its mode change leaves it; one that holds entries cannot be
  claimed and is not broken: the wait ends in `LeaseUnavailable` naming the owner. That is unreachable with the
  packaged modes: every
  lease is `shared_dir_mode` or `lease.dir_mode` (group-listable; `lease.dir_mode` is a profile key), except the label map's, which is
  `PRIVATE_DIR_MODE` under a `0o700` directory that no other user can reach. `PRIVATE_DIR_MODE` is an invariant
  in the P-34 form, not config: the label map is readable by its owner only, and no profile may open it.
- **Take.** After `mkdir <name>.lease/<token>`, the taker lists the lease. Another token beside its own means its
  lease directory was swapped under it; it removes its token and does not hold. It then `rmdir`s `<name>.lease`,
  which succeeds only if the directory is empty, so two takers that both see each other's token and both withdraw
  leave nothing at the name (otherwise an empty, fresh lease that no one holds would stall every waiter for
  `stale_after`). The rule stays asymmetric: whoever sees another token withdraws; a tie-break by token order
  would let one keep `{A, B}` while the other keeps `{B}`. A third taker that opened the directory before that
  `rmdir` gets `ENOENT` on its token and retries. Finally the taker checks its token afresh by name. A lost reply
  on the first `mkdir` answers `EEXIST`; the taker treats it as another's and breaks its own empty lease one bound
  later.
- **Break and release: a claim by rename.** A breaker that judged a lease stale, or a holder releasing its lease,
  first claims it by renaming an entry inside it: its token `<token>` to `gone-<token>`, or, in a lease with no token
  (a claimant that died after its claim), another entry to `claim-<uuid4>`. An entry is one directory's, so of any
  number of contenders exactly one claim succeeds; the others get `ENOENT` (or `EINVAL` on APFS while the lease is
  torn down) and move nothing. The claimant stamps the entry before the rename and again after, so a judge that
  lists the lease at any point of the claim, even with the lease's own stat served from an attribute cache, ages it
  as fresh and does not claim it again; a `claim-` entry is chosen only when the lease holds no other entry, and only the claimant renames `<name>.lease` aside (after checking that the directory at the name
  is the one it claimed in) and removes it. A lease with no entry at all is removed by `rmdir <name>.lease`, which
  succeeds only while it is empty, so it can never remove a lease a taker has put its token in; a taker between its
  two `mkdir`s just retries. Under N contending breakers each lease is moved at most once, by its claimant. No
  `lstat` identity compare remains: the one fix round 1 kept for tokenless leases was not a one-syscall gap but the
  client's attribute-cache window on NFS, during which a stale `lstat` could let a loser move a fresh lease. A moved
  lease is never put back (the D-2 link-back is gone), and nothing renames onto `<name>.lease`.
- **Renewer.** `Lease.keep_alive(renew_fraction)` starts a daemon thread named for the lease, with its own handle
  on the directory. Beats are due on a monotonic cadence of `renew_fraction × stale_after`: each is due one
  interval after the last was due, whatever the beats took, so beat latency does not stretch the period. Each beat
  reopens `<name>.lease` and `<token>` by name; `ENOENT` or `ESTALE` (an NFS handle to a removed directory) marks
  the state lost at once. Any other failure is retried at the next beat, and one unrenewed for longer than
  `stale_after` is lost. A renewer that ends for any reason other than being stopped marks the state lost. On
  release the renewer is stopped and joined for one interval; one held in a hung beat is left to end on its own.
- **Proof before commit.** `Lease.prove()` makes and removes a directory in the holder's token directory. A `mkdir`
  is answered by the storage, never from a client's attribute cache, so it fails once the lease was claimed and
  removed. Every commit under a lease is preceded by `prove()`, and a holder whose proof fails is lost and does not
  commit. The run manifest and the label map are read-merge-write, so each commit is one atomic replace of the
  merged content (`write_atomically`) made after a successful proof. The manifest's holder re-takes the lease,
  re-reads and merges, up to `lease.record_attempts` times; the label map's raises `RunIOError`.
- **Callers.** Every lease holder beats under the profile's `lease` section (`envs.lease_stale_after_s` is folded
  into it). On a lost lease, or a failed `prove()` right before publishing, the env build is used but its record
  is not published; the manifest and the label map
  are not written. The vLLM start lease only orders starts: lost before the launch (checked by `prove()`), the
  start is refused; lost after it, the healthy server is kept and the loss is logged.
- **The beat bound is checked at load.** `lease.attribute_delay_s` is data (the NFS default acdirmax, 60 s, in
  the `slurm` template; 0 on local disk) and a floor: `probe_capabilities` measures, from one client, how long a directory's mtime takes to show a new entry, and `check_storage` checks the bound at plan time against the larger of the two. One client cannot see another's stale attribute cache, so the probe only ever raises the floor. `ExecutionProfile.load` refuses a profile where
  `(1 + lease.missed_beats) × renew_fraction × stale_after_s + attribute_delay_s >= stale_after_s` with
  `LeaseBoundError`; the packaged profiles allow one missed beat (`lease.missed_beats` 1). Both packaged profiles load (slurm, once filled: 198 + 60 = 258 < 300).
- **Unsupported roots.** `StorageUnsupported` is raised when `mkdir` fails as an unsupported operation (`ENOTSUP`,
  `EOPNOTSUPP`, `ENOSYS`). `probe_capabilities` measures directory rename, server-clock mtimes, file rename, exclusive `link` and the attribute delay on the root, and `require` raises the same type naming each missing primitive.
- **Not guaranteed.** On NFS a breaker's `lstat` of a token may be served from its attribute cache for up to
  `acdirmax`; the load-time check holds only as far as `attribute_delay_s` is right. On the holder's side, `owns()`
  and the beat's reopen by name are dentry lookups that the client may trust for up to `attribute_delay_s`, so a
  holder whose lease was claimed may believe it holds for one interval plus that delay. That is why commits rest on
  `prove()` rather than `owns()`, and why `ESTALE` counts as lost. A proof's `mkdir` can still land in a lease that
  a claimant has renamed aside but not yet removed (one syscall gap). The cross-node check (three nodes, holder
  SIGKILLed) is `scripts/probe_lease_cross_node.py`, run on ORCD.

## Amendment log

- 2026-10-31: paths, kinds and entry binding under a storage root (R-100, R-101; H9b).
- 2026-10-31: lock files `flock`ed with heartbeat and takeover (H-R40..H-R42). Reverted 2026-11-01 (D-2).
- 2026-11-01: locks are leases by rename and `O_EXCL` (D-2).
- 2026-11-01: one transient classifier (`faults.is_transient`); retry takes a `RetryPolicy` from data; no ambient retry knob (F1, F3).
- 2026-11-01: services are process-scoped in Phase 2; the leased registry and `svc` pool worker arrive with lease renewal in Phase 3 (P2-14, P2-38; Task 9b).
- 2026-11-03: a lease is a directory taken by `mkdir` with the holder's token, renewed by beat directories from a renewer independent of the work; replaces the `O_EXCL` link (owner item 58).
- 2026-11-06: renewal as built: probe-directory clock, fresh re-read before a break, claim by rename before a break or release (tokenless leases included; empty ones by `rmdir`), withdrawn leases removed, no put-back, `keep_alive` renewer on a monotonic cadence held by every lease caller, `prove()` before every commit, `ESTALE` as lost, the beat bound (one missed beat allowed) checked at profile load, `StorageUnsupported` on an unsupported `mkdir`; closes R-111.
- 2026-11-07: the Phase 3 service stage built: registry entries `{identity, endpoint}` under renewed leases, consumer leases, the `svc` pool worker; drops `heartbeat` (Phase 3 Task 17, P3-40).
- 2026-11-07: a server backend is placed on a `svc` class first, a providing set consumes a live entry of the identity before reserving its own, the registry is per host and visible device set, and a server whose device lacks the memory it asks for is refused before launch (Phase 3 Task 20c).
- 2026-11-07: the lease constants this ADR names are profile keys (`lease.record_attempts`, `lease.dir_mode`, `lease.missed_beats`), per P-34 as amended; `PRIVATE_DIR_MODE` stays an invariant (a security property). The other code constants move into the profile, with derivations in `config-derivations.md`, under the fixed-parameter audit.
- 2026-11-08: measures roots, layered (shared, then own) under these rules: write-once files, a hardware index, a probe lease per measured unit, a serve lease pinning a probe queue to one worker, bounded peer waits, a read-only root that falls back and never fails, one trust filter over every record kind, and `collect` of the collector's own files (Phase 3 Task 20R).
- 2026-11-08: the outcome record names the epoch of the claim that wrote it (Phase 3 Task 21R1).
- 2026-11-10: the lease section as built lags the 2026-11-07 decision: three keys so far; the beat-bound factor, the manifest retry count and the lease directory mode are not yet keys (pending under R-219; Phase 3 Task 22a).

## Amendment 2026-11-06: queues of plan fragments (Phase 3 Task 13)

- A queue segment is a directory; a put takes `slot-<n>` by an exclusive `symlink` (taking `n+1` on `EEXIST`), so no
  shared file is appended to. A segment retires only when it is full and every slot is resolved, and it retires in
  place: an exclusive `retired` mark inside `segments/<seq>/`, never a rename, so a segment number is made once and a
  stale lookup cannot make it again. A slot is marked resolved only once `outcome/<digest>` exists and the slot, read
  again, names that digest. Each item has one
  first-writer-wins outcome, `outcome/<digest>` (done or dead); the outcome record is staged, then the claim is proven,
  then one link is made. A run re-puts every unfinished fragment on start and resume.
- Amended 2026-11-07 (Task 16 split): the put that takes `slot-<n>` then writes `jobs-<n>-<count>` in the same
  segment by an exclusive create (`count` ≥ 1, the item's jobs), so a segment's listing gives the queue's work in jobs
  without reading an item. A slot without the mark is counted from its item.
- **Not guaranteed.** A crash, or a lost NFS reply on the `symlink`, between taking a slot and writing the item's
  `slotted` marker gives the item a second slot on a later put or reconcile. Exactly-once still holds (one outcome per
  digest); `depth()`, an upper bound, counts the item twice until a claimer resolves the second slot.

## Amendment 2026-11-07: the Phase 3 service stage built (Task 17, P3-40)

Completes the 2026-11-01 staging. A registry entry is `{identity, endpoint}` under a renewed lease, replacing
`{identity, endpoint, heartbeat}`: the lease's beats are the liveness (2026-11-03 amendment).

- **Registry.** One directory per host, `<server env root>/services/<host token>`, because a server listens on loopback
  only. A service is named `svc-<canonical digest of its Attestation>`: `<name>.json` is its `service-entry@1.0`
  entry, written by atomic replace after `prove()`, and `<name>.lease` is its server process's lease, renewed by
  `keep_alive`. An entry is live only while that lease is held and younger than `stale_after`. A provider reserves
  the lease before it starts its server, removing an entry a dead holder left, and publishes the entry once the
  server is up, so a provider that loses the race starts nothing and no consumer is sent to a dead holder's endpoint.
  A second reservation of an identity whose lease is live fails at once (`LeaseTimeout`).
- **Consumers.** Each consumer holds `<name>.consumers/<uuid>.lease`, renewed the same way; counting the live ones
  removes the stale ones. A process's worker set consumes a live entry that attests what its backend, env lock,
  model and settings declare instead of starting a server; with none live it starts a process-scoped server as in
  Phase 2.
- **Placement (amended 2026-11-07).** A server backend is placed on the first `svc:<id>` class of a device it
  declares before any other class; no other backend is placed on a service class; a profile with no service class
  places it as any backend. A providing set, like a consuming one, consumes a live entry attesting the identity
  before it reserves one, so the `svc` workers of one identity on a host start one server; one that finds the
  registration reserved but not yet published gets `LeaseTimeout` and gives its items back. The registry is one
  directory per host and set of visible devices (`<server env root>/services/<host token>/<devices>`, the devices named as
  the start lease names them), so each card registers its own server and a consumer binds to its own card's.
- **Device memory (amended 2026-11-07).** Inside the start lease, before the launch, a server reads the free and
  total memory of its devices (`nvidia-smi`) and is refused with `VllmOutOfMemory`, a resource fault, when any device
  has less free than `gpu_memory_utilization` of its total. A start that still fails carries its log's last
  device-memory error line (else its last error line) and the log's path as detail.
- **The `svc` pool worker.** A pool worker of `svc:<id>` classes registers each server it starts. Once it holds one,
  empty queues decide nothing: it exits when no consumer lease of any service it holds has been seen live for
  `services.idle_exit_s`, read every `pool.poll_s` (so the exit can come up to one poll early); the jobs it runs
  itself count as consumption, and a lost registration lease ends it. The scaler never stops a running worker of a
  service class for an empty queue (it still stops pending ones); a coordinator closing its run still stops every
  worker it started.
- **Per-reply check (I-8).** A consumer reads the entry before and after each request. The reply is refused
  (`StaleAttestation`) unless the entry is live both times at the same endpoint and the reply names the attested
  model; a connection the server drops is `StaleAttestation` too. A restart with another config is another identity,
  so another name, and the old identity's consumers find no live entry. Not covered: between the two reads, another server of the same model bound on the old port while
  the old entry's lease is still live.
- **The key uses the attested value (I-8).** A server backend's call carries the attestation of the service
  answering it (`Call.service`), and its logical key holds it (`LogicalKey.service`, ADR-002). The attestation is
  computed before any server starts or entry is read, from the env lock the declaring package names
  (`Packaged.env_lock`), the model and the server settings, so a store lookup needs no server and a coordinator's
  key agrees with its workers'. The registry is keyed by that attestation, so the entry a call consumes attests
  exactly it, and a started server that attests anything else is refused (`AttestationMismatch`). A restart with a
  new config changes the key; a restart with the same config keeps it.

## Amendment 2026-11-08: measures roots and probe leases (Phase 3 Task 20R)

Design: [`calibration-probe.md`](../calibration-probe.md) §3, §4.4. Adds to the Decision; replaces nothing in it.

- **Layered roots.** `calibration.measures_roots` lists roots read in order and pooled: on a cluster, the group's
  shared root beside the result store, then the user's own. Each holds `measures@1/`: `classes/<declared>/<seen>`
  (the hardware index) and `units/<u[:2]>/<u>/` per `MeasureUnit` digest `u` (ADR-002 amendment of this date) with
  `unit.json` (first writer wins), `obs/`, `drift/`, `supersede/`, `regime/`, `oom/` and `probe.lease/`. The
  2026-10-31 rules apply unchanged: walks by descriptor, nothing modified in place, modes from the store policy, a
  wrong kind at an owned path moved aside. A reader recomputes `u` from `unit.json`; a directory whose digest disagrees
  is misfiled, under the malformed-file rule below.
- **Writes go to the first writable root.** Writability is probed once at run start, before any work, and the plan
  names the write root. Any later `OSError` on a root (permission, quota, stale handle, unreachable) makes it unreadable
  or unwritable for the run: logged once, never raised; reads use the other roots, writes the next writable one. A
  read-only member reuses the shared root and writes to their own.
- **Write-once, so no lock guards a write.** Every file is staged under `tmp/<uuid>/` and renamed onto a uuid name no
  other writer uses.
- **One probe per unit, for efficiency.** A coordinator looks for a live `<u>/probe.lease` in every root and, finding
  none, takes it in its write root (the 2026-11-03 directory lease, `keep_alive` renewed, `prove()` before its
  observations are renamed in). Finding one live, it waits at most `peer_wait_s` (`probe_s +
  lease.stale_after_s`, a derived property of the profile), reading once per `pool.poll_s`, then plans again with what it found.
  Correctness does not rest on the lease: two probes give two valid sets of observations.
- **The pin is a lease.** A probe's items go on a probe queue `<queue>~probe-<id>`. A worker claims from it only while
  holding its `serve.lease`, one item per claim. A dead holder's lease goes stale and another worker takes the rest;
  its first requests report a load.
- **Trust is a reader's filter over every record kind, by owner uid.** `obs`, `drift`, `supersede`, `regime` and
  `oom` records carry their writer's scheduler job id, run, senselab version and commit, profile digest and hardware
  class, and no account name or hostname: the roots are readable by other users (ADR-007, 2026-11-08). The writer is
  the file's owner uid. `calibration.trust` `group` reads all, `own` only files the current uid owns (the run config
  overrides the profile); a record acts only for readers that admit it. Run provenance names probe
  ids and counts.
- **A reader moves only its own files.** A file of an unknown format tag is skipped and logged. A malformed or
  misfiled file is quarantined when the reader's uid owns it, else skipped; never fatal.
- **`collect` removes only files the collector's uid owns** (superseded observations beyond `observations_kept`,
  staging, quarantine past the policy's ages).
- **Not guaranteed.** One probe per unit; that a peer's estimate arrives within `peer_wait_s`; agreement between
  measures written by different senselab versions under one `MeasureUnit` (a confirming probe after drift settles it).

## Amendment 2026-11-08: the outcome record names the claim's epoch (Phase 3 Task 21R1)

- **`outcome/<digest>` carries `epoch`.** The record `{"outcome": "done" | "dead", "worker", "finish", ...}` gains
  `"epoch": <int>`: the coordinator epoch current when the claim that wrote it was taken, read from the queue's epoch
  source (0 when the queue has none). It is fixed at claim time, so a claim taken in epoch 1 that commits after epoch 2
  put the same item still names 1.
- **Meaning.** A coordinator of epoch `e` gives an outcome as executed in its attempt only when the record's `epoch` is
  `e`. A record of another epoch, or without the field, is another attempt's work: the coordinator gives it as served
  under `exact`, or under the policy the worker already recorded when that is `equivalent` or `tolerant`. The pool
  decides this where it settles an outcome; it reads no outcome at put time. The epoch therefore also decides
  `finished` against `cache_hit` in the run's events, not only the efficiency fence of the safety section: results
  stay correct without it (idempotence, first-writer-wins), but which attempt is credited with the work depends on
  it.
- **One writer shape.** Every writer of `outcome/<digest>`, a claim's commit or dead and the queue's own dead record
  for a corrupt item, builds the record the same way and names an epoch.
- **Unchanged.** First-writer-wins, the staged-then-linked write, the one-outcome-per-item rule and fencing by epoch.

## Amendment 2026-11-10: the lease section as built (Phase 3 Task 22a), closed by Task 24-6

The 2026-11-07 amendment made `lease.record_attempts`, `lease.dir_mode` and `lease.missed_beats` profile keys under P-34
as amended. At the Phase 3 head the profile's `lease` section had three keys, `stale_after_s`, `renew_fraction` and
`attribute_delay_s`; Task 24-6 added the three above and `wait_s`, `poll_s` and `remove_attempts`, all required and
set in both packaged profiles, each with its derivation in config-derivations.md.

- **`lease.missed_beats`.** The beat bound is the ADR's: `(1 + missed_beats) × renew_fraction × stale_after_s +
  attribute_delay_s < stale_after_s`, refused with `LeaseBoundError` at load (`JobLease.check_bound`); `check_storage`
  applies it at plan time with the larger of the profile's `attribute_delay_s`, a floor, and the delay measured on each
  root. The value is at least 1 (the beats a live holder may miss) and is 1 in both packaged profiles.
- **`lease.record_attempts`.** The manifest's holder, and the merge of a run's `inputs.json`, re-take the lease and merge
  up to this many times.
- **`lease.dir_mode`.** A lease directory is made with this octal mode, `2770` as packaged. The private mode stays an
  invariant (`runtime/storage.py`).
- **`lease.wait_s`, `lease.poll_s`, `lease.remove_attempts`.** The default wait for a lease, the wait between takes, and
  the removals of a moved-aside lease; they were constants of `runtime/locks.py` (R-254 closed: `wait_s` is a floor
  derived from `stale_after_s`, not a build time). `JobLease` refuses `wait_s < stale_after_s + attribute_delay_s`, the
  time a taker needs to break a dead holder's lease, and the run's combined lease takes the longest `wait_s` of its
  epochs (never less than its combined stale bound plus delay), as it takes the longest `stale_after_s`.
- **The tag is bumped.** `lease.json` is the run's whole lease section, so these keys enter it, and `run-lease@1`
  becomes `run-lease@2` (R-300). No reader of the old tag: a `lease.json` tagged `run-lease@1` is refused as unreadable
  (`RunLeaseUnreadable`) with a message naming both tags.
- **Services, as built.** The idle exit of a service worker is the profile key `services.idle_exit_s`, 600 s in both
  packaged profiles, compared with the idle time of each service the worker provides. The registry entry carries no
  heartbeat. The 2026-11-07 service amendment stands otherwise.

## Amendment 2026-11-11: the run's lease, the ready epoch and the live item (Phase 3 Task 24, T24-2)

- **An epoch is current once it is bound.** The current epoch is the highest `n` whose `epoch/<n>/attempt.json`
  exists. A coordinator makes `epoch/<n>` and `lease.json` and takes its lease, sets up, and publishes `attempt.json`
  last. On any exit before that it writes `ended`. An epoch that never became current fences no coordinator and
  changes no status. `Epoch.take` numbers from the highest directory, so two coordinators never share a number. When no
  epoch is bound, an external waiter judges the highest directory by its `ended` mark, else its age.
- **`lease.json` holds the coordinator's whole `lease` section** (`run-lease@2`), read with the section's own model;
  a missing or unknown key makes the run unreadable (`RunLeaseUnreadable`): a run directory written by another build is
  not resumed. The run's lease is the composite of every bound epoch's section and the starting coordinator's own: the
  longest `stale_after_s` and `attribute_delay_s`, the smallest `renew_fraction`, every other key from the newest.
  Every judge of a run-scoped lease (claims, serve leases, the coordinator lease, `inputs.json`, liveness, probe
  abandonment) and every holder of one uses it (`QueueSet.for_run`). A coordinator whose composite fails the beat
  bound, with the attribute delay measured on the run's storage, takes no epoch. A worker whose profile's section is not
  the newest bound epoch's refuses to start (`LeaseMismatch`), and one that sees a newer epoch with another section
  stops before its next claim. A worker of an older epoch judges by its own bound until that next turn, so for at most one
  turn it can break a lease an epoch with a longer bound still beats.
- **Replaces** "A run re-puts every unfinished fragment on start and resume" (2026-11-06) with: a run re-puts every
  unfinished job on start and resume, except a job a live item already holds on the queue the job is now placed on, under
  the same key. The coordinator follows that item with the jobs of its own fragment, so each job is given and charged
  once and an item's observations land once. A job whose live item is on another queue is put again; that duplicate is
  the fresh placement of ADR-004, bounded by the old items, and accepted.
- **Settle after the request.** A worker settles each item's outcome after the request that stored it, not after its
  claim's group; the renewal proofs run once per renewal interval, not per emit. At most one request's items, and one
  multi-job item still waiting on its other requests, lie between "stored" and "settled". A request's report goes with its outcomes, so every item keeps its own wall and
  observations. A holder killed between staging and linking leaves its temporary in `outcome/`; it names no outcome,
  so the item is claimed again, and a later settle removes only its own temporary.
