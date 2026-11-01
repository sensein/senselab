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
  - The coordinator epoch file (incremented by `O_EXCL` link) is a fence for efficiency only. A
    stale coordinator sees a newer epoch and stops releasing.
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
  - A service is a registry entry `{identity, endpoint, heartbeat}` under a lease.
  - Each consumer holds its own lease on the service. The service exits when no consumer lease has
    been live for T.
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
- **A lock is a lease.** The lease is the file `<name>.lease`. It is taken by writing the holder's
  identity to a private temporary, fsyncing it, and linking it at the name with an `O_EXCL` link relative
  to the directory's descriptor; the link fails if anything is at the name. A link error is judged by the
  temporary's link count, so a lost NFS reply does not lose a lease that was taken. The lease is never
  rewritten, so its identity is whole whenever it exists.
- **Stale and broken.** A lease is stale when its age passes `stale_after`. Age is read on the storage
  clock (`fs_now()` against the lease's mtime) once per distinct lease, and then grows by local monotonic
  elapsed time. Anything but a regular file at the name is quarantined, not judged.
- **Identity of a lease.** A holder owns its lease while the file at the name carries the uuid token it
  wrote into the identity record; `owns()` and release match by that token, never by inode. A breaker
  identifies the lease it judged stale by device, inode and mtime, which rename keeps. Inode numbers alone
  are not identity: ext4 reuses a freed inode number at once, so a new lease can carry the number of the one
  just broken, while its mtime is fresh and a judged-stale lease is older than `stale_after`.
  (`runtime/locks.py`: `_carries`, `_is`.)
- **Broken or released by rename.** Both rename the name to a fresh uuid and compare the moved file with
  the expected one (by token for a holder, by device, inode and mtime for a breaker), removing it only on a
  match; a regular file moved by mistake is linked back. The stale
  lease itself is therefore removed exactly once. POSIX has no conditional rename, so a narrow window
  remains: a late breaker that judged an old lease can rename a newer lease aside before it can link it
  back, and another taker may take the empty name. The displaced holder sees `owns()` false and its
  release logs it. The overlap is duplicate work, which is within "correctness by idempotence".
- **Bounded.** `timeout` and `stale_after` are at least 0. A wait polls and ends in `LeaseTimeout`
  naming the holder; a directory that refuses the temporary, or a lease that cannot be judged or moved,
  raises `LeaseUnavailable` naming the path and its owner's remedy. Correctness still rests on
  idempotence, not on the lock.
- **Kept from the 2026-10-31 amendment**, because they are path safety and not lock semantics: the
  descriptor walk with `O_NOFOLLOW`, no modification in place, quarantine of wrong kinds, and private
  staging.

## Amendment 2026-11-01: one transient classifier; retry takes a policy

F1 and F3 ([`rulings.md`](../rulings.md), H-R57).

- **`runtime/faults.py::is_transient` is the one classifier** of a fault as transient (a retry may succeed)
  or permanent. For an OS error it reads `errno` (a fixed set: `EAGAIN`, `EINTR`, `EBUSY`, `ETIMEDOUT`,
  `ECONNRESET`, `EIO`, `ESTALE` and similar; `FileNotFoundError`, `PermissionError` and `ENOSPC` are
  permanent). For an HTTP error from any client it reads `response.status_code` (429 and 500 and above are
  transient, every other status is permanent); `urllib.error.HTTPError.code` is read the same way.
- **`storage_fault` uses it.** A `StoreUnavailable` says "retry" only when its cause is transient; it no
  longer claims every storage fault is retryable.
- **Retry takes an explicit `RetryPolicy`**, `retry_on_transient_error(fn, policy)`, loaded from
  `runtime/data/retry_policy.yaml` (`retry-policy@1.0`). Its values (3 attempts, 1 s first backoff, doubling)
  are the 1.x defaults and are not measured; the file says so. `SENSELAB_HF_MAX_RETRIES` is no longer read.

## Amendment log

- 2026-10-31: paths, kinds and entry binding under a storage root (R-100, R-101; H9b).
- 2026-10-31: lock files `flock`ed with heartbeat and takeover (H-R40..H-R42). Reverted 2026-11-01 (D-2).
- 2026-11-01: locks are leases by rename and `O_EXCL` (D-2).
- 2026-11-01: one transient classifier (`faults.is_transient`); retry takes a `RetryPolicy` from data; no ambient retry knob (F1, F3).
