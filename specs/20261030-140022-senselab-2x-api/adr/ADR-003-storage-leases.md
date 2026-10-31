# ADR-003: Storage and lease protocol on shared filesystems

**Status:** accepted, 2026-10-30.

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

### Storage paths, kinds and entry binding (H9b, R-100, R-101)
The result store, the fetch cache and the run manifests live under shared, group-writable roots, so
anyone in the group can plant a symlink or a file there. The H8c re-attack showed what followed: fetch
replaced a file outside the cache through a planted `files/sub -> /victim` (XS-1); a directory at
`blobs/<sha256>` or a file at a store directory wedged its key behind a retryable-looking
`StoreUnavailable` (XS-2, XS-3); a run id `../../escaped` wrote outside `runs/`, and under umask 0 the
run manifest was world-writable (XS-7). The rules below make those states unconstructible.

Fix round 1 (the H9b review, ruling H-R40) found that the lock beside each run manifest still created,
chmodded and truncated its sidecars by path, so a planted symlink or hard link reached a file outside the
root (C-1, I-1). The rules below are stated as what is guaranteed after that round.

- **One path primitive.** `storage.StorageDir` reaches every directory below a storage root (the result
  store, the fetch cache, the HF cache's `runs/`, and the directory a `SharedFileLock` lives in) one
  component at a time, opening each with `O_DIRECTORY | O_NOFOLLOW` relative to its parent's descriptor,
  on macOS and Linux. The root itself (and, for a lock, its directory) is opened as given and may be a
  symlink, an operator's choice. Below it, every create, rename, link, symlink, removal, mode change and
  time change is made relative to a held descriptor. No symlink below a root is followed by a write, a
  mode change or a removal; reads of files the runtime owns open them with `O_NOFOLLOW | O_NONBLOCK` and
  check with `fstat` that they are regular, so a symlink is never read through and a fifo never blocks.
- **Nothing under a root is modified in place.** A file that changes (a lock's holder, heartbeat and
  clock, a store or cache `.clock`, a run manifest, a dataset's completion marker) is written to a fresh
  temporary created `0o600` with `O_EXCL | O_NOFOLLOW` relative to the directory's descriptor, given its
  mode with `fchmod` before any byte is written, and renamed over its name. Replacing a symlink or a hard link replaces the name; the file it pointed at or shared is
  never opened for writing, truncated, chmodded or re-stamped. Modes are set only with `fchmod` on a
  descriptor this process opened; nothing is chmodded by path. A directory or other non-file at such a
  name is moved aside before the rename.
- **The lock file is the one exception, and it must be what it seems.** `<name>.lock` must stay put to be
  `flock`ed, so it is opened (created if missing, `O_EXCL` first) relative to the directory's descriptor
  with `O_NOFOLLOW | O_NONBLOCK`, never truncated, and accepted only if `fstat` shows a regular file with
  `st_nlink == 1`. A hard link (`st_nlink > 1`), symlink, directory or fifo there is corrupt and moved
  aside beside it; its target is not touched. A regular file there that this user cannot open read-write
  (`EACCES`, `EPERM`, as another user's `0o600` file) is moved aside the same way; if the directory refuses
  that rename too, the lock raises `LockUnavailable`, naming the file, its owner and the command that
  removes it, never a bare `PermissionError`. 1.x `reproduce.build.file_lock` takes the same lock (shared
  or exclusive) through `SharedFileLock`. After each `flock` the descriptor must still be the file at
  the name (device and inode through a fresh open), else the attempt is retried. `filelock` is no longer
  used; `SharedFileLock` is the one lock implementation.
- **Holders and waits.** A holder file that is not a whole identity (`user`, `host`, `pid`, `taken_at`
  of their types), not a regular file, or unreadable, is corrupt: once the later of its mtime and the
  heartbeat's is older than `stale_after` on the storage clock, it is moved aside and the lock is taken.
  With `timeout >= 0` every wait in `SharedFileLock.__enter__` ends in `LockTimeout` at `timeout`. With a
  negative `timeout` (1.x `build.exclusive_lock` and `file_lock`) the wait for another process that holds
  the `flock` has no bound (H-R41); the wait on a corrupt holder still ends at `stale_after`.
  `record_resolution` waits once, for 600 s, and does not retry.
- **The heartbeat stops at release.** The heartbeat thread writes through its own directory handle,
  checks a stop flag before every write, and closes its handle as it ends; release sets the flag and
  joins the thread for up to 5 s, then removes the sidecars and closes the lock's descriptors. A heartbeat
  write already in progress when that join gives up (a stalled filesystem) can still land after release;
  this is logged.
- **Private staging for downloads.** Fetch stages each download and each marker in `tmp/<uuid>/`, made
  `0o700` by this process and checked with `fstat` to be its own. The download is written and hashed
  through one descriptor; it is hard-linked (first writer wins) or renamed into `blobs/<sha256>` from
  that directory by descriptor. The marker records only the stamp of an inode this call hashed (or found
  recorded): if `blobs/<sha256>` no longer names the hashed inode after the link, no stamp is recorded and
  the next `verify="size"` fetch hashes it again. A blob nobody hashed is never trusted by size.
- **Every written name is listed.** Each runtime module that writes under a root exports
  `WRITTEN_NAMES`, the grammars of the file names it writes, and every `StorageDir` write takes the
  grammar its name must match. A lint over every module that writes under a storage root (store, fetch,
  locks, storage, provenance, BEP-028, run manifests, 1.x `reproduce.build` and `reproduce.cache`)
  reports any write without a listed grammar, and any write, rename, link, removal, mode or time change or
  write-mode open by `os`, `io`, `shutil`, `tempfile`, builtin `open` or `pathlib.Path`, under any alias
  the shared import resolver folds, outside `StorageDir`. The raw sites that remain (BEP-028's sidecars
  and the 1.x reproduce trees, by path in caller-chosen trees, deferred to the 1.x cleanup) are counted in
  the lint, so a new one fails it. A contract plants a symlink, a hard link, a directory and corrupt
  content at every name a clean run writes.
- **A wrong kind at an owned path is corrupt, never retryable.** A symlink or a non-directory where a
  store or cache directory belongs, or anything but a regular file at `blobs/<sha256>` or a completion
  marker, is moved aside by rename (into `quarantine/` for the store, into `tmp/` for the cache, where age
  collection removes it; beside itself if that is refused, which `collect` also ages out under the root,
  `store@4/` and each prefix, and `record_resolution` under `runs/`). An object that cannot be moved
  raises `WrongKindError`, a `ValueError`, naming the path, its owner and the command that removes it.
  Under a dataset's `files/`, whose paths come from the manifest, a wrong kind is a
  `ManifestConflictError`, as a foreign file there already is.
- **Ids that become path components are one component.** Run ids (including `SENSELAB_RUN_ID`),
  dataset names and versions and each part of a manifest path pass `storage.path_component`: non-empty,
  not `.` or `..`, no `/`, `\\` or NUL.
- **Modes come from `StorePolicy`, never the umask.** Every store, cache and run-manifest directory and
  file is created private (`0o700`, `0o600`) and then set to its policy mode through its descriptor. A
  root's missing ancestors, which lie outside the root, are made as `os.makedirs` makes them. A lock's own
  directory, when the lock does not manage it (`manage_dir_mode=False`) and it is missing, is made under
  the umask; a caller's file written by `write_atomically` without a mode keeps the mode the file had, or
  gets the umask's default for a new file, set through the descriptor.
- **An entry is bound to its address and to the context it was written in.** `result.json` records the
  key digest and the context fingerprint (`result@2.0`, `store@4`, ADR-006). A reader requires the stored
  key to digest to the recorded digest, the recorded fingerprint to equal the fingerprint of
  `context.json`, and both to equal the path; a mismatch is misfiled and quarantined. `put` verifies an
  existing entry's arrays before returning its identity.
- **What is not guaranteed.** Store members are read through the entry directory's descriptor, but the
  directories above the entry were reached by an earlier walk; a rename between the walk and the read can
  redirect a read to another entry, which must still verify against the address and the context. A group
  member who hard-links a live lock file makes the next acquirer move it aside and lock a new file while
  the old holder keeps the old one; that needs write access to the lock's directory, which already allows
  removing the lock file. Old `store@3/` trees and other unrecognised names are never collected.
- **Lock takeover is read from the holder file once the lock is held.** A holder writes `.holder` only
  while holding the `flock` and a clean release removes it before unlocking, so `took_over` is whether a
  holder file (whole or corrupt) is still there after the acquire.

## Consequences
- `work_pool` gains index segments and quantum yielding. Items become plan fragments.
- The packed tier and the token arbiter wait for measurements.

## Alternatives rejected
- **Refusing a wrong kind with an error and leaving it** (H9b). In a shared root another user's stray
  object would wedge a key for everyone until its owner acted; moving it aside heals the key and keeps
  the object for its owner, and an error remains only for what cannot be moved.
- **Resolving paths with `realpath` and checking the prefix.** It races with a rename between the check
  and the use; descriptor-relative operations do not.
- **A database or lock service for coordination.** It adds an operated dependency to every site.
- **Strict fencing on NFS.** It cannot be made atomic with check-then-act; idempotence makes it
  unnecessary for correctness.
