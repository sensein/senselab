# H8c findings, surface S (store, fetch, locks, revision, bep028) at b42d3cf0, macOS APFS (case-insensitive)
Probes: <scratch>/h8c-store/probes/ (run with `uv run python <probe> <scratch dir>` in a worktree at b42d3cf0; worktree since removed).

## XS-1 (high) fetch writes through a planted directory symlink under files/
Invariant: symlink tricks inside the store root must not reach outside it; fetch.py docstring "every deletion/write owned".
Repro (probes/fetch_symlink.py): create `cache/datasets/ds@1/files/sub -> /victim` (victim/f is a regular file), fetch a manifest with path `sub/f`.
Observed: `victim/f` is replaced by a symlink to `../../../../blobs/<sha>`; manifest never rejected. `_foreign_files` lists a symlink-to-dir as a dir "above" a manifest file and passes it; `_link` -> `ensure_dirs`/`os.symlink`/`os.replace` follow it.
Expected: ManifestConflictError (a non-directory/symlink component under files/), no write outside the dataset. Needs write access to the group-writable cache (ADR-003 multi-user model). Not a regression of a closed A-row that I know of.

## XS-2 (medium) a directory at blobs/<sha256> wedges the dataset forever, with a misleading error
Repro (probes/blobdir.py): `mkdir cache/blobs/<sha256 of file>`; fetch.
Observed: every call raises `StoreUnavailable [Errno 21] Is a directory: ... retry` naming the cache root, not the blob. `_blob_state` calls it "absent", `_commit` does `os.replace(file, dir)`. A symlink or fifo there heals; a directory does not and nothing quarantines it.
Expected: heal (set aside, as the store does) or a typed corrupt error naming the path and owner. Invariant: no key wedged by another user's entry.

## XS-3 (medium) a file at store@2/<xx> or store@2/<xx>/<digest> wedges put for that key
Repro (probes/wedge.py): replace the digest directory (or its 2-char prefix directory) with a regular file. get returns None; put raises `StoreUnavailable [Errno 17] File exists ... retry` on every call.
Expected: corrupt/quarantine path like a bad entry, or at least a non-retryable typed error naming the owner. `ensure_dirs` -> `mkdir(exist_ok=True)` raises FileExistsError, which `_ensure` maps to retryable StoreUnavailable. (Quarantine-as-file, dangling symlink, symlink to /tmp and fifo at the entry path all healed correctly.)

## XS-4 (medium) result.json is not bound to context.json: swapped outputs serve under the wrong context
Invariant: ADR-002/003 "an entry is bound to its address"; store.py module docstring.
Repro (probes/swap.py): put the same key under contexts linux and macos with different outputs; swap the two result.json files (arrays too, if any) and re-seal.
Observed: `get(k, linux, EXACT)` returns macos's outputs and vice versa, no quarantine. Binding checks key digest from result.json and fingerprint from context.json separately; result.json carries no context fingerprint, and outputs are only bound to their own output_identity.
Expected: result.json records the context fingerprint (or hashes of both) and reads require it to equal the path. Needs fs write access (accidental copy/rsync mix-ups are enough).

## XS-5 (low) put returns "existing" for an entry whose array file is corrupt; get then drops it
Repro (probes/put_trust.py): corrupt the .npy of a committed entry; `put` returns the identity as success (arrays only checked present), immediately `get` -> None (quarantined); entry is only rewritten by the next put. Documented in the put docstring, but the writer's success is false.
Also seen in probes/stress.py: after a vandal process, put-then-get returned None.

## XS-6 (low) took_over is True after a clean release
Repro (probes/lock_tookover.py): A holds the lock and releases cleanly after 0.3 s while B is waiting. B reports `took_over=True` ("previous holder never released"). The snapshot of `.holder` is taken before the acquire and the 0.5 s contended threshold is time-based. Real crash case also reports True (correct). Only a misleading flag/logs.

## XS-7 (low) revision manifests: umask and run id
Repro (probes/rev.py): under umask 0, `runs/` is 0777 and `resolutions@2.json` is 0666 (immutable-for-the-run manifest becomes world-writable). `record_resolution(..., run="../../escaped")` writes outside `runs/` (run id from SENSELAB_RUN_ID is not validated). Leftover `resolutions@2.json.tmp.<pid>` on crash is never collected.

## Held (tried, no break)
- Concurrent fetch, 6 processes on an empty cache: all succeed with correct bytes; os._exit in `_link` mid-dataset then refetch succeeds.
- 4 get/put processes plus a vandal corrupting arrays: no wrong data served, no exceptions (only misses).
- path_key vs real APFS case-insensitive volume: every code point 1..0x2FFFF with any case/NFC/NFD variant, 194558 candidates: 0 fs collisions that path_key fails to join. HFS+ ignorables (ZWJ/ZWNJ/LRM/BOM/soft hyphen etc.) do not join on APFS and path_key does not either.
- Name/version collision (`a@b`+`c` vs `a`+`b@c`) and name-case collision: caught by manifest digest in marker.
- Planted entries (file, dangling symlink, symlink to dir, fifo) at the fingerprint path: quarantined or set aside, put heals.
- collect: removes in-place set-aside, sealed staging, quarantine uuid dirs; non-uuid-named junk is left alone (by design). Quarantine-is-a-file falls back to in-place set aside.
- umask 0 on store, fetch cache (dirs, blobs, files, markers, .clock): no world bits. (Lock dirs/files are 0o2775/0o664 by design.)
- Concurrent record_resolution, 8 processes: one winner.
- https-only fetch, redirects to non-https, size+1 bound: code read, no break found (not exercised against a network).
- Not run: case-sensitive APFS disk image (hdiutil), NFS clock skew; foreign-uid ownership cases (single user machine); bep028 only read.
