# Exit attack: storage, locks, provenance (target c54e9712)

## ES-1 (medium-high) A run id of 32 lowercase hex is collected as a set-aside object, deleting its manifest
- Invariant: revision.py header and ADR-006: a run resolves each (repo_id, ref) once and every participant binds
  to that answer; `2x/runs/<run id>/` is the run's namespace. `_remove_leftovers` documents that it removes
  only "objects moved aside under 2x/runs/".
- Cause: `choose_run_id` accepts any one path component. `_remove_leftovers` runs `runs.remove_older(cutoff)`
  whose default pattern is `UUID_HEX` (`^[0-9a-f]{32}$`), the same grammar as set-aside names. A run directory
  named by `uuid4().hex` (a plausible SENSELAB_RUN_ID) is indistinguishable from a set-aside object and is
  rmtree'd once its mtime is older than quarantine_retention_s (7 days; the run dir mtime moves only when
  the manifest gains a key).
- Repro (<scratch>): record_resolution(run="0123456789abcdef0123456789abcdef"),
  utime the run dir and manifest 8 days back, record_resolution(run="otherrun") by any other run:
  observed `2x/runs/<id>` gone, read_manifest == {}, the same (repo, ref) then rebinds to a different SHA
  (`bbbb...` replacing `aaaa...`). Expected: the live run's binding survives; a run id must not be able to
  take a set-aside name (reject it in `choose_run_id`, or give set-asides a name no run id can have).
- Default uuid4 ids (dashed) and Slurm job ids (digits) are not affected.

## ES-2 (low) Run id longer than NAME_MAX is "one path component" but fails with a raw OSError
- Invariant: `choose_run_id` / `record_resolution` docstrings: the only error for a bad run id is ValueError;
  storage faults are StoreUnavailable.
- Repro (p3.py): run="x"*300: `record_resolution` raises raw `OSError` [Errno 63]/ENAMETOOLONG (not
  StoreUnavailable, not ValueError), `Resolver.resolve` leaks it; `read_manifest` returns {} silently, so
  every resolve re-asks the Hub and then fails at record. Expected: refused up front in `choose_run_id`.
  Same for a lease over a resource whose name makes `.<name>.lease.<pid>.<tid>.<32hex>.tmp` exceed 255.

## ES-3 (low) merge() is documented order-independent but accessor answers depend on order
- Invariant: `ProvStore.merge` docstring "order-independent"; `fingerprint` ignores order.
- Repro (p2.py): two stores each with `was_generated_by(e, aN)` for different activities.
  `merge([A,B]).generated_by(e) == a1`, `merge([B,A]).generated_by(e) == a2`, fingerprints equal.
  The `_targets_of` lists are in first-seen order, so `generated_by` (first element), `derived_from`,
  `associated_with` etc. depend on merge order while the fingerprint says the stores are equal.
  Expected: sorted lists, or one-generator enforced, or the claim narrowed.

## ES-4 (low) The relation index accepts dangling and wrong-typed endpoints, and round-trips them
- Invariant: read_jsonl docstring says relations "pass through the same membership check as the relation
  methods"; only the relation NAME is checked.
- Repro (p2.py): `was_generated_by("nope","nada")`, `used(entity_id, entity_id)` are recorded; write_jsonl then
  read_jsonl returns them unchanged. A relation whose ends are not held records, or whose end is of the wrong
  record type (used(activity, entity) given two entities), is silently valid. Expected: refusal at `_relate`
  (held ids of the right kind) at least on read.

## ES-5 (low) Wrong-kind objects the lease sets aside are never collected
- Repro (p3.py): symlink at `res2.lease`: moved to `<uuid>` beside it in the resource directory. For the run
  manifest this is `2x/runs/<run>/<uuid>`, and `_remove_leftovers` collects only children of `2x/runs/` and
  manifest temporaries, so it stays forever. Victim file untouched (checked). Expected: collected like the
  other set-asides, or put in the store's quarantine.

## ES-6 (low) Lost manifest update after a displaced lease contradicts "a concurrent writer's entries are kept"
- The documented narrow race (ADR-003 amendment) lets a releaser or late breaker displace a newer holder
  (`_take_aside` renames a foreign lease, then links it back; the name is empty in between so a third taker
  can take it and the link-back then deletes the second holder's lease). The ADR accepts duplicate WORK, but
  `_record_in` is a read-modify-write under the lease: two overlapping holders lose one's key, and a ref that
  moved between the two answers then yields a different SHA for a participant that already used the first.
  Not reproduced with real timing (needs the window); reasoned from locks.py `_take_aside` and revision.py
  `_record_in`. The same lease also ages from the temporary's mtime, so a taker stalled in link for
  > stale_after owns an already-stale lease.

## ES-7 (info) audio_io.check fast path trusts (size, mtime_ns, ino, dev), no ctime
- A same-size in-place rewrite with mtime restored passes `check`; `SoundFileReader.decode` re-hashes and
  raises StaleSource, so no wrong samples are served. Fetch blobs learned this (R-77); audio_io did not.
  `stamp`/`open` on a FIFO path blocks forever (`open(path,"rb")`, no O_NONBLOCK, unlike StorageDir).

## duplicates (P-37)
- `ProvStore._hold`: `_put(tables[record._KIND], record, where)` is called twice in a row (identical line).
- Temporary-name format `.{name}.{pid}.{tid}.{uuid}.tmp` built twice: `locks._temporary_name` and
  `StorageDir.write_atomically`.
- Run-directory path re-derived from `manifest_path(...).parents[3]` / `.parents[2]` in `_run_dir` and
  `_remove_leftovers`, instead of one `open_runs()`.
- `store._write_synced` is a one-line alias of `StorageDir.write_new`.
- `store._commit` and `fetch._commit` are two stage-then-publish-then-fsync sequences with different publish
  verbs (rename onto dir vs link-first/replace); `store._seal`/`_reseal` and `StorageDir.remove_older`'s
  set_mode before delete repeat the unseal/seal dance.
- `fetch._regular` (lstat + S_ISREG) repeats `StorageDir.kind`/`open_regular`; `fetch._hashed` repeats
  `file_digest` over a dup'd descriptor.
- `Lease._record_at/_carries` and `_is` are two predicates of one shape (name -> bool) with separate
  lstat/read paths.

## held
- Mutual exclusion of Lease under stale holder / late breaker: acknowledged in ADR-003 (efficiency hint).
- Token ownership: A broken while held, B takes; A.owns() False, B.owns() True, A's exit leaves B's lease (p3.py).
- Inode reuse: judged lease keyed by (dev, ino, mtime_ns); a fresh lease cannot share a stale one's mtime.
- Planted hard link at `<name>.lease` (victim in scratch, old mtime): broken as a name, victim content and
  data intact (nlink back to 1); symlink at the name: set aside, victim untouched.
- Run ids `..`, `.`, slashes, backslash, NUL, empty: refused by `path_component`.
- Dangling/symlinked `2x/`, `runs/` or run dir: `walk` refuses or sets aside, never follows.
- Staging private-mode check (uid + exact 0o700) holds; store staging uses shared mode but entries are
  verified by content/key binding on read.
- Fetch blob link/replace path: no victim reached; `keeps()` re-verifies a racing blob.
