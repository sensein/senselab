# The ORCD CPU stampede: `owns()` self-sabotage, a too-short timeout, and a bounded retry

Sibling to `design.md` (the heartbeat-`inf` fix). That fix's own item 5 introduced
`SharedFileLock.owns()` and `ensure_venv`'s post-build check on it. This document covers a second,
independent defect in that same check, found while investigating a fresh failure it did not
predict.

## The failure

A 37-task pre-emptible CPU array on ORCD ran one subject's task battery. 36 of 37 failed, all with:

```
RuntimeError: PREPROCESS: 2 block(s) failed unexpectedly:
  asr_crisperwhisper: RuntimeError: Venv 'crisperwhisper' lost its lock to a concurrent process during build; retry.
  asr_qwen: RuntimeError: Venv 'qwen-asr' lost its lock to a concurrent process during build; retry.
```

The parallel GPU array survived 35 of 37, but only because `QOSMaxGRESPerUser` throttled it to a
couple of concurrent tasks, so the venv was already built before the rest arrived. The CPU
partition has no such cap: 36 tasks raced an unbuilt venv at once.

## Diagnosis: the builder loses to its own waiters, not to a real takeover

The message says the *builder* lost its lock, not that a waiter timed out. Tracing the code as it
stood at `d349b216`:

`ensure_venv` (`subprocess_venv.py`) wraps lock *acquisition* in an unbounded retry that only
catches `TimeoutError`:

```python
lock = SharedFileLock(venv_dir, timeout=600)
while True:
    try:
        lock.__enter__()
        break
    except TimeoutError:
        logger.info(...)
        continue
```

A waiter that cannot get the flock within 600s raises `filelock.Timeout`, `SharedFileLock.__enter__`
converts it to `TimeoutError`, and this loop retries forever — a waiter never raises. So the
`RuntimeError` seen in the logs is not that path. It comes from further down, reached only by
whichever process *did* get the flock and finished its (real, successful) install:

```python
if not lock.owns():
    logger.error(...)
    shutil.rmtree(venv_dir, ignore_errors=True)
    raise RuntimeError(f"Venv '{name}' lost its lock to a concurrent process during build; retry.")
```

`owns()` (`file_lock.py`, pre-fix) was:

```python
def owns(self) -> bool:
    current = lock_holder(self._lock_path)
    return current is not None and self._token is not None and current.get("token") == self._token
```

`self._lock_path` is the **same file** `filelock.FileLock` wraps for the OS-level `flock()`. That
file's own docstring already named the mechanism that defeats reading it after the fact:

> `filelock`'s `UnixFileLock._acquire` opens the lock file with `O_TRUNC` on *every* poll attempt
> including the ones that fail to take the lock, so a waiter's own polling erases the live
> holder's payload within one poll interval.

Confirmed against the installed `filelock` package (`_unix.py`):

```python
def _acquire(self) -> None:
    open_flags = os.O_RDWR | os.O_TRUNC | os.O_CREAT
    ...
    fd = os.open(self.lock_file, open_flags, open_mode)
    ...
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exception:
        os.close(fd)
        ...  # EAGAIN/EWOULDBLOCK: someone else holds it, try again later
```

`os.open(..., O_TRUNC)` runs **before** the `flock()` attempt and **unconditionally** — including
on the branch where `flock()` then fails because another process holds the lock. `filelock`'s
default poll interval is 0.05s (`BaseFileLock.acquire`). So every waiter, on every one of its
non-blocking polls — successful or not — truncates the shared `.lock` file's content to empty.

The real builder writes its identity into that same file once, via `_write_holder()`, immediately
after acquiring. With 35 other processes each polling at 20 Hz for the ~600–800s the build takes,
the file is emptied roughly every `0.05/35 ≈ 1.4ms` on average. The window in which
`lock_holder(self._lock_path)` could read the builder's own payload back is a few milliseconds
inside a multi-hundred-second build — so by the time the *actual, still-flock-holding* builder
finishes and calls `owns()` to certify its own successful, uncorrupted build, `lock_holder()`
reads back `None` almost certainly. `owns()` returns `False`. The builder concludes — wrongly —
that another process took over, deletes the venv it just finished, and raises.

This is a self-inflicted false negative, not a real takeover: the OS-level `flock()` was never
lost (mutual exclusion held throughout — only one process is ever inside the build section at a
time); only the *content* used to verify that fact was destroyed by innocent, unrelated failed
polls from processes that never got anywhere near acquiring it.

**Why 36 of 37, and why the survivor.** Once a builder self-sabotages, it `rmtree`s the venv and
releases the lock. The next process to acquire finds no marker, becomes the new builder, and is
just as likely to be undone by whichever other processes are still cycling their own acquire
attempts. This repeats — cascading rebuild-then-self-sabotage — until either contention thins
enough (fewer processes still polling) for one attempt to survive uncorrupted, or a process
exhausts its own wall-clock budget. This matches "one survivor, the rest each raise once" without
requiring any process to have genuinely held the flock more than once.

**Ruled out:**
- *A waiter's own timeout being fatal* — ruled out above; the acquisition retry loop only catches
  `TimeoutError` and never raises it further.
- *The staleness takeover (`cdde8ed8`'s fixed logic) still firing* — would require an actual
  second `flock()` acquisition. Mutual exclusion at the OS level is not in question here; nothing
  in this failure needed a stale-detection branch to fire at all.
- *The heartbeat thread dying* — irrelevant to this path; `owns()` doesn't consult the heartbeat.
- *`filelock`'s own timeout behaviour* — behaves exactly as documented (`Timeout` on a failed
  bounded wait); the defect is in what senselab layered on top of its lock file's content, not in
  `filelock` itself.

## Fix 1: identity moves to its own file

`SharedFileLock` now derives a third path, `.holder` (alongside the existing `.lock` and
`.heartbeat`), and stores the JSON identity payload there instead of in `.lock`. Nothing but a
genuine new holder's own `_write_holder()` call — reached only after actually acquiring the
`flock()` — ever writes to `.holder`; a waiter's failed, non-blocking polls (`filelock`'s own
`_acquire()`) never touch it, because `filelock` only ever opens `self._lock_path`. This applies
uniformly to `owns()` and to the pre-acquire `previous_holder` read in `__enter__` (used for the
`Timeout` message and the stale-takeover check) — both now read `.holder`. `__exit__` unlinks
`.holder` alongside `.heartbeat` instead of truncating `.lock`'s content, since nothing meaningful
is written to `.lock` any more.

This is the same separation the heartbeat file already had (heartbeat has always lived in its own
file, never inside `.lock`) — the identity payload was the one piece of state still sharing a file
with `filelock`'s own polled lock, and it inherited that file's corruption exposure as a result.

## Fix 2: the timeout (and its fallback bound) raised from measurement

`600s` was never fitted to a build time; it was inherited unchanged from the plain `FileLock` this
class replaced. Measured cold `crisperwhisper` builds
(`specs/20260817-triage-workflow-dag/benchmarks/orcd-scheduling-2026-09-08.md`):

| run | reported | what it actually measures |
|---|---|---|
| CPU, this run | 579.69s | clean cold build (timestamps span exactly this) |
| CPU, first attempt (personal scratch, prior run) | 590s | clean cold build |
| GPU, this run | 794.11s reported, **362.6s actual** | build + ~430s of *lock-contention wait* conflated into one logged figure; the build itself, isolated by its own `Creating isolated venv` → `Venv ... ready` timestamps, is 362.6s |

So confirmed pure build time ranges **362.6–590s** across both hosts and both runs (n=2 per
point, per that benchmark's own caveats — no claim of higher precision). The 794s/800s figures
quoted from the incident report are the mislabeled build+wait number, not a build time; using them
as if they were a build measurement would double-count the very lock contention this fix
addresses.

**Chosen default: `1200s`.** This is roughly 2x the highest *confirmed pure build* measurement
(590s) and still comfortably above the mislabeled 794s/800s figures, which is deliberate given n=2
per data point: a fitted, unpadded ceiling on two samples is not a ceiling on the population. This
value is `SharedFileLock`'s `timeout` for the venv-build lock in `subprocess_venv.py`, is
configurable via `SENSELAB_VENV_LOCK_TIMEOUT` (`_venv_lock_timeout()`), and is not a bare literal
in the call site.

**The fallback staleness bound is raised for free.** `SharedFileLock._staleness()` falls back to
`self._timeout` (not `self._stale_after`) as the bound when a holder's heartbeat file cannot be
read at all:

```python
taken_at = holder.get("taken_at")
if isinstance(taken_at, (int, float)):
    return max(0.0, time.time() - taken_at), "taken_at", self._timeout
```

Since `ensure_venv` constructs one `SharedFileLock(venv_dir, timeout=_venv_lock_timeout())`
instance and `_staleness()` reads `self._timeout` off that same instance, raising the constructor
argument raises this fallback bound identically — confirmed by inspection, no separate change
needed. A holder with an unreadable heartbeat but a legible `taken_at` up to ~1200s old (instead of
~600s) is now tolerated as still-building before staleness is even considered.

## Fix 3: waiter reuse — verified already correct, not touched

`_ensure_venv_once`'s marker check runs immediately inside the critical section, before any build
work:

```python
if marker.is_file():
    stored = json.loads(marker.read_text())
    ...
    if stored.get("requirements") == sorted(requirements) and stored_index_url == expected_index_url:
        logger.debug("Reusing existing venv: %s", venv_dir)
        return venv_dir
```

Any process that acquires the lock — whether it waited from the start or is a later retry after
losing an earlier race — hits this check first. If a previous holder finished and wrote the
marker, the new acquirer reuses it and returns immediately, without calling `_find_uv()` or
`subprocess.run` at all. This logic was already correct; the reason it was invisible on the ORCD
run is Fix 1's bug: `owns()`'s false negative meant *no* builder ever survived long enough to
*write* the marker under heavy contention, so there was nothing for a waiter to find. Fixing
`owns()` is what lets this pre-existing reuse path actually fire; it required no changes on its
own, and the concurrency test below (`test_stampede_on_a_slow_build_lets_every_late_arrival_reuse_the_winners_venv`)
asserts exactly one real build occurs among several concurrent callers, which is this path
working.

## Fix 4: a lost-lock during build is now retried, bounded

Even after Fix 1, a *genuine* takeover remains possible (the original heartbeat-staleness path
this lock exists to support, e.g. a builder that really did crash and get displaced mid-build by
a legitimate stale-lock takeover). Before this change that raised straight into
`ensure_venv`'s caller — in production, into PREPROCESS's `hard_failures`, killing the whole node
for that recording on what may be a recoverable race.

`ensure_venv` now retries a `_VenvLockLost` (the internal exception `_ensure_venv_once` raises from
the `owns()` check) up to `_MAX_LOCK_LOST_RETRIES = 3` times. Each retry re-enters
`_ensure_venv_once` from scratch: it re-acquires the lock and re-checks the marker first (Fix 3),
so if the process that took over finished in the meantime, the retry reuses it instead of
rebuilding. `_MAX_LOCK_LOST_RETRIES` is a small engineering ceiling, not a value fitted to
measurement — CLAUDE.md's "thresholds belong in `data/` with a derivation" is about tuned
detection thresholds (a silhouette cutoff, an HNR ramp), not a bounded-retry count for a now-rare
race; no derivation is claimed for it beyond "more than one, not unbounded."

This is judged clean to implement: the existing per-attempt body was already self-contained and
idempotent (its first action after acquiring is the marker check), so wrapping it in a bounded
retry loop needed no restructuring beyond extracting it into `_ensure_venv_once` and renaming the
raise to a dedicated exception type the wrapper can catch selectively (never swallowing a genuine
install failure, e.g. `subprocess.CalledProcessError`, which is a different exception type and
still propagates immediately).

## Verification

`src/tests/utils/file_lock_test.py` — updated to read/write the fabricated holder payloads used by
the pre-existing staleness tests against `.holder` instead of `.lock`, matching the new derivation
(`_holder_file()` helper alongside the existing `_lock_file()` / `_heartbeat_file()`).

`src/tests/utils/subprocess_venv_test.py` —
`test_stampede_on_a_slow_build_lets_every_late_arrival_reuse_the_winners_venv`: several concurrent
threads call `ensure_venv` for the same never-before-built, torch-free backend, with
`subprocess.run` stubbed so the "install" step is a real `time.sleep()` (releases the GIL, so
`fcntl.flock()`'s per-open-file-description semantics give genuine OS-level contention across
threads in one process — no subprocess/env-var plumbing needed) longer than an artificially
shortened `SENSELAB_VENV_LOCK_TIMEOUT`, mirroring the ORCD ratio of a too-short timeout against a
too-long build at test scale. Verified against the code at `d349b216` (pre-fix): the test fails,
reproducing multiple `RuntimeError: ... lost its lock ...` results and more than one real "install"
call, matching the cluster's shape. Against the fix: zero errors, all callers return the same venv
directory, and exactly one real install call is recorded — the rest reused the marker.
