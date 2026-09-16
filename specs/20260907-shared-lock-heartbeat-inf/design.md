# `SharedFileLock`: a live holder judged stale, and why the age read `inf`

## The failure

Two Slurm jobs (GPU node3805, CPU node2803) shared `$HOME/.cache/senselab/venvs/` and both
built `crisperwhisper`. The GPU job logged:

```
WARNING - Stale lock at .../crisperwhisper.lock: heartbeat is infs old (> stale_after=120.0s).
Previous holder was user=satra host=node2803 pid=245707. Breaking lock and taking over.
```

`node2803` (the CPU job) was alive and installing at that instant. A second, independent run
(`specs/20260817-triage-workflow-dag/benchmarks/orcd-scheduling-2026-09-08.md`) reproduced the
same message **in both directions on one run** — each side declared the other's live, in-progress
build dead — and that run's own `full_call` attempt then failed:

```
RuntimeError: Importing the numpy C-extensions failed ...
libscipy_openblas64_-f48b354e.so: cannot open shared object file: No such file or directory
```

a `.so` missing from an install that otherwise looked complete: the signature of two processes
having mutated the same venv tree. The same log also shows ~430s of the GPU job's reported 794.11s
`crisperwhisper` build was lock-contention wait, not build time (confirmed from that run's own
timestamps) — the false "stale, breaking" declaration doesn't just risk corruption, it also
misattributes wait time as build time in every downstream benchmark that reads `ensure_venv`'s
wall clock.

## Tracing the `inf`

`SharedFileLock._heartbeat_age()` (`src/senselab/utils/file_lock.py`) computed:

```python
def _heartbeat_age(self) -> float:
    try:
        return time.time() - self._heartbeat_path.stat().st_mtime
    except OSError:
        return float("inf")
```

Any `OSError` from `stat()` — overwhelmingly a missing file — became `float("inf")`, which is
`> stale_after` unconditionally. That alone would make *every* holder without a currently-readable
heartbeat instantly stale. But the heartbeat file is not, in general, missing while a legitimate
holder is running: `__enter__` touches it immediately on acquire and a background thread
(`_heartbeat_loop`) refreshes it every `heartbeat_interval` (30s default) for as long as the lock
is held, including across a blocking `subprocess.run` (the GIL is released for the underlying
`waitpid`, so the thread keeps ticking). That part of the design already worked; it is not what
produced the field failure.

**The actual mechanism.** `__enter__` read the previous holder's identity *before* calling
`self._lock.acquire(timeout=...)`:

```python
previous_holder = lock_holder(self._lock_path)
try:
    self._lock.acquire(timeout=self._timeout)
except Timeout:
    ...
else:
    if previous_holder is not None:
        age = self._heartbeat_age()
        if age > self._stale_after:
            self._warn_stale_takeover(previous_holder, age)
```

`__exit__` unlinks the heartbeat file and clears the lock file's content *before* releasing the
flock:

```python
def __exit__(self, *exc):
    ...
    self._heartbeat_path.unlink(missing_ok=True)
    self._lock_path.write_text("")
    self._lock.release()
```

So: when a second process's `acquire()` call has to **wait** (the first process is still holding
the flock) and the first process then finishes and exits *normally*, the wait resolves via a clean
release. The `previous_holder` the waiter captured *before* it started waiting is still sitting in
its local variable, but the heartbeat file it names is now gone — deleted as part of the exit that
just happened, not because anyone crashed. `_heartbeat_age()` reads the missing file and returns
`inf`; the `else` branch, having no way to tell "cleanly finished" from "leftover from a crash",
treats it as stale and logs the takeover.

This reproduces deterministically and needs no NFS caching or clock-skew story: every ordinary
contended-then-succeeded handoff — a completely normal event — hits this path. `_UNCONTENDED_ACQUIRE_THRESHOLD`
below is what the field timestamps corroborate: the GPU job's "stale" log and its own "Creating
isolated venv" line are 0.3s apart, and the second CPU-side "stale" log follows the GPU job's own
"Venv ready" line by 40ms — in both cases, the taking-over process's `acquire()` call had been
blocked on the other side's still-running build and fired the instant it released.

**Confirmed by `lock_holder`'s own docstring**, which had already documented the mechanism that
defeats any attempt to fix this by re-reading content *after* acquiring:
`filelock.UnixFileLock._acquire()` opens the lock file with `os.O_TRUNC` on *every* attempt,
including failed non-blocking polls — so the file's content cannot be trusted to still show
anything by the time an `acquire()` call returns, regardless of whether the previous holder
crashed or exited cleanly. A "re-read after acquire and check if it's now empty" fix was tried
first and rejected: it made `test_a_stale_heartbeat_is_taken_over` fail, because the file is
*already* empty by the time any acquire (contended or not) returns.

## The fix

1. **`_heartbeat_age()` now returns `Optional[float]`.** `None` means "cannot currently read a
   heartbeat" — not "infinitely old". Every caller must treat `None` as unknown, never as stale.
2. **`__enter__`'s `else` branch now gates on how long the acquire took, not on re-read content.**
   `previous_holder` is still captured once, before the first acquire attempt (the only read
   guaranteed to precede any truncating `_acquire()` call). After a successful acquire, elapsed
   wall time since the attempt began decides which of two cases this is:
   - **Elapsed ≤ `_UNCONTENDED_ACQUIRE_THRESHOLD` (0.5s):** the flock was free the moment we
     asked — a crashed process's flock is kernel-released the instant it exits, so any
     `previous_holder` identity left on disk is a genuine leftover. Evaluate it for staleness.
   - **Elapsed > the threshold:** we waited for someone, and they released during that wait.
     Per the class's own existing invariant (a live flock cannot be held continuously and also
     release early), this can only be a normal exit. `previous_holder` is not evaluated at all —
     no "stale" log, no takeover reasoning, because there is nothing to take over.

   0.5s is comfortably above `filelock`'s own 0.05s poll interval (so a single failed poll cannot
   be mistaken for genuine contention) and comfortably below any real contention in this codebase,
   which runs to minutes.
3. **A missing/unreadable heartbeat still doesn't mean infinitely old when staleness *is*
   evaluated** (the genuinely-uncontended case above). `_staleness()` falls back to the previous
   holder's own recorded `taken_at`, bounded by `self._timeout` (the same "this can legitimately
   take minutes" grace the class already grants a live acquire attempt) rather than the tighter
   `stale_after` (calibrated for heartbeat cadence, not for "no heartbeat at all"). If neither the
   heartbeat nor `taken_at` is readable, the lock is not taken over silently: `__enter__` releases
   the flock it just took and raises `TimeoutError`, matching the existing "cannot confirm — do not
   break it" contract of the `except Timeout` branch.
4. **Takeover logging now names the basis** (`heartbeat` vs `taken_at`) and the threshold that
   was crossed, in addition to the previous holder's user/host/pid, so a takeover that does happen
   is loud and explains itself.
5. **A completed build now verifies it still owns the lock before certifying itself.**
   `SharedFileLock._write_holder()` stamps a random `token` alongside user/host/pid/taken_at;
   `SharedFileLock.owns()` reports whether the identity currently on disk still carries this
   instance's token. `ensure_venv` (`subprocess_venv.py`) calls `owns()` immediately before writing
   `.senselab-installed`; if it returns `False` — a takeover happened elsewhere during this build,
   however that takeover was justified — the tree is removed and a loud `RuntimeError` is raised
   instead of certifying a build that a second builder may already be treating as its own. This
   does not by itself prevent every concurrent-write scenario (see the residual risk below), but it
   closes the specific "marker present on an incomplete/interleaved tree" gap the corrupted
   `crisperwhisper` build exposed.

## Verification

`src/tests/utils/file_lock_test.py`:

- `test_a_normal_handoff_after_waiting_is_not_logged_as_a_stale_takeover` — the primary
  regression test. A real child process (via `multiprocessing`) holds the lock for 2s and exits
  cleanly; the parent waits for it. Before the fix this logged
  `Stale lock ...: heartbeat is infs old (> stale_after=120.0s) ... Breaking lock and taking over`
  — reproduced verbatim by running the test against the pre-fix code. After the fix, no "Stale
  lock" message is logged.
- `test_a_holder_without_a_heartbeat_yet_is_not_taken_over` / `..._mid_long_install_without_a_readable_heartbeat_is_not_taken_over` —
  manufacture a lock file with a legible, recent-`taken_at` holder and no heartbeat file at all
  (missing outright, or 400s into what `crisperwhisper`'s measured 579.7–794.1s cold builds show
  is a normal in-progress install). Both fail pre-fix (`inf > stale_after` fires every time) and
  pass after.
- `test_a_holder_with_no_heartbeat_and_an_old_taken_at_is_eventually_taken_over` — same shape, but
  `taken_at` is older than `timeout` (700s against 600s). Asserts the takeover **does** fire:
  `assert "Stale lock" in message and "alice" in message and "node1234" in message and "4211" in message`.
  Proves the fix does not simply disable staleness detection.
- `test_heartbeat_is_refreshed_while_held` — asserts the heartbeat file's mtime advances more than
  once while the lock is held (short `heartbeat_interval` for test speed), confirming the existing
  background-thread refresh mechanism actually ticks across a hold.

`src/tests/utils/subprocess_venv_test.py`:

- `test_a_takeover_during_build_refuses_to_certify_the_venv` — monkeypatches `SharedFileLock.owns`
  to `False` and asserts `ensure_venv` raises (`match="lost its lock"`) and removes the venv
  directory instead of writing `.senselab-installed`.

## Residual risk found, not fixed here

`ensure_venv`'s lock only protects the **build** phase. `call_in_venv` calls `ensure_venv` (which
acquires and releases the lock internally), then runs the venv's Python **after** the lock has
been released, unprotected. A second process's `ensure_venv` call for the same name — triggered by
a legitimate marker mismatch, not a bug — can `shutil.rmtree` that same directory while the first
process's subprocess is still executing against it. This is independent of the heartbeat defect
fixed here and is plausibly the more direct mechanism behind the observed missing `.so`: GPU and
CPU jobs need genuinely different `torch_index` builds for the same venv **name**
(`crisperwhisper`), so every handoff between them ends in a real rebuild regardless of staleness
detection — the fix above only ensures that rebuild happens strictly after the other side's build
fully finishes and released, rather than while it is still writing. The benchmark's own
recommendation #3 (moving venvs to a single, group-scratch build referenced read-only by every job)
removes the collision surface entirely; this task does not implement that, per scope.

## Model downloads: same primitive, already covered

`ensure_hf_model` (`src/senselab/utils/dependencies.py`) does not reimplement locking or rely on
`huggingface_hub`'s own per-blob `filelock` locks for its own "check cache, else download, then
cache the result" critical section — it wraps the identical `SharedFileLock` class fixed above,
with the same retry-on-`TimeoutError` shape as `ensure_venv`. No separate heartbeat/staleness logic
exists there (confirmed: no `heartbeat`, `stale`, or `inf` handling in `dependencies.py` beyond
what it inherits from `SharedFileLock`). The fix in `file_lock.py` therefore applies to the HF
download path automatically; no separate defect of this class was found there, and this task does
not otherwise touch `dependencies.py`. `huggingface_hub`'s own internal per-file lock (used to
make concurrent downloads of the *same blob* safe) is a distinct, lower-level mechanism with no
heartbeat or staleness logic of its own — a plain `flock` with a fixed timeout — and is unaffected
by, and out of scope for, this fix.
