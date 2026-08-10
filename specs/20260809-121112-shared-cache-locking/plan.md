# Shared-Cache Locking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One file lock that works when several users share a cache directory — group-writable, with a heartbeat that is actually read, and a holder identity a takeover can name — replacing both existing implementations.

**Architecture:** A new `senselab/utils/file_lock.py` importing only stdlib and `filelock`, so both call sites can use it without circular imports (`dependencies.py` imports nothing from senselab today, and must keep it that way). `dependencies.py` and `subprocess_venv.py` then each drop their own copy.

**Tech Stack:** Python 3.12, `filelock`, stdlib `os`/`socket`/`getpass`/`threading`/`json`.

## Global Constraints

- **No new host dependency.** `filelock` is already used by both call sites.
- **`file_lock.py` must import nothing from `senselab`.** `dependencies.py` has zero senselab imports and that is deliberate; adding one via this module would reintroduce a circular-import hazard. Use `logging.getLogger("senselab")`, as both existing modules do.
- **Lock and heartbeat files are always group-writable** (`0664`), their directories `2775`. Not behind a flag — the failure it prevents is silent.
- **A `chmod` that fails must not fail the lock.** On a shared tree the file may be owned by another user; the mode is already correct in that case, and raising would break the very scenario this exists for.
- **Do not change where any cache lives.** `SENSELAB_VENV_CACHE`, `SENSELAB_CACHE` and `HF_HOME` keep their current meanings and defaults.
- **Do not change how a venv is built** — only who can build, take over, and reuse one.
- **Every Python command runs through `uv run`.** Never bare `python` or `pip`.
- **Run pytest in the FOREGROUND, never backgrounded**, scoped so it returns in seconds. Check `uptime` first; this machine is shared and a run showing minutes elapsed against seconds of CPU is contention, not a hang.
- **Never `pytest -n auto`** or any `-n` flag.
- **Never `git add -A` unqualified.** Use `git add -A -- src/`.
- **Tests must not sleep to test staleness.** Use `os.utime` to age a heartbeat. A sleeping test is a slow test that still doesn't prove much.
- **Test docstrings in Google style**; ruff enforces `D205`/`D209`. The pre-commit `mypy` hook runs with `--extra-checks`, which a plain `uv run mypy` does not — annotate bindings explicitly (`x: T = T(...)`) or CI fails where local passes.

## File Structure

| Path | Responsibility | Action |
|---|---|---|
| `src/senselab/utils/file_lock.py` | `SharedFileLock`: modes, holder identity, heartbeat, stale takeover | Create |
| `src/senselab/utils/dependencies.py` | Drop `_HeartbeatLock`, use `SharedFileLock` | Modify |
| `src/senselab/utils/subprocess_venv.py` | Drop `_FileLockWithHeartbeat`, use `SharedFileLock`; group-accessible venv trees | Modify |
| `src/tests/utils/file_lock_test.py` | All lock tests | Create |

---

### Task 1: `SharedFileLock`

**Files:**
- Create: `src/senselab/utils/file_lock.py`
- Test: `src/tests/utils/file_lock_test.py`

**Interfaces:**
- Produces:
  ```python
  LOCK_FILE_MODE = 0o664
  LOCK_DIR_MODE = 0o2775

  def lock_holder(lock_path: Path) -> dict | None: ...

  class SharedFileLock:
      def __init__(self, path: Path, *, timeout: float = 600.0,
                   heartbeat_interval: float = 30.0, stale_after: float = 120.0) -> None: ...
      def __enter__(self) -> "SharedFileLock": ...
      def __exit__(self, *exc: object) -> None: ...
  ```
  `path` is the resource being guarded; the lock and heartbeat are derived from it (`.lock`, `.heartbeat`), matching what both existing implementations do.

- [ ] **Step 1: Write the failing tests**

```python
"""A file lock that several users can share on one directory."""

import json
import os
import stat
from pathlib import Path

import pytest

from senselab.utils.file_lock import LOCK_DIR_MODE, LOCK_FILE_MODE, SharedFileLock, lock_holder


def test_lock_and_heartbeat_files_are_group_writable(tmp_path: Path) -> None:
    """A second user must be able to refresh the heartbeat and break a stale lock.

    Created under the default umask these land at 0644, so user B's heartbeat
    touch fails — and because both old implementations swallowed that error, B's
    heartbeat silently stopped refreshing and a third user read it as stale and
    broke a live lock.
    """
    resource = tmp_path / "thing"
    with SharedFileLock(resource):
        lock_file = resource.with_suffix(".lock")
        heartbeat = resource.with_suffix(".heartbeat")
        assert stat.S_IMODE(lock_file.stat().st_mode) & 0o060 == 0o060
        assert stat.S_IMODE(heartbeat.stat().st_mode) & 0o060 == 0o060


def test_lock_directory_is_setgid_and_group_writable(tmp_path: Path) -> None:
    """Group ownership must propagate to files the next user creates."""
    resource = tmp_path / "nested" / "thing"
    with SharedFileLock(resource):
        mode = stat.S_IMODE(resource.parent.stat().st_mode)
        assert mode & stat.S_ISGID, "directory should be setgid"
        assert mode & 0o070 == 0o070, "directory should be group rwx"


def test_holder_identity_is_recorded_while_held(tmp_path: Path) -> None:
    """A takeover must be able to name who it displaced.

    On a cluster the holder is often on another node, so 'stale lock detected'
    without a user, host and pid gives nobody enough to check whether that job
    actually died.
    """
    resource = tmp_path / "thing"
    with SharedFileLock(resource):
        holder = lock_holder(resource.with_suffix(".lock"))
        assert holder is not None
        assert holder["pid"] == os.getpid()
        assert holder["user"] and holder["host"]
        assert isinstance(holder["taken_at"], (int, float))


def test_holder_is_cleared_on_release(tmp_path: Path) -> None:
    resource = tmp_path / "thing"
    with SharedFileLock(resource):
        pass
    assert lock_holder(resource.with_suffix(".lock")) is None


def test_a_stale_heartbeat_is_taken_over(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A holder that died must not block waiters until timeout.

    This is what subprocess_venv's plain FileLock lacked: a crashed install held
    the lock for the full 600s and then raised, rather than being detected as
    dead and taken over.
    """
    import logging

    resource = tmp_path / "thing"
    lock_file = resource.with_suffix(".lock")
    heartbeat = resource.with_suffix(".heartbeat")
    resource.parent.mkdir(parents=True, exist_ok=True)
    lock_file.write_text(json.dumps({"user": "alice", "host": "node1234", "pid": 4211, "taken_at": 0}))
    heartbeat.touch()
    os.utime(heartbeat, (0, 0))  # aged, not slept

    with caplog.at_level(logging.WARNING):
        with SharedFileLock(resource, timeout=1.0, stale_after=60.0):
            assert lock_holder(lock_file)["pid"] == os.getpid()
    message = " ".join(r.message for r in caplog.records)
    assert "alice" in message and "node1234" in message and "4211" in message


def test_a_fresh_heartbeat_is_not_taken_over(tmp_path: Path) -> None:
    """A live holder must be waited for, not displaced.

    The mirror of the test above, and the one that matters more: breaking a live
    lock produces exactly the concurrent clobber the lock exists to prevent.
    """
    resource = tmp_path / "thing"
    lock_file = resource.with_suffix(".lock")
    heartbeat = resource.with_suffix(".heartbeat")
    resource.parent.mkdir(parents=True, exist_ok=True)

    with SharedFileLock(resource):
        # Held by us; a second acquisition with a short timeout must fail rather
        # than break in, because the heartbeat is current.
        with pytest.raises(TimeoutError):
            with SharedFileLock(resource, timeout=0.5, stale_after=3600.0):
                pass
    assert heartbeat.exists() is False


def test_chmod_failure_does_not_break_the_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """On a shared tree the file may belong to someone else.

    Its mode is already correct in that case, so a failed chmod must be ignored —
    raising would break precisely the multi-user scenario this class exists for.
    """
    def _boom(*_args: object, **_kwargs: object) -> None:
        raise PermissionError("not the owner")

    monkeypatch.setattr(os, "chmod", _boom)
    with SharedFileLock(tmp_path / "thing"):
        pass  # must not raise


def test_lock_holder_returns_none_for_a_missing_or_junk_file(tmp_path: Path) -> None:
    """A partially written or corrupt lock file must read as unheld, not crash."""
    assert lock_holder(tmp_path / "nope.lock") is None
    junk = tmp_path / "junk.lock"
    junk.write_text("not json")
    assert lock_holder(junk) is None
```

- [ ] **Step 2: Run them and watch them fail**

```bash
uv run pytest src/tests/utils/file_lock_test.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'senselab.utils.file_lock'`.

- [ ] **Step 3: Implement**

Write `src/senselab/utils/file_lock.py`. Requirements, each of which a test above pins:

- Module docstring stating the two invisible assumptions: staleness compares local `time.time()` against a server-set mtime, so `stale_after` must exceed plausible node clock skew; and `filelock` uses `fcntl`, which needs NFSv4 or equivalent to be correct over a network filesystem.
- `_ensure_dir(path)` — `mkdir(parents=True, exist_ok=True)` then `os.chmod(..., LOCK_DIR_MODE)`, chmod failures ignored.
- `_touch_shared(path)` — create if absent, then `os.chmod(..., LOCK_FILE_MODE)`, failures ignored. Use this for both the lock file and the heartbeat.
- `lock_holder(lock_path)` — read and JSON-parse the lock file; return `None` on missing file, empty file, or `JSONDecodeError`. A holder mid-write must read as unheld rather than raise.
- `SharedFileLock.__enter__` — acquire with `timeout`; on `filelock.Timeout`, read the heartbeat mtime. If older than `stale_after`, log a warning **naming the previous holder from `lock_holder`** (user, host, pid, and heartbeat age), unlink the lock, and retry once. If fresh, re-raise as `TimeoutError`. Then write our own holder payload, start the heartbeat thread, and touch the heartbeat.
- `SharedFileLock.__exit__` — stop the thread, join with a timeout, unlink the heartbeat, truncate the lock file's payload (so `lock_holder` reports unheld), release.
- The heartbeat thread swallows `OSError` — but note in a comment that this is why group-writable modes matter: a swallowed permission error is how a live holder's heartbeat goes silently stale.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest src/tests/utils/file_lock_test.py -q
```

Expected: PASS, 8 tests.

- [ ] **Step 5: Add the one real concurrency test**

Everything above is single-process. Add a test that forks a second process which holds the lock briefly, and assert the parent waited rather than proceeding — the one case a mocked test cannot establish.

```python
def test_a_second_process_waits_rather_than_proceeding(tmp_path: Path) -> None:
    """Two live processes must serialise, not interleave.

    Every other test here is single-process and could pass against a lock that
    does nothing. This is the one that cannot.
    """
    import multiprocessing as mp
    import time

    resource = tmp_path / "thing"

    def _hold(path: str, seconds: float) -> None:
        from senselab.utils.file_lock import SharedFileLock

        with SharedFileLock(Path(path)):
            time.sleep(seconds)

    proc = mp.get_context("spawn").Process(target=_hold, args=(str(resource), 1.5))
    proc.start()
    time.sleep(0.3)  # let it acquire
    started = time.monotonic()
    with SharedFileLock(resource, timeout=10.0):
        waited = time.monotonic() - started
    proc.join(timeout=10)
    assert waited > 0.5, f"second acquirer did not wait (waited {waited:.2f}s)"
```

Run it and confirm it passes. If `spawn` cannot pickle the local function, move `_hold` to module scope.

- [ ] **Step 6: Lint, type-check, commit**

```bash
uv run ruff format src/ && uv run ruff check src/ && uv run mypy src/senselab/
uv run --with pre-commit pre-commit run mypy --all-files
git add -A -- src/
git commit -m "feat(utils): a file lock several users can share

Group-writable modes, a holder identity a takeover can name, and a heartbeat that
is actually read. Created under the default umask, lock files land at 0644, so a
second user's heartbeat touch fails — and both previous implementations swallowed
that, so the heartbeat silently stopped refreshing and a third user read it as
stale and broke a live lock.

A failed chmod is ignored rather than raised: on a shared tree the file belongs to
someone else and its mode is already right, so raising would break exactly the
scenario this exists for."
```

---

### Task 2: `dependencies.py` adopts it

**Files:**
- Modify: `src/senselab/utils/dependencies.py` — remove `_HeartbeatLock`, use `SharedFileLock`
- Test: `src/tests/utils/dependencies_test.py`

**Interfaces:**
- Consumes: `SharedFileLock` from Task 1.
- Produces: no new public interface; `_HeartbeatLock` ceases to exist.

- [ ] **Step 1: Find every call site**

```bash
grep -n "_HeartbeatLock" src/senselab/ -r
grep -rn "_HeartbeatLock" src/tests/ || echo "no test references"
```

Record what you find in your report. Replace each, and delete the class. Pre-alpha convention is rename-and-replace outright: no alias, no shim.

- [ ] **Step 2: Preserve the behaviour that already worked**

`_HeartbeatLock` acquired with a 60 s initial timeout and looped — waiting again when the heartbeat was fresh, breaking when stale. `SharedFileLock` does the same thing with `timeout` and `stale_after`. Choose values that preserve the effective behaviour (its defaults were `heartbeat_interval=30`, `stale_threshold=90`) and say in your report what you chose and why.

Do **not** silently shorten how long a waiter tolerates a live download: these locks guard multi-GB model downloads, and a waiter that gives up early turns one slow download into two concurrent ones.

- [ ] **Step 3: Run the tests**

```bash
uptime
uv run pytest src/tests/utils/dependencies_test.py -q
uv run pytest src/tests/utils -q
```

Expected: no regression. Report both counts.

- [ ] **Step 4: Commit**

```bash
uv run ruff format src/ && uv run ruff check src/ && uv run mypy src/senselab/
git add -A -- src/
git commit -m "refactor(dependencies): use the shared file lock

Drops the local _HeartbeatLock. Its stale-detection logic was the correct one and
is preserved; what it gains is group-writable modes and a holder identity, so a
takeover on a shared cache can name the job it displaced."
```

---

### Task 3: `subprocess_venv.py` adopts it, and shared venvs become usable

**Files:**
- Modify: `src/senselab/utils/subprocess_venv.py`
- Test: `src/tests/utils/subprocess_venv_test.py`

**Interfaces:**
- Consumes: `SharedFileLock` from Task 1.
- Produces: no new public interface; `_FileLockWithHeartbeat` ceases to exist.

- [ ] **Step 1: Replace both locks in this module**

Two distinct sites:
1. `ensure_venv`'s `with FileLock(str(lock_path), timeout=600):` — the plain lock with no heartbeat. This is the one whose holder, if it dies mid-install, blocks every waiter for 600 s and then raises.
2. `_FileLockWithHeartbeat` — used once (around line 867). It writes a heartbeat nothing reads. Delete the class and use `SharedFileLock` at its call site.

**Do not change the sequence inside `ensure_venv`'s lock.** The marker check, `shutil.rmtree`, install, and marker write stay exactly as they are; only the lock wrapping them changes. That sequence is what makes a half-built venv un-reusable and it is already correct.

- [ ] **Step 2: Make a shared venv runnable by the group**

Building it group-writable is only half the job — a second user has to execute the interpreter the first user created. After a successful install, walk the venv tree and add group read to files and group read+execute to directories, ignoring failures on entries owned by someone else.

Do this **before** the marker is written, so an interrupted chmod cannot leave a venv that looks complete but is half-permissioned.

(An earlier revision of this step said "after", with the same rationale — which the order it prescribed did not deliver. Writing the marker first means a kill between the two steps leaves a venv advertising itself as ready with its modes half-fixed, and because the chmod pass only runs on the fresh-build path, every later call takes the reuse fast path and never repairs it. Chmod first, then mark: an interrupted pass leaves no marker, the next call's marker check fails, `rmtree` fires, and the rebuild completes the chmod.)

- [ ] **Step 3: Write the failing test first**

```python
def test_a_completed_venv_is_group_readable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A second user must be able to run the interpreter the first user built.

    Group-writable lock files let user B take over a stale build; they do not let
    B execute A's venv. Venv trees are created under the default umask, so without
    this a shared cache is buildable but not usable.
    """
```

Drive it against a fake venv tree rather than a real install — building a real venv downloads packages and takes minutes. Create a directory structure, call the permission-fixing helper directly, and assert the modes. Factor that helper out so it is callable without running an install.

- [ ] **Step 4: Verify no regression in the venv path**

```bash
uptime
uv run pytest src/tests/utils/subprocess_venv_test.py -q
uv run pytest src/tests/utils -q
```

The existing `_cache_dir_path` / `_cache_dir` tests must still pass unchanged — they pin the side-effect-free path query a test skip gate depends on.

- [ ] **Step 5: Confirm `ensure_venv` still works end to end**

This is the one place a real venv build is worth it, because the lock change wraps it:

```bash
uv run python -c "
from senselab.utils.subprocess_venv import ensure_venv
p = ensure_venv('lock-smoke', ['packaging'], python_version='3.12')
print('built at', p)
print('marker:', (p / '.senselab-installed').is_file())
"
```

Expected: builds, marker present. Run it twice — the second must reuse rather than rebuild, proving the marker path still short-circuits. Report both timings.

- [ ] **Step 6: Update the module docstring and commit**

`subprocess_venv.py`'s docstring says "file locks with heartbeat for concurrent access safety". Make it accurate about what that now means: shared-safe modes, a heartbeat that is read, and takeover of a dead holder.

```bash
uv run ruff format src/ && uv run ruff check src/ && uv run mypy src/senselab/
uv run --with pre-commit pre-commit run --all-files
git add -A -- src/
git commit -m "refactor(subprocess_venv): shared-safe locking, and shared venvs the group can run

ensure_venv held a plain FileLock with no heartbeat: a holder that died mid-install
blocked every waiter for the full 600 s and then raised. It now uses the shared
lock, so a dead holder is detected and taken over.

_FileLockWithHeartbeat is deleted. It wrote a heartbeat every 15 s that nothing in
the repository ever read, while its docstring claimed it let other processes
distinguish a live holder from a crashed one.

A completed venv is also made group-readable and group-executable, because building
it group-writable only lets a second user take over the build — not run the
interpreter the first user produced."
```

---

## After this plan

`SENSELAB_VENV_CACHE`, `SENSELAB_CACHE` and `HF_HOME` can each be pointed at a group location, and the group then shares venvs, resolution state and weights with locking that survives a crashed holder and names who it displaced. Nothing about where caches live changes by default.
