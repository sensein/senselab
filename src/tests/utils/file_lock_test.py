"""A file lock that several users can share on one directory."""

import getpass
import json
import os
import stat
from multiprocessing.synchronize import Event as MpEvent
from pathlib import Path

import pytest

from senselab.utils.file_lock import LOCK_DIR_MODE, LOCK_FILE_MODE, SharedFileLock, lock_holder


def _lock_file(resource: Path) -> Path:
    """Mirror SharedFileLock's own derivation: append, not `Path.with_suffix`.

    Kept as a named helper (rather than inlined `Path(str(resource) + ".lock")` at each
    call site) so a future change to that derivation shows up as one helper edit, not a
    dozen silently-stale call sites.
    """
    return Path(str(resource) + ".lock")


def _heartbeat_file(resource: Path) -> Path:
    """Mirror SharedFileLock's own derivation: append, not `Path.with_suffix`."""
    return Path(str(resource) + ".heartbeat")


def test_lock_and_heartbeat_files_are_group_writable(tmp_path: Path) -> None:
    """A second user must be able to refresh the heartbeat and break a stale lock.

    Created under the default umask these land at 0644, so user B's heartbeat
    touch fails — and because both old implementations swallowed that error, B's
    heartbeat silently stopped refreshing and a third user read it as stale and
    broke a live lock.
    """
    resource = tmp_path / "thing"
    with SharedFileLock(resource):
        lock_file = _lock_file(resource)
        heartbeat = _heartbeat_file(resource)
        assert stat.S_IMODE(lock_file.stat().st_mode) & 0o060 == 0o060
        assert stat.S_IMODE(heartbeat.stat().st_mode) & 0o060 == 0o060


def test_lock_directory_is_setgid_and_group_writable(tmp_path: Path) -> None:
    """Group ownership must propagate to files the next user creates."""
    resource = tmp_path / "nested" / "thing"
    with SharedFileLock(resource):
        mode = stat.S_IMODE(resource.parent.stat().st_mode)
        assert mode & stat.S_ISGID, "directory should be setgid"
        assert mode & 0o070 == 0o070, "directory should be group rwx"


def test_a_caller_directory_is_not_chmodded_when_unmanaged(tmp_path: Path) -> None:
    """A lock over a caller-supplied path must not touch that directory's permissions.

    Finding #8 of the #550 review: locking a FileRef in a caller's own private directory
    chmodded that directory from 0700 to 0o2775 -- setgid and group-write, and also
    other-read and other-execute, so a directory its owner deliberately made private became
    world-traversable as a side effect of dropping a ``.lock`` in it. (A directory owned by
    someone *else* was never at risk: chmod returns EPERM there and ``_ensure_dir`` swallows
    it. The victim is the invoking user.) With ``manage_dir_mode=False`` the directory is
    left exactly as found.
    """
    data_dir = tmp_path / "restricted"
    data_dir.mkdir()
    os.chmod(data_dir, 0o700)  # pin an exact, non-shared mode regardless of umask
    resource = data_dir / "input.wav"
    resource.write_bytes(b"x")

    before = stat.S_IMODE(data_dir.stat().st_mode)
    with SharedFileLock(resource, manage_dir_mode=False):
        pass
    after = stat.S_IMODE(data_dir.stat().st_mode)

    assert after == before == 0o700, f"caller dir mode changed: {oct(before)} -> {oct(after)}"
    assert not (after & stat.S_ISGID), "setgid must not be set on a caller-owned directory"


def test_holder_identity_is_recorded_while_held(tmp_path: Path) -> None:
    """A takeover must be able to name who it displaced.

    On a cluster the holder is often on another node, so 'stale lock detected'
    without a user, host and pid gives nobody enough to check whether that job
    actually died.
    """
    resource = tmp_path / "thing"
    with SharedFileLock(resource):
        holder = lock_holder(_lock_file(resource))
        assert holder is not None
        assert holder["pid"] == os.getpid()
        assert holder["user"] and holder["host"]
        assert isinstance(holder["taken_at"], (int, float))


def test_holder_is_cleared_on_release(tmp_path: Path) -> None:
    """After release, lock_holder must read as unheld."""
    resource = tmp_path / "thing"
    with SharedFileLock(resource):
        pass
    assert lock_holder(_lock_file(resource)) is None


def test_a_stale_heartbeat_is_taken_over(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A holder that died must not block waiters until timeout.

    This is what subprocess_venv's plain FileLock lacked: a crashed install held
    the lock for the full 600s and then raised, rather than being detected as
    dead and taken over.
    """
    import logging

    resource = tmp_path / "thing"
    lock_file = _lock_file(resource)
    heartbeat = _heartbeat_file(resource)
    resource.parent.mkdir(parents=True, exist_ok=True)
    lock_file.write_text(json.dumps({"user": "alice", "host": "node1234", "pid": 4211, "taken_at": 0}))
    heartbeat.touch()
    os.utime(heartbeat, (0, 0))  # aged, not slept

    with caplog.at_level(logging.WARNING):
        with SharedFileLock(resource, timeout=1.0, stale_after=60.0):
            holder = lock_holder(lock_file)
            assert holder is not None
            assert holder["pid"] == os.getpid()
    message = " ".join(r.message for r in caplog.records)
    assert "alice" in message and "node1234" in message and "4211" in message


def test_a_fresh_heartbeat_is_not_taken_over(tmp_path: Path) -> None:
    """A live holder must be waited for, not displaced.

    The mirror of the test above, and the one that matters more: breaking a live
    lock produces exactly the concurrent clobber the lock exists to prevent.
    """
    resource = tmp_path / "thing"
    heartbeat = _heartbeat_file(resource)
    resource.parent.mkdir(parents=True, exist_ok=True)

    with SharedFileLock(resource):
        # Held by us; a second acquisition with a short timeout must fail rather
        # than break in, because the heartbeat is current.
        with pytest.raises(TimeoutError):
            with SharedFileLock(resource, timeout=0.5, stale_after=3600.0):
                pass
    assert heartbeat.exists() is False


def test_a_live_holder_with_a_stale_heartbeat_is_not_displaced(tmp_path: Path) -> None:
    """A ``filelock.Timeout`` means a live process holds the lock -- never break it.

    Reaching ``except Timeout:`` proves the OS-level flock was held
    continuously for the entire wait, which a crashed process cannot do (the
    kernel drops its flock the instant it exits). This forces exactly that
    branch against a holder whose heartbeat has gone stale independently of
    its still-live flock -- e.g. its heartbeat thread died or was starved of
    scheduling, which is exactly what ``_heartbeat_loop``'s bare
    ``except OSError`` cannot see. A second acquirer must still raise
    ``TimeoutError`` naming the holder rather than unlinking and taking over:
    doing so would hand it a lock on a fresh inode while the first holder
    still holds the ``flock`` on the orphaned one, so both would believe they
    hold the lock -- the exact clobber this class exists to prevent.
    """
    resource = tmp_path / "thing"
    heartbeat = _heartbeat_file(resource)

    with SharedFileLock(resource):
        # The flock is genuinely held (by us, in this process) for the whole
        # test below, but age the heartbeat file to simulate its refresh
        # thread having stalled or died independently of the work it guards.
        os.utime(heartbeat, (0, 0))
        with pytest.raises(TimeoutError) as excinfo:
            with SharedFileLock(resource, timeout=0.5, stale_after=1.0):
                pass  # pragma: no cover - must not be reached
        message = str(excinfo.value)
        assert str(os.getpid()) in message
        assert getpass.getuser() in message


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


def test_dotted_resource_names_get_distinct_lock_files(tmp_path: Path) -> None:
    """Two resources differing only after a dot must never share a lock file.

    `Path.with_suffix(".lock")` replaces everything from the *last* dot onward, so two
    distinct resources like two revisions of the same model --
    "org--model--v1.5--main" and "org--model--v1.6--main" -- both reduced to the
    identical "org--model--v1.lock" under that derivation, silently merging their
    locks. This forces exactly that case: nesting locks on both resources must succeed
    immediately (append-based derivation makes them different files, so there is no
    contention), which fails against the old `with_suffix` derivation because the
    second, mistakenly-identical acquire blocks until its own short timeout and then
    raises.
    """
    a = tmp_path / "org--model--v1.5--main"
    b = tmp_path / "org--model--v1.6--main"
    with SharedFileLock(a):
        with SharedFileLock(b, timeout=0.5):
            locks_while_both_held = sorted(p.name for p in tmp_path.glob("*.lock"))
    assert locks_while_both_held == [
        "org--model--v1.5--main.lock",
        "org--model--v1.6--main.lock",
    ]


def test_lock_holder_returns_none_for_a_missing_or_junk_file(tmp_path: Path) -> None:
    """A partially written or corrupt lock file must read as unheld, not crash."""
    assert lock_holder(tmp_path / "nope.lock") is None
    junk = tmp_path / "junk.lock"
    junk.write_text("not json")
    assert lock_holder(junk) is None


def _hold(path: str, seconds: float, acquired: MpEvent) -> None:
    """Acquire the lock on ``path``, signal readiness, then hold it for ``seconds``.

    Module-scoped so ``multiprocessing``'s ``spawn`` context can pickle it.
    ``spawn`` re-imports the interpreter and this module from scratch, which
    under load can itself take longer than a fixed sleep would assume — hence
    the explicit ``acquired`` handshake instead of a timed guess by the parent.

    Args:
        path: String form of the resource path (Path is not always picklable
            across the spawn boundary depending on platform).
        seconds: How long to sleep while holding the lock, after signaling.
        acquired: Set once the lock is actually held, so the parent can wait
            on a real event instead of guessing how long spawn + import takes.
    """
    from senselab.utils.file_lock import SharedFileLock

    with SharedFileLock(Path(path)):
        acquired.set()
        import time

        time.sleep(seconds)


def test_a_second_process_waits_rather_than_proceeding(tmp_path: Path) -> None:
    """Two live processes must serialise, not interleave.

    Every other test here is single-process and could pass against a lock that
    does nothing. This is the one that cannot.

    The child signals an ``Event`` right after it acquires, so the parent times
    its own acquire attempt from a real handshake rather than a fixed sleep —
    under load, a ``spawn``-started child re-importing the interpreter and this
    module can take much longer than any fixed guess to reach the acquire.
    """
    import multiprocessing as mp
    import time

    resource = tmp_path / "thing"
    ctx = mp.get_context("spawn")
    acquired = ctx.Event()

    proc = ctx.Process(target=_hold, args=(str(resource), 2.0, acquired))
    proc.start()
    signaled = acquired.wait(timeout=30)
    assert signaled, "child never acquired the lock within 30s"
    started = time.monotonic()
    with SharedFileLock(resource, timeout=10.0):
        waited = time.monotonic() - started
    proc.join(timeout=10)
    assert waited > 0.5, f"second acquirer did not wait (waited {waited:.2f}s)"


def test_timeout_tolerates_a_holder_without_taken_at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A holder file senselab did not write must still raise TimeoutError, not KeyError.

    Finding #7 of the #550 review: the Timeout branch read ``previous_holder["taken_at"]`` as a
    bare subscript while every neighbour used ``.get()``, so a payload lacking that key made
    ``__enter__`` raise ``KeyError`` — which no caller expects: the retry loops in
    ensure_hf_model / ensure_venv / record_resolution catch only ``TimeoutError``, and
    ``call_in_venv``'s FileRef locks catch nothing at all.

    The reachable source is a hand-edited lock file, or a foreign ``<resource>.lock`` on the
    FileRef path — *not* an older senselab, which never wrote such a payload (see the comment at
    the fix). The fabricated holder below is therefore a stand-in for a hand-edited file.
    """
    import multiprocessing as mp

    resource = tmp_path / "thing"
    ctx = mp.get_context("spawn")
    acquired = ctx.Event()
    proc = ctx.Process(target=_hold, args=(str(resource), 3.0, acquired))
    proc.start()
    try:
        assert acquired.wait(timeout=30), "child never acquired the lock within 30s"
        # A hand-edited holder payload: legible JSON, correct shape, no `taken_at`.
        monkeypatch.setattr(
            "senselab.utils.file_lock.lock_holder",
            lambda *_a, **_k: {"user": "alice", "host": "node1", "pid": 4211},
        )
        with pytest.raises(TimeoutError):
            with SharedFileLock(resource, timeout=0.5):
                pass
    finally:
        proc.join(timeout=10)
