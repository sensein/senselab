"""A lease by `O_EXCL` link and rename, one planted contract per invariant ADR-003 states."""

import errno
import json
import os
import stat
import threading
import time
from pathlib import Path

import pytest
from senselab_core.runtime import storage as storage_module
from senselab_core.runtime.locks import Lease, LeaseTimeout, LeaseUnavailable

pytestmark = pytest.mark.logic

JOIN_S = 10.0
OLD = 10**9


def _lease_file(resource: Path) -> Path:
    return Path(str(resource) + ".lease")


def _plant_lease(resource: Path, *, at: float = OLD) -> Path:
    lease = _lease_file(resource)
    lease.write_text(json.dumps({"user": "u", "host": "h", "pid": 1}))
    os.utime(lease, (at, at))
    return lease


def _join(threads: list[threading.Thread]) -> None:
    for thread in threads:
        thread.join(JOIN_S)
    assert not any(t.is_alive() for t in threads), "a thread outlived its bound"


def _snapshot(path: Path) -> tuple[bytes, int, int]:
    st = os.lstat(path)
    return path.read_bytes(), stat.S_IMODE(st.st_mode), st.st_mtime_ns


# -- taken by an O_EXCL link: one taker wins --


def test_of_concurrent_takers_exactly_one_holds_the_lease(tmp_path: Path) -> None:
    """Eight takers start together with no wait; exactly one holds the lease and the rest time out."""
    resource = tmp_path / "thing"
    takers = 8
    barrier, attempted = threading.Barrier(takers), threading.Barrier(takers)
    outcomes: list[str] = []

    def take() -> None:
        barrier.wait(JOIN_S)
        try:
            with Lease(resource, timeout=0):
                outcomes.append("held")
                attempted.wait(JOIN_S)
                return
        except LeaseTimeout:
            outcomes.append("timeout")
        attempted.wait(JOIN_S)

    threads = [threading.Thread(target=take) for _ in range(takers)]
    for thread in threads:
        thread.start()
    _join(threads)
    assert sorted(outcomes) == ["held"] + ["timeout"] * (takers - 1)


def test_a_waiter_takes_the_lease_once_it_is_released_and_breaks_nothing(tmp_path: Path) -> None:
    """A waiter blocks while a live lease is held, takes it after the release, and records no broken lease."""
    resource = tmp_path / "thing"
    held, release = threading.Event(), threading.Event()

    def hold() -> None:
        with Lease(resource, timeout=1):
            held.set()
            release.wait(JOIN_S)

    holder = threading.Thread(target=hold)
    holder.start()
    assert held.wait(JOIN_S)
    threading.Timer(0.3, release.set).start()
    started = time.monotonic()
    with Lease(resource, timeout=5) as waiter:
        waited = time.monotonic() - started
        assert waiter.owns() and waiter.broke == [] and waiter.set_aside == []
    _join([holder])
    assert waited >= 0.2


def test_a_link_whose_reply_is_lost_still_takes_the_lease(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A link made but reported as failed (a lost NFS reply) is judged by the link count: the lease is held."""
    real_link = os.link

    def link(*args: object, **kwargs: object) -> None:
        real_link(*args, **kwargs)  # type: ignore[arg-type]
        raise OSError(errno.EIO, "reply lost")

    monkeypatch.setattr(os, "link", link)
    with Lease(tmp_path / "thing", timeout=0) as lease:
        assert lease.owns()
    assert list(tmp_path.iterdir()) == []


def test_the_holder_identity_is_whole_whenever_the_lease_exists(tmp_path: Path) -> None:
    """The identity is written before the link, so a reader racing repeated takes never sees a partial one."""
    resource, lease_file = tmp_path / "thing", _lease_file(tmp_path / "thing")
    done = threading.Event()

    def churn() -> None:
        try:
            for _ in range(300):
                with Lease(resource, timeout=5):
                    pass
        finally:
            done.set()

    writer = threading.Thread(target=churn)
    writer.start()
    reads = partial = 0
    while not done.is_set():
        try:
            data = lease_file.read_bytes()
        except FileNotFoundError:
            continue
        reads += 1
        try:
            record = json.loads(data)
        except ValueError:
            record = None
        partial += not (isinstance(record, dict) and {"user", "host", "pid"} <= record.keys())
    _join([writer])
    assert partial == 0, f"{partial} of {reads} reads saw a partial identity"


# -- released by removing it --


def test_release_removes_the_lease_and_leaves_nothing_behind(tmp_path: Path) -> None:
    """After an uncontended take and release the directory is empty: no lease, no temporary, no clock."""
    with Lease(tmp_path / "thing") as lease:
        assert _lease_file(tmp_path / "thing").exists() and lease.owns()
    assert list(tmp_path.iterdir()) == []
    assert not lease.owns()


def test_a_release_never_removes_a_lease_another_holder_took(tmp_path: Path) -> None:
    """A holder whose lease was broken releases nothing: the new holder's lease stays and it still owns it."""
    resource = tmp_path / "thing"
    first = Lease(resource, timeout=0)
    first.__enter__()
    os.utime(_lease_file(resource), (OLD, OLD))
    with Lease(resource, timeout=1, stale_after=60) as second:
        assert not first.owns() and second.owns()
        first.__exit__(None, None, None)
        assert second.owns()


# -- judged stale from its timestamp on the storage clock --


def test_a_lease_older_than_stale_after_is_broken_and_taken(tmp_path: Path) -> None:
    """A lease stamped long ago is renamed away and removed, and the taker holds a fresh one."""
    resource = tmp_path / "thing"
    _plant_lease(resource)
    with Lease(resource, timeout=1, stale_after=60) as lease:
        assert lease.owns() and lease.broke == [_lease_file(resource)]
    assert sorted(p.name for p in tmp_path.iterdir()) == [".clock"]


def test_a_fresh_lease_is_broken_after_the_stale_bound_and_not_before(tmp_path: Path) -> None:
    """A lease stamped now is waited on until its age passes `stale_after`, then broken."""
    resource = tmp_path / "thing"
    _plant_lease(resource, at=time.time())
    started = time.monotonic()
    with Lease(resource, timeout=10, stale_after=0.5) as lease:
        waited = time.monotonic() - started
        assert lease.broke == [_lease_file(resource)]
    assert 0.3 <= waited < 5


def test_staleness_is_read_on_the_storage_clock_not_the_local_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A local clock an hour ahead does not age a fresh lease; a storage clock an hour ahead does."""
    resource = tmp_path / "thing"
    real_time = time.time
    with Lease(resource, timeout=0):
        monkeypatch.setattr(time, "time", lambda: real_time() + 3600)
        with pytest.raises(LeaseTimeout):
            with Lease(resource, timeout=0.2, stale_after=60):
                pass
        monkeypatch.setattr(time, "time", real_time)
        real_now = storage_module.StorageDir.fs_now
        monkeypatch.setattr(storage_module.StorageDir, "fs_now", lambda self: real_now(self) + 3600)
        with Lease(resource, timeout=1, stale_after=60) as breaker:
            assert breaker.broke == [_lease_file(resource)]


# -- a stale lease is broken by rename, so two breakers cannot both win --


def test_a_breaker_that_judged_an_old_lease_never_removes_the_new_one(tmp_path: Path) -> None:
    """When another breaker took the name between a judgement and its rename, the new lease is put back."""
    resource = tmp_path / "thing"
    planted = os.lstat(_plant_lease(resource))
    late = Lease(resource, timeout=0, stale_after=60)
    with Lease(resource, timeout=1, stale_after=60) as winner:
        assert winner.broke
        assert late._take_aside((planted.st_dev, planted.st_ino)) is False
        assert winner.owns()
    late._close()
    assert [p.name for p in tmp_path.iterdir() if p.name != ".clock"] == []


def test_concurrent_breakers_remove_one_stale_lease_once_and_a_displaced_holder_knows(tmp_path: Path) -> None:
    """Eight takers break one stale lease together: it is removed once, and holders overlap only when one was displaced.

    A late breaker can move a newer lease before linking it back (ADR-003 guarantees no exactly-once); the
    holder it displaced then sees `owns()` false, so the overlap is never silent.
    """
    resource = tmp_path / "thing"
    _plant_lease(resource)
    takers = 8
    barrier = threading.Barrier(takers)
    lock = threading.Lock()
    inside, most, broke, displaced = [0], [0], [0], [0]

    def take() -> None:
        barrier.wait(JOIN_S)
        with Lease(resource, timeout=5, stale_after=60) as lease:
            with lock:
                inside[0] += 1
                most[0] = max(most[0], inside[0])
                broke[0] += len(lease.broke)
            time.sleep(0.02)
            with lock:
                inside[0] -= 1
                displaced[0] += not lease.owns()

    threads = [threading.Thread(target=take) for _ in range(takers)]
    for thread in threads:
        thread.start()
    _join(threads)
    assert broke[0] == 1
    assert most[0] == 1 or displaced[0] >= 1, (most, displaced)
    assert [p.name for p in tmp_path.iterdir() if p.name != ".clock"] == []


# -- every wait is bounded --


def test_a_wait_on_a_live_lease_ends_at_its_timeout(tmp_path: Path) -> None:
    """A taker with a 0.3 s timeout against a live lease raises `LeaseTimeout` naming the holder, near 0.3 s."""
    resource = tmp_path / "thing"
    with Lease(resource, timeout=0) as holder:
        started = time.monotonic()
        with pytest.raises(LeaseTimeout, match=r"thing\.lease, held by .*'pid'"):
            with Lease(resource, timeout=0.3, stale_after=3600):
                pass
        assert time.monotonic() - started < 2 and holder.owns()


@pytest.mark.parametrize(("timeout", "stale_after"), [(-1.0, 60.0), (1.0, -1.0)])
def test_a_negative_bound_is_refused(tmp_path: Path, timeout: float, stale_after: float) -> None:
    """There is no unbounded wait: a negative timeout or stale bound is a `ValueError`."""
    with pytest.raises(ValueError, match="not negative"):
        Lease(tmp_path / "thing", timeout=timeout, stale_after=stale_after)


# -- path safety under a storage root (2026-10-31 amendment) --


@pytest.mark.parametrize("plant", ["symlink", "dangling", "directory", "fifo"])
def test_a_wrong_kind_at_the_lease_name_is_moved_aside_and_its_target_untouched(tmp_path: Path, plant: str) -> None:
    """A symlink, directory or fifo at the lease name is quarantined beside it; a symlink's target is not touched."""
    root, victim = tmp_path / "root", tmp_path / "victim"
    root.mkdir()
    victim.mkdir()
    (victim / "f").write_bytes(b"belongs to someone else")
    before = _snapshot(victim / "f")
    lease_file = _lease_file(root / "thing")
    if plant == "symlink":
        os.symlink(victim / "f", lease_file)
    elif plant == "dangling":
        os.symlink(victim / "created-outside", lease_file)
    elif plant == "directory":
        lease_file.mkdir()
    else:
        os.mkfifo(lease_file)
    with Lease(root / "thing", timeout=1) as lease:
        assert lease.owns() and len(lease.set_aside) == 1 and os.path.lexists(lease.set_aside[0])
    assert _snapshot(victim / "f") == before and sorted(p.name for p in victim.iterdir()) == ["f"]


def test_a_hard_link_at_the_lease_name_is_broken_by_name_and_its_file_untouched(tmp_path: Path) -> None:
    """An old hard link at the lease name is renamed and unlinked; the file it shares keeps its bytes, mode and time."""
    root, victim = tmp_path / "root", tmp_path / "victim"
    root.mkdir()
    victim.mkdir()
    (victim / "f").write_bytes(b"belongs to someone else")
    os.chmod(victim / "f", 0o600)
    os.utime(victim / "f", (OLD, OLD))
    before = _snapshot(victim / "f")
    os.link(victim / "f", _lease_file(root / "thing"))
    with Lease(root / "thing", timeout=1, stale_after=60) as lease:
        assert lease.owns()
    assert _snapshot(victim / "f") == before


@pytest.mark.parametrize("umask", [0o000, 0o077])
def test_the_lease_mode_is_set_through_its_descriptor_whatever_the_umask(tmp_path: Path, umask: int) -> None:
    """The lease file has the mode the caller asks for under any umask."""
    previous = os.umask(umask)
    try:
        with Lease(tmp_path / "thing", file_mode=0o660):
            mode = stat.S_IMODE(os.lstat(_lease_file(tmp_path / "thing")).st_mode)
    finally:
        os.umask(previous)
    assert mode == 0o660


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes in a read-only directory")
def test_a_directory_that_refuses_the_lease_is_a_typed_error_naming_its_owner(tmp_path: Path) -> None:
    """A lease that cannot be created is `LeaseUnavailable` naming the lease and who can fix the directory."""
    root = tmp_path / "root"
    root.mkdir()
    os.chmod(root, 0o555)
    try:
        with pytest.raises(LeaseUnavailable, match=r"thing\.lease.*owned by"):
            with Lease(root / "thing", timeout=1):
                pass
    finally:
        os.chmod(root, 0o755)
