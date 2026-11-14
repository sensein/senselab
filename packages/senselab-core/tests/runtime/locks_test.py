"""A lease is a directory taken by `mkdir` and renewed by beats, one planted contract per invariant ADR-003 states."""

import contextlib
import errno
import multiprocessing
import os
import shutil
import stat
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import lease_stress_fixture
import pytest
from lease_clock_fixture import ManualClock, age
from senselab_core.runtime.locks import GONE_PREFIX, RENEWER_PREFIX, Lease, LeaseTimeout, LeaseUnavailable
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.storage import StorageUnsupported

POLL_S = ExecutionProfile.packaged().lease.poll_s

pytestmark = pytest.mark.logic

JOIN_S = 10.0
OLD = 10**9
NFS_ENV = "SENSELAB_WORK_POOL_NFS_DIR"
UUID_LEN = 32


@pytest.fixture(params=["local", "nfs"])
def lease_dir(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Path]:
    """A scratch directory: `tmp_path`, or a fresh one under `SENSELAB_WORK_POOL_NFS_DIR` (skipped when unset)."""
    if request.param == "local":
        yield tmp_path
        return
    base = os.environ.get(NFS_ENV, "")
    if not base:
        pytest.skip(f"{NFS_ENV} is not set")
    root = Path(base) / f"lease-{uuid.uuid4().hex[:12]}"
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _lease_dir(resource: Path) -> Path:
    return Path(str(resource) + ".lease")


def _plant_lease(resource: Path, *, at: float = OLD) -> Path:
    """Plant a dead holder's lease, its token directory and the lease stamped `at`; return the token directory."""
    token = _lease_dir(resource) / uuid.uuid4().hex
    token.mkdir(parents=True)
    for path in (token, token.parent):
        os.utime(path, (at, at))
    return token


def _join(threads: list[threading.Thread]) -> None:
    for thread in threads:
        thread.join(JOIN_S)
    assert not any(t.is_alive() for t in threads), "a thread outlived its bound"


def _renewers() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith(RENEWER_PREFIX)]


def _snapshot(path: Path) -> tuple[bytes, int, int]:
    st = os.lstat(path)
    return path.read_bytes(), stat.S_IMODE(st.st_mode), st.st_mtime_ns


# -- taken by mkdir: one taker wins --


def test_of_concurrent_takers_exactly_one_holds_the_lease(lease_dir: Path) -> None:
    """Eight takers start together with no wait; exactly one holds the lease and the rest time out."""
    resource = lease_dir / "thing"
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


def test_a_waiter_takes_the_lease_once_it_is_released_and_breaks_nothing(
    tmp_path: Path, lease_clock: Callable[..., ManualClock]
) -> None:
    """A waiter blocks while a live lease is held, takes it after the release, and records no broken lease."""
    resource = tmp_path / "thing"
    held, release = threading.Event(), threading.Event()

    def hold() -> None:
        with Lease(resource, timeout=0):
            held.set()
            release.wait(JOIN_S)

    def release_after_two(_: float) -> None:
        if len(clock.slept) == 2:  # noqa: PLR2004 - the second poll
            release.set()
            holder.join(JOIN_S)

    holder = threading.Thread(target=hold)
    holder.start()
    assert held.wait(JOIN_S)
    clock = lease_clock(on_sleep=release_after_two)
    with Lease(resource, timeout=5, clock=clock) as waiter:
        assert waiter.owns() and [b.path for b in waiter.broke] == [] and waiter.set_aside == []
    assert clock.slept == [POLL_S] * 2
    _join([holder])


def test_lost_mkdir_reply_with_own_token_is_held(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A token `mkdir` made but answered `EEXIST` (a lost NFS reply, retransmitted) is the holder's own."""
    real_mkdir, lease_made = os.mkdir, []

    def mkdir(path: Any, mode: int = 0o777, *, dir_fd: int | None = None) -> None:  # noqa: ANN401
        real_mkdir(path, mode, dir_fd=dir_fd)
        name = os.fsdecode(path)
        if name.endswith(".lease"):
            lease_made.append(name)
        elif lease_made and len(name) == UUID_LEN:
            lease_made.clear()
            raise FileExistsError(errno.EEXIST, "reply lost; the retransmission found it", name)

    monkeypatch.setattr(os, "mkdir", mkdir)
    with Lease(tmp_path / "thing", timeout=0) as lease:
        assert lease.owns()
    assert list(tmp_path.iterdir()) == []


def test_a_taker_that_finds_another_token_beside_its_own_withdraws(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lease that holds another token by the time this taker's token is made is not taken, and is left as found."""
    real_mkdir, other = os.mkdir, uuid.uuid4().hex

    def mkdir(path: Any, mode: int = 0o777, *, dir_fd: int | None = None) -> None:  # noqa: ANN401
        real_mkdir(path, mode, dir_fd=dir_fd)
        if os.fsdecode(path).endswith(".lease"):
            real_mkdir(other, 0o700, dir_fd=os.open(path, os.O_RDONLY, dir_fd=dir_fd))

    monkeypatch.setattr(os, "mkdir", mkdir)
    with pytest.raises(LeaseTimeout):
        with Lease(tmp_path / "thing", timeout=0, stale_after=60):
            pass
    assert [p.name for p in _lease_dir(tmp_path / "thing").iterdir()] == [other]


# -- renewed by beats, from a thread the lease owns --


def test_keep_alive_renews_while_the_work_writes_nothing(
    tmp_path: Path, lease_clock: Callable[..., ManualClock]
) -> None:
    """While the work is idle the renewer beats: `<token>/` is restamped, one beat is kept, and the thread ends."""
    resource = tmp_path / "thing"
    clock = lease_clock()
    with Lease(resource, timeout=0, stale_after=1.0, clock=clock) as lease, lease.keep_alive(0.1) as state:
        token = next(_lease_dir(resource).iterdir())
        clock.settle()
        for beat in range(1, 4):
            age(tmp_path, 1)
            aged = os.lstat(token).st_mtime_ns
            clock.advance(0.1)
            clock.settle()
            assert state.beats == beat and os.lstat(token).st_mtime_ns > aged
        assert len(list(token.iterdir())) == 1 and not state.lost and lease.owns()
    _join(_renewers())
    assert _renewers() == [] and list(tmp_path.iterdir()) == []


def test_beat_after_break_marks_lost(tmp_path: Path, lease_clock: Callable[..., ManualClock]) -> None:
    """A beat that finds the lease gone (`ENOENT`) marks it lost; the release removes nothing; no thread is left."""
    resource = tmp_path / "thing"
    clock = lease_clock()
    with Lease(resource, timeout=0, stale_after=1.0, clock=clock) as lease, lease.keep_alive(0.1) as state:
        clock.settle()
        clock.advance(0.1)
        clock.settle()
        assert state.beats == 1
        aside = tmp_path / uuid.uuid4().hex
        os.rename(_lease_dir(resource), aside)
        shutil.rmtree(aside)
        clock.advance(0.1)
        assert state.wait_lost(JOIN_S) and not lease.owns()
        _join(_renewers())
        assert _renewers() == []
        with Lease(resource, timeout=0) as other:
            assert other.owns()
    _join(_renewers())
    assert _renewers() == [] and list(tmp_path.iterdir()) == []


def test_keep_alive_needs_a_held_lease_and_a_fraction_below_one(tmp_path: Path) -> None:
    """A renewer is refused on a lease not held, and a renew fraction outside (0, 1) is a `ValueError`."""
    lease = Lease(tmp_path / "thing")
    with pytest.raises(RuntimeError, match="held lease"):
        with lease.keep_alive(0.3):
            pass
    with lease, pytest.raises(ValueError, match=r"\(0, 1\)"):
        with lease.keep_alive(1.0):
            pass


# -- judged stale on the storage clock --


def test_dead_holder_lease_broken_after_bound(tmp_path: Path, lease_clock: Callable[..., ManualClock]) -> None:
    """A holder that stops beating is not broken 59 s on the storage clock after its last stamp, and is at 61 s."""
    resource = tmp_path / "thing"
    dead = _plant_lease(resource).name
    clock = lease_clock(storage_origin=OLD)
    clock.advance(59)
    with pytest.raises(LeaseTimeout):
        with Lease(resource, timeout=0, stale_after=60, clock=clock):
            pass
    assert [p.name for p in _lease_dir(resource).iterdir()] == [dead]
    clock.advance(2)
    with Lease(resource, timeout=0, stale_after=60, clock=clock) as breaker:
        assert [b.path for b in breaker.broke] == [_lease_dir(resource)] and breaker.owns()
    assert list(tmp_path.iterdir()) == []


def test_breaker_never_breaks_a_beating_lease(tmp_path: Path, lease_clock: Callable[..., ManualClock]) -> None:
    """Over 100 s on the storage clock, more than three stale bounds, a beating lease is never broken."""
    resource = tmp_path / "thing"
    clock = lease_clock()
    with Lease(resource, timeout=0, stale_after=30, clock=clock) as holder, holder.keep_alive(0.01) as state:
        clock.settle()
        for _ in range(10):
            age(tmp_path, 10)
            clock.advance(10)
            clock.settle()
            with pytest.raises(LeaseTimeout):
                with Lease(resource, timeout=0, stale_after=30):
                    pass
        assert holder.owns() and not state.lost
    _join(_renewers())
    assert _renewers() == []


def test_staleness_is_read_on_the_storage_clock_not_the_local_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A local clock an hour ahead does not age a fresh lease; an hour on the storage clock does."""
    resource = tmp_path / "thing"
    real_time = time.time
    with Lease(resource, timeout=0):
        monkeypatch.setattr(time, "time", lambda: real_time() + 3600)
        with pytest.raises(LeaseTimeout):
            with Lease(resource, timeout=0, stale_after=60):
                pass
        monkeypatch.setattr(time, "time", real_time)
        age(tmp_path, 3600)
        with Lease(resource, timeout=0, stale_after=60) as breaker:
            assert [b.path for b in breaker.broke] == [_lease_dir(resource)]


def test_a_lease_older_than_stale_after_is_broken_and_taken(tmp_path: Path) -> None:
    """A lease stamped long ago is renamed away and removed, the taker holds a fresh one, and nothing is left."""
    resource = tmp_path / "thing"
    _plant_lease(resource)
    with Lease(resource, timeout=1, stale_after=60) as lease:
        assert lease.owns() and [b.path for b in lease.broke] == [_lease_dir(resource)]
    assert list(tmp_path.iterdir()) == []


def test_an_empty_lease_a_taker_left_is_judged_by_its_own_stamp(tmp_path: Path) -> None:
    """A lease directory with no token, as a taker that died between its two `mkdir`s leaves, is broken when stale."""
    lease_dir = _lease_dir(tmp_path / "thing")
    lease_dir.mkdir()
    os.utime(lease_dir, (OLD, OLD))
    with Lease(tmp_path / "thing", timeout=1, stale_after=60) as lease:
        assert lease.owns() and [b.path for b in lease.broke] == [lease_dir]


def test_a_lease_a_claimant_died_in_is_claimed_and_broken_when_stale(tmp_path: Path) -> None:
    """A lease left holding only a dead claimant's `gone-<token>` is claimed by renaming that entry, then broken."""
    lease_dir = _lease_dir(tmp_path / "thing")
    claimed = lease_dir / f"gone-{uuid.uuid4().hex}"
    claimed.mkdir(parents=True)
    for path in (claimed, lease_dir):
        os.utime(path, (OLD, OLD))
    with Lease(tmp_path / "thing", timeout=1, stale_after=60) as lease:
        assert lease.owns() and [b.path for b in lease.broke] == [lease_dir]
    assert list(tmp_path.iterdir()) == []


def test_a_fresh_taker_after_a_crash_is_never_displaced_by_a_late_breaker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A late breaker that judged a crashed, empty lease never removes the fresh lease that replaced it.

    Before the late breaker acts, another breaks the empty lease and takes the name; the late breaker's `lstat` of
    the name is then served from a cache, as an NFS client serves it, and still shows the old lease.
    """
    resource, real_lstat, real_listdir = tmp_path / "thing", os.lstat, os.listdir
    lease_dir = _lease_dir(resource)
    lease_dir.mkdir()
    os.utime(lease_dir, (OLD, OLD))
    winner = Lease(resource, timeout=1, stale_after=60)
    last: list[os.stat_result] = []
    cached: list[os.stat_result] = []
    listings, running = [0], [False]

    def lstat(path: Any, *, dir_fd: int | None = None) -> os.stat_result:  # noqa: ANN401
        if os.fsdecode(path) == "thing.lease" and cached and not running[0]:
            return cached[0]
        st = real_lstat(path, dir_fd=dir_fd)
        if os.fsdecode(path) == "thing.lease":
            last[:] = [st]
        return st

    def listdir(path: Any = ".") -> list[str]:  # noqa: ANN401
        names = real_listdir(path)
        if isinstance(path, int) and not running[0] and not cached:
            listings[0] += 1
            if listings[0] == 2:  # noqa: PLR2004 - the end of the late breaker's second, fresh read of the lease
                running[0] = True
                winner.__enter__()
                running[0] = False
                cached.append(last[0])
        return names

    monkeypatch.setattr(os, "listdir", listdir)
    monkeypatch.setattr(os, "lstat", lstat)
    late = Lease(resource, timeout=0, stale_after=60)
    with pytest.raises(LeaseTimeout):
        late.__enter__()
    monkeypatch.setattr(os, "lstat", real_lstat)
    monkeypatch.setattr(os, "listdir", real_listdir)
    assert [b.path for b in winner.broke] == [lease_dir] and [b.path for b in late.broke] == [] and winner.owns()
    winner.__exit__(None, None, None)
    assert list(tmp_path.iterdir()) == []


# -- broken and released by rename --


def test_a_release_never_removes_a_lease_another_holder_took(tmp_path: Path) -> None:
    """A holder whose lease was broken releases nothing: the new holder's lease stays and it still owns it."""
    resource = tmp_path / "thing"
    first = Lease(resource, timeout=0)
    first.__enter__()
    for path in (*_lease_dir(resource).iterdir(), _lease_dir(resource)):
        os.utime(path, (OLD, OLD))
    with Lease(resource, timeout=1, stale_after=60) as second:
        assert not first.owns() and second.owns()
        first.__exit__(None, None, None)
        assert second.owns()


def test_a_late_breaker_cannot_claim_a_lease_another_breaker_already_broke(tmp_path: Path) -> None:
    """A breaker claims the lease it judged by its token; once another broke it, the claim fails; the new one stays."""
    resource = tmp_path / "thing"
    judged = frozenset({_plant_lease(resource).name})
    late = Lease(resource, timeout=0, stale_after=60)
    with Lease(resource, timeout=1, stale_after=60) as winner:
        assert winner.broke and winner.owns()
        assert late._break(judged, frozenset()) is False
        assert winner.owns()
    late._close()
    assert list(tmp_path.iterdir()) == []


def test_concurrent_breakers_remove_one_stale_lease_once_and_displace_no_holder(
    lease_dir: Path, lease_clock: Callable[..., ManualClock]
) -> None:
    """Eight takers break one stale lease together: it is removed once, each holds in turn, and none is displaced.

    None times out waiting on an unknown holder (an orphaned lease).
    """
    resource = lease_dir / "thing"
    _plant_lease(resource)
    takers = 8
    barrier = threading.Barrier(takers)
    lock = threading.Lock()
    inside, most, broke, displaced = [0], [0], [0], [0]
    timeouts: list[str] = []

    def take() -> None:
        barrier.wait(JOIN_S)
        try:
            with Lease(resource, timeout=5, stale_after=60, clock=lease_clock(spends=False)) as lease:
                with lock:
                    inside[0] += 1
                    most[0] = max(most[0], inside[0])
                    broke[0] += len(lease.broke)
                with lock:
                    inside[0] -= 1
                    displaced[0] += not lease.owns()
        except LeaseTimeout as err:
            timeouts.append(str(err))

    threads = [threading.Thread(target=take) for _ in range(takers)]
    for thread in threads:
        thread.start()
    _join(threads)
    assert timeouts == []
    assert broke[0] == 1 and most[0] == 1 and displaced[0] == 0, (most, displaced)
    assert list(lease_dir.iterdir()) == []


# -- nothing is written under the lease --


def test_no_file_is_written_under_the_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lease_clock: Callable[..., ManualClock]
) -> None:
    """Taking, beating, judging and releasing a lease create no file: only directories, under any name."""
    resource, real_open, created = tmp_path / "thing", os.open, []

    def opened(path: Any, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:  # noqa: ANN401
        if flags & os.O_CREAT:
            created.append(os.fsdecode(path))
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", opened)
    clock = lease_clock()
    with Lease(resource, timeout=0, stale_after=1.0, clock=clock) as lease, lease.keep_alive(0.1) as state:
        clock.settle()
        for _ in range(2):
            clock.advance(0.1)
            clock.settle()
        assert state.beats == 2  # noqa: PLR2004
        with pytest.raises(LeaseTimeout):
            with Lease(resource, timeout=0.2, stale_after=1.0, clock=clock):
                pass
        assert [f for _, _, files in os.walk(_lease_dir(resource)) for f in files] == []
    assert created == [] and list(tmp_path.iterdir()) == []


# -- every wait is bounded --


def test_a_wait_on_a_live_lease_ends_at_its_timeout(tmp_path: Path, lease_clock: Callable[..., ManualClock]) -> None:
    """A taker with a 0.3 s timeout against a live lease raises `LeaseTimeout` naming the holder's token."""
    resource = tmp_path / "thing"
    clock = lease_clock()
    with Lease(resource, timeout=0) as holder:
        with pytest.raises(LeaseTimeout, match=r"thing\.lease, held by .*'token'"):
            with Lease(resource, timeout=0.3, stale_after=3600, clock=clock):
                pass
        assert holder.owns()
    assert clock.now() == pytest.approx(0.3)


@pytest.mark.parametrize(("timeout", "stale_after"), [(-1.0, 60.0), (1.0, -1.0)])
def test_a_negative_bound_is_refused(tmp_path: Path, timeout: float, stale_after: float) -> None:
    """There is no unbounded wait: a negative timeout or stale bound is a `ValueError`."""
    with pytest.raises(ValueError, match="not negative"):
        Lease(tmp_path / "thing", timeout=timeout, stale_after=stale_after)


# -- the root and path safety under a storage root --


def test_root_without_atomic_mkdir_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A root whose filesystem refuses `mkdir` as unsupported (an object store) is `StorageUnsupported`."""

    def mkdir(path: Any, mode: int = 0o777, *, dir_fd: int | None = None) -> None:  # noqa: ANN401
        raise OSError(errno.ENOTSUP, "Operation not supported", os.fsdecode(path))

    monkeypatch.setattr(os, "mkdir", mkdir)
    with pytest.raises(StorageUnsupported, match="atomic mkdir"):
        with Lease(tmp_path / "thing", timeout=1):
            pass


@pytest.mark.parametrize("plant", ["symlink", "dangling", "file", "fifo"])
def test_a_wrong_kind_at_the_lease_name_is_moved_aside_and_its_target_untouched(tmp_path: Path, plant: str) -> None:
    """A symlink, file or fifo at the lease name is quarantined beside it; a symlink's target is not touched."""
    root, victim = tmp_path / "root", tmp_path / "victim"
    root.mkdir()
    victim.mkdir()
    (victim / "f").write_bytes(b"belongs to someone else")
    before = _snapshot(victim / "f")
    lease_dir = _lease_dir(root / "thing")
    if plant == "symlink":
        os.symlink(victim, lease_dir)
    elif plant == "dangling":
        os.symlink(victim / "created-outside", lease_dir)
    elif plant == "file":
        lease_dir.write_bytes(b'{"user": "u", "host": "h", "pid": 1}')
    else:
        os.mkfifo(lease_dir)
    with Lease(root / "thing", timeout=1) as lease:
        assert lease.owns() and len(lease.set_aside) == 1 and os.path.lexists(lease.set_aside[0])
    assert _snapshot(victim / "f") == before and sorted(p.name for p in victim.iterdir()) == ["f"]


def test_a_hard_link_at_the_lease_name_is_moved_aside_by_name_and_its_file_untouched(tmp_path: Path) -> None:
    """A hard link at the lease name is renamed aside; the file it shares keeps its bytes, mode and time."""
    root, victim = tmp_path / "root", tmp_path / "victim"
    root.mkdir()
    victim.mkdir()
    (victim / "f").write_bytes(b"belongs to someone else")
    os.chmod(victim / "f", 0o600)
    os.utime(victim / "f", (OLD, OLD))
    before = _snapshot(victim / "f")
    os.link(victim / "f", _lease_dir(root / "thing"))
    with Lease(root / "thing", timeout=1, stale_after=60) as lease:
        assert lease.owns()
    assert _snapshot(victim / "f") == before


@pytest.mark.parametrize("umask", [0o000, 0o077])
def test_the_lease_mode_is_set_through_its_descriptor_whatever_the_umask(tmp_path: Path, umask: int) -> None:
    """The lease directory and its token directory have the mode the caller asks for under any umask."""
    previous = os.umask(umask)
    try:
        with Lease(tmp_path / "thing", mode=0o770):
            lease_dir = _lease_dir(tmp_path / "thing")
            modes = {stat.S_IMODE(os.lstat(p).st_mode) for p in (lease_dir, *lease_dir.iterdir())}
    finally:
        os.umask(previous)
    assert modes == {0o770}


def test_an_empty_lease_this_user_cannot_list_is_removed_and_taken(tmp_path: Path) -> None:
    """An empty `0o000` lease, as a crash between a lease's `mkdir` and its mode change leaves, is removed (`rmdir`)."""
    lease_dir = _lease_dir(tmp_path / "thing")
    lease_dir.mkdir()
    os.utime(lease_dir, (OLD, OLD))
    os.chmod(lease_dir, 0o000)
    try:
        with Lease(tmp_path / "thing", timeout=1, stale_after=60) as lease:
            assert lease.owns() and [b.path for b in lease.broke] == [lease_dir]
    finally:
        for path in tmp_path.iterdir():
            os.chmod(path, 0o700)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root lists a 0o000 directory")
def test_a_stale_lease_this_user_cannot_list_is_a_typed_error_naming_its_owner(tmp_path: Path) -> None:
    """A stale lease another user left unlistable cannot be claimed, so it is not broken: the error names its owner."""
    resource = tmp_path / "thing"
    _plant_lease(resource)
    os.chmod(_lease_dir(resource), 0o000)
    try:
        with pytest.raises(LeaseUnavailable, match=r"cannot be listed.*owned by"):
            with Lease(resource, timeout=1, stale_after=60):
                pass
        assert _lease_dir(resource).exists()
    finally:
        for path in tmp_path.iterdir():
            os.chmod(path, 0o700)


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


# -- fix round: interleavings forced at the `os` boundary --


def test_takers_that_both_withdraw_leave_no_orphan_at_the_name(  # noqa: C901 - the seams of one interleaving
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A's lease is renamed away by a late breaker, B makes a new one, A's token lands beside B's, both withdraw.

    The emptied lease is removed, so a third taker holds at once instead of waiting `stale_after` on an orphan.
    """
    resource, real_mkdir, real_listdir = tmp_path / "thing", os.mkdir, os.listdir
    b_go, b_made = threading.Event(), threading.Event()
    both_listed = threading.Barrier(2, timeout=JOIN_S)
    calls, listed = [0], [0]
    lock = threading.Lock()

    def mkdir(path: Any, mode: int = 0o777, *, dir_fd: int | None = None) -> None:  # noqa: ANN401
        if os.fsdecode(path) != "thing.lease":
            real_mkdir(path, mode, dir_fd=dir_fd)
            return
        with lock:
            calls[0] += 1
            n = calls[0]
        real_mkdir(path, mode, dir_fd=dir_fd)
        if n == 1:
            os.rename(path, uuid.uuid4().hex, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
            b_go.set()
            assert b_made.wait(JOIN_S)
        elif n == 2:  # noqa: PLR2004 - the second taker
            b_made.set()

    def listdir(path: Any = ".") -> list[str]:  # noqa: ANN401
        names = real_listdir(path)
        with lock:
            armed = isinstance(path, int) and calls[0] == 2 and listed[0] < 2  # noqa: PLR2004 - the two takers
            listed[0] += armed
        if armed:
            with contextlib.suppress(threading.BrokenBarrierError):
                both_listed.wait()
            names = real_listdir(path)
        return names

    monkeypatch.setattr(os, "mkdir", mkdir)
    monkeypatch.setattr(os, "listdir", listdir)
    outcomes: dict[str, str] = {}

    def take(who: str, wait: threading.Event | None) -> None:
        if wait is not None:
            assert wait.wait(JOIN_S)
        try:
            with Lease(resource, timeout=0, stale_after=60):
                outcomes[who] = "held"
        except LeaseTimeout:
            outcomes[who] = "timeout"

    threads = [threading.Thread(target=take, args=("A", None)), threading.Thread(target=take, args=("B", b_go))]
    for thread in threads:
        thread.start()
    _join(threads)
    monkeypatch.setattr(os, "listdir", real_listdir)
    assert sorted(outcomes.values()) in (["timeout", "timeout"], ["held", "timeout"]), outcomes
    with Lease(resource, timeout=0, stale_after=60) as third:
        assert third.owns()


def test_a_breaker_never_renames_a_lease_other_than_the_one_it_judged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Between a late breaker's fresh judgement and its rename another breaks and takes; the new lease is left alone."""
    resource, real_lstat = tmp_path / "thing", os.lstat
    planted = _plant_lease(resource).name
    winner = Lease(resource, timeout=1, stale_after=60)
    seen, running = [0], [False]

    def lstat(path: Any, *, dir_fd: int | None = None) -> os.stat_result:  # noqa: ANN401
        st = real_lstat(path, dir_fd=dir_fd)
        if os.fsdecode(path) == planted and not running[0]:
            seen[0] += 1
            if seen[0] == 2:  # noqa: PLR2004 - the end of the late breaker's second, fresh read of the stale lease
                running[0] = True
                winner.__enter__()
        return st

    monkeypatch.setattr(os, "lstat", lstat)
    late = Lease(resource, timeout=0, stale_after=60)
    with pytest.raises(LeaseTimeout):
        late.__enter__()
    assert (
        [b.path for b in winner.broke] == [_lease_dir(resource)]
        and [b.path for b in late.broke] == []
        and winner.owns()
    )
    winner.__exit__(None, None, None)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("where", ["beat", "owns"])
def test_a_stale_nfs_handle_is_a_lost_lease_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str, lease_clock: Callable[..., ManualClock]
) -> None:
    """`ESTALE` on a beat marks the lease lost at that beat, not after `stale_after`; on `owns()` it reads as False."""
    real_mkdir, real_open = os.mkdir, os.open

    def mkdir(path: Any, mode: int = 0o777, *, dir_fd: int | None = None) -> None:  # noqa: ANN401
        if where == "beat" and os.fsdecode(path).startswith("beat-"):
            raise OSError(errno.ESTALE, "Stale file handle", os.fsdecode(path))
        real_mkdir(path, mode, dir_fd=dir_fd)

    def opened(path: Any, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:  # noqa: ANN401
        if where == "owns" and armed and os.fsdecode(path) == "thing.lease":
            raise OSError(errno.ESTALE, "Stale file handle", os.fsdecode(path))
        return real_open(path, flags, mode, dir_fd=dir_fd)

    armed = False
    monkeypatch.setattr(os, "mkdir", mkdir)
    monkeypatch.setattr(os, "open", opened)
    clock = lease_clock()
    with Lease(tmp_path / "thing", timeout=0, stale_after=60, clock=clock) as lease, lease.keep_alive(0.005) as state:
        if where == "beat":
            clock.settle()
            clock.advance(0.3)
            assert state.wait_lost(JOIN_S)
        else:
            armed = True
            assert lease.owns() is False and lease.prove() is False
            armed = False
    _join(_renewers())
    assert _renewers() == []


def test_prove_is_a_fresh_mkdir_that_fails_once_the_lease_is_gone(tmp_path: Path) -> None:
    """`prove` makes and removes a directory in the token; it is True while held and False once the lease is gone."""
    resource = tmp_path / "thing"
    with Lease(resource, timeout=0) as lease:
        assert lease.prove()
        token = next(_lease_dir(resource).iterdir())
        assert list(token.iterdir()) == []
        aside = tmp_path / uuid.uuid4().hex
        _lease_dir(resource).rename(aside)
        shutil.rmtree(aside)
        assert not lease.prove()


def test_a_renewer_ended_by_any_error_marks_the_lease_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lease_clock: Callable[..., ManualClock]
) -> None:
    """An exception that is not an `OSError` ends the renewer, and the lease is then marked lost, not left unrenewed."""
    real_mkdir = os.mkdir

    def mkdir(path: Any, mode: int = 0o777, *, dir_fd: int | None = None) -> None:  # noqa: ANN401
        if os.fsdecode(path).startswith("beat-"):
            raise MemoryError("a beat ran out of memory")
        real_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "mkdir", mkdir)
    clock = lease_clock()
    with Lease(tmp_path / "thing", timeout=0, stale_after=60, clock=clock) as lease, lease.keep_alive(0.005) as state:
        clock.settle()
        clock.advance(0.3)
        assert state.wait_lost(JOIN_S)
    _join(_renewers())
    assert _renewers() == []


def test_a_hung_beat_does_not_hold_up_the_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lease_clock: Callable[..., ManualClock]
) -> None:
    """A beat that never returns does not hold up the release; the renewer ends when the beat returns."""
    real_mkdir, hung, unhang = os.mkdir, threading.Event(), threading.Event()

    def mkdir(path: Any, mode: int = 0o777, *, dir_fd: int | None = None) -> None:  # noqa: ANN401
        if os.fsdecode(path) == "beat-2":
            hung.set()
            unhang.wait(JOIN_S)
        real_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "mkdir", mkdir)
    clock = lease_clock()
    with Lease(tmp_path / "thing", timeout=0, stale_after=1, clock=clock) as lease, lease.keep_alive(0.001) as state:
        clock.settle()
        clock.advance(0.001)
        clock.settle()
        assert state.beats == 1
        clock.advance(0.001)
        assert hung.wait(JOIN_S)
    assert not lease.owns() and len(_renewers()) == 1
    unhang.set()
    for thread in _renewers():
        thread.join(JOIN_S)
    assert _renewers() == []


def test_a_judge_listing_during_a_claim_ages_the_lease_fresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A judge that lists the lease in the middle of a claim ages it fresh, and the claimant breaks it once.

    The judge lists right after the claimant's rename, with the lease's own stat served from a cache as an NFS client
    serves it; the claimed entry was stamped before its rename, so the judge does not claim it again.
    """
    resource, real_rename, real_lstat = tmp_path / "thing", os.rename, os.lstat
    planted = _plant_lease(resource).name
    cached = [real_lstat(_lease_dir(resource))]
    judge = Lease(resource, timeout=0, stale_after=60)
    serving = [False]
    outcome: list[str] = []

    def lstat(path: Any, *, dir_fd: int | None = None) -> os.stat_result:  # noqa: ANN401
        if serving[0] and os.fsdecode(path) == "thing.lease":
            return cached[0]
        return real_lstat(path, dir_fd=dir_fd)

    def rename(src: Any, dst: Any, *, src_dir_fd: int | None = None, dst_dir_fd: int | None = None) -> None:  # noqa: ANN401
        real_rename(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)
        if os.fsdecode(src) == planted and not outcome:
            serving[0] = True
            try:
                judge.__enter__()
                outcome.append("held")
            except LeaseTimeout:
                outcome.append("refused")
            finally:
                serving[0] = False

    monkeypatch.setattr(os, "lstat", lstat)
    monkeypatch.setattr(os, "rename", rename)
    with Lease(resource, timeout=1, stale_after=60) as claimant:
        assert [b.path for b in claimant.broke] == [_lease_dir(resource)] and claimant.owns()
    assert outcome == ["refused"] and [b.path for b in judge.broke] == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("where", [("lstat",), ("listdir",), ("lstat", "listdir")])
def test_a_lease_judged_while_another_node_removes_it_reads_estale_as_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: tuple[str, ...]
) -> None:
    """`ESTALE` on a `gone-` entry's `lstat` or the lease's `listdir` while judging reads as gone; the taker holds.

    On NFS `ESTALE` answers only for an object that no longer exists, so a token it drops from the judged listing
    is gone, not live; the judge reads the lease again on a fresh clock before it breaks anything.
    """
    lease_dir = _lease_dir(tmp_path / "thing")
    claimed = lease_dir / f"gone-{uuid.uuid4().hex}"
    claimed.mkdir(parents=True)
    for path in (claimed, lease_dir):
        os.utime(path, (OLD, OLD))
    real_lstat, real_listdir, raised = os.lstat, os.listdir, []

    def stale(call: str, path: Any) -> None:  # noqa: ANN401
        if call in where and call not in raised:
            raised.append(call)
            raise OSError(errno.ESTALE, "Stale file handle", str(path))

    def lstat(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:  # noqa: ANN401
        if os.fsdecode(path).startswith("gone-"):
            stale("lstat", path)
        return real_lstat(path, *args, **kwargs)

    def listdir(path: Any = ".") -> list[str]:  # noqa: ANN401
        if isinstance(path, int) and lease_dir.exists() and os.fstat(path).st_ino == real_lstat(lease_dir).st_ino:
            stale("listdir", path)
        return real_listdir(path)

    monkeypatch.setattr(os, "lstat", lstat)
    monkeypatch.setattr(os, "listdir", listdir)
    with Lease(tmp_path / "thing", timeout=5, stale_after=60) as lease:
        assert lease.owns()
    assert sorted(raised) == sorted(where)


def test_a_take_that_reads_estale_after_its_mkdir_withdraws_and_takes_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A token `mkdir` that lands but answers `ESTALE` leaves no orphan: the taker withdraws, then holds the lease."""
    real_mkdir, raised = os.mkdir, list[str]()

    def mkdir(path: Any, mode: int = 0o777, *, dir_fd: int | None = None) -> None:  # noqa: ANN401
        real_mkdir(path, mode, dir_fd=dir_fd)
        if not raised and len(os.fsdecode(path)) == UUID_LEN and dir_fd is not None:
            raised.append(os.fsdecode(path))
            raise OSError(errno.ESTALE, "Stale file handle", os.fsdecode(path))

    monkeypatch.setattr(os, "mkdir", mkdir)
    with Lease(tmp_path / "thing", timeout=5, stale_after=60) as lease:
        assert lease.owns() and raised and [b.path for b in lease.broke] == []


# -- how long a wait runs, on a clock the test advances --


def test_a_dead_holders_lease_is_broken_at_the_first_poll_past_its_stale_bound(
    tmp_path: Path, lease_clock: Callable[..., ManualClock]
) -> None:
    """A lease stamped at the storage origin and never beaten is broken at the first poll past `stale_after`."""
    resource = tmp_path / "thing"
    _plant_lease(resource)
    stale_after = 10.5 * POLL_S
    clock = lease_clock(storage_origin=OLD)
    with Lease(resource, timeout=10, stale_after=stale_after, clock=clock) as taker:
        assert [b.path for b in taker.broke] == [_lease_dir(resource)]
    assert stale_after < clock.now() <= stale_after + POLL_S
    assert clock.slept == [POLL_S] * (int(stale_after // POLL_S) + 1)


def test_beats_keep_their_cadence_when_each_beat_is_slow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lease_clock: Callable[..., ManualClock]
) -> None:
    """Beats that take 0.15 s each are still due every 0.3 s: the cadence does not drift by the beats' latency."""
    clock, real_mkdir, starts = lease_clock(), os.mkdir, []

    def mkdir(path: Any, mode: int = 0o777, *, dir_fd: int | None = None) -> None:  # noqa: ANN401
        if os.fsdecode(path).startswith("beat-"):
            starts.append(clock.now())
            clock.advance(0.15)
        real_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "mkdir", mkdir)
    with Lease(tmp_path / "thing", timeout=0, stale_after=30, clock=clock) as lease, lease.keep_alive(0.01):
        clock.settle()
        for k in range(1, 8):
            clock.advance_to(0.3 * k + 0.01)
            clock.settle()
    periods = [b - a for a, b in zip(starts, starts[1:])]
    assert len(periods) == 6 and periods == pytest.approx([0.3] * 6), periods  # noqa: PLR2004


ROUNDS = 25
TAKERS = 8


def test_contending_processes_break_a_stale_lease_once_and_never_orphan_it(
    tmp_path: Path, slack_tests: dict[str, Any], tier_bound: float
) -> None:
    """Eight processes contend for one stale lease for 25 rounds, and nothing is left behind.

    In every round there is one break, every taker holds in turn and none is displaced (no overlapping holds), no taker
    times out on an unknown holder (no orphan), and a process moves only the stale lease it judged or its own. A
    moved lease may also hold the token of a taker that reached it before the move, which then withdrew; it never
    holds another process's held token.
    """
    poll_s = slack_tests["lease_clock"]["poll_s"]
    context = multiprocessing.get_context("spawn")
    with context.Pool(TAKERS, initializer=lease_stress_fixture.setup) as pool:
        for n in range(ROUNDS):
            root = tmp_path / f"round-{n}"
            root.mkdir()
            planted = _plant_lease(root / "thing").name
            gate = tmp_path / f"gate-{n}"
            gate.mkdir()
            reports = pool.starmap(
                lease_stress_fixture.take, [(str(root / "thing"), str(gate), TAKERS, poll_s, tier_bound)] * TAKERS
            )
            assert [r["timeout"] for r in reports if r["timeout"]] == [], (n, reports)
            assert all(r["held"] and not r["displaced"] for r in reports), (n, reports)
            assert sum(r["broke"] for r in reports) == 1, (n, reports)
            held = {r["token"] for r in reports}
            for r in reports:
                for tokens in r["renamed"]:
                    moved = {t.removeprefix(GONE_PREFIX) for t in tokens}
                    assert planted in moved or r["token"] in moved, (n, r)
                    assert not moved & (held - {r["token"]}), (n, r)
            assert list(root.iterdir()) == [], (n, list(root.iterdir()))
