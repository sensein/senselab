"""A lease is a directory taken by `mkdir` and renewed by beats, one planted contract per invariant ADR-003 states."""

import errno
import os
import shutil
import stat
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from senselab_core.runtime.locks import RENEWER_PREFIX, Lease, LeaseState, LeaseTimeout, LeaseUnavailable
from senselab_core.runtime.storage import StorageUnsupported

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


class _Shifted:
    """A stat result whose mtime reads `by` seconds later."""

    def __init__(self, st: os.stat_result, by: float) -> None:
        self._st, self._by = st, by

    def __getattr__(self, name: str) -> object:
        if name == "st_mtime":
            return self._st.st_mtime + self._by
        if name == "st_mtime_ns":
            return self._st.st_mtime_ns + int(self._by * 1e9)
        return getattr(self._st, name)


class StorageClock:
    """The storage clock, advanced at the `os` boundary: a stamp made after an advance reads that much later."""

    def __init__(self) -> None:
        """Start with no advance."""
        self.advances: list[tuple[float, float]] = []

    def advance(self, seconds: float) -> None:
        """Move the storage clock `seconds` ahead from now on, and let the storage's stamps pass the moment."""
        time.sleep(0.05)
        self.advances.append((time.time(), seconds))
        time.sleep(0.05)

    def offset(self, mtime: float) -> float:
        """Return how far the storage clock was advanced before the stamp `mtime`."""
        return sum(by for at, by in self.advances if mtime >= at)


@pytest.fixture
def storage_clock(monkeypatch: pytest.MonkeyPatch) -> StorageClock:
    """A storage clock the test advances; every `lstat` relative to a directory reads its stamps on it."""
    clock, real = StorageClock(), os.lstat

    def lstat(path: Any, *, dir_fd: int | None = None) -> object:  # noqa: ANN401
        st = real(path, dir_fd=dir_fd)
        return st if dir_fd is None else _Shifted(st, clock.offset(st.st_mtime))

    monkeypatch.setattr(os, "lstat", lstat)
    return clock


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


def _until_beats(state: LeaseState, beats: int) -> None:
    deadline = time.monotonic() + JOIN_S
    while state.beats < beats and time.monotonic() < deadline:
        time.sleep(0.01)
    assert state.beats >= beats, f"{state.beats} beats within {JOIN_S}s"


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
    with Lease(resource, timeout=5) as waiter:
        assert waiter.owns() and waiter.broke == [] and waiter.set_aside == []
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


def test_keep_alive_renews_while_the_work_writes_nothing(tmp_path: Path) -> None:
    """While the work is idle the renewer beats: `<token>/` is restamped, one beat is kept, and the thread ends."""
    resource = tmp_path / "thing"
    with Lease(resource, timeout=0, stale_after=1.0) as lease, lease.keep_alive(0.1) as state:
        token = next(_lease_dir(resource).iterdir())
        taken = os.lstat(token).st_mtime_ns
        _until_beats(state, 3)
        assert os.lstat(token).st_mtime_ns > taken
        assert len(list(token.iterdir())) == 1 and not state.lost and lease.owns()
    assert _renewers() == [] and list(tmp_path.iterdir()) == []


def test_beat_after_break_marks_lost(tmp_path: Path) -> None:
    """A beat that finds the lease gone (`ENOENT`) marks it lost; the release removes nothing; no thread is left."""
    resource = tmp_path / "thing"
    with Lease(resource, timeout=0, stale_after=1.0) as lease, lease.keep_alive(0.1) as state:
        _until_beats(state, 1)
        aside = tmp_path / uuid.uuid4().hex
        os.rename(_lease_dir(resource), aside)
        shutil.rmtree(aside)
        assert state.wait_lost(JOIN_S) and not lease.owns()
        assert _renewers() == []
        with Lease(resource, timeout=0) as other:
            assert other.owns()
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


def test_dead_holder_lease_broken_after_bound(tmp_path: Path, storage_clock: StorageClock) -> None:
    """A holder that stops beating is not broken 59 s on the storage clock after its last stamp, and is at 61 s."""
    resource = tmp_path / "thing"
    dead = Lease(resource, timeout=0, stale_after=60)
    dead.__enter__()
    storage_clock.advance(59)
    with pytest.raises(LeaseTimeout):
        with Lease(resource, timeout=0, stale_after=60):
            pass
    storage_clock.advance(2)
    with Lease(resource, timeout=0, stale_after=60) as breaker:
        assert breaker.broke == [_lease_dir(resource)] and breaker.owns() and not dead.owns()
    dead.__exit__(None, None, None)


def test_breaker_never_breaks_a_beating_lease(tmp_path: Path, storage_clock: StorageClock) -> None:
    """Over 100 s on the storage clock, more than three stale bounds, a beating lease is never broken."""
    resource = tmp_path / "thing"
    with Lease(resource, timeout=0, stale_after=30) as holder, holder.keep_alive(0.01) as state:
        for _ in range(10):
            storage_clock.advance(10)
            _until_beats(state, state.beats + 1)
            with pytest.raises(LeaseTimeout):
                with Lease(resource, timeout=0, stale_after=30):
                    pass
        assert holder.owns() and not state.lost
    assert _renewers() == []


def test_staleness_is_read_on_the_storage_clock_not_the_local_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, storage_clock: StorageClock
) -> None:
    """A local clock an hour ahead does not age a fresh lease; an hour on the storage clock does."""
    resource = tmp_path / "thing"
    real_time = time.time
    with Lease(resource, timeout=0):
        monkeypatch.setattr(time, "time", lambda: real_time() + 3600)
        with pytest.raises(LeaseTimeout):
            with Lease(resource, timeout=0.2, stale_after=60):
                pass
        monkeypatch.setattr(time, "time", real_time)
        storage_clock.advance(3600)
        with Lease(resource, timeout=1, stale_after=60) as breaker:
            assert breaker.broke == [_lease_dir(resource)]


def test_a_lease_older_than_stale_after_is_broken_and_taken(tmp_path: Path) -> None:
    """A lease stamped long ago is renamed away and removed, the taker holds a fresh one, and nothing is left."""
    resource = tmp_path / "thing"
    _plant_lease(resource)
    with Lease(resource, timeout=1, stale_after=60) as lease:
        assert lease.owns() and lease.broke == [_lease_dir(resource)]
    assert list(tmp_path.iterdir()) == []


def test_an_empty_lease_a_taker_left_is_judged_by_its_own_stamp(tmp_path: Path) -> None:
    """A lease directory with no token, as a taker that died between its two `mkdir`s leaves, is broken when stale."""
    lease_dir = _lease_dir(tmp_path / "thing")
    lease_dir.mkdir()
    os.utime(lease_dir, (OLD, OLD))
    with Lease(tmp_path / "thing", timeout=1, stale_after=60) as lease:
        assert lease.owns() and lease.broke == [lease_dir]


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


def test_a_breaker_that_moved_a_newer_lease_by_mistake_reports_no_break_and_its_holder_knows(tmp_path: Path) -> None:
    """A breaker that renamed a newer lease records no break, and the newer holder finds its lease lost."""
    resource = tmp_path / "thing"
    judged = frozenset({_plant_lease(resource).name})
    late = Lease(resource, timeout=0, stale_after=60)
    with Lease(resource, timeout=1, stale_after=60) as winner:
        assert winner.broke and winner.owns()
        assert late._take_aside(lambda found: found == judged) is False
        assert not winner.owns()
    late._close()
    assert list(tmp_path.iterdir()) == []


def test_concurrent_breakers_remove_one_stale_lease_once_and_a_displaced_holder_knows(lease_dir: Path) -> None:
    """Eight takers break one stale lease together: it is removed once; holders overlap only when one was displaced."""
    resource = lease_dir / "thing"
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
    assert list(lease_dir.iterdir()) == []


# -- nothing is written under the lease --


def test_no_file_is_written_under_the_lease(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Taking, beating, judging and releasing a lease create no file: only directories, under any name."""
    resource, real_open, created = tmp_path / "thing", os.open, []

    def opened(path: Any, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:  # noqa: ANN401
        if flags & os.O_CREAT:
            created.append(os.fsdecode(path))
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", opened)
    with Lease(resource, timeout=0, stale_after=1.0) as lease, lease.keep_alive(0.1) as state:
        _until_beats(state, 2)
        with pytest.raises(LeaseTimeout):
            with Lease(resource, timeout=0.2, stale_after=1.0):
                pass
        assert [f for _, _, files in os.walk(_lease_dir(resource)) for f in files] == []
    assert created == [] and list(tmp_path.iterdir()) == []


# -- every wait is bounded --


def test_a_wait_on_a_live_lease_ends_at_its_timeout(tmp_path: Path) -> None:
    """A taker with a 0.3 s timeout against a live lease raises `LeaseTimeout` naming the holder's token."""
    resource = tmp_path / "thing"
    with Lease(resource, timeout=0) as holder:
        with pytest.raises(LeaseTimeout, match=r"thing\.lease, held by .*'token'"):
            with Lease(resource, timeout=0.3, stale_after=3600):
                pass
        assert holder.owns()


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


@pytest.mark.skipif(os.geteuid() == 0, reason="root lists a 0o000 directory")
def test_a_stale_lease_this_user_cannot_list_is_still_broken(tmp_path: Path) -> None:
    """A stale lease another user left unlistable is judged by its own stamp, moved aside, and taken."""
    resource = tmp_path / "thing"
    _plant_lease(resource)
    os.chmod(_lease_dir(resource), 0o000)
    try:
        with Lease(resource, timeout=1, stale_after=60) as lease:
            assert lease.owns() and lease.broke == [_lease_dir(resource)] and len(lease.set_aside) == 1
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
