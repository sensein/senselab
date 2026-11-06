"""How long a lease waits: for a release, for staleness, for its own timeout, and against a beating holder."""

import multiprocessing
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import lease_stress_fixture
import pytest
from senselab_core.runtime.locks import GONE_PREFIX, Lease, LeaseTimeout

pytestmark = pytest.mark.limits

JOIN_S = 10.0


def _plant_lease(resource: Path, *, at: float) -> str:
    token = Path(str(resource) + ".lease") / uuid.uuid4().hex
    token.mkdir(parents=True)
    for path in (token, token.parent):
        os.utime(path, (at, at))
    return token.name


def test_a_waiter_blocks_until_the_live_lease_is_released(tmp_path: Path) -> None:
    """A waiter against a live lease released after 0.3 s is held for at least 0.2 s."""
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
    with Lease(resource, timeout=5):
        waited = time.monotonic() - started
    holder.join(JOIN_S)
    assert not holder.is_alive()
    assert waited >= 0.2


def test_a_dead_holders_lease_is_broken_when_its_stale_bound_passes(tmp_path: Path) -> None:
    """A lease stamped now and never beaten is waited on for about `stale_after` (0.5 s) and not broken before 0.3 s."""
    resource = tmp_path / "thing"
    _plant_lease(resource, at=time.time())
    started = time.monotonic()
    with Lease(resource, timeout=10, stale_after=0.5):
        waited = time.monotonic() - started
    assert 0.3 <= waited < 5


def test_a_beating_lease_outlives_four_stale_bounds(tmp_path: Path) -> None:
    """A holder renewing every 0.1 s keeps a 0.5 s lease for 2 s against a waiter, which then times out."""
    resource = tmp_path / "thing"
    with Lease(resource, timeout=0, stale_after=0.5) as holder, holder.keep_alive(0.2) as state:
        with pytest.raises(LeaseTimeout):
            with Lease(resource, timeout=2, stale_after=0.5):
                pass
        assert holder.owns() and not state.lost and state.beats >= 10


def test_a_wait_on_a_live_lease_ends_near_its_timeout(tmp_path: Path) -> None:
    """A taker with a 0.3 s timeout against a live lease raises `LeaseTimeout` within 2 s."""
    resource = tmp_path / "thing"
    with Lease(resource, timeout=0):
        started = time.monotonic()
        with pytest.raises(LeaseTimeout):
            with Lease(resource, timeout=0.3, stale_after=3600):
                pass
        assert time.monotonic() - started < 2


def test_beats_keep_their_cadence_when_each_beat_is_slow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Beats that take 0.15 s each are still due every 0.3 s: the cadence does not drift by the beats' latency."""
    real_mkdir, starts = os.mkdir, []

    def mkdir(path: Any, mode: int = 0o777, *, dir_fd: int | None = None) -> None:  # noqa: ANN401
        if os.fsdecode(path).startswith("beat-"):
            starts.append(time.monotonic())
            time.sleep(0.15)
        real_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "mkdir", mkdir)
    with Lease(tmp_path / "thing", timeout=0, stale_after=30) as lease, lease.keep_alive(0.01):
        deadline = time.monotonic() + JOIN_S
        while len(starts) < 7 and time.monotonic() < deadline:  # noqa: PLR2004 - six periods
            time.sleep(0.01)
    periods = [b - a for a, b in zip(starts, starts[1:])][:6]
    assert len(periods) == 6 and sum(periods) / len(periods) < 0.36, periods  # noqa: PLR2004


ROUNDS = 25
TAKERS = 8


def test_contending_processes_break_a_stale_lease_once_and_never_orphan_it(tmp_path: Path) -> None:
    """Eight processes contend for one stale lease for 25 rounds, and nothing is left behind.

    In every round there is one break, every taker holds in turn, no taker times out on an unknown holder, and a process
    renames only the stale lease it judged or its own.
    """
    context = multiprocessing.get_context("spawn")
    with context.Pool(TAKERS, initializer=lease_stress_fixture.setup) as pool:
        for n in range(ROUNDS):
            root = tmp_path / f"round-{n}"
            root.mkdir()
            planted = _plant_lease(root / "thing", at=10**9)
            start_at = time.time() + 0.2
            reports = pool.starmap(lease_stress_fixture.take, [(str(root / "thing"), start_at)] * TAKERS)
            assert [r["timeout"] for r in reports if r["timeout"]] == [], (n, reports)
            assert all(r["held"] and not r["displaced"] for r in reports), (n, reports)
            assert sum(r["broke"] for r in reports) == 1, (n, reports)
            for r in reports:
                moved = [[t.removeprefix(GONE_PREFIX) for t in tokens] for tokens in r["renamed"]]
                assert all(tokens in ([planted], [r["token"]]) for tokens in moved), (n, r)
            assert list(root.iterdir()) == [], (n, list(root.iterdir()))
