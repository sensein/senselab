"""How long a lease waits: for a release, for staleness, for its own timeout, and against a beating holder.

Every wait runs on a `ManualClock` the test advances; nothing here sleeps or reads a wall clock.
"""

import multiprocessing
import os
import threading
import uuid
from pathlib import Path
from typing import Any

import lease_stress_fixture
import pytest
from lease_clock_fixture import ManualClock, age
from senselab_core.runtime.locks import GONE_PREFIX, POLL_S, Lease, LeaseTimeout

pytestmark = pytest.mark.logic

JOIN_S = 10.0
OLD = 10**9


def _plant_lease(resource: Path, *, at: float) -> str:
    token = Path(str(resource) + ".lease") / uuid.uuid4().hex
    token.mkdir(parents=True)
    for path in (token, token.parent):
        os.utime(path, (at, at))
    return token.name


def test_a_waiter_blocks_until_the_live_lease_is_released(tmp_path: Path) -> None:
    """A waiter polls a live lease until it is released after three polls, then takes it inside its timeout."""
    resource = tmp_path / "thing"
    held, release = threading.Event(), threading.Event()

    def hold() -> None:
        with Lease(resource, timeout=0):
            held.set()
            release.wait(JOIN_S)

    holder = threading.Thread(target=hold)
    holder.start()
    assert held.wait(JOIN_S)

    def release_after_three(_: float) -> None:
        if len(clock.slept) == 3:  # noqa: PLR2004 - the third poll
            release.set()
            holder.join(JOIN_S)

    clock = ManualClock(on_sleep=release_after_three)
    with Lease(resource, timeout=5, clock=clock) as waiter:
        assert waiter.owns()
    assert not holder.is_alive()
    assert clock.slept == [POLL_S] * 3 and clock.now() < 5  # noqa: PLR2004


def test_a_dead_holders_lease_is_broken_when_its_stale_bound_passes(tmp_path: Path) -> None:
    """A lease stamped at the storage origin and never beaten is broken at the first poll past `stale_after`."""
    resource = tmp_path / "thing"
    _plant_lease(resource, at=OLD)
    stale_after = 0.52
    clock = ManualClock(storage_origin=OLD)
    with Lease(resource, timeout=10, stale_after=stale_after, clock=clock) as taker:
        assert taker.broke == [Path(str(resource) + ".lease")]
    assert stale_after < clock.now() <= stale_after + POLL_S
    assert clock.slept == [POLL_S] * 11  # noqa: PLR2004 - the first poll whose elapsed time passes the bound


def test_a_beating_lease_outlives_four_stale_bounds(tmp_path: Path) -> None:
    """A holder renewing every 0.6 s keeps a 60 s lease through 240 s of storage time against a waiter."""
    resource = tmp_path / "thing"
    clock = ManualClock()
    with Lease(resource, timeout=0, stale_after=60, clock=clock) as holder, holder.keep_alive(0.01) as state:
        clock.settle()
        for _ in range(12):
            age(tmp_path, 20)
            clock.advance(20)
            clock.settle()
            with pytest.raises(LeaseTimeout):
                with Lease(resource, timeout=0, stale_after=60):
                    pass
        assert holder.owns() and not state.lost and state.beats >= 12  # noqa: PLR2004


def test_a_wait_on_a_live_lease_ends_at_its_timeout(tmp_path: Path) -> None:
    """A taker with a 0.3 s timeout against a live lease raises `LeaseTimeout` once 0.3 s have passed on its clock."""
    resource = tmp_path / "thing"
    clock = ManualClock()
    with Lease(resource, timeout=0):
        with pytest.raises(LeaseTimeout):
            with Lease(resource, timeout=0.3, stale_after=3600, clock=clock):
                pass
    assert clock.now() == pytest.approx(0.3)


def test_beats_keep_their_cadence_when_each_beat_is_slow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Beats that take 0.15 s each are still due every 0.3 s: the cadence does not drift by the beats' latency."""
    clock, real_mkdir, starts = ManualClock(), os.mkdir, []

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


def test_contending_processes_break_a_stale_lease_once_and_never_orphan_it(tmp_path: Path) -> None:
    """Eight processes contend for one stale lease for 25 rounds, and nothing is left behind.

    In every round there is one break, every taker holds in turn and none is displaced (no overlapping holds), no taker
    times out on an unknown holder (no orphan), and a process moves only the stale lease it judged or its own. A
    moved lease may also hold the token of a taker that reached it before the move, which then withdrew; it never
    holds another process's held token.
    """
    context = multiprocessing.get_context("spawn")
    with context.Pool(TAKERS, initializer=lease_stress_fixture.setup) as pool:
        for n in range(ROUNDS):
            root = tmp_path / f"round-{n}"
            root.mkdir()
            planted = _plant_lease(root / "thing", at=OLD)
            gate = tmp_path / f"gate-{n}"
            gate.mkdir()
            reports = pool.starmap(lease_stress_fixture.take, [(str(root / "thing"), str(gate), TAKERS)] * TAKERS)
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
