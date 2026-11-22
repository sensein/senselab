"""`served`: a worker is live whenever its class has waiting tickets, and closing the helper leaves no worker thread."""

import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from pool_stack_fixture import packaged, planned, registry, scoring
from senselab_core.runtime.execution.run import start
from senselab_core.testing.pools import POOL_PROFILE, served

pytestmark = pytest.mark.contract

TASK = scoring("served_pool_test.score")


@pytest.fixture(autouse=True)
def stack(tmp_path: Path) -> Iterator[None]:
    """This module's registry, every env this interpreter under an MIT lock; the cache under `tmp_path`."""
    with packaged(registry(TASK), tmp_path / "cache"):
        yield


def test_a_run_submitted_after_the_first_worker_idled_out_completes(tmp_path: Path) -> None:
    """The first job arrives after the worker started at open has exited idle; another worker serves it."""
    idle_s = POOL_PROFILE.pool.idle_polls * POOL_PROFILE.pool.poll_s
    placed, values = planned(TASK, POOL_PROFILE, ["a", "b"])
    with served(tmp_path / "pool", store=tmp_path / "store") as (session, pool):
        time.sleep(idle_s + 1.0)
        assert not any(t.is_alive() for t in pool.threads)
        run = start(placed, session, values=values)
        assert run.wait(timeout=30).done
    assert sorted(str(given.outputs["value"]) for _, given in run.results()) == ["a", "b"]  # type: ignore[union-attr]


def test_closing_the_helper_stops_its_workers_without_waiting_for_them_to_idle_out(tmp_path: Path) -> None:
    """`served` exits within the workers' idle window, and no thread it or the pool's workers started is alive."""
    idle_s = POOL_PROFILE.pool.idle_polls * POOL_PROFILE.pool.poll_s
    before = set(threading.enumerate())
    placed, values = planned(TASK, POOL_PROFILE, ["a"])
    with served(tmp_path / "pool", store=tmp_path / "store") as (session, pool):
        assert start(placed, session, values=values).wait(timeout=30).done
        assert any(t.is_alive() for t in pool.threads)
        closing = time.monotonic()
    assert time.monotonic() - closing < idle_s / 2
    left = [t.name for t in threading.enumerate() if t not in before and t.is_alive()]
    assert left == []
