"""How long a lease waits: for a release, for staleness and for its own timeout."""

import json
import os
import threading
import time
from pathlib import Path

import pytest
from senselab_core.runtime.locks import Lease, LeaseTimeout

pytestmark = pytest.mark.limits

JOIN_S = 10.0
OLD = 10**9


def _lease_file(resource: Path) -> Path:
    return Path(str(resource) + ".lease")


def _plant_lease(resource: Path, *, at: float = OLD) -> Path:
    lease = _lease_file(resource)
    lease.write_text(json.dumps({"user": "u", "host": "h", "pid": 1}))
    os.utime(lease, (at, at))
    return lease


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


def test_a_fresh_lease_is_broken_when_its_stale_bound_passes(tmp_path: Path) -> None:
    """A lease stamped now is waited on for about `stale_after` (0.5 s) and not broken before 0.3 s."""
    resource = tmp_path / "thing"
    _plant_lease(resource, at=time.time())
    started = time.monotonic()
    with Lease(resource, timeout=10, stale_after=0.5):
        waited = time.monotonic() - started
    assert 0.3 <= waited < 5


def test_a_wait_on_a_live_lease_ends_near_its_timeout(tmp_path: Path) -> None:
    """A taker with a 0.3 s timeout against a live lease raises `LeaseTimeout` within 2 s."""
    resource = tmp_path / "thing"
    with Lease(resource, timeout=0):
        started = time.monotonic()
        with pytest.raises(LeaseTimeout):
            with Lease(resource, timeout=0.3, stale_after=3600):
                pass
        assert time.monotonic() - started < 2
