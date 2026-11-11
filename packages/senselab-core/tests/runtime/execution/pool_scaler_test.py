"""The scaler thread of a pool ends when the pool is collected, not only when it is closed."""

import gc
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pool_stack_fixture
import pytest
from lease_clock_fixture import ManualClock
from pool_test import (  # noqa: F401 - `stack` is autouse
    CONTENT,
    _Idle,
    _pool,
    _profile,
    _stack,
    stack,
)
from senselab_core.runtime.execution.in_process import node_fragment


@pytest.mark.contract
def test_a_pool_dropped_without_close_ends_its_scaler(
    tmp_path: Path, lease_clock: Callable[..., ManualClock], slack_tests: dict[str, Any]
) -> None:
    """The scaler thread parked on a clock nobody advances ends once its pool is collected; no hang guard ends it."""
    hang_s = slack_tests["stop_test"]["hang_s"]
    profile = _profile(**CONTENT)
    placed, values, store, _ = _stack(tmp_path, profile, ["one"])
    pool = _pool(tmp_path, store, _Idle(), profile, clock=lease_clock(spends="park", hang_s=hang_s))
    pool.submit(node_fragment(placed, pool.admit(placed, values), elements=[(0,)]))
    scaler = pool._scaler  # noqa: SLF001 - the thread under test
    assert scaler is not None and scaler.is_alive()
    pool_stack_fixture._POOLS.remove(pool)  # noqa: SLF001 - handed over to be collected, not closed
    del pool
    deadline = time.monotonic() + hang_s
    while scaler.is_alive() and time.monotonic() < deadline:
        gc.collect()
        scaler.join(slack_tests["fake_worker"]["poll_s"])
    assert not scaler.is_alive()
