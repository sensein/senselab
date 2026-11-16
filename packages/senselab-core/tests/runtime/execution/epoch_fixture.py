"""Epochs a test makes current: `take` and `bind` in one call, as a coordinator that is ready does.

`waited_past_queued` fails a run stuck before its first submit in seconds.
"""

import time
from pathlib import Path

from senselab_core.runtime.execution.epoch import Epoch
from senselab_core.runtime.execution.events import RunStatus, read_events
from senselab_core.runtime.execution.run import Run, prepare
from senselab_core.runtime.execution_settings import JobLease
from senselab_core.runtime.planner import Plan
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.store import ResultStore


def taken(epochs: Epoch, lease: JobLease | None = None) -> int:
    """Take the next epoch under `lease` and bind it to a synthetic attempt `a<n>`; return it."""
    n = epochs.take(lease)
    epochs.bind(n, f"a{n}", n)
    return n


def prepared(placed: Plan, run_dir: Path, store: ResultStore) -> tuple[int, str, int]:
    """Claim an attempt for `placed` (`prepare`), take the next epoch and bind it; return both and the ordinal."""
    attempt, ordinal = prepare(placed, run_dir, store)
    epochs = Epoch(run_dir)
    profile = placed.execution if placed.execution is not None else ExecutionProfile.load()
    n = epochs.take(profile.lease)
    epochs.bind(n, attempt, ordinal)
    return n, attempt, ordinal


def waited_past_queued(run: Run, stall_s: float, hang_s: float) -> RunStatus:
    """Return `run`'s status once it is done or `hang_s` passed; fail after `stall_s` on a run stuck before a submit.

    Stuck: its own epoch (`Run.epoch`, which `start` sets) has logged nothing but `queued` and `placement` by then.
    """
    assert run.epoch is not None, "a run `start` made"
    deadline = time.monotonic() + stall_s
    while time.monotonic() < deadline:
        status = run.wait(stall_s / 10)
        if status.done:
            return status
    kinds = {e.kind for e in read_events(run.run_dir) if e.epoch == run.epoch}
    assert kinds - {"queued", "placement"}, f"the run logged only {sorted(kinds)} in {stall_s} s: stuck before a submit"
    return run.wait(hang_s)
