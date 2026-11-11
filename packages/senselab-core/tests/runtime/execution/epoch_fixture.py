"""Epochs a test makes current: `take` and `bind` in one call, as a coordinator that is ready does."""

from pathlib import Path

from senselab_core.runtime.execution.epoch import Epoch
from senselab_core.runtime.execution.run import prepare
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
