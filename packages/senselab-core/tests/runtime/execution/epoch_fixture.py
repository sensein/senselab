"""Epochs a test makes current: `take` and `bind` in one call, as a coordinator that is ready does.

`left_by_a_killed_coordinator` writes what a coordinator killed while it ran leaves of pydra's lease of the run.
"""

import socket
from pathlib import Path

from senselab_core.runtime.execution.epoch import Epoch
from senselab_core.runtime.execution.run import PYDRA_DIR, prepare
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


KILLED_HOLDERS = {
    "another host": f"1\n{socket.gethostname()}.elsewhere\n",
    "a live process of this host": f"1\n{socket.gethostname()}\n",
}
"""The holder a killed coordinator's lock file names, as `filelock` writes it (pid, host): neither provably dead."""


def left_by_a_killed_coordinator(run_dir: Path, holder: str) -> None:
    """Write the lock file of pydra's run-long lease of the run's workflow job, naming `KILLED_HOLDERS[holder]`."""
    root = run_dir / PYDRA_DIR
    [job] = [d for d in root.iterdir() if d.is_dir()]
    [context] = [d for d in job.iterdir() if d.is_dir()]
    (root / f"{job.name}-{context.name}.lock").write_text(KILLED_HOLDERS[holder])
