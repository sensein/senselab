"""Epochs a test makes current: `take` and `bind` in one call, as a coordinator that is ready does.

`left_by_a_killed_coordinator` writes a pydra lock a coordinator killed while it held it leaves, and
`waited_past_queued` fails a run stuck before its first submit in seconds.
"""

import socket
import time
from pathlib import Path

from senselab_core.runtime.execution.epoch import Epoch
from senselab_core.runtime.execution.events import RunStatus, read_events
from senselab_core.runtime.execution.run import PYDRA_DIR, Run, prepare
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

KILLED_LOCKS = ("lease", "save")
"""The pydra locks a coordinator killed in the middle of them leaves: its run-long lease, or one pickle's save."""


def left_by_a_killed_coordinator(run_dir: Path, holder: str, lock: str = "lease") -> Path:
    """Write the pydra lock `lock` of the run's workflow job naming `KILLED_HOLDERS[holder]`, and return it.

    `lease` is the run-long lease's `<checksum>-<context>.lock`; `save` is `<checksum>/<context>_save.lock` with the
    job's pickle cut short, as a kill in the middle of `save` leaves them.
    """
    root = run_dir / PYDRA_DIR
    [job] = [d for d in root.iterdir() if d.is_dir() and d.name != "pkl_files"]
    [context] = [d for d in job.iterdir() if d.is_dir()]
    if lock == "save":
        pickled = context / "_job.pklz"
        pickled.write_bytes(pickled.read_bytes()[:16])
        path = job / f"{context.name}_save.lock"
    else:
        path = root / f"{job.name}-{context.name}.lock"
    path.write_text(KILLED_HOLDERS[holder])
    return path


def waited_past_queued(run: Run, stall_s: float, hang_s: float) -> RunStatus:
    """Return `run`'s status once it is done or `hang_s` passed; fail after `stall_s` on a run stuck before a submit.

    Stuck: its own epoch has logged nothing but `queued` and `placement` by then.
    """
    deadline = time.monotonic() + stall_s
    while time.monotonic() < deadline:
        status = run.wait(stall_s / 10)
        if status.done:
            return status
    kinds = {e.kind for e in read_events(run.run_dir) if e.epoch == run.epoch}
    assert kinds - {"queued", "placement"}, f"the run logged only {sorted(kinds)} in {stall_s} s: stuck before a submit"
    return run.wait(hang_s)
