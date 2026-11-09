"""Process probes for tests, read from `ps` state: whether a process or any process of a group can still run code."""

import subprocess

from senselab_core.runtime.clock import Clock, SystemClock

POLL_S = 0.05
ROW_FIELDS = 2
"""A `ps -A -o pgid=,stat=` row: the group and the state."""


def can_run(stat: str) -> bool:
    """Whether a `ps` state string names a process that can still run user code: not a zombie, not exiting.

    The first letter is the run state (`Z` zombie, `X` dead); macOS adds an `E` flag while a process exits.
    """
    return bool(stat) and stat[0] not in "ZX" and "E" not in stat[1:]


def alive(pid: int) -> bool:
    """Whether `pid` names a process that can still run user code."""
    return can_run(state(pid))


def state(pid: int) -> str:
    """The `ps` state letters of `pid`, empty if it has gone."""
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False)
    return out.stdout.strip()


def group_alive(pgid: int) -> bool:
    """Whether any process of group `pgid` can still run user code, by `ps` over every process."""
    out = subprocess.run(["ps", "-A", "-o", "pgid=,stat="], capture_output=True, text=True, check=True)
    rows = (line.split() for line in out.stdout.splitlines())
    return any(len(row) == ROW_FIELDS and int(row[0]) == pgid and can_run(row[1]) for row in rows)


def group_gone(pgid: int, timeout_s: float, clock: Clock = SystemClock()) -> bool:  # noqa: B008 - stateless
    """Whether group `pgid` has no process that can run user code within `timeout_s` seconds of `clock`."""
    deadline = clock.now() + timeout_s
    while group_alive(pgid):
        if clock.now() > deadline:
            return False
        clock.sleep(POLL_S)
    return True
