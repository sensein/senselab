"""The liveness probes the lifecycle tests rely on, against real processes in each state they must tell apart."""

import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator

import pytest
from lifecycle_probes_fixture import alive, can_run, group_alive, state

SETTLE_S = 10.0
SLEEPER = "import time; time.sleep(3600)"
FORKER = "import os, time\nif os.fork() == 0:\n    time.sleep(3600)\nelse:\n    os._exit(0)"


def _until(predicate: object, what: str) -> None:
    end = time.monotonic() + SETTLE_S
    while not predicate():  # type: ignore[operator]
        assert time.monotonic() < end, what
        time.sleep(0.01)


@pytest.fixture
def leader() -> Iterator[subprocess.Popen[bytes]]:
    """A sleeping process that leads its own process group; its group is killed and it is reaped afterwards."""
    proc = subprocess.Popen([sys.executable, "-c", SLEEPER], start_new_session=True)
    yield proc
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    proc.wait()


@pytest.mark.contract
@pytest.mark.parametrize(
    ("stat", "runs"),
    [("S", True), ("Ss", True), ("R+", True), ("T", True), ("U", True), ("D", True), ("I", True)]
    + [("Z", False), ("Z+", False), ("X", False), ("", False), ("RE", False), ("SE", False), ("TE", False)],
)
def test_ps_states_read_as_runnable_unless_zombie_dead_or_exiting(stat: str, runs: bool) -> None:
    """Zombie, dead, gone and the macOS exiting flag `E` cannot run user code; every other state can."""
    assert can_run(stat) is runs


@pytest.mark.contract
def test_probes_follow_a_real_process_through_its_states(leader: subprocess.Popen[bytes]) -> None:
    """Sleeping and stopped read as alive; an unreaped zombie leader and a reaped one do not, nor does its group."""
    pid = leader.pid
    assert alive(pid) and group_alive(pid)
    os.kill(pid, signal.SIGSTOP)
    _until(lambda: state(pid).startswith("T"), "the process never stopped")
    assert alive(pid) and group_alive(pid)
    os.kill(pid, signal.SIGKILL)
    _until(lambda: state(pid).startswith("Z"), "the process never became a zombie")
    assert not alive(pid) and not group_alive(pid)
    leader.wait()
    assert not alive(pid) and not group_alive(pid)


@pytest.mark.contract
def test_group_with_a_zombie_leader_and_a_live_member_is_alive() -> None:
    """A live member keeps its group alive after the leader has exited unreaped."""
    proc = subprocess.Popen([sys.executable, "-c", FORKER], start_new_session=True)
    try:
        _until(lambda: state(proc.pid).startswith("Z"), "the leader never exited")
        assert not alive(proc.pid) and group_alive(proc.pid)
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait()
    _until(lambda: not group_alive(proc.pid), "the forked member outlived SIGKILL")
