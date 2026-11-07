"""The supervisor: a worker's orphans are adopted, reaped and killed with it; group liveness is read from its table."""

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from senselab_core.runtime.execution.supervisor import Keeper, Supervisor, subreaper_supported
from senselab_core.runtime.worker_client import WorkerClient
from senselab_core.testing.interpreter import source_tree_python
from senselab_core.testing.processes import alive, group_alive, group_gone

PYTHON = source_tree_python()
ESCAPING = "backends.escaping_backend_fixture"
SETTLE_S = 10.0
ONE = [{"id": "a", "inputs": {}}]
SLEEPER = "import time; time.sleep(3600)"
FORK_THEN_SLEEP = "import os, sys, time\nchild = os.fork()\nif child:\n    print(child, flush=True)\ntime.sleep(3600)\n"

linux_only = pytest.mark.skipif(not subreaper_supported(), reason="a subreaper and its children list are Linux-only")


def _until(predicate: Callable[[], bool], what: str) -> None:
    deadline = time.monotonic() + SETTLE_S
    while not predicate():
        assert time.monotonic() < deadline, f"not within {SETTLE_S} s: {what}"
        time.sleep(0.01)


def _not_my_child(pid: int) -> bool:
    try:
        os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except ChildProcessError:
        return True
    return False


def _spawned(sup: Supervisor) -> tuple[WorkerClient, int, int]:
    """Start an escaping worker under `sup`; return it, its forked member and its escapee once `sup` adopted that."""
    client = WorkerClient(PYTHON, ESCAPING, init={}, supervisor=sup)
    member, escapee = client.run(ONE, {})[0]["outputs"]["pids"]
    _until(lambda: escapee in sup.orphans(), "the escapee is adopted")
    return client, member, escapee


@linux_only
@pytest.mark.contract
def test_setsid_escapee_killed_with_worker(backends_importable: None) -> None:
    """An escapee left by a worker is adopted while the worker runs, then SIGKILLed and reaped when it is closed."""
    with Supervisor() as sup:
        client, _, escapee = _spawned(sup)
        assert os.getpgid(escapee) != client.hello["pid"] and alive(escapee)
        client.close()
        _until(lambda: (escapee, -signal.SIGKILL) in sup.reaped, "the escapee is killed and reaped")
    assert _not_my_child(escapee) and not alive(escapee)


@linux_only
@pytest.mark.contract
def test_fork_without_exec_child_reaped(backends_importable: None) -> None:
    """A forked member outliving its leader is adopted into the leader's group, stopped with it and reaped."""
    with Supervisor() as sup:
        client, member, escapee = _spawned(sup)
        pgid = client.hello["pid"]
        client.close()
        _until(lambda: sup.orphans() == {}, "the member and the escapee are reaped")
        assert member in dict(sup.reaped) and not group_alive(pgid)
    assert _not_my_child(member) and not alive(member) and not alive(escapee)


class _Asked(Supervisor):
    """A supervisor that records each group whose liveness it is asked."""

    def __init__(self) -> None:
        super().__init__()
        self.asked: list[int] = []

    def group_live(self, pgid: int) -> bool:
        self.asked.append(pgid)
        return super().group_live(pgid)


@linux_only
@pytest.mark.contract
def test_liveness_from_supervisor_not_ps(backends_importable: None) -> None:
    """Liveness is the table's: a group launched elsewhere is never live to it, a group with a running orphan is.

    A worker's close asks the supervisor it was launched through.
    """
    elsewhere = subprocess.Popen([sys.executable, "-c", SLEEPER], start_new_session=True)
    try:
        with _Asked() as sup:
            assert not sup.group_live(elsewhere.pid) and group_alive(elsewhere.pid)
            leader = sup.popen([sys.executable, "-c", FORK_THEN_SLEEP], stdout=subprocess.PIPE)
            assert leader.stdout is not None
            member = int(leader.stdout.readline())
            assert sup.group_live(leader.pid)
            leader.kill()
            _until(lambda: member in sup.orphans(), "the member is adopted")
            assert sup.group_live(leader.pid)
            os.kill(member, signal.SIGKILL)
            _until(lambda: not sup.group_live(leader.pid), "the group is dead to the supervisor")
            sup.released(leader.pid)
            leader.wait()
            leader.stdout.close()
            client, _, _ = _spawned(sup)
            client.close()
            assert client.hello["pid"] in sup.asked
    finally:
        elsewhere.kill()
        elsewhere.wait()


@pytest.mark.contract
def test_leaving_a_supervisor_kills_its_groups() -> None:
    """Every group launched through a supervisor, and not yet released, is SIGKILLed and reaped when it is left."""
    with Supervisor(subreaper=False) as sup:
        leader = sup.popen([sys.executable, "-c", SLEEPER])
    assert leader.returncode == -signal.SIGKILL


DRIVER = """
import json, sys, time
from senselab_core.runtime.execution.supervisor import Supervisor
from senselab_core.runtime.worker_client import WorkerClient

with Supervisor() as sup:
    client = WorkerClient(sys.argv[1], sys.argv[2], init={})
    member, escapee = client.run([{"id": "a", "inputs": {}}], {})[0]["outputs"]["pids"]
    while escapee not in sup.orphans():
        time.sleep(0.01)
    print(json.dumps([client.hello["pid"], member, escapee]), flush=True)
    time.sleep(3600)
"""


@linux_only
@pytest.mark.limits
def test_supervisor_host_sigkill_leaves_no_descendant(backends_importable: None, tmp_path: Path) -> None:
    """A SIGKILL of the supervising host leaves no descendant: its keeper kills the worker's group and its orphans."""
    host = subprocess.Popen([sys.executable, "-c", DRIVER, str(PYTHON), ESCAPING], stdout=subprocess.PIPE)
    assert host.stdout is not None
    try:
        leader, member, escapee = json.loads(host.stdout.readline())
        assert all(map(alive, (leader, member, escapee)))
        host.kill()
        host.wait()
        _until(lambda: not any(map(alive, (leader, member, escapee))), "every descendant is gone")
        assert not group_alive(leader)
    finally:
        host.kill()
        host.wait()
        host.stdout.close()


def _exited(pid: int) -> bool:
    return os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None


@linux_only
@pytest.mark.contract
def test_released_leader_keeps_its_exit_status(backends_importable: None) -> None:
    """A leader released while the reaper runs is never reaped by it: its `Popen` and its client read -9."""
    with Supervisor(poll_s=0.001) as sup:
        for _ in range(20):
            leader = sup.popen([sys.executable, "-c", SLEEPER])
            leader.kill()
            _until(lambda: _exited(leader.pid), "the leader has exited")
            sup.released(leader.pid)
            sup.reap()
            assert leader.wait() == -signal.SIGKILL
        client = WorkerClient(PYTHON, ESCAPING, init={}, supervisor=sup)
        os.kill(client.hello["pid"], signal.SIGKILL)
        client.close()
        assert client.returncode == -signal.SIGKILL


@pytest.mark.contract
def test_keeper_kills_only_groups_it_still_owns() -> None:
    """At its end the keeper SIGKILLs a recorded group whose leader has the recorded start time, never another."""
    owned = subprocess.Popen([sys.executable, "-c", SLEEPER], start_new_session=True)
    reused = subprocess.Popen([sys.executable, "-c", SLEEPER], start_new_session=True)
    try:
        keeper = Keeper()
        keeper.record("+", "g", owned.pid)
        keeper.record("+", "g", reused.pid, start="another-start")
        keeper.stop(SETTLE_S)
        assert owned.wait(SETTLE_S) == -signal.SIGKILL
        assert reused.poll() is None
    finally:
        for proc in (owned, reused):
            proc.kill()
            proc.wait()


UNSUPERVISED = """
import json, sys, time
from senselab_core.runtime.worker_client import WorkerClient

client = WorkerClient(sys.argv[1], sys.argv[2], init={})
member, _ = client.run([{"id": "a", "inputs": {}}], {})[0]["outputs"]["pids"]
print(json.dumps([client.hello["pid"], member]), flush=True)
time.sleep(3600)
"""


@pytest.mark.contract
def test_an_unsupervised_owner_sigkill_leaves_no_group(backends_importable: None) -> None:
    """With no supervisor entered, a SIGKILL of a worker's owner still takes the worker's group with it."""
    host = subprocess.Popen([sys.executable, "-c", UNSUPERVISED, str(PYTHON), ESCAPING], stdout=subprocess.PIPE)
    assert host.stdout is not None
    try:
        leader, member = json.loads(host.stdout.readline())
        assert alive(leader) and alive(member)
        host.kill()
        host.wait()
        assert group_gone(leader, SETTLE_S) and not alive(member)
    finally:
        host.kill()
        host.wait()
        host.stdout.close()
