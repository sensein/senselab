"""A SIGTERM handler that calls `close()` while the main thread is inside each wait `close()` can reach."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_HEAD = """
import signal, sys, threading
from pathlib import Path
from signal_in_lock_fixture import THREADING, deliver_in
from senselab_core.runtime import worker_pipes
from senselab_core.runtime.worker_client import WorkerClient, WorkerLost
from senselab_core.testing.interpreter import source_tree_python
from senselab_worker.fake import FAULT_KEY

FILES = (THREADING, worker_pipes.__file__)
box, delivered = [], [False]

def on_term(signum, frame):
    delivered[0] = True
    box[0].close()

signal.signal(signal.SIGTERM, on_term)
parent = Path(sys.argv[1])
"""

_BUILT = """
client = WorkerClient(source_tree_python(), "senselab_worker.fake", init={}, array_parent=parent)
box.append(client)
"""

_FINISH = """
sys.settrace(None)
box[0].close()
print("delivered" if delivered[0] else "not-delivered", result, flush=True)
"""


def _in_wait_on(event: str) -> str:
    return (
        "state = deliver_in(signal.SIGTERM, lambda f: f.f_code.co_name == 'wait' and f.f_locals.get('self') is "
        f"{event}, FILES)\n"
    )


CASES = {
    # file:line of the wait, as of this change
    "wait_exited": _BUILT
    + _in_wait_on("client._exited")
    + "result = client.wait_exited(30.0)\n",  # worker_client.py wait_exited, on `_exited`
    "fail_exit_code_wait": _BUILT
    + _in_wait_on("client._exited")
    + "try:\n    client.run([{'id': 'a', 'inputs': {}}], {FAULT_KEY: {'crash_after': 0}})\n"
    "except WorkerLost:\n    result = 'WorkerLost'\n",  # `_fail`, on `_exited`
    "terminate_reaped": _BUILT + _in_wait_on("client._reaped") + "client.close()\nresult = 'closed'\n",
    "terminate_stdout_ended": _BUILT + _in_wait_on("client._pumps[0].ended") + "client.close()\nresult = 'closed'\n",
    "terminate_stderr_ended": _BUILT + _in_wait_on("client._pumps[1].ended") + "client.close()\nresult = 'closed'\n",
    "shutdown_lock_held": _BUILT
    + "state = deliver_in(signal.SIGTERM, lambda f: f.f_code.co_name == '_stop' and f.f_locals.get('self') is client,"
    " (worker_pipes.__file__.replace('worker_pipes', 'worker_client'),))\n"
    "client.close()\nresult = 'closed'\n",
    "pump_lock_held": _BUILT
    + "state = deliver_in(signal.SIGTERM, lambda f: f.f_code.co_name == 'finish' and f.f_locals.get('self') is "
    "client._pumps[0], FILES)\nclient.close()\nresult = 'closed'\n",
    "channel_lock_held_in_init": "client_file = worker_pipes.__file__.replace('worker_pipes', 'worker_client')\n"
    "def under(frame):\n    box[:] = [frame.f_locals['self']]\n"
    "    return frame.f_locals['self']._channel_lock.locked()\n"
    "state = deliver_in(signal.SIGTERM, lambda f: f.f_code.co_name == '_send', (client_file,), at=under)\n"
    "try:\n    WorkerClient(source_tree_python(), 'senselab_worker.fake', init={}, array_parent=parent)\n"
    "except WorkerLost:\n    result = 'WorkerLost'\n",
}


@pytest.mark.contract
@pytest.mark.parametrize("case", list(CASES))
def test_a_sigterm_handler_closing_the_client_inside_a_wait_finishes_and_leaves_nothing(
    case: str, tmp_path: Path
) -> None:
    """SIGTERM lands inside the wait while a handler calls `close()`; the process ends and leaves nothing."""
    script = _HEAD + CASES[case] + _FINISH
    arrays = tmp_path / "arrays"
    arrays.mkdir()
    proc = subprocess.run(
        [sys.executable, "-c", script, str(arrays)],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert proc.stdout.split()[0] == "delivered", proc.stdout + proc.stderr[-1000:]
    assert list(arrays.iterdir()) == []


_CLOSE_DURING_CONSTRUCTION = (
    _HEAD
    + """
stops, box = [], []

class Probed(WorkerClient):
    def _write_stop(self):
        stops.append(threading.current_thread().name)
        super()._write_stop()

    def _handshake(self, *args, **kwargs):
        box.append(self)
        closer = threading.Thread(target=self.close, name="closer")
        closer.start()
        closer.join(30)
        assert not closer.is_alive()
        return super()._handshake(*args, **kwargs)

try:
    Probed(source_tree_python(), "senselab_worker.fake", init={}, array_parent=parent)
except WorkerLost:
    outcome = "WorkerLost"
else:
    outcome = "constructed"
proc = box[0]._proc
print(outcome, stops, proc.returncode is not None, box[0]._array_dir.exists(), flush=True)
"""
)


@pytest.mark.contract
def test_a_close_from_another_thread_during_construction_is_graceful_and_leaves_nothing(tmp_path: Path) -> None:
    """The stop line is written, the worker is reaped and the array directory is removed, as before."""
    arrays = tmp_path / "arrays"
    arrays.mkdir()
    proc = subprocess.run(
        [sys.executable, "-c", _CLOSE_DURING_CONSTRUCTION, str(arrays)],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert proc.stdout.split() == ["WorkerLost", "['closer']", "True", "False"], proc.stdout
    assert list(arrays.iterdir()) == []
