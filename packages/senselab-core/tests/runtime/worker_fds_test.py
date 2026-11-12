"""A client's descriptors: it works with more than 1024 open, and a closed client leaves none behind."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_HEAD = """
import os, resource, sys, threading
from pathlib import Path
from senselab_core.runtime.worker_client import WorkerClient
from senselab_core.testing.interpreter import source_tree_python

def open_fds():
    found = set()
    for name in os.listdir("/dev/fd"):
        if name.isdigit():
            try:
                os.fstat(int(name))
            except OSError:
                continue  # the descriptor of the listing itself
            found.add(int(name))
    return found

def new_client():
    return WorkerClient(source_tree_python(), "senselab_worker.fake", init={}, array_parent=Path(sys.argv[1]))
"""

_MANY = """
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
want = 2400
if hard != resource.RLIM_INFINITY and hard < want:
    print("skip-hard-limit", hard)
    sys.exit(0)
resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, want), hard))
held = [os.open(os.devnull, os.O_RDONLY) for _ in range(1100)]
client = new_client()
assert client._exited._r >= 1024, client._exited._r
assert client.wait_exited(0.05) is False
assert client.run([{"id": "a", "inputs": {}}], {})[0]["id"] == "a"
client.close()
assert client.wait_exited(5.0) is True
print("ok")
"""

_CLOSED = """
new_client().close()  # the process-wide supervisor opens its own descriptors on first use
before, threads = open_fds(), set(threading.enumerate())
client = new_client()
client.run([{"id": "a", "inputs": {}}], {})
client.close()
for thread in set(threading.enumerate()) - threads:
    thread.join(30)  # a descriptor a thread may still write is released when that thread is done with it
print(sorted(open_fds() - before))
"""

_STRAGGLERS = """
from senselab_core.runtime.clock import SystemClock

watched, held, resume, all_held, raised = [], [], threading.Event(), threading.Event(), []
SETTERS = 3  # the reaper on `_exited`, each pump on its `ended`

def hook(frame, event, arg):
    if event == "c_call" and arg is os.write and threading.current_thread() is not threading.main_thread():
        while frame is not None and frame.f_code.co_name != "set":
            frame = frame.f_back
        if frame is not None and any(frame.f_locals.get("self") is e for e in watched):
            held.append(threading.current_thread())
            if len(held) == SETTERS:
                all_held.set()
            resume.wait(60)

class Armed(SystemClock):
    armed = False
    def wait(self, event, timeout):
        if not self.armed:
            return super().wait(event, timeout)
        assert all_held.wait(60), held
        return event.is_set()  # every later wait is spent at once: the pipe readers do not stop in time

threading.excepthook = lambda args: raised.append(args.exc_value)
threading.setprofile(hook)
clock = Armed()
client = WorkerClient(
    source_tree_python(), "senselab_worker.fake", init={}, array_parent=Path(sys.argv[1]), clock=clock
)
events = [client._exited, client._reaped, *(pump.ended for pump in client._pumps)]
numbers = {fd for e in events for fd in (e._r, e._w)}
watched.extend([client._exited, *(pump.ended for pump in client._pumps)])
clock.armed = True
client.close()  # the setters are held with their flags set and their descriptors read
assert len(held) == SETTERS and not client._reaped.is_set(), (held, client._reaped.is_set())
probes = [os.pipe() for _ in range(16)]
resume.set()
for thread in held:
    thread.join(30)
    assert not thread.is_alive()
written = []
for r, w in probes:
    os.set_blocking(r, False)
    try:
        written.append(os.read(r, 16))
    except BlockingIOError:
        pass
    os.close(r)
    os.close(w)
print(written, raised, sorted(numbers & open_fds()))
"""


def _run(body: str, arrays: Path) -> str:
    arrays.mkdir()
    proc = subprocess.run(
        [sys.executable, "-c", _HEAD + body, str(arrays)],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert list(arrays.iterdir()) == []
    return proc.stdout.strip()


@pytest.mark.contract
def test_a_client_waits_and_closes_with_more_than_1024_descriptors_open(tmp_path: Path) -> None:
    """With 1100 descriptors held, the client's event pipes lie above 1023 and `wait_exited`, `run` and `close` work."""
    out = _run(_MANY, tmp_path / "arrays")
    if out.startswith("skip-hard-limit"):
        pytest.skip(f"the hard descriptor limit is {out.split()[-1]}")
    assert out == "ok"


@pytest.mark.contract
def test_a_closed_client_leaves_no_descriptor_open(tmp_path: Path) -> None:
    """After a request and `close()`, the process holds the descriptors it held before the client."""
    assert _run(_CLOSED, tmp_path / "arrays") == "[]"


@pytest.mark.contract
def test_setters_still_running_when_the_readers_did_not_stop_write_into_no_later_descriptor(tmp_path: Path) -> None:
    """Setters held mid-`set` while `close` gives up on the pipe readers write into no pipe opened after `close`.

    Once they have returned, the events' descriptors are closed.
    """
    assert _run(_STRAGGLERS, tmp_path / "arrays") == "[] [] []"
