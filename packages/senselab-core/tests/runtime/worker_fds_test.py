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
before = open_fds()
client = new_client()
client.run([{"id": "a", "inputs": {}}], {})
client.close()
for thread in threading.enumerate():
    if thread.name == "senselab-lines-relay":
        thread.join(30)
print(sorted(open_fds() - before))
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
