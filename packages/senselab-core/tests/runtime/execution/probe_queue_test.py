"""Probe queues and worker presence: the queue's name, its serve lease and item marks, the ready mark's contents."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from queue_race_fixture import fragment
from senselab_core.runtime.calibration import HardwareClass
from senselab_core.runtime.execution.queue import PROBE_MARK, QueueSet, item_digest, probe_of, probe_queue
from senselab_core.runtime.execution.substrates import (
    WORKERS_DIR,
    class_of,
    host_token,
    mark_ready,
    read_mark,
    workers_on_host,
)
from senselab_core.runtime.hardware import device_cotenants
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_worker.canonical import jcs_dumps

LEASE = {"stale_after_s": 0.4, "renew_fraction": 0.25, "attribute_delay_s": 0.0}
CPU = HardwareClass(device="cpu", name="a cpu", cpus=2)
CUDA = HardwareClass(device="cuda", name="a gpu", memory_mib=1024, compute_capability="9.0", cpus=2)


def _queues(root: Path) -> QueueSet:
    return QueueSet(root, ExecutionProfile.load(override={"lease": LEASE}))


@pytest.mark.logic
@pytest.mark.parametrize("queue", ["cpu", "gpu:h100#" + "a" * 40])
def test_a_probe_queue_is_named_under_its_queue_and_has_its_class(queue: str) -> None:
    """`<queue>~probe-<id>` names its queue and probe, is a queue name of the queue's class, and nests no further."""
    name = probe_queue(queue, "p1")
    assert name == f"{queue}{PROBE_MARK}p1" and probe_of(name) == (queue, "p1")
    assert class_of(name) == class_of(queue) and probe_of(queue) is None
    with pytest.raises(ValueError):
        probe_queue(name, "p2")
    with pytest.raises(ValueError):
        probe_queue(queue, "has space")


@pytest.mark.logic
def test_one_worker_holds_a_probe_queues_serve_lease_until_it_is_released_or_stale(tmp_path: Path) -> None:
    """A second taker finds the lease live and gets None; once released, or stale, another takes it."""
    queue = _queues(tmp_path).queue(probe_queue("cpu", "p1"))
    first = queue.serve_lease()
    assert first is not None and first.held and queue.serve_lease() is None
    first.release()
    second = queue.serve_lease()
    assert second is not None and second.held
    second.release()
    assert queue.serve_lease() is not None


@pytest.mark.logic
def test_a_probe_items_phase_and_stratum_are_marked_once(tmp_path: Path) -> None:
    """The first mark of an item stays; an item with none reads None."""
    queue = _queues(tmp_path).queue(probe_queue("cpu", "p1"))
    item = fragment("a-0")
    queue.put(item)
    digest = item_digest(item)
    assert queue.probe_mark(digest) is None
    queue.mark_probe(digest, "SINGLE", 2)
    queue.mark_probe(digest, "BATCHED", 0)
    assert queue.probe_mark(digest) == ("SINGLE", 2)


@pytest.mark.logic
def test_a_ready_mark_carries_the_hardware_per_queue_and_names_the_host_by_digest(tmp_path: Path) -> None:
    """The mark reads back as written, holds no host name, and writes nothing but the mark."""
    mark_ready(tmp_path, {"cpu": CPU, "gpu": CUDA})
    [path] = list((tmp_path / WORKERS_DIR).iterdir())
    mark = read_mark(path)
    assert mark is not None and mark.hardware == {"cpu": CPU, "gpu": CUDA} and mark.pid == os.getpid()
    assert mark.host == host_token() and os.uname().nodename not in path.read_text()
    assert [p.name for p in tmp_path.iterdir()] == [WORKERS_DIR]


@pytest.mark.logic
def test_an_empty_or_foreign_mark_is_no_mark(tmp_path: Path) -> None:
    """A mark with nothing in it (a worker of an older kind), or of no known shape, reads as None."""
    (tmp_path / "empty").write_bytes(b"")
    (tmp_path / "list").write_text("[1]")
    (tmp_path / "partial").write_bytes(jcs_dumps({"host": "x"}))
    assert [read_mark(tmp_path / n) for n in ("empty", "list", "partial", "missing")] == [None] * 4


@pytest.mark.logic
def test_workers_on_host_counts_live_workers_of_this_host_and_itself(tmp_path: Path) -> None:
    """A peer alive on this host counts; one that exited, or one of another host, does not; this process always does."""
    assert workers_on_host(tmp_path) == 1
    peer = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    try:
        directory = tmp_path / WORKERS_DIR
        directory.mkdir()
        host = host_token()
        marks = {
            "peer": (host, peer.pid),
            "gone": (host, gone.pid),
            "away": ("0" * 64, peer.pid),
            "me": (host, os.getpid()),
        }
        for name, (where, pid) in marks.items():
            (directory / name).write_bytes(jcs_dumps({"host": where, "pid": pid, "hardware": {}}))
        assert workers_on_host(tmp_path) == 2
    finally:
        peer.kill()
        peer.wait()


@pytest.mark.logic
def test_device_cotenants_counts_the_other_processes_on_the_card_and_the_caller() -> None:
    """Processes on other cards and the caller's own do not count; the caller does, once, though it holds no memory."""
    rows = "10, GPU-a\n11, GPU-a\n12, GPU-b\n13, GPU-a\nbroken\n"
    seen: list[float] = []

    def listing(cmd: list[str], timeout_s: float) -> str:
        seen.append(timeout_s)
        return rows

    assert device_cotenants("GPU-a", {10}, timeout_s=3.0, run=listing) == 3
    assert device_cotenants("GPU-a", set(), timeout_s=3.0, run=listing) == 4
    assert device_cotenants("GPU-c", {10}, timeout_s=3.0, run=listing) == 1 and seen == [3.0] * 3

    def absent(cmd: list[str], timeout_s: float) -> str:
        raise FileNotFoundError("nvidia-smi")

    assert device_cotenants("GPU-a", {10}, timeout_s=3.0, run=absent) is None


@pytest.mark.logic
def test_a_names_listing_reads_each_queue_once(tmp_path: Path) -> None:
    """A queue made after a listing is in the next one, and the first is not read again."""
    queues = _queues(tmp_path)
    queues.queue("cpu")
    assert queues.names() == ["cpu"]
    queues.queue(probe_queue("cpu", "p1"))
    assert queues.names() == ["cpu", probe_queue("cpu", "p1")]
