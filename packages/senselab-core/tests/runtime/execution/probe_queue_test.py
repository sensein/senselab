"""Probe queues and worker presence: the queue's name, its serve lease and marked puts, the ready mark and peers."""

import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from lease_clock_fixture import ManualClock
from queue_race_fixture import fragment
from senselab_core.kernel.formats import WORKER_READY_FORMAT
from senselab_core.runtime.calibration import HardwareClass
from senselab_core.runtime.execution.observing import Presence
from senselab_core.runtime.execution.queue import (
    PROBE_MARK,
    ProbeQueueError,
    QueueSet,
    UnmarkedProbeItem,
    item_digest,
    probe_of,
    probe_queue,
)
from senselab_core.runtime.execution.substrates import (
    WORKERS_DIR,
    Peers,
    QueueReady,
    class_of,
    mark_ready,
    read_mark,
)
from senselab_core.runtime.hardware import GpuInfo, HardwareInfo, device_parties, host_secret, host_token
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_worker.canonical import jcs_dumps
from senselab_worker.digest import identity_digest

CPU = HardwareClass(device="cpu", name="a cpu", cpus=2)
CUDA = HardwareClass(device="cuda", name="a gpu", memory_mib=1024, compute_capability="9.0", cpus=2)
GPU = GpuInfo(name="a gpu", memory_bytes=2**30, driver="1", compute_capability="9.0", uuid="GPU-a", index=0)
HOST = HardwareInfo(cpu_count=2, memory_bytes=2**30, gpus=(GPU,), apple_silicon=False, platform="linux-x86_64")


def _queues(root: Path, slack: dict[str, Any]) -> QueueSet:
    lease = {"stale_after_s": slack["stale_after_s"], "renew_fraction": slack["renew_fraction"], "attribute_delay_s": 0}
    return QueueSet(root, ExecutionProfile.load(override={"lease": lease}))


@pytest.fixture
def probe(slack_tests: dict[str, Any]) -> dict[str, Any]:
    """The probe-queue test slack."""
    section: dict[str, Any] = slack_tests["probe_queue"]
    return section


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
def test_one_worker_holds_a_probe_queues_serve_lease_until_it_is_released(
    tmp_path: Path, probe: dict[str, Any]
) -> None:
    """A second taker finds the lease live and gets None; once released, another takes it."""
    queue = _queues(tmp_path, probe).queue(probe_queue("cpu", "p1"))
    first = queue.serve_lease()
    assert first is not None and first.held and queue.serve_lease() is None
    first.release()
    second = queue.serve_lease()
    assert second is not None and second.held
    second.release()
    assert queue.serve_lease() is not None


@pytest.mark.logic
def test_a_probe_item_is_put_with_its_mark_and_a_plain_put_is_refused(tmp_path: Path, probe: dict[str, Any]) -> None:
    """`put_probe` writes the mark before the slot, the first mark staying; `put` and `put_all` refuse a probe queue."""
    queues = _queues(tmp_path, probe)
    queue, plain = queues.queue(probe_queue("cpu", "p1")), queues.queue("cpu")
    item = fragment("a-0")
    queue.put_probe(item, "SINGLE", 2)
    queue.put_probe(item, "BATCHED", 0)
    assert queue.probe_mark(item_digest(item)) == ("SINGLE", 2) and queue.depth() == 1
    for refused in (lambda: queue.put(fragment("a-1")), lambda: queue.put_all([fragment("a-1")])):
        with pytest.raises(ProbeQueueError):
            refused()
    with pytest.raises(ProbeQueueError):
        plain.put_probe(fragment("a-1"), "SINGLE", 0)
    assert queue.depth() == 1


@pytest.mark.logic
def test_an_item_of_a_probe_queue_without_a_mark_is_a_typed_error(tmp_path: Path, probe: dict[str, Any]) -> None:
    """A probe item whose mark is gone raises `UnmarkedProbeItem` rather than reading as no probe."""
    queue = _queues(tmp_path, probe).queue(probe_queue("cpu", "p1"))
    item = fragment("a-0")
    queue.put_probe(item, "SINGLE", 0)
    (queue.path / "probe" / item_digest(item)).unlink()
    with pytest.raises(UnmarkedProbeItem):
        queue.probe_mark(item_digest(item))


@pytest.mark.logic
def test_a_ready_mark_carries_the_hardware_per_queue_and_names_the_host_by_token(tmp_path: Path) -> None:
    """The mark reads back as written, holds no host name, and writes nothing but the mark."""
    mark_ready(tmp_path, {"cpu": QueueReady(CPU, {}), "gpu": QueueReady(CUDA, {})})
    [path] = list((tmp_path / WORKERS_DIR).iterdir())
    mark = read_mark(path)
    assert (
        mark is not None
        and {q: r.hardware for q, r in mark.queues.items()} == {"cpu": CPU, "gpu": CUDA}
        and mark.pid == os.getpid()
    )
    assert mark.host == host_token() and os.uname().nodename not in path.read_text()
    assert [p.name for p in tmp_path.iterdir()] == [WORKERS_DIR]


@pytest.mark.logic
def test_the_host_token_is_stable_and_not_derived_from_the_host_name() -> None:
    """The token is the same in this process and another, and no hash of the host name gives it."""
    probe_code = "from senselab_core.runtime.hardware import host_token; print(host_token())"
    other = subprocess.run(
        [sys.executable, "-c", probe_code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert host_token() == host_token() == other
    for name in (os.uname().nodename, os.uname().nodename.split(".")[0]):
        assert host_token() != identity_digest("host@1", name)
    assert host_secret() == host_secret()


@pytest.mark.logic
def test_without_a_boot_id_the_secret_is_a_private_random_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Off Linux and macOS, the value is random bytes in a file only its user reads, the same on every call."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    first = host_secret(run=lambda _: "", boot_id=_no_boot_id)
    files = [p for p in tmp_path.iterdir() if p.name.startswith("senselab-host-")]
    assert first and host_secret(run=lambda _: "", boot_id=_no_boot_id) == first and len(files) == 1
    assert all(p.stat().st_mode & 0o077 == 0 for p in files)


def _no_boot_id() -> str:
    raise OSError("no boot id")


@pytest.mark.logic
def test_host_secret_is_created_once_under_a_race(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A rival that creates the file between this process's write and its link wins; both read the one value."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    rival: list[str] = []

    def link(staged: Path, path: Path) -> None:
        assert staged.read_text() != "", "the staged file is complete when it is linked"
        rival.append(host_secret(run=lambda _: "", boot_id=_no_boot_id))
        os.link(staged, path)

    mine = host_secret(run=lambda _: "", boot_id=_no_boot_id, link=link)
    files = sorted(p.name for p in tmp_path.iterdir())
    assert rival and mine == rival[0] != "" and files == [f"senselab-host-{os.getuid()}"]


@pytest.mark.logic
def test_a_stale_empty_secret_file_is_passed_over(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty file is no secret; it stays as it is, so two writers that both read it empty cannot each replace it.

    A rival that links the next name while this process writes there wins, and both read its one value.
    """
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    stale = tmp_path / f"senselab-host-{os.getuid()}"
    stale.write_bytes(b"")
    inode = stale.stat().st_ino
    rival: list[str] = []

    def link(staged: Path, path: Path) -> None:
        rival.append(host_secret(run=lambda _: "", boot_id=_no_boot_id))
        os.link(staged, path)

    mine = host_secret(run=lambda _: "", boot_id=_no_boot_id, link=link)
    assert rival and mine == rival[0] != "" and host_secret(run=lambda _: "", boot_id=_no_boot_id) == mine
    assert stale.stat().st_ino == inode and stale.read_bytes() == b""


@pytest.mark.logic
def test_an_empty_or_foreign_mark_is_no_mark(tmp_path: Path) -> None:
    """A mark with nothing in it (a worker of an older kind), or of no known shape, reads as None."""
    (tmp_path / "empty").write_bytes(b"")
    (tmp_path / "list").write_text("[1]")
    (tmp_path / "partial").write_bytes(jcs_dumps({"host": "x"}))
    assert [read_mark(tmp_path / n) for n in ("empty", "list", "partial", "missing")] == [None] * 4


def _mark(directory: Path, name: str, host: str, pid: int, started: str | None = None) -> None:
    directory.mkdir(exist_ok=True)
    (directory / name).write_bytes(
        jcs_dumps({"format": WORKER_READY_FORMAT, "host": host, "pid": pid, "started": started, "queues": {}})
    )


@pytest.mark.logic
def test_peers_are_the_other_live_workers_of_this_host_and_each_mark_is_read_once(
    tmp_path: Path, probe: dict[str, Any]
) -> None:
    """A live peer counts; an exited one, one of another host and this process do not; a mark is read once."""
    peer = subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({probe['peer_lifetime_s']})"])
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    try:
        directory = tmp_path / WORKERS_DIR
        host = host_token()
        for name, where, pid in (("peer", host, peer.pid), ("gone", host, gone.pid), ("away", "0" * 64, peer.pid)):
            _mark(directory, name, where, pid)
        _mark(directory, "me", host, os.getpid())
        peers = Peers(tmp_path)
        assert list(peers.alive()) == [peer.pid] and list(peers.alive()) == [peer.pid] and peers.reads == 4
        _mark(directory, "late", host, os.getpid())
        peers.alive()
        assert peers.reads == 5
    finally:
        peer.kill()
        peer.wait()


@pytest.mark.logic
def test_a_pid_reused_by_another_process_is_not_a_peer(tmp_path: Path, probe: dict[str, Any]) -> None:
    """A mark whose pid is alive but started at another time names a process that is not its worker."""
    from senselab_core.runtime.hardware import process_started  # noqa: PLC0415 - beside its one use

    peer = subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({probe['peer_lifetime_s']})"])
    try:
        started = process_started(peer.pid)
        if started is None:
            pytest.skip("this host has no /proc start times")
        directory = tmp_path / WORKERS_DIR
        _mark(directory, "same", host_token(), peer.pid, started)
        assert list(Peers(tmp_path).alive()) == [peer.pid]
        (directory / "same").unlink()
        _mark(directory, "other", host_token(), peer.pid, started + "0")
        assert Peers(tmp_path).alive() == {}
    finally:
        peer.kill()
        peer.wait()


@pytest.mark.logic
def test_device_parties_counts_workers_not_pids() -> None:
    """A worker holding two processes on the card is one party from every side; k workers read k."""
    rows = "10, GPU-a\n11, GPU-a\n12, GPU-a\n13, GPU-b\nbroken\n"

    def listing(cmd: list[str], timeout_s: float) -> str:
        return rows

    workers = {1: {10, 11}, 2: {12}}
    assert device_parties(["GPU-a"], workers, {99}, timeout_s=1, run=listing) == 3
    assert device_parties(["GPU-a"], {1: {10, 11}}, {12}, timeout_s=1, run=listing) == 2
    assert device_parties(["GPU-a"], {1: {10}}, {11, 12}, timeout_s=1, run=listing) == 2
    assert device_parties(["GPU-a"], {}, {99}, timeout_s=1, run=listing) == 4
    assert device_parties(["GPU-c"], workers, {99}, timeout_s=1, run=listing) == 1
    assert device_parties(["GPU-a", "GPU-b"], workers, {99}, timeout_s=1, run=listing) == 3


@pytest.mark.logic
def test_device_parties_is_none_where_nvidia_smi_cannot_answer() -> None:
    """An absent or failing `nvidia-smi` is unread, not a count."""

    def absent(cmd: list[str], timeout_s: float) -> str:
        raise FileNotFoundError("nvidia-smi")

    assert device_parties(["GPU-a"], {}, {1}, timeout_s=1, run=absent) is None


@pytest.mark.logic
def test_presence_keeps_an_unread_device_as_none_and_reads_once_per_refresh(
    tmp_path: Path, lease_clock: Callable[..., ManualClock]
) -> None:
    """A failing device read gives `device=None` beside the host count; claims within a refresh read nothing again."""
    profile = ExecutionProfile.load()
    asked: list[int] = []

    def failing(cmd: list[str], timeout_s: float) -> str:
        asked.append(1)
        raise OSError("nvidia-smi timed out")

    clock = lease_clock()
    presence = Presence(profile, tmp_path, refresh_s=10, clock=clock, report=lambda: HOST, run=failing)
    cuda = next(n for n, c in profile.resource_classes.items() if c.device == "cuda")
    first = presence.cotenancy(cuda)
    assert (first.device, first.host) == (None, 1)
    for _ in range(100):
        presence.cotenancy(cuda)
    assert len(asked) == 1
    clock.advance(11.0)
    presence.cotenancy(cuda)
    assert len(asked) == 2
    cpu = next(n for n, c in profile.resource_classes.items() if c.device == "cpu")
    assert presence.cotenancy(cpu).device == 1


@pytest.mark.logic
def test_marks_read_per_refresh_do_not_grow_with_the_claims_or_the_workers_already_read(
    tmp_path: Path, lease_clock: Callable[..., ManualClock]
) -> None:
    """With 100 peer marks, a thousand claims read each mark once, not a hundred thousand."""
    directory, host, me = tmp_path / WORKERS_DIR, host_token(), os.getpid()
    for i in range(100):
        _mark(directory, f"w{i}", host, me + 1 + i)
    clock = lease_clock()
    presence = Presence(
        ExecutionProfile.load(), tmp_path, refresh_s=1, clock=clock, report=lambda: HOST, run=lambda *_: ""
    )
    for claim in range(1000):
        clock.advance_to(claim * 0.1)
        presence.cotenancy("cpu")
    assert presence.reads == 100


@pytest.mark.logic
def test_a_names_listing_reads_each_queue_once_and_forgets_a_removed_one(tmp_path: Path, probe: dict[str, Any]) -> None:
    """A queue made after a listing is in the next one; one whose directory is gone is out of it."""
    queues = _queues(tmp_path, probe)
    queues.queue("cpu")
    assert queues.names() == ["cpu"]
    made = queues.queue(probe_queue("cpu", "p1"))
    assert queues.names() == ["cpu", probe_queue("cpu", "p1")]
    import shutil  # noqa: PLC0415 - beside its one use

    shutil.rmtree(made.path)
    assert queues.names() == ["cpu"]


@pytest.mark.logic
def test_only_another_major_of_the_ready_mark_is_no_mark(tmp_path: Path) -> None:
    """A later minor of `worker-ready@1` is read; `worker-ready@2` and an untagged mark are not."""
    for tag, readable in (
        ("worker-ready@1.0", True),
        ("worker-ready@1.3", True),
        ("worker-ready@2.0", False),
        (None, False),
    ):
        document = {"host": "x", "pid": 1, "started": None, "queues": {"cpu": {"hardware": None, "envs": {}}}}
        path = tmp_path / f"mark-{tag}"
        path.write_bytes(jcs_dumps({**document, **({} if tag is None else {"format": tag})}))
        assert (read_mark(path) is not None) is readable, tag


@pytest.mark.logic
def test_workers_of_one_slurm_allocation_have_distinct_marks_and_see_each_other(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, probe: dict[str, Any]
) -> None:
    """Two workers under one `SLURM_JOB_ID` write two marks, and each reads a co-tenancy of 2."""
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    code = (
        "import sys, time; from pathlib import Path; from senselab_core.runtime.execution.substrates import mark_ready;"
        f"mark_ready(Path(sys.argv[1])); print('ready', flush=True); time.sleep({probe['peer_lifetime_s']})"
    )
    peer = subprocess.Popen([sys.executable, "-c", code, str(tmp_path)], stdout=subprocess.PIPE, text=True)
    try:
        assert peer.stdout is not None and peer.stdout.readline().strip() == "ready"
        mark_ready(tmp_path)
        marks = [p for p in (tmp_path / WORKERS_DIR).iterdir() if not p.name.startswith(".")]
        assert len(marks) == 2
        presence = Presence(ExecutionProfile.load(), tmp_path, report=lambda: HOST, run=lambda *_: "")
        assert presence.cotenancy("cpu").host == 2
    finally:
        peer.kill()
        peer.wait()


@pytest.mark.logic
def test_a_worker_id_ends_in_a_per_process_nonce_in_or_out_of_slurm(monkeypatch: pytest.MonkeyPatch) -> None:
    """The id carries the process id and a nonce fixed for the process, so a process id reused names another worker."""
    from senselab_core.runtime.execution.substrates import worker_id  # noqa: PLC0415 - beside its one use

    code = "from senselab_core.runtime.execution.substrates import worker_id; print(worker_id())"
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    here = worker_id()
    assert here == worker_id() and here.split("-")[1] == str(os.getpid())
    ids = {subprocess.check_output([sys.executable, "-c", code], text=True).strip() for _ in range(2)}
    assert len(ids) == 2 and all(len(i.split("-")[-1]) == 8 for i in ids)
    monkeypatch.setenv("SLURM_JOB_ID", "77")
    assert worker_id().startswith("77-") and worker_id().split("-")[2] == str(os.getpid())
