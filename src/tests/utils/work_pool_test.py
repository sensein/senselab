"""The work pool: exclusive claims under contention, leases, heartbeats, reaping, retries and the worker."""

from __future__ import annotations

import errno
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

from senselab.utils import work_pool as wp

NFS_ENV = "SENSELAB_WORK_POOL_NFS_DIR"
SRC = Path(wp.__file__).resolve().parents[2]

# One claimer: waits for the start time, claims and completes until the queue is empty, then records its wins.
_CLAIMER = r"""
import json, os, sys, time
from pathlib import Path
from senselab.utils.work_pool import WorkPool

root, me, start, hold = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
pool = WorkPool(root)
while time.time() < start:
    time.sleep(0.005)
won, deadline = [], time.time() + 300
while time.time() < deadline:
    lease = pool.claim(["main"], me)
    if lease is None:
        if not os.listdir(root / "queue"):
            break
        time.sleep(0.05)
        continue
    won.append([lease.item.id, lease.item.attempt])
    if hold:
        time.sleep(hold)
    assert pool.complete(lease, {"exit_code": 0})
(root.parent / "wins" / (me + ".json")).write_text(json.dumps(won))
"""

# A child that records its pid, item id and attempt, then sleeps.
_SLEEPER = (
    "import json, os, sys, time; "
    "open(sys.argv[1], 'w').write(json.dumps({'pid': os.getpid(), 'id': os.environ['POOL_ITEM_ID'], "
    "'attempt': os.environ['POOL_ATTEMPT']})); time.sleep(60)"
)


def _env() -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")])}


def _pool(root: Path, **kwargs: Any) -> wp.WorkPool:  # noqa: ANN401
    settings: dict[str, Any] = {"lease_s": 60.0, "heartbeat_s": 1.0, **kwargs}
    return wp.WorkPool.create(root / "pool", **settings)


def _item(i: int, cls: str = "main", command: tuple[str, ...] = ("true",), **kwargs: Any) -> wp.Item:  # noqa: ANN401
    return wp.Item(id=f"item-{i:04d}", resource_class=cls, command=command, **kwargs)


def _names(path: Path) -> list[str]:
    return sorted(os.listdir(path))


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _expire(pool: wp.WorkPool, item_id: str) -> None:
    beat = pool.root / "leases" / item_id / "heartbeat"
    os.utime(beat, (0, 0))


def _worker(pool: wp.WorkPool, *extra: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen(  # noqa: S603 — this interpreter
        [sys.executable, "-m", "senselab.utils.work_pool", "work", str(pool.root), "--classes", "main", *extra],
        env=_env(),
    )


def _wait_for(path: Path, timeout_s: float = 30.0) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            return _read(path)
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.05)
    pytest.fail(f"{path.name} did not appear")


def _gone(pid: int) -> bool:
    """Whether a process is gone (a zombie counts as gone: it runs nothing)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z"
    except (FileNotFoundError, IndexError):
        return True


def _eventually(check: Any, timeout_s: float = 15.0) -> bool:  # noqa: ANN401
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if check():
            return True
        time.sleep(0.05)
    return bool(check())


def _race(root: Path, *, processes: int, items: int, hold: float = 0.0) -> None:
    """Publish ``items`` items and let ``processes`` claimers race for them; assert each was claimed once."""
    pool = _pool(root)
    pool.publish([_item(i) for i in range(items)])
    (root / "wins").mkdir()
    start = time.time() + 2.0
    procs = [
        subprocess.Popen(  # noqa: S603 — this interpreter
            [sys.executable, "-c", _CLAIMER, str(pool.root), f"w{k}", str(start), str(hold)], env=_env()
        )
        for k in range(processes)
    ]
    assert [p.wait(timeout=600) for p in procs] == [0] * processes
    claims: dict[str, int] = {}
    for name in _names(root / "wins"):
        wins: list[list[Any]] = json.loads((root / "wins" / name).read_text())
        for item_id, attempt in wins:
            assert attempt == 1
            claims[item_id] = claims.get(item_id, 0) + 1
    assert sorted(claims) == [f"item-{i:04d}" for i in range(items)]
    assert set(claims.values()) == {1}
    assert len(_names(pool.root / "done")) == items
    assert _names(pool.root / "queue") == _names(pool.root / "leases") == _names(pool.root / "tmp") == []


def test_publish_then_claim_is_exclusive_under_contention(tmp_path: Path) -> None:
    """16 processes racing for 200 items on local storage claim each item exactly once."""
    _race(tmp_path, processes=16, items=200)


def test_claims_are_exclusive_on_shared_storage() -> None:
    """The same race on the shared (NFS) directory named by SENSELAB_WORK_POOL_NFS_DIR."""
    base = os.environ.get(NFS_ENV)
    if not base:
        pytest.skip(f"{NFS_ENV} is not set")
    root = Path(base) / f"work-pool-race-{uuid.uuid4().hex[:12]}"
    root.mkdir(parents=True)
    try:
        _race(root, processes=16, items=200)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_a_claim_is_never_visible_without_owner_and_heartbeat(tmp_path: Path) -> None:
    """A watcher polling leases/ during 200 claims never sees a lease dir without its files."""
    pool = _pool(tmp_path)
    pool.publish([_item(i) for i in range(200)])
    (tmp_path / "wins").mkdir()
    leases = pool.root / "leases"
    seen, bad, stop = [0], [], threading.Event()

    def watch() -> None:
        while not stop.is_set():
            for name in os.listdir(leases):
                for part in ("item.json", "owner.json", "heartbeat", "attempt"):
                    if not (leases / name / part).exists() and (leases / name).is_dir():
                        bad.append((name, part))
                seen[0] += 1

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    start = time.time() + 1.0
    procs = [
        subprocess.Popen(  # noqa: S603 — this interpreter
            [sys.executable, "-c", _CLAIMER, str(pool.root), f"w{k}", str(start), "0.01"], env=_env()
        )
        for k in range(4)
    ]
    assert [p.wait(timeout=300) for p in procs] == [0] * 4
    stop.set()
    watcher.join()
    assert seen[0] > 0
    assert bad == []
    assert len(_names(pool.root / "done")) == 200


def test_a_lost_claim_kills_the_child(tmp_path: Path) -> None:
    """A lease reaped under a live worker: the worker kills its child and writes no done or failed record."""
    pool = _pool(tmp_path, heartbeat_s=0.5, max_interrupts=1)
    record = tmp_path / "child.json"
    [item] = pool.publish([_item(0, command=(sys.executable, "-c", _SLEEPER, str(record)))])
    worker = _worker(pool, "--linger", "1", "--poll", "0.2")
    child = _wait_for(record)
    reaped = 0
    for _ in range(20):  # the worker's heartbeat may land between the expiry and the reap; try again
        _expire(pool, item.id)
        reaped = pool.reap()
        if reaped:
            break
    assert reaped == 1
    t0 = time.time()
    assert worker.wait(timeout=30) == 0
    assert time.time() - t0 < 15
    assert _gone(child["pid"])
    assert _names(pool.root / "done") == _names(pool.root / "failed") == []
    assert _names(pool.root / "interrupted") == [f"{item.id}.1.json"]
    assert _read(pool.root / "interrupted" / f"{item.id}.1.json")["reason"] == "reap"
    assert _read(pool.root / "dead" / f"{item.id}.json")["dead_reason"] == "interrupts"


def test_max_interrupts_defaults_to_six_and_sends_the_item_to_dead(tmp_path: Path) -> None:
    """Each release and each reap increments the attempt; the sixth interruption sends the item to dead/."""
    pool = _pool(tmp_path)
    assert pool.max_interrupts == 6
    assert _read(pool.root / "pool.json")["max_interrupts"] == 6
    [item] = pool.publish([_item(0)])
    for k in range(1, 7):
        lease = pool.claim(["main"], "w")
        assert lease is not None and lease.item.attempt == k
        assert (pool.root / "leases" / item.id / "attempt").read_text().strip() == str(k)
        if k % 2:
            _expire(pool, item.id)
            assert pool.reap() == 1
        else:
            assert pool.release(lease) == ("dead" if k == 6 else "requeued")
        if k < 6:
            assert _read(pool.root / "queue" / f"{item.id}.json")["attempt"] == k + 1
    assert _names(pool.root / "queue") == _names(pool.root / "failed") == []
    assert len(_names(pool.root / "interrupted")) == 6
    assert _read(pool.root / "dead" / f"{item.id}.json")["dead_reason"] == "interrupts"
    assert pool.claim(["main"], "w") is None


class _Died(BaseException):
    """A worker dying mid-claim."""


def test_a_dead_workers_lease_is_republished_after_lease_s(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Leases of dead workers and claims of workers that died between renames are requeued after lease_s.

    Both come back with attempt 2 once lease_s has passed, and not before; claim() reaps first.
    """
    pool = _pool(tmp_path, lease_s=1.5, heartbeat_s=0.5)
    first, second = pool.publish([_item(0), _item(1)])
    script = "import sys; from senselab.utils.work_pool import WorkPool; WorkPool(sys.argv[1]).claim(['main'], 'w0')"
    subprocess.run([sys.executable, "-c", script, str(pool.root)], env=_env(), check=True)  # noqa: S603
    assert _names(pool.root / "leases") == [first.id]

    rename = os.rename

    def dies_publishing_the_claim(src: str | Path, dst: str | Path) -> None:
        if Path(dst).parent.name == "leases":
            raise _Died
        rename(src, dst)

    monkeypatch.setattr(wp.os, "rename", dies_publishing_the_claim)
    with pytest.raises(_Died):
        pool.claim(["main"], "w1")
    monkeypatch.setattr(wp.os, "rename", rename)
    assert _names(pool.root / "queue") == []
    assert [n for n in _names(pool.root / "tmp") if (pool.root / "tmp" / n / "item.json").exists()]

    assert pool.reap() == 0
    assert _names(pool.root / "leases") == [first.id]
    time.sleep(2.0)
    lease = pool.claim(["main"], "w2")  # claim reaps first: both items are back, attempt 2
    assert lease is not None and lease.item.attempt == 2
    assert {lease.item.id, *[n.removesuffix(".json") for n in _names(pool.root / "queue")]} == {first.id, second.id}
    assert _read(pool.root / "queue" / f"{({first.id, second.id} - {lease.item.id}).pop()}.json")["attempt"] == 2
    assert sorted(_read(pool.root / "interrupted" / n)["reason"] for n in _names(pool.root / "interrupted")) == [
        "reap",
        "reap",
    ]
    assert _names(pool.root / "tmp") == []

    # On NFS a lease dir renamed away by another node between lookup and open reports ESTALE: it reads as gone.
    def stale(opener: Any) -> Any:  # noqa: ANN401
        def wrapped(path: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            if Path(path).parent.parent.name == "leases":
                raise OSError(errno.ESTALE, os.strerror(errno.ESTALE), str(path))
            return opener(path, *args, **kwargs)

        return wrapped

    monkeypatch.setattr(wp.os, "open", stale(os.open))
    monkeypatch.setattr(wp, "open", stale(open), raising=False)
    os.utime(lease.path / "heartbeat", (0, 0))
    assert pool.reap() == 0
    assert not pool.heartbeat(lease)
    monkeypatch.undo()
    assert pool.heartbeat(lease)


def test_sigterm_releases_without_spending_a_retry(tmp_path: Path) -> None:
    """SIGTERM to a worker stops its child and puts the item back with attempt 2 and no failure recorded."""
    pool = _pool(tmp_path, heartbeat_s=0.5)
    record = tmp_path / "child.json"
    [item] = pool.publish([_item(0, command=(sys.executable, "-c", _SLEEPER, str(record)))])
    worker = _worker(pool, "--linger", "1", "--poll", "0.2")
    child = _wait_for(record)
    assert child["id"] == item.id and child["attempt"] == "1"
    worker.send_signal(signal.SIGTERM)
    assert worker.wait(timeout=30) == 143
    assert _gone(child["pid"])
    assert _names(pool.root / "failed") == _names(pool.root / "done") == _names(pool.root / "leases") == []
    assert _read(pool.root / "queue" / f"{item.id}.json")["attempt"] == 2
    assert _read(pool.root / "interrupted" / f"{item.id}.1.json")["reason"] == "release"


def test_failures_spend_the_budget_then_go_dead(tmp_path: Path) -> None:
    """max_attempts failures send an item to dead/, through the API and through a worker."""
    pool = _pool(tmp_path)
    one, two = pool.publish([_item(0), _item(1, max_attempts=1)])
    lease = pool.claim(["main"], "w")
    assert lease is not None and lease.item.id == one.id
    assert pool.fail(lease, 3) == "retry"
    assert _read(pool.root / "queue" / f"{one.id}.json")["attempt"] == 2
    assert _read(pool.root / "failed" / f"{one.id}.1.json")["exit_code"] == 3
    lease = pool.claim(["main"], "w")
    assert lease is not None and lease.item.id == one.id and lease.item.attempt == 2
    assert pool.fail(lease, 3) == "dead"
    assert _read(pool.root / "dead" / f"{one.id}.json")["dead_reason"] == "failures"
    lease = pool.claim(["main"], "w")
    assert lease is not None and lease.item.id == two.id
    assert pool.fail(lease, 1) == "dead"
    assert pool.claim(["main"], "w") is None

    pool = wp.WorkPool.create(tmp_path / "worker-pool", lease_s=60.0, heartbeat_s=1.0)
    [item] = pool.publish([_item(2, command=("sh", "-c", 'test "$POOL_ATTEMPT" = 9 || exit 5'))])
    assert _worker(pool, "--linger", "1", "--poll", "0.1").wait(timeout=60) == 0
    assert [_read(pool.root / "failed" / n)["exit_code"] for n in _names(pool.root / "failed")] == [5, 5]
    assert _read(pool.root / "dead" / f"{item.id}.json")["dead_reason"] == "failures"
    assert _names(pool.root / "queue") == _names(pool.root / "done") == []


def test_complete_is_idempotent_and_atomic(tmp_path: Path) -> None:
    """Complete writes one done record once; a finished or lost lease writes nothing anywhere."""
    pool = _pool(tmp_path)
    first, second = pool.publish([_item(0), _item(1)])
    lease = pool.claim(["main"], "w")
    assert lease is not None and lease.item.id == first.id
    assert pool.complete(lease, {"exit_code": 0, "start": 1.0, "end": 2.0})
    done = pool.root / "done" / f"{first.id}.json"
    record, raw = _read(done), done.read_bytes()
    assert record["id"] == first.id and record["attempt"] == 1 and record["exit_code"] == 0
    assert record["resource_class"] == "main" and record["worker"] == "w" and record["host"]
    assert record["start"] == 1.0 and record["end"] == 2.0
    assert not pool.complete(lease, {"exit_code": 0})
    assert pool.fail(lease, 1) == "lost" and pool.release(lease) == "lost" and not pool.heartbeat(lease)
    assert done.read_bytes() == raw

    stale = pool.claim(["main"], "w-stalled")
    assert stale is not None and stale.item.id == second.id
    _expire(pool, second.id)
    assert pool.reap() == 1
    fresh = pool.claim(["main"], "w-new")
    assert fresh is not None and fresh.item.attempt == 2
    assert not pool.complete(stale, {"exit_code": 0})
    assert pool.fail(stale, 1) == "lost" and pool.release(stale) == "lost" and not pool.heartbeat(stale)
    assert _read(pool.root / "leases" / second.id / "owner.json")["token"] == fresh.token
    assert _names(pool.root / "failed") == [] and _names(pool.root / "done") == [f"{first.id}.json"]
    assert pool.complete(fresh, {"exit_code": 0})
    assert _read(pool.root / "done" / f"{second.id}.json")["attempt"] == 2
    assert _names(pool.root / "tmp") == _names(pool.root / "leases") == _names(pool.root / "queue") == []
    assert all(not n.startswith(".") for n in _names(pool.root / "done"))


def test_resource_classes_filter_claims(tmp_path: Path) -> None:
    """A worker claims only items of the classes it names."""
    pool = _pool(tmp_path)
    pool.publish([_item(0, "gpu-a"), _item(1), _item(2)])
    assert pool.claim(["spare"], "w") is None
    lease = pool.claim(["gpu-a"], "w")
    assert lease is not None and lease.item.id == "item-0000"
    assert pool.claim(["gpu-a"], "w") is None
    got = [pool.claim(["spare", "main"], "w") for _ in range(3)]
    assert sorted(lease.item.id for lease in got if lease is not None) == ["item-0001", "item-0002"]
    assert got[2] is None


def test_status_counts_per_state_and_class(tmp_path: Path) -> None:
    """status() counts items per state and per class and writes status.json."""
    pool = _pool(tmp_path)
    pool.publish([_item(0), _item(1), _item(2, max_attempts=1), _item(3, "gpu-a"), _item(4, "gpu-a")])
    with pytest.raises(ValueError, match="item-0001"):
        pool.publish([_item(1)])
    lease = pool.claim(["main"], "w")
    assert lease is not None and pool.complete(lease, {"exit_code": 0})
    pool.claim(["gpu-a"], "w")
    released, failed = pool.claim(["main"], "w"), pool.claim(["main"], "w")
    assert released is not None and failed is not None and failed.item.id == "item-0002"
    assert pool.release(released) == "requeued"
    assert pool.fail(failed, 2) == "dead"
    status = pool.status()
    assert status["counts"] == {
        "queued": 2,
        "leased": 1,
        "done": 1,
        "dead": 1,
        "failed_attempts": 1,
        "interrupted": 1,
        "staged": 0,
        "conflicts": 0,
    }
    assert status["classes"] == {
        "main": {"queued": 1, "leased": 0, "done": 1, "dead": 1},
        "gpu-a": {"queued": 1, "leased": 1, "done": 0, "dead": 0},
    }
    assert _read(pool.root / "status.json") == status


def test_a_put_back_does_not_strip_a_live_owner(tmp_path: Path) -> None:
    """A lease renamed aside and back (a reaper's put-back) stays the owner's.

    Heartbeat, complete, and a worker that keeps its child.
    """
    pool = _pool(tmp_path, heartbeat_s=0.2)
    first, second = pool.publish([_item(0), _item(1, "spare")])
    aside = tmp_path / "aside"

    def aside_and_back(path: Path, seconds: float) -> threading.Thread:
        os.rename(path, aside)

        def put_back() -> None:
            time.sleep(seconds)
            os.rename(aside, path)

        back = threading.Thread(target=put_back)
        back.start()
        return back

    lease = pool.claim(["main"], "w")
    assert lease is not None
    back = aside_and_back(lease.path, 0.4)
    assert pool.heartbeat(lease)
    back.join()
    back = aside_and_back(lease.path, 0.4)
    assert pool.complete(lease, {"exit_code": 0})
    back.join()
    assert _read(pool.root / "done" / f"{first.id}.json")["attempt"] == 1

    record = tmp_path / "child.json"
    napper = _SLEEPER.replace("time.sleep(60)", "time.sleep(6)")
    worker_pool = wp.WorkPool.create(tmp_path / "worker-pool", lease_s=60.0, heartbeat_s=0.2)
    [item] = worker_pool.publish([_item(2, command=(sys.executable, "-c", napper, str(record)))])
    worker = _worker(worker_pool, "--linger", "0", "--poll", "0.1")
    child = _wait_for(record)
    for _ in range(3):  # each window spans a heartbeat; each gap is longer than the owner's re-check wait
        aside_and_back(worker_pool.root / "leases" / item.id, 0.3).join()
        time.sleep(1.2)
    assert worker.wait(timeout=30) == 0
    assert _gone(child["pid"])
    assert _read(worker_pool.root / "done" / f"{item.id}.json")["exit_code"] == 0
    assert _names(worker_pool.root / "interrupted") == []


def test_a_reaper_puts_back_a_lease_claimed_again_under_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The reaper reads the owner's token before its rename.

    A lease that changed hands is put back, even when its heartbeat also looks old.
    """
    pool = _pool(tmp_path)
    [item] = pool.publish([_item(0)])
    stale = pool.claim(["main"], "w-stalled")
    assert stale is not None
    _expire(pool, item.id)
    rename = os.rename
    fresh: list[wp.Lease | None] = []

    def swap_first(src: Any, dst: Any) -> None:  # noqa: ANN401
        if Path(dst).name.startswith("reap-") and Path(src).parent.name == "leases" and not fresh:
            monkeypatch.setattr(wp.os, "rename", rename)
            assert pool.release(stale) == "requeued"
            fresh.append(pool.claim(["main"], "w-new"))
            _expire(pool, item.id)
            monkeypatch.setattr(wp.os, "rename", swap_first)
        rename(src, dst)

    monkeypatch.setattr(wp.os, "rename", swap_first)
    assert pool.reap() == 0
    monkeypatch.setattr(wp.os, "rename", rename)
    [lease] = fresh
    assert lease is not None and lease.item.attempt == 2
    assert _read(pool.root / "leases" / item.id / "owner.json")["token"] == lease.token
    assert _names(pool.root / "interrupted") == [f"{item.id}.1.json"]
    assert pool.heartbeat(lease) and pool.complete(lease, {"exit_code": 0})


def test_a_stalled_heartbeat_thread_cannot_touch_the_next_item(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A heartbeat thread stalled past the join timeout, then reporting its lease lost, leaves the next run alone."""
    monkeypatch.setattr(wp, "JOIN_GRACE_S", 0.1, raising=False)
    monkeypatch.setattr(wp, "PUTBACK_WAIT_S", 0.1, raising=False)
    pool = _pool(tmp_path, heartbeat_s=0.2)
    record = tmp_path / "child.json"
    napper = _SLEEPER.replace("time.sleep(60)", "time.sleep(2)")
    first, second = pool.publish(
        [_item(0, command=("sleep", "0.6")), _item(1, command=(sys.executable, "-c", napper, str(record)))]
    )
    gate, real = threading.Event(), pool.heartbeat

    def stalls_for_the_first(lease: wp.Lease) -> bool:
        if lease.item.id == first.id:
            gate.wait(30)
            return False
        return real(lease)

    monkeypatch.setattr(pool, "heartbeat", stalls_for_the_first)
    worker = wp._Worker(pool, ["main"], "w", 0, 0.1)
    lease = pool.claim(["main"], "w")
    assert lease is not None and lease.item.id == first.id
    worker._run(lease)
    assert _read(pool.root / "done" / f"{first.id}.json")["exit_code"] == 0

    def open_the_gate() -> None:
        _wait_for(record)
        gate.set()

    opener = threading.Thread(target=open_the_gate)
    opener.start()
    lease = pool.claim(["main"], "w")
    assert lease is not None and lease.item.id == second.id
    worker._run(lease)
    opener.join()
    assert _read(pool.root / "done" / f"{second.id}.json")["exit_code"] == 0


def test_the_childs_whole_process_group_is_killed(tmp_path: Path) -> None:
    """What the child leaves running in its process group dies when the item finishes."""
    pool = _pool(tmp_path)
    grandchild = tmp_path / "grandchild.pid"
    [item] = pool.publish([_item(0, command=("sh", "-c", f"sleep 60 & echo $! > {grandchild}; exit 0"))])
    assert _worker(pool, "--linger", "0", "--poll", "0.1").wait(timeout=60) == 0
    assert _read(pool.root / "done" / f"{item.id}.json")["exit_code"] == 0
    pid = int(grandchild.read_text())
    assert _eventually(lambda: _gone(pid))


def test_reaping_uses_the_storage_clock_not_the_local_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A local clock an hour ahead reaps no live lease.

    A staged claim named an hour ago is not reaped while its dir is fresh; a lease that has really expired is reaped
    with the local clock an hour behind.
    """
    pool = _pool(tmp_path)
    first, second = pool.publish([_item(0), _item(1)])
    lease = pool.claim(["main"], "w0")
    assert lease is not None and lease.item.id == first.id
    real_now, real_rename = wp._now, os.rename
    monkeypatch.setattr(wp, "_now", lambda: real_now() + 3600)
    assert pool.reap() == 0
    assert _names(pool.root / "leases") == [first.id]

    def dies_publishing_the_claim(src: Any, dst: Any) -> None:  # noqa: ANN401
        if Path(dst).parent.name == "leases":
            raise _Died
        real_rename(src, dst)

    monkeypatch.setattr(wp, "_now", lambda: real_now() - 3600)
    monkeypatch.setattr(wp.os, "rename", dies_publishing_the_claim)
    with pytest.raises(_Died):
        pool.claim(["main"], "w1")
    monkeypatch.setattr(wp.os, "rename", real_rename)
    monkeypatch.setattr(wp, "_now", real_now)
    assert pool.reap() == 0
    assert [n for n in _names(pool.root / "tmp") if (pool.root / "tmp" / n / "item.json").exists()]
    monkeypatch.setattr(wp, "_now", lambda: real_now() - 3600)
    _expire(pool, first.id)
    assert pool.reap() == 1
    assert _read(pool.root / "queue" / f"{first.id}.json")["attempt"] == 2


def test_duplicates_go_to_conflicts_and_never_run_twice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Queued copies of done or dead items, and an item whose lease dir exists, go to conflicts/.

    Publish sees items held in tmp/, links exclusively and reads back a link that landed but raised.
    """
    pool = _pool(tmp_path)
    done_item, dead_item, leased_item = pool.publish([_item(0), _item(1), _item(2)])
    lease = pool.claim(["main"], "w")
    assert lease is not None and lease.item.id == done_item.id and pool.complete(lease, {"exit_code": 0})
    for _ in range(2):
        lease = pool.claim(["main"], "w")
        assert lease is not None and lease.item.id == dead_item.id
        pool.fail(lease, 1)
    assert (pool.root / "dead" / f"{dead_item.id}.json").exists()
    for item in (done_item, dead_item):
        (pool.root / "queue" / f"{item.id}.json").write_text(json.dumps(item.to_dict()))
    other = pool.root / "leases" / leased_item.id
    other.mkdir()
    (other / "owner.json").write_text(json.dumps({"token": "other"}))
    (other / "item.json").write_text(json.dumps(leased_item.to_dict()))
    (other / "heartbeat").touch()
    done_raw = (pool.root / "done" / f"{done_item.id}.json").read_bytes()

    assert pool.claim(["main"], "w") is None
    assert sorted(n.split(".")[0:2] for n in _names(pool.root / "conflicts")) == [
        [done_item.id, "done"],
        [dead_item.id, "dead"],
        [leased_item.id, "leased"],
    ]
    assert _names(pool.root / "queue") == []
    assert _read(other / "owner.json")["token"] == "other"
    assert (pool.root / "done" / f"{done_item.id}.json").read_bytes() == done_raw
    assert pool.status()["counts"]["conflicts"] == 3

    rename = os.rename

    def dies_publishing_the_claim(src: Any, dst: Any) -> None:  # noqa: ANN401
        if Path(dst).parent.name == "leases":
            raise _Died
        rename(src, dst)

    pool.publish([_item(3, "spare")])
    monkeypatch.setattr(wp.os, "rename", dies_publishing_the_claim)
    with pytest.raises(_Died):
        pool.claim(["spare"], "w")
    monkeypatch.setattr(wp.os, "rename", rename)
    with pytest.raises(ValueError, match="item-0003"):
        pool.publish([_item(3, "spare")])

    link = os.link

    def lands_then_raises(src: Any, dst: Any) -> None:  # noqa: ANN401
        link(src, dst)
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(dst))

    def another_publisher_first(src: Any, dst: Any) -> None:  # noqa: ANN401
        Path(dst).write_text(json.dumps(_item(5).to_dict()))
        link(src, dst)

    monkeypatch.setattr(wp.os, "link", lands_then_raises)
    [published] = pool.publish([_item(4)])
    assert _read(pool.root / "queue" / f"{published.id}.json")["payload_sha256"] == published.payload_sha256
    monkeypatch.setattr(wp.os, "link", another_publisher_first)
    with pytest.raises(ValueError, match="item-0005"):
        pool.publish([_item(5)])
    monkeypatch.setattr(wp.os, "link", link)
    assert "placed" not in _read(pool.root / "queue" / "item-0005.json")
    assert [n for n in _names(pool.root / "tmp") if n.endswith(".json")] == []


def test_renames_and_links_that_land_and_then_raise_are_read_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retransmitted NFS call that did its work but reports an error.

    Publish, claim, complete, release, fail and the move to dead/ all still succeed exactly once.
    """
    pool = _pool(tmp_path)
    rename, link = os.rename, os.link

    def rename_lands_then_raises(src: Any, dst: Any) -> None:  # noqa: ANN401
        rename(src, dst)
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(src))

    def link_lands_then_raises(src: Any, dst: Any) -> None:  # noqa: ANN401
        link(src, dst)
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(dst))

    monkeypatch.setattr(wp.os, "rename", rename_lands_then_raises)
    monkeypatch.setattr(wp.os, "link", link_lands_then_raises)
    first, second, third = pool.publish([_item(0), _item(1, "spare"), _item(2, "gpu-a", max_attempts=1)])
    lease = pool.claim(["main"], "w")
    assert lease is not None and lease.item.id == first.id
    assert pool.complete(lease, {"exit_code": 0})
    lease = pool.claim(["spare"], "w")
    assert lease is not None and pool.release(lease) == "requeued"
    lease = pool.claim(["spare"], "w")
    assert lease is not None and lease.item.attempt == 2 and pool.fail(lease, 1) == "retry"
    lease = pool.claim(["gpu-a"], "w")
    assert lease is not None and pool.fail(lease, 1) == "dead"
    monkeypatch.setattr(wp.os, "rename", rename)
    monkeypatch.setattr(wp.os, "link", link)
    assert _read(pool.root / "done" / f"{first.id}.json")["attempt"] == 1
    assert _read(pool.root / "queue" / f"{second.id}.json")["attempt"] == 3
    assert _read(pool.root / "dead" / f"{third.id}.json")["dead_reason"] == "failures"
    assert _names(pool.root / "tmp") == _names(pool.root / "leases") == _names(pool.root / "conflicts") == []


def test_a_stalled_finisher_loses_cleanly_to_the_reaper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A finisher stalled in its hand-off loses it to the tmp/ reaper and writes nothing.

    A stale complete() that renamed a re-claimed lease aside puts it back; a hand-off left after done/ was written
    is dropped, not run.
    """
    pool = _pool(tmp_path, lease_s=1.0, heartbeat_s=0.3)
    first, second, third = pool.publish([_item(0), _item(1, "spare"), _item(2, "gpu-a")])

    held = pool.claim(["main"], "w")
    assert held is not None
    lease: wp.Lease = held
    write, entered, go = wp._write_json, threading.Event(), threading.Event()
    result: list[bool] = []

    def stalls_writing_done(path: Path, data: Any) -> None:  # noqa: ANN401
        if path.name == "done.json" and not go.is_set():
            entered.set()
            go.wait(30)
        write(path, data)

    monkeypatch.setattr(wp, "_write_json", stalls_writing_done)
    finisher = threading.Thread(target=lambda: result.append(pool.complete(lease, {"exit_code": 0})))
    finisher.start()
    assert entered.wait(10)
    time.sleep(1.6)
    assert pool.reap() == 1
    go.set()
    finisher.join()
    monkeypatch.setattr(wp, "_write_json", write)
    assert result == [False]
    assert not (pool.root / "done" / f"{first.id}.json").exists()
    assert _read(pool.root / "queue" / f"{first.id}.json")["attempt"] == 2
    assert _read(pool.root / "interrupted" / f"{first.id}.1.json")["reason"] == "reap"

    stale = pool.claim(["spare"], "w-stalled")
    assert stale is not None
    rename = os.rename
    fresh: list[wp.Lease | None] = []

    def reclaimed_before_the_rename(src: Any, dst: Any) -> None:  # noqa: ANN401
        if Path(dst).name.startswith("done-") and Path(src) == stale.path and not fresh:
            monkeypatch.setattr(wp.os, "rename", rename)
            _expire(pool, second.id)
            assert pool.reap() == 1
            fresh.append(pool.claim(["spare"], "w-new"))
            monkeypatch.setattr(wp.os, "rename", reclaimed_before_the_rename)
        rename(src, dst)

    monkeypatch.setattr(wp.os, "rename", reclaimed_before_the_rename)
    assert not pool.complete(stale, {"exit_code": 0})
    monkeypatch.setattr(wp.os, "rename", rename)
    [new] = fresh
    assert new is not None and new.item.attempt == 2
    assert _read(pool.root / "leases" / second.id / "owner.json")["token"] == new.token
    assert not (pool.root / "done" / f"{second.id}.json").exists()
    assert pool.heartbeat(new) and pool.complete(new, {"exit_code": 0})
    assert _read(pool.root / "done" / f"{second.id}.json")["attempt"] == 2

    last = pool.claim(["gpu-a"], "w")
    assert last is not None
    rmtree = shutil.rmtree

    def dies_cleaning_up(path: Any, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        if Path(path).name.startswith("done-"):
            raise _Died
        rmtree(path, *args, **kwargs)

    monkeypatch.setattr(wp.shutil, "rmtree", dies_cleaning_up)
    with pytest.raises(_Died):
        pool.complete(last, {"exit_code": 0})
    monkeypatch.setattr(wp.shutil, "rmtree", rmtree)
    assert (pool.root / "done" / f"{third.id}.json").exists()
    assert [n for n in _names(pool.root / "tmp") if (pool.root / "tmp" / n / "item.json").exists()]
    time.sleep(1.6)
    pool.reap()
    assert not (pool.root / "queue" / f"{third.id}.json").exists()
    assert not (pool.root / "interrupted" / f"{third.id}.1.json").exists()
    assert _names(pool.root / "tmp") == []


def test_a_re_reap_does_not_skip_an_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A release that dies after staging the next state but before placing it: the reaper re-derives attempt 2."""
    pool = _pool(tmp_path, lease_s=1.0, heartbeat_s=0.3)
    [item] = pool.publish([_item(0)])
    lease = pool.claim(["main"], "w")
    assert lease is not None
    rename, link = os.rename, os.link

    def dies_placing(real: Any) -> Any:  # noqa: ANN401
        def wrapped(src: Any, dst: Any) -> None:  # noqa: ANN401
            if Path(dst).parent.name == "queue":
                raise _Died
            real(src, dst)

        return wrapped

    monkeypatch.setattr(wp.os, "rename", dies_placing(rename))
    monkeypatch.setattr(wp.os, "link", dies_placing(link))
    with pytest.raises(_Died):
        pool.release(lease)
    monkeypatch.setattr(wp.os, "rename", rename)
    monkeypatch.setattr(wp.os, "link", link)
    assert _names(pool.root / "queue") == []
    time.sleep(1.6)
    assert pool.reap() == 1
    assert _read(pool.root / "queue" / f"{item.id}.json")["attempt"] == 2
    assert _names(pool.root / "interrupted") == [f"{item.id}.1.json"]
    assert _names(pool.root / "tmp") == []


# -- controller ----------------------------------------------------------------------------------------------------
class _FakeSlurm:
    """A fake ``runner``: records sbatch/squeue/scancel argv and keeps a job table that squeue prints."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.jobs: list[dict[str, Any]] = []
        self.next_id = 1000

    def add(self, name: str, partition: str, state: str, age_s: float = 0.0, reason: str = "None") -> str:
        self.next_id += 1
        self.jobs.append(
            {
                "id": str(self.next_id),
                "name": name,
                "partition": partition,
                "state": state,
                "submit": time.time() - age_s,
                "reason": reason,
            }
        )
        return str(self.next_id)

    def __call__(self, argv: Any) -> str:  # noqa: ANN401
        argv = list(argv)
        self.calls.append(argv)
        if argv[0] == "sbatch":
            opts = dict(a[2:].split("=", 1) for a in argv if a.startswith("--") and "=" in a)
            return self.add(opts["job-name"], opts["partition"], "PENDING", reason="Priority") + "\n"
        if argv[0] == "squeue":
            return "".join(
                f"{j['id']}|{j['partition']}|{j['state']}|"
                f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(j['submit']))}|{j['name']}|{j['reason']}\n"
                for j in self.jobs
            )
        if argv[0] == "scancel":
            self.jobs = [j for j in self.jobs if j["id"] not in argv[1:]]
            return ""
        raise AssertionError(argv)

    def sbatches(self) -> list[list[str]]:
        return [c for c in self.calls if c[0] == "sbatch"]

    def per_partition(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for argv in self.sbatches():
            part = next(a.split("=", 1)[1] for a in argv if a.startswith("--partition="))
            counts[part] = counts.get(part, 0) + 1
        return counts


def _cls(name: str = "main", **max_workers: int) -> wp.ResourceClass:
    parts = max_workers or {"main": 2, "spare": 2}
    return wp.ResourceClass(
        name=name, partitions=tuple(wp.PartitionSpec(partition=p, max_workers=n) for p, n in parts.items())
    )


def _controller(pool: wp.WorkPool, slurm: _FakeSlurm, *classes: wp.ResourceClass, **kwargs: Any) -> wp.Controller:  # noqa: ANN401
    return wp.Controller(pool, list(classes) or [_cls()], runner=slurm, **kwargs)


def test_controller_tops_up_workers_on_every_partition_of_a_class(tmp_path: Path) -> None:
    """Queued items get workers spread over the class's partitions; later ones go where workers start."""
    pool = _pool(tmp_path)
    pool.publish([_item(i) for i in range(3)])
    slurm = _FakeSlurm()
    control = _controller(pool, slurm)
    summary = control.tick()
    assert slurm.per_partition() == {"main": 2, "spare": 1}
    assert len(summary["submitted"]) == 3
    name = f"work-pool-{control.pool8}-main"
    for argv in slurm.sbatches():
        assert "--parsable" in argv and f"--job-name={name}" in argv
        wrap = argv[argv.index("--wrap") + 1]
        assert wrap.startswith("exec ")  # the worker, not a shell, gets Slurm's SIGTERM
        assert f"work_pool work {pool.root} --classes main" in wrap
    squeue = next(c for c in slurm.calls if c[0] == "squeue")
    assert "--me" in squeue and f"--name={name}" in squeue

    control.tick()  # three live workers for three queued items: nothing more
    assert len(slurm.sbatches()) == 3

    # main's workers start, spare's stays pending: the next items go to main, which has no pending workers.
    for job in slurm.jobs:
        if job["partition"] == "main":
            job["state"], job["reason"] = "RUNNING", "None"
    pool.publish([_item(i) for i in range(3, 6)])
    slurm.calls.clear()
    wider = _controller(pool, slurm, _cls(main=4, spare=4))
    wider.tick()
    assert slurm.per_partition() == {"main": 2, "spare": 1}


def test_a_partition_with_stale_pending_workers_is_skipped(tmp_path: Path) -> None:
    """Workers pending past pending_s for Priority, QOS* or ReqNodeNotAvail* skip the partition; Resources does not."""
    cases = [
        ("Priority", 1200.0, True),
        ("QOSMaxJobsPerUserLimit", 1200.0, True),
        ("ReqNodeNotAvail,_UnavailableNodes:n1", 1200.0, True),
        ("Resources", 1200.0, False),
        ("Priority", 10.0, False),
    ]
    for k, (reason, age, skipped) in enumerate(cases):
        pool = _pool(tmp_path / str(k))
        pool.publish([_item(i) for i in range(6)])
        slurm = _FakeSlurm()
        control = _controller(pool, slurm, _cls(main=4, spare=4), pending_s=600.0)
        slurm.add(f"work-pool-{control.pool8}-main", "spare", "PENDING", age_s=age, reason=reason)
        summary = control.tick()
        counts = slurm.per_partition()
        if skipped:
            assert counts == {"main": 4}, reason
            assert summary["partitions"]["spare"]["stale"] is True
        else:
            assert counts.get("spare", 0) >= 1, (reason, age)


def test_max_workers_per_partition_is_respected(tmp_path: Path) -> None:
    """No partition gets more live workers (pending plus running) of a class than its max_workers."""
    pool = _pool(tmp_path)
    pool.publish([_item(i) for i in range(10)])
    slurm = _FakeSlurm()
    control = _controller(pool, slurm, _cls(main=1, spare=2))
    control.tick()
    assert slurm.per_partition() == {"main": 1, "spare": 2}
    for job in slurm.jobs:
        job["state"] = "RUNNING"
    control.tick()
    control.tick()
    assert len(slurm.sbatches()) == 3
    slurm.jobs.pop()  # a spare worker ends: one more there
    control.tick()
    assert slurm.per_partition() == {"main": 1, "spare": 3}


def test_idle_pending_workers_are_cancelled_when_the_queue_drains(tmp_path: Path) -> None:
    """With nothing queued, the class's pending workers are cancelled; running and foreign jobs are left alone."""
    pool = _pool(tmp_path)
    pool.publish([_item(0)])
    lease = pool.claim(["main"], "w")
    assert lease is not None
    slurm = _FakeSlurm()
    control = _controller(pool, slurm, _cls(), _cls("gpu-a", main=1))
    mine = f"work-pool-{control.pool8}-main"
    pending = [slurm.add(mine, "main", "PENDING"), slurm.add(mine, "spare", "PENDING", reason="Resources")]
    running = slurm.add(mine, "spare", "RUNNING")
    foreign = slurm.add("someone-else", "main", "PENDING")
    other_pool = slurm.add("work-pool-00000000-main", "main", "PENDING")
    summary = control.tick()
    assert sorted(summary["cancelled"]) == sorted(pending)
    scancels = [c for c in slurm.calls if c[0] == "scancel"]
    assert [sorted(c[1:]) for c in scancels] == [sorted(pending)]
    assert {j["id"] for j in slurm.jobs} == {running, foreign, other_pool}
    assert slurm.sbatches() == []


def test_exclude_and_requeue_are_rendered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """gres, exclude, requeue, cpus, mem and time reach sbatch; classes load from YAML; the CLI and controller job."""
    pool = _pool(tmp_path)
    pool.publish([_item(0, "gpu-a"), _item(1, "gpu-a")])
    classes_file = tmp_path / "classes.yaml"
    classes_file.write_text(
        "classes:\n"
        "  - name: gpu-a\n"
        "    partitions:\n"
        "      - {partition: main, gres: 'gpu:1', cpus: 4, mem: 16G, time: '00:10:00', exclude: [n1, n2],"
        " requeue: true, max_workers: 1}\n"
        "      - {partition: spare, max_workers: 1}\n"
    )
    [gpu] = wp.load_classes(classes_file)
    assert gpu.name == "gpu-a" and [p.partition for p in gpu.partitions] == ["main", "spare"]
    slurm = _FakeSlurm()
    _controller(pool, slurm, gpu, log_dir=tmp_path / "logs").tick()
    by_part = {next(a for a in c if a.startswith("--partition=")): c for c in slurm.sbatches()}
    first, second = by_part["--partition=main"], by_part["--partition=spare"]
    for flag in ("--gres=gpu:1", "--cpus-per-task=4", "--mem=16G", "--time=00:10:00", "--exclude=n1,n2", "--requeue"):
        assert flag in first, flag
    assert not [a for a in second if a.startswith(("--gres", "--exclude", "--requeue"))]
    assert f"--output={tmp_path / 'logs'}/%x-%j.out" in first

    # submit_controller renders the controller's own job.
    job = wp.submit_controller(pool, classes_file, partition="main", time="00:30:00", runner=slurm, poll_s=15.0)
    argv = slurm.calls[-1]
    assert job == slurm.jobs[-1]["id"] and "--time=00:30:00" in argv and "--partition=main" in argv
    assert f"control {pool.root} --classes {classes_file} --poll 15.0" in argv[argv.index("--wrap") + 1]

    # The CLI, through fake sbatch/squeue/scancel on PATH.
    bin_dir, log = tmp_path / "bin", tmp_path / "calls.log"
    bin_dir.mkdir()
    for tool, out in (("sbatch", "4242"), ("squeue", ""), ("scancel", "")):
        script = bin_dir / tool
        script.write_text(f'#!/bin/sh\necho "{tool} $*" >> {log}\necho "{out}"\n')
        script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    assert wp.main(["control", str(pool.root), "--classes", str(classes_file), "--once"]) == 0
    lines = log.read_text().splitlines()
    assert lines[0].startswith("squeue --me -h") and any("--requeue" in line for line in lines if "sbatch" in line)


def test_each_tick_reaps_expired_leases(tmp_path: Path) -> None:
    """With no live worker, one tick puts an expired lease back in the queue (attempt 2) and submits a worker."""
    pool = _pool(tmp_path, lease_s=1.5, heartbeat_s=0.5)
    [item] = pool.publish([_item(0)])
    assert pool.claim(["main"], "gone") is not None
    _expire(pool, item.id)
    slurm = _FakeSlurm()
    summary = _controller(pool, slurm).tick()
    assert summary["reaped"] == 1
    assert _read(pool.root / "queue" / f"{item.id}.json")["attempt"] == 2
    assert _names(pool.root / "leases") == []
    assert len(slurm.sbatches()) == 1


def test_a_partition_hitting_the_release_cap_gets_no_workers(tmp_path: Path) -> None:
    """Workers failing --host-check release their item and exit; release_cap such releases pause the partition."""
    pool = _pool(tmp_path)
    pool.publish([_item(i) for i in range(4)])
    env = {**_env(), "SLURM_JOB_PARTITION": "spare"}
    for k in range(3):
        rc = subprocess.run(  # noqa: S603 — this interpreter
            [sys.executable, "-m", "senselab.utils.work_pool", "work", str(pool.root), "--classes", "main"]
            + ["--host-check", "false", "--poll", "0.1", "--linger", "0", "--worker", f"w{k}"],
            env=env,
            timeout=60,
        ).returncode
        assert rc == wp.HOST_CHECK_EXIT
    assert len(_names(pool.root / "hostcheck")) == 3
    assert _names(pool.root / "done") == [] and len(_names(pool.root / "queue")) == 4
    assert (
        sorted(_read(pool.root / "interrupted" / n)["reason"] for n in _names(pool.root / "interrupted"))
        == ["release"] * 3
    )

    slurm = _FakeSlurm()
    control = _controller(pool, slurm, _cls(main=4, spare=4), release_cap=3)
    summary = control.tick()
    assert slurm.per_partition() == {"main": 4}
    assert summary["partitions"]["spare"]["paused"] == "host-check"
    assert _read(pool.root / "status.json")["partitions"]["spare"]["paused"] == "host-check"
    assert _read(pool.root / "status.json")["partitions"]["main"]["paused"] is None

    # Releases older than the window no longer count.
    oldest = sorted(_names(pool.root / "hostcheck"))[0]
    os.utime(pool.root / "hostcheck" / oldest, (0, 0))
    slurm = _FakeSlurm()
    _controller(pool, slurm, _cls(main=4, spare=4), release_cap=3).tick()
    assert slurm.per_partition().get("spare", 0) >= 1

    # A worker whose host check passes runs items as usual.
    rc = subprocess.run(  # noqa: S603 — this interpreter
        [sys.executable, "-m", "senselab.utils.work_pool", "work", str(pool.root), "--classes", "main"]
        + ["--host-check", "true", "--poll", "0.1", "--linger", "0"],
        env=env,
        timeout=60,
    ).returncode
    assert rc == 0 and len(_names(pool.root / "done")) == 4
