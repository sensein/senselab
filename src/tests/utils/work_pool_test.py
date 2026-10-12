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
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


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
    }
    assert status["classes"] == {
        "main": {"queued": 1, "leased": 0, "done": 1, "dead": 1},
        "gpu-a": {"queued": 1, "leased": 1, "done": 0, "dead": 0},
    }
    assert _read(pool.root / "status.json") == status
