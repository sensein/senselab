"""Queues of plan fragments: items by identity, index segments, oldest-first claims under leases (ADR-003)."""

import collections
import json
import multiprocessing
import os
import pickle
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import queue_race_fixture
from senselab_core.kernel.placement import NodePlacement
from senselab_core.runtime.execution.executor import FaultRecord, JobRef, PlanFragment
from senselab_core.runtime.execution.queue import ITEM_TAG, LABEL_TAG, Queue, QueueSet
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_worker.canonical import jcs_dumps
from senselab_worker.digest import identity_digest

pytestmark = pytest.mark.logic

PLAN = "ab" * 32
OLD = 10**9
RACE = {"stale_after_s": 0.5, "renew_fraction": 0.25, "attribute_delay_s": 0.0}
COUNTED = (
    "open",
    "close",
    "read",
    "write",
    "mkdir",
    "rmdir",
    "rename",
    "replace",
    "link",
    "unlink",
    "lstat",
    "stat",
    "fstat",
    "listdir",
    "scandir",
    "utime",
    "fsync",
    "fchmod",
    "dup",
)


def _fragments(n: int, prefix: str = "a") -> list[PlanFragment]:
    placement = NodePlacement(
        backend="b",
        backend_version="1",
        backend_params={},
        stack="s",
        env="host",
        device="cpu",
        batch_size=1,
        resource_class="cpu",
    )
    job = (JobRef(plan_hash=PLAN, node="n", element=(0,)),)
    return [
        PlanFragment(plan_hash=PLAN, node="n", admission=f"{prefix}-{i}", jobs=job, inputs=({},), placement=placement)
        for i in range(n)
    ]


def _digest(fragment: PlanFragment) -> str:
    return identity_digest(ITEM_TAG, fragment.model_dump(mode="json"))


def _queue(root: Path, segment_items: int = 3, lease: dict[str, float] | None = None) -> Queue:
    override: dict[str, Any] = {"pool": {"segment_items": segment_items}}
    if lease is not None:
        override["lease"] = lease
    profile = ExecutionProfile.load(override=override)
    return QueueSet(root, profile).queue("cpu")


def _drain(queue: Queue, worker: str = "w") -> list[str]:
    order: list[str] = []
    while (claim := queue.claim(worker, 1)) is not None:
        order.extend(claim.items)
        claim.commit()
    return order


def test_claims_oldest_first_across_segments(tmp_path: Path) -> None:
    """Seven puts fill three segments of three; claims come back in put order, and a release is claimed next.

    A finished segment below the head is retired.
    """
    queue = _queue(tmp_path)
    fragments = _fragments(7)
    for fragment in fragments:
        queue.put(fragment)
    digests = [_digest(f) for f in fragments]
    segments = queue.path / "segments"
    assert sorted(p.name for p in segments.glob("*.jsonl")) == [f"{s:012d}.jsonl" for s in range(3)]
    assert queue.depth() == 7 and queue.oldest_age_s() is not None

    first = queue.claim("w", 2)
    assert first is not None and first.items == tuple(digests[:2]) and first.segment(digests[0]) == 0
    second = queue.claim("w", 1)
    assert second is not None and second.items == (digests[2],)
    first.release([digests[1]])
    assert first.commit() == (digests[0],)
    again = queue.claim("w", 1)
    assert again is not None and again.items == (digests[1],)
    again.commit()
    second.commit()

    assert _drain(queue) == digests[3:]
    assert queue.depth() == 0 and queue.oldest_age_s() is None
    assert sorted(p.name for p in segments.glob("*.retired")) == [f"{s:012d}.retired" for s in range(2)]
    queue.put(fragments[0])
    assert queue.depth() == 0


def test_put_is_first_writer_wins_and_names_the_queue_by_label(tmp_path: Path) -> None:
    """Two puts of one fragment write one item; a queue for a resident model lives under its label digest."""
    queues = QueueSet(tmp_path, ExecutionProfile.load())
    model = queues.queue("gpu:h100#model-1")
    assert model.path == tmp_path / identity_digest(LABEL_TAG, "gpu:h100#model-1")
    assert (model.path / "name").read_text() == "gpu:h100#model-1"
    fragment = _fragments(1)[0]
    model.put(fragment)
    model.put(fragment)
    assert [p.name for p in (model.path / "items").iterdir()] == [f"{_digest(fragment)}.json"]
    assert model.depth() == 1
    with pytest.raises(ValueError, match="queue name"):
        queues.queue("gpu h100")


def test_dead_holds_the_fault_and_finishes_the_item(tmp_path: Path) -> None:
    """A claim moved to `dead/` records its `FaultRecord`; the item is not claimed again."""
    queue = _queue(tmp_path)
    queue.put(_fragments(1)[0])
    claim = queue.claim("w", 1)
    assert claim is not None
    fault = FaultRecord(fault="content", type="ContentFault", message="bad", gap=False, charged=True)
    (digest,) = claim.dead(fault)
    record = json.loads((queue.path / "dead" / f"{digest}.json").read_bytes())
    assert FaultRecord.model_validate(record["fault"]) == fault and record["worker"] == "w"
    assert queue.claim("w", 1) is None and queue.depth() == 0


def test_expired_claim_reclaimed(tmp_path: Path) -> None:
    """A dead claimer's stale lease is broken and the item claimed again; a live claim is never taken."""
    queue = _queue(tmp_path)
    fragments = _fragments(2)
    queue.put_all(fragments)
    dead_token = queue.path / "claims" / f"{_digest(fragments[0])}.lease" / uuid.uuid4().hex
    dead_token.mkdir(parents=True)
    for path in (dead_token, dead_token.parent):
        os.utime(path, (OLD, OLD))

    claim = queue.claim("w", 1)
    assert claim is not None and claim.items == (_digest(fragments[0]),) and claim.broke(claim.items[0]) == 1
    other = _queue(tmp_path).claim("v", 2)
    assert other is not None and other.items == (_digest(fragments[1]),)
    (held,) = claim.items
    assert claim.renew() and claim.commit() == (held,)
    other.release()


def test_double_claim_commits_once(tmp_path: Path) -> None:
    """A claimer stalled past the stale bound is broken by a second process; the item is committed once.

    The stalled claimer's beats and commit link sleep at the `os` boundary; its late commit is refused as a duplicate.
    """
    queue = _queue(tmp_path, lease=RACE)
    (fragment,) = _fragments(1)
    queue.put(fragment)
    override = {"pool": {"segment_items": 3}, "lease": RACE}
    context = multiprocessing.get_context("spawn")
    with context.Pool(2) as pool:
        assert len(set(pool.map(queue_race_fixture.ready, [0.2, 0.2], chunksize=1))) == 2
        start = time.time() + 0.3
        slow = pool.apply_async(queue_race_fixture.claim_and_commit, (str(tmp_path), override, start, 0.95))
        fast = pool.apply_async(queue_race_fixture.claim_and_commit, (str(tmp_path), override, start + 0.6, 0.0))
        reports = [slow.get(30), fast.get(30)]
    assert sum(len(r["committed"]) for r in reports) == 1, reports
    assert [p.name for p in (queue.path / "done").iterdir()] == [f"{_digest(fragment)}.json"]
    assert [r["claimed"] for r in reports] == [True, True] and reports[1]["broke"] == 1, reports
    assert list(reports[0]["outcomes"].values()) == ["duplicate"], reports
    assert queue.claim("w", 1) is None


def _counting(monkeypatch: pytest.MonkeyPatch) -> collections.Counter[str]:
    counts: collections.Counter[str] = collections.Counter()

    def wrap(name: str, real: Callable[..., Any]) -> Callable[..., Any]:
        def counted(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            counts[name] += 1
            return real(*args, **kwargs)

        return counted

    for name in COUNTED:
        if hasattr(os, name):
            monkeypatch.setattr(os, name, wrap(name, getattr(os, name)))
    return counts


def _claim_cost(root: Path, items: int, claims: int, monkeypatch: pytest.MonkeyPatch) -> int:
    queue = _queue(root, segment_items=50)
    queue.put_all(_fragments(items))
    total = 0
    for _ in range(claims):
        with monkeypatch.context() as patch:
            counts = _counting(patch)
            claim = queue.claim("w", 1)
        total += sum(counts.values())
        assert claim is not None
        claim.commit()
    return total


def test_claim_cost_flat_in_queue_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Claiming 100 items, across a segment retirement, costs the same syscalls at 10^2 and 10^4 items queued."""
    small = _claim_cost(tmp_path / "small", 10**2, 100, monkeypatch)
    large = _claim_cost(tmp_path / "large", 10**4, 100, monkeypatch)
    assert small == large, (small, large)


def test_fragment_is_never_pickled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An item is the fragment's JCS document and claims read it back as a `PlanFragment`; pickle is never called."""

    def refuse(*args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        raise AssertionError("a queue item was pickled")

    for name in ("dump", "dumps", "load", "loads"):
        monkeypatch.setattr(pickle, name, refuse)
    queue = _queue(tmp_path)
    (fragment,) = _fragments(1)
    queue.put(fragment)
    item = queue.path / "items" / f"{_digest(fragment)}.json"
    assert item.read_bytes() == jcs_dumps(fragment.model_dump(mode="json"))
    claim = queue.claim("w", 1)
    assert claim is not None and claim.fragments == (fragment,)
    claim.commit()
