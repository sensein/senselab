"""Queues of plan fragments: items by identity, slots in segments, oldest-first claims and one outcome per item."""

import errno
import gc
import multiprocessing
import os
import pickle
import threading
import uuid
from collections.abc import Iterator
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Any

import pytest
import queue_race_fixture
from queue_race_fixture import FAULT, fragment
from senselab_core.runtime.execution.executor import FaultRecord, PlanFragment
from senselab_core.runtime.execution.queue import ITEM_TAG, LABEL_TAG, Queue, QueueSet
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_worker.canonical import jcs_dumps
from senselab_worker.digest import identity_digest
from syscall_faults_fixture import await_file, count_calls, refuse_call, stale_lookup

pytestmark = pytest.mark.logic

OLD = 10**9
RACE = {"stale_after_s": 0.5, "renew_fraction": 0.25, "attribute_delay_s": 0.0}
POLL_S = RACE["stale_after_s"] * RACE["renew_fraction"]
OVERRIDE = {"pool": {"segment_items": 3}, "lease": RACE}


def _fragments(n: int, prefix: str = "a") -> list[PlanFragment]:
    return [fragment(f"{prefix}-{i}") for i in range(n)]


def _digest(fragment_: PlanFragment) -> str:
    return identity_digest(ITEM_TAG, fragment_.model_dump(mode="json"))


def _queue(root: Path, segment_items: int = 3, lease: dict[str, float] | None = None) -> Queue:
    override: dict[str, Any] = {"pool": {"segment_items": segment_items}}
    if lease is not None:
        override["lease"] = lease
    return QueueSet(root, ExecutionProfile.load(override=override)).queue("cpu")


def _drain(queue: Queue, worker: str = "w") -> list[str]:
    order: list[str] = []
    while (claim := queue.claim(worker, 1)) is not None:
        order.extend(claim.items)
        claim.commit()
    return order


@pytest.fixture
def pool() -> Iterator[Pool]:
    """One spawned process."""
    with multiprocessing.get_context("spawn").Pool(1) as processes:
        yield processes


def _age(lease: Path) -> None:
    """Stamp a lease and its tokens `OLD`, as a holder that stopped beating long ago leaves them."""
    for path in (*lease.iterdir(), lease):
        os.utime(path, (OLD, OLD))


def test_claims_oldest_first_across_segments(tmp_path: Path) -> None:
    """Seven puts take slots in three segments of three; claims come back in put order, and a release is next.

    A full segment whose slots are all resolved is retired out of `segments/`.
    """
    queue = _queue(tmp_path)
    fragments = _fragments(7)
    for each in fragments:
        queue.put(each)
    digests = [_digest(f) for f in fragments]
    assert sorted(p.name for p in (queue.path / "segments").iterdir()) == [f"{s:012d}" for s in range(3)]
    assert os.readlink(queue.path / "segments" / f"{1:012d}" / "slot-0") == digests[3]
    assert queue.depth() == 7 and queue.oldest_age_s() is not None

    first = queue.claim("w", 2)
    assert first is not None and first.items == tuple(digests[:2]) and first.slot(digests[0]) == (0, 0)
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
    retired = [(queue.path / "segments" / f"{s:012d}" / "retired").exists() for s in range(3)]
    assert retired == [True, True, False]
    queue.put(fragments[0])
    assert queue.depth() == 0


def test_a_stale_lookup_never_lets_a_retired_segment_number_be_made_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A putter's stale lookup of a retired segment's name never makes that segment again.

    A long-lived claimer drains and retires two segments; the putter, its lookup of segment 1 answered `ENOENT` once,
    puts one more item, which the same claimer then claims and commits.
    """
    putter, claimer = _queue(tmp_path), _queue(tmp_path)
    putter.put_all(_fragments(6))
    assert len(_drain(claimer)) == 6 and claimer.claim("w", 1) is None
    late = fragment("late")
    with monkeypatch.context() as patch:
        stale_lookup(patch, f"{1:012d}")
        putter.put(late)
    assert claimer.depth() == 1 and _queue(tmp_path).depth() == 1
    claim = claimer.claim("w", 1)
    assert claim is not None and claim.items == (_digest(late),) and claim.commit() == (_digest(late),)
    record = _queue(tmp_path).outcome(_digest(late))
    assert record is not None and record["outcome"] == "done"


def test_put_is_idempotent_and_names_the_queue_by_label(tmp_path: Path) -> None:
    """Two puts of one fragment write one item and one slot; a queue for a resident model lives under its label."""
    queues = QueueSet(tmp_path, ExecutionProfile.load())
    model = queues.queue("gpu:h100#model-1")
    assert model.path == tmp_path / identity_digest(LABEL_TAG, "gpu:h100#model-1")
    assert (model.path / "name").read_text() == "gpu:h100#model-1"
    (each,) = _fragments(1)
    assert model.put_all([each]) == 1 and model.put_all([each]) == 0
    assert [p.name for p in (model.path / "items").iterdir()] == [f"{_digest(each)}.json"]
    assert model.depth() == 1
    with pytest.raises(ValueError, match="queue name"):
        queues.queue("gpu h100")


def test_dead_is_the_one_outcome_and_the_item_is_not_claimed_again(tmp_path: Path) -> None:
    """A claim killed with a fault records `dead` and the fault at `outcome/<digest>`; nothing claims it again."""
    queue = _queue(tmp_path)
    queue.put(_fragments(1)[0])
    claim = queue.claim("w", 1)
    assert claim is not None
    (digest,) = claim.dead(FAULT)
    record = queue.outcome(digest)
    assert record is not None and record["outcome"] == "dead" and record["worker"] == "w"
    assert FaultRecord.model_validate(record["fault"]) == FAULT
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
    assert queue.oldest_age_s() is not None

    claim = queue.claim("w", 1)
    assert claim is not None and claim.items == (_digest(fragments[0]),) and claim.broke(claim.items[0]) == 1
    other = _queue(tmp_path).claim("v", 2)
    assert other is not None and other.items == (_digest(fragments[1]),)
    (held,) = claim.items
    assert claim.renew() and claim.commit() == (held,)
    other.release()


def _race(tmp_path: Path, pool: Pool, tier_bound: float, kinds: tuple[str, str]) -> tuple[str, list[dict[str, Any]]]:
    """Put one item; a claimer stalled after its proof has its lease aged, and a second claimer finishes the item.

    The first claimer is released only after the second is done; its report is the first of the two returned.
    """
    bound, gate = tier_bound, tmp_path / "gate"
    gate.mkdir()
    queue = _queue(tmp_path, lease=RACE)
    (each,) = _fragments(1)
    queue.put(each)
    slow = pool.apply_async(
        queue_race_fixture.claim_and_finish, (str(tmp_path), OVERRIDE, kinds[0], str(gate), (bound, POLL_S))
    )
    await_file(gate / "stalled", bound, POLL_S)
    _age(queue.path / "claims" / f"{_digest(each)}.lease")
    fast = queue_race_fixture.claim_and_finish(str(tmp_path), OVERRIDE, kinds[1])
    (gate / "release").touch()
    reports = [slow.get(bound), fast]
    assert [r["claimed"] for r in reports] == [True, True] and reports[1]["broke"] == 1, reports
    return _digest(each), reports


def test_double_claim_commits_once(tmp_path: Path, pool: Pool, tier_bound: float) -> None:
    """A claimer stalled past the stale bound after its proof is broken by a second process; one commit is kept.

    The stalled claimer's beats wait and its outcome link waits at the `os` boundary; its late link is refused as a
    duplicate.
    """
    digest, reports = _race(tmp_path, pool, tier_bound, ("done", "done"))
    assert [r["finished"] for r in reports] == [[], [digest]] and reports[0]["outcomes"] == {digest: "duplicate"}
    assert [p.name for p in (_queue(tmp_path).path / "outcome").iterdir()] == [digest]


def test_a_stalled_commit_and_a_fast_dead_give_one_outcome(tmp_path: Path, pool: Pool, tier_bound: float) -> None:
    """A commit stalled after its proof and a `dead` from the claimer that broke it: the item has one outcome."""
    digest, reports = _race(tmp_path, pool, tier_bound, ("done", "dead"))
    queue = _queue(tmp_path)
    record = queue.outcome(digest)
    assert record is not None and record["outcome"] == "dead"
    assert reports[0]["outcomes"] == {digest: "duplicate"} and reports[1]["finished"] == [digest]
    assert [p.name for p in (queue.path / "outcome").iterdir()] == [digest]


def test_a_stalled_putter_keeps_its_item_indexed(tmp_path: Path, pool: Pool, tier_bound: float) -> None:
    """A putter whose slot `symlink` stalls while others fill, finish and retire its segment still takes a slot."""
    bound, gate = tier_bound, tmp_path / "gate"
    gate.mkdir()
    late = pool.apply_async(
        queue_race_fixture.put_stalled, (str(tmp_path), OVERRIDE, "late", str(gate), (bound, POLL_S))
    )
    await_file(gate / "stalled", bound, POLL_S)
    queue = _queue(tmp_path, lease=RACE)
    queue.put_all(_fragments(3))
    assert len(_drain(queue)) == 3
    assert (queue.path / "segments" / f"{0:012d}" / "retired").exists()
    (gate / "release").touch()
    assert late.get(bound) == 1
    claim = queue.claim("w", 1)
    assert claim is not None and claim.items == (_digest(fragment("late")),)
    claim.commit()


def test_a_missing_item_stays_ready_and_a_corrupt_one_is_dead(tmp_path: Path) -> None:
    """A missing item file is an infrastructure fault and stays ready; one that is not its digest's is a dead item."""
    queue = _queue(tmp_path)
    missing, corrupt = _fragments(2)
    queue.put_all([missing, corrupt])
    item = queue.path / "items" / f"{_digest(missing)}.json"
    held = item.read_bytes()
    item.unlink()
    (queue.path / "items" / f"{_digest(corrupt)}.json").write_bytes(held)
    assert queue.claim("w", 1) is None
    assert [(d, f.fault, f.charged) for d, f in queue.faults] == [(_digest(missing), "infrastructure", False)]
    corrupt_outcome = queue.outcome(_digest(corrupt))
    assert corrupt_outcome is not None and corrupt_outcome["outcome"] == "dead"
    assert FaultRecord.model_validate(corrupt_outcome["fault"]).fault == "content"
    assert queue.outcome(_digest(missing)) is None
    item.write_bytes(held)
    claim = _queue(tmp_path).claim("w", 1)
    assert claim is not None and claim.items == (_digest(missing),)
    claim.commit()


def test_a_crash_after_the_outcome_marks_the_slot_and_does_not_run_again(tmp_path: Path) -> None:
    """A claimer that dies between its outcome link and its slot's mark: the next claimer marks it and runs nothing."""
    queue = _queue(tmp_path, lease=RACE)
    (each,) = _fragments(1)
    queue.put(each)
    child = multiprocessing.get_context("spawn").Process(
        target=queue_race_fixture.commit_then_crash, args=(str(tmp_path), OVERRIDE)
    )
    child.start()
    child.join(30)
    assert child.exitcode == 9
    record = queue.outcome(_digest(each))
    assert record is not None and record["worker"] == "crashed"
    assert not (queue.path / "segments" / f"{0:012d}" / "res-0").exists()
    _age(queue.path / "claims" / f"{_digest(each)}.lease")
    assert queue.claim("w", 1) is None
    assert (queue.path / "segments" / f"{0:012d}" / "res-0").exists()


def test_a_lost_reply_on_the_outcome_link_is_this_claims_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A link the storage made but answered `EEXIST` (a retransmission) is read back as this claim's commit."""
    queue = _queue(tmp_path)
    (each,) = _fragments(1)
    queue.put(each)
    claim = queue.claim("w", 1)
    assert claim is not None
    real_link = os.link

    def link(*args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        real_link(*args, **kwargs)
        raise FileExistsError(errno.EEXIST, "lost reply")

    monkeypatch.setattr(os, "link", link)
    assert claim.commit() == (_digest(each),) and claim.outcomes == {_digest(each): "committed"}


def test_an_interrupted_put_is_repaired_by_reconcile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A put that wrote its item and failed before its slot leaves the item unindexed until `reconcile` puts it."""
    queues = QueueSet(tmp_path, ExecutionProfile.load(override={"pool": {"segment_items": 3}}))
    queue = queues.queue("cpu")
    (each,) = _fragments(1)
    with monkeypatch.context() as patch:
        refuse_call(patch, "symlink", "slot-", OSError(errno.EIO, "interrupted"))
        with pytest.raises(OSError, match="interrupted"):
            queue.put(each)
    assert queue.depth() == 0 and queue.claim("w", 1) is None
    assert QueueSet(tmp_path).reconcile() == 1 and QueueSet(tmp_path).reconcile() == 0
    claim = queue.claim("w", 1)
    assert claim is not None and claim.items == (_digest(each),)
    claim.commit()


def _claim_cost(root: Path, items: int, claims: int, monkeypatch: pytest.MonkeyPatch) -> tuple[int, int]:
    queue = _queue(root, segment_items=50)
    queue.put_all(_fragments(items))
    calls = read = 0
    for _ in range(claims):
        with count_calls(monkeypatch) as totals:
            claim = queue.claim("w", 1)
        calls, read = calls + totals["calls"], read + totals["bytes"]
        assert claim is not None
        claim.commit()
    return calls, read


def test_claim_cost_flat_in_queue_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Claiming 100 items, through a segment retirement, costs the same syscalls and bytes read at 10^2 and 10^4."""
    small = _claim_cost(tmp_path / "small", 10**2, 100, monkeypatch)
    large = _claim_cost(tmp_path / "large", 10**4, 100, monkeypatch)
    assert small == large, (small, large)


class _ClosesOnCollection:
    """A pipe end in a reference cycle that closes its descriptor when the garbage collector frees it."""

    def __init__(self) -> None:
        """Open a pipe and refer to itself."""
        self.fds, self.cycle = os.pipe(), self

    def __del__(self) -> None:
        """Close both ends."""
        for fd in self.fds:
            os.close(fd)


def test_claim_cost_counts_only_the_code_under_test(monkeypatch: pytest.MonkeyPatch) -> None:
    """A collection inside the counted block, or another thread, adds no call: garbage left earlier is not counted."""
    _ClosesOnCollection()
    with count_calls(monkeypatch) as totals:
        gc.collect()
        worker = threading.Thread(target=os.stat, args=(os.curdir,))
        worker.start()
        worker.join()
    assert totals == {"calls": 0, "bytes": 0}


def test_fragment_is_never_pickled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An item is the fragment's JCS document and claims read it back as a `PlanFragment`; pickle is never called."""

    def refuse(*args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        raise AssertionError("a queue item was pickled")

    for name in ("dump", "dumps", "load", "loads"):
        monkeypatch.setattr(pickle, name, refuse)
    queue = _queue(tmp_path)
    (each,) = _fragments(1)
    queue.put(each)
    item = queue.path / "items" / f"{_digest(each)}.json"
    assert item.read_bytes() == jcs_dumps(each.model_dump(mode="json"))
    claim = queue.claim("w", 1)
    assert claim is not None and claim.fragments == (each,)
    claim.commit()
