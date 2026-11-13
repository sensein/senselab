"""A resumed coordinator follows the live item that already holds a job instead of putting the job again."""

from collections.abc import Mapping
from pathlib import Path

import pytest
from epoch_fixture import KILLED_HOLDERS, KILLED_LOCKS, left_by_a_killed_coordinator, prepared, waited_past_queued
from lease_clock_fixture import ManualClock
from pool_stack_fixture import planned, unresolving
from pool_test import (  # noqa: F401 - `stack` is autouse
    CHARGED,
    CLASSES,
    CONTENT,
    SCORE,
    FakeWorkers,
    _coordinator,
    _late_claim,
    _planned,
    _pool,
    _profile,
    _stack,
    stack,
)
from senselab_core.kernel.identity import InputIdentity
from senselab_core.runtime.call import RunConfig
from senselab_core.runtime.execution.epoch import Epoch
from senselab_core.runtime.execution.events import EventLog, read_events, run_dir_of
from senselab_core.runtime.execution.executor import PlanFragment, follow
from senselab_core.runtime.execution.in_process import node_fragment
from senselab_core.runtime.execution.pool import Pool, split
from senselab_core.runtime.execution.queue import RUN_QUEUES, QueueSet, item_digest, probe_queue
from senselab_core.runtime.execution.run import Output, stored_policy
from senselab_core.runtime.licenses import LicensePolicy
from senselab_core.runtime.planner import Plan, resume
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.provenance import ProvStore
from senselab_core.runtime.store import ResultStore


def _live_items(
    tmp_path: Path, store: ResultStore, workers: FakeWorkers, profile: ExecutionProfile, cuts: int
) -> tuple[Plan, Mapping[InputIdentity, object], list[PlanFragment]]:
    """Put the run's one fragment as items of `cuts` jobs from epoch 1, held back; return the plan, values and items."""
    placed, values = workers.placed, workers.values
    workers.ready = lambda: False
    first = _pool(tmp_path, store, workers, profile)
    items = split(node_fragment(placed, first.admit(placed, values)), cuts)
    first.queues.queue("cpu").put_all(items)
    return placed, values, items


@pytest.mark.contract
def test_a_resume_follows_the_live_item_that_holds_a_job(tmp_path: Path) -> None:
    """Epoch 1 cut four jobs one to an item and was killed; epoch 2 cuts them as one and puts nothing."""
    profile = _profile(**CONTENT)
    placed, values, store, workers = _stack(tmp_path, profile, ["a", "b", "c", "d"])
    _, _, old = _live_items(tmp_path, store, workers, profile, 1)
    second = _pool(tmp_path, store, workers, profile)
    handle = second.submit(node_fragment(placed, second.admit(placed, values)))
    assert second.queues.queue("cpu").depth() == 4
    workers.ready = lambda: True
    try:
        refs = list(follow(second, handle))
    finally:
        workers.stop()
    assert sorted(r.job.element for r in refs) == [(0,), (1,), (2,), (3,)] and all(r.failure is None for r in refs)
    assert {item_digest(f) for f in workers.served} == {item_digest(i) for i in old}
    assert len([e for e in read_events(tmp_path / "run") if e.kind == "claimed"]) == 4


@pytest.mark.contract
def test_a_job_whose_live_item_is_on_another_queue_is_put_again(tmp_path: Path) -> None:
    """A resume that moves a job to another class puts it again; the old item stays (ADR-004)."""
    profile = _profile(**CONTENT)
    placed, values, store, workers = _stack(tmp_path, profile, ["a", "b"])
    workers.ready = lambda: False
    first = _pool(tmp_path, store, workers, profile)
    (whole,) = split(node_fragment(placed, first.admit(placed, values)), 2)
    moved = whole.model_copy(update={"placement": whole.placement.model_copy(update={"resource_class": "cpu:large"})})
    first.queues.queue("cpu:large").put_all([moved])
    second = _pool(tmp_path, store, workers, profile)
    second.submit(node_fragment(placed, second.admit(placed, values)))
    second.close()
    assert second.queues.queue("cpu").depth() == 1 and second.queues.queue("cpu:large").depth() == 1


@pytest.mark.contract
def test_two_fragments_following_one_item_are_given_and_charged_once_each(tmp_path: Path) -> None:
    """One live item of four jobs, followed by two fragments of two: each handle gets its jobs, each is charged once."""
    profile = _profile(**CONTENT)
    placed, values, store, workers = _stack(tmp_path, profile, ["a", "b", "c", "d"])
    _live_items(tmp_path, store, workers, profile, 4)
    workers.act = lambda fragment: CHARGED if fragment.attempt == 1 else None
    second = _pool(tmp_path, store, workers, profile)
    admission = second.admit(placed, values)
    handles = [second.submit(node_fragment(placed, admission, elements=[(i,), (i + 1,)])) for i in (0, 2)]
    workers.ready = lambda: True
    try:
        refs = [list(follow(second, h)) for h in handles]
    finally:
        workers.stop()
    assert [sorted(r.job.element for r in each) for each in refs] == [[(0,), (1,)], [(2,), (3,)]]
    assert all(r.failure is None for each in refs for r in each)
    assert len([e for e in read_events(tmp_path / "run") if e.kind == "failed"]) == 4


@pytest.mark.contract
def test_a_job_held_on_two_queues_is_followed_on_the_queue_it_is_placed_on(tmp_path: Path) -> None:
    """A job live on `cpu` and on `cpu:large` is followed where it is placed now, not on the first queue by name."""
    profile = _profile(**CONTENT)
    placed, values, store, workers = _stack(tmp_path, profile, ["a", "b"])
    workers.ready = lambda: False
    first = _pool(tmp_path, store, workers, profile)
    (whole,) = split(node_fragment(placed, first.admit(placed, values)), 2)
    on_large = whole.model_copy(
        update={"placement": whole.placement.model_copy(update={"resource_class": "cpu:large"})}
    )
    first.queues.queue("cpu").put_all([whole])
    first.queues.queue("cpu:large").put_all([on_large])
    second = _pool(tmp_path, store, workers, profile)
    second.submit(
        node_fragment(placed, second.admit(placed, values)).model_copy(update={"placement": on_large.placement})
    )
    second.close()
    assert second.queues.queue("cpu:large").depth() == 1 and second.queues.queue("cpu").depth() == 1


@pytest.mark.contract
def test_a_probe_queue_item_and_a_damaged_live_item_hold_no_job(tmp_path: Path) -> None:
    """A probe item naming the job and a live item that no longer reads are not followed: the job is put again."""
    profile = _profile(**CONTENT)
    placed, values, store, workers = _stack(tmp_path, profile, ["a"])
    workers.ready = lambda: False
    first = _pool(tmp_path, store, workers, profile)
    (whole,) = split(node_fragment(placed, first.admit(placed, values)), 1)
    first.queues.queue(probe_queue("cpu", "p1")).put_probe(whole, "SINGLE", 1)
    second = _pool(tmp_path, store, workers, profile)
    second.submit(node_fragment(placed, second.admit(placed, values)))
    queue = second.queues.queue("cpu")
    assert queue.depth() == 1
    ((digest, _),) = queue.live_items()
    (queue.path / "items" / f"{digest}.json").write_bytes(b"not an item")
    third = _pool(tmp_path, store, workers, profile)
    third.submit(node_fragment(placed, third.admit(placed, values)))
    third.close()
    assert queue.depth() == 1 and queue.live_items() == []


@pytest.mark.contract
def test_coordinate_runs_under_the_policy_the_run_was_started_under(tmp_path: Path) -> None:
    """`coordinate` reads the stored policy, `prepare` accepts it as its own, and the plan it runs carries it."""
    pytest.importorskip("pydra")
    from senselab_core.runtime.execution.pool import coordinate, run_plan  # noqa: PLC0415 - after the pydra check

    policy = LicensePolicy(frozenset({"mit"}))
    profile = _profile(**CONTENT)
    placed, values = planned(SCORE, profile, ["one"], RunConfig(licenses=policy))
    store, run_dir = ResultStore(tmp_path / "store"), tmp_path / "run"
    workers = FakeWorkers(run_dir, store, placed, values)
    prepared(placed, run_dir, store)
    try:
        outcome = coordinate(run_dir, substrate=workers, profile=profile, store=store)
    finally:
        workers.stop()
    config = run_plan(run_dir, profile).config
    assert config is not None and config.licenses == policy == stored_policy(run_dir)
    assert [r.failure for r in outcome.results["a"]] == [None]


STALL_S = 10.0
"""Real seconds a resume may log nothing but `queued` before the test calls it stuck; a failure bound only."""


@pytest.mark.contract
@pytest.mark.parametrize("lock", KILLED_LOCKS)
@pytest.mark.parametrize("holder", sorted(KILLED_HOLDERS))
def test_a_resume_after_a_killed_coordinator_starts_a_worker_and_commits_every_live_item(
    tmp_path: Path, clock: ManualClock, hang_s: float, holder: str, lock: str
) -> None:
    """Epoch 1 put every item and was killed with no worker alive; epoch 2 starts one and follows each item to the end.

    The killed coordinator's pydra lock (its run-long lease, or a save it was in) names a holder that cannot be proven
    dead; the resume neither waits on it nor puts any job again.
    """
    pytest.importorskip("pydra")
    profile = ExecutionProfile.load(
        "laptop",
        {
            "devices": ["cpu"],
            "resource_classes": CLASSES,
            "pool": {"poll_s": 0.2, "quantum_items": 1},
            "faults": CONTENT,
        },
    )
    placed, values = _planned(profile, ["one", "two", "three", "four"])
    run_dir = run_dir_of(placed.hash)
    killed = FakeWorkers(run_dir, ResultStore(tmp_path / "store"), placed, values)
    killed.ready = lambda: False
    first = _coordinator(placed, profile, killed, tmp_path / "store")
    try:
        started = placed.run(first, store=first.store)
        queue = first.queues.queue("cpu")
        clock.until(lambda: queue.depth() == len(placed.manifest.rows), "epoch 1 put every item", step=0.0)
        started.cancel()
        started.wait(hang_s)
    finally:
        first.close()
    old = {digest for digest, _ in queue.live_items()}
    assert len(old) == 4 and not killed.live()
    planted = left_by_a_killed_coordinator(run_dir, holder, lock)
    workers = FakeWorkers(run_dir, ResultStore(tmp_path / "store"), placed, values)
    second = Pool(
        run_dir,
        queues=QueueSet(run_dir / RUN_QUEUES, profile, lease=profile.lease, epoch=Epoch(run_dir).current),
        store=ResultStore(tmp_path / "store"),
        events=EventLog(run_dir, "a0", 1, epoch=0),
        profile=profile,
        substrate=workers,
        epoch=0,
        clock=clock,
        resolver=unresolving(profile),
    )
    try:
        resumed = resume(placed.hash, execution=profile, store=second.store, executor=second)
        status = waited_past_queued(resumed, STALL_S, hang_s)
    finally:
        second.close()
        workers.stop()
    assert status.done, f"the resume did not end within {hang_s} s: {status.counts}"
    assert workers.names == {"cpu"} and {item_digest(f) for f in workers.served} == old
    assert all(isinstance(o, Output) for _, o in resumed.results()) and planted.exists()


@pytest.mark.contract
def test_a_late_commit_settled_as_a_reuse_is_on_disk_when_it_is_given(tmp_path: Path) -> None:
    """The coordinator's Reuse of an earlier epoch's late commit is written before `poll` gives the result."""
    refs, _ = _late_claim(tmp_path, lambda: 1)
    [reuse] = [a for a in ProvStore.read(tmp_path / "run").activities() if "senselab:Reuse" in a.types()]
    assert [r.served for r in refs] == ["exact"] and "prov:wasAssociatedWith" not in reuse.node()
