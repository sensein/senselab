"""A resumed coordinator follows the live item that already holds a job instead of putting the job again."""

from collections.abc import Mapping
from pathlib import Path

import pytest
from pool_test import CHARGED, CONTENT, FakeWorkers, _pool, _profile, _stack, stack  # noqa: F401 - `stack` is autouse
from senselab_core.kernel.identity import InputIdentity
from senselab_core.runtime.execution.events import read_events
from senselab_core.runtime.execution.executor import PlanFragment, follow
from senselab_core.runtime.execution.in_process import node_fragment
from senselab_core.runtime.execution.pool import split
from senselab_core.runtime.execution.queue import item_digest
from senselab_core.runtime.planner import Plan
from senselab_core.runtime.profiles import ExecutionProfile
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
