"""A load that runs out of memory at run time is kept as a fact about its unit; the next plan avoids the class."""

from collections.abc import Mapping
from pathlib import Path

import pytest
from lease_clock_fixture import ManualClock
from pool_probe_test import (
    JOBS,
    SEEN,
    _drive,
    _plan,
    _pool,
    _probe_queues,
    _submit,
    _unit,
    stack,  # noqa: F401 - the registry and env the plans need
)
from pool_telemetry_test import Worker
from senselab_core.kernel.identity import InputIdentity
from senselab_core.runtime.execution.executor import FaultRecord, PlanFragment, QueueHeld
from senselab_core.runtime.execution.in_process import fragment_jobs, fragment_keys
from senselab_core.runtime.execution.job import Failed, JobOutcome
from senselab_core.runtime.execution.pool import Pool
from senselab_core.runtime.fit import Unplaceable
from senselab_core.runtime.planner import Plan
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.units import measure_roots

pytestmark = pytest.mark.logic

SMALL, LARGE = 2**33, 2**34
LOAD_OOM = FaultRecord(
    fault="resource", type="WorkerLoadFailed", message="CUDA out of memory", gap=False, charged=False, load=True
)


class LoadsFailOn(Worker):
    """A worker whose production items fail at load on the class `fails`, and run as a `Worker` runs them elsewhere."""

    fails: tuple[str, ...] = ("cpu",)

    def _production(self, item: PlanFragment, digest: str) -> tuple[list[JobOutcome], list[object]]:  # type: ignore[override]
        if item.placement.resource_class not in self.fails:
            return super()._production(item, digest)  # type: ignore[return-value]
        jobs, _ = fragment_jobs(item, self.placed, self.values)
        keys = fragment_keys(item, self.placed, self.values)
        return [Failed(j.ref, k, LOAD_OOM) for j, k in zip(jobs, keys, strict=True)], []


def _profile(tmp_path: Path) -> ExecutionProfile:
    classes = {
        "cpu": {"device": "cpu", "mem_bytes": SMALL, "max_workers": 2},
        "cpu:big": {"device": "cpu", "mem_bytes": LARGE, "max_workers": 2},
    }
    override = {
        "devices": ["cpu"],
        "resource_classes": classes,
        "pool": {"poll_s": 0.2, "first_poll_s": 0.01, "quantum_items": 10},
        "calibration": {"measures_roots": [str(tmp_path / "measures")], "warmup_requests": 1},
        "faults": {"resource_replacements": 1},
    }
    return ExecutionProfile.load("laptop", override)


def _more(pool: Pool, profile: ExecutionProfile, char: str) -> tuple[str, Plan, Mapping[InputIdentity, object]]:
    placed, values = _plan(profile, range(1, JOBS + 1), char=char)
    _, handle = _submit(pool, placed, values)
    return handle, placed, values


def test_a_runtime_load_oom_records_unfit_and_the_replan_avoids_the_class(tmp_path: Path, clock: ManualClock) -> None:
    """A measured unit's load fails on `cpu`; the fragment runs on `cpu:big`; the next one starts there."""
    profile = _profile(tmp_path)
    for resource in profile.resource_classes.values():
        assert measure_roots(profile).note_seen(resource, SEEN) is not None
    placed, values = _plan(profile, range(1, JOBS + 1))
    pool = _pool(tmp_path, profile, clock=clock)
    _, handle = _submit(pool, placed, values)
    _drive(pool, handle, Worker(pool, placed, values), rounds=200)
    unit = _unit(pool, placed, values)
    assert not pool.roots.unfit_records(unit)

    second, placed2, values2 = _more(pool, profile, "y")
    refs = _drive(pool, second, LoadsFailOn(pool, placed2, values2), rounds=200)
    assert len(refs) == JOBS and all(r.failure is None for r in refs)
    [record] = pool.roots.unfit_records(unit)
    assert (record.resource, record.grant_bytes) == ("host", SMALL)

    probes = len(_probe_queues(pool))
    third, placed3, values3 = _more(pool, profile, "z")
    assert len(_probe_queues(pool)) == probes and pool.queues.queue("cpu:big").work() == JOBS
    assert pool.queues.queue("cpu").work() == 0
    refs = _drive(pool, third, LoadsFailOn(pool, placed3, values3), rounds=200)
    assert len(refs) == JOBS and len(pool.roots.unfit_records(unit)) == 1


def test_a_load_that_fails_on_every_class_records_each_and_the_replan_is_unplaceable(
    tmp_path: Path, clock: ManualClock
) -> None:
    """The re-placed retry keeps its origin: `cpu:big` records its own fact, and no class is left on the next plan."""
    profile = _profile(tmp_path)
    for resource in profile.resource_classes.values():
        assert measure_roots(profile).note_seen(resource, SEEN) is not None
    placed, values = _plan(profile, range(1, JOBS + 1))
    pool = _pool(tmp_path, profile, clock=clock)
    _, handle = _submit(pool, placed, values)
    _drive(pool, handle, Worker(pool, placed, values), rounds=200)
    unit = _unit(pool, placed, values)

    second, placed2, values2 = _more(pool, profile, "y")
    hand = LoadsFailOn(pool, placed2, values2)
    hand.fails = ("cpu", "cpu:big")
    with pytest.raises(QueueHeld):
        _drive(pool, second, hand, rounds=200)
    assert sorted((r.resource, r.grant_bytes) for r in pool.roots.unfit_records(unit)) == [
        ("host", SMALL),
        ("host", LARGE),
    ]
    again = _pool(tmp_path, profile, run="run2", store="store2", clock=clock)
    with pytest.raises(Unplaceable) as raised:
        _more(again, profile, "z")
    assert [t.verdict for t in raised.value.tried] == ["unfit", "unfit"]
