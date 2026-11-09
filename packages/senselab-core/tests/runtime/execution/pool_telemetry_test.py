"""What the pool does with production requests: drift, regime and supersede records, `oom` records, closing actuals.

Workers are played by hand (`Worker`): a probe item reports as a probe does, a production item reports its requests
with a wall `rho` times the probe's, under a co-tenancy and a hardware class the test names. Nothing reads a wall time.
"""

import itertools
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest
from lease_clock_fixture import ManualClock
from pool_probe_test import (
    ELSEWHERE,
    JOBS,
    SEEN,
    Hand,
    _drive,
    _plan,
    _pool,
    _probe_queues,
    _profile,
    _seen,
    _submit,
    _unit,
    stack,  # noqa: F401 - the registry and env the plans need
)
from senselab_core.kernel.identity import InputIdentity
from senselab_core.runtime.calibration import Cotenancy, HardwareClass, Observation, OomRecord
from senselab_core.runtime.execution.events import placements, read_events
from senselab_core.runtime.execution.executor import FaultRecord, Handle, PlanFragment
from senselab_core.runtime.execution.in_process import fragment_jobs, fragment_keys
from senselab_core.runtime.execution.job import Failed, JobOutcome
from senselab_core.runtime.execution.observing import request_units
from senselab_core.runtime.execution.pool import Pool
from senselab_core.runtime.execution.substrates import Fleet, WorkerHandle
from senselab_core.runtime.execution.telemetry import Actuals
from senselab_core.runtime.measures import MeasureUnit
from senselab_core.runtime.planner import Plan
from senselab_core.runtime.probe import probe_regime
from senselab_core.runtime.profiles import ExecutionProfile

pytestmark = pytest.mark.logic

ALONE = Cotenancy(device=1, host=1)
SHARED = Cotenancy(device=2, host=2)
SLOW = 2.0


class Worker(Hand):
    """A `Hand` that reports production items too.

    A probe request takes `probe_rho` times the base wall and a production request `production_rho` times, each
    jittered by the same fixed pattern so the fits have a spread. A production item reports `cotenancy` and
    `production_hardware`. An item of `queue` whose longest job has `fault_units` units or more fails every job with a
    resource fault on its first attempt; `peak` gives the device bytes a request reports as `(peak, after)`.
    """

    def __init__(  # noqa: PLR0913 - the plan, then what the worker does differently from a probe's hand
        self,
        pool: Pool,
        placed: Plan,
        values: Mapping[InputIdentity, object],
        *,
        probe_rho: float = 1.0,
        production_rho: float = 1.0,
        cotenancy: Cotenancy = ALONE,
        production_hardware: HardwareClass = SEEN,
        fault_units: float | None = None,
        limit: int | None = None,
        peak: Callable[[float], tuple[int, int]] | None = None,
    ) -> None:
        """Serve `pool`'s queues for `placed`."""
        self._pattern = itertools.cycle((-1.0, 0.0, 1.0))
        self.probe_rho, self.production_rho = probe_rho, production_rho
        self.cotenancy, self.production_hardware = cotenancy, production_hardware
        self.fault_units, self.peak = fault_units, peak
        self.peaks: list[int] = []
        super().__init__(pool, placed, values, wall=lambda units: self._base(units) * probe_rho, limit=limit)

    def _base(self, units: float) -> float:
        return (0.05 + 0.01 * units) * (1 + 0.01 * next(self._pattern))

    def _run(
        self, name: str, digest: str, item: PlanFragment, *, probe: bool
    ) -> tuple[list[JobOutcome], list[Observation]]:
        """Return a probe item as a `Hand` does, a production item with its own observations."""
        return super()._run(name, digest, item, probe=True) if probe else self._production(item, digest)

    def _production(self, item: PlanFragment, digest: str) -> tuple[list[JobOutcome], list[Observation]]:
        jobs, _ = fragment_jobs(item, self.placed, self.values)
        longest = max(request_units([j])[1] for j in jobs)
        faulted = self.fault_units is not None and item.attempt == 1 and longest >= self.fault_units
        found: list[Observation] = []
        for index, job in enumerate(jobs):
            kind, units, padded = request_units([job])
            device = self.peak(units) if self.peak is not None else (None, None)
            self.peaks.extend(p for p in device[:1] if p is not None)
            found.append(
                Observation.model_validate(
                    {
                        "probe": None,
                        "item": digest,
                        "request_index": index,
                        "jobs": 1,
                        "batch": 1,
                        "unit_kind": kind,
                        "units": units,
                        "padded_units": padded,
                        "wall_s": self._base(units) * self.production_rho,
                        "item_wall_s": None,
                        "since_load": None if faulted else self.since,
                        "load_s": None,
                        "device_peak": device[0],
                        "device_after": device[1],
                        "host_peak": None,
                        "host_after": None,
                        "cotenancy": self.cotenancy,
                        "outcome": "failed" if faulted else "ok",
                        "hardware": self.production_hardware,
                        "fault": "resource" if faulted else None,
                    }
                )
            )
            self.since += 1
        if not faulted:
            return self._outcomes(item), found
        fault = FaultRecord(
            fault="resource", type="OutOfMemoryError", message="out of memory", gap=False, charged=False
        )
        keys = fragment_keys(item, self.placed, self.values)
        return [Failed(j.ref, k, fault) for j, k in zip(jobs, keys, strict=True)], found


def _first(
    tmp_path: Path, clock: ManualClock, **worker: object
) -> tuple[ExecutionProfile, Pool, Plan, Mapping[InputIdentity, object], MeasureUnit]:
    """Measure the unit in a first fragment (its probe at `probe_rho`) and run its production items as `worker` says."""
    profile = _profile(tmp_path)
    _seen(profile, SEEN)
    placed, values = _plan(profile, range(1, JOBS + 1))
    pool = _pool(tmp_path, profile, clock=clock)
    _, handle = _submit(pool, placed, values)
    _drive(pool, handle, Worker(pool, placed, values, **worker), rounds=200)  # type: ignore[arg-type]
    return profile, pool, placed, values, _unit(pool, placed, values)


def _more(pool: Pool, profile: ExecutionProfile, char: str) -> tuple[str, Plan, Mapping[InputIdentity, object]]:
    """Submit another fragment of the same unit over the same lengths of `char`; return its handle, plan and values."""
    placed, values = _plan(profile, range(1, JOBS + 1), char=char)
    _, handle = _submit(pool, placed, values)
    return handle, placed, values


def _kinds(pool: Pool, unit: MeasureUnit) -> list[str]:
    return [r.kind for r in pool.roots.records(unit)]


def _only(pool: Pool, unit: MeasureUnit, kind: str) -> dict[str, object]:
    [found] = [r for r in pool.roots.records(unit) if r.kind == kind]
    return dict(found.record)


def test_drift_confirms_before_superseding(tmp_path: Path, clock: ManualClock) -> None:
    """A k = 1 drift writes `drift` and supersedes nothing; a confirming probe that disagrees writes `supersede`."""
    profile, pool, placed, values, unit = _first(tmp_path, clock, production_rho=SLOW)
    first = next(iter(pool.probes.values())).probe
    assert "drift" in _kinds(pool, unit) and "supersede" not in _kinds(pool, unit)
    before = pool.measures.get(unit)
    assert before is not None and first in before.source.probe_ids
    handle, placed2, values2 = _more(pool, profile, "y")
    assert "supersede" not in _kinds(pool, unit)
    _drive(pool, handle, Worker(pool, placed2, values2, probe_rho=SLOW, production_rho=SLOW), rounds=200)
    assert _only(pool, unit, "supersede")["outdates"] == (first,)
    after = pool.measures.get(unit)
    assert after is not None and first not in after.source.probe_ids
    assert before.single is not None and after.single is not None and after.single.r > 1.5 * before.single.r


def test_a_confirming_probe_that_agrees_writes_nothing_more(tmp_path: Path, clock: ManualClock) -> None:
    """The slowness was another process: the confirming probe measures the old cost and no `supersede` is written."""
    profile, pool, _, _, unit = _first(tmp_path, clock, production_rho=SLOW)
    first = next(iter(pool.probes.values())).probe
    handle, placed2, values2 = _more(pool, profile, "y")
    _drive(pool, handle, Worker(pool, placed2, values2), rounds=200)
    assert "supersede" not in _kinds(pool, unit)
    estimate = pool.measures.get(unit)
    assert estimate is not None and first in estimate.source.probe_ids


def test_a_confirming_probe_holds_the_units_jobs(tmp_path: Path, clock: ManualClock) -> None:
    """While the confirming probe runs, the unit's next jobs are put on no production queue; they follow its end."""
    profile, pool, _, _, _ = _first(tmp_path, clock, production_rho=SLOW)
    probes = len(_probe_queues(pool))
    handle, placed2, values2 = _more(pool, profile, "y")
    assert len(_probe_queues(pool)) == probes + 1 and pool.queues.queue("cpu").work() == 0
    Worker(pool, placed2, values2, limit=1).serve()
    early, _ = pool.poll(Handle(id=handle))
    assert pool.queues.queue("cpu").work() == 0
    refs = [*early, *_drive(pool, handle, Worker(pool, placed2, values2), rounds=200)]
    assert len(refs) == JOBS and all(r.failure is None for r in refs)


def test_drift_under_cotenancy_records_a_regime_without_a_probe(tmp_path: Path, clock: ManualClock) -> None:
    """Drift under k = 2 is a regime effect: one `regime` record, `size` reads it, and no probe runs again."""
    profile, pool, _, _, unit = _first(tmp_path, clock, production_rho=SLOW, cotenancy=SHARED)
    probes = len(_probe_queues(pool))
    regime = _only(pool, unit, "regime")
    assert (regime["k"], regime["n"]) == (2, profile.calibration.drift_window)
    assert regime["rho"] == pytest.approx(SLOW, rel=0.1)
    estimate = pool.measures.get(unit)
    assert estimate is not None and [r.k for r in estimate.regimes] == [2]
    handle, placed2, values2 = _more(pool, profile, "y")
    assert len(_probe_queues(pool)) == probes and pool.queues.queue("cpu").work() == JOBS
    _drive(pool, handle, Worker(pool, placed2, values2, production_rho=SLOW, cotenancy=SHARED), rounds=200)
    kinds = _kinds(pool, unit)
    assert kinds.count("regime") == 1 and "supersede" not in kinds and len(_probe_queues(pool)) == probes


def test_requests_of_another_hardware_class_feed_no_drift(tmp_path: Path, clock: ManualClock) -> None:
    """Slow requests reported under another hardware class are no evidence about this class's estimate.

    The same requests reported under this class do drift it.
    """
    profile, pool, _, _, unit = _first(tmp_path, clock, production_rho=SLOW, production_hardware=ELSEWHERE)
    probes = len(_probe_queues(pool))
    handle, placed2, values2 = _more(pool, profile, "y")
    _drive(
        pool,
        handle,
        Worker(pool, placed2, values2, production_rho=SLOW, production_hardware=ELSEWHERE),
        rounds=200,
    )
    assert _kinds(pool, unit) == [] and len(_probe_queues(pool)) == probes
    handle, placed3, values3 = _more(pool, profile, "v")
    _drive(pool, handle, Worker(pool, placed3, values3, production_rho=SLOW), rounds=200)
    assert "drift" in _kinds(pool, unit)


def test_drift_in_one_run_does_not_supersede_another_runs_regime(tmp_path: Path, clock: ManualClock) -> None:
    """A drift record alone changes no estimate; a run's supersede outdates probes, never another run's regime."""
    profile, first, _, _, unit = _first(tmp_path, clock, production_rho=SLOW, cotenancy=SHARED)
    probe = next(iter(first.probes.values())).probe
    other = _pool(tmp_path, profile, run="run2", store="store2", clock=clock)
    placed, values = _plan(profile, range(1, JOBS + 1), char="z")
    _, handle = _submit(other, placed, values)
    assert not _probe_queues(other) and other.queues.queue("cpu").work() == JOBS
    _drive(other, handle, Worker(other, placed, values, production_rho=SLOW), rounds=200)
    assert "drift" in _kinds(other, unit) and "supersede" not in _kinds(other, unit)
    handle, placed2, values2 = _more(other, profile, "w")
    _drive(other, handle, Worker(other, placed2, values2, probe_rho=SLOW, production_rho=SLOW), rounds=200)
    assert _only(other, unit, "supersede")["outdates"] == (probe,)
    estimate = other.measures.get(unit)
    assert estimate is not None and [(r.k, r.n) for r in estimate.regimes] == [(2, profile.calibration.drift_window)]


def test_an_oom_at_batch_one_is_replaced_not_raised(tmp_path: Path, clock: ManualClock) -> None:
    """A resource fault of a batch of one is recorded, raises nothing, and its jobs run on the next larger class."""
    override = {
        "devices": ["cpu"],
        "resource_classes": {
            "cpu": {"device": "cpu", "mem_bytes": 2**33, "max_workers": 2},
            "cpu:big": {"device": "cpu", "mem_bytes": 2**34, "max_workers": 2},
        },
        "pool": {"poll_s": 0.2, "first_poll_s": 0.01, "quantum_items": 10},
        "calibration": {"measures_roots": [str(tmp_path / "measures")], "warmup_requests": 1},
        "faults": {"resource_replacements": 1},
    }
    profile = ExecutionProfile.load("laptop", override)
    _seen(profile, SEEN)
    placed, values = _plan(profile, range(1, JOBS + 1))
    pool = _pool(tmp_path, profile, clock=clock)
    _, handle = _submit(pool, placed, values)
    refs = _drive(pool, handle, Worker(pool, placed, values, fault_units=40.0), rounds=200)
    unit = _unit(pool, placed, values)
    assert len(refs) == JOBS and [r.failure for r in refs if r.failure is not None] == []
    ooms = [OomRecord.model_validate(r.record) for r in pool.roots.records(unit) if r.kind == "oom"]
    probed = next(iter(pool.probes.values())).probe
    assert ooms and len({o.length for o in ooms}) == len(ooms) and all(o.length > 0 for o in ooms)
    assert {(o.resource, o.batch, o.cotenancy, o.grant_bytes, o.basis) for o in ooms} == {
        ("host", 1, 1, 2**33, (probed,))
    }
    assert not any(probe_regime(o) for o in ooms)
    estimate = pool.measures.get(unit)
    assert estimate is not None and set(estimate.oom) == set(ooms) and estimate.device is None and estimate.host is None


def test_the_placement_event_closes_with_actuals_beside_the_request(tmp_path: Path, clock: ManualClock) -> None:
    """Closing the pool appends each placed class's record again: wall, device peak and the request it was placed on."""
    profile = _profile(tmp_path)
    _seen(profile, SEEN)
    placed, values = _plan(profile, range(1, JOBS + 1))
    pool = _pool(tmp_path, profile, clock=clock)
    pool.granted("cpu")
    _, handle = _submit(pool, placed, values)
    worker = Worker(pool, placed, values, peak=lambda units: (10**9 + 10**6 * int(units), 10**9))
    for _ in range(200):
        worker.serve()
        clock.advance(1.0)
        if pool.poll(Handle(id=handle))[1]:
            break
    pool.close()
    opened, closed = placements(read_events(pool.run_dir))
    assert opened.placement is not None and opened.placement.closing is None
    assert closed.placement is not None and closed.placement.closing is not None
    assert closed.placement.request is not None and closed.placement.request.resource_class == "cpu"
    assert closed.placement.closing.wall_s is not None and closed.placement.closing.wall_s > 0
    assert closed.placement.closing.gpu_hours is None
    assert closed.placement.closing.device_peak_bytes == max(worker.peaks)


def test_actuals_read_gpu_hours_from_worker_lifetimes_and_the_largest_device_peak(clock: ManualClock) -> None:
    """A worker lives from the first listing that shows it started to the first that omits it; pending ones do not."""
    workers: list[WorkerHandle] = [WorkerHandle("gpu", "w1"), WorkerHandle("gpu", "w2", pending=True)]

    class Listing:
        def live(self) -> list[WorkerHandle]:
            return list(workers)

        def start(self, resource_class: str, count: int) -> list[WorkerHandle]:  # noqa: ARG002
            return []

        def stop(self, handles: object = ()) -> None:
            pass

    fleet = Fleet(Listing(), workers_dir=None, budget=3, first_s=1.0, max_s=8.0, clock=clock)  # type: ignore[arg-type]
    fleet.live()
    clock.advance(3600.0)
    fleet.live()
    assert fleet.worker_seconds("gpu") == pytest.approx(3600.0)
    workers.pop(0)
    fleet.live()
    clock.advance(7200.0)
    assert fleet.worker_seconds("gpu") == pytest.approx(3600.0)
    acts = Actuals(clock)
    acts.began("gpu")
    clock.advance(10.0)
    closing = acts.closing("gpu", fleet.worker_seconds("gpu"))
    assert closing.gpu_hours == pytest.approx(1.0) and closing.wall_s is None and closing.device_peak_bytes is None
