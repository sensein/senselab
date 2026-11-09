"""`oom` records and the device-peak raise, through the machine, the roots, `default_measures` and `size`.

The unit is measured from a seed of observations (one probe, `SEED`) on a device grant; a record is written by the
machine and read back by what sizes the next jobs, so the whole path a fault takes is the one under test.
"""

from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from lease_clock_fixture import ManualClock
from pool_probe_test import (
    JOBS,
    SEEN,
    stack,  # noqa: F401 - the registry and env the plans need
)
from probe_machine_test import FakeHost, FakeQueue, _machine  # noqa: F401 - the fake host and its machine
from senselab_core.runtime.calibration import Cotenancy, LengthProfile, Observation, OomRecord, ProbeRef
from senselab_core.runtime.execution.in_process import fragment_jobs
from senselab_core.runtime.execution.observing import request_units
from senselab_core.runtime.execution.pool_parts import Held
from senselab_core.runtime.execution.probe_machine import ProbeMachine
from senselab_core.runtime.execution.queue import probe_of
from senselab_core.runtime.execution_settings import ResourceClass
from senselab_core.runtime.measures import MeasureUnit, Provenance
from senselab_core.runtime.probe import Probe, probe_regime
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.sizing import Grant, size
from senselab_core.runtime.units import Unresolved, decoder_of

pytestmark = pytest.mark.logic

SEED = "seed"
RESIDENT = 10**9
M1 = 10**7
GRANT = Grant(max_workers=2, memory_bytes=2**33, quantum_items=10, quantum_s=60.0, on_device=True)
LENGTHS = (1.0, float(JOBS))
FAST = settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])


class DeviceHost(FakeHost):
    """A `FakeHost` whose grant is a device's, and whose probe queues remember each item's phase and stratum."""

    def grant(self, queue: str) -> Grant:  # noqa: ARG002
        """Return a device grant of one card's memory."""
        return GRANT

    def queue(self, name: str) -> FakeQueue:
        """Return the queue `name`, one that keeps the marks of its items."""
        return self.queues.setdefault(name, Marked())


class Marked(FakeQueue):
    """A `FakeQueue` that keeps each item's phase and stratum."""

    def __init__(self) -> None:
        """Start empty."""
        super().__init__()
        self.marks: dict[str, tuple[str, int]] = {}

    def put_probe(self, fragment: Any, phase: str, stratum: int) -> None:  # noqa: ANN401 - a fragment
        """Keep the item and its mark."""
        super().put_probe(fragment, phase, stratum)
        from senselab_core.runtime.execution.queue import item_digest  # noqa: PLC0415 - one use

        self.marks[item_digest(fragment)] = (phase, stratum)


def _obs(  # noqa: PLR0913 - one request: where it ran, its shape, and what it cost
    x: float, *, batch: int = 1, probe: str = SEED, phase: str = "SINGLE", stratum: int = 0, wall: float | None = None
) -> Observation:
    padded = batch * x
    return Observation.model_validate(
        {
            "probe": ProbeRef(id=probe, phase=phase, stratum=stratum),
            "item": f"{probe}-{phase}-{x}-{batch}",
            "request_index": 0,
            "jobs": batch,
            "batch": batch,
            "unit_kind": "chars",
            "units": padded,
            "padded_units": padded,
            "wall_s": wall if wall is not None else 0.05 + (0.01 if batch == 1 else 0.005) * padded,
            "item_wall_s": None,
            "since_load": 2,
            "load_s": None,
            "device_peak": RESIDENT + int(M1 * padded),
            "device_after": RESIDENT,
            "host_peak": None,
            "host_after": None,
            "cotenancy": Cotenancy(device=1, host=1),
            "outcome": "ok",
            "hardware": SEEN,
        }
    )


def _seed(batch: int = 4) -> list[Observation]:
    """`min_requests` warm requests at one job and at `batch` jobs, in each of two strata of length."""
    return [
        _obs(x, batch=b, phase="SINGLE" if b == 1 else "BATCHED", stratum=i)
        for i, x in enumerate(LENGTHS)
        for b in (1, batch)
        for _ in range(3)
    ]


def _provenance(profile: ExecutionProfile) -> Provenance:
    return Provenance(job=None, run="seed", senselab=None, profile=profile.canonical_digest(), hardware=SEEN)


def _unit(machine: ProbeMachine, host: FakeHost, held: Held) -> MeasureUnit:
    jobs, placement = fragment_jobs(held.part, host.placed, host.values)
    decoder = decoder_of(jobs)
    assert not isinstance(decoder, Unresolved)
    unit = machine.resolver.unit(jobs[0].call, placement, decoder, None)
    assert isinstance(unit, MeasureUnit)
    return unit


def _device(tmp_path: Path, clock: ManualClock) -> tuple[ProbeMachine, DeviceHost, Held, ExecutionProfile, MeasureUnit]:
    """A machine over a device host whose roots hold the seed: placing its part gives a measured unit."""
    plain, fake, held, profile = _machine(tmp_path, True, clock)
    host = DeviceHost(fake.placed, fake.values)
    machine = ProbeMachine(profile, plain.roots, plain.measures, plain.resolver, clock, host)
    unit = _unit(machine, host, held)
    assert machine.roots.put_observations(unit, SEED, _seed(), _provenance(profile)) is not None
    return machine, host, held, profile, unit


def _faulted(x: float, batch: int, *, fault: str = "resource", device: int | None = 1) -> Observation:
    """A request of `batch` jobs of `x` units that raised `fault` and was split, under `device` co-tenancy."""
    return _obs(x, batch=batch).model_copy(
        update={
            "outcome": "bisected",
            "fault": fault,
            "since_load": None,
            "probe": None,
            "cotenancy": Cotenancy(device=device, host=1),
        }
    )


def _fault(  # noqa: PLR0913 - the machine and unit, then the fault's shape
    machine: ProbeMachine, host: DeviceHost, held: Held, *, length: float, batch: int, device: int | None = 1
) -> OomRecord:
    """Size the unit's part, fault a request of `batch` jobs of `length` units, and return the record written."""
    machine.place(held)
    [release] = host.released
    assert release.estimate is not None and release.estimate.device is not None
    machine.note(release, ["item"])
    machine.oom("item", [_faulted(length, batch, device=device)], "4242")
    [record] = [r for r in machine.roots.regime_records(release.unit) if isinstance(r, OomRecord)]  # type: ignore[arg-type]
    return record


def _sized(machine: ProbeMachine, unit: MeasureUnit, host: DeviceHost, *lengths: float) -> Any:  # noqa: ANN401 - Sizing
    estimate = machine.measures.get(unit)
    assert estimate is not None
    profile = machine.profile
    return size(
        estimate,
        host.grant("cpu"),
        profile.sizing,
        LengthProfile(units=tuple(lengths)),
        calibration=profile.calibration,
        bucket_edges=(),
    )


def test_an_oom_caps_only_the_lengths_at_or_above_it(tmp_path: Path, clock: ManualClock) -> None:
    """A fault at batch 4 and length 60 leaves lengths up to 45 their batch and gives 60 a batch below 4."""
    machine, host, held, _, unit = _device(tmp_path, clock)
    before = _sized(machine, unit, host, 1.0, 10.0, 20.0, 40.0, 60.0)
    assert [before.batch_at(n) for n in (1.0, 10.0, 20.0, 40.0, 60.0)] == [4, 4, 4, 4, 4]
    record = _fault(machine, host, held, length=60.0, batch=4)
    assert (record.resource, record.batch, record.length, record.cotenancy, record.worker) == (
        "device",
        4,
        60.0,
        1,
        "4242",
    )
    assert record.grant_bytes == GRANT.memory_bytes and record.basis == (SEED,)
    after = _sized(machine, unit, host, 1.0, 10.0, 20.0, 40.0, 60.0)
    assert [after.batch_at(n) for n in (1.0, 10.0, 20.0, 40.0)] == [4, 4, 4, 4]
    assert after.batch_at(60.0) < 4


def test_an_oom_reopens_its_stratum_and_the_next_probe_lifts_it(tmp_path: Path, clock: ManualClock) -> None:
    """The next run plans a batched probe over the stratum; its requests outside the basis lift the record."""
    machine, host, held, profile, unit = _device(tmp_path, clock)
    record = _fault(machine, host, held, length=60.0, batch=4)
    raised = machine.measures.get(unit)
    assert raised is not None and raised.oom == (record,) and raised.device is not None and raised.device.m1 > M1
    plain, fake, held2, _ = _machine(tmp_path, True, clock)
    again = DeviceHost(fake.placed, fake.values)
    second = ProbeMachine(profile, machine.roots, plain.measures, plain.resolver, clock, again)
    second.place(held2)
    assert not again.released and [n for n in again.queues if probe_of(n)]
    for _ in range(8):
        _answer(again)
        second.advance()
        if again.released:
            break
    [release] = again.released
    lifted = machine.measures.get(unit)
    assert release.estimate is not None and release.estimate.oom == () and lifted is not None and lifted.oom == ()
    assert lifted.device is not None and lifted.device.m1 == pytest.approx(M1, rel=0.05)
    assert any(
        o.probe is not None and o.probe.phase == "BATCHED" and o.batch >= 2 for o in machine.roots.observations(unit)
    )


def _answer(host: DeviceHost) -> None:
    """Give every probe item with no outcome the observations a worker reports: its requests at the item's batch."""
    for name, queue in host.queues.items():
        found = probe_of(name)
        if found is None or not isinstance(queue, Marked):
            continue
        for digest, item in queue.items.items():
            if digest in queue.outcomes:
                continue
            jobs, _ = fragment_jobs(item, host.placed, host.values)
            size_ = item.placement.batch_size
            phase, stratum = queue.marks[digest]
            reported = []
            for index, start in enumerate(range(0, len(jobs), size_)):
                _, units, padded = request_units(jobs[start : start + size_])
                batch = len(jobs[start : start + size_])
                reported.append(
                    _obs(padded / batch, batch=batch, probe=found[1], phase=phase, stratum=stratum).model_copy(
                        update={"item": digest, "request_index": index, "since_load": 3 + index}
                    )
                )
            queue.outcomes[digest] = {"outcome": "done", "observations": [o.model_dump(mode="json") for o in reported]}


def test_a_fault_that_reports_no_co_tenancy_is_recorded_at_one(tmp_path: Path, clock: ManualClock) -> None:
    """An unread device count is filed as co-tenancy 1, the regime a probe running alone can lift."""
    machine, host, held, _, _ = _device(tmp_path, clock)
    assert _fault(machine, host, held, length=60.0, batch=4, device=None).cotenancy == 1


def test_a_second_fault_no_longer_than_the_first_adds_no_record(tmp_path: Path, clock: ManualClock) -> None:
    """One record per batch, co-tenancy, grant and basis holds the shortest length that faulted."""
    machine, host, held, _, unit = _device(tmp_path, clock)
    _fault(machine, host, held, length=54.0, batch=4)
    machine.note(host.released[0], ["again", "shorter"])
    machine.oom("again", [_faulted(58.0, 4)], None)
    assert len(machine.roots.regime_records(unit)) == 1
    machine.oom("shorter", [_faulted(44.0, 4)], None)
    assert sorted(r.length for r in machine.roots.regime_records(unit) if isinstance(r, OomRecord)) == [44.0, 54.0]


def test_a_batch_that_faults_and_bisects_cleanly_writes_a_record_at_its_batch(
    tmp_path: Path, clock: ManualClock
) -> None:
    """The faulting request is the measurement: no job failed, yet the next `size` stays below its batch."""
    machine, host, held, _, unit = _device(tmp_path, clock)
    machine.place(held)
    machine.note(host.released[0], ["item"])
    halves = [_obs(60.0, batch=2).model_copy(update={"probe": None})] * 2
    machine.oom("item", [_faulted(60.0, 4), *halves], None)
    [record] = [r for r in machine.roots.regime_records(unit) if isinstance(r, OomRecord)]
    assert (record.batch, record.length, record.cotenancy) == (4, 60.0, 1)
    assert _sized(machine, unit, host, 60.0).batch_at(60.0) < 4


def test_a_job_failing_alone_is_a_batch_one_record_that_raises_nothing(tmp_path: Path, clock: ManualClock) -> None:
    """A fault of one job is filed at batch 1: it is re-placed, `m1` is not raised and no probe re-opens for it."""
    machine, host, held, _, unit = _device(tmp_path, clock)
    before = machine.measures.get(unit)
    record = _fault(machine, host, held, length=60.0, batch=1)
    after = machine.measures.get(unit)
    assert before is not None and after is not None and record.batch == 1 and not probe_regime(record)
    assert after.device == before.device and after.oom == (record,)


def test_a_fault_that_is_not_a_resource_fault_writes_nothing(tmp_path: Path, clock: ManualClock) -> None:
    """A lost worker splits the request but is not an out-of-memory: no record."""
    machine, host, held, _, unit = _device(tmp_path, clock)
    machine.place(held)
    machine.note(host.released[0], ["item"])
    machine.oom("item", [_faulted(60.0, 4, fault="infrastructure")], None)
    assert machine.roots.regime_records(unit) == ()


def test_an_oom_under_shared_devices_is_read_when_the_card_shares_it_again(tmp_path: Path, clock: ManualClock) -> None:
    """A fault with two parties on the card caps the batch when two workers share it, though a probe never lifts it."""
    machine, host, held, _, unit = _device(tmp_path, clock)
    record = _fault(machine, host, held, length=60.0, batch=4, device=2)
    assert record.cotenancy == 2 and not probe_regime(record)
    shared = Grant(
        max_workers=2, memory_bytes=GRANT.memory_bytes, quantum_items=10, quantum_s=60.0, on_device=True, devices=1
    )
    estimate = machine.measures.get(unit)
    assert estimate is not None
    profile = machine.profile
    for grant, below in ((GRANT, False), (shared, True)):
        sized = size(estimate, grant, profile.sizing, LengthProfile(units=(60.0,)), calibration=profile.calibration)
        assert (sized.batch_at(60.0) < 4) is below and sized.workers_per_device == (2 if below else 1)


class HostGrant(DeviceHost):
    """A `DeviceHost` whose grant is `host_grant`, a host's memory."""

    host_grant = GRANT

    def grant(self, queue: str) -> Grant:  # noqa: ARG002
        """Return the host grant."""
        return self.host_grant


@pytest.mark.parametrize(("shared_host", "filed"), [(False, 1), (True, 3)])
def test_a_host_fault_is_filed_at_the_co_tenancy_of_its_grant(
    tmp_path: Path,
    clock: ManualClock,
    shared_host: bool,
    filed: int,  # noqa: FBT001 - parametrized
) -> None:
    """Beside three live peers, a host fault is filed at 1 under a per-worker grant and at 3 under shared host memory.

    The first is one a probe lifts; the second is read by `size` when the class's workers share the host again.
    """
    plain, fake, held, profile = _machine(tmp_path, True, clock)
    host = HostGrant(fake.placed, fake.values)
    granted = ResourceClass(device="cpu", max_workers=4, mem_bytes=None if shared_host else 2**35)
    host.host_grant = Grant.of(granted, profile.pool, 2**35)
    machine = ProbeMachine(profile, plain.roots, plain.measures, plain.resolver, clock, host)
    unit = _unit(machine, host, held)
    seed = [
        o.model_copy(
            update={"device_peak": None, "device_after": None, "host_peak": o.device_peak, "host_after": RESIDENT}
        )
        for o in _seed()
    ]
    assert machine.roots.put_observations(unit, SEED, seed, _provenance(profile)) is not None
    assert _sized(machine, unit, host, 60.0).batch_at(60.0) == 4
    machine.place(held)
    machine.note(host.released[0], ["item"])
    peers = _faulted(60.0, 4, device=None).model_copy(update={"cotenancy": Cotenancy(device=None, host=3)})
    machine.oom("item", [peers], None)
    [record] = [r for r in machine.roots.regime_records(unit) if isinstance(r, OomRecord)]
    assert (record.resource, record.batch, record.cotenancy) == ("host", 4, filed)
    assert probe_regime(record) is not shared_host
    assert _sized(machine, unit, host, 60.0).batch_at(60.0) < 4


def test_a_device_peak_above_its_band_raises_m1_for_the_next_jobs(tmp_path: Path, clock: ManualClock) -> None:
    """A warm production request whose peak is beyond its band raises the unit's `m1` at once, noted as `drift`."""
    machine, host, held, profile, unit = _device(tmp_path, clock)
    machine.place(held)
    [first] = host.released
    assert first.estimate is not None and first.estimate.device is not None
    machine.note(first, ["item"])
    padded = 30.0
    request = _obs(padded, wall=0.05 + 0.01 * padded).model_copy(
        update={"probe": None, "device_peak": RESIDENT + int(5 * M1 * padded)}
    )
    machine.settled("item", [request])
    machine.place(Held(held.followed, held.queue, held.part))
    raised = host.released[-1].estimate
    assert raised is not None and raised.device is not None and raised.device.m1 > first.estimate.device.m1
    assert raised.device.m1 == pytest.approx(5 * M1, rel=0.01)
    notes = [r for r in machine.roots.records(unit) if r.kind == "drift"]
    assert [dict(r.record)["metric"] for r in notes] == ["device_peak"]


def _estimate_after(batch: int, length: float, grant: int) -> Any:  # noqa: ANN401 - an Estimate
    record = OomRecord(resource="device", batch=batch, length=length, cotenancy=1, grant_bytes=grant, basis=(SEED,))
    profile = ExecutionProfile.load("laptop")
    return Probe.estimate(
        _seed(batch),
        [record],
        profile.calibration,
        hardware=SEEN,
        memory_headroom=profile.sizing.memory_headroom,
    )


@given(
    batch=st.integers(2, 12),
    length=st.floats(1.0, float(JOBS)),
    above=st.floats(1.0, 4.0),
)
@FAST
def test_after_an_oom_size_never_returns_its_batch_at_or_above_its_length(
    batch: int, length: float, above: float
) -> None:
    """After an `oom` record at (B, L), `size` gives a batch below B at every length L′ at or above L."""
    profile = ExecutionProfile.load("laptop")
    estimate = _estimate_after(batch, length, GRANT.memory_bytes or 0)
    longer = min(length * above, float(JOBS))
    sized = size(
        estimate,
        GRANT,
        profile.sizing,
        LengthProfile(units=(length, longer)),
        calibration=profile.calibration,
        bucket_edges=(),
    )
    assert sized.batch_at(length) < batch and sized.batch_at(longer) < batch
