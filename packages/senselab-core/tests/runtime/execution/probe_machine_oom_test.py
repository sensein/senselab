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
from senselab_core.runtime.execution.in_process import fragment_jobs, job_units
from senselab_core.runtime.execution.observing import request_units
from senselab_core.runtime.execution.pool_parts import Dumped, Held
from senselab_core.runtime.execution.probe_machine import ProbeMachine
from senselab_core.runtime.execution.queue import probe_of
from senselab_core.runtime.measures import MeasureUnit, Provenance
from senselab_core.runtime.probe import Probe
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


def _part(host: FakeHost, held: Held, lengths: range, batch: int) -> Any:  # noqa: ANN401 - a fragment
    """Return the item of `held`'s jobs whose length is in `lengths`, at request size `batch`."""
    jobs, _ = fragment_jobs(held.part, host.placed, host.values)
    picks = [i for i, j in enumerate(jobs) if job_units(j, "chars") in lengths]
    moved = held.part.placement.model_copy(update={"batch_size": batch})
    return Dumped.of(held.part).part(picks, placement=moved.model_dump(mode="json"))


def _fault(machine: ProbeMachine, host: DeviceHost, held: Held, *, lengths: range, batch: int) -> OomRecord:
    """Size the unit's part, put its jobs of `lengths` as an item of `batch`, fault it, and return the record."""
    machine.place(held)
    [release] = host.released
    assert release.estimate is not None and release.estimate.device is not None
    machine.note(release, ["item"])
    seen = [_obs(float(lengths[-1]), batch=1)]
    machine.oom("item", _part(host, held, lengths, batch), seen, "4242")
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
    record = _fault(machine, host, held, lengths=range(57, 61), batch=4)
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
    record = _fault(machine, host, held, lengths=range(57, 61), batch=4)
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


def test_a_fault_that_reports_no_co_tenancy_writes_no_record(tmp_path: Path, clock: ManualClock) -> None:
    """The co-tenancy a record is filed under comes from the item's requests; with none, nothing is invented."""
    machine, host, held, _, unit = _device(tmp_path, clock)
    machine.place(held)
    machine.note(host.released[0], ["item"])
    machine.oom("item", _part(host, held, range(57, 61), 4), [], None)
    assert machine.roots.regime_records(unit) == ()


def test_a_second_fault_no_longer_than_the_first_adds_no_record(tmp_path: Path, clock: ManualClock) -> None:
    """One record per batch, co-tenancy, grant and basis holds the shortest length that faulted."""
    machine, host, held, _, unit = _device(tmp_path, clock)
    _fault(machine, host, held, lengths=range(51, 55), batch=4)
    machine.note(host.released[0], ["again", "shorter"])
    seen = [_obs(54.0)]
    machine.oom("again", _part(host, held, range(57, 61), 4), seen, None)
    assert len(machine.roots.regime_records(unit)) == 1
    machine.oom("shorter", _part(host, held, range(41, 45), 4), seen, None)
    assert sorted(r.length for r in machine.roots.regime_records(unit) if isinstance(r, OomRecord)) == [44.0, 54.0]


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
    machine.place(Held(held.followed, held.queue, _part(host, held, range(1, 5), 1)))
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
