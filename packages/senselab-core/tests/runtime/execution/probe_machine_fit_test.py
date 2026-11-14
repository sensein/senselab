"""The probe machine's fit gate: every release is judged against its class, and a load that ran out of memory is kept.

The unit is measured from a seed of observations (resident `RESIDENT`) on device grants a test sets per class; the
machine releases a held part, or moves it to a larger class, or stops the plan as `Unplaceable`.
"""

import ast
import inspect
from pathlib import Path

import pytest
from lease_clock_fixture import ManualClock
from pool_probe_test import (
    HOST,
    JOBS,
    SEEN,
    _plan,
    stack,  # noqa: F401 - the registry and env the plans need
)
from probe_machine_oom_test import RESIDENT, DeviceHost, _provenance, _seed
from probe_machine_test import FakeHost, _machine  # noqa: F401 - the fake host and its machine
from senselab_core.runtime.calibration import UnfitRecord
from senselab_core.runtime.execution import probe_machine as machine_module
from senselab_core.runtime.execution.executor import Admission
from senselab_core.runtime.execution.in_process import fragment_jobs, node_fragment
from senselab_core.runtime.execution.pool_parts import Followed, Held, Release
from senselab_core.runtime.execution.probe_machine import ProbeMachine
from senselab_core.runtime.execution.queue import probe_of
from senselab_core.runtime.execution.substrates import class_of
from senselab_core.runtime.fit import Unplaceable, code_of
from senselab_core.runtime.measures import MeasureUnit
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.sizing import Grant
from senselab_core.runtime.units import UnitResolver, Unresolved, decoder_of, default_measures, measure_roots

pytestmark = pytest.mark.logic

SMALL = RESIDENT  # a grant whose headroom share (0.9) is below the seeded resident memory
LARGE = 4 * RESIDENT


def _code(unit: MeasureUnit) -> dict[str, str]:
    return {"source": code_of(unit).source, "overlay": code_of(unit).overlay}


class GrantedHost(DeviceHost):
    """A device host whose grant is the memory a test gives each class."""

    grants: dict[str, int]

    def grant(self, queue: str) -> Grant:
        """Return a device grant of the class's memory."""
        return Grant(
            max_workers=2, memory_bytes=self.grants[class_of(queue)], quantum_items=10, quantum_s=60.0, on_device=True
        )


def _profile(tmp_path: Path, *, large: bool) -> ExecutionProfile:
    classes = {"cpu": {"device": "cpu", "mem_bytes": 2**33, "max_workers": 2}}
    if large:
        classes["cpu:large"] = {"device": "cpu", "mem_bytes": 2**34, "max_workers": 2}
    override = {
        "devices": ["cpu"],
        "resource_classes": classes,
        "pool": {"poll_s": 0.2, "first_poll_s": 0.01, "quantum_items": 10},
        "calibration": {"measures_roots": [str(tmp_path / "measures")], "warmup_requests": 1},
    }
    return ExecutionProfile.load("laptop", override)


def _fitting(
    tmp_path: Path, clock: ManualClock, *, small: int = SMALL, large: bool = True, seed: bool = True
) -> tuple[ProbeMachine, GrantedHost, Held, MeasureUnit]:
    """A machine over a host granting `small` to `cpu` and `LARGE` to `cpu:large`, its unit seeded when `seed`."""
    profile = _profile(tmp_path, large=large)
    roots = measure_roots(profile)
    for resource in profile.resource_classes.values():
        assert roots.note_seen(resource, SEEN) is not None
    placed, values = _plan(profile, range(1, JOBS + 1))
    host = GrantedHost(placed, values)
    host.grants = {"cpu": small, "cpu:large": LARGE}
    machine = ProbeMachine(
        profile, roots, default_measures(profile, roots), UnitResolver(profile, roots, hardware=HOST), clock, host
    )
    fragment = node_fragment(placed, Admission(id=f"run-{placed.hash}", plan_hash=placed.hash))
    held = Held(Followed([]), "cpu", fragment)
    jobs, placement = fragment_jobs(held.part, placed, values)
    decoder = decoder_of(jobs)
    assert not isinstance(decoder, Unresolved)
    unit = machine.resolver.unit(jobs[0].call, placement, decoder, None)
    assert isinstance(unit, MeasureUnit)
    if seed:
        assert roots.put_observations(unit, "seed", _seed(), _provenance(profile)) is not None
    return machine, host, held, unit


def test_an_oversized_unit_is_released_on_the_larger_class(tmp_path: Path, clock: ManualClock) -> None:
    """The seeded resident memory is over the headroom of `cpu`'s grant: the part is put on `cpu:large`, sized there."""
    machine, host, held, _ = _fitting(tmp_path, clock)
    machine.place(held)
    [release] = host.released
    assert release.held.queue == "cpu:large"
    assert release.held.part.placement.resource_class == "cpu:large"
    assert release.estimate is not None and release.estimate.device is not None


def test_a_unit_that_fits_no_class_stops_the_plan_unplaceable(tmp_path: Path, clock: ManualClock) -> None:
    """With no larger class the release raises `Unplaceable`, naming each class tried and the figures."""
    machine, host, held, _ = _fitting(tmp_path, clock, large=False)
    with pytest.raises(Unplaceable) as raised:
        machine.place(held)
    assert not host.released
    [tried] = raised.value.tried
    assert (tried.resource_class, tried.verdict, tried.grant_bytes) == ("cpu", "too small", SMALL)
    assert tried.resident == RESIDENT and f"resident {RESIDENT}" in str(raised.value)


def test_a_unit_that_fits_is_released_where_it_is(tmp_path: Path, clock: ManualClock) -> None:
    """A grant with room for the resident memory leaves the release on its own class."""
    machine, host, held, _ = _fitting(tmp_path, clock, small=2 * RESIDENT)
    machine.place(held)
    [release] = host.released
    assert release.held.queue == "cpu"


def test_a_batch_one_load_oom_is_recorded_as_unfit_for_the_class(tmp_path: Path, clock: ManualClock) -> None:
    """A probe item's load that ran out of memory is a fact about the unit and the grant, not an ignored B=1 oom."""
    machine, host, held, unit = _fitting(tmp_path, clock, small=2 * RESIDENT, seed=False)
    machine.place(held)
    [name] = [n for n in host.queues if probe_of(n)]
    digest = next(iter(host.queues[name].items))
    assert not machine.roots.unfit_records(unit)
    machine.unfit(digest, "4242")
    [record] = machine.roots.unfit_records(unit)
    assert (record.resource, record.grant_bytes, record.worker) == ("device", 2 * RESIDENT, "4242")
    assert not machine.roots.regime_records(unit)


def test_a_recorded_unfit_class_is_avoided_by_the_next_placement(tmp_path: Path, clock: ManualClock) -> None:
    """Two runs apart: the first records the load fault; the second, with room by measure, still moves up a class."""
    machine, host, held, unit = _fitting(tmp_path, clock, small=2 * RESIDENT)
    machine.place(held)
    [release] = host.released
    assert release.held.queue == "cpu"
    machine.note(release, ["item"])
    machine.unfit("item", None)
    again, other, held_again, _ = _fitting(tmp_path, clock, small=2 * RESIDENT, seed=False)
    assert again.roots.unfit_records(unit)
    again.place(held_again)
    assert [r.held.queue for r in other.released] == ["cpu:large"]


def test_an_unfit_record_in_a_larger_grant_leaves_no_class_to_place_on(tmp_path: Path, clock: ManualClock) -> None:
    """A load that ran out of memory in the larger grant too leaves the unit unplaceable."""
    machine, host, held, unit = _fitting(tmp_path, clock, small=2 * RESIDENT)
    assert machine.roots.put_record(
        unit,
        "unfit",
        UnfitRecord(resource="device", grant_bytes=LARGE, **_code(unit)),
        _provenance(machine.profile),
    )
    with pytest.raises(Unplaceable) as raised:
        machine.place(held)
    assert [t.verdict for t in raised.value.tried] == ["unfit", "unfit"]
    assert not host.released


def test_an_unmeasured_unit_recorded_unfit_is_not_probed_on_that_class(tmp_path: Path, clock: ManualClock) -> None:
    """A unit with no measure and an `unfit` record for its class is placed above it before any probe item is put."""
    machine, host, held, unit = _fitting(tmp_path, clock, small=2 * RESIDENT, seed=False)
    assert machine.roots.put_record(
        unit,
        "unfit",
        UnfitRecord(resource="device", grant_bytes=2 * RESIDENT, **_code(unit)),
        _provenance(machine.profile),
    )
    machine.place(held)
    probing = [n for n in host.queues if probe_of(n)]
    assert [class_of(n) for n in probing] == ["cpu:large"]


def test_a_probe_abandoned_after_a_load_fault_releases_its_jobs_above_the_class(
    tmp_path: Path, clock: ManualClock
) -> None:
    """The release of an abandoned probe has no estimate, and still reads the load fault the probe recorded."""
    machine, host, held, unit = _fitting(tmp_path, clock, small=2 * RESIDENT, seed=False)
    machine.place(held)
    digest = next(iter(host.queues[next(n for n in host.queues if probe_of(n))].items))
    machine.unfit(digest, None)
    clock.advance(machine.profile.lease.stale_after_s + 1)
    machine.advance()
    [release] = host.released
    assert release.estimate is None and release.held.queue == "cpu:large"
    assert release.held.part.placement.resource_class == "cpu:large"


def test_a_unit_with_no_unit_to_key_a_record_is_released_unjudged(tmp_path: Path, clock: ManualClock) -> None:
    """An unresolved unit has no digest to read a record by: it is released as before, and nothing is raised."""
    machine, host, held, _ = _fitting(tmp_path, clock, small=2 * RESIDENT, seed=False)
    machine._release(Release(held, None, None), [], fragment_jobs(held.part, host.placed, host.values)[1])  # noqa: SLF001 - the gate, by its one caller
    assert [(r.held.queue, r.unit) for r in host.released] == [("cpu", None)]


def test_every_release_goes_through_the_one_fit_gate() -> None:
    """The machine puts jobs through `host.produce` at one call site, the gate: no path releases a unit unjudged."""
    tree = ast.parse(inspect.getsource(machine_module))
    sites = [
        (fn.name, node.lineno)
        for fn in ast.walk(tree)
        if isinstance(fn, ast.FunctionDef)
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "produce"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "host"
    ]
    assert {name for name, _ in sites} == {"_release"}, sites


def test_a_record_of_other_code_does_not_bind_the_unit(tmp_path: Path, clock: ManualClock) -> None:
    """The record carries the senselab source and worker-overlay digest of the code that failed: others ignore it."""
    machine, host, held, unit = _fitting(tmp_path, clock, small=2 * RESIDENT)
    stale = UnfitRecord(resource="device", grant_bytes=LARGE, source="0" * 64, overlay="0" * 64)
    assert machine.roots.put_record(unit, "unfit", stale, _provenance(machine.profile))
    assert (stale.source, stale.overlay) != (code_of(unit).source, code_of(unit).overlay)
    machine.place(held)
    [release] = host.released
    assert release.held.queue == "cpu"


def test_a_written_record_carries_the_code_digest_and_a_group_can_clear_it(
    tmp_path: Path, clock: ManualClock, capsys: pytest.CaptureFixture[str]
) -> None:
    """`senselab unfit` lists a unit's records and `--clear` removes them, so a fixed loader is tried again."""
    from senselab_core.cli import main as cli_main  # noqa: PLC0415 - one use

    machine, host, held, unit = _fitting(tmp_path, clock, small=2 * RESIDENT)
    machine.place(held)
    [release] = host.released
    machine.note(release, ["item"])
    machine.unfit("item", None)
    [record] = machine.roots.unfit_records(unit)
    assert {"source": record.source, "overlay": record.overlay} == _code(unit)
    profile = tmp_path / "profile.json"
    profile.write_text(machine.profile.model_dump_json())
    assert cli_main(["unfit", "--profile", str(profile)]) == 0
    listed = capsys.readouterr().out
    assert unit.label() in listed and str(2 * RESIDENT) in listed
    assert cli_main(["unfit", unit.label(), "--clear", "--profile", str(profile)]) == 0
    assert "removed 1" in capsys.readouterr().out
    assert machine.roots.unfit_records(unit) == ()
    assert cli_main(["unfit", "--clear", "--profile", str(profile)]) == 1
    assert "names no unit" in capsys.readouterr().err


def test_clearing_needs_one_unit_and_an_ambiguous_prefix_is_refused(
    tmp_path: Path, clock: ManualClock, capsys: pytest.CaptureFixture[str]
) -> None:
    """A prefix that names several units clears none and lists them; a unique one clears its own."""
    from senselab_core.cli import main as cli_main  # noqa: PLC0415 - one use

    machine, host, held, unit = _fitting(tmp_path, clock, small=2 * RESIDENT)
    other = next(
        o
        for o in (unit.model_copy(update={"backend": f"elsewhere{n}"}) for n in range(500))
        if o.digest()[0] == unit.digest()[0]
    )
    provenance = _provenance(machine.profile)
    for each in (unit, other):
        record = UnfitRecord(resource="device", grant_bytes=RESIDENT, **_code(each))
        assert machine.roots.put_record(each, "unfit", record, provenance)
    ambiguous = unit.digest()[0]
    profile = tmp_path / "profile.json"
    profile.write_text(machine.profile.model_dump_json())
    assert cli_main(["unfit", ambiguous, "--clear", "--profile", str(profile)]) == 1
    err = capsys.readouterr().err
    assert "match" in err and unit.label() in err and other.label() in err
    assert machine.roots.unfit_records(unit) and machine.roots.unfit_records(other)
    assert cli_main(["unfit", unit.label(), "--clear", "--profile", str(profile)]) == 0
    assert machine.roots.unfit_records(unit) == () and machine.roots.unfit_records(other)
