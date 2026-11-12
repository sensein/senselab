"""Placement judges a unit against its class: an oversized one moves up, and one that fits none is unplaceable."""

from collections.abc import Sequence
from typing import Any

import pytest
from planner_test import (
    NO_GPU,
    _call,
    _plan,
    _records,
    packaged,  # noqa: F401 - the autouse fixture that installs the toy registry
)
from senselab_core.runtime.calibration import Estimate, EstimateSource, MemoryFit, RequestFit, StratumBatch, UnfitRecord
from senselab_core.runtime.fit import Unplaceable, choose, judge
from senselab_core.runtime.measures import MeasureSource, MeasureUnit
from senselab_core.runtime.planner import CALL_NODE, Plan
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.testing.calibration_samples import UNIT, observation

pytestmark = pytest.mark.logic

_GB = 10**9


def _estimate(resident: int, m0: float = 0.0) -> Estimate:
    return Estimate(
        status="measured",
        unit_kind="audio_s",
        load_s=2.0,
        load_n=3,
        item_s=0.05,
        item_n=3,
        single=RequestFit(batch=1, r=0.5, u=0.05, n=6, spread=0.1, lo=2.0, hi=1000.0),
        batch=(StratumBatch(lo=2.0, hi=1000.0, batch=1),),
        host=MemoryFit(resident=resident, m0=m0, m1=1.0, n=6),
        source=EstimateSource(probe_ids=("p1",), observations=6),
    )


class _Roots:
    def observations(self, unit: MeasureUnit) -> tuple[Any, ...]:
        return (observation(),)


def _source(estimate: Estimate | None) -> MeasureSource:
    return MeasureSource([_Roots()], estimate=lambda unit, observed: estimate or _unmeasured())


def _unmeasured() -> Estimate:
    return Estimate(status="unmeasured", unit_kind="none", reasons=("no_usable_observations",))


def _profile(*, large: bool = True) -> ExecutionProfile:
    classes: dict[str, Any] = {"cpu": {"device": "cpu", "mem_bytes": _GB, "max_workers": 1}}
    if large:
        classes["cpu:large"] = {"device": "cpu", "mem_bytes": 8 * _GB, "max_workers": 1}
    return ExecutionProfile.load("laptop", {"devices": ["cpu"], "resource_classes": classes})


def _planned(profile: ExecutionProfile) -> Plan:
    return _plan(_call(), _records("x"), hardware=NO_GPU, execution=profile)


def _none(unit: MeasureUnit) -> Sequence[UnfitRecord]:
    return ()


def _read(placed: Plan, estimate: Estimate | None, unfit: Sequence[UnfitRecord] = ()) -> dict[str, Any]:
    return {"measures": _source(estimate), "units": lambda node: UNIT, "unfit": lambda unit: unfit, "hardware": NO_GPU}


def test_an_oversized_unit_is_placed_on_the_larger_class() -> None:
    """Resident plus m0 over the headroom of a class's grant skips it; the next larger class that fits is taken."""
    placed = _planned(_profile())
    assert placed.placement.nodes[CALL_NODE].resource_class == "cpu"
    big = _estimate(950_000_000, 1000.0)
    fitted = placed.fit(**_read(placed, big))
    node = fitted.placement.nodes[CALL_NODE]
    assert node.resource_class == "cpu:large" and "moved from cpu" in node.why and "resident 950000000" in node.why
    assert fitted.hash == placed.hash
    assert placed.fit(**_read(placed, _estimate(100_000_000))).placement == placed.placement


def test_a_unit_that_fits_no_class_is_unplaceable_and_explain_names_the_numbers() -> None:
    """With no larger class the plan stops; `explain` shows the unit, each class tried and the figures."""
    placed = _planned(_profile(large=False))
    big = _estimate(950_000_000, 1000.0)
    with pytest.raises(Unplaceable) as raised:
        placed.fit(**_read(placed, big))
    [tried] = raised.value.tried
    assert (tried.resource_class, tried.verdict, tried.resident, tried.m0) == ("cpu", "too small", 950_000_000, 1000.0)
    assert "node 'call'" in str(raised.value) and "950000000" in str(raised.value) and "900000000" in str(raised.value)
    text = placed.explain(**_read(placed, big), lengths=None)
    fit = next(line for line in text.splitlines() if line.startswith("  fit:"))
    assert "unplaceable" in fit and "class cpu" in fit and "resident 950000000 + m0 1000" in fit


def test_explain_shows_the_move_with_the_class_tried() -> None:
    """A node that moves is printed on the class it moves to, after the class it left and why."""
    placed = _planned(_profile())
    text = placed.explain(**_read(placed, _estimate(950_000_000)))
    fit = next(line for line in text.splitlines() if line.startswith("  fit:"))
    assert "class cpu (1000000000 bytes) is too small" in fit and fit.endswith("placed on class cpu:large")
    assert "resource class cpu:large" in text


def test_an_unfit_record_skips_the_class_it_names_and_every_smaller_one() -> None:
    """A load that ran out of memory in a grant makes that grant and any below it unfit, whatever was measured."""
    placed = _planned(_profile())
    record = UnfitRecord(resource="host", grant_bytes=_GB)
    fitted = placed.fit(**_read(placed, _estimate(10_000_000), [record]))
    assert fitted.placement.nodes[CALL_NODE].resource_class == "cpu:large"
    assert "a load ran out of memory in a grant of 1000000000" in fitted.placement.nodes[CALL_NODE].why
    both = UnfitRecord(resource="host", grant_bytes=8 * _GB)
    with pytest.raises(Unplaceable) as raised:
        placed.fit(**_read(placed, None, [both]))
    assert [t.verdict for t in raised.value.tried] == ["unfit", "unfit"]
    other = UnfitRecord(resource="device", grant_bytes=8 * _GB)
    assert placed.fit(**_read(placed, None, [other])).placement == placed.placement


def test_a_unit_with_no_measure_and_no_record_stays_where_it_is() -> None:
    """Nothing to judge by: the placement is the plan's own."""
    placed = _planned(_profile())
    assert placed.fit(**_read(placed, None)).placement == placed.placement


def test_judge_is_the_one_rule() -> None:
    """At the boundary resident + m0 equal to the limit fits; above it is too small; no grant is unknown."""
    fit = MemoryFit(resident=90, m0=0.0, m1=1.0, n=3)
    assert judge("c", 100, fit, "host", (), 0.9).verdict == "fits"
    assert judge("c", 99, fit, "host", (), 0.9).verdict == "too small"
    assert judge("c", None, fit, "host", (), 0.9).verdict == "grant unknown"
    assert judge("c", 100, None, "host", (), 0.9).verdict == "fits"
    profile = _profile()
    chosen = choose(profile, "cpu", fit=fit, resource="host", unfit=(), grant_of={"cpu": 50, "cpu:large": 500}.get)
    assert chosen.to == "cpu:large" and [t.verdict for t in chosen.tried] == ["too small", "fits"]
