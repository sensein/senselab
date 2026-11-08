"""The calibration data types: what each refuses, and that an `Observation` and an `Estimate` round-trip."""

from typing import Any

import pytest
from senselab_core.runtime.calibration import (
    Cotenancy,
    Estimate,
    HardwareClass,
    LengthProfile,
    Observation,
    ProbeRef,
    RequestFit,
)

HARDWARE = HardwareClass(
    device="cuda", name="NVIDIA H100 80GB HBM3", memory_mib=81559, compute_capability="9.0", cpus=8
)


def _observation(**update: Any) -> Observation:  # noqa: ANN401
    fields: dict[str, Any] = {
        "probe": ProbeRef(id="p1", phase="SINGLE", stratum=0),
        "item": "i1",
        "request_index": 0,
        "jobs": 1,
        "batch": 1,
        "units": 4.0,
        "padded_units": 4.0,
        "wall_s": 0.5,
        "item_wall_s": None,
        "since_load": 3,
        "load_s": None,
        "device_peak": None,
        "device_after": None,
        "host_peak": None,
        "host_after": None,
        "cotenancy": Cotenancy(device=1, host=1),
        "outcome": "ok",
        "hardware": HARDWARE,
    }
    return Observation.model_validate({**fields, **update})


@pytest.mark.logic
@pytest.mark.parametrize(
    "bad",
    [
        {"batch": 0},
        {"jobs": 0},
        {"units": -1.0},
        {"padded_units": 3.0},
        {"wall_s": float("nan")},
        {"since_load": -1},
        {"outcome": "lost"},
        {"probe": {"id": "p", "phase": "IDLE", "stratum": 0}},
        {"device_peak": 5, "device_after": 6},
        {"cotenancy": {"device": 0, "host": 1}},
    ],
)
def test_an_observation_refuses_what_a_worker_cannot_have_measured(bad: dict[str, Any]) -> None:
    """Padding never shrinks a request, a peak is not below what stayed allocated, a phase is one of three."""
    with pytest.raises(ValueError):
        _observation(**bad)


@pytest.mark.logic
def test_a_production_request_carries_no_probe_and_round_trips() -> None:
    """`probe` is None for a production request; the value survives its own JSON."""
    o = _observation(probe=None, device_peak=10, device_after=4)
    assert Observation.model_validate_json(o.model_dump_json()) == o


@pytest.mark.logic
def test_a_hardware_class_names_its_card_and_its_cores() -> None:
    """A cuda class has a product name and memory; every class has at least one cpu."""
    with pytest.raises(ValueError):
        HardwareClass(device="cuda", name="x", memory_mib=None, cpus=1)
    with pytest.raises(ValueError):
        HardwareClass(device="cpu", name="x", cpus=0)
    assert HardwareClass(device="cpu", name="Xeon", cpus=4).memory_mib is None


@pytest.mark.logic
def test_an_estimate_is_measured_by_its_per_request_cost_or_says_why_not() -> None:
    """`measured` needs the single fit's `r`; `unmeasured` needs reasons and holds no device figure as zero."""
    fit = RequestFit(batch=1, r=0.2, u=0.05, n=6, spread=0.1, lo=2.0, hi=9.0)
    ok = Estimate(status="measured", unit_kind="audio_s", load_s=3.0, load_n=1, single=fit)
    assert ok.estimator == "estimate@1" and ok.device is None and ok.reasons == ()
    with pytest.raises(ValueError, match="single"):
        Estimate(status="measured", unit_kind="audio_s")
    with pytest.raises(ValueError, match="reasons"):
        Estimate(status="unmeasured", unit_kind="none")
    assert Estimate(status="unmeasured", unit_kind="none", reasons=("no warm request",)).single is None
    with pytest.raises(ValueError):
        RequestFit(batch=1, r=0.2, u=None, n=1, spread=None, lo=5.0, hi=2.0)


@pytest.mark.logic
def test_a_length_profile_holds_units_in_put_order_with_unknowns() -> None:
    """A job whose length is not yet known is None; a length is not negative."""
    assert LengthProfile(units=(3.0, None, 1.0)).units == (3.0, None, 1.0)
    with pytest.raises(ValueError):
        LengthProfile(units=(-1.0,))
