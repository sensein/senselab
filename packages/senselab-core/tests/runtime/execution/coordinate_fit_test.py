"""`coordinate` fits the plan to its classes as `start` does: an unplaceable run stops before an attempt is claimed."""

from pathlib import Path

import pytest
from epoch_fixture import prepared
from pool_probe_test import (
    JOBS,
    Idle,
    _plan,
    _profile,
    stack,  # noqa: F401 - the registry and env the plans need
)
from probe_machine_oom_test import _provenance, _seed
from senselab_core.cli import main as cli_main
from senselab_core.runtime.calibration import HardwareClass, HardwareGrant
from senselab_core.runtime.execution.epoch import Epoch
from senselab_core.runtime.execution.plan_reads import unit_of_node
from senselab_core.runtime.fit import Unplaceable
from senselab_core.runtime.measures import MeasureUnit
from senselab_core.runtime.store import ResultStore
from senselab_core.runtime.units import UnitResolver, measure_roots

pytestmark = pytest.mark.logic

RESIDENT = 9 * 10**9  # above the headroom share of the profile's 2**33-byte class


def test_coordinate_stops_an_unplaceable_run_before_it_claims_an_attempt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The unit does not fit its one class: `coordinate` raises, and the CLI exits typed without a traceback."""
    pytest.importorskip("pydra")
    from senselab_core.runtime.execution.pool import coordinate  # noqa: PLC0415 - after the pydra check

    profile = _profile(tmp_path)
    placed, values = _plan(profile, range(1, JOBS + 1))
    store, run_dir = ResultStore(tmp_path / "store"), tmp_path / "run"
    prepared(placed, run_dir, store)
    before = Epoch(run_dir).current()
    roots = measure_roots(profile)
    resolver = UnitResolver(profile, roots)
    seen = HardwareClass.from_report(resolver.hardware, HardwareGrant(device="cpu", cpus=None))
    assert roots.note_seen(profile.resource_classes["cpu"], seen) is not None
    unit = unit_of_node(placed, resolver)("a")
    assert isinstance(unit, MeasureUnit)
    observations = [
        o.model_copy(
            update={
                "device_peak": None,
                "device_after": None,
                "host_peak": RESIDENT + (o.device_peak or 0) - 10**9,
                "host_after": RESIDENT,
                "hardware": seen,
            }
        )
        for o in _seed()
    ]
    assert roots.put_observations(
        unit, "seed", observations, _provenance(profile).model_copy(update={"hardware": seen})
    )
    with pytest.raises(Unplaceable) as raised:
        coordinate(run_dir, substrate=Idle(), profile=profile, store=store)
    assert f"resident {RESIDENT}" in str(raised.value)
    assert Epoch(run_dir).current() == before
    saved = tmp_path / "profile.json"
    saved.write_text(profile.model_dump_json())
    assert cli_main(["coordinate", str(run_dir), "--profile", str(saved), "--external"]) == 3
    err = capsys.readouterr().err
    assert err.startswith("senselab: unplaceable:") and "Traceback" not in err and f"resident {RESIDENT}" in err
