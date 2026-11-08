"""Measure identity, the canonical hardware class and the lookup order, each against what `Measures.get` did."""

import getpass
import socket
from collections.abc import Sequence
from pathlib import Path

import pytest
from senselab_core.kernel.placement import NodePlacement
from senselab_core.runtime.calibration import Estimate, HardwareClass, HardwareGrant, Observation
from senselab_core.runtime.config import dump_yaml
from senselab_core.runtime.hardware import GpuInfo, HardwareInfo
from senselab_core.runtime.measures import MeasureSource, MeasureUnit
from senselab_core.runtime.resources import Derivation, MeasureRow, MeasureTable, PackagedTable
from senselab_core.testing.calibration_samples import ENV, UNIT
from senselab_worker.digest import identity_digest

PLACEMENT = NodePlacement(
    backend="whisper",
    backend_version="3",
    backend_params={"dtype": "float16"},
    stack="s",
    env="e",
    device="cuda",
    batch_size=1,
    resource_class="gpu:h100",
)
ENV_FIELDS = ENV.model_dump()
MEASURED = Estimate.model_validate(
    {
        "status": "measured",
        "unit_kind": "audio_s",
        "single": {"batch": 1, "r": 0.1, "u": 0.01, "n": 4, "spread": 0.0, "lo": 1.0, "hi": 5.0},
    }
)
UNMEASURED = Estimate(status="unmeasured", unit_kind="none", reasons=("no warm request",))


def _unit(placement: NodePlacement = PLACEMENT, env: dict[str, object] | None = None) -> MeasureUnit:
    return MeasureUnit.of(
        placement,
        UNIT.model,
        env or ENV_FIELDS,
        UNIT.hardware,
        task=UNIT.task,
        task_version=UNIT.task_version,
        formation="mixed",
    )


@pytest.mark.logic
def test_units_differing_only_in_backend_params_have_different_measures() -> None:
    """`Measures.get` keyed on (task, backend, model, class), so engine and dtype shared one row; these do not."""
    other = PLACEMENT.model_copy(update={"backend_params": {"dtype": "float32"}})
    assert _unit().digest() != _unit(other).digest()
    assert _unit().digest() == _unit().digest()
    assert _unit() == UNIT


@pytest.mark.logic
def test_units_differing_only_in_wheel_set_have_different_measures() -> None:
    """An env rebuilt on another torch index changes the wheel set, and the old row no longer answers for it."""
    rebuilt = {**ENV_FIELDS, "wheel_set": "f" * 64, "driver": {"torch_index": "cu121"}}
    assert _unit().digest() != _unit(env=rebuilt).digest()


@pytest.mark.logic
def test_a_unit_refuses_missing_env_fields() -> None:
    """A unit built without an env field raises, naming it."""
    with pytest.raises(ValueError, match="wheel_set"):
        _unit(env={k: v for k, v in ENV_FIELDS.items() if k != "wheel_set"})


@pytest.mark.logic
def test_the_digest_is_the_tagged_identity_of_the_fields() -> None:
    """The digest is `identity_digest` of the unit's JSON fields under its tag."""
    assert UNIT.digest() == identity_digest("measure-unit@1", UNIT.model_dump(mode="json"))


def _host(**update: object) -> HardwareInfo:
    fields: dict[str, object] = {
        "cpu_count": 32,
        "memory_bytes": 2**36,
        "gpus": (
            GpuInfo(name="NVIDIA L40S", memory_bytes=46068 * 2**20, driver="1", compute_capability="8.9", index=0),
        ),
        "apple_silicon": False,
        "platform": "linux-x86_64",
        "cpu_model": "AMD EPYC 9654",
    }
    return HardwareInfo(**{**fields, **update})  # type: ignore[arg-type]


@pytest.mark.logic
def test_a_cuda_class_carries_its_granted_cpus() -> None:
    """The class names the card and the cores the worker was granted, not the host's 32."""
    granted = HardwareClass.from_report(_host(), HardwareGrant(device="cuda", cpus=4))
    assert (granted.name, granted.memory_mib, granted.compute_capability, granted.cpus) == (
        "NVIDIA L40S",
        46068,
        "8.9",
        4,
    )
    assert HardwareClass.from_report(_host(), HardwareGrant(device="cuda")).cpus == 32
    assert granted != HardwareClass.from_report(_host(), HardwareGrant(device="cuda", cpus=8))


@pytest.mark.logic
def test_mps_and_cpu_classes_are_normalised() -> None:
    """An mps class carries the chip and unified memory, a cpu class the model and no memory."""
    mps = HardwareClass.from_report(
        _host(gpus=(), chip="Apple M2 Max", memory_bytes=32 * 2**30), HardwareGrant(device="mps", cpus=8)
    )
    assert (mps.device, mps.name, mps.memory_mib, mps.cpus) == ("mps", "Apple M2 Max", 32768, 8)
    cpu = HardwareClass.from_report(_host(), HardwareGrant(device="cpu", cpus=2))
    assert (cpu.name, cpu.memory_mib, cpu.cpus) == ("AMD EPYC 9654", None, 2)


@pytest.mark.logic
@pytest.mark.parametrize(
    ("grant", "host"),
    [
        (HardwareGrant(device="cuda", gpu=1), {}),
        (HardwareGrant(device="mps"), {}),
        (HardwareGrant(device="cpu"), {"cpu_model": None}),
    ],
)
def test_a_grant_the_report_cannot_satisfy_is_refused(grant: HardwareGrant, host: dict[str, object]) -> None:
    """A GPU, chip or CPU model the report lacks is refused."""
    with pytest.raises(ValueError):
        HardwareClass.from_report(_host(**host), grant)


class _Root:
    def __init__(self, observed: bool) -> None:
        self.observed = observed

    def observations(self, unit: MeasureUnit) -> Sequence[Observation]:
        return (object(),) if self.observed else ()  # type: ignore[return-value]


class _Packaged:
    def __init__(self, estimate: Estimate | None) -> None:
        self.estimate = estimate

    def get(self, unit: MeasureUnit) -> Estimate | None:
        return self.estimate


@pytest.mark.logic
@pytest.mark.parametrize(
    ("roots", "packaged", "derived", "expected"),
    [
        ([True], MEASURED, MEASURED, MEASURED),
        ([True], None, UNMEASURED, None),
        ([False], MEASURED, MEASURED, MEASURED),
        ([True], UNMEASURED, UNMEASURED, None),
        ([], None, MEASURED, None),
    ],
)
def test_the_first_measured_estimate_wins_then_the_packaged_table(
    roots: list[bool], packaged: Estimate | None, derived: Estimate, expected: Estimate | None
) -> None:
    """A root's measured estimate wins; an unmeasured one falls through to the packaged row; none probes."""
    source = MeasureSource([_Root(o) for o in roots], _Packaged(packaged), estimate=lambda u, obs: derived)
    assert source.get(UNIT) == expected


def _table() -> MeasureTable:
    row = MeasureRow(
        unit=UNIT, estimate=MEASURED, derivation=Derivation(commit="d" * 40, jobs=("j1",), probe_ids=("p1",))
    )
    return MeasureTable(task=UNIT.task, backend=UNIT.backend, rows=(row,))


def _write(root: Path, table: MeasureTable) -> Path:
    path = root / table.task / f"{table.backend}.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(dump_yaml(table.model_dump(mode="json")))
    return path


@pytest.mark.logic
def test_a_packaged_row_is_read_only_on_exact_identity(tmp_path: Path) -> None:
    """`Unit` matched a row on (model, class) alone; a row now answers for its own env, params and card only."""
    _write(tmp_path, _table())
    packaged = PackagedTable.load(tmp_path)
    assert packaged.get(UNIT) == MEASURED
    for update in (
        {"backend_params": {"dtype": "float32"}},
        {"env": ENV.model_copy(update={"wheel_set": "f" * 64})},
        {"hardware": UNIT.hardware.model_copy(update={"cpus": 4})},
        {"model": None},
    ):
        assert packaged.get(UNIT.model_copy(update=update)) is None


@pytest.mark.logic
def test_packaged_rows_carry_no_user_or_host(tmp_path: Path) -> None:
    """I7: no account or host name in a row, and a row that adds one is refused, not ignored."""
    path = _write(tmp_path, _table())
    text = path.read_text()
    assert getpass.getuser() not in text.split() and socket.gethostname() not in text
    row = _table().model_dump(mode="json")["rows"][0]
    for part in (row, row["unit"], row["unit"]["env"], row["derivation"]):
        assert not {"user", "username", "host", "hostname", "account"} & set(part)
    doc = _table().model_dump(mode="json")
    doc["rows"][0]["derivation"]["host"] = "node1"
    path.write_text(dump_yaml(doc))
    with pytest.raises(ValueError):
        PackagedTable.load(tmp_path)


@pytest.mark.logic
def test_a_table_refuses_a_row_of_another_backend_or_a_unit_twice() -> None:
    """A table holds rows of its own task and backend, one per unit."""
    row = _table().rows[0]
    with pytest.raises(ValueError, match="holds a row of"):
        MeasureTable(task="speech_to_text", backend="other", rows=(row,))
    with pytest.raises(ValueError, match="two rows"):
        MeasureTable(task=UNIT.task, backend=UNIT.backend, rows=(row, row))


@pytest.mark.logic
def test_an_absent_root_holds_no_rows(tmp_path: Path) -> None:
    """A root that is not there holds no packaged rows."""
    assert PackagedTable.load(tmp_path / "none").get(UNIT) is None
