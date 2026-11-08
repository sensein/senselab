"""Measure identity, the canonical hardware class and the lookup order, each against what `Measures.get` did."""

import getpass
import platform
import re
import socket
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

import pytest
from senselab_core.kernel.identity import Attestation, LogicalKey, ModelIdentity
from senselab_core.runtime.calibration import Estimate, HardwareClass, HardwareGrant, Observation
from senselab_core.runtime.config import dump_yaml
from senselab_core.runtime.fake_backend import FAKE
from senselab_core.runtime.hardware import GpuInfo, HardwareInfo, _cpu_model, probe_hardware
from senselab_core.runtime.keys import Placement
from senselab_core.runtime.measures import MeasureSource, MeasureUnit
from senselab_core.runtime.resources import Derivation, MeasureRow, MeasureTable, PackagedTable, load_resource_tables
from senselab_core.testing.calibration_samples import ENV, UNIT
from senselab_worker.digest import identity_digest

PLACEMENT = Placement(backend=FAKE, backend_params=FAKE.params(), device="cpu", batch_formation="mixed")
KEY = LogicalKey(
    task=UNIT.task,
    task_version=UNIT.task_version,
    backend=UNIT.backend,
    backend_version=UNIT.backend_version,
    model=UNIT.model,
    params={"word_timestamps": False},
    backend_params={"dtype": "float16"},
    inputs={},
)
ENV_FIELDS = {**ENV.model_dump(), "driver": {"torch_index": "https://download.pytorch.org/whl/cu128"}}
OBSERVATION = Observation.model_validate(
    {
        "probe": None,
        "item": "i",
        "request_index": 0,
        "jobs": 1,
        "batch": 1,
        "units": 1.0,
        "padded_units": 1.0,
        "wall_s": 0.1,
        "item_wall_s": None,
        "since_load": 1,
        "load_s": None,
        "device_peak": None,
        "device_after": None,
        "host_peak": None,
        "host_after": None,
        "cotenancy": {"device": 1, "host": 1},
        "outcome": "ok",
        "hardware": UNIT.hardware,
    }
)
MEASURED = Estimate.model_validate(
    {
        "status": "measured",
        "unit_kind": "audio_s",
        "single": {"batch": 1, "r": 0.1, "u": 0.01, "n": 4, "spread": 0.0, "lo": 1.0, "hi": 5.0},
    }
)
UNMEASURED = Estimate(status="unmeasured", unit_kind="none", reasons=("no warm request",))


def _unit(placement: Placement = PLACEMENT, key: LogicalKey = KEY, env: dict[str, object] | None = None) -> MeasureUnit:
    return MeasureUnit.of(placement, key, env or ENV_FIELDS, UNIT.hardware)


@pytest.mark.logic
def test_units_differing_only_in_backend_params_have_different_measures() -> None:
    """`Measures.get` keyed on (task, backend, model, class), so engine and dtype shared one row; these do not."""
    other = KEY.model_copy(update={"backend_params": {"dtype": "float32"}})
    assert _unit().digest() != _unit(key=other).digest()
    assert _unit().digest() == _unit().digest()
    assert _unit() == UNIT


@pytest.mark.logic
def test_units_differing_only_in_task_params_or_chunk_grid_have_different_measures() -> None:
    """A task param that shapes cost, or a chunk grid, changes the measure; `Measures.get` had neither."""
    words = KEY.model_copy(update={"params": {"word_timestamps": True}})
    assert _unit().digest() != _unit(key=words).digest()
    gridded = replace(PLACEMENT, chunk_grid="30s")
    assert _unit().digest() != _unit(placement=gridded).digest()
    assert _unit(placement=gridded).chunk_grid == "30s"


@pytest.mark.logic
def test_the_formation_is_read_from_the_placement() -> None:
    """The formation is the placement's, and an undecided one is refused rather than guessed."""
    assert _unit(placement=replace(PLACEMENT, batch_formation="single")).formation == "single"
    with pytest.raises(ValueError, match="not decided"):
        _unit(placement=replace(PLACEMENT, batch_formation=None))


@pytest.mark.logic
def test_units_differing_only_in_wheel_set_have_different_measures() -> None:
    """An env rebuilt on another torch index changes the wheel set, and the old row no longer answers for it."""
    rebuilt = {**ENV_FIELDS, "wheel_set": "f" * 64, "driver": {"torch_index": "https://download.pytorch.org/whl/cu121"}}
    assert _unit().digest() != _unit(env=rebuilt).digest()


@pytest.mark.logic
@pytest.mark.parametrize(
    ("index", "build"),
    [
        ("https://download.pytorch.org/whl/cu128", "cu128"),
        ("https://download.pytorch.org/whl/cpu", "cpu"),
        ("https://deploy:s3cret@mirror.corp.internal/pytorch/cu121/", "cu121"),
        ("https://deploy:s3cret@mirror.corp.internal/simple", "other"),
    ],
)
def test_the_unit_holds_the_torch_build_never_the_index_url(index: str, build: str) -> None:
    """The index URL may carry userinfo or an internal host; neither reaches the unit, its JSON or its digest input."""
    unit = _unit(env={**ENV_FIELDS, "driver": {"torch_index": index}})
    text = unit.model_dump_json()
    assert unit.env.driver == {"torch_build": build}
    assert "s3cret" not in text and "internal" not in text and "://" not in text
    other = _unit(env={**ENV_FIELDS, "driver": {"torch_index": "https://other.example/whl/cu999"}})
    assert unit.digest() != other.digest()


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
    def __init__(self, observed: int) -> None:
        self.observed = observed

    def observations(self, unit: MeasureUnit) -> Sequence[Observation]:
        return tuple(OBSERVATION for _ in range(self.observed))


class _Packaged:
    def __init__(self, estimate: Estimate | None) -> None:
        self.estimate = estimate

    def get(self, unit: MeasureUnit) -> Estimate | None:
        return self.estimate


def _estimator(minimum: int) -> Callable[[MeasureUnit, Sequence[Observation]], Estimate]:
    def estimate(unit: MeasureUnit, observations: Sequence[Observation]) -> Estimate:
        return MEASURED if len(observations) >= minimum else UNMEASURED

    return estimate


@pytest.mark.logic
@pytest.mark.parametrize(
    ("roots", "packaged", "expected"),
    [
        ([3], None, MEASURED),
        ([1], None, None),
        ([0], MEASURED, MEASURED),
        ([1], UNMEASURED, None),
        ([], None, None),
        ([2, 1], None, MEASURED),
        ([1, 1], None, None),
    ],
)
def test_roots_are_pooled_then_estimated_once_then_the_packaged_table(
    roots: list[int], packaged: Estimate | None, expected: Estimate | None
) -> None:
    """Observations of every root are estimated together, so two roots below the threshold alone can reach it."""
    source = MeasureSource([_Root(n) for n in roots], _Packaged(packaged), estimate=_estimator(3))
    assert source.get(UNIT) == expected


@pytest.mark.logic
def test_the_estimator_is_called_once_over_the_pooled_observations() -> None:
    """One call, with the roots' observations in root order."""
    calls: list[int] = []

    def estimate(unit: MeasureUnit, observations: Sequence[Observation]) -> Estimate:
        calls.append(len(observations))
        return MEASURED

    MeasureSource([_Root(2), _Root(4)], _Packaged(None), estimate=estimate).get(UNIT)
    assert calls == [6]


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


@pytest.mark.logic
def test_both_table_readers_refuse_a_file_under_another_name(tmp_path: Path) -> None:
    """The directory walk is one function: a `@2.0` file and a `@1.0` file are held to their path alike."""
    path = _write(tmp_path, _table())
    moved = path.with_name("other.yaml")
    path.rename(moved)
    with pytest.raises(ValueError, match="holds the table of"):
        PackagedTable.load(tmp_path)
    with pytest.raises(ValueError):
        load_resource_tables(tmp_path)


LSCPU_X86 = "Architecture: x86_64\nModel name: AMD EPYC 9654 96-Core Processor\nCPU(s): 192\n"
LSCPU_ARM = "Architecture: aarch64\nModel name: Neoverse-N1\n"


def _fail(command: list[str]) -> str:
    raise FileNotFoundError(command[0])


@pytest.mark.logic
@pytest.mark.parametrize(
    ("system", "output", "expected"),
    [
        ("Linux", LSCPU_X86, "AMD EPYC 9654 96-Core Processor"),
        ("Linux", LSCPU_ARM, "Neoverse-N1"),
        ("Linux", "Architecture: aarch64\n", None),
        ("Darwin", "Intel(R) Core(TM) i7-8700B CPU @ 3.20GHz\n", "Intel(R) Core(TM) i7-8700B CPU @ 3.20GHz"),
        ("Darwin", "\n", None),
    ],
)
def test_the_cpu_model_is_read_from_the_host_command_never_the_architecture(
    system: str, output: str, expected: str | None
) -> None:
    """A host that does not name its CPU gets None, not `aarch64` or `i386`, so classes never merge by architecture."""
    assert _cpu_model(lambda command: output, system, lambda: "") == expected
    assert _cpu_model(_fail, system, lambda: "") is None


CPUINFO = "processor\t: 0\nmodel name\t: Intel(R) Xeon(R) Gold 6248\n"
LSCPU_DE = "Architektur: x86_64\nModellname: AMD EPYC 9654\n"


@pytest.mark.logic
def test_cpuinfo_names_the_cpu_when_lscpu_is_absent() -> None:
    """A slim image without `lscpu` still forms a cpu class from `/proc/cpuinfo`."""
    assert _cpu_model(_fail, "Linux", lambda: CPUINFO) == "Intel(R) Xeon(R) Gold 6248"


@pytest.mark.logic
def test_lscpu_runs_in_the_c_locale_and_a_translated_answer_names_nothing() -> None:
    """`lscpu` is asked under `LC_ALL=C`; a translated `Modellname` is not read as a model, so the result is None."""
    asked: list[list[str]] = []

    def run(command: list[str]) -> str:
        asked.append(command)
        return LSCPU_DE

    assert _cpu_model(run, "Linux", lambda: "") is None
    assert asked == [["env", "LC_ALL=C", "lscpu"]]


@pytest.mark.logic
def test_neither_source_names_the_cpu() -> None:
    """With no cpuinfo model and no lscpu, the model is None, never the architecture."""
    assert _cpu_model(_fail, "Linux", lambda: "processor\t: 0\n") is None


@pytest.mark.logic
def test_probe_hardware_records_the_cpu_model() -> None:
    """The probe's `cpu_model` comes from the injected sources (a Darwin host takes the sysctl answer)."""
    info = probe_hardware(run=_fail, cpuinfo=lambda: CPUINFO)
    expected = None if platform.system() == "Darwin" else "Intel(R) Xeon(R) Gold 6248"
    assert info.cpu_model == expected


ATTESTATION = Attestation(
    engine_version="0.11.0",
    served_model="/home/user/models/m",
    model_identity="e" * 64,
    config_digest="f" * 64,
)
SERVER_DRIVER = {
    "torch_index": "https://download.pytorch.org/whl/cu128",
    "engine": "vllm",
    **ATTESTATION.driver(),
    "launch_env": "9" * 64,
}


@pytest.mark.logic
def test_a_server_unit_over_a_local_model_names_no_path_or_user() -> None:
    """The service is held as a digest and the attestation's served path is dropped, so no host string remains."""
    key = KEY.model_copy(update={"model": ModelIdentity(id=None, identity="e" * 64), "service": ATTESTATION})
    unit = _unit(key=key, env={**ENV_FIELDS, "driver": SERVER_DRIVER})
    text = unit.model_dump_json()
    assert "/home" not in text and "user" not in text and "models/m" not in text
    assert unit.service is not None and re.fullmatch(r"[0-9a-f]{64}", unit.service)
    assert set(unit.env.driver) == {"torch_build", "engine", "engine_version", "config_digest", "launch_env"}
    other = key.model_copy(update={"service": ATTESTATION.model_copy(update={"config_digest": "0" * 64})})
    assert _unit(key=other, env={**ENV_FIELDS, "driver": SERVER_DRIVER}).digest() != unit.digest()


@pytest.mark.logic
def test_a_driver_key_that_is_not_declared_is_refused() -> None:
    """A driver entry outside the declared tags and digests is refused, whatever its value looks like."""
    with pytest.raises(ValueError, match="holds"):
        _unit(env={**ENV_FIELDS, "driver": {"torch_index": "https://x/whl/cpu", "cache_dir": "/home/user/c"}})
