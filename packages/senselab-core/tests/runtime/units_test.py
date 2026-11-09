"""The plan-time unit resolver: `explain` and the in-process executor read measures before any worker starts."""

import platform
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path

import pytest
from senselab_core.kernel.dataset import Record, ValueInput
from senselab_core.kernel.declare import BackendSpec, Env, Registry, TaskParams, TaskSpec
from senselab_core.kernel.graph import Graph
from senselab_core.runtime.calibration import (
    Estimate,
    EstimateSource,
    HardwareClass,
    MemoryFit,
    Observation,
    RequestFit,
    StratumBatch,
)
from senselab_core.runtime.call import Packaged, RunConfig, bind_input, use_packaged
from senselab_core.runtime.catalogue import Catalogue
from senselab_core.runtime.datasets import RecordSet
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.execution.in_process import InProcess, fragment_jobs, node_fragment
from senselab_core.runtime.execution.workers import WorkerSet
from senselab_core.runtime.hardware import HardwareInfo
from senselab_core.runtime.licenses import license_record
from senselab_core.runtime.measures import MeasureSource, MeasureUnit
from senselab_core.runtime.planner import Plan, plan
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.store import ResultStore
from senselab_core.runtime.units import measure_roots
from senselab_core.testing.calibration_samples import observation
from senselab_core.testing.interpreter import source_tree_python

SCORE = TaskSpec(
    name="units_test.score",
    version="1",
    kind="transform",
    inputs={"text": "json"},
    outputs={"value": "sha256"},
    params=TaskParams,
)
REGISTRY = Registry()
REGISTRY.add_task(SCORE)
REGISTRY.add_backend(
    BackendSpec(
        task=SCORE.name,
        name="score",
        version="1",
        stack="dsp",
        env=Env.host(),
        devices=("cpu",),
        batch_invariant=True,
        worker="backends.sizes_backend_fixture",
        load_stub={},
    )
)
CATALOGUE = Catalogue.load("format: catalogue@2.0\nentries: []\n")
HOST = HardwareInfo(
    cpu_count=4, memory_bytes=2**33, gpus=(), apple_silicon=False, platform="linux-x86_64", cpu_model="Test CPU"
)
SEEN = HardwareClass(device="cpu", name="Test CPU", cpus=4)
OTHER = HardwareClass(device="cpu", name="Another CPU", cpus=4)


def _handle(backend: BackendSpec, profile: ExecutionProfile) -> EnvHandle:  # noqa: ARG001 - an EnvOpener
    return EnvHandle(
        python=source_tree_python(),
        python_version=platform.python_version(),
        lock_sha="1" * 64,
        wheel_set="2" * 64,
        torch_index=None,
    )


def _unbuilt(backend: BackendSpec, profile: ExecutionProfile) -> None:  # noqa: ARG001 - an EnvRead
    return None


def _use(read: object = _handle) -> Packaged:
    records = {
        "pkg": license_record("MIT", source="lock", verified={"date": "2026-11-02", "how": "pep639"}, gated=False)
    }
    return Packaged(CATALOGUE, _handle, REGISTRY, lambda _: records, env_read=read)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def packaged() -> Iterator[None]:
    """Plans read this module's registry; its host env is built unless a test says otherwise."""
    previous = use_packaged(_use())
    yield
    use_packaged(previous)


def _profile(root: Path, **settings: object) -> ExecutionProfile:
    given = {"devices": ["cpu"], "calibration": {"measures_roots": [str(root / "measures")]}, **settings}
    return ExecutionProfile.load("laptop", given)


def _plan(rows: Sequence[str], profile: ExecutionProfile) -> tuple[Plan, dict[str, object]]:
    graph = Graph.model_validate(
        {
            "inputs": {"text": "list[json]"},
            "nodes": [{"id": "a", "task": SCORE.name, "task_version": "1"}],
            "edges": [{"source": "input.text", "target": "a.text", "kind": "map"}],
        }
    )
    records = [Record(id=f"r{i}", inputs={"text": ValueInput(value=v)}) for i, v in enumerate(rows)]
    placed = plan(graph, RecordSet(records), config=RunConfig(), execution=profile)
    return placed, {bind_input("x", "json", v)[0]: v for v in rows}


class _Root:
    def observations(self, unit: MeasureUnit) -> Sequence[Observation]:
        return (observation(),)


class _NoTable:
    def get(self, unit: MeasureUnit) -> None:
        return None


def _estimate() -> Estimate:
    return Estimate(
        status="measured",
        unit_kind="none",
        load_s=1.0,
        load_n=3,
        item_s=0.01,
        item_n=3,
        single=RequestFit(batch=1, r=0.5, u=0.0, n=6, spread=0.1, lo=0.0, hi=0.0),
        batch=(StratumBatch(lo=0.0, hi=0.0, batch=1),),
        host=MemoryFit(resident=2**20, m0=0.0, m1=1.0, n=6),
        source=EstimateSource(probe_ids=("p1", "p2"), observations=12),
    )


def _sizing_line(text: str) -> str:
    return next(line for line in text.splitlines() if line.startswith("  sizing:"))


def _seen(profile: ExecutionProfile, *classes: HardwareClass) -> None:
    roots = measure_roots(profile)
    for seen in classes:
        assert roots.note_seen(profile.resource_classes["cpu"], seen) is not None


@pytest.mark.logic
def test_explain_is_measured_when_the_store_has_the_unit(tmp_path: Path) -> None:
    """With the env built and one hardware class seen, the plan's unit is resolved and its figures are measured."""
    profile = _profile(tmp_path)
    _seen(profile, SEEN)
    placed, bound = _plan(["x"], profile)
    asked: list[MeasureUnit] = []

    def estimate(unit: MeasureUnit, observed: Sequence[Observation]) -> Estimate:  # noqa: ARG001 - an estimator
        asked.append(unit)
        return _estimate()

    source = MeasureSource([_Root()], _NoTable(), estimate=estimate)
    text = placed.explain(measures=source, hardware=HOST, values=bound)
    (unit,) = {u.digest(): u for u in asked}.values()
    assert unit.hardware == SEEN and unit.env.wheel_set == "2" * 64 and unit.env.env_lock == "1" * 64
    assert (unit.task, unit.backend, unit.formation, unit.model) == (SCORE.name, "score", "mixed", None)
    assert f"  unit {unit.label()}\n" in text and unit.digest() not in text and SCORE.name not in text.split("unit ")[1]
    assert _sizing_line(text).count("(measured: 2 probes, 12 observations)") == 2 and "unmeasured" not in _sizing_line(
        text
    )


@pytest.mark.logic
def test_explain_says_why_a_unit_is_unresolved(tmp_path: Path) -> None:
    """Each unresolved unit names the state that is true: no hardware seen, an unbuilt env, or several classes seen."""
    profile = _profile(tmp_path)
    placed, bound = _plan(["x"], profile)
    source = MeasureSource([_Root()], _NoTable(), estimate=lambda unit, observed: _estimate())

    def read(host: HardwareInfo = HOST) -> str:
        return placed.explain(measures=source, hardware=host, values=bound)

    def unresolved(text: str) -> str:
        assert "  unit unresolved\n" in text and "(measured:" not in text
        return _sizing_line(text)

    assert "hardware not yet seen: no worker of class 'cpu' reported one" in unresolved(read())
    _seen(profile, SEEN)
    use_packaged(_use(_unbuilt))
    assert "the env of backend 'score' is not built" in unresolved(read())
    use_packaged(_use())
    assert "  unit unresolved" not in read() and "(measured:" in _sizing_line(read())
    _seen(profile, OTHER)
    third = replace(HOST, cpu_model="Third CPU")
    assert "maps to 2 seen hardware classes, none this host's; the first worker's hello decides" in unresolved(
        read(third)
    )
    assert "  unit unresolved" not in read()
    for stale in ("no measure source given", "no measure unit for the node"):
        assert stale not in read(third)


@pytest.mark.contract
def test_explain_counts_servable_and_computed_items(tmp_path: Path) -> None:
    """Per class, the jobs the store admits are servable and the others are computed; no result is read."""
    profile = _profile(tmp_path)
    store = ResultStore(tmp_path / "store")
    warmed, values = _plan(["x"], profile)
    executor = InProcess(store=store, profile=profile, run=RunConfig(), workers=WorkerSet(2))
    admission = executor.admit(warmed, values)
    handle = executor.submit(node_fragment(warmed, admission))
    executor.forget(handle)
    placed, bound = _plan(["x", "y", "z"], profile)
    text = placed.explain(store=store, values=bound, hardware=HOST)
    assert "class cpu: 1 servable, 2 computed" in text
    assert text.index("class cpu:") < text.index("waits on another run")
    use_packaged(_use(_unbuilt))
    unbuilt = placed.explain(store=store, values=bound, hardware=HOST)
    assert "class cpu: 0 servable, 0 computed, 3 undetermined (the env of backend 'score' is not built)" in unbuilt
    assert "items:" in placed.explain(hardware=HOST) and "servable" in placed.explain(hardware=HOST)


@pytest.mark.logic
def test_the_in_process_executor_sizes_from_the_unit_it_resolves(tmp_path: Path) -> None:
    """With no `units` given, the executor resolves the unit itself: measured once hardware is seen, else unmeasured."""
    profile = _profile(tmp_path)
    placed, bound = _plan(["x", "y"], profile)
    source = MeasureSource([_Root()], _NoTable(), estimate=lambda unit, observed: _estimate())
    executor = InProcess(store=ResultStore(tmp_path / "store"), profile=profile, workers=WorkerSet(1), measures=source)
    fragment = node_fragment(placed, executor.admit(placed, bound))
    jobs, placement = fragment_jobs(fragment, placed, bound)
    assert not executor.sizing(jobs, placement).measured.time
    _seen(profile, SEEN)
    assert executor.sizing(jobs, placement).measured.time
