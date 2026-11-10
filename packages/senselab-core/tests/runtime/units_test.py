"""The plan-time unit resolver: `explain` and the in-process executor read measures before any worker starts."""

import logging
import platform
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from pathlib import Path

import numpy
import pytest
import soundfile
from senselab_core.kernel.dataset import Record, ValueInput
from senselab_core.kernel.declare import BackendSpec, Env, Registry, ServerConfig, TaskParams, TaskSpec
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
from senselab_core.runtime.call import (
    EnvRead,
    EnvUnreadable,
    InputUnreadable,
    Packaged,
    RunConfig,
    bind_input,
    input_decoder,
    use_packaged,
)
from senselab_core.runtime.catalogue import Catalogue
from senselab_core.runtime.datasets import RecordSet
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.execution.in_process import InProcess, fragment_jobs, node_call, node_fragment
from senselab_core.runtime.execution.job import Job
from senselab_core.runtime.execution.plan_reads import count_items, plan_decoder, unit_of_node
from senselab_core.runtime.execution.run import InputChanged
from senselab_core.runtime.execution.workers import WorkerSet
from senselab_core.runtime.hardware import HardwareInfo
from senselab_core.runtime.keys import Placement
from senselab_core.runtime.licenses import license_record
from senselab_core.runtime.measures import MeasureSource, MeasureUnit, Provenance
from senselab_core.runtime.planner import Plan, plan
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.store import ResultStore
from senselab_core.runtime.units import UnitResolver, Unresolved, decoder_of, default_measures, measure_roots
from senselab_core.testing.calibration_samples import UNIT, observation
from senselab_core.testing.interpreter import source_tree_python

CLUSTER = (
    next(p for p in Path(__file__).resolve().parents if p.name == "packages")
    / "senselab-core/tests/data/profiles/cluster.yaml"
)

SCORE = TaskSpec(
    name="units_test.score",
    version="1",
    kind="transform",
    inputs={"text": "json"},
    outputs={"value": "sha256"},
    params=TaskParams,
)
AUDIO = SCORE.model_copy(update={"name": "units_test.audio", "inputs": {"text": "Audio"}})
TEXTY = SCORE.model_copy(update={"name": "units_test.texty", "inputs": {"text": "Text"}})
REGISTRY = Registry()
for _task in (SCORE, AUDIO, TEXTY):
    REGISTRY.add_task(_task)


def _backend(
    task: TaskSpec, name: str = "score", env: Env | None = None, server: ServerConfig | None = None
) -> BackendSpec:
    return BackendSpec(
        task=task.name,
        name=name,
        version="1",
        stack="dsp",
        env=env or Env.host(),
        devices=("cpu",),
        batch_invariant=True,
        worker="backends.sizes_backend_fixture",
        load_stub={},
        server=server,
    )


for _task in (SCORE, AUDIO, TEXTY):
    REGISTRY.add_backend(_backend(_task))
SERVED = _backend(SCORE, "served", Env.server("s"), ServerConfig(settings={}, startup_timeout_s=1.0))
CATALOGUE = Catalogue.load("format: catalogue@2.0\nentries: []\n")
HOST = HardwareInfo(
    cpu_count=4, memory_bytes=2**33, gpus=(), apple_silicon=False, platform="linux-x86_64", cpu_model="Test CPU"
)
SEEN = HardwareClass(device="cpu", name="Test CPU", cpus=4)
OTHER = HardwareClass(device="cpu", name="Another CPU", cpus=4)
PROVENANCE = Provenance(job=None, run="r1", senselab=None, profile="d" * 64, hardware=SEEN)


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


def _unreadable(backend: BackendSpec, profile: ExecutionProfile) -> None:  # noqa: ARG001 - an EnvRead
    raise EnvUnreadable("uv is not installed")


def _use(read: EnvRead = _handle) -> Packaged:
    records = {
        "pkg": license_record("MIT", source="lock", verified={"date": "2026-11-02", "how": "pep639"}, gated=False)
    }
    return Packaged(CATALOGUE, _handle, REGISTRY, lambda _: records, env_read=read)


@pytest.fixture(autouse=True)
def packaged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Plans read this module's registry, its host env built unless a test says so, and a cache root of its own."""
    monkeypatch.setenv("SENSELAB_CACHE", str(tmp_path / "cache"))
    previous = use_packaged(_use())
    yield
    use_packaged(previous)


def _profile(root: Path, **settings: object) -> ExecutionProfile:
    given = {"devices": ["cpu"], "calibration": {"measures_roots": [str(root / "measures")]}, **settings}
    return ExecutionProfile.load("laptop", given)


def _plan(
    rows: Sequence[object], profile: ExecutionProfile, task: TaskSpec = SCORE, kind: str = "json"
) -> tuple[Plan, dict[str, object]]:
    graph = Graph.model_validate(
        {
            "inputs": {"text": f"list[{kind}]"},
            "nodes": [{"id": "a", "task": task.name, "task_version": "1"}],
            "edges": [{"source": "input.text", "target": "a.text", "kind": "map"}],
        }
    )
    records = [Record(id=f"r{i}", inputs={"text": ValueInput(value=v)}) for i, v in enumerate(rows)]
    placed = plan(graph, RecordSet(records), config=RunConfig(), execution=profile)
    return placed, {bind_input("x", kind, v)[0]: bind_input("x", kind, v)[1] for v in rows}


class _Root:
    def observations(self, unit: MeasureUnit) -> Sequence[Observation]:
        return (observation(),)


def _estimate(unit_kind: str = "none", hi: float = 0.0) -> Estimate:
    return Estimate(
        status="measured",
        unit_kind=unit_kind,
        load_s=1.0,
        load_n=3,
        item_s=0.01,
        item_n=3,
        single=RequestFit(batch=1, r=0.5, u=0.0, n=6, spread=0.1, lo=0.0 if unit_kind == "none" else 1.0, hi=hi),
        batch=(StratumBatch(lo=0.0, hi=hi, batch=1),),
        host=MemoryFit(resident=2**20, m0=0.0, m1=1.0, n=6),
        source=EstimateSource(probe_ids=("p1", "p2"), observations=12),
    )


def _sizing_line(text: str) -> str:
    return next(line for line in text.splitlines() if line.startswith("  sizing:"))


def _seen(profile: ExecutionProfile, *classes: HardwareClass) -> None:
    roots = measure_roots(profile)
    for seen in classes:
        assert roots.note_seen(profile.resource_classes["cpu"], seen) is not None


def _source() -> MeasureSource:
    return MeasureSource([_Root()], estimate=lambda unit, observed: _estimate())


def _resolve(profile: ExecutionProfile, placed: Plan, hardware: HardwareInfo = HOST) -> MeasureUnit | Unresolved:
    registry = REGISTRY
    call = node_call(placed, "a", registry)
    placement = Placement(
        backend=registry.backends(call.task.name)[0],
        backend_params=registry.backends(call.task.name)[0].params.model_validate({}),
        device="cpu",
    )
    return UnitResolver(profile, hardware=hardware).unit(call, placement)


@pytest.mark.logic
def test_explain_is_measured_when_the_store_has_the_unit(tmp_path: Path) -> None:
    """With the env built and this host's class seen, the plan's unit is resolved and its figures are measured."""
    profile = _profile(tmp_path)
    _seen(profile, SEEN)
    placed, _ = _plan(["x"], profile)
    asked: list[MeasureUnit] = []

    def estimate(unit: MeasureUnit, observed: Sequence[Observation]) -> Estimate:  # noqa: ARG001 - an estimator
        asked.append(unit)
        return _estimate()

    text = placed.explain(measures=MeasureSource([_Root()], estimate=estimate), hardware=HOST)
    (unit,) = {u.digest(): u for u in asked}.values()
    assert unit.hardware == SEEN and unit.env.wheel_set == "2" * 64 and unit.env.env_lock == "1" * 64
    assert (unit.task, unit.backend, unit.formation, unit.model, unit.env.decoder) == (
        SCORE.name,
        "score",
        "mixed",
        None,
        None,
    )
    assert f"  unit {unit.label()}\n" in text and unit.digest() not in text
    line = _sizing_line(text)
    assert line.count("(measured: 2 probes, 12 observations)") == 2 and "unmeasured" not in line


@pytest.mark.logic
def test_a_measure_written_under_the_resolved_unit_is_read_by_the_default_source(tmp_path: Path) -> None:
    """Observations written to a real root under the resolver's unit come back measured through `default_measures`."""
    profile = _profile(tmp_path)
    _seen(profile, SEEN)
    placed, _ = _plan(["x"], profile)
    unit = _resolve(profile, placed)
    assert isinstance(unit, MeasureUnit)
    written = [
        observation(hardware=SEEN, unit_kind="none", units=0.0, padded_units=0.0, request_index=i) for i in range(6)
    ]
    assert measure_roots(profile).put_observations(unit, "p1", written, PROVENANCE) is not None
    assert default_measures(profile).get(unit) is not None
    line = _sizing_line(placed.explain(hardware=HOST))
    assert "wall 0.5 s (measured: 1 probes, 6 observations)" in line


@pytest.mark.logic
def test_a_unit_no_measures_root_holds_is_unmeasured_in_explain(tmp_path: Path) -> None:
    """A unit whose identity no root matches is read as no estimate, and `explain` says unmeasured, naming no table."""
    profile = _profile(tmp_path)
    _seen(profile, SEEN)
    placed, _ = _plan(["x"], profile)
    unit = _resolve(profile, placed)
    assert isinstance(unit, MeasureUnit)
    other = unit.model_copy(update={"env": unit.env.model_copy(update={"wheel_set": "f" * 64})})
    written = [
        observation(hardware=SEEN, unit_kind="none", units=0.0, padded_units=0.0, request_index=i) for i in range(6)
    ]
    assert measure_roots(profile).put_observations(other, "p1", written, PROVENANCE) is not None
    assert default_measures(profile).get(other) is not None and default_measures(profile).get(unit) is None
    line = _sizing_line(placed.explain(hardware=HOST))
    assert "unmeasured" in line and "measured:" not in line and "packaged" not in placed.explain(hardware=HOST)


@pytest.mark.logic
def test_every_unresolved_reason_is_true_of_its_state(tmp_path: Path) -> None:
    """Each fault branch of the resolver gives the reason that is true: hardware, env, substrate, decoder, server."""
    profile = _profile(tmp_path)
    placed, bound = _plan(["x"], profile)

    def reason(profile_: ExecutionProfile = profile, hardware: HardwareInfo = HOST) -> str:
        found = _resolve(profile_, placed, hardware)
        assert isinstance(found, Unresolved)
        return found.reason

    assert reason() == "hardware not yet seen on this host: no worker of class 'cpu' reported it"
    _seen(profile, OTHER)
    assert reason().startswith("hardware not yet seen on this host")
    assert reason(hardware=replace(HOST, cpu_model=None)).startswith("this host reports no hardware class for 'cpu'")
    _seen(profile, SEEN)
    assert isinstance(_resolve(profile, placed), MeasureUnit)
    use_packaged(_use(_unbuilt))
    assert reason() == "the env of backend 'score' is not built"
    use_packaged(_use(_unreadable))
    assert reason() == "the env of backend 'score' cannot be read: uv is not installed"
    use_packaged(_use())
    slurm = ExecutionProfile.load(CLUSTER)
    assert "chosen at submission" in reason(slurm)
    call = node_call(placed, "a", REGISTRY)
    served = Placement(backend=SERVED, backend_params=SERVED.params.model_validate({}), device="cpu")
    found = UnitResolver(profile, hardware=HOST).unit(call, served)
    assert isinstance(found, Unresolved) and "the attestation of backend 'served'" in found.reason
    assert bound


@pytest.mark.logic
def test_the_decoder_is_read_from_the_plans_sources_never_assumed(tmp_path: Path) -> None:
    """A path source of an audio port is the file decoder, a JSON value none, and both together are two units."""
    soundfile.write(tmp_path / "a.wav", numpy.zeros(1600, dtype=numpy.float32), 16000)
    profile = _profile(tmp_path)
    files, _ = _plan([str(tmp_path / "a.wav")], profile, AUDIO, "Audio")
    values, _ = _plan(["x"], profile)
    assert isinstance(plan_decoder(files, "a"), str)
    assert plan_decoder(values, "a") is None
    mixed = replace(
        files,
        live={"held": None},
        sources={"rows": [{"text": str(tmp_path / "a.wav")}, {"text": {"memory": "held"}}], "shared": {}},
    )
    assert isinstance(plan_decoder(mixed, "a"), Unresolved)
    _seen(profile, SEEN)
    asked: list[MeasureUnit] = []

    def estimate(unit: MeasureUnit, observed: Sequence[Observation]) -> Estimate:  # noqa: ARG001 - an estimator
        asked.append(unit)
        return _estimate()

    files.explain(measures=MeasureSource([_Root()], estimate=estimate), hardware=HOST)
    assert {u.env.decoder for u in asked} == {asked[0].env.decoder} and asked[0].env.decoder is not None


@pytest.mark.logic
def test_jobs_of_two_decoders_are_two_units(tmp_path: Path) -> None:
    """Jobs that read a file and jobs that do not cannot be sized as one unit."""
    soundfile.write(tmp_path / "a.wav", numpy.zeros(1600, dtype=numpy.float32), 16000)
    profile = _profile(tmp_path)
    placed, bound = _plan([str(tmp_path / "a.wav")], profile, AUDIO, "Audio")
    jobs, _ = fragment_jobs(node_fragment(placed, InProcess(profile=profile).admit(placed, bound)), placed, bound)
    bare = Job(jobs[0].ref, jobs[0].call, {})
    assert isinstance(decoder_of([jobs[0], bare]), Unresolved)
    assert isinstance(decoder_of([jobs[0]]), str) and decoder_of([bare]) is None


def _warm(tmp_path: Path, profile: ExecutionProfile, rows: Sequence[str]) -> None:
    """Run a plan over `rows` into the result store a run uses by default."""
    warmed, values = _plan(rows, profile)
    executor = InProcess(profile=profile, run=RunConfig(), workers=WorkerSet(2))
    executor.forget(executor.submit(node_fragment(warmed, executor.admit(warmed, values))))


@pytest.mark.logic
def test_explain_counts_servable_and_computed_items_by_default(tmp_path: Path) -> None:
    """With no store given, the plan counts against the store a run uses: what it serves and what it computes."""
    profile = _profile(tmp_path)
    _warm(tmp_path, profile, ["x"])
    placed, _ = _plan(["x", "y", "z"], profile)
    text = placed.explain(hardware=HOST)
    assert "class cpu: 1 servable, 2 computed" in text
    assert text.index("class cpu:") < text.index("waits on another run")
    other = ResultStore(tmp_path / "elsewhere")
    assert "class cpu: 0 servable, 3 computed" in placed.explain(store=other, hardware=HOST)
    use_packaged(_use(_unbuilt))
    unbuilt = placed.explain(hardware=HOST)
    assert "class cpu: 0 servable, 0 computed, 3 undetermined (the env of backend 'score' is not built)" in unbuilt


@pytest.mark.logic
def test_jobs_whose_context_is_unknowable_are_undetermined(tmp_path: Path) -> None:
    """On a slurm substrate the worker's platform is unknown, so no job is counted as served or computed."""
    profile = _profile(tmp_path)
    _warm(tmp_path, profile, ["x"])
    placed, bound = _plan(["x", "y"], profile)
    counted = count_items(
        placed,
        bound,
        store=ResultStore(tmp_path / "cache" / "results"),
        run=RunConfig(),
        profile=ExecutionProfile.load(CLUSTER),
    )
    assert counted["cpu"].servable == 0 and counted["cpu"].computed == 0 and counted["cpu"].undetermined == 2
    assert "chosen at submission" in counted["cpu"].why[0]


@pytest.mark.logic
def test_the_in_process_executor_sizes_from_the_unit_it_resolves(tmp_path: Path) -> None:
    """With no `units` given, the executor resolves the unit itself: measured once this host's class is seen."""
    profile = _profile(tmp_path)
    placed, bound = _plan(["x", "y"], profile)
    resolver = UnitResolver(profile, hardware=HOST)
    executor = InProcess(profile=profile, workers=WorkerSet(1), measures=_source(), units=resolver.unit)
    jobs, placement = fragment_jobs(node_fragment(placed, executor.admit(placed, bound)), placed, bound)
    assert not executor.sizing(jobs, placement).measured.time
    _seen(profile, SEEN)
    assert executor.sizing(jobs, placement).measured.time


@pytest.mark.logic
def test_the_extrapolation_log_starts_afresh_with_each_run(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """An executor reused for a second run logs a unit's extrapolation again: once per unit per run."""
    profile = _profile(tmp_path)
    placed, bound = _plan(["a long enough text"], profile, TEXTY, "Text")
    source = MeasureSource([_Root()], estimate=lambda unit, observed: _estimate("chars", hi=5.0))
    executor = InProcess(
        profile=profile, workers=WorkerSet(1), measures=source, units=lambda call, placement, decoder: UNIT
    )
    jobs, placement = fragment_jobs(node_fragment(placed, executor.admit(placed, bound)), placed, bound)
    with caplog.at_level(logging.INFO, logger="senselab_core.runtime.sizing"):
        executor.sizing(jobs, placement)
        executor.sizing(jobs, placement)
        assert [UNIT.label() in r.getMessage() for r in caplog.records] == [True]
        executor.bind_epoch(2, None)  # type: ignore[arg-type]
        executor.sizing(jobs, placement)
    assert len(caplog.records) == 2


ISOLATED = SCORE.model_copy(update={"name": "units_test.isolated", "outputs": {"label": "sha256"}})
REGISTRY.add_task(ISOLATED)
REGISTRY.add_backend(_backend(ISOLATED, env=Env.isolated("units-test")))


def _isolated_unbuilt(backend: BackendSpec, profile: ExecutionProfile) -> EnvHandle | None:  # an EnvRead
    return None if backend.env.kind == "isolated" else _handle(backend, profile)


def _isolated_built(backend: BackendSpec, profile: ExecutionProfile) -> EnvHandle:  # an EnvRead
    handle = _handle(backend, profile)
    return replace(handle, lock_sha="3" * 64) if backend.env.kind == "isolated" else handle


@pytest.mark.logic
def test_a_backend_name_and_version_shared_by_two_envs_reads_each_env(tmp_path: Path) -> None:
    """A host backend and an isolated one named alike in one plan each get their env's handle, unit and context."""
    profile = _profile(tmp_path)
    _seen(profile, SEEN)
    graph = Graph.model_validate(
        {
            "inputs": {"text": "list[json]"},
            "nodes": [
                {"id": "a", "task": SCORE.name, "task_version": "1"},
                {"id": "b", "task": ISOLATED.name, "task_version": "1"},
            ],
            "edges": [
                {"source": "input.text", "target": "a.text", "kind": "map"},
                {"source": "input.text", "target": "b.text", "kind": "map"},
            ],
        }
    )
    placed = plan(graph, RecordSet([Record(id="r0", inputs={"text": ValueInput(value="x")})]), execution=profile)
    assert [(p.backend, p.backend_version) for p in placed.placement.nodes.values()] == [("score", "1")] * 2
    use_packaged(_use(_isolated_built))
    resolve = unit_of_node(placed, UnitResolver(profile, hardware=HOST))
    a, b = resolve("a"), resolve("b")
    assert isinstance(a, MeasureUnit) and isinstance(b, MeasureUnit)
    assert (a.env.env_lock, b.env.env_lock) == ("1" * 64, "3" * 64)
    use_packaged(_use(_isolated_unbuilt))
    text = placed.explain(hardware=HOST)
    assert "class cpu: 0 servable, 1 computed, 1 undetermined (the env of backend 'score' is not built)" in text


def _memory_plan(tmp_path: Path) -> Plan:
    """A plan read back from a run directory in another process: its one input held in memory, which is not here."""
    placed, _ = _plan(["x"], _profile(tmp_path))
    (row,) = placed.manifest.rows
    return replace(placed, sources={"rows": [{"text": {"memory": row.inputs["text"]}}], "shared": {}}, live={})


@pytest.mark.logic
def test_a_plan_whose_inputs_cannot_be_bound_explains_with_counts_undetermined(tmp_path: Path) -> None:
    """No recorded sources, or inputs held by another process: the plan renders, its counts undetermined, why given."""
    _seen(_profile(tmp_path), SEEN)
    held = _memory_plan(tmp_path)
    for placed, why in (
        (replace(held, sources={}), "the plan records no input sources"),
        (held, "were held in memory by the process that planned the run"),
    ):
        text = placed.explain(hardware=HOST)
        assert text.startswith(f"plan {placed.hash}") and "node a (units_test.score)" in text
        (line,) = [x for x in text.splitlines() if x.startswith("class cpu:")]
        assert line.startswith("class cpu: servable and computed undetermined (") and why in line
        found = plan_decoder(placed, "a")
        assert isinstance(found, Unresolved) and "unit unresolved" in text
    assert plan_decoder(replace(held, sources={}), "a") == Unresolved("the plan records no input sources")


@pytest.mark.logic
def test_lengths_of_unbound_inputs_are_unknown_with_the_reason(tmp_path: Path) -> None:
    """A node whose unit is measured but whose inputs cannot be bound says its lengths are unknown, and why."""
    profile = _profile(tmp_path)
    _seen(profile, SEEN)
    held = _memory_plan(tmp_path)
    unit = _resolve(profile, held)
    assert isinstance(unit, MeasureUnit)
    line = _sizing_line(held.explain(measures=_source(), units=lambda node: unit, hardware=HOST))
    assert "unmeasured (lengths unknown: inputs ['text'] of records ['r0'] were held in memory" in line


@pytest.mark.logic
def test_a_path_source_is_read_by_its_port_types_reader(tmp_path: Path) -> None:
    """The decoder of a path source is the one the reader table gives its port type."""
    soundfile.write(tmp_path / "a.wav", numpy.zeros(1600, dtype=numpy.float32), 16000)
    files, _ = _plan([str(tmp_path / "a.wav")], _profile(tmp_path), AUDIO, "Audio")
    assert plan_decoder(files, "a") == input_decoder("Audio") is not None
    assert input_decoder("json") is None


Content = bytes | Callable[[bytes], bytes]
"""What replaces a planned sound file: bytes, or a function of its bytes."""


def _unreadable_plan(tmp_path: Path, content: Content) -> tuple[Plan, Path]:
    """A plan over a sound file whose content is then replaced by `content`, which its reader cannot decode."""
    path = tmp_path / "a.wav"
    soundfile.write(path, numpy.zeros(1600, dtype=numpy.float32), 16000)
    placed, _ = _plan([str(path)], _profile(tmp_path), AUDIO, "Audio")
    path.write_bytes(content(path.read_bytes()) if callable(content) else content)
    return placed, path


UNREADABLE = pytest.mark.parametrize(
    "content",
    [lambda wav: wav[:20], b"RIFF" + bytes(56)],
    ids=["truncated", "garbage"],
)


@pytest.mark.logic
@UNREADABLE
def test_a_plan_over_an_undecodable_file_explains_with_counts_undetermined(tmp_path: Path, content: Content) -> None:
    """A recorded file truncated or not audio: the plan renders, its counts undetermined with the binding reason."""
    placed, path = _unreadable_plan(tmp_path, content)
    with pytest.raises(InputUnreadable, match="'text'"):
        bind_input("text", "Audio", str(path))
    text = placed.explain(hardware=HOST)
    (line,) = [x for x in text.splitlines() if x.startswith("class cpu:")]
    assert line.startswith("class cpu: servable and computed undetermined (inputs ['text'] of records ['r0']")


@pytest.mark.logic
@UNREADABLE
def test_a_run_over_an_undecodable_file_raises_input_unreadable(tmp_path: Path, content: Content) -> None:
    """Planning over an undecodable file raises `InputUnreadable`; running a plan whose file became one refuses it."""
    pytest.importorskip("pydra")
    placed, path = _unreadable_plan(tmp_path, content)
    with pytest.raises(InputUnreadable):
        _plan([str(path)], _profile(tmp_path), AUDIO, "Audio")
    with pytest.raises(InputChanged) as raised:
        placed.run(store=ResultStore(tmp_path / "store"))
    assert isinstance(raised.value.__cause__, InputUnreadable) and raised.value.changed == (("r0", "text"),)
