"""`resolve`: a graph or one call over records to a logical plan, with no execution input (fake stack)."""

import dataclasses
import inspect
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import senselab_core.runtime.fake_backend as fake
from pydantic import ValidationError
from senselab_core.kernel.declare import BackendSpec, Env, Registry, TaskParams, TaskSpec
from senselab_core.kernel.graph import Graph
from senselab_core.kernel.identity import ReusePolicy
from senselab_core.kernel.license import License
from senselab_core.kernel.plan import plan_hash, slice_identity
from senselab_core.runtime import planner
from senselab_core.runtime.calibration import (
    Estimate,
    EstimateSource,
    LengthProfile,
    MemoryFit,
    Observation,
    RequestFit,
    StratumBatch,
)
from senselab_core.runtime.call import Packaged, RunConfig, TaskCall, use_packaged
from senselab_core.runtime.call import packaged as packaged_now
from senselab_core.runtime.catalogue import Catalogue
from senselab_core.runtime.datasets import from_records
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.execution.in_process import InProcess
from senselab_core.runtime.execution.workers import WorkerSet
from senselab_core.runtime.hardware import GpuInfo, HardwareInfo
from senselab_core.runtime.keys import MissingProfileSlice
from senselab_core.runtime.licenses import LicensePolicy, LicenseRefused, license_record
from senselab_core.runtime.measures import MeasureSource, MeasureUnit
from senselab_core.runtime.placement import BackendParamsMismatch
from senselab_core.runtime.planner import CALL_NODE, InvalidGraph, RecordBinding, UnmappedOutput, plan, resolve
from senselab_core.runtime.profiles import ExecutionProfile, NoDevice
from senselab_core.runtime.revision import PROCESS_RESOLVER, Resolver
from senselab_core.runtime.store import ResultStore
from senselab_core.testing.calibration_samples import UNIT, observation

TOY = "toyorg/toy"
PIN = "a" * 40
MOVED = "c" * 40


class _Params(TaskParams):
    """Params of the planner test task."""

    w: int = 1


SCORE = TaskSpec(
    name="planner_test.score",
    version="1",
    kind="transform",
    inputs={"text": "json"},
    outputs={"value": "json"},
    params=_Params,
)
READS = SCORE.model_copy(
    update={
        "name": "planner_test.reads",
        "outputs": {"score": "json"},
        "profile_slices": ("dataset.lexicon", "run.aggregator"),
    }
)


def _backend(name: str, task: TaskSpec, devices: tuple[str, ...] = ("cpu",), env: Env | None = None) -> BackendSpec:
    return BackendSpec(
        task=task.name,
        name=name,
        version="1",
        stack=name,
        env=env or Env.host(),
        devices=devices,
        batch_invariant=True,
        worker=fake.WORKER_MODULE,
        load_stub={},
    )


PAIR = SCORE.model_copy(
    update={"name": "planner_test.pair", "inputs": {"a": "json", "b": "json"}, "outputs": {"pair": "json"}}
)

TAIL = SCORE.model_copy(update={"name": "planner_test.tail", "outputs": {"tail": "json"}})
SPLIT = SCORE.model_copy(update={"name": "planner_test.split", "outputs": {"items": "list[json]"}})
GPU_ONLY = SCORE.model_copy(update={"name": "planner_test.gpu", "outputs": {"on_gpu": "json"}})
ISOLATED = SCORE.model_copy(update={"name": "planner_test.isolated", "outputs": {"boxed": "json"}})

REGISTRY = Registry()
for _task in (SCORE, READS, PAIR, TAIL, GPU_ONLY, ISOLATED, SPLIT):
    REGISTRY.add_task(_task)
REGISTRY.add_backend(_backend("alpha", SCORE))
REGISTRY.add_backend(_backend("beta", SCORE))
REGISTRY.add_backend(_backend("alpha", READS))
REGISTRY.add_backend(_backend("gamma", TAIL))
REGISTRY.add_backend(_backend("cuda-only", GPU_ONLY, devices=("cuda",)))
REGISTRY.add_backend(_backend("boxed", ISOLATED, env=Env.isolated("boxed")))

CATALOGUE = Catalogue.load(
    f"""
format: catalogue@2.0
entries:
- id: {TOY}
  revision: {PIN}
  tasks: [{SCORE.name}]
  backends: [alpha]
  stacks: [alpha]
  default_for: [{SCORE.name}]
  license:
    expression: MIT
    source: toy card
    verified: {{date: 2026-11-01, how: hf-card}}
    gated: false
"""
)


def _no_env(backend: BackendSpec, profile: ExecutionProfile) -> EnvHandle:  # noqa: ARG001 - an EnvOpener
    raise AssertionError("resolving a plan opens no env")


@pytest.fixture(autouse=True)
def packaged() -> Iterator[None]:
    """`resolve` reads the toy registry and catalogue."""
    previous = use_packaged(Packaged(CATALOGUE, _no_env, REGISTRY))
    yield
    use_packaged(previous)


def _hub(answers: dict[str, str], seen: list[tuple[str, str]]) -> Callable[[str, str, str | None], str]:
    def hub(repo: str, ref: str, token: str | None) -> str:  # noqa: ARG001 - a hub lookup
        seen.append((repo, ref))
        return answers[ref]

    return hub


def _resolver(tmp_path: Path, answers: dict[str, str], seen: list[tuple[str, str]] | None = None) -> Resolver:
    return Resolver(cache_dir=tmp_path / "hf-hub", hub=_hub(answers, [] if seen is None else seen), cache_root=tmp_path)


def _call(task: TaskSpec = SCORE, **fields: Any) -> TaskCall:  # noqa: ANN401 - call fields
    return TaskCall.model_validate({"task": task.name, "inputs": {}, **fields}, context={"registry": REGISTRY})


def _records(*texts: str) -> Any:  # noqa: ANN401 - a Dataset
    return from_records({"id": f"r{i}", "inputs": {"text": t}} for i, t in enumerate(texts))


PROFILE = {"lexicon": ["a", "b"], "cohort": "x"}
SETTINGS = {"aggregator": "mean", "other": 1}


@pytest.mark.logic
def test_resolve_takes_no_execution_input() -> None:
    """`resolve` has no execution profile, hardware or placement parameter."""
    names = set(inspect.signature(resolve).parameters)
    assert names == {"target", "dataset", "config", "dataset_profile", "bind", "inputs", "resolver", "admit"}
    assert not names & {"execution", "profile", "hardware", "placement", "device", "backend"}


@pytest.mark.logic
def test_call_over_records_is_a_one_node_graph_mapped_over_them() -> None:
    """A call over a dataset is one node fed by a `map` edge; the manifest has one row per record."""
    plan, manifest = resolve(_call(params={"w": 2}), _records("x", "y"), config=RunConfig())
    graph = plan.graph
    assert [n.id for n in graph.nodes] == [CALL_NODE] and dict(graph.nodes[0].params) == {"w": 2}
    assert dict(graph.inputs) == {"text": "list[json]"}
    assert [(e.source, e.target, e.kind) for e in graph.edges] == [("input.text", "call.text", "map")]
    assert [r.id for r in manifest.rows] == ["r0", "r1"] and plan.manifest_digest == manifest.digest()
    model = plan.models[CALL_NODE]
    assert model is not None and model.identity == PIN
    assert plan.reuse == ReusePolicy.EQUIVALENT.value


@pytest.mark.logic
def test_single_call_is_its_own_record() -> None:
    """Without a dataset the call's inputs are one record, and a call over a dataset gives none itself."""
    plan, manifest = resolve(_call(inputs={"text": "x"}), None, config=RunConfig())
    over, records = resolve(_call(), _records("x"), config=RunConfig())
    assert [r.id for r in manifest.rows] == ["call"] and manifest.rows[0].inputs == records.rows[0].inputs
    assert plan.graph == over.graph
    with pytest.raises(ValueError, match="takes its inputs from the records"):
        resolve(_call(inputs={"text": "x"}), _records("x"), config=RunConfig())


@pytest.mark.logic
def test_ref_resolved_to_commit_in_plan(tmp_path: Path) -> None:
    """A ref is pinned to the commit it names now; the plan names that commit, so a moved ref is a new plan."""
    seen: list[tuple[str, str]] = []
    call = _call(model="otherorg/m@main")
    plan, _ = resolve(call, _records("x"), config=RunConfig(), resolver=_resolver(tmp_path / "1", {"main": PIN}, seen))
    model = plan.models[CALL_NODE]
    assert model is not None and model.identity == PIN and model.ref.ref == PIN and seen == [("otherorg/m", "main")]
    again, _ = resolve(call, _records("x"), config=RunConfig(), resolver=_resolver(tmp_path / "3", {"main": PIN}))
    pinned, _ = resolve(_call(model=f"otherorg/m@{PIN}"), _records("x"), config=RunConfig())
    assert plan_hash(again) == plan_hash(pinned) == plan_hash(plan)
    moved, _ = resolve(call, _records("x"), config=RunConfig(), resolver=_resolver(tmp_path / "2", {"main": MOVED}))
    assert plan_hash(moved) != plan_hash(plan)


@pytest.mark.logic
def test_catalogue_pin_needs_no_lookup(tmp_path: Path) -> None:
    """A node naming no model gets the task's catalogue default at its pinned revision, without a hub lookup."""
    seen: list[tuple[str, str]] = []
    plan, _ = resolve(_call(), _records("x"), config=RunConfig(), resolver=_resolver(tmp_path, {}, seen))
    model = plan.models[CALL_NODE]
    assert model is not None and str(model.ref) == f"{TOY}@{PIN}" and model.license is not None and seen == []


@pytest.mark.logic
def test_declared_slice_keyed_by_identity() -> None:
    """Each declared slice is the identity of its value, cut from the dataset profile or the run settings."""
    config = RunConfig(settings=SETTINGS)
    plan, _ = resolve(_call(READS), _records("x"), config=config, dataset_profile=PROFILE)
    assert dict(plan.profile_slices[CALL_NODE]) == {
        "dataset.lexicon": slice_identity(["a", "b"]),
        "run.aggregator": slice_identity("mean"),
    }
    changed, _ = resolve(_call(READS), _records("x"), config=config, dataset_profile={**PROFILE, "lexicon": ["a"]})
    assert plan_hash(changed) != plan_hash(plan)
    with pytest.raises(MissingProfileSlice, match="dataset.lexicon"):
        resolve(_call(READS), _records("x"), config=config)


@pytest.mark.logic
@pytest.mark.parametrize(
    ("profile", "settings"),
    [({**PROFILE, "cohort": "y"}, SETTINGS), ({**PROFILE, "new": 1}, SETTINGS), (PROFILE, {**SETTINGS, "other": 2})],
    ids=["dataset-changed", "dataset-added", "run-changed"],
)
def test_undeclared_profile_key_perturbation_leaves_plan(profile: dict[str, Any], settings: dict[str, Any]) -> None:
    """A profile key no node declares never reaches the plan."""
    base, _ = resolve(_call(READS), _records("x"), config=RunConfig(settings=SETTINGS), dataset_profile=PROFILE)
    other, _ = resolve(_call(READS), _records("x"), config=RunConfig(settings=settings), dataset_profile=profile)
    assert other == base and plan_hash(other) == plan_hash(base)


@pytest.mark.logic
def test_call_backend_params_are_held_per_backend_as_given() -> None:
    """A call's backend params are keyed by backend name and reach the node unchanged; a flat entry is refused."""
    plan, _ = resolve(_call(backend_params={"beta": {"beam": 2}}), _records("x"), config=RunConfig())
    assert {k: dict(v) for k, v in plan.graph.nodes[0].backend_params.items()} == {"beta": {"beam": 2}}
    with pytest.raises(ValidationError):
        _call(backend_params={"beam": 2})


@pytest.mark.logic
@pytest.mark.parametrize("given", [{}, {"alpha": {"beam": 2}}], ids=["none", "alpha"])
def test_installed_backends_never_reach_the_plan(given: dict[str, Any]) -> None:
    """Installing a backend for the task, or removing one, leaves the plan hash unchanged."""
    base, _ = resolve(_call(backend_params=given), _records("x"), config=RunConfig())
    for change in ("add", "remove"):
        registry = Registry()
        registry.restore(REGISTRY.snapshot())
        if change == "add":
            registry.add_backend(_backend("gamma", SCORE))
            other = registry
        else:
            other = Registry()
            for task in (SCORE, READS, PAIR):
                other.add_task(task)
            other.add_backend(_backend("alpha", SCORE))
        previous = use_packaged(Packaged(CATALOGUE, _no_env, other))
        try:
            changed, _ = resolve(_call(backend_params=given), _records("x"), config=RunConfig())
        finally:
            use_packaged(previous)
        assert plan_hash(changed) == plan_hash(base), change


@pytest.mark.logic
def test_node_params_are_completed_with_defaults() -> None:
    """`{}` and the explicit default hash the same, and a call hashes as its hand-written graph."""
    call_plan, _ = resolve(_call(), _records("x"), config=RunConfig())

    def written(params: dict[str, Any]) -> Graph:
        return Graph.model_validate(
            {
                "inputs": {"text": "list[json]"},
                "nodes": [{"id": CALL_NODE, "task": SCORE.name, "task_version": "1", "params": params}],
                "edges": [{"source": "input.text", "target": "call.text", "kind": "map"}],
                "outputs": {"value": "call.value"},
            }
        )

    empty, _ = resolve(written({}), _records("x"), config=RunConfig())
    explicit, _ = resolve(written({"w": 1}), _records("x"), config=RunConfig())
    assert dict(empty.graph.nodes[0].params) == {"w": 1}
    assert plan_hash(empty) == plan_hash(explicit) == plan_hash(call_plan)
    other, _ = resolve(written({"w": 2}), _records("x"), config=RunConfig())
    assert plan_hash(other) != plan_hash(empty)


@pytest.mark.logic
def test_a_graph_over_a_dataset_needs_a_record_list_and_its_broadcast_values() -> None:
    """Over a dataset a graph maps at least one record list; each other input is broadcast, given once in `inputs`."""
    with pytest.raises(RecordBinding) as raised:
        resolve(_call(PAIR), _records("x"), config=RunConfig())
    assert raised.value.ports == ("b",)
    plain = Graph.model_validate(
        {
            "inputs": {"text": "json"},
            "nodes": [{"id": "n", "task": SCORE.name, "task_version": "1"}],
            "edges": [{"source": "input.text", "target": "n.text"}],
        }
    )
    with pytest.raises(RecordBinding, match="text"):
        resolve(plain, _records("x"), config=RunConfig())
    with pytest.raises(RecordBinding, match="b"):
        resolve(_call(PAIR), _records("x"), config=RunConfig(), bind={"b": "text"}, inputs={"b": 2})
    with pytest.raises(ValueError, match="no dataset"):
        resolve(_call(PAIR, inputs={"a": 1, "b": 2}), None, config=RunConfig(), inputs={"b": 2})
    single, manifest = resolve(_call(PAIR, inputs={"a": 1, "b": 2}), None, config=RunConfig())
    assert set(manifest.rows[0].inputs) == {"a", "b"} and single.manifest_digest == manifest.digest()


@pytest.mark.logic
def test_record_bound_inputs_are_zipped_and_the_rest_broadcast() -> None:
    """Two record lists are read from each record, row by row; a broadcast input has one identity in every row."""
    graph = Graph.model_validate(
        {
            "inputs": {"first": "list[json]", "second": "list[json]", "both": "json"},
            "nodes": [
                {"id": "f", "task": SCORE.name, "task_version": "1"},
                {"id": "s", "task": SCORE.name, "task_version": "1"},
                {"id": "p", "task": PAIR.name, "task_version": "1"},
            ],
            "edges": [
                {"source": "input.first", "target": "f.text", "kind": "map"},
                {"source": "input.second", "target": "s.text", "kind": "map"},
                {"source": "f.value", "target": "p.a"},
                {"source": "input.both", "target": "p.b"},
            ],
        }
    )
    records = from_records({"id": f"r{i}", "inputs": {"first": f"one-{i}", "second": f"two-{i}"}} for i in range(3))
    logical, manifest = resolve(graph, records, config=RunConfig(), inputs={"both": 7})
    rows = [dict(r.inputs) for r in manifest.rows]
    assert len({r["both"] for r in rows}) == 1 and len({(r["first"], r["second"]) for r in rows}) == 3
    _, other = resolve(graph, records, config=RunConfig(), inputs={"both": 8})
    assert other.digest() != manifest.digest() and logical.manifest_digest == manifest.digest()
    with pytest.raises(RecordBinding, match="both"):
        resolve(graph, records, config=RunConfig())


@pytest.mark.logic
def test_two_record_lists_into_one_node_are_bound_per_row() -> None:
    """One node mapped over two record lists reads both from each record's row."""
    graph = Graph.model_validate(
        {
            "inputs": {"first": "list[json]", "second": "list[json]"},
            "nodes": [{"id": "p", "task": PAIR.name, "task_version": "1"}],
            "edges": [
                {"source": "input.first", "target": "p.a", "kind": "map"},
                {"source": "input.second", "target": "p.b", "kind": "map"},
            ],
        }
    )
    records = from_records({"id": f"r{i}", "inputs": {"first": f"f{i}", "second": f"s{i}"}} for i in range(2))
    _, manifest = resolve(graph, records, config=RunConfig())
    assert [set(r.inputs) for r in manifest.rows] == [{"first", "second"}] * 2
    assert len({r.inputs["first"] for r in manifest.rows} | {r.inputs["second"] for r in manifest.rows}) == 4


@pytest.mark.logic
def test_refused_source_raises_before_any_lookup(tmp_path: Path) -> None:
    """`admit` sees each hub model before it is resolved; a refusal raises and the hub is never asked."""
    seen: list[tuple[str, str]] = []

    def admit(model_id: str) -> None:
        raise PermissionError(f"{model_id} is not an allowed source")

    with pytest.raises(PermissionError, match="otherorg/m"):
        resolve(
            _call(model="otherorg/m@main"),
            _records("x"),
            config=RunConfig(),
            resolver=_resolver(tmp_path, {"main": PIN}, seen),
            admit=admit,
        )
    assert seen == []
    plan, _ = resolve(_call(), _records("x"), config=RunConfig(), admit=admit)
    assert plan.models[CALL_NODE] is not None


@pytest.mark.logic
def test_execution_settings_leave_the_plan(tmp_path: Path) -> None:
    """Verify sampling, thresholds and the resolver's network policy never reach the plan."""
    base, _ = resolve(_call(), _records("x"), config=RunConfig())
    offline = Resolver(cache_dir=tmp_path, cache_root=tmp_path, network="offline")
    for config, resolver in ((RunConfig(verify=0.5), PROCESS_RESOLVER), (RunConfig(), offline)):
        other, _ = resolve(_call(), _records("x"), config=config, resolver=resolver)
        assert plan_hash(other) == plan_hash(base)


@pytest.mark.logic
def test_graph_with_issues_is_refused() -> None:
    """A graph `validate` finds issues in does not plan, and the issues travel with the error."""
    graph = Graph.model_validate({"nodes": [{"id": "n", "task": SCORE.name, "task_version": "1"}]})
    with pytest.raises(InvalidGraph) as raised:
        resolve(graph, None, config=RunConfig())
    assert [i.code for i in raised.value.issues] == ["unbound_input"]


@pytest.mark.logic
def test_reuse_policy_and_bind_reach_the_plan() -> None:
    """The run config's reuse policy and the binding are recorded; neither is execution."""
    rows = from_records([{"id": "r", "inputs": {"utterance": "x"}}])
    plan, _ = resolve(_call(), rows, config=RunConfig(reuse=ReusePolicy.EXACT), bind={"text": "utterance"})
    assert plan.reuse == ReusePolicy.EXACT.value and dict(plan.bind) == {"text": "utterance"}


@pytest.mark.logic
def test_an_automatic_bind_is_the_plan_of_the_explicit_one() -> None:
    """A port binding to the records' single input is recorded in `bind`; a port binding by name records nothing."""
    rows = from_records([{"id": "r", "inputs": {"utterance": "x"}}])
    auto, _ = resolve(_call(), rows, config=RunConfig())
    explicit, _ = resolve(_call(), rows, config=RunConfig(), bind={"text": "utterance"})
    assert dict(auto.bind) == {"text": "utterance"} and plan_hash(auto) == plan_hash(explicit)
    by_name, _ = resolve(_call(), _records("x"), config=RunConfig())
    assert dict(by_name.bind) == {}


NO_GPU = HardwareInfo(cpu_count=2, memory_bytes=2**33, gpus=(), apple_silicon=False, platform="linux-x86_64")
GPU = HardwareInfo(
    cpu_count=2,
    memory_bytes=2**33,
    gpus=(GpuInfo(name="NVIDIA H100 80GB HBM3", memory_bytes=2**36, driver="1"),),
    apple_silicon=False,
    platform="linux-x86_64",
)


def _plan(*args: Any, hardware: HardwareInfo = NO_GPU, **fields: Any) -> planner.Plan:  # noqa: ANN401
    """`plan` on a host with `hardware`, a CPU-only host by default."""
    return plan(*args, hardware=hardware, **fields)


def _license(expression: str) -> License:
    return license_record(expression, source="lock", verified={"date": "2026-11-04", "how": "pep639"}, gated=False)


def _use(**fields: Any) -> None:  # noqa: ANN401 - Packaged fields
    """Replace fields of the packaged state for one test; the autouse fixture restores it."""
    use_packaged(dataclasses.replace(packaged_now(), **fields))


def _chain() -> Graph:
    """`a` scores each record and `b` (task `tail`, no model) scores `a`'s value."""
    return Graph.model_validate(
        {
            "inputs": {"text": "list[json]"},
            "nodes": [
                {"id": "a", "task": SCORE.name, "task_version": "1"},
                {"id": "b", "task": TAIL.name, "task_version": "1"},
            ],
            "edges": [
                {"source": "input.text", "target": "a.text", "kind": "map"},
                {"source": "a.value", "target": "b.text"},
            ],
            "outputs": {"tail": "b.tail"},
        }
    )


ORCD = {"orcd": ExecutionProfile.load("orcd"), "laptop": ExecutionProfile.load("laptop")}


@pytest.mark.logic
def test_plan_hash_ignores_execution_fields() -> None:
    """Under the laptop and orcd profiles, on a CPU host and a GPU host, one plan hash: the logical plan's."""
    logical, _ = resolve(_call(), _records("x"), config=RunConfig())
    seen = set()
    for hardware in (NO_GPU, GPU):
        for profile in ORCD.values():
            placed = _plan(_call(), _records("x"), execution=profile, hardware=hardware)
            seen.add((placed.hash, placed.placement.profile_digest))
            assert placed.hash == plan_hash(placed.logical) == plan_hash(logical)
    assert len({h for h, _ in seen}) == 1 and len({d for _, d in seen}) == 2


@pytest.mark.logic
def test_gpu_only_node_without_gpu_class_fails_naming_node() -> None:
    """A CUDA-only node fails planning, naming the node, where no GPU class runs it; the cluster's H100 class does."""
    with pytest.raises(NoDevice, match="node 'call'"):
        _plan(_call(GPU_ONLY), _records("x"))
    without = ExecutionProfile.load("orcd", {"devices": ["cpu"]})
    with pytest.raises(NoDevice, match="node 'call'.*resource classes"):
        _plan(_call(GPU_ONLY), _records("x"), execution=without)
    placed = _plan(_call(GPU_ONLY), _records("x"), execution=ORCD["orcd"]).placement.nodes[CALL_NODE]
    assert (placed.device, placed.resource_class, placed.partition) == ("cuda", "gpu:h100", "${GPU_PARTITION}")
    assert _plan(_call(GPU_ONLY), _records("x"), hardware=GPU).placement.nodes[CALL_NODE].resource_class == "gpu"


@pytest.mark.logic
def test_explain_lists_every_placement_field() -> None:
    """`explain` names the plan hash and env builds, and per node every placement field and its licenses."""
    placed = _plan(_call(backend_params={"beta": {}}), _records("x"))
    text = placed.explain()
    for field in (placed.hash, "envs to build: none"):
        assert field in text, field
    node = next(line for line in text.splitlines() if line.startswith("node call"))
    for field in (
        "node call",
        "backend 'alpha' 1",
        "stack 'alpha'",
        "env host",
        "device cpu",
        "batch 1",
        "resource class cpu",
        "partition none",
        f"model {TOY}@{PIN} MIT",
        "env of backend planner_test.score/alpha no record",
        "first declared",
    ):
        assert field in node, field


@pytest.mark.logic
def test_env_builds_announced() -> None:
    """An isolated env not yet built is announced, and a built one is not; a host env never is."""
    _use(env_built=lambda backend, profile: False)
    announced = _plan(_call(ISOLATED), _records("x"))
    assert announced.placement.env_builds == ("isolated:boxed",)
    assert "envs to build: isolated:boxed" in announced.explain()
    assert _plan(_call(), _records("x")).placement.env_builds == ()
    _use(env_built=lambda backend, profile: True)
    assert _plan(_call(ISOLATED), _records("x")).placement.env_builds == ()


@pytest.mark.logic
def test_taint_unions_upstream_licenses() -> None:
    """A node's license set holds its own env and model and every upstream node's; a policy refuses the taint."""
    by_backend = {"alpha": _license("CC-BY-NC-4.0"), "gamma": _license("MIT")}
    _use(lock_records=lambda backend: {"pkg": by_backend[backend.name]})
    sets = _plan(_chain(), _records("x")).placement.licenses
    own_a = {f"env of backend {SCORE.name}/alpha": "CC-BY-NC-4.0", f"model {TOY}@{PIN}": "MIT"}
    assert {k: v.expression for k, v in sets["a"].records.items() if v} == own_a
    assert {k: v.expression for k, v in sets["b"].records.items() if v} == {
        **own_a,
        f"env of backend {TAIL.name}/gamma": "MIT",
    }
    permissive = RunConfig(licenses=LicensePolicy(refuse=frozenset({"CC-BY-NC-4.0"})))
    with pytest.raises(LicenseRefused, match="node 'a'.*CC-BY-NC-4.0"):
        _plan(_chain(), _records("x"), config=permissive)
    _plan(_chain(), _records("x"), config=RunConfig(licenses=LicensePolicy(allow=frozenset({"MIT", "CC-BY-NC-4.0"}))))


@pytest.mark.logic
def test_backend_params_mismatch_at_plan_time() -> None:
    """The placed backend's entry is validated at plan time; another backend's is not; an undeclared name is refused."""
    with pytest.raises(BackendParamsMismatch, match="node 'call'.*'alpha'.*beam") as raised:
        _plan(_call(backend_params={"alpha": {"beam": 2}}), _records("x"))
    assert (raised.value.node, raised.value.backend) == (CALL_NODE, "alpha")
    placed = _plan(_call(backend_params={"beta": {"beam": 2}}), _records("x"))
    assert dict(placed.placement.nodes[CALL_NODE].backend_params) == {}
    with pytest.raises(BackendParamsMismatch, match="declares no backend 'delta'") as raised:
        _plan(_call(backend_params={"delta": {}}), _records("x"))
    assert raised.value.backend == "delta"


@pytest.mark.logic
def test_a_map_over_a_node_output_is_refused_at_plan_time() -> None:
    """A valid graph mapping a node's list output does not plan: one item of a stored list output has no identity."""
    graph = Graph.model_validate(
        {
            "inputs": {"text": "list[json]", "seed": "json"},
            "nodes": [
                {"id": "s", "task": SPLIT.name, "task_version": "1"},
                {"id": "n", "task": SCORE.name, "task_version": "1"},
                {"id": "m", "task": SCORE.name, "task_version": "1"},
            ],
            "edges": [
                {"source": "input.seed", "target": "s.text"},
                {"source": "s.items", "target": "n.text", "kind": "map"},
                {"source": "input.text", "target": "m.text", "kind": "map"},
            ],
        }
    )
    with pytest.raises(UnmappedOutput, match="has no identity yet") as raised:
        resolve(graph, _records("x"), config=RunConfig(), bind={"text": "text"}, inputs={"seed": 1})
    assert (raised.value.node, raised.value.source) == ("n", "s.items")


_GB = 10**9


def _measured(spread: float | None = 0.1, resident: int = 10 * _GB) -> Estimate:
    return Estimate(
        status="measured",
        unit_kind="audio_s",
        load_s=2.0,
        load_n=3,
        item_s=0.05,
        item_n=3,
        single=RequestFit(batch=1, r=0.5, u=0.05, n=6, spread=spread, lo=2.0, hi=1000.0),
        batch=(StratumBatch(lo=2.0, hi=1000.0, batch=1),),
        device=MemoryFit(resident=resident, m0=0.0, m1=1e7, n=6),
        source=EstimateSource(probe_ids=("p1", "p2"), observations=12),
    )


class _Reads:
    """A measures root that records every call: `explain` may read observations and do nothing else."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def observations(self, unit: MeasureUnit) -> tuple[Observation, ...]:
        self.calls.append("observations")
        return (observation(),)


def _gpu_plan(cap: int | None = None) -> planner.Plan:
    gpu = {} if cap is None else {"memory_cap_bytes": cap}
    profile = ExecutionProfile.load("laptop", {"resource_classes": {"gpu": gpu}})
    return _plan(_call(GPU_ONLY), _records("x"), hardware=GPU, execution=profile)


@pytest.mark.logic
def test_explain_labels_each_figure_measured_or_unmeasured_and_never_probes() -> None:
    """Each sized figure says `measured` with its source or `unmeasured` with why; nothing is probed."""
    placed = _gpu_plan()
    root = _Reads()
    source = MeasureSource([root], estimate=lambda unit, observed: _measured())
    work = {CALL_NODE: LengthProfile(units=(60.0, 120.0))}
    text = placed.explain(measures=source, units=lambda node: UNIT, lengths=work, hardware=GPU)
    sizing = next(line for line in text.splitlines() if line.startswith("  sizing:"))
    for figure in ("wall ", "gpu ", "peak device memory "):
        assert figure in sizing
    assert sizing.count("measured: 2 probes, 12 observations") == 3 and "unmeasured" not in sizing
    assert root.calls == ["observations"]
    cases: tuple[tuple[dict[str, Any], str], ...] = (
        ({"measures": source, "units": lambda node: UNIT}, "a length is unknown"),
        (
            {
                "measures": MeasureSource([root], estimate=lambda u, o: _measured()),
                "units": lambda n: UNIT,
                "lengths": {CALL_NODE: LengthProfile(units=(None,))},
            },
            "a length is unknown",
        ),
    )
    for kwargs, why in cases:
        text = placed.explain(**kwargs)
        line = next(line for line in text.splitlines() if line.startswith("  sizing:"))
        assert f"wall unmeasured ({why}" in line and "gpu unmeasured" in line and "(measured:" not in line
    assert not any(
        "measured:" in line for line in _gpu_plan().explain(measures=source).splitlines() if "sizing" in line
    )


@pytest.mark.logic
def test_explain_states_the_waits_on_another_run() -> None:
    """The text names where measures are written, the lease takeover delay and the wait for a peer's probe."""
    placed = _plan(_call(), _records("x"))
    text = placed.explain()
    profile = placed.execution
    assert profile is not None
    assert "measures are written to the first writable of:" in text and "2x/measures" in text
    assert f"a lease is taken over after {profile.lease.stale_after_s:g} s" in text
    assert f"waited for up to {profile.peer_wait_s:g} s" in text
    assert text.splitlines()[-1].startswith("waits on another run")


@pytest.mark.logic
def test_explain_flags_a_job_whose_batch_one_peak_exceeds_the_cap() -> None:
    """A job whose `resident + m0 + m1·L` passes the class's memory cap is flagged with the bytes it needs."""
    cap = 12 * _GB
    source = MeasureSource([_Reads()], estimate=lambda unit, observed: _measured())
    work = {CALL_NODE: LengthProfile(units=(100.0, 500.0))}
    text = _gpu_plan(cap).explain(measures=source, units=lambda node: UNIT, lengths=work, hardware=GPU)
    needed = 10 * _GB + 1e7 * 500.0
    assert (
        f"needs {round(needed)} bytes of device memory at batch 1 for its longest job; class gpu caps it at {cap}"
        in text
    )
    short = {CALL_NODE: LengthProfile(units=(100.0,))}
    fits = _gpu_plan(cap).explain(measures=source, units=lambda node: UNIT, lengths=short, hardware=GPU)
    assert "needs " not in fits


def _sized(batched_s: float = 0.1) -> tuple[MeasureSource, ExecutionProfile]:
    """A source whose measures hold an estimate of `UNIT` that batches 64 where `batched_s` is no slower than 0.1."""
    fit = {"n": 6, "spread": 0.1, "lo": 1.0, "hi": 100.0}
    estimate = Estimate(
        status="measured",
        unit_kind="audio_s",
        load_s=2.5,
        load_n=3,
        item_s=0.01,
        item_n=3,
        single=RequestFit(batch=1, r=0.0, u=0.1, **fit),
        batched=(RequestFit(batch=64, r=0.0, u=batched_s, **fit),),
        batch=(StratumBatch(lo=1.0, hi=100.0, batch=64 if batched_s <= 0.1 else 1),),
        host=MemoryFit(resident=1024, m0=0.0, m1=1.0, n=6),
        source=EstimateSource(probe_ids=("p9",), observations=12),
    )
    profile = ExecutionProfile.load("laptop", {"resource_classes": {"cpu": {"mem_bytes": 10_000}}})
    return MeasureSource([_Reads()], estimate=lambda unit, observed: estimate), profile


@pytest.mark.logic
def test_a_measured_unit_batches_in_process(tmp_path: Path) -> None:
    """The executor sizes a node's fragment from `size` over its estimate: many jobs where it batches, else one."""
    source, profile = _sized()
    placed = _plan(_call(), _records("x", "y"), execution=profile)
    assert placed.placement.nodes[CALL_NODE].batch_size == 1
    executor = InProcess(
        store=ResultStore(tmp_path / "a"), profile=profile, workers=WorkerSet(1), measures=source, units=_unit
    )
    assert executor.fragment_size(placed, CALL_NODE) == profile.pool.quantum_items
    slower, _ = _sized(batched_s=0.2)
    unbatched = _plan(_call(), _records("x"), execution=profile)
    assert (
        InProcess(
            store=ResultStore(tmp_path / "b"), profile=profile, workers=WorkerSet(1), measures=slower, units=_unit
        ).fragment_size(unbatched, CALL_NODE)
        == 1
    )


def _unit(call: Any, placement: Any, decoder: Any) -> MeasureUnit:  # noqa: ANN401, ARG001 - a UnitOf
    return UNIT


@pytest.mark.logic
def test_explain_prints_no_cost_line_the_nodes_sizing_is_the_one_estimate() -> None:
    """A plan states no cost of its own: wall time, GPU hours and memory are each node's `Sizing`, from one source."""
    text = _plan(_call(), _records("x")).explain()
    assert "cost:" not in text and "; cost " not in text and "\nenvs to build" in text


@pytest.mark.logic
def test_a_gpu_without_a_class_cap_is_flagged_against_the_device_it_sees() -> None:
    """A class with no `memory_cap_bytes` is judged against the seen device's memory."""
    source = MeasureSource([_Reads()], estimate=lambda unit, observed: _measured(resident=70 * _GB))
    work = {CALL_NODE: LengthProfile(units=(100.0,))}
    text = _gpu_plan().explain(measures=source, units=lambda node: UNIT, lengths=work, hardware=GPU)
    assert f"caps it at {2**36} bytes" in text
    roomy = MeasureSource([_Reads()], estimate=lambda unit, observed: _measured())
    assert "needs " not in _gpu_plan().explain(measures=roomy, units=lambda node: UNIT, lengths=work, hardware=GPU)


@pytest.mark.logic
def test_a_gpu_node_with_a_host_only_estimate_prints_device_memory_unmeasured() -> None:
    """A device grant never reads the host fit: peak device memory is unmeasured, not the host's figure."""
    host_only = _measured().model_copy(update={"device": None, "host": MemoryFit(resident=_GB, m0=0.0, m1=1e6, n=6)})
    source = MeasureSource([_Reads()], estimate=lambda unit, observed: host_only)
    work = {CALL_NODE: LengthProfile(units=(60.0,))}
    text = _gpu_plan().explain(measures=source, units=lambda node: UNIT, lengths=work, hardware=GPU)
    assert "peak device memory unmeasured (memory unmeasured)" in text
