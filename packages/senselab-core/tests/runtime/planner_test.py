"""`resolve`: a graph or one call over records to a logical plan, with no execution input (fake stack)."""

import inspect
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import senselab_core.runtime.fake_backend as fake
from senselab_core.kernel.declare import BackendSpec, Env, Registry, TaskParams, TaskSpec
from senselab_core.kernel.graph import Graph
from senselab_core.kernel.identity import ReusePolicy
from senselab_core.kernel.plan import plan_hash, slice_identity
from senselab_core.runtime.call import Packaged, RunConfig, TaskCall, use_packaged
from senselab_core.runtime.catalogue import Catalogue
from senselab_core.runtime.datasets import from_records
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.keys import MissingProfileSlice
from senselab_core.runtime.planner import CALL_NODE, InvalidGraph, resolve
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.revision import Resolver

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


def _backend(name: str, task: TaskSpec) -> BackendSpec:
    return BackendSpec(
        task=task.name,
        name=name,
        version="1",
        stack=name,
        env=Env.host(),
        devices=("cpu",),
        batch_invariant=True,
        worker=fake.WORKER_MODULE,
        load_stub={},
    )


REGISTRY = Registry()
for _task in (SCORE, READS):
    REGISTRY.add_task(_task)
REGISTRY.add_backend(_backend("alpha", SCORE))
REGISTRY.add_backend(_backend("beta", SCORE))
REGISTRY.add_backend(_backend("alpha", READS))

CATALOGUE = Catalogue.load(
    f"""
format: catalogue@1.0
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
    assert names == {"target", "dataset", "config", "dataset_profile", "bind", "resolver"}
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
    assert plan_hash(again) == plan_hash(plan)
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
def test_call_backend_params_are_held_for_every_backend_it_may_be_placed_on() -> None:
    """A call's backend params become the entry of each backend declared for its task."""
    plan, _ = resolve(_call(backend_params={"beam": 2}), _records("x"), config=RunConfig())
    assert {k: dict(v) for k, v in plan.graph.nodes[0].backend_params.items()} == {
        "alpha": {"beam": 2},
        "beta": {"beam": 2},
    }
    bare, _ = resolve(_call(), _records("x"), config=RunConfig())
    assert dict(bare.graph.nodes[0].backend_params) == {}


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
