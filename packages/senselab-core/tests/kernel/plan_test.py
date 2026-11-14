"""The logical plan document and its hash, the run id."""

from typing import Any

import pytest
from pydantic import ValidationError
from senselab_core.kernel.graph import Edge, Graph, Node
from senselab_core.kernel.identity import ModelRef, ResolvedModel, ReusePolicy
from senselab_core.kernel.license import License
from senselab_core.kernel.plan import LogicalPlan, plan_hash, slice_identity
from senselab_core.runtime.licenses import license_record

COMMIT = "a" * 40
MANIFEST = "b" * 64


def _graph(
    params: dict[str, Any] | None = None, backend_params: dict[str, Any] | None = None, model: str | None = None
) -> Graph:
    node = Node(
        id="n",
        task="t.score",
        task_version="1",
        params=params or {"w": 1},
        backend_params=backend_params or {},
        model=model,
    )
    return Graph(
        inputs={"text": "list[json]"},
        nodes=(node,),
        edges=(Edge(source="input.text", target="n.text", kind="map"),),
        outputs={"value": "n.value"},
    )


def _model(commit: str = COMMIT, ref: str | None = None, **fields: Any) -> ResolvedModel:  # noqa: ANN401
    return ResolvedModel(ref=ModelRef.parse(f"org/m@{ref or commit}"), identity=commit, **fields)


def _license(date: str) -> License:
    return license_record("MIT", source="card", verified={"date": date, "how": "hf-card"}, gated=False)


def _plan(**update: Any) -> LogicalPlan:  # noqa: ANN401 - field overrides
    fields: dict[str, Any] = {
        "graph": _graph(),
        "models": {"n": _model()},
        "profile_slices": {"n": {"dataset.lexicon": slice_identity(["a", "b"])}},
        "manifest_digest": MANIFEST,
        "bind": {"text": "transcript"},
        "reuse": ReusePolicy.EQUIVALENT,
    }
    return LogicalPlan(**{**fields, **update})


@pytest.mark.logic
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("models", {"n": _model("c" * 40)}),
        ("models", {"n": None}),
        ("graph", _graph(params={"w": 2})),
        ("graph", _graph(backend_params={"fake": {"beam": 2}})),
        ("profile_slices", {"n": {"dataset.lexicon": slice_identity(["a"])}}),
        ("profile_slices", {"n": {}}),
        ("manifest_digest", "c" * 64),
        ("bind", {}),
        ("reuse", ReusePolicy.EXACT),
    ],
    ids=["model-pin", "no-model", "params", "backend-params", "slice", "no-slice", "manifest", "bind", "reuse"],
)
def test_plan_hash_changes_with_each_logical_field(field: str, value: object) -> None:
    """Every logical field reaches the hash; the plan alone does not change it."""
    assert plan_hash(_plan()) == plan_hash(_plan())
    assert plan_hash(_plan(**{field: value})) != plan_hash(_plan())


@pytest.mark.logic
@pytest.mark.parametrize(
    "update",
    [
        {"graph": _graph(model="org/m@main"), "models": {"n": _model(ref="main")}},
        {"graph": _graph(model=f"org/m@{COMMIT}")},
        {"models": {"n": _model(license=_license("2026-11-01"))}},
        {"models": {"n": _model(license=_license("2026-11-02"))}},
    ],
    ids=["ref-main-same-commit", "ref-text-in-graph", "license", "license-date"],
)
def test_plan_hash_ignores_reference_text_and_license(update: dict[str, Any]) -> None:
    """Only a model's pinned identity reaches the hash: not its reference text nor its license."""
    assert plan_hash(_plan(**update)) == plan_hash(_plan())


@pytest.mark.logic
def test_local_model_moved_keeps_the_hash() -> None:
    """A local model at another path, with the same weights, keeps the hash."""
    weights = "d" * 64

    def local(path: str) -> LogicalPlan:
        model = ResolvedModel(ref=ModelRef.parse(f"local:{path}"), identity=weights)
        return _plan(graph=_graph(model=f"local:{path}"), models={"n": model})

    assert plan_hash(local("/laptop/m")) == plan_hash(local("/cluster/scratch/m"))
    assert plan_hash(local("/laptop/m")) != plan_hash(_plan())


@pytest.mark.logic
def test_plan_hash_ignores_mapping_and_edge_order() -> None:
    """Models, slices and edges given in another order hash the same."""
    nodes = (Node(id="a", task="t", task_version="1"), Node(id="b", task="t", task_version="1"))
    inputs = {"x": "list[json]", "y": "list[json]"}
    edges = (Edge(source="input.x", target="a.x", kind="map"), Edge(source="input.y", target="b.y", kind="map"))
    one = Graph(inputs=inputs, nodes=nodes, edges=edges)
    two = Graph(inputs=inputs, nodes=nodes, edges=tuple(reversed(edges)))
    slices = {"a": {"run.k": slice_identity(1)}, "b": {}}
    first = _plan(graph=one, models={"a": _model(), "b": None}, profile_slices=slices, bind={})
    second = _plan(graph=two, models={"b": None, "a": _model()}, profile_slices=dict(reversed(slices.items())), bind={})
    assert plan_hash(first) == plan_hash(second)


@pytest.mark.logic
def test_later_minor_graph_in_a_plan_has_no_hash() -> None:
    """A 1.0 plan holding a later-minor graph is refused a hash through the graph's digest."""
    graph = Graph.model_validate({**_graph().model_dump(mode="json"), "format": "graph@1.1"})
    with pytest.raises(ValueError, match=r"graph@1\.1"):
        plan_hash(_plan(graph=graph))


@pytest.mark.logic
def test_plan_round_trips_json_and_keeps_its_hash() -> None:
    """A plan read back from its JSON is equal to it and hashes the same."""
    plan = _plan()
    again = LogicalPlan.model_validate_json(plan.model_dump_json())
    assert again == plan and plan_hash(again) == plan_hash(plan)
    assert plan.model_dump(mode="json")["format"] == "logical-plan@1.0"


@pytest.mark.logic
def test_plan_hash_ignores_node_order() -> None:
    """The graph enters by its digest, so the order nodes are written in does not reach the hash."""
    two = Graph(nodes=(Node(id="a", task="t", task_version="1"), Node(id="b", task="t", task_version="1")))
    flipped = two.model_copy(update={"nodes": tuple(reversed(two.nodes))})
    common = {"models": {"a": None, "b": None}, "profile_slices": {"a": {}, "b": {}}, "bind": {}}
    assert plan_hash(_plan(graph=two, **common)) == plan_hash(_plan(graph=flipped, **common))


@pytest.mark.logic
@pytest.mark.parametrize(
    "update",
    [{"models": {}}, {"profile_slices": {"n": {}, "m": {}}}, {"bind": {"other": "x"}}, {"manifest_digest": "x"}],
    ids=["missing-model", "unknown-slice-node", "bind-unknown-input", "bad-digest"],
)
def test_plan_refuses_entries_that_do_not_match_its_graph(update: dict[str, Any]) -> None:
    """Models and slices name exactly the graph's nodes, bind only its inputs, and the digest is 64 hex."""
    with pytest.raises(ValidationError):
        _plan(**update)


@pytest.mark.logic
def test_later_minor_plan_has_no_hash() -> None:
    """A reader of 1.0 reads a 1.1 plan but gives it no hash."""
    later = LogicalPlan.model_validate({**_plan().model_dump(mode="json"), "format": "logical-plan@1.1", "x": 1})
    with pytest.raises(ValueError, match=r"logical-plan@1\.1"):
        plan_hash(later)
    with pytest.raises(ValidationError):
        LogicalPlan.model_validate({**_plan().model_dump(mode="json"), "format": "logical-plan@2.0"})


@pytest.mark.logic
def test_plan_reuse_names_every_policy() -> None:
    """The plan holds a reuse policy by its value, and every policy has one."""
    assert {_plan(reuse=p).reuse for p in ReusePolicy} == {p.value for p in ReusePolicy}
    with pytest.raises(ValidationError):
        _plan(reuse="loose")


@pytest.mark.logic
def test_slice_identity_is_the_value_digest() -> None:
    """A slice identity depends on its value only, not on key order."""
    assert slice_identity({"a": 1, "b": 2}) == slice_identity({"b": 2, "a": 1}) != slice_identity({"a": 1})
    assert slice_identity(1).startswith("profile:")
