"""The planner: a graph, or one task call, and the records it runs over resolved to a `LogicalPlan`.

`resolve` reads no execution profile and no hardware: it pins every model, cuts every declared profile slice and
builds the input manifest, so the plan and its hash are the same on every host.
"""

import re
from collections.abc import Mapping
from typing import Any

from senselab_core.kernel.dataset import Dataset, InputManifest, ManifestRow
from senselab_core.kernel.declare import Registry, TaskSpec
from senselab_core.kernel.graph import Edge, Graph, GraphIssue, Node, validate
from senselab_core.kernel.identity import ModelRef, ResolvedModel
from senselab_core.kernel.plan import LogicalPlan
from senselab_core.runtime.call import ITEM_ID, RunConfig, TaskCall, bind_input, packaged, pin_model
from senselab_core.runtime.catalogue import Catalogue
from senselab_core.runtime.datasets import build_manifest
from senselab_core.runtime.keys import profile_slices
from senselab_core.runtime.revision import PROCESS_RESOLVER, Resolver

CALL_NODE = "call"
"""The id of the one node a task call is planned as."""

_LIST = re.compile(r"^list\[(.+)\]$")


class InvalidGraph(ValueError):
    """A graph `validate` finds issues in; `issues` holds them."""

    def __init__(self, issues: list[GraphIssue]) -> None:
        """Hold `issues` and name each in the message."""
        super().__init__("; ".join(i.message for i in issues))
        self.issues = issues


def one_node_graph(call: TaskCall, registry: Registry) -> Graph:
    """Return the one-node graph of `call`: its first input port mapped over the record list, the rest per record.

    The node is `call`; the graph input of the first declared input port is `list[T]` and joined by a `map` edge,
    every other is `T` and joined by a plain edge, and each output port is a graph output of the same name. The
    call's `backend_params`, when given, are the entry of every backend declared for the task.
    """
    spec = registry.spec(call.task)
    ports = list(spec.inputs.items())
    inputs = {port: f"list[{kind}]" if i == 0 else kind for i, (port, kind) in enumerate(ports)}
    edges = tuple(
        Edge(source=f"input.{port}", target=f"{CALL_NODE}.{port}", kind="map" if i == 0 else "plain")
        for i, (port, _) in enumerate(ports)
    )
    backend_params = (
        {b.name: dict(call.backend_params) for b in registry.backends(spec.name)} if call.backend_params else {}
    )
    node = Node(
        id=CALL_NODE,
        task=call.task,
        task_version=call.task_version,
        params=call.params.model_dump(mode="json"),
        backend_params=backend_params,
        model=None if call.model is None else str(call.model),
    )
    outputs = {port: f"{CALL_NODE}.{port}" for port in spec.outputs}
    return Graph(inputs=inputs, nodes=(node,), edges=edges, outputs=outputs)


def _item_type(port_type: str) -> str:
    listed = _LIST.match(port_type)
    return listed[1] if listed else port_type


def _manifest(
    graph: Graph, dataset: Dataset | None, call: TaskCall | None, spec: TaskSpec | None, bind: Mapping[str, str]
) -> InputManifest:
    """Return the manifest of the records `graph` runs over: the dataset's, or the call's one record."""
    if dataset is not None:
        if call is not None and call.inputs:
            raise ValueError("a call planned over a dataset takes its inputs from the records; it gives none itself")
        return build_manifest(dataset, {port: _item_type(t) for port, t in graph.inputs.items()}, bind=bind)
    if bind:
        raise ValueError("bind names the record inputs of a dataset; there is no dataset")
    if call is None or spec is None:
        if graph.inputs:
            raise ValueError(f"the graph has inputs {sorted(graph.inputs)}; they are bound from a dataset's records")
        return InputManifest()
    if set(call.inputs) != set(spec.inputs):
        raise ValueError(f"{call.task!r} takes inputs {sorted(spec.inputs)}; the call gives {sorted(call.inputs)}")
    identities = {port: bind_input(port, spec.inputs[port], value)[0] for port, value in call.inputs.items()}
    return InputManifest(rows=(ManifestRow(id=ITEM_ID, entities={}, inputs=identities),))


def _pinned(spec: TaskSpec, node: Node, catalogue: Catalogue, resolver: Resolver) -> ResolvedModel | None:
    """Return the node's model pinned: a hub model's reference names the commit its identity is.

    Raises:
        UnpinnedModel: the model's identity does not pin its weights.
    """
    model, _ = pin_model(
        spec.name, None if node.model is None else ModelRef.parse(node.model), catalogue, resolver=resolver
    )
    if model is None:
        return None
    model.key_identity()
    if model.ref.source != "hf":
        return model
    ref = ModelRef(source="hf", id=model.ref.id, ref=model.identity)
    return ResolvedModel(ref=ref, identity=model.identity, license=model.license)


def resolve(  # noqa: PLR0913 - the target and records, then keyword-only resolution inputs
    target: Graph | TaskCall,
    dataset: Dataset | None,
    *,
    config: RunConfig,
    dataset_profile: Mapping[str, Any] | None = None,
    bind: Mapping[str, str] | None = None,
    resolver: Resolver = PROCESS_RESOLVER,
) -> tuple[LogicalPlan, InputManifest]:
    """Return the logical plan of `target` over `dataset`, and the input manifest whose digest it holds.

    A task call becomes `one_node_graph`; without a dataset its own inputs are the one record, `call`. Each node's
    model is the catalogue's pin, or `resolve()`d through `resolver` (the task's catalogue default when the node names
    none); each node's declared profile slices are cut from `config.settings` and `dataset_profile`
    (`keys.profile_slices`). `bind` maps graph inputs to the record inputs they read (`build_manifest`). The plan's
    reuse policy is `config.reuse`.

    Raises:
        InvalidGraph: the graph has issues against the packaged registry.
        ValueError: the records do not bind the graph's inputs, or a call over a dataset gives inputs itself.
        MissingProfileSlice: a declared profile slice is absent.
        UnpinnedModel, RevisionResolutionError, ModelResolutionError: a model cannot be pinned.
    """
    registry, catalogue = packaged().registry, packaged().catalogue
    call = target if isinstance(target, TaskCall) else None
    spec = None if call is None else registry.spec(call.task)
    graph = one_node_graph(target, registry) if isinstance(target, TaskCall) else target
    issues = validate(graph, registry)
    if issues:
        raise InvalidGraph(issues)
    manifest = _manifest(graph, dataset, call, spec, bind or {})
    specs = {n.id: registry.spec(n.task) for n in graph.nodes}
    plan = LogicalPlan(
        graph=graph,
        models={n.id: _pinned(specs[n.id], n, catalogue, resolver) for n in graph.nodes},
        profile_slices={n.id: profile_slices(specs[n.id], config.settings, dataset_profile) for n in graph.nodes},
        manifest_digest=manifest.digest(),
        bind=bind or {},
        reuse=config.reuse,
    )
    return plan, manifest
