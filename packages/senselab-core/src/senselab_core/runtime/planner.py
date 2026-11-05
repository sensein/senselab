"""The planner: a graph, or one task call, and the records it runs over resolved to a `LogicalPlan`, then placed.

`resolve` reads no execution profile and no hardware: it pins every model, cuts every declared profile slice and
builds the input manifest, so the plan and its hash are the same on every host. `plan` resolves and then places the
logical plan under an execution profile on this host (`runtime.placement`), giving a `Plan`.
"""

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from senselab_core.kernel.dataset import Dataset, InputManifest, ManifestRow
from senselab_core.kernel.declare import Registry, TaskSpec
from senselab_core.kernel.graph import Edge, Graph, GraphIssue, Node, validate
from senselab_core.kernel.identity import ModelRef, ResolvedModel
from senselab_core.kernel.placement import PlanPlacement
from senselab_core.kernel.plan import LogicalPlan, plan_hash
from senselab_core.runtime.call import (
    ITEM_ID,
    RunConfig,
    TaskCall,
    bind_input,
    env_license,
    host_hardware,
    packaged,
    pin_model,
)
from senselab_core.runtime.catalogue import Catalogue
from senselab_core.runtime.datasets import build_manifest
from senselab_core.runtime.explain import explain
from senselab_core.runtime.keys import profile_slices
from senselab_core.runtime.placement import check_caps, check_licenses, node_licenses, place_nodes, plan_cost
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.revision import PROCESS_RESOLVER, Resolver

CALL_NODE = "call"
"""The id of the one node a task call is planned as."""

_LIST = re.compile(r"^list\[(.+)\]$")


class RecordBinding(ValueError):
    """A graph over a dataset whose inputs are not one record list fed by `map` edges; `ports` names its inputs."""

    def __init__(self, ports: tuple[str, ...], message: str) -> None:
        """Hold the graph's input `ports` and name them in the message."""
        super().__init__(f"graph inputs {list(ports)}: {message}")
        self.ports = ports


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
    node holds the call's params, its per-backend `backend_params` and its model reference as given.
    """
    spec = registry.spec(call.task)
    ports = list(spec.inputs.items())
    inputs = {port: f"list[{kind}]" if i == 0 else kind for i, (port, kind) in enumerate(ports)}
    edges = tuple(
        Edge(source=f"input.{port}", target=f"{CALL_NODE}.{port}", kind="map" if i == 0 else "plain")
        for i, (port, _) in enumerate(ports)
    )
    node = Node(
        id=CALL_NODE,
        task=call.task,
        task_version=call.task_version,
        params=call.params.model_dump(mode="json"),
        backend_params=call.backend_params,
        model=None if call.model is None else str(call.model),
    )
    outputs = {port: f"{CALL_NODE}.{port}" for port in spec.outputs}
    return Graph(inputs=inputs, nodes=(node,), edges=edges, outputs=outputs)


def _record_port(graph: Graph) -> tuple[str, str]:
    """Return the one graph input a dataset's records bind, and the type each record gives it.

    Raises:
        RecordBinding: the graph has no input or more than one, or its input is not a `list[T]` every edge of which
            is a `map` edge.
    """
    ports = tuple(sorted(graph.inputs))
    if len(ports) != 1:
        raise RecordBinding(ports, "a graph over a dataset has exactly one input, the record list")
    port = ports[0]
    listed = _LIST.match(graph.inputs[port])
    kinds = {e.kind for e in graph.edges if e.source == f"input.{port}"}
    if listed is None or kinds != {"map"}:
        raise RecordBinding(ports, f"the record list is a list[T] fed only by map edges, got {graph.inputs[port]}")
    return port, listed[1]


def _manifest(
    graph: Graph, dataset: Dataset | None, call: TaskCall | None, spec: TaskSpec | None, bind: Mapping[str, str]
) -> InputManifest:
    """Return the manifest of the records `graph` runs over: the dataset's, or the call's one record."""
    if dataset is not None:
        if call is not None and call.inputs:
            raise ValueError("a call planned over a dataset takes its inputs from the records; it gives none itself")
        port, item = _record_port(graph)
        return build_manifest(dataset, {port: item}, bind=bind)
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


def _completed(graph: Graph, registry: Registry) -> Graph:
    """Return `graph` with each node's params validated against its task and completed with their defaults.

    Raises:
        pydantic.ValidationError, InconsistentInputs: as `TaskCall.parse_params`.
    """
    nodes = tuple(
        n.model_copy(update={"params": TaskCall.parse_params(n.task, n.params, registry).model_dump(mode="json")})
        for n in graph.nodes
    )
    return graph.model_copy(update={"nodes": nodes})


def _pinned(
    spec: TaskSpec, node: Node, catalogue: Catalogue, resolver: Resolver, admit: Callable[[str], None] | None
) -> ResolvedModel | None:
    """Return the node's model pinned: a hub model's reference names the commit its identity is.

    Raises:
        UnpinnedModel: the model's identity does not pin its weights.
    """
    ref = None if node.model is None else ModelRef.parse(node.model)
    model, _ = pin_model(spec.name, ref, catalogue, resolver=resolver, admit=admit)
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
    admit: Callable[[str], None] | None = None,
) -> tuple[LogicalPlan, InputManifest]:
    """Return the logical plan of `target` over `dataset`, and the input manifest whose digest it holds.

    A task call becomes `one_node_graph`; without a dataset its own inputs are the one record, `call`. Each node's
    params are completed with their declared defaults. Each node's model is the catalogue's pin, or `resolve()`d
    through `resolver` once `admit` (a source allowlist, raising to refuse) accepts its hub id; a node naming none
    takes the task's catalogue default. Each node's declared profile slices are cut from `config.settings` and
    `dataset_profile` (`keys.profile_slices`). Over a dataset the graph has one input, the record list, and `bind`
    maps it to the record input it reads (`build_manifest`). The plan's reuse policy is `config.reuse`. Models and
    slices are resolved before any input is read.

    Raises:
        InvalidGraph: the graph has issues against the packaged registry.
        RecordBinding: over a dataset, the graph's inputs are not one record list fed by `map` edges.
        ValueError: the records do not bind the graph's input, or a call over a dataset gives inputs itself.
        MissingProfileSlice: a declared profile slice is absent.
        UnpinnedModel, RevisionResolutionError, ModelResolutionError: a model cannot be pinned; or what `admit` raises.
    """
    registry, catalogue = packaged().registry, packaged().catalogue
    call = target if isinstance(target, TaskCall) else None
    spec = None if call is None else registry.spec(call.task)
    graph = one_node_graph(target, registry) if isinstance(target, TaskCall) else target
    issues = validate(graph, registry)
    if issues:
        raise InvalidGraph(issues)
    graph = _completed(graph, registry)
    specs = {n.id: registry.spec(n.task) for n in graph.nodes}
    models = {n.id: _pinned(specs[n.id], n, catalogue, resolver, admit) for n in graph.nodes}
    slices = {n.id: profile_slices(specs[n.id], config.settings, dataset_profile) for n in graph.nodes}
    manifest = _manifest(graph, dataset, call, spec, bind or {})
    plan = LogicalPlan(
        graph=graph,
        models=models,
        profile_slices=slices,
        manifest_digest=manifest.digest(),
        bind=bind or {},
        reuse=config.reuse,
    )
    return plan, manifest


@dataclass(frozen=True)
class Plan:
    """A placed plan: the logical plan and its manifest, its placement, and its hash, the run id (`plan_hash`)."""

    logical: LogicalPlan
    manifest: InputManifest
    placement: PlanPlacement
    hash: str

    def explain(self) -> str:
        """Return, per node, the backend, stack, env, device, batch, resource class, partition, licenses and reason.

        The first lines give the plan hash, the envs to build and the cost, naming what is unmeasured.
        """
        return explain(self.hash, self.logical, self.placement)


def plan(  # noqa: PLR0913 - the target and records, then keyword-only run inputs
    target: Graph | TaskCall,
    dataset: Dataset | None = None,
    *,
    config: RunConfig | None = None,
    dataset_profile: Mapping[str, Any] | None = None,
    execution: ExecutionProfile | None = None,
    bind: Mapping[str, str] | None = None,
) -> Plan:
    """Return `target` over `dataset` resolved (`resolve`) and placed under `execution` on this host.

    Models are resolved under the profile's network policy and its source allowlist. Each node is placed
    (`place_nodes`) with its placed backend's params validated, gets its license set (`node_licenses`), and the run
    config's license policy is checked on every record of it. The memory caps of each resource class are checked
    (`check_caps`), the envs not yet built are named and the cost is summed from the resource tables (`plan_cost`).
    `None` means `RunConfig()` and the `laptop` profile.

    Raises:
        InvalidGraph, RecordBinding, ValueError, MissingProfileSlice, UnpinnedModel: as `resolve`.
        UnlistedSource: a model's organisation is not an allowed source.
        BackendParamsMismatch: a node's backend params name an undeclared backend, or its placed backend rejects them.
        NoBackend, NoDevice: no backend or resource class runs a node; the error names the node.
        LicenseRefused: the run config's license policy refuses a record of a node's license set.
        MemoryCapExceeded: a resource class's resident models exceed its memory cap.
    """
    config = config if config is not None else RunConfig()
    profile = execution if execution is not None else ExecutionProfile.load()
    registry, catalogue = packaged().registry, packaged().catalogue
    logical, manifest = resolve(
        target,
        dataset,
        config=config,
        dataset_profile=dataset_profile,
        bind=bind,
        resolver=replace(PROCESS_RESOLVER, network=profile.network),
        admit=lambda model_id: profile.check_source(model_id, catalogue),
    )
    placed = place_nodes(logical, profile, host_hardware(), registry, catalogue)
    licenses = node_licenses(logical, placed, env_license)
    check_licenses(config.licenses, licenses)
    check_caps(logical, placed, profile, catalogue)
    builds = sorted(
        {
            p.env()
            for p in placed.values()
            if p.backend.env.kind != "host" and not packaged().env_built(p.backend, profile)
        }
    )
    placement = PlanPlacement(
        profile_digest=profile.canonical_digest(),
        nodes={n: p.record() for n, p in placed.items()},
        env_builds=tuple(builds),
        licenses=licenses,
        cost=plan_cost(logical, placed, catalogue),
    )
    return Plan(logical=logical, manifest=manifest, placement=placement, hash=plan_hash(logical))
