"""The planner: a graph, or one task call, and the records it runs over resolved to a `LogicalPlan`, then placed.

`resolve` reads no execution profile and no hardware: it pins every model, cuts every declared profile slice and
builds the input manifest, so the plan and its hash are the same on every host. `plan` resolves and then places the
logical plan under an execution profile on this host (`runtime.placement`), giving a `Plan`.
"""

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

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
from senselab_core.runtime.datasets import sourced_manifest
from senselab_core.runtime.explain import explain
from senselab_core.runtime.hardware import HardwareInfo
from senselab_core.runtime.keys import profile_slices
from senselab_core.runtime.placement import check_caps, check_licenses, node_licenses, place_nodes, plan_cost
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.revision import PROCESS_RESOLVER, Resolver

if TYPE_CHECKING:
    from senselab_core.runtime.execution.executor import Executor
    from senselab_core.runtime.execution.run import Run
    from senselab_core.runtime.store import ResultStore

CALL_NODE = "call"
"""The id of the one node a task call is planned as."""

_LIST = re.compile(r"^list\[(.+)\]$")


class RecordBinding(ValueError):
    """A graph over a dataset whose inputs the records and the broadcast values do not bind; `ports` names them."""

    def __init__(self, ports: tuple[str, ...], message: str) -> None:
        """Hold the graph's input `ports` and name them in the message."""
        super().__init__(f"graph inputs {list(ports)}: {message}")
        self.ports = ports


class UnmappedOutput(ValueError):
    """A `map` over a node's list output, which is not run: one item of a stored list output has no identity yet.

    `node` is the node the map feeds and `source` the output it reads.
    """

    def __init__(self, node: str, source: str) -> None:
        """Name the node and the output it maps over."""
        super().__init__(
            f"node {node!r} maps over {source}, a node's list output; one item of a stored list output has no "
            "identity yet, so such a map is not run"
        )
        self.node = node
        self.source = source


def check_mapped_outputs(graph: Graph) -> None:
    """Refuse a `map` over a node's output; a map reads a graph input.

    Raises:
        UnmappedOutput: a `map` edge reads a node's output.
    """
    for e in graph.edges:
        if e.kind == "map" and not e.source.startswith("input."):
            raise UnmappedOutput(e.target.split(".", 1)[0], e.source)


class InvalidGraph(ValueError):
    """A graph `validate` finds issues in; `issues` holds them."""

    def __init__(self, issues: list[GraphIssue]) -> None:
        """Hold `issues` and name each in the message."""
        super().__init__("; ".join(i.message for i in issues))
        self.issues = issues


def one_node_graph(call: TaskCall, registry: Registry, *, every_port: bool = False) -> Graph:
    """Return the one-node graph of `call`: its first input port mapped over the record list, the rest per record.

    The node is `call`; the graph input of the first declared input port is `list[T]` and joined by a `map` edge,
    every other is `T` and joined by a plain edge, and each output port is a graph output of the same name. With
    `every_port`, every input port is a `list[T]` joined by a `map` edge, so each record binds every port. The
    node holds the call's params, its per-backend `backend_params` and its model reference as given.
    """
    spec = registry.spec(call.task)
    ports = list(spec.inputs.items())
    mapped = [every_port or i == 0 for i in range(len(ports))]
    inputs = {port: f"list[{kind}]" if m else kind for m, (port, kind) in zip(mapped, ports, strict=True)}
    edges = tuple(
        Edge(source=f"input.{port}", target=f"{CALL_NODE}.{port}", kind="map" if m else "plain")
        for m, (port, _) in zip(mapped, ports, strict=True)
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


def _record_ports(graph: Graph) -> tuple[dict[str, str], dict[str, str]]:
    """Return the graph inputs a dataset's records bind, with the type each record gives, and the broadcast inputs.

    A record-bound input is a `list[T]` every edge of which is a `map` edge; every record gives each one, so they are
    zipped by record. Every other input is broadcast: one value, given once, for every element.

    Raises:
        RecordBinding: no input is record-bound, or a `list[T]` input is fed by both `map` and other edges.
    """
    record: dict[str, str] = {}
    broadcast: dict[str, str] = {}
    for port, declared in sorted(graph.inputs.items()):
        listed = _LIST.match(declared)
        kinds = {e.kind for e in graph.edges if e.source == f"input.{port}"}
        if listed is not None and kinds == {"map"}:
            record[port] = listed[1]
        elif "map" in kinds:
            raise RecordBinding((port,), f"a record list is fed only by map edges, got {sorted(kinds)}")
        else:
            broadcast[port] = declared
    if not record:
        raise RecordBinding(tuple(sorted(graph.inputs)), "a graph over a dataset maps at least one list[T] input")
    return record, broadcast


def _manifest(  # noqa: PLR0913 - the graph and what binds its inputs
    graph: Graph,
    dataset: Dataset | None,
    call: TaskCall | None,
    spec: TaskSpec | None,
    bind: Mapping[str, str],
    inputs: Mapping[str, object],
) -> tuple[InputManifest, dict[str, Any]]:
    """Return the manifest of the records `graph` runs over and the sources its inputs are read from.

    The sources are `{"rows": [{port: source}], "shared": {port: source}}`, a row per manifest row and the broadcast
    inputs once; a source is a path as text or a JSON value. Over a dataset each row holds its
    record's identity for every record-bound input and the broadcast identity of every other input.
    """
    if dataset is not None:
        if call is not None and call.inputs:
            raise ValueError("a call planned over a dataset takes its inputs from the records; it gives none itself")
        record, broadcast = _record_ports(graph)
        if set(inputs) != set(broadcast):
            raise RecordBinding(
                tuple(sorted(broadcast)), f"the broadcast inputs are given once as inputs; got {sorted(inputs)}"
            )
        if set(bind) - set(record):
            raise RecordBinding(tuple(sorted(set(bind) - set(record))), "bind names record-bound inputs only")
        shared = {port: bind_input(port, kind, inputs[port])[0] for port, kind in broadcast.items()}
        built, rows = sourced_manifest(dataset, record, bind=bind)
        manifest = InputManifest(
            rows=tuple(ManifestRow(id=r.id, entities=r.entities, inputs={**r.inputs, **shared}) for r in built.rows)
        )
        return manifest, {"rows": rows, "shared": {port: inputs[port] for port in broadcast}}
    if bind or inputs:
        raise ValueError("bind and inputs name the inputs of a graph over a dataset; there is no dataset")
    if call is None or spec is None:
        if graph.inputs:
            raise ValueError(f"the graph has inputs {sorted(graph.inputs)}; they are bound from a dataset's records")
        return InputManifest(), {"rows": [], "shared": {}}
    if set(call.inputs) != set(spec.inputs):
        raise ValueError(f"{call.task!r} takes inputs {sorted(spec.inputs)}; the call gives {sorted(call.inputs)}")
    bound = {port: bind_input(port, spec.inputs[port], value) for port, value in call.inputs.items()}
    identities = {port: identity for port, (identity, _, _) in bound.items()}
    manifest = InputManifest(rows=(ManifestRow(id=ITEM_ID, entities={}, inputs=identities),))
    return manifest, {"rows": [dict(call.inputs)], "shared": {}}


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


@dataclass(frozen=True)
class Resolved:
    """A resolved plan with what it runs over: its input manifest and the sources its inputs are read from."""

    logical: LogicalPlan
    manifest: InputManifest
    sources: Mapping[str, Any]


def resolve(  # noqa: PLR0913 - the target and records, then keyword-only resolution inputs
    target: Graph | TaskCall,
    dataset: Dataset | None,
    *,
    config: RunConfig,
    dataset_profile: Mapping[str, Any] | None = None,
    bind: Mapping[str, str] | None = None,
    inputs: Mapping[str, object] | None = None,
    resolver: Resolver = PROCESS_RESOLVER,
    admit: Callable[[str], None] | None = None,
) -> tuple[LogicalPlan, InputManifest]:
    """Return the logical plan of `target` over `dataset`, and the input manifest whose digest it holds.

    See `resolve_bound`, which also returns the input sources.
    """
    found = resolve_bound(
        target,
        dataset,
        config=config,
        dataset_profile=dataset_profile,
        bind=bind,
        inputs=inputs,
        resolver=resolver,
        admit=admit,
    )
    return found.logical, found.manifest


def resolve_bound(  # noqa: PLR0913 - as `resolve`
    target: Graph | TaskCall,
    dataset: Dataset | None,
    *,
    config: RunConfig,
    dataset_profile: Mapping[str, Any] | None = None,
    bind: Mapping[str, str] | None = None,
    inputs: Mapping[str, object] | None = None,
    resolver: Resolver = PROCESS_RESOLVER,
    admit: Callable[[str], None] | None = None,
) -> Resolved:
    """Return the logical plan of `target` over `dataset`, its input manifest and the sources of its inputs.

    A task call becomes `one_node_graph`; without a dataset its own inputs are the one record, `call`. Each node's
    params are completed with their declared defaults. Each node's model is the catalogue's pin, or `resolve()`d
    through `resolver` once `admit` (a source allowlist, raising to refuse) accepts its hub id; a node naming none
    takes the task's catalogue default. Each node's declared profile slices are cut from `config.settings` and
    `dataset_profile` (`keys.profile_slices`). Over a dataset each `list[T]` input fed only by `map` edges is bound
    from every record, zipped by record, and `bind` maps such an input to the record input it reads
    (`build_manifest`); every other input is broadcast to every element, its one value given in `inputs`. The plan's
    reuse policy is `config.reuse`. Models and slices are resolved before any input is read.

    Raises:
        InvalidGraph: the graph has issues against the packaged registry.
        UnmappedOutput: a `map` edge reads a node's output.
        RecordBinding: over a dataset, no input is a record list fed only by `map` edges, a list is fed by both
            kinds, `inputs` does not give exactly the broadcast inputs, or `bind` names another input.
        ValueError: the records do not bind the graph's inputs, a call over a dataset gives inputs itself, or `bind`
            or `inputs` is given without a dataset.
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
    check_mapped_outputs(graph)
    graph = _completed(graph, registry)
    specs = {n.id: registry.spec(n.task) for n in graph.nodes}
    models = {n.id: _pinned(specs[n.id], n, catalogue, resolver, admit) for n in graph.nodes}
    slices = {n.id: profile_slices(specs[n.id], config.settings, dataset_profile) for n in graph.nodes}
    manifest, sources = _manifest(graph, dataset, call, spec, bind or {}, inputs or {})
    plan = LogicalPlan(
        graph=graph,
        models=models,
        profile_slices=slices,
        manifest_digest=manifest.digest(),
        bind=bind or {},
        reuse=config.reuse,
    )
    return Resolved(plan, manifest, sources)


class ExecutorUnavailable(RuntimeError):
    """The executor a profile's substrate needs is not available yet; `needs` names it."""

    def __init__(self, substrate: str, needs: str) -> None:
        """Name the substrate and the executor it needs."""
        super().__init__(
            f"the {substrate!r} substrate runs on {needs}, which is not available; pass an executor to run"
        )
        self.substrate = substrate
        self.needs = needs


@dataclass(frozen=True)
class Plan:
    """A placed plan: the logical plan and its manifest, its placement, and its hash, the run id (`plan_hash`).

    Beside the documents it holds what running it needs, none of which is part of its identity: the execution
    profile and run config it was placed under and the sources its inputs are read from (`_manifest`).
    """

    logical: LogicalPlan
    manifest: InputManifest
    placement: PlanPlacement
    hash: str
    execution: ExecutionProfile | None = field(default=None, compare=False, repr=False)
    config: RunConfig | None = field(default=None, compare=False, repr=False)
    sources: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def run(self, executor: "Executor | None" = None, *, store: "ResultStore | None" = None) -> "Run":
        """Start this plan on `executor` and return its `Run`, which runs in the background.

        `None` means `InProcess` over the result store under `cache_root()/results`, under the plan's own profile and
        run config, when the profile's substrate is local. `store` is the result store the run's results are read from;
        give it with an executor that stores results.

        Raises:
            ExecutorUnavailable: no executor is given and the profile's substrate is `slurm`, which runs on `Pool`.
            ValueError: as `Executor.admit`.
            MissingExtra: the `pydra` extra is not installed.
        """
        from senselab_core.runtime.execution.run import start  # noqa: PLC0415 - run imports this module

        return start(self, executor, store=store)

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
    inputs: Mapping[str, object] | None = None,
    hardware: HardwareInfo | None = None,
) -> Plan:
    """Return `target` over `dataset` resolved (`resolve`) and placed under `execution` on this host.

    Models are resolved under the profile's network policy and its source allowlist. Each node is placed
    (`place_nodes`) with its placed backend's params validated, gets its license set (`node_licenses`), and the run
    config's license policy is checked on every record of it. The memory caps of each resource class are checked
    (`check_caps`), the envs not yet built are named and the cost is summed from the resource tables (`plan_cost`).
    `None` means `RunConfig()`, the `laptop` profile and this host's probed hardware (`host_hardware`).

    Raises:
        InvalidGraph, UnmappedOutput, RecordBinding, ValueError, MissingProfileSlice, UnpinnedModel: as `resolve`.
        UnlistedSource: a model's organisation is not an allowed source.
        BackendParamsMismatch: a node's backend params name an undeclared backend, or its placed backend rejects them.
        NoBackend, NoDevice: no backend or resource class runs a node; the error names the node.
        LicenseRefused: the run config's license policy refuses a record of a node's license set.
        LicenseConflict: one subject reaches a node with two license records.
        MemoryCapExceeded: a resource class's resident models exceed its memory cap.
    """
    config = config if config is not None else RunConfig()
    profile = execution if execution is not None else ExecutionProfile.load()
    registry, catalogue = packaged().registry, packaged().catalogue
    found = resolve_bound(
        target,
        dataset,
        config=config,
        dataset_profile=dataset_profile,
        bind=bind,
        inputs=inputs,
        resolver=replace(PROCESS_RESOLVER, network=profile.network),
        admit=lambda model_id: profile.check_source(model_id, catalogue),
    )
    logical = found.logical
    placed = place_nodes(logical, profile, hardware or host_hardware(), registry, catalogue)
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
    return Plan(
        logical=logical,
        manifest=found.manifest,
        placement=placement,
        hash=plan_hash(logical),
        execution=profile,
        config=config,
        sources=found.sources,
    )
