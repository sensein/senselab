"""The planner: a graph, or one task call, and the records it runs over resolved to a `LogicalPlan`, then placed.

`resolve` reads no execution profile and no hardware: it pins every model, cuts every declared profile slice and
builds the input manifest, so the plan and its hash are the same on every host. `plan` resolves and then places the
logical plan under an execution profile on this host (`runtime.placement`), giving a `Plan`.
"""

import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from senselab_core.kernel import plan as kernel_plan
from senselab_core.kernel.dataset import Dataset, InputManifest, ManifestRow
from senselab_core.kernel.declare import Registry, TaskSpec
from senselab_core.kernel.graph import Edge, Graph, GraphIssue, Node, validate
from senselab_core.kernel.identity import InputIdentity, ModelRef, ResolvedModel, ReusePolicy
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
from senselab_core.runtime.revision import PROCESS_RESOLVER, Resolver, cache_root
from senselab_core.runtime.storage import StorageCapabilities, probe_capabilities

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
) -> tuple[InputManifest, dict[str, Any], dict[InputIdentity, object]]:
    """Return the manifest of the records `graph` runs over, the sources its inputs are read from and the live values.

    The sources are `{"rows": [{port: source}], "shared": {port: source}}`, a row per manifest row and the broadcast
    inputs once (`_source`); the live values are those of the `{"memory": <identity>}` sources, by identity. Over a
    dataset each row holds its record's identity for every record-bound input and the broadcast identity of every
    other input.
    """
    live: dict[InputIdentity, object] = {}
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
        bound = {port: bind_input(port, kind, inputs[port]) for port, kind in broadcast.items()}
        shared = {port: identity for port, (identity, _, _) in bound.items()}
        built, rows = sourced_manifest(dataset, record, bind=bind)
        manifest = InputManifest(
            rows=tuple(ManifestRow(id=r.id, entities=r.entities, inputs={**r.inputs, **shared}) for r in built.rows)
        )
        sources = {port: _source(inputs[port], identity, value, live) for port, (identity, value, _) in bound.items()}
        return manifest, {"rows": rows, "shared": sources}, live
    if bind or inputs:
        raise ValueError("bind and inputs name the inputs of a graph over a dataset; there is no dataset")
    if call is None or spec is None:
        if graph.inputs:
            raise ValueError(f"the graph has inputs {sorted(graph.inputs)}; they are bound from a dataset's records")
        return InputManifest(), {"rows": [], "shared": {}}, live
    if set(call.inputs) != set(spec.inputs):
        raise ValueError(f"{call.task!r} takes inputs {sorted(spec.inputs)}; the call gives {sorted(call.inputs)}")
    bound = {port: bind_input(port, spec.inputs[port], value) for port, value in call.inputs.items()}
    identities = {port: identity for port, (identity, _, _) in bound.items()}
    manifest = InputManifest(rows=(ManifestRow(id=ITEM_ID, entities={}, inputs=identities),))
    row = {port: _source(call.inputs[port], identity, value, live) for port, (identity, value, _) in bound.items()}
    return manifest, {"rows": [row], "shared": {}}, live


def _source(given: object, identity: InputIdentity, value: object, live: dict[InputIdentity, object]) -> object:
    """Return what an input given as `given` is read from.

    That is its path as text or its JSON value; any other object is `{"memory": identity}`, its bound `value` added to
    `live`.
    """
    if isinstance(given, os.PathLike):
        return os.fspath(given)
    try:
        json.dumps(given)
    except (TypeError, ValueError):
        live[identity] = value
        return {"memory": identity}
    return given


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
    """A resolved plan with what it runs over: its input manifest, the sources its inputs are read from, live values."""

    logical: LogicalPlan
    manifest: InputManifest
    sources: Mapping[str, Any]
    live: Mapping[InputIdentity, object]


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
    manifest, sources, live = _manifest(graph, dataset, call, spec, bind or {}, inputs or {})
    plan = LogicalPlan(
        graph=graph,
        models=models,
        profile_slices=slices,
        manifest_digest=manifest.digest(),
        bind=bind or {},
        reuse=config.reuse,
    )
    return Resolved(plan, manifest, sources, live)


@dataclass(frozen=True)
class Plan:
    """A placed plan: the logical plan and its manifest, its placement, and its hash, the run id (`plan_hash`).

    Beside the documents it holds what running it needs, none of which is part of its identity: the execution
    profile and run config it was placed under, the sources its inputs are read from (`_manifest`) and the values
    of the inputs given as objects (`live`), which only this process holds.
    """

    logical: LogicalPlan
    manifest: InputManifest
    placement: PlanPlacement
    hash: str
    execution: ExecutionProfile | None = field(default=None, compare=False, repr=False)
    config: RunConfig | None = field(default=None, compare=False, repr=False)
    sources: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)
    live: Mapping[InputIdentity, object] = field(default_factory=dict, compare=False, repr=False)

    def run(self, executor: "Executor | None" = None, *, store: "ResultStore | None" = None) -> "Run":
        """Start this plan on `executor` and return its `Run`, which runs in the background.

        `None` means `InProcess` over the result store under `cache_root()/results`, under the plan's own profile and
        run config, when the profile's substrate is local, and a `Pool` over `Slurm` when it is `slurm`. `store` is the
        result store the run's results are read from; give it with an executor that stores results.

        Raises:
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
    resolver: Resolver | None = None,
) -> Plan:
    """Return `target` over `dataset` resolved (`resolve`) and placed under `execution` on this host (`place`).

    Models are resolved through `resolver` (`None`: the process resolver under the profile's network policy) and
    the profile's source allowlist. `None` means `RunConfig()`, the `laptop` profile and this host's probed hardware
    (`host_hardware`).

    Raises:
        InvalidGraph, UnmappedOutput, RecordBinding, ValueError, MissingProfileSlice, UnpinnedModel: as `resolve`.
        UnlistedSource: a model's organisation is not an allowed source.
        BackendParamsMismatch, NoBackend, NoDevice, LicenseRefused, LicenseConflict, MemoryCapExceeded: as `place`.
        StorageUnsupported: the run directory's or the result store's root lacks a primitive the leases need.
        LeaseBoundError: the root's measured attribute delay breaks the profile's lease bound.
    """
    config = config if config is not None else RunConfig()
    profile = execution if execution is not None else ExecutionProfile.load()
    check_storage(profile, *storage_roots())
    catalogue = packaged().catalogue
    found = resolve_bound(
        target,
        dataset,
        config=config,
        dataset_profile=dataset_profile,
        bind=bind,
        inputs=inputs,
        resolver=resolver if resolver is not None else replace(PROCESS_RESOLVER, network=profile.network),
        admit=lambda model_id: profile.check_source(model_id, catalogue),
    )
    return Plan(
        logical=found.logical,
        manifest=found.manifest,
        placement=place(found.logical, profile, hardware or host_hardware(), config),
        hash=plan_hash(found.logical),
        execution=profile,
        config=config,
        sources=found.sources,
        live=found.live,
    )


def storage_roots() -> tuple[Path, ...]:
    """Return the roots a run writes under by default: the run directory root and the result store's."""
    from senselab_core.runtime.execution.events import runs_root  # noqa: PLC0415 - execution imports the planner
    from senselab_core.runtime.execution.run import RESULTS_DIR  # noqa: PLC0415 - as above

    return runs_root(), cache_root() / RESULTS_DIR


def check_storage(
    profile: ExecutionProfile, *roots: Path, probe: Callable[[Path], StorageCapabilities] = probe_capabilities
) -> None:
    """Refuse the plan unless each of `roots` has the primitives leases need and a delay the lease bound allows.

    Each root is probed (`probe`, by default `probe_capabilities`). A root whose nearest existing ancestor cannot be
    written is not judged: a run there could not write either.

    Raises:
        StorageUnsupported: a root lacks a primitive; names it.
        LeaseBoundError: a root's measured attribute delay makes `2 × renew_fraction × stale_after_s +
            attribute_delay_s` not below `stale_after_s`.
    """
    for root in roots:
        ancestor = root
        while not ancestor.exists():
            ancestor = ancestor.parent
        if not os.access(ancestor, os.W_OK | os.X_OK):
            continue
        found = probe(root)
        found.require(root)
        profile.lease.model_copy(update={"attribute_delay_s": found.attribute_delay_s}).check_bound()


def place(logical: LogicalPlan, profile: ExecutionProfile, hardware: HardwareInfo, config: RunConfig) -> PlanPlacement:
    """Return the placement of `logical` under `profile` on `hardware`; it reads the plan and resolves nothing.

    Each node is placed (`place_nodes`) with its placed backend's params validated, gets its license set
    (`node_licenses`), and the run config's license policy is checked on every record of it. The memory caps of each
    resource class are checked (`check_caps`), the envs not yet built are named and the cost is summed from the
    resource tables (`plan_cost`).

    Raises:
        UnlistedSource: a model's organisation is not an allowed source.
        BackendParamsMismatch: a node's backend params name an undeclared backend, or its placed backend rejects them.
        NoBackend, NoDevice: no backend or resource class runs a node; the error names the node.
        LicenseRefused: the run config's license policy refuses a record of a node's license set.
        LicenseConflict: one subject reaches a node with two license records.
        MemoryCapExceeded: a resource class's resident models exceed its memory cap.
    """
    registry, catalogue = packaged().registry, packaged().catalogue
    placed = place_nodes(logical, profile, hardware, registry, catalogue)
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
    return PlanPlacement(
        profile_digest=profile.canonical_digest(),
        nodes={n: p.record() for n, p in placed.items()},
        env_builds=tuple(builds),
        licenses=licenses,
        cost=plan_cost(logical, placed, catalogue),
    )


class NoRun(FileNotFoundError):
    """No run of the plan hash is on disk: `<cache root>/2x/runs/<plan hash>/plan.json` is absent."""


def resume(  # noqa: PLR0913 - the plan hash, then keyword-only run inputs
    plan_hash: str,
    *,
    execution: ExecutionProfile | str | None = None,
    store: "ResultStore | None" = None,
    executor: "Executor | None" = None,
    config: RunConfig | None = None,
    hardware: HardwareInfo | None = None,
) -> "Run":
    """Start the run of `plan_hash` again from its run directory, as a new attempt under a fresh placement.

    The logical plan, input manifest and input sources are read from `plan.json` and `inputs.json`; the plan's
    pinned models are run as they are and no model reference is resolved. The plan is placed (`place`) under
    `execution` (a profile, the name of a packaged one, or `None` for the packaged default) on `hardware` (`None`:
    this host, probed); every input is then bound again and checked against the manifest's identity, all before
    anything is written. The plan is started (`Plan.run`), the placement written to a new
    `placements/<attempt>.json`. The plan hash is unchanged; a node placed on another backend has another key and is
    computed again, and a job the store holds is served from it. `store` is where results are read and written: by
    default the one `manifest.json` names. `executor` defaults as in `Plan.run`.
    `config` defaults to `RunConfig()` under the plan's reuse policy; given with an executor, it is that executor's
    run config.

    Raises:
        NoRun: no run of `plan_hash` is on disk.
        ValueError: `plan.json` is not the plan `plan_hash` names, or `config` reuses under another policy.
        InputChanged: an input changed since the plan was made; it names every record, and nothing is written.
        InputsUnavailable: an input was held in memory by the process that planned the run.
        UnlistedSource, BackendParamsMismatch, NoBackend, NoDevice, LicenseRefused, LicenseConflict,
            MemoryCapExceeded: as `place`.
        MissingExtra: as `Plan.run`.
    """
    from senselab_core.runtime.execution.events import run_dir_of  # noqa: PLC0415 - execution imports the planner
    from senselab_core.runtime.execution.run import Run, bind_sources, load_plan, start  # noqa: PLC0415 - as above
    from senselab_core.runtime.store import ResultStore  # noqa: PLC0415 - as above

    run_dir = run_dir_of(plan_hash)
    try:
        stored = load_plan(run_dir)
    except FileNotFoundError as err:
        raise NoRun(f"there is no run of plan {plan_hash}; plan and run it first") from err
    if stored.hash != plan_hash or kernel_plan.plan_hash(stored.logical) != plan_hash:
        raise ValueError(f"the run directory of plan {plan_hash} holds another plan")
    if stored.manifest.digest() != stored.logical.manifest_digest:
        raise ValueError(f"the input manifest of plan {plan_hash} is not the one its plan holds")
    reuse = ReusePolicy(stored.logical.reuse)
    config = config if config is not None else RunConfig(reuse=reuse)
    if config.reuse is not reuse:
        raise ValueError(f"plan {plan_hash} reuses under {reuse.value}, not {config.reuse.value}")
    profile = ExecutionProfile.load(execution) if isinstance(execution, str) else execution or ExecutionProfile.load()
    placement = place(stored.logical, profile, hardware or host_hardware(), config)
    values = bind_sources(stored)
    placed = replace(stored, placement=placement, execution=profile, config=config)
    if store is None:
        named = Run(run_dir).manifest().get("store")
        store = ResultStore(Path(named)) if isinstance(named, str) else None
    return start(placed, executor, store=store, values=values)
