"""Placing a logical plan: each node's placement under a profile, its license set, the memory caps and the cost.

`place_nodes` picks each node's backend, resource class and device (`resolve_execution`) and validates its backend
params; `node_licenses` gives each node the union of its own licenses and its upstream nodes'; `check_caps` refuses a
resource class whose resident models exceed its memory cap; `plan_cost` sums the measured resource tables.
"""

from collections.abc import Callable, Mapping
from dataclasses import replace

from pydantic import ValidationError

from senselab_core.kernel.declare import BackendParams, BackendSpec, Registry
from senselab_core.kernel.graph import Node, NodeId, split
from senselab_core.kernel.graph import order as graph_order
from senselab_core.kernel.identity import ResolvedModel
from senselab_core.kernel.license import License, LicenseConflict, LicenseSet
from senselab_core.kernel.placement import Cost
from senselab_core.kernel.plan import LogicalPlan
from senselab_core.runtime.catalogue import Catalogue, CatalogueEntry
from senselab_core.runtime.hardware import HardwareInfo
from senselab_core.runtime.keys import Placement, request_components, request_size
from senselab_core.runtime.licenses import LicensePolicy
from senselab_core.runtime.profiles import ExecutionProfile, NoBackend, NoDevice, resolve_execution
from senselab_core.runtime.resources import ResourceMeasure, ResourceRow

_HOUR_S = 3600.0


class BackendParamsMismatch(ValueError):
    """A node's `backend_params` entry its backend's schema rejects, or one naming a backend its task lacks."""

    def __init__(self, node: str, backend: str, errors: list[str]) -> None:
        """Name the node, the backend and each error."""
        super().__init__(f"node {node!r}: backend params for {backend!r} are refused: {'; '.join(errors)}")
        self.node = node
        self.backend = backend
        self.errors = errors


class MemoryCapExceeded(ValueError):
    """A resource class whose resident models need more memory than its static cap."""

    def __init__(self, resource_class: str, cap: int, needed: int, nodes: list[str]) -> None:
        """Name the class, its cap, what its resident models need and the nodes placed on it."""
        super().__init__(
            f"resource class {resource_class!r} caps memory at {cap} bytes; the models resident at once for nodes "
            f"{nodes} need {needed}"
        )
        self.resource_class = resource_class
        self.cap = cap
        self.needed = needed
        self.nodes = nodes


def _entry(catalogue: Catalogue, model: ResolvedModel | None) -> CatalogueEntry | None:
    if model is None:
        return None
    return next((e for e in catalogue.entries if (e.id, e.revision) == (model.ref.id, model.identity)), None)


def _params(node: Node, backend: BackendSpec, registry: Registry) -> BackendParams:
    unknown = sorted(set(node.backend_params) - {b.name for b in registry.backends(node.task)})
    if unknown:
        raise BackendParamsMismatch(node.id, unknown[0], [f"task {node.task!r} declares no backend {unknown[0]!r}"])
    try:
        return backend.params.model_validate(dict(node.backend_params.get(backend.name, {})))
    except ValidationError as err:
        errors = [f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}" for e in err.errors()]
        raise BackendParamsMismatch(node.id, backend.name, errors) from err


def place_nodes(
    plan: LogicalPlan, profile: ExecutionProfile, hardware: HardwareInfo, registry: Registry, catalogue: Catalogue
) -> dict[NodeId, Placement]:
    """Return each node's placement under `profile`, its backend params validated against the placed backend.

    A `slurm` substrate places onto the cluster's classes, so `hardware` is not consulted; a `local` one places onto
    the classes this host has. A node with no entry for its placed backend takes that backend's defaults.

    Raises:
        BackendParamsMismatch: an entry names a backend the task does not declare, or the placed backend rejects it.
        NoBackend, NoDevice: as `resolve_execution`, naming the node.
    """
    here = None if profile.substrate.kind == "slurm" else hardware
    placed: dict[NodeId, Placement] = {}
    for node in plan.graph.nodes:
        spec = registry.spec(node.task)
        try:
            placement = resolve_execution(
                spec, _entry(catalogue, plan.models[node.id]), profile, here, registry, catalogue
            )
        except (NoBackend, NoDevice) as err:
            raise type(err)(f"node {node.id!r}: {err}") from err
        params = _params(node, placement.backend, registry)
        size = request_size(placement.backend, placement.batch_size)
        placed[node.id] = replace(placement, backend_params=params, **request_components(placement.backend, size))
    return placed


def node_licenses(
    plan: LogicalPlan, placed: Mapping[NodeId, Placement], env_license: Callable[[BackendSpec], License | None]
) -> dict[NodeId, LicenseSet]:
    """Return each node's license set: its env's and model's records, joined with every upstream node's.

    A subject is `env of backend <task>/<name>` or `model <ref>`; one without a record maps to None.

    Raises:
        LicenseConflict: a subject is recorded with two records; names the node.
    """
    upstream: dict[str, set[str]] = {n.id: set() for n in plan.graph.nodes}
    for edge in plan.graph.edges:
        source, target = split(edge.source)[0], split(edge.target)[0]
        if source != "input" and target in upstream:
            upstream[target].add(source)
    sets: dict[NodeId, LicenseSet] = {}
    for node_id in graph_order(plan.graph):
        backend = placed[node_id].backend
        own: dict[str, License | None] = {f"env of backend {backend.task}/{backend.name}": env_license(backend)}
        model = plan.models[node_id]
        if model is not None:
            own[f"model {model.ref}"] = model.license
        try:
            sets[node_id] = LicenseSet(records=own).union(*(sets[u] for u in sorted(upstream[node_id])))
        except LicenseConflict as err:
            raise LicenseConflict(err.subject, node_id) from err
    return sets


def check_licenses(policy: LicensePolicy | None, licenses: Mapping[NodeId, LicenseSet]) -> None:
    """Raise `LicenseRefused` when `policy` refuses any record of any node's license set; None refuses nothing."""
    if policy is None:
        return
    for node_id, found in licenses.items():
        for subject, record in found.records.items():
            policy.check(f"node {node_id!r}: {subject}", record)


def _measure(
    catalogue: Catalogue, placement: Placement, model: ResolvedModel | None
) -> tuple[ResourceRow, ResourceMeasure] | None:
    backend = placement.backend
    ident = (None, None) if model is None else (model.ref.id, model.identity)
    row = catalogue.resources(backend.task, backend.name, *ident)
    measure = None if row is None else row.classes.get(placement.resource_class)
    return None if row is None or measure is None else (row, measure)


def check_caps(
    plan: LogicalPlan, placed: Mapping[NodeId, Placement], profile: ExecutionProfile, catalogue: Catalogue
) -> None:
    """Refuse a resource class whose models resident at once need more than its `memory_cap_bytes`.

    The models resident at once on a class are the distinct backend and model pairs of its nodes, each at its
    measured peak; a worker may hold several, so the sum is an upper bound. A node without a resource table is
    admitted (`plan_cost` lists it as unmeasured).

    Raises:
        MemoryCapExceeded: a class's resident peaks sum past its cap.
    """
    resident: dict[str, dict[tuple[str, str | None], int]] = {}
    nodes: dict[str, list[str]] = {}
    for node_id, placement in placed.items():
        model = plan.models[node_id]
        measured = _measure(catalogue, placement, model)
        if measured is None:
            continue
        pair = (placement.backend.name, None if model is None else model.identity)
        resident.setdefault(placement.resource_class, {})[pair] = measured[1].peak_memory_bytes
        nodes.setdefault(placement.resource_class, []).append(node_id)
    for name, peaks in sorted(resident.items()):
        cap = profile.resource_classes[name].memory_cap_bytes
        if cap is not None and sum(peaks.values()) > cap:
            raise MemoryCapExceeded(name, cap, sum(peaks.values()), sorted(nodes[name]))


def plan_cost(
    plan: LogicalPlan,
    placed: Mapping[NodeId, Placement],
    catalogue: Catalogue,
    input_s: float | None = None,
) -> Cost:
    """Return the plan's cost from the measured resource tables over `input_s` seconds of input per node.

    A node's wall time is its load plus `s_per_input_s` times `input_s`; GPU hours are the wall time of nodes on a
    `cuda` class; download bytes sum once per model. A figure is None when any node or model is unmeasured, and wall
    time and GPU hours are None when `input_s` is None; nothing is estimated.
    """
    unmeasured: set[str] = set()
    downloads: dict[tuple[str, str | None], int] = {}
    wall, gpu = 0.0, 0.0
    for node_id, placement in placed.items():
        model = plan.models[node_id]
        measured = _measure(catalogue, placement, model)
        if measured is None:
            unmeasured.update([node_id] + ([] if model is None else [str(model.ref)]))
            continue
        row, measure = measured
        downloads[(row.id, row.revision) if row.id is not None else (placement.backend.name, None)] = row.download_bytes
        node_wall = measure.load_s + measure.s_per_input_s * (input_s or 0.0)
        wall += node_wall
        gpu += node_wall / _HOUR_S if placement.device == "cuda" else 0.0
    timed = not unmeasured and input_s is not None
    return Cost(
        wall_s=wall if timed else None,
        gpu_hours=gpu if timed else None,
        download_bytes=None if unmeasured else sum(downloads.values()),
        unmeasured=tuple(sorted(unmeasured)),
    )
