"""Job formation over a placed plan: a node's jobs as a fragment, their keys and checksums, inputs and units.

`node_fragment` names the jobs of one node with their input identities; `fragment_jobs` gives each its graph-input
values and upstream inputs; `job_key` and `job_checksum` give the key each job runs under before anything runs.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from senselab_worker.units import NS_PER_S

from senselab_core.kernel.declare import BackendSpec, Registry, TaskSpec
from senselab_core.kernel.graph import Edge, Node, order
from senselab_core.kernel.identity import InputIdentity, LogicalKey, gather_identity
from senselab_core.kernel.placement import NodePlacement
from senselab_core.runtime.calibration import UnitKind
from senselab_core.runtime.call import TaskCall, packaged
from senselab_core.runtime.execution import batching
from senselab_core.runtime.execution.batching import BucketGrid, job_length_ns, job_text_chars
from senselab_core.runtime.execution.executor import Admission, JobRef, PlanFragment, ResultRef, UpstreamRef
from senselab_core.runtime.execution.job import Job, Upstream
from senselab_core.runtime.execution.outputs import anchors_of
from senselab_core.runtime.execution.workers import service_of
from senselab_core.runtime.keys import (
    Call,
    FormationUnknown,
    Placement,
    call_key,
    fixed_formation,
    request_components,
    request_size,
)
from senselab_core.runtime.planner import Plan
from senselab_core.runtime.units import Unresolved, decoder_of


def _source(edge: Edge) -> tuple[str, str]:
    node, _, port = edge.source.partition(".")
    return node, port


def node_fragment(  # noqa: PLR0913 - the plan and admission, then keyword-only selections
    placed: Plan,
    admission: Admission,
    node: str | None = None,
    *,
    upstream: Mapping[str, Sequence[ResultRef]] | None = None,
    elements: Sequence[tuple[int, ...]] | None = None,
) -> PlanFragment:
    """Return the fragment of the jobs of `node` (the plan's one node when None) under `admission`.

    A node fed by a graph input has one job per manifest record, its element the record's index when a `map` edge
    feeds it; an unmapped node fed only by broadcast inputs has one job. A node fed only by upstream nodes has one job
    per element of their per-element results (one job when every edge into it is a `gather`), `upstream` giving each
    upstream node's results. An element reads its own record and the upstream result of the same element, or of the
    unmapped one; a `gather` edge reads every element's result, in element order, as one `gather:` input. `elements`
    selects some of the node's jobs; None selects every one.

    Raises:
        ValueError: `node` is None and the plan has more than one node; the admission is of another plan; an
            upstream node's results are not given, are missing an element or hold a failure; the manifest holds no
            record, or records that differ in a broadcast input of an unmapped node; `elements` names a job the
            node lacks.
    """
    graph = placed.logical.graph
    if node is None:
        if len(graph.nodes) != 1:
            raise ValueError(f"the plan has nodes {sorted(n.id for n in graph.nodes)}; name the node")
        node = graph.nodes[0].id
    if admission.plan_hash != placed.hash:
        raise ValueError(f"admission {admission.id} is of plan {admission.plan_hash}, not {placed.hash}")
    feeds = {e.target.split(".", 1)[1]: e for e in graph.edges if e.target.split(".", 1)[0] == node}
    given = {p: _source(e)[1] for p, e in feeds.items() if _source(e)[0] == "input"}
    results = {p: _results(node, _source(e)[0], upstream or {}) for p, e in feeds.items() if _source(e)[0] != "input"}
    every = _elements(node, placed, {feeds[p].kind for p in given}, given, results, feeds)
    chosen = every if elements is None else [tuple(e) for e in elements]
    if not set(chosen) <= set(every):
        raise ValueError(f"node {node!r} has elements {every}; {sorted(set(chosen) - set(every))} are not among them")
    rows = placed.manifest.rows
    inputs: list[dict[str, str]] = []
    read: dict[str, UpstreamRef] = {}
    gathers: dict[str, tuple[str, ...]] = {}
    for element in chosen:
        entry: dict[str, str] = {p: rows[element[0] if element else 0].inputs[g] for p, g in given.items()}
        for port, refs in results.items():
            source_port = _source(feeds[port])[1]
            picked = (
                [refs[e] for e in sorted(refs)] if feeds[port].kind == "gather" else [refs.get(element, refs.get(()))]
            )
            stored = [_stored(node, element, port, source_port, ref) for ref in picked]
            read.update({ref.identity(): ref for ref in stored})
            if feeds[port].kind == "gather":
                ordered = tuple(ref.identity() for ref in stored)
                entry[port] = gather_identity(ordered)
                gathers[entry[port]] = ordered
            else:
                entry[port] = stored[0].identity()
        inputs.append(entry)
    return PlanFragment(
        plan_hash=placed.hash,
        node=node,
        admission=admission.id,
        jobs=tuple(JobRef(plan_hash=placed.hash, node=node, element=e) for e in chosen),
        inputs=tuple(inputs),
        upstream=read,
        gathered=gathers,
        placement=placed.placement.nodes[node],
    )


def _elements(  # noqa: PLR0913 - the node, its plan and what feeds it
    node: str,
    placed: Plan,
    kinds: set[str],
    given: Mapping[str, str],
    results: Mapping[str, Mapping[tuple[int, ...], ResultRef]],
    feeds: Mapping[str, Edge],
) -> list[tuple[int, ...]]:
    """Return every element of `node`: a record's index per record when a `map` edge feeds it, else per upstream."""
    rows = placed.manifest.rows
    if "map" in kinds:
        if not rows:
            raise ValueError(f"node {node!r} is mapped over a manifest of no record")
        return [(i,) for i in range(len(rows))]
    per_element = [refs for p, refs in results.items() if feeds[p].kind != "gather"]
    if per_element:
        return sorted({element for refs in per_element for element in refs})
    if given and (not rows or len({tuple(rows[i].inputs[g] for g in given.values()) for i in range(len(rows))}) > 1):
        raise ValueError(f"node {node!r} has {len(rows)} records; an unmapped node reads one value of each input")
    return [()]


def _stored(node: str, element: tuple[int, ...], port: str, source_port: str, ref: ResultRef | None) -> UpstreamRef:
    """Return where port `source_port` of the upstream result `ref` is stored, refusing a missing result."""
    if ref is None or ref.output_identity is None:
        raise ValueError(f"node {node!r} element {element} has no upstream result for input {port!r}")
    length = ref.lengths_ns.get(source_port)
    return UpstreamRef(key_digest=ref.key_digest, outputs=ref.output_identity, port=source_port, length_ns=length)


def _results(node: str, source: str, upstream: Mapping[str, Sequence[ResultRef]]) -> dict[tuple[int, ...], ResultRef]:
    """Return the results of upstream node `source` by element, refusing a missing node or a failed element."""
    if source not in upstream:
        raise ValueError(f"node {node!r} reads node {source!r}; give its results")
    refs = {r.job.element: r for r in upstream[source]}
    failed = sorted(e for e, r in refs.items() if r.failure is not None)
    if failed:
        raise ValueError(f"node {source!r} failed elements {failed}; the runner fails their descendants")
    return refs


def placement_of_record(record: NodePlacement, spec: TaskSpec, registry: Registry) -> Placement:
    """Return the placement a `NodePlacement` records for a node of task `spec`, its backend params validated.

    Raises:
        ValueError: the task declares no such backend at that version, or the params do not validate.
    """
    found = [b for b in registry.backends(spec.name) if b.name == record.backend]
    if not found or found[0].version != record.backend_version:
        raise ValueError(f"{spec.name!r} declares no backend {record.backend!r} at version {record.backend_version}")
    backend = found[0]
    return Placement(
        backend=backend,
        backend_params=backend.params.model_validate(dict(record.backend_params)),
        device=record.device,
        why=record.why,
        resource_class=record.resource_class,
        batch_size=record.batch_size,
        partition=record.partition,
    )


def node_call(placed: Plan, node_id: str, registry: Registry) -> Call:
    """Return the call node `node_id` of `placed` makes, its params parsed, before any input is bound.

    Raises:
        ValueError: the plan has no such node, or names another version of its task than the declared one.
    """
    nodes = {n.id: n for n in placed.logical.graph.nodes}
    if node_id not in nodes:
        raise ValueError(f"plan {placed.hash} has no node {node_id!r}")
    node = nodes[node_id]
    spec = registry.spec(node.task)
    if node.task_version != spec.version:
        raise ValueError(f"{node.task!r} is declared at version {spec.version!r}, not {node.task_version!r}")
    params = TaskCall.parse_params(node.task, node.params, registry)
    return Call(spec, params, {}, placed.logical.models[node_id], placed.logical.profile_slices[node_id])


def job_key(
    placed: Plan,
    node: str,
    inputs: Mapping[str, InputIdentity],
    placement: NodePlacement | None = None,
    *,
    formation: str | None = None,
) -> LogicalKey:
    """Return the logical key `execute` gives the job of `node` of `placed` that reads `inputs`, sent on `placement`.

    `placement` is the node's placement in the plan when None. `formation` is the job's batch formation
    (`batching.formation`), needed only when its length picks it.

    Raises:
        FormationUnknown: the job's length picks its formation and none is given.
        ValueError: as `placement_of_record` or `request_components`, or the plan has no such node.
    """
    registry = packaged().registry
    call = replace(node_call(placed, node, registry), inputs=dict(inputs))
    record = placed.placement.nodes[node] if placement is None else placement
    placed_on = placement_of_record(record, call.task, registry)
    if formation is None and fixed_formation(placed_on.backend) is None:
        raise FormationUnknown(f"node {node!r}: the batch formation of backend {placed_on.backend.name!r} is its job's")
    call = replace(call, service=service_of(placed_on.backend, call.model))
    return call_key(call, replace(placed_on, **request_components(placed_on.backend, formation)))


def fragment_keys(fragment: PlanFragment, placed: Plan, values: Mapping[InputIdentity, object]) -> list[str]:
    """Return the key digest `execute` gives each job of `fragment`.

    Raises:
        ValueError: as `fragment_logical_keys`.
    """
    return [k.digest() for k in fragment_logical_keys(fragment, placed, values)]


def fragment_logical_keys(
    fragment: PlanFragment, placed: Plan, values: Mapping[InputIdentity, object]
) -> list[LogicalKey]:
    """Return the logical key `execute` gives each job of `fragment`.

    A job whose length picks its formation is measured from its graph-input `values` and its upstream refs' recorded
    lengths; no stored value is read.

    Raises:
        ValueError: as `job_key` or `fragment_jobs`.
    """
    record, backend = fragment.placement, _backend(placed, fragment)
    if fixed_formation(backend) is not None:
        return [job_key(placed, fragment.node, i, record) for i in fragment.inputs]
    jobs, _ = fragment_jobs(fragment, placed, values)
    grid = BucketGrid.load_default()
    return [
        job_key(placed, fragment.node, i, record, formation=batching.keyed_formation(j, backend, grid)[0])
        for i, j in zip(fragment.inputs, jobs, strict=True)
    ]


def _backend(placed: Plan, fragment: PlanFragment) -> BackendSpec:
    registry = packaged().registry
    return placement_of_record(fragment.placement, node_call(placed, fragment.node, registry).task, registry).backend


def job_checksum(placed: Plan, node: str, inputs: Mapping[str, InputIdentity]) -> str:
    """Return the digest of the job's key (`job_key`), its bucket left out when its length picks its formation.

    The bucket is a function of the inputs the key names, so the digest still names one job.
    """
    try:
        return job_key(placed, node, inputs).digest()
    except FormationUnknown:
        return job_key(placed, node, inputs, formation=BucketGrid.load_default().unindexed).digest()


def node_request_size(placed: Plan, node: str, placement: NodePlacement | None = None) -> int:
    """Return the most jobs per request `execute` sends node `node` of `placed` on `placement` (the plan's when None).

    Raises:
        ValueError: as `placement_of_record`, or the plan has no such node.
    """
    registry = packaged().registry
    record = placed.placement.nodes[node] if placement is None else placement
    placed_on = placement_of_record(record, node_call(placed, node, registry).task, registry)
    return request_size(placed_on.backend, placed_on.batch_size)


@dataclass(frozen=True)
class PlannedNode:
    """A node's jobs and the key digest each has before anything runs, in the fragment's order."""

    fragment: PlanFragment
    keys: tuple[str, ...]


def plan_nodes(placed: Plan, values: Mapping[InputIdentity, object]) -> dict[str, PlannedNode]:
    """Return each node of `placed` whose job keys are known before any job runs, with them (`fragment_keys`).

    An upstream input is identified by its upstream job's key digest and port, never by its outputs, so every key is
    known before any job runs, a job whose length picks its formation measured from its graph-input `values`. A node
    with no job (mapped over no record) has no key, nor does a node whose length-picked formation reads an upstream
    input, whose length is known only once it ran, or any node below either.
    """
    admission = Admission(id="keys", plan_hash=placed.hash)
    nodes: dict[str, PlannedNode] = {}
    refs: dict[str, list[ResultRef]] = {}
    unread = "output:" + "0" * 64
    for node in order(placed.logical.graph):
        refs[node] = []
        try:
            fragment = node_fragment(placed, admission, node, upstream=refs)
        except ValueError:
            continue
        if fragment.upstream and fixed_formation(_backend(placed, fragment)) is None:
            del refs[node]
            continue
        known = fragment_keys(fragment, placed, values)
        nodes[node] = PlannedNode(fragment, tuple(known))
        for job, key in zip(fragment.jobs, known, strict=True):
            refs[node].append(ResultRef(job=job, key_digest=key, output_identity=unread))
    return nodes


def plan_keys(placed: Plan, values: Mapping[InputIdentity, object]) -> dict[tuple[str, tuple[int, ...]], str]:
    """Return the key digest of every job `plan_nodes` knows, by (node, element)."""
    return {
        (node, job.element): key
        for node, known in plan_nodes(placed, values).items()
        for job, key in zip(known.fragment.jobs, known.keys, strict=True)
    }


def node_jobs(placed: Plan, node: str, values: Mapping[InputIdentity, object]) -> list[Job] | None:
    """Return the jobs of `node` of `placed` over the graph-input `values`; None while they wait on an upstream."""
    try:
        fragment = node_fragment(placed, Admission(id="jobs", plan_hash=placed.hash), node)
        return fragment_jobs(fragment, placed, values)[0]
    except ValueError:
        return None


def node_decoder(placed: Plan, node: str, values: Mapping[InputIdentity, object]) -> str | None | Unresolved:
    """Return the decoder the jobs of `node` read their file-backed inputs by; None when none is or they are unknown."""
    jobs = node_jobs(placed, node, values)
    return None if jobs is None else decoder_of(jobs)


def job_units(job: Job, kind: UnitKind) -> float | None:
    """Return `job`'s input units of `kind`: recording seconds, held characters, or 0 for none; None when unknown."""
    if kind == "none":
        return 0.0
    try:
        length = job_length_ns(job) if kind == "audio_s" else job_text_chars(job)
    except batching.LengthUnrecorded:
        return None
    return None if length is None else length / NS_PER_S if kind == "audio_s" else float(length)


def fragment_jobs(
    fragment: PlanFragment, placed: Plan, values: Mapping[InputIdentity, object]
) -> tuple[list[Job], Placement]:
    """Return the jobs of `fragment` of `placed`, with their graph-input `values` and upstream inputs, and placement.

    Raises:
        ValueError: a job names an input no edge of its node feeds, or a graph input with no value.
    """
    registry = packaged().registry
    base = node_call(placed, fragment.node, registry)
    nodes = {n.id: n for n in placed.logical.graph.nodes}
    edges = placed.logical.graph.edges
    feeds = {e.target.split(".", 1)[1]: _source(e) for e in edges if e.target.split(".", 1)[0] == fragment.node}
    jobs = []
    for ref, identities in zip(fragment.jobs, fragment.inputs, strict=True):
        held, upstream = {}, {}
        for port, identity in identities.items():
            if port not in feeds:
                raise ValueError(f"{ref} names input {port!r}, which no edge of node {fragment.node!r} feeds")
            if feeds[port][0] != "input":
                upstream[port] = _upstream_input(fragment, identity, feeds[port], nodes, registry)
            elif identity in values:
                held[port] = values[identity]
            else:
                raise ValueError(f"no value is admitted for input {port!r} of {ref}")
        call = replace(base, inputs=dict(identities))
        jobs.append(Job(ref, call, held, anchors_of(held.values()), upstream))
    return jobs, placement_of_record(fragment.placement, base.task, registry)


def _upstream_input(
    fragment: PlanFragment,
    identity: str,
    source: tuple[str, str],
    nodes: Mapping[str, Node],
    registry: Registry,
) -> Upstream:
    """Return the upstream input `identity` of `fragment`, read from port `source` (node, port) of the graph.

    Raises:
        ValueError: a stored element is of another port than the edge's.
    """
    node, port = source
    stored = tuple(fragment.upstream[e] for e in fragment.gathered.get(identity, (identity,)))
    if any(s.port != port for s in stored):
        raise ValueError(f"input {identity} reads port {port!r} of node {node!r}, and is stored from another")
    task = nodes[node].task
    return Upstream(stored, task, registry.spec(task).outputs[port], gathered=identity in fragment.gathered)
