"""`InProcess`: the `Executor` that runs a fragment's jobs through `execute` in this process."""

import atexit
import itertools
import threading
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import ClassVar

from senselab_core.kernel.declare import Registry, TaskSpec
from senselab_core.kernel.graph import Edge, Node, order
from senselab_core.kernel.identity import InputIdentity, LogicalKey, ReusePolicy, gather_identity
from senselab_core.kernel.placement import NodePlacement
from senselab_core.runtime.call import Result, RunConfig, TaskCall, bind_input, packaged
from senselab_core.runtime.execution.batching import BucketGrid
from senselab_core.runtime.execution.events import EventLog
from senselab_core.runtime.execution.executor import (
    Admission,
    FaultRecord,
    Handle,
    JobRef,
    PlanFragment,
    ResultRef,
    Status,
    UpstreamRef,
)
from senselab_core.runtime.execution.job import Failed, Job, JobOutcome, Upstream, execute, job_formation
from senselab_core.runtime.execution.outputs import anchors_of
from senselab_core.runtime.execution.records import add_plan_entity
from senselab_core.runtime.execution.services import service_registry
from senselab_core.runtime.execution.workers import WorkerSet, service_of
from senselab_core.runtime.keys import (
    BUCKET,
    Call,
    FormationUnknown,
    Placement,
    call_key,
    fixed_formation,
    request_components,
    request_size,
)
from senselab_core.runtime.planner import Plan, plan
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.revision import cache_root
from senselab_core.runtime.store import ResultStore


class CallFailed(RuntimeError):
    """The backend failed the call's item; `failure` is its record, with its fault class."""

    def __init__(self, failure: FaultRecord) -> None:
        """Hold the failure and name it in the message."""
        super().__init__(f"{failure.type} ({failure.fault}): {failure.message}")
        self.failure = failure


_PROCESS: list[WorkerSet] = []
_PROCESS_LOCK = threading.Lock()


def process_workers(profile: ExecutionProfile) -> WorkerSet:
    """Return this process's one worker set, bounded by the workers `profile`'s resource classes keep alive at once.

    The first profile sets the bound and the registry whose services it consumes (`service_registry`); a later profile
    with a smaller bound tightens it (`WorkerSet.tighten`), and a larger one leaves it. The set is closed at exit.
    """
    bound = sum(c.max_workers for c in profile.resource_classes.values())
    with _PROCESS_LOCK:
        if not _PROCESS:
            _PROCESS.append(WorkerSet(bound, registry=service_registry(profile)))
        workers = _PROCESS[0]
    workers.tighten(bound)
    return workers


@atexit.register
def _close_process_workers() -> None:
    for workers in _PROCESS:
        workers.close()


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
    return UpstreamRef(key_digest=ref.key_digest, outputs=ref.output_identity, port=source_port)


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
        env_root=record.env_root,
        why=record.why,
        resource_class=record.resource_class,
        batch_size=record.batch_size,
        partition=record.partition,
    )


def _node_call(placed: Plan, node_id: str, registry: Registry) -> Call:
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
    (`job_formation`), needed only when its length picks it.

    Raises:
        FormationUnknown: the job's length picks its formation and none is given.
        ValueError: as `placement_of_record` or `request_components`, or the plan has no such node.
    """
    registry = packaged().registry
    call = replace(_node_call(placed, node, registry), inputs=dict(inputs))
    record = placed.placement.nodes[node] if placement is None else placement
    placed_on = placement_of_record(record, call.task, registry)
    if formation is None and fixed_formation(placed_on.backend) is None:
        raise FormationUnknown(f"node {node!r}: the batch formation of backend {placed_on.backend.name!r} is its job's")
    call = replace(call, service=service_of(placed_on.backend, call.model))
    return call_key(call, replace(placed_on, **request_components(placed_on.backend, formation)))


def fragment_keys(
    fragment: PlanFragment, placed: Plan, values: Mapping[InputIdentity, object], store: ResultStore
) -> list[str]:
    """Return the key digest `execute` gives each job of `fragment`.

    A job whose length picks its formation is read from its graph-input `values` and its upstream entries in `store`.

    Raises:
        ValueError: as `job_key` or `fragment_jobs`.
    """
    registry, record = packaged().registry, fragment.placement
    backend = placement_of_record(record, _node_call(placed, fragment.node, registry).task, registry).backend
    if fixed_formation(backend) is not None:
        return [job_key(placed, fragment.node, i, record).digest() for i in fragment.inputs]
    jobs, _ = fragment_jobs(fragment, placed, values)
    grid = BucketGrid.load_default()
    return [
        job_key(placed, fragment.node, i, record, formation=job_formation(j, backend, store, grid)).digest()
        for i, j in zip(fragment.inputs, jobs, strict=True)
    ]


def job_checksum(placed: Plan, node: str, inputs: Mapping[str, InputIdentity]) -> str:
    """Return the digest of the job's key (`job_key`), its bucket left out when its length picks its formation.

    The bucket is a function of the inputs the key names, so the digest still names one job.
    """
    try:
        return job_key(placed, node, inputs).digest()
    except FormationUnknown:
        return job_key(placed, node, inputs, formation=f"{BUCKET}:{BucketGrid.load_default().id}").digest()


def node_request_size(placed: Plan, node: str, placement: NodePlacement | None = None) -> int:
    """Return the most jobs per request `execute` sends node `node` of `placed` on `placement` (the plan's when None).

    Raises:
        ValueError: as `placement_of_record`, or the plan has no such node.
    """
    registry = packaged().registry
    record = placed.placement.nodes[node] if placement is None else placement
    placed_on = placement_of_record(record, _node_call(placed, node, registry).task, registry)
    return request_size(placed_on.backend, placed_on.batch_size)


def result_ref(outcome: JobOutcome) -> ResultRef:
    """Return the `ResultRef` of one job's outcome: its key digest and output identity, or its failure."""
    if isinstance(outcome, Failed):
        return ResultRef(job=outcome.ref, key_digest=outcome.key_digest, failure=outcome.failure)
    reused = outcome.result.reused
    return ResultRef(
        job=outcome.ref,
        key_digest=outcome.result.key.digest(),
        output_identity=outcome.result.output_identity,
        served=None if reused is None else reused.value,
    )


def plan_keys(placed: Plan) -> dict[tuple[str, tuple[int, ...]], str]:
    """Return the key digest of every job of `placed` by (node, element), as `job_key` gives it under the plan.

    An upstream input is identified by its upstream job's key digest and port, never by its outputs, so every key is
    known before any job runs. A node with no job (mapped over no record) has no key, nor does a node whose jobs'
    lengths pick their formation, or any node below it.
    """
    admission = Admission(id="keys", plan_hash=placed.hash)
    keys: dict[tuple[str, tuple[int, ...]], str] = {}
    refs: dict[str, list[ResultRef]] = {}
    unread = "output:" + "0" * 64
    for node in order(placed.logical.graph):
        refs[node] = []
        try:
            fragment = node_fragment(placed, admission, node, upstream=refs)
        except ValueError:
            continue
        try:
            known = [job_key(placed, node, inputs).digest() for inputs in fragment.inputs]
        except FormationUnknown:
            del refs[node]
            continue
        for job, key in zip(fragment.jobs, known, strict=True):
            keys[(node, job.element)] = key
            refs[node].append(ResultRef(job=job, key_digest=key, output_identity=unread))
    return keys


class InProcess:
    """An `Executor` whose `submit` runs a fragment's jobs through `execute` in this process, before returning.

    A fragment is submitted under an admission of its plan (`admit`), which holds the value of each graph-input
    identity its jobs name; an upstream input is read from the store.
    Jobs are sent in batches of at most the placement's batch size (`form_batches`). `None` means the packaged
    default: the store under `cache_root()/results`, the `laptop` profile, `RunConfig()` and this process's worker
    set for the profile (`process_workers`).
    """

    def __init__(
        self,
        *,
        store: ResultStore | None = None,
        profile: ExecutionProfile | None = None,
        run: RunConfig | None = None,
        workers: WorkerSet | None = None,
    ) -> None:
        """Run fragments under `profile` and `run`, storing in `store`, on `workers`."""
        self.profile = profile if profile is not None else ExecutionProfile.load()
        self.run = run if run is not None else RunConfig()
        self.store = store if store is not None else ResultStore(cache_root() / "results")
        self.workers = workers if workers is not None else process_workers(self.profile)
        self._lock = threading.Lock()
        self._admissions: dict[str, tuple[Plan, dict[InputIdentity, object]]] = {}
        self._outcomes: dict[str, list[JobOutcome]] = {}
        self._status: dict[str, Status] = {}
        self._ids = itertools.count()

    logs_claims: ClassVar[bool] = False
    splits_fragments: ClassVar[bool] = False

    def bind_epoch(self, epoch: int, events: EventLog) -> None:
        """Do nothing: an in-process run has no queue for an older coordinator to write."""

    def admit(self, placed: Plan, values: Mapping[InputIdentity, object]) -> Admission:
        """Hold `placed` and the values of the graph-input identities its fragments name, until `release`.

        Each admission is its own: two admissions of one plan never share or replace each other's values.

        Raises:
            ValueError: the plan was placed under another profile, or resolved under another reuse policy.
        """
        if placed.placement.profile_digest != self.profile.canonical_digest():
            raise ValueError(f"plan {placed.hash} was placed under another execution profile than this executor's")
        if ReusePolicy(placed.logical.reuse) is not self.run.reuse:
            raise ValueError(f"plan {placed.hash} reuses under {placed.logical.reuse}, not {self.run.reuse.value}")
        admission = Admission(id=f"admission-{uuid.uuid4().hex}", plan_hash=placed.hash)
        with self._lock:
            self._admissions[admission.id] = (placed, dict(values))
        return admission

    def release(self, admission: Admission) -> None:
        """Let go of `admission` and its values; an admission of another plan, or not held, is ignored."""
        with self._lock:
            held = self._admissions.get(admission.id)
            if held is not None and held[0].hash == admission.plan_hash:
                del self._admissions[admission.id]

    def submit(self, fragment: PlanFragment) -> Handle:
        """Run every job of `fragment` and return its handle, whose status is then `done`.

        Raises:
            ValueError: the fragment's admission is not held or is of another plan, names a node the plan lacks or a
                graph input with no value, or its placement names an undeclared backend; and as `execute`.
            InconsistentInputs, AmbiguousClock: as `execute`.
            LicenseRefused, LeaseTimeout, LeaseUnavailable, WorkerLoadFailed, OutputTypeMismatch, WireFormatRetired,
                WorkerFatal, WorkerLost, ProtocolError: as `execute`.
        """
        jobs, placement = self._jobs(fragment)
        handle = Handle(id=f"fragment-{next(self._ids)}")
        with self._lock:
            self._status[handle.id] = Status.RUNNING
        try:
            outcomes = execute(
                jobs, placement, store=self.store, run=self.run, profile=self.profile, workers=self.workers
            )
        except BaseException:
            with self._lock:
                self._status.pop(handle.id)
            raise
        with self._lock:
            self._outcomes[handle.id] = outcomes
            self._status[handle.id] = Status.DONE
        return handle

    def status(self, handle: Handle) -> Status:
        """Return where the fragment of `handle` is: `done` once `submit` returned it.

        Raises:
            KeyError: the handle is not this executor's, or was forgotten.
        """
        with self._lock:
            return self._status[handle.id]

    def cancel(self, handle: Handle) -> None:
        """Do nothing: every job of a fragment has run once `submit` returns its handle."""

    def poll(self, handle: Handle) -> tuple[list[ResultRef], bool]:
        """Return every result of `handle`, in the fragment's order, and forget the handle (`forget`).

        Raises:
            KeyError: the handle is not this executor's, or was forgotten.
        """
        with self._lock:
            outcomes = self._outcomes.pop(handle.id)
            self._status.pop(handle.id, None)
        return [result_ref(o) for o in outcomes], True

    def outcomes(self, handle: Handle) -> list[JobOutcome]:
        """Return the outcome of each job of `handle`, results and their outputs included, in the fragment's order.

        Raises:
            KeyError: the handle is not this executor's, or was forgotten.
        """
        with self._lock:
            return list(self._outcomes[handle.id])

    def forget(self, handle: Handle) -> None:
        """Let go of the outcomes of `handle`."""
        with self._lock:
            self._outcomes.pop(handle.id, None)
            self._status.pop(handle.id, None)

    def call(self, call: TaskCall) -> Result:
        """Run `call`: plan it as one node under this executor's profile and run config, submit its one fragment.

        Each input is bound once (`bind_input`), its constraints checked, and the plan admitted with the bound
        values; the result is the fragment's one outcome.

        Raises:
            UnknownTask, InconsistentInputs: the task is not declared; the inputs contradict a constraint it declares.
            AmbiguousClock: two recording inputs of one content carry different anchors.
            UnlistedSource, NoBackend, NoDevice, BackendParamsMismatch: as `plan` places the node.
            LicenseRefused, LicenseConflict, MemoryCapExceeded: as `plan` checks the placement.
            MissingProfileSlice, ModelResolutionError: as `plan` resolves the node.
            ValueError: the call names another task version or other inputs than the task's; an input has no reader
                for its port's type or, for an isolated or server backend, no wire form; or a server backend got no
                model.
            WireFormatRetired, OutputTypeMismatch: a stored or computed output is not of its declared form.
            LeaseTimeout, LeaseUnavailable: the backend's server start lease is busy or cannot be read.
            WorkerLoadFailed: the backend's worker or server did not start or load (`VllmStartError` and
                `AttestationMismatch` among its causes).
            WorkerFatal, WorkerLost, ProtocolError: as the worker reports them.
            CallFailed: the backend failed the item.
        """
        spec = packaged().registry.spec(call.task)
        if call.task_version != spec.version:
            raise ValueError(f"{call.task!r} is declared at version {spec.version!r}, not {call.task_version!r}")
        if set(call.inputs) != set(spec.inputs):
            raise ValueError(f"{call.task!r} takes inputs {sorted(spec.inputs)}; the call gives {sorted(call.inputs)}")
        bound = {port: bind_input(port, spec.inputs[port], value)[:2] for port, value in call.inputs.items()}
        values = {port: value for port, (_, value) in bound.items()}
        spec.check_inputs(values)
        anchors_of(values.values())
        bound_call = TaskCall(
            task=call.task,
            task_version=call.task_version,
            inputs=values,
            params=call.params,
            model=call.model,
            backend_params=call.backend_params,
        )
        placed = plan(bound_call, config=self.run, execution=self.profile)
        add_plan_entity(self.store.provenance, placed.hash)
        admission = self.admit(placed, dict(bound.values()))
        try:
            handle = self.submit(node_fragment(placed, admission))
            [outcome] = self.outcomes(handle)
            self.forget(handle)
        finally:
            self.release(admission)
        if isinstance(outcome, Failed):
            raise CallFailed(outcome.failure)
        return outcome.result

    def _jobs(self, fragment: PlanFragment) -> tuple[list[Job], Placement]:
        """Return the jobs of `fragment` with their graph-input values and upstream inputs, and its placement."""
        return fragment_jobs(fragment, *self._held(fragment))

    def _held(self, fragment: PlanFragment) -> tuple[Plan, dict[InputIdentity, object]]:
        """Return the plan and graph-input values of the admission `fragment` names.

        Raises:
            ValueError: the admission is not held, or is of another plan.
        """
        with self._lock:
            if fragment.admission not in self._admissions:
                raise ValueError(f"admission {fragment.admission} is not held by this executor")
            placed, values = self._admissions[fragment.admission]
        if placed.hash != fragment.plan_hash:
            raise ValueError(f"admission {fragment.admission} is of plan {placed.hash}, not {fragment.plan_hash}")
        return placed, values


def fragment_jobs(
    fragment: PlanFragment, placed: Plan, values: Mapping[InputIdentity, object]
) -> tuple[list[Job], Placement]:
    """Return the jobs of `fragment` of `placed`, with their graph-input `values` and upstream inputs, and placement.

    Raises:
        ValueError: a job names an input no edge of its node feeds, or a graph input with no value.
    """
    registry = packaged().registry
    base = _node_call(placed, fragment.node, registry)
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
