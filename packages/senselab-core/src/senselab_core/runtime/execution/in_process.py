"""`InProcess`: the `Executor` that runs a fragment's jobs through `execute` in this process."""

import atexit
import itertools
import threading
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace

from senselab_core.kernel.declare import Registry, TaskSpec
from senselab_core.kernel.graph import Edge
from senselab_core.kernel.identity import InputIdentity, ReusePolicy
from senselab_core.kernel.placement import NodePlacement
from senselab_core.runtime.call import Result, RunConfig, TaskCall, bind_input, packaged
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
from senselab_core.runtime.execution.job import Completed, Failed, Job, JobOutcome, Upstream, execute
from senselab_core.runtime.execution.outputs import anchors_of
from senselab_core.runtime.execution.workers import WorkerSet
from senselab_core.runtime.keys import Call, Placement
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

    The first profile sets the bound; a later profile with a smaller bound tightens it (`WorkerSet.tighten`), and a
    larger one leaves it. The set is closed at exit.
    """
    bound = sum(c.max_workers for c in profile.resource_classes.values())
    with _PROCESS_LOCK:
        if not _PROCESS:
            _PROCESS.append(WorkerSet(bound))
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


def node_fragment(
    placed: Plan,
    admission: Admission,
    node: str | None = None,
    *,
    upstream: Mapping[str, Sequence[ResultRef]] | None = None,
) -> PlanFragment:
    """Return the fragment of every job of `node` (the plan's one node when None) under `admission`.

    A node fed by a graph input has one job per manifest record, its element the record's index when a `map` edge
    feeds it. A node fed only by upstream nodes has one job per element of their results, `upstream` giving each
    upstream node's results; an element reads the upstream result of the same element, or of the unmapped one.

    Raises:
        ValueError: `node` is None and the plan has more than one node; the admission is of another plan; an
            upstream node's results are not given, are missing an element or hold a failure; the manifest holds no
            record, or more than one for an unmapped node.
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
    rows = placed.manifest.rows
    if given:
        mapped = any(feeds[p].kind == "map" for p in given)
        if not rows or (not mapped and len(rows) > 1):
            raise ValueError(f"node {node!r} has {len(rows)} records; a fragment holds one job per record")
        elements: list[tuple[int, ...]] = [(i,) if mapped else () for i in range(len(rows))]
    else:
        elements = sorted({element for refs in results.values() for element in refs})
    inputs: list[dict[str, str]] = []
    read: dict[str, UpstreamRef] = {}
    for i, element in enumerate(elements):
        entry: dict[str, str] = {p: rows[i].inputs[g] for p, g in given.items()}
        for port, refs in results.items():
            ref = refs.get(element, refs.get(()))
            if ref is None or ref.output_identity is None:
                raise ValueError(f"node {node!r} element {element} has no upstream result for input {port!r}")
            stored = UpstreamRef(key_digest=ref.key_digest, outputs=ref.output_identity, port=_source(feeds[port])[1])
            entry[port] = stored.identity()
            read[stored.identity()] = stored
        inputs.append(entry)
    return PlanFragment(
        plan_hash=placed.hash,
        node=node,
        admission=admission.id,
        jobs=tuple(JobRef(plan_hash=placed.hash, node=node, element=e) for e in elements),
        inputs=tuple(inputs),
        upstream=read,
        placement=placed.placement.nodes[node],
    )


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


class InProcess:
    """An `Executor` whose `submit` runs a fragment's jobs through `execute` in this process, before returning.

    A fragment is submitted under an admission of its plan (`admit`), which holds the value of each graph-input
    identity its jobs name; an upstream input is read from the store.
    Jobs are sent in requests of the placement's batch size (one item per request for a backend that is not
    batch-invariant). `None` means the packaged default: the store under `cache_root()/results`, the `laptop`
    profile, `RunConfig()` and this process's worker set for the profile (`process_workers`).
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
        admission = Admission(id=f"admission-{next(self._ids)}", plan_hash=placed.hash)
        with self._lock:
            self._admissions[admission.id] = (placed, dict(values))
        return admission

    def release(self, admission: Admission) -> None:
        """Let go of `admission` and its values."""
        with self._lock:
            self._admissions.pop(admission.id, None)

    def submit(self, fragment: PlanFragment) -> Handle:
        """Run every job of `fragment` and return its handle, whose status is then `done`.

        Raises:
            ValueError: the fragment's admission is not held or is of another plan, names a node the plan lacks or a
                graph input with no value, or its placement names an undeclared backend; and as `execute`.
            InconsistentInputs, AmbiguousClock: as `execute`.
            LicenseRefused, VllmStartError, LeaseTimeout, LeaseUnavailable, AttestationMismatch, OutputTypeMismatch,
                WireFormatRetired, WorkerFatal, WorkerLost, ProtocolError: as `execute`.
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

    def results(self, handle: Handle) -> Iterator[ResultRef]:
        """Yield one result per job of `handle`, in the fragment's order.

        Raises:
            KeyError: the handle is not this executor's, or was forgotten.
        """
        for outcome in self.outcomes(handle):
            if isinstance(outcome, Completed):
                key = outcome.result.key.digest()
                yield ResultRef(job=outcome.ref, key_digest=key, output_identity=outcome.result.output_identity)
            else:
                yield ResultRef(job=outcome.ref, key_digest=outcome.key_digest, failure=outcome.failure)

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
            VllmStartError, LeaseTimeout, LeaseUnavailable, AttestationMismatch: the backend's server did not start.
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
        with self._lock:
            if fragment.admission not in self._admissions:
                raise ValueError(f"admission {fragment.admission} is not held by this executor")
            placed, values = self._admissions[fragment.admission]
        if placed.hash != fragment.plan_hash:
            raise ValueError(f"admission {fragment.admission} is of plan {placed.hash}, not {fragment.plan_hash}")
        nodes = {n.id: n for n in placed.logical.graph.nodes}
        if fragment.node not in nodes:
            raise ValueError(f"plan {fragment.plan_hash} has no node {fragment.node!r}")
        node, registry = nodes[fragment.node], packaged().registry
        spec = registry.spec(node.task)
        if node.task_version != spec.version:
            raise ValueError(f"{node.task!r} is declared at version {spec.version!r}, not {node.task_version!r}")
        params = TaskCall.parse_params(node.task, node.params, registry)
        base = Call(spec, params, {}, placed.logical.models[node.id], placed.logical.profile_slices[node.id])
        edges = placed.logical.graph.edges
        feeds = {e.target.split(".", 1)[1]: _source(e) for e in edges if e.target.split(".", 1)[0] == node.id}
        jobs = []
        for ref, identities in zip(fragment.jobs, fragment.inputs, strict=True):
            held, upstream = {}, {}
            for port, identity in identities.items():
                source, source_port = feeds[port]
                if source != "input":
                    stored = fragment.upstream[identity]
                    if stored.port != source_port:
                        raise ValueError(f"input {port!r} of {ref} reads port {source_port!r}, not {stored.port!r}")
                    upstream[port] = Upstream(stored, registry.spec(nodes[source].task).outputs[source_port])
                elif identity in values:
                    held[port] = values[identity]
                else:
                    raise ValueError(f"no value is admitted for input {port!r} of {ref}")
            call = replace(base, inputs=dict(identities))
            jobs.append(Job(ref, call, held, anchors_of(held.values()), upstream))
        return jobs, placement_of_record(fragment.placement, spec, registry)
