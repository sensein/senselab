"""`InProcess`: the `Executor` that runs a fragment's jobs through `execute` in this process."""

import atexit
import itertools
import threading
from collections.abc import Iterator, Mapping
from dataclasses import replace

from senselab_core.kernel.declare import Registry, TaskSpec
from senselab_core.kernel.identity import InputIdentity, ReusePolicy
from senselab_core.kernel.placement import NodePlacement
from senselab_core.runtime.call import Result, RunConfig, TaskCall, bind_input, packaged
from senselab_core.runtime.execution.executor import FaultRecord, Handle, JobRef, PlanFragment, ResultRef, Status
from senselab_core.runtime.execution.job import Completed, Failed, Job, JobOutcome, execute
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


_WORKER_SETS: dict[int, WorkerSet] = {}
_WORKER_SETS_LOCK = threading.Lock()


def process_workers(profile: ExecutionProfile) -> WorkerSet:
    """Return this process's worker set bounded by `profile`: the workers its resource classes keep alive at once.

    One set exists per bound; every set is closed at exit.
    """
    bound = sum(c.max_workers for c in profile.resource_classes.values())
    with _WORKER_SETS_LOCK:
        if bound not in _WORKER_SETS:
            _WORKER_SETS[bound] = WorkerSet(bound)
        return _WORKER_SETS[bound]


@atexit.register
def _close_worker_sets() -> None:
    for workers in _WORKER_SETS.values():
        workers.close()


def node_fragment(placed: Plan, node: str | None = None) -> PlanFragment:
    """Return the fragment of every job of `node` (the plan's one node when None), one per manifest record.

    The node's every input is a graph input; a `map` edge makes a job's element its record's index.

    Raises:
        ValueError: `node` is None and the plan has more than one node; an input of the node is another node's
            output; the manifest holds no record, or more than one for an unmapped node.
    """
    graph = placed.logical.graph
    if node is None:
        if len(graph.nodes) != 1:
            raise ValueError(f"the plan has nodes {sorted(n.id for n in graph.nodes)}; name the node")
        node = graph.nodes[0].id
    feeds = {e.target.split(".", 1)[1]: e for e in graph.edges if e.target.split(".", 1)[0] == node}
    if any(not e.source.startswith("input.") for e in feeds.values()):
        raise ValueError(f"node {node!r} reads another node's output; the plan runner builds its jobs")
    mapped = any(e.kind == "map" for e in feeds.values())
    rows = placed.manifest.rows
    if not rows or (not mapped and len(rows) > 1):
        raise ValueError(f"node {node!r} has {len(rows)} records; a fragment holds one job per record")
    return PlanFragment(
        plan_hash=placed.hash,
        node=node,
        jobs=tuple(JobRef(plan_hash=placed.hash, node=node, element=(i,) if mapped else ()) for i in range(len(rows))),
        inputs=tuple({port: row.inputs[e.source.split(".", 1)[1]] for port, e in feeds.items()} for row in rows),
        placement=placed.placement.nodes[node],
    )


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

    A fragment is submitted once its plan, and the value of each input identity its jobs name, are `admit`ted.
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
        self._plans: dict[str, tuple[Plan, dict[InputIdentity, object], int]] = {}
        self._outcomes: dict[str, list[JobOutcome]] = {}
        self._status: dict[str, Status] = {}
        self._ids = itertools.count()

    def admit(self, placed: Plan, values: Mapping[InputIdentity, object]) -> None:
        """Hold `placed` and the values of the input identities its fragments name, until as many `release`s.

        Raises:
            ValueError: the plan was placed under another profile, or resolved under another reuse policy.
        """
        if placed.placement.profile_digest != self.profile.canonical_digest():
            raise ValueError(f"plan {placed.hash} was placed under another execution profile than this executor's")
        if ReusePolicy(placed.logical.reuse) is not self.run.reuse:
            raise ValueError(f"plan {placed.hash} reuses under {placed.logical.reuse}, not {self.run.reuse.value}")
        with self._lock:
            _, held, count = self._plans.get(placed.hash, (placed, {}, 0))
            self._plans[placed.hash] = (placed, {**held, **values}, count + 1)

    def release(self, plan_hash: str) -> None:
        """Drop one `admit` of the plan `plan_hash`; its values are let go with the last."""
        with self._lock:
            placed, held, count = self._plans.pop(plan_hash)
            if count > 1:
                self._plans[plan_hash] = (placed, held, count - 1)

    def submit(self, fragment: PlanFragment) -> Handle:
        """Run every job of `fragment` and return its handle, whose status is then `done`.

        Raises:
            ValueError: the fragment's plan is not admitted, names a node it lacks or an input with no value, or its
                placement names an undeclared backend; and as `execute`.
            InconsistentInputs: a job's inputs contradict a constraint the task declares.
            AmbiguousClock: two recording inputs of one content carry different anchors.
            LicenseRefused, VllmStartError, LeaseTimeout, LeaseUnavailable, AttestationMismatch, OutputTypeMismatch,
                WireFormatRetired, WorkerFatal, WorkerLost, ProtocolError: as `execute`.
        """
        jobs, placement = self._jobs(fragment)
        handle = Handle(id=f"in-process-{next(self._ids)}")
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
        self.admit(placed, dict(bound.values()))
        try:
            handle = self.submit(node_fragment(placed))
            [outcome] = self.outcomes(handle)
            self.forget(handle)
        finally:
            self.release(placed.hash)
        if isinstance(outcome, Failed):
            raise CallFailed(outcome.failure)
        return outcome.result

    def _jobs(self, fragment: PlanFragment) -> tuple[list[Job], Placement]:
        """Return the jobs of `fragment` with their values, and its placement."""
        with self._lock:
            if fragment.plan_hash not in self._plans:
                raise ValueError(f"plan {fragment.plan_hash} is not admitted to this executor")
            placed, values, _ = self._plans[fragment.plan_hash]
        nodes = {n.id: n for n in placed.logical.graph.nodes}
        if fragment.node not in nodes:
            raise ValueError(f"plan {fragment.plan_hash} has no node {fragment.node!r}")
        node, registry = nodes[fragment.node], packaged().registry
        spec = registry.spec(node.task)
        if node.task_version != spec.version:
            raise ValueError(f"{node.task!r} is declared at version {spec.version!r}, not {node.task_version!r}")
        params = TaskCall.parse_params(node.task, node.params, registry)
        model, slices = placed.logical.models[node.id], placed.logical.profile_slices[node.id]
        base = Call(spec, params, {}, model, slices)
        jobs = []
        for ref in fragment.jobs:
            identities = fragment.inputs_of(ref)
            missing = sorted(port for port, identity in identities.items() if identity not in values)
            if missing:
                raise ValueError(f"no value is admitted for inputs {missing} of {ref}")
            held = {port: values[identity] for port, identity in identities.items()}
            spec.check_inputs(held)
            jobs.append(Job(ref, replace(base, inputs=dict(identities)), held, anchors_of(held.values())))
        return jobs, placement_of_record(fragment.placement, spec, registry)
