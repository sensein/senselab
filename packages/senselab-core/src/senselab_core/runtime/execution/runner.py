"""The plan runner: a placed plan compiled to a pydra workflow, its jobs run on a warm pool (`PydraRunner`).

`compile_plan` turns a plan into a pydra workflow: a `records` node split over the manifest's rows, one node per
graph node (a `map` edge reads the row, so every mapped node shares the one record axis, and a node with several
mapped upstreams is split over them zipped), and, for each node a `gather` edge reads, a copy of it combined over the
rows. Each node's checksum is the job's senselab key digest (`job_checksum`), so pydra hashes no senselab code.
The runner drives pydra's `GraphRunner` from outside (RFC 0002 §3.5) on one thread: each round's ready jobs become
pool jobs (`prepare`), submitted to the session's `WarmPool` in one `submit`; the pool's follower hands each outcome to
the driver thread through a queue, and the driver alone calls the `GraphRunner`. A job pydra settles as the twin of
another job of its key is sent again, to be served by the store as a reuse. The `GraphRunner`'s store is
`SenselabStore`, the run's completion records in memory; results live in the session's store. Each attempt has a
pydra root of its own, `<work dir>/<attempt>`.
"""

import contextlib
import queue
import threading
import typing
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from pydra.compose import python, workflow
from pydra.engine.job import Job
from pydra.engine.lease import Lease
from pydra.engine.result import Result
from pydra.engine.runner import Failed, GraphRunner, NotRun, Round
from pydra.engine.store import context_key
from pydra.engine.submitter import Submitter, WorkflowError
from pydra.utils.resources import ComputeResources
from pydra.workers.pool.pool import UnplaceableJob

from senselab_core.kernel.graph import Edge, fan_out_issues, mapped_nodes, order, split
from senselab_core.kernel.identity import InputIdentity, ReusePolicy, gather_identity, port_output_identity
from senselab_core.runtime.call import RunConfig, packaged
from senselab_core.runtime.execution.batching import BucketGrid, keyed_formation, recording_span_ns
from senselab_core.runtime.execution.events import TERMINAL, UNPLACEABLE, EventKind, EventLog, run_dir_of
from senselab_core.runtime.execution.executor import (
    Admission,
    FaultRecord,
    JobRef,
    PlanFragment,
    QueueHeld,
    ResultRef,
    RunCancelled,
)
from senselab_core.runtime.execution.fragments import fragment_jobs, job_checksum, node_fragment, plan_keys
from senselab_core.runtime.execution.job import code_license, job_context, resolved, stale, used
from senselab_core.runtime.execution.outputs import decoded
from senselab_core.runtime.execution.portable import OUTPUTS, JobDoc, read_outputs
from senselab_core.runtime.execution.providers import Loader
from senselab_core.runtime.execution.run import PYDRA_DIR
from senselab_core.runtime.execution.warm import PoolSession, item_failure
from senselab_core.runtime.execution.workers import service_of
from senselab_core.runtime.keys import build_key, request_components
from senselab_core.runtime.licenses import LicenseRefused
from senselab_core.runtime.planner import Plan, check_mapped_outputs
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.revision import attempt_scope
from senselab_core.runtime.storage import PRIVATE_DIR, private_dir
from senselab_core.runtime.store import output_identity

RECORDS = "records"
"""The pydra node split over the manifest's rows; every mapped node reads its row from it."""
RECORD_FIELD = "row"
"""The records node's input, a row index; `records.row` is the split every gather combines over."""
REF = "ref"
"""The one output of every compiled node: the job's `ResultRef` as JSON (the row index, for the records node)."""


class ElementFailed(RuntimeError):
    """An element the executor failed; `failure` is its record."""

    def __init__(self, failure: FaultRecord) -> None:
        """Hold the failure and name it in the message."""
        super().__init__(f"{failure.type} ({failure.fault}): {failure.message}")
        self.failure = failure


@dataclass(frozen=True)
class RunOutcome:
    """A finished run: each node's results in element order, and the elements a failed upstream element blocked."""

    plan_hash: str
    results: Mapping[str, tuple[ResultRef, ...]]
    blocked: Mapping[str, tuple[tuple[int, ...], ...]]

    @property
    def done(self) -> bool:
        """Whether every job of the plan completed."""
        failed = any(r.failure is not None for refs in self.results.values() for r in refs)
        return not failed and not any(self.blocked.values())


class _EpochLease:
    """pydra's lease of a job, held without a lock file: the run's records are its own, in memory."""

    def __init__(self, checksum: str, cache_dir: Path, lockfile: Path) -> None:
        """Name the job `checksum` whose directory is `cache_dir` and whose (unwritten) lock is `lockfile`."""
        self.checksum = checksum
        self.cache_dir = cache_dir
        self.lockfile = lockfile

    def __enter__(self) -> "_EpochLease":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    async def __aenter__(self) -> "_EpochLease":
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    def try_take(self) -> float | None:
        """Return None: the lease is held at once, as entering it is."""
        return None


class SenselabStore:
    """pydra's `ResultStore` over one run: completion records in memory by checksum and context slot, never pickled.

    It holds what pydra reads to resolve a downstream input and to mark a job done; `lookup` returns a record of its
    own context's slot only. `cache_root` holds only the workflow job's own files; a lease writes nothing there.
    """

    def __init__(self, cache_root: Path) -> None:
        """Hold the run's records under `cache_root`."""
        self.cache_root = Path(cache_root)
        self._records: dict[str, dict[str, Result]] = {}

    def __getstate__(self) -> dict[str, Any]:
        """Return the state a pickled job carries: the root, never the records."""
        return {"cache_root": self.cache_root, "_records": {}}

    def cache_dir(self, checksum: str, context: Mapping[str, Any]) -> Path:
        """Return the directory of the job `checksum` in `context`."""
        return self.cache_root / checksum / str(context_key(context))

    def lookup(
        self,
        checksum: str,
        context: Mapping[str, Any],
        task: object,  # noqa: ARG002
        stored: list[tuple[Path, Mapping[str, Any]]] | None = None,  # noqa: ARG002
    ) -> Result | None:
        """Return the record of `checksum` in `context`'s slot, or None."""
        return self._records.get(checksum, {}).get(context_key(context))

    def stored_contexts(self, checksum: str) -> Iterator[tuple[Path, Mapping[str, Any]]]:
        """Yield the slot and context of each record of `checksum`."""
        for slot, result in sorted(self._records.get(checksum, {}).items()):
            yield self.cache_root / checksum / slot, result.context

    def lockfile(self, checksum: str, context: Mapping[str, Any]) -> Path:
        """Return the name pydra's `ResultStore` gives the lock of job `checksum` in `context`; nothing writes it."""
        return self.cache_root / f"{checksum}-{context_key(context)}.lock"

    def acquire(self, checksum: str, context: Mapping[str, Any]) -> Lease:
        """Return the lease of the job `checksum` in `context`: one that takes no lock (`_EpochLease`)."""
        return _EpochLease(checksum, self.cache_dir(checksum, context), self.lockfile(checksum, context))

    def commit(self, lease: Lease, result: Result, job: object = None) -> None:  # noqa: ARG002
        """Record `result` as the job's held under `lease`."""
        self.record(lease.checksum, result)

    def record(self, checksum: str, result: Result) -> None:
        """Record `result` for `checksum` in its own context's slot."""
        self._records.setdefault(checksum, {})[context_key(result.context)] = result

    def checksums(self) -> list[str]:
        """Return the checksum of every job recorded, sorted."""
        return sorted(self._records)


@dataclass(frozen=True)
class _Feed:
    """How one input port of a compiled node is fed: the graph edge and its source node (`input` for a graph input)."""

    port: str
    edge: Edge
    source: str
    source_port: str


def _feeds(placed: Plan, node: str) -> list[_Feed]:
    return [
        _Feed(split(e.target)[1], e, *split(e.source)) for e in placed.logical.graph.edges if split(e.target)[0] == node
    ]


def _ref(value: object) -> ResultRef:
    return ResultRef.model_validate(value)


def _at(ref: ResultRef, node: str, element: tuple[int, ...]) -> ResultRef:
    """Return `ref` as the result of the job of `node` at `element`."""
    return ref.model_copy(update={"job": JobRef(plan_hash=ref.job.plan_hash, node=node, element=element)})


def _refs(value: object, source: str) -> list[ResultRef]:
    """Return a gathered list of `source`'s results, each held at its position, the record it combines."""
    return [_at(_ref(v), source, (i,)) for i, v in enumerate(typing.cast(Sequence[object], value))]


class _NodeKey:
    """The checksum of a compiled node's job (`job_checksum`), from the identities its inputs resolve to."""

    def __init__(self, placed: Plan, node: str) -> None:
        """Key jobs of `node` of `placed`."""
        self.placed = placed
        self.node = node

    def identities(self, inputs: Mapping[str, object]) -> dict[str, InputIdentity]:
        """Return the input identity of each port, as `node_fragment` names it, from the job's resolved inputs."""
        rows = self.placed.manifest.rows
        found: dict[str, InputIdentity] = {}
        for feed in _feeds(self.placed, self.node):
            value = inputs[feed.port]
            if feed.source == "input":
                row = typing.cast(Mapping[str, int], value)[RECORD_FIELD] if feed.edge.kind == "map" else 0
                found[feed.port] = rows[row].inputs[feed.source_port]
            elif feed.edge.kind == "gather":
                ports = [port_output_identity(r.key_digest, feed.source_port) for r in _refs(value, feed.source)]
                found[feed.port] = gather_identity(ports)
            else:
                found[feed.port] = port_output_identity(_ref(value).key_digest, feed.source_port)
        return found

    def checksum(self, inputs: Mapping[str, object]) -> str:
        """Return the job's `job_checksum`."""
        return job_checksum(self.placed, self.node, self.identities(inputs))


def _node_checksum(key: _NodeKey) -> Callable[[object, Mapping[str, object]], str]:
    def checksum(task: object, inputs: Mapping[str, object]) -> str:  # noqa: ARG001
        return key.checksum(inputs)

    return checksum


def _records_key(plan_hash: str) -> Callable[[object, Mapping[str, object]], str]:
    def checksum(task: object, inputs: Mapping[str, object]) -> str:  # noqa: ARG001
        return f"{RECORDS}-{plan_hash}-{inputs[RECORD_FIELD]}"

    return checksum


def _never(**inputs: object) -> object:
    """Stand in for a compiled node's function, which the runner never calls: its executor runs the job."""
    raise RuntimeError(f"a compiled node is run by its executor, not called; got {sorted(inputs)}")


def _record(row: int) -> object:
    """Stand in for the records node's function, whose result the runner gives itself."""
    return {RECORD_FIELD: row}


def _node_class(placed: Plan, node: str) -> Any:  # noqa: ANN401 - a pydra task class
    ports = {f.port: Any for f in _feeds(placed, node)}
    cls = python.define(
        _never, inputs=ports, outputs={REF: Any}, name=f"n_{node}", checksum=_node_checksum(_NodeKey(placed, node))
    )
    cls.senselab_node = node
    return cls


def compile_plan(placed: Plan) -> Any:  # noqa: ANN401 - a pydra workflow task
    """Return `placed` as a pydra workflow task, every node's checksum the senselab key digest of its job.

    A `map` edge from a graph input reads the `records` node's row; a broadcast graph input is the one identity every
    row holds; a node with several mapped upstreams (the records included) is split over them zipped; a `gather` edge
    reads a copy of its source combined over `records.row`. Each node carries its resource class as a routing tag.

    Raises:
        ValueError: the graph has a nested fan-out (`fan_out_issues`).
        UnmappedOutput: a `map` edge reads a node's output (`check_mapped_outputs`).
    """
    graph = placed.logical.graph
    refused = fan_out_issues(graph)
    if refused:
        raise ValueError("; ".join(i.message for i in refused))
    check_mapped_outputs(graph)
    mapped = mapped_nodes(placed.logical.graph)
    classes = {n.id: _node_class(placed, n.id) for n in graph.nodes}
    records = python.define(
        _record, inputs={RECORD_FIELD: Any}, outputs={REF: Any}, name=RECORDS, checksum=_records_key(placed.hash)
    )
    records.senselab_node = None
    gathered = {split(e.source)[0] for e in graph.edges if e.kind == "gather"}
    ordered = order(graph)

    def construct(rows: list[int]) -> Any:  # noqa: ANN401 - the lazy outputs pydra reads
        added: dict[str, Any] = {}
        if mapped:
            added[RECORDS] = workflow.add(records().split(row=rows), name=RECORDS)
        outputs = []
        for node in ordered:
            resources = ComputeResources(tags=(placed.placement.nodes[node].resource_class,))
            added[f"n_{node}"] = workflow.add(
                _wired(placed, node, classes, added, mapped), name=f"n_{node}", resources=resources
            )
            if node in gathered:
                combined = _wired(placed, node, classes, added, mapped).combine(f"{RECORDS}.{RECORD_FIELD}")
                added[f"g_{node}"] = workflow.add(combined, name=f"g_{node}", resources=resources)
            outputs.append(getattr(added[f"n_{node}"], REF))
        return tuple(outputs)

    names = {f"n_{node}": Any for node in ordered}
    plan_check = f"plan-{placed.hash}"
    return workflow.define(
        construct, inputs={"rows": list[int]}, outputs=names, name="plan", checksum=lambda t, i: plan_check
    )(rows=list(range(len(placed.manifest.rows))))


def _wired(placed: Plan, node: str, classes: Mapping[str, Any], added: Mapping[str, Any], mapped: set[str]) -> Any:  # noqa: ANN401 - a pydra task
    """Return a task of `node`, each input wired to its lazy source, split over its mapped upstreams zipped."""
    values: dict[str, Any] = {}
    upstreams: list[str] = []
    rows = placed.manifest.rows
    for feed in _feeds(placed, node):
        if feed.source == "input" and feed.edge.kind == "map":
            values[feed.port] = getattr(added[RECORDS], REF)
            upstreams.append(RECORDS)
        elif feed.source == "input":
            values[feed.port] = rows[0].inputs[feed.source_port] if rows else None
        elif feed.edge.kind == "gather":
            values[feed.port] = getattr(added[f"g_{feed.source}"], REF)
        else:
            values[feed.port] = getattr(added[f"n_{feed.source}"], REF)
            if feed.source in mapped:
                upstreams.append(f"n_{feed.source}")
    task = classes[node](**values)
    states = sorted(set(upstreams))
    return task.split(tuple(f"_{u}" for u in states)) if len(states) > 1 else task


def _senselab(job: Job) -> tuple[str | None, tuple[int, ...]]:
    """Return the senselab node of a pydra job (None for a `records` job) and its element: the row, or `()`."""
    node = getattr(type(job.task), "senselab_node", None)
    return node, () if job.state_index is None else (int(job.state_index),)


def _label(job_name: str, state_index: object) -> str:
    """Return a job's name with its state index, as an error entry names it."""
    return job_name if state_index is None else f"{job_name}({state_index})"


@dataclass(eq=False)
class _Batch:
    """The jobs of one node sent in one round: the pydra job of each element; `served` for jobs pydra settled as twins.

    A twin's result is senselab's reuse of another job's: it is logged and held, and never reported to pydra.
    """

    node: str
    jobs: dict[tuple[int, ...], Job]
    served: bool = False
    given: set[tuple[int, ...]] = field(default_factory=set)


Message = tuple[str, object, object]
"""What the pool's follower hands the driver: (`outcome`, the pool job, its outcome)."""


@dataclass(frozen=True)
class _Sent:
    """A pool job in flight: its batch, element, key digest, and the senselab job it runs."""

    batch: _Batch
    element: tuple[int, ...]
    doc: JobDoc


class _Driver:
    """One run: the `GraphRunner` of a placed plan driven on this thread (RFC 0002 §3.5), its rounds sent to a pool.

    Every `GraphRunner` call is made here; the pool's follower only posts outcomes to `inbox`. A pool result is a pydra
    `Result` holding the job's `ResultRef`, or `Failed(ElementFailed)`. A node whose jobs cannot be formed is, for each
    of its jobs: `Failed` for a license refusal or any other exception (the latter raised after the run). A held job
    is `NotRun(held=True)`, and the run raises the hold after the loop; a cancelled one is `NotRun`.
    """

    def __init__(  # noqa: PLR0913 - the run's parts
        self,
        placed: Plan,
        values: Mapping[InputIdentity, object],
        session: PoolSession,
        run: RunConfig,
        store: "SenselabStore",
        events: EventLog,
        cancel: threading.Event,
        cancel_check_s: float,
    ) -> None:
        """Run `placed` on `session`'s pool under `run`, its graph inputs `values`, logging to `events`."""
        self.placed, self.values, self.session, self.run = placed, values, session, run
        self.store, self.events, self.cancel, self.cancel_check_s = store, events, cancel, cancel_check_s
        self.inbox: queue.Queue[Message] = queue.Queue()
        self.results: dict[str, dict[tuple[int, ...], ResultRef]] = {}
        self.ended: set[tuple[str, tuple[int, ...]]] = set()
        self.errors: list[BaseException] = []
        self.sent: dict[object, _Sent] = {}
        self.handles: list[Any] = []
        self.refs: dict[Job, ResultRef] = {}
        self.first: dict[str, ResultRef] = {}
        self.twins: dict[str, dict[tuple[int, ...], Job]] = {}
        self.reported: set[str] = set()
        self.held: list[str] = []
        self.cancelled = False
        self.completed = 0

    def drive(self, task: Any) -> None:  # noqa: ANN401 - a pydra workflow task
        """Drive `task`'s `GraphRunner` until no job can progress and no pool job is open.

        Raises:
            Exception: the first error of the run that is no job's failure, or a cancellation.
            QueueHeld: the pool held jobs of the run; they have no terminal event.
            WorkflowError: pydra errored a job no pool job failed.
            RuntimeError: nothing can progress, but the run is not done.
        """
        pool = self.session.pool
        pool.bind(self._report)
        with Submitter(worker="debug", store=self.store, telemetry=False) as submitter:
            graph = GraphRunner(
                task, self.store, on_error="raise_at_end", lookup_before_dispatch=False, submitter=submitter
            )
            try:
                self._loop(graph)
                foreign = [e for e in graph.entries if _label(e.job_name, e.state_index) not in self.reported]
            finally:
                if self.sent:
                    for handle in self.handles:
                        with contextlib.suppress(Exception):
                            pool.cancel(handle)
                pool.unbind(self._report)
                graph.close()
        if self.errors:
            raise self.errors[0]
        if self.held:
            raise QueueHeld("pool", "pool", len(self.held), self.held[0])
        if foreign:
            listed = "\n".join(f"{_label(e.job_name, e.state_index)} ({e.phase}): {e.message}" for e in foreign)
            raise WorkflowError(
                f"pydra errored jobs no pool job failed:\n{listed}", [e.message for e in foreign], foreign
            )

    def _report(self, job: object, outcome: object) -> None:
        """The pool's report callable, called on its follower thread: hand the outcome to the driver."""
        self.inbox.put(("outcome", job, outcome))

    def _loop(self, graph: GraphRunner) -> None:
        while True:
            self._check_cancel()
            rnd = graph.ready()
            progressed = self._settle_round(rnd)
            progressed = self._dispatch(graph, rnd.dispatch) or progressed
            if progressed:
                continue
            if not self.sent:
                if graph.done:
                    return
                if rnd.retry_after is None:
                    raise RuntimeError(graph.stuck() or "nothing can progress, but the run is not done")
            wait = self.cancel_check_s if rnd.retry_after is None else min(rnd.retry_after, self.cancel_check_s)
            self._receive(graph, wait)

    def _check_cancel(self) -> None:
        """Once `cancel` is set, record the run's cancellation and withdraw its jobs still open in the pool."""
        if self.cancel.is_set() and not self.cancelled:
            self.cancelled = True
            self._record(RunCancelled(f"run {self.placed.hash} was cancelled"))
            for handle in self.handles:
                with contextlib.suppress(Exception):
                    self.session.pool.cancel(handle)

    def _settle_round(self, rnd: Round) -> bool:
        """Take the jobs pydra settled as twins: a failed one fails, a served one is sent with the round."""
        for settled in rnd.settled:
            if settled.how != "shared":
                raise RuntimeError(f"pydra settled {settled.job.name} as {settled.how}; the run's store serves none")
            self._shared(settled.job)
        return bool(rnd.settled or rnd.unrunnable)

    def _shared(self, job: Job) -> None:
        node, element = _senselab(job)
        if node is None:
            return
        ref = _at(self.first[job.checksum], node, element)
        self.refs[job] = ref
        if element in self.results.get(node, {}) or element in self.twins.get(node, {}):
            return  # the same senselab job: a gather's copy of its node
        if ref.failure is not None:
            self.settle(node, ref)
        else:
            self.twins.setdefault(node, {})[element] = job

    def _dispatch(self, graph: GraphRunner, jobs: Sequence[Job]) -> bool:
        """Send `jobs` and the twins to serve as one round to the pool; return whether any job was settled here."""
        grouped: dict[str, dict[tuple[int, ...], Job]] = {}
        settled = False
        for job in jobs:
            node, element = _senselab(job)
            if node is None:
                graph.complete(job, self._row(job))
            elif self._stopped():
                graph.complete(job, NotRun("the run was stopped before it was sent", held=bool(self.held)))
            else:
                grouped.setdefault(node, {}).setdefault(element, job)
                continue
            settled = True
        before = self.completed
        twins, self.twins = self.twins, {}
        sends = [_Batch(n, js) for n, js in grouped.items()]
        sends += [] if self._stopped() else [_Batch(n, js, served=True) for n, js in twins.items()]
        pool_jobs: list[object] = []
        for batch in sends:
            pool_jobs += self._formed(graph, batch)
        if pool_jobs:
            self.handles.append(self.session.pool.submit(pool_jobs))
        return settled or self.completed != before

    def _formed(self, graph: GraphRunner, batch: _Batch) -> list[object]:
        """Return the pool jobs of `batch`, each logged `started`; a job that cannot be formed is settled here."""
        try:
            fragment = node_fragment(
                self.placed,
                ADMISSION.model_copy(update={"plan_hash": self.placed.hash}),
                batch.node,
                upstream=self._upstream(graph, batch),
                elements=list(batch.jobs),
            )
            prepared = prepare(fragment, self.placed, self.values, self.run, self.session, self.events.attempt)
        except Exception as error:  # noqa: BLE001 - the batch's refusal
            self._unsent(graph, batch, list(batch.jobs), error)
            return []
        formed = []
        for (element, job), made in zip(batch.jobs.items(), prepared, strict=True):
            if isinstance(made, FaultRecord):
                self._given(
                    graph,
                    batch,
                    ResultRef(job=self._job_ref(batch.node, element), key_digest=job.checksum, failure=made),
                )
                continue
            doc, pool_job = made
            self.log("started", batch.node, element, key=doc.key.digest())
            self.sent[pool_job] = _Sent(batch, element, doc)
            formed.append(pool_job)
        return formed

    def _row(self, job: Job) -> Result:
        """Return the result of a `records` job: its row."""
        outputs = {REF: {RECORD_FIELD: int(job.task.row)}}
        return self._result(job, outputs)

    def _result(self, job: Job, outputs: Mapping[str, object]) -> Result:
        result = Result(
            outputs=job.task.Outputs(**outputs),
            runtime=None,
            errored=False,
            cache_dir=self.store.cache_dir(job.checksum, job.context),
            task=job.task,
            context=job.context,
        )
        self.store.record(job.checksum, result)
        return result

    def _stopped(self) -> bool:
        """Whether no job is sent any more: the run is cancelled, or the pool held its jobs."""
        return self.cancel.is_set() or bool(self.held)

    def _record(self, error: Exception) -> None:
        """Hold `error` as one of the run's, a cancellation once."""
        cancelled = isinstance(error, RunCancelled) and any(isinstance(e, RunCancelled) for e in self.errors)
        if not cancelled and not any(e is error for e in self.errors):
            self.errors.append(error)

    def _upstream(self, graph: GraphRunner, batch: _Batch) -> dict[str, list[ResultRef]]:
        """Return each upstream node's results the jobs of `batch` read (`GraphRunner.upstream`), by element.

        A node read by a gather over no element is given no results.
        """
        found: dict[str, dict[tuple[int, ...], ResultRef]] = {
            f.source: {} for f in _feeds(self.placed, batch.node) if f.source != "input"
        }
        for job in batch.jobs.values():
            for reads in graph.upstream(job).values():
                for source_job, _ in reads:
                    source = _senselab(source_job)[0]
                    if source is not None:
                        ref = self.refs[source_job]
                        found.setdefault(source, {})[ref.job.element] = ref
        return {source: list(refs.values()) for source, refs in found.items()}

    def _receive(self, graph: GraphRunner, timeout: float | None) -> None:
        """Apply the next outcome, waiting up to `timeout`, then every outcome waiting."""
        try:
            message = self.inbox.get(timeout=timeout)
        except queue.Empty:
            return
        while True:
            self._apply(graph, message)
            try:
                message = self.inbox.get_nowait()
            except queue.Empty:
                return

    def _apply(self, graph: GraphRunner, message: Message) -> None:
        _, pool_job, outcome = message
        sent = self.sent.pop(pool_job, None)
        if sent is None:
            return
        if isinstance(outcome, NotRun):
            self._not_run(graph, sent, outcome)
            return
        self._given(graph, sent.batch, ref_of(sent, outcome))

    def _not_run(self, graph: GraphRunner, sent: _Sent, outcome: NotRun) -> None:
        """Give a job the pool did not run its outcome: a hold (raised after the run) or a cancellation."""
        if outcome.held:
            self.held.append(str(outcome.reason))
        job = sent.batch.jobs[sent.element]
        if not sent.batch.served:
            graph.complete(job, NotRun(str(outcome.reason), held=outcome.held))

    def _given(self, graph: GraphRunner, batch: _Batch, ref: ResultRef) -> None:
        """Settle an element's result; report it to pydra unless pydra settled the job as a twin."""
        self.completed += 1
        element = ref.job.element
        job = batch.jobs[element]
        batch.given.add(element)
        self.settle(batch.node, ref)
        self.first.setdefault(job.checksum, ref)
        if batch.served:
            return
        self.refs[job] = ref
        if ref.failure is not None:
            self.reported.add(_label(job.name, job.state_index))
            graph.complete(job, Failed(ElementFailed(ref.failure)))
            return
        try:
            result = self._result(job, {REF: ref.model_dump(mode="json")})
        except Exception as error:  # noqa: BLE001 - pydra's error of the job, which the run raises
            graph.complete(job, Failed(error))
            return
        graph.complete(job, result)

    def _unsent(
        self, graph: GraphRunner, batch: _Batch, elements: Sequence[tuple[int, ...]], error: BaseException
    ) -> None:
        """Give each job of `elements` its outcome for `error`, which kept it from the pool (`_Driver`)."""
        if not isinstance(error, Exception):
            raise error
        failure = FaultRecord.from_exception(error)
        self.completed += 1
        if not isinstance(error, LicenseRefused):
            self._record(error)
        for element in elements:
            job = batch.jobs[element]
            ref = ResultRef(job=self._job_ref(batch.node, element), key_digest=job.checksum, failure=failure)
            self.settle(batch.node, ref)
            self.first.setdefault(job.checksum, ref)
            self.refs[job] = ref
            self.reported.add(_label(job.name, job.state_index))
            if not batch.served:
                graph.complete(job, Failed(error))

    def _job_ref(self, node: str, element: tuple[int, ...]) -> JobRef:
        return JobRef(plan_hash=self.placed.hash, node=node, element=element)

    def settle(self, node: str, ref: ResultRef) -> None:
        """Hold `ref` as its element's result and log it `failed`, `cache_hit` or `finished`."""
        self.results.setdefault(node, {})[ref.job.element] = ref
        kind: EventKind = "failed" if ref.failure is not None else "cache_hit" if ref.served else "finished"
        self.log(
            kind,
            node,
            ref.job.element,
            key=ref.key_digest,
            failure=ref.failure,
            output=ref.output_identity,
            reuse=ref.served,
        )

    def expected(self) -> list[tuple[str, tuple[int, ...]]]:
        """Return every job of the plan as (node, element): a mapped node's per record, any other once."""
        graph = self.placed.logical.graph
        mapped = mapped_nodes(graph)
        rows = len(self.placed.manifest.rows)
        return [(n, (i,)) for n in order(graph) if n in mapped for i in range(rows)] + [
            (n, ()) for n in order(graph) if n not in mapped
        ]

    def close(self) -> None:
        """Give every job with no terminal event its own: `cancelled` once cancelled, else `blocked`.

        A run whose jobs the pool held leaves them without one, for a later run.
        """
        if self.held:
            return
        cancelled = any(isinstance(e, RunCancelled) for e in self.errors)
        for node, element in self.expected():
            if (node, element) not in self.ended:
                self.log("cancelled" if cancelled else "blocked", node, element)

    def log(  # noqa: PLR0913 - the fields of an event
        self,
        kind: EventKind,
        node: str,
        element: tuple[int, ...],
        *,
        key: str | None = None,
        failure: FaultRecord | None = None,
        output: str | None = None,
        reuse: str | None = None,
    ) -> None:
        """Append the event of `kind` for the job of `node` at `element`, with the node's placement and env."""
        placement = self.placed.placement.nodes[node]
        if kind in TERMINAL:
            self.ended.add((node, element))
        self.events.record(
            kind,
            node,
            job=self._job_ref(node, element),
            key=key,
            placement_digest=placement.canonical_digest(),
            failure=failure,
            env=placement.env,
            output=output,
            reuse=reuse,
        )


ADMISSION = Admission(id="run", plan_hash="0" * 64)
"""The admission a run's fragments name: the run holds its graph-input values itself."""


def ref_of(sent: _Sent, outcome: object) -> ResultRef:
    """Return the `ResultRef` of a pool job's outcome: its output identity and lengths, or its failure."""
    doc = sent.doc
    ref, key = doc.ref, doc.key.digest()
    if isinstance(outcome, Failed):
        return ResultRef(job=ref, key_digest=key, failure=pool_failure(outcome.error))
    assert isinstance(outcome, Result)
    if outcome.errored:
        errors = outcome.errors or {}
        text = "\n".join(errors.get("error message", [])) if isinstance(errors, dict) else str(errors)
        last = outcome.failure or " ".join(text.strip().splitlines()[-1:])
        failure = item_failure(text) or FaultRecord.from_pool({"kind": "content", "message": last}, "TaskError")
        return ResultRef(job=ref, key_digest=key, failure=failure)
    outputs = dict(getattr(outcome.outputs, OUTPUTS))
    read = decoded(doc.job.call.task, outputs)
    hit = outcome.committed is None or outcome.committed.served
    served = (doc.run.reuse.value if outcome.cross_context else ReusePolicy.EXACT.value) if hit else None
    return ResultRef(
        job=ref,
        key_digest=key,
        output_identity=output_identity(outputs),
        served=served,
        lengths_ns={p: n for p, v in read.items() if (n := recording_span_ns(v)) is not None},
    )


def pool_failure(error: BaseException) -> FaultRecord:
    """Return the senselab fault record of a job the pool failed: from the pool's own record when it carries one."""
    fault = getattr(error, "fault", None)
    if isinstance(fault, Mapping):
        carried = item_failure(str(fault.get("message", "")))
        return carried if carried is not None else FaultRecord.from_pool(fault, type(error).__name__)
    if isinstance(error, UnplaceableJob):
        return FaultRecord.from_pool({"kind": "resource", "message": str(error)}, UNPLACEABLE)
    return FaultRecord.from_exception(error)


def prepare(  # noqa: PLR0913 - the fragment and the run it is of
    fragment: PlanFragment,
    placed: Plan,
    values: Mapping[InputIdentity, object],
    run: RunConfig,
    session: PoolSession,
    attempt: str,
) -> list[tuple[JobDoc, object] | FaultRecord]:
    """Return, per job of `fragment`, its `JobDoc` and pool job, or the failure that keeps it from the pool.

    The backend's env is opened and its license checked once; each job's upstream inputs are read from the session's
    store, its stamps checked, its formation picked by its length and its key and lookup context built.

    Raises:
        LicenseRefused: the run's policy refuses the backend's env or the model.
        ValueError: as `fragment_jobs`.
    """
    profile = session.profile
    jobs, placement = fragment_jobs(fragment, placed, values)
    backend, call = placement.backend, jobs[0].call
    licensed = code_license(run, placement, call)
    env = packaged().open_env(backend, profile)
    expected = service_of(backend, call.model)
    grid = BucketGrid.load_default()
    store = session.store
    made: list[tuple[JobDoc, object] | FaultRecord] = []
    for given in jobs:
        job, unread = resolved(
            replace(given, call=replace(given.call, service=expected)), lambda k, o: read_outputs(store, k, o)
        )
        picked, unrecorded = keyed_formation(job, backend, grid)
        keyed = replace(placement, **request_components(backend, picked))
        gap = None if unrecorded is None else FaultRecord.gap_of(type(unrecorded).__name__, str(unrecorded))
        failure = unread or gap or stale(job)
        if failure is not None:
            made.append(failure)
            continue
        context = job_context(keyed, env, expected, profile, job, licensed, declared=True)
        doc = JobDoc(
            job=job,
            placement=keyed,
            env=env,
            expected=expected,
            profile=profile,
            code_license=licensed,
            key=build_key(job.call, keyed, context),
            context=context,
            run=run,
            used=used(job),
            attempt=attempt,
        )
        loader = Loader(backend, env, call.model, keyed.device, profile, expected)
        made.append((doc, session.job(doc, loader, f"{fragment.node}-{doc.key.digest()[:12]}")))
    return made


class PydraRunner:
    """Runs placed plans on a pool session: pydra's `GraphRunner` expands the graph and releases its jobs.

    `cancel_check_s` is how often the driver looks at a set cancel while it waits on the pool (the profile's).
    """

    def __init__(self, *, profile: ExecutionProfile | None = None, work_dir: Path | None = None) -> None:
        """Run under `profile`, pydra's files under `work_dir` (`<run dir>/private/pydra`)."""
        self.profile = profile if profile is not None else ExecutionProfile.load()
        self.work_dir = work_dir

    def run_plan(  # noqa: PLR0913 - the plan and its session, then keyword-only run inputs
        self,
        placed: Plan,
        session: PoolSession,
        *,
        values: Mapping[InputIdentity, object],
        events: EventLog,
        store: SenselabStore | None = None,
        cancel: threading.Event | None = None,
    ) -> RunOutcome:
        """Run every job of `placed` on `session`'s pool, its graph inputs `values`, and return what became of each.

        Each job is first logged `queued` with its key under this plan (`plan_keys`).

        A failed element fails only its descendants, which are reported `blocked`; every other element runs. A job
        sharing its key with another job of the run is that job's reuse, or its failure. A license refusal fails each
        job of its node and is not raised. Each job's `queued`, `started` and `finished`, `cache_hit` or `failed`
        event is appended to `events`. Once `cancel` is set no further job is sent, open jobs are withdrawn from the
        pool and the run raises `RunCancelled`. While it runs, `attempt_id()` is the attempt of `events`.

        Raises:
            ValueError: as `compile_plan`, or the plan was placed under another profile than the session's.
            RunCancelled: `cancel` was set before every job ran.
            QueueHeld: the pool held jobs of the run; they have no terminal event.
            WorkflowError: pydra errored a job no pool job failed.
            Exception: the first other error forming a node's jobs raised, after the run.
        """
        if placed.placement.profile_digest != session.profile.canonical_digest():
            raise ValueError(f"plan {placed.hash} was placed under another execution profile than the session's")
        with attempt_scope(events.attempt):
            return self._run_plan(placed, session, values, events, store, cancel)

    def _run_plan(  # noqa: PLR0913 - as `run_plan`
        self,
        placed: Plan,
        session: PoolSession,
        values: Mapping[InputIdentity, object],
        events: EventLog,
        store: SenselabStore | None,
        cancel: threading.Event | None,
    ) -> RunOutcome:
        if self.work_dir is None:
            private_dir(run_dir_of(placed.hash) / PRIVATE_DIR)
        base = private_dir(self.work_dir if self.work_dir is not None else run_dir_of(placed.hash) / PYDRA_DIR)
        root = private_dir(base / events.attempt)
        store = store if store is not None else SenselabStore(root)
        config = placed.config if placed.config is not None else RunConfig(reuse=ReusePolicy(placed.logical.reuse))
        cancel = cancel if cancel is not None else threading.Event()
        driver = _Driver(placed, values, session, config, store, events, cancel, self.profile.pool.cancel_check_s)
        keys = plan_keys(placed, values)
        for node, element in driver.expected():
            driver.log("queued", node, element, key=keys.get((node, element)))
        try:
            driver.drive(compile_plan(placed))
        finally:
            driver.close()
        return _outcome(placed, driver.results)


def _outcome(placed: Plan, results: Mapping[str, Mapping[tuple[int, ...], ResultRef]]) -> RunOutcome:
    """Return the run's outcome; each element given no result is `blocked`."""
    mapped = mapped_nodes(placed.logical.graph)
    rows = len(placed.manifest.rows)
    found, blocked = {}, {}
    for node in order(placed.logical.graph):
        every = [(i,) for i in range(rows)] if node in mapped else [()]
        got = results.get(node, {})
        found[node] = tuple(got[e] for e in every if e in got)
        blocked[node] = tuple(e for e in every if e not in got)
    return RunOutcome(placed.hash, found, blocked)


def run_plan(
    placed: Plan, session: PoolSession, *, values: Mapping[InputIdentity, object], events: EventLog
) -> RunOutcome:
    """Run `placed` on `session` with a `PydraRunner` under the plan's profile."""
    return PydraRunner(profile=placed.execution).run_plan(placed, session, values=values, events=events)
