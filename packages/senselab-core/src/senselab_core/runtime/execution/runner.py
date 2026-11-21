"""The plan runner: a placed plan compiled to a pydra workflow, its jobs sent through an `Executor` (`PydraRunner`).

`compile_plan` turns a plan into a pydra workflow: a `records` node split over the manifest's rows, one node per
graph node (a `map` edge reads the row, so every mapped node shares the one record axis, and a node with several
mapped upstreams is split over them zipped), and, for each node a `gather` edge reads, a copy of it combined over the
rows. Each node's checksum is the job's senselab key digest (`job_checksum`), so pydra hashes no senselab code.
The runner drives pydra's `GraphRunner` from outside (RFC 0002 §3.5) on one thread: each round's ready jobs go to
the `Executor` in one `submit_many` call, as fragments of the placement's batch size (`node_fragment`), or one
fragment per node to an executor that splits fragments. Submissions run on sender threads and one poll thread
follows every open fragment; both hand what they learn to the driver thread through a queue, which alone calls the
`GraphRunner`. A job pydra settles as the twin of another job of its key is sent again, to be served as a reuse.
`SenselabStore` holds the run's completion records; no result is pickled. Its job leases take no lock file: the
run's coordinator epoch is the one exclusion of a run (ADR-003). Each attempt has a pydra root of its own,
`<work dir>/<attempt>`.
"""

import asyncio
import contextvars
import queue
import threading
import typing
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import CancelledError as FutureCancelled
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
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

from senselab_core.kernel.graph import Edge, fan_out_issues, mapped_nodes, order, split
from senselab_core.kernel.identity import InputIdentity, gather_identity, port_output_identity
from senselab_core.runtime.clock import Clock, SystemClock
from senselab_core.runtime.execution.epoch import StaleCoordinator
from senselab_core.runtime.execution.events import TERMINAL, EventKind, EventLog, run_dir_of
from senselab_core.runtime.execution.executor import (
    Admission,
    Executor,
    FaultRecord,
    Handle,
    JobRef,
    PlanFragment,
    QueueHeld,
    ResultRef,
)
from senselab_core.runtime.execution.in_process import fragment_keys, job_checksum, node_fragment, plan_keys
from senselab_core.runtime.execution.run import PYDRA_DIR
from senselab_core.runtime.licenses import LicenseRefused
from senselab_core.runtime.planner import Plan, check_mapped_outputs
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.revision import attempt_scope
from senselab_core.runtime.storage import PRIVATE_DIR, private_dir

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


Interrupted = (QueueHeld, StaleCoordinator)
"""The errors that hold a run's unsent jobs without failing them, each `NotRun(held=True)`: a held env or queue, a
newer coordinator."""
_CANCELLED = (asyncio.CancelledError, FutureCancelled)
"""The cancellations a submission can raise; each is the run's `RunCancelled`."""


class RunCancelled(RuntimeError):
    """The run was cancelled: a job not yet sent to the executor was not sent."""


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
    """pydra's lease of a job, held without a lock file: the run's coordinator epoch excludes other coordinators."""

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

    A job's results live in senselab's store, which the executor serves under the run's reuse policy; this store
    holds what pydra reads to resolve a downstream input and to mark a job done, and `lookup` returns a record of its
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
    """One fragment sent: its node and the pydra job of each element; `served` for jobs pydra settled as twins.

    A twin's result is senselab's reuse of another job's: it is logged and held, and never reported to pydra.
    """

    node: str
    jobs: dict[tuple[int, ...], Job]
    served: bool = False
    given: set[tuple[int, ...]] = field(default_factory=set)


Message = tuple[str, _Batch, object]
"""What a sender or the poll thread hands the driver: (`result`, `ended` or `refused`, the batch, the payload)."""


class _Poller:
    """The poll thread: `Executor.poll` per open handle per round, each result and each end handed to `post`.

    A handle ends once: with None once every result was given, with the error its poll raised, or with
    `RunCancelled` once `cancel` is set (the handle is cancelled through the executor after its landed results are
    posted). A round that gives nothing waits `first_s`, doubling up to `max_s`; a new handle or `cancel` ends the
    wait, which checks `cancel` every `cancel_check_s` on `clock`. The thread runs in the context it was made in.
    """

    def __init__(  # noqa: PLR0913 - the executor, cancel and post, then keyword-only waits and the clock
        self,
        executor: Executor,
        cancel: threading.Event,
        post: Callable[[Message], object],
        *,
        first_s: float,
        max_s: float,
        cancel_check_s: float,
        clock: Clock = SystemClock(),  # noqa: B008 - stateless
    ) -> None:
        """Poll the handles of `executor`, posting to `post` and stopping at `cancel`; waits run on `clock`."""
        self.executor, self.cancel, self.post, self._clock = executor, cancel, post, clock
        self.first_s, self.max_s, self.cancel_check_s = first_s, max_s, cancel_check_s
        self._open: dict[str, tuple[Handle, _Batch]] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._closed = False
        self._thread: threading.Thread | None = None
        self._context = contextvars.copy_context()

    def follow(self, handle: Handle, batch: _Batch) -> None:
        """Poll `handle`, the fragment of `batch`, from the next round, which starts at once."""
        with self._lock:
            self._open[handle.id] = (handle, batch)
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._context.run, args=(self._loop,), name="senselab-poll", daemon=True
                )
                self._thread.start()
        self._wake.set()

    def _round(self) -> bool:
        """Poll every open handle once; return whether any result was given."""
        with self._lock:
            followed = list(self._open.values())
        gave = False
        for handle, batch in followed:
            try:
                refs, done = self.executor.poll(handle)
            except Exception as error:  # noqa: BLE001 - the handle's end, which the driver classifies
                self._end(handle, batch, error)
                continue
            for ref in refs:
                self.post(("result", batch, ref))
            gave = gave or bool(refs)
            if done:
                self._end(handle, batch, None)
            elif self.cancel.is_set():
                self._end(handle, batch, self._cancelled(handle))
        return gave

    def _cancelled(self, handle: Handle) -> Exception:
        """Cancel `handle` through the executor; return `RunCancelled`, or the error the cancel raised."""
        try:
            self.executor.cancel(handle)
        except Exception as error:  # noqa: BLE001 - the handle's end
            return error
        return RunCancelled(f"the run was cancelled while {handle.id} was polled")

    def _end(self, handle: Handle, batch: _Batch, error: Exception | None) -> None:
        with self._lock:
            self._open.pop(handle.id, None)
        self.post(("ended", batch, error))

    def _loop(self) -> None:
        pause = self.first_s
        while not self._closed:
            self._wake.clear()
            if self._round():
                pause = self.first_s
            elif self._open:
                self._nap(pause)
                pause = min(2 * pause, self.max_s)
            else:
                self._wake.wait()

    def _nap(self, seconds: float) -> None:
        """Wait up to `seconds`, ending early on a new handle, on `close`, or once `cancel` is set."""
        deadline = self._clock.now() + seconds
        while not self._wake.is_set() and not self.cancel.is_set() and (left := deadline - self._clock.now()) > 0:
            self._clock.wait(self._wake, min(left, self.cancel_check_s))

    def close(self) -> None:
        """Stop polling and cancel every handle still open through the executor."""
        self._closed = True
        self._wake.set()
        if self._thread is not None:
            self._thread.join()
        with self._lock:
            left, self._open = list(self._open.values()), {}
        for handle, _ in left:
            self._cancelled(handle)


class _Driver:
    """One run: the `GraphRunner` of a placed plan driven on this thread (RFC 0002 §3.5), its rounds sent and settled.

    Senders and the poll thread only post to `inbox`; every `GraphRunner` call is made here. An executor's result is
    a pydra `Result` holding the job's `ResultRef`, or `Failed(ElementFailed)`. A fragment the executor refused is,
    for each of its jobs: `Failed` for a license refusal or any other exception (the latter raised after the run),
    `NotRun(held=True)` for a held env or queue or a newer coordinator, and `NotRun` once the run is cancelled.
    """

    def __init__(  # noqa: PLR0913 - the run's parts
        self,
        placed: Plan,
        admission: Admission,
        values: Mapping[InputIdentity, object],
        executor: Executor,
        store: "SenselabStore",
        events: EventLog,
        cancel: threading.Event,
        poller: Callable[[Callable[[Message], object]], _Poller],
        senders: int,
    ) -> None:
        """Run `placed` through `executor` under `admission`, on `senders` sender threads and the poller made."""
        self.placed, self.admission, self.values, self.executor = placed, admission, values, executor
        self.store, self.events, self.cancel = store, events, cancel
        self.inbox: queue.Queue[Message] = queue.Queue()
        self.poller = poller(self.inbox.put)
        self.senders = ThreadPoolExecutor(senders, thread_name_prefix="senselab-send")
        self.results: dict[str, dict[tuple[int, ...], ResultRef]] = {}
        self.ended: set[tuple[str, tuple[int, ...]]] = set()
        self.errors: list[BaseException] = []
        self.open: set[_Batch] = set()
        self.refs: dict[Job, ResultRef] = {}
        self.first: dict[str, ResultRef] = {}
        self.twins: dict[str, dict[tuple[int, ...], Job]] = {}
        self.reported: set[str] = set()

    def drive(self, task: Any) -> None:  # noqa: ANN401 - a pydra workflow task
        """Drive `task`'s `GraphRunner` until no job can progress and no fragment is open.

        Raises:
            Exception: the first error of the run that is no job's failure: an executor's, a hold, a cancellation.
            WorkflowError: pydra errored a job no executor failed.
            RuntimeError: nothing can progress, but the run is not done.
        """
        with Submitter(worker="debug", store=self.store, telemetry=False) as submitter:
            graph = GraphRunner(
                task, self.store, on_error="raise_at_end", lookup_before_dispatch=False, submitter=submitter
            )
            try:
                self._loop(graph)
                foreign = [e for e in graph.entries if _label(e.job_name, e.state_index) not in self.reported]
            finally:
                graph.close()
                self.poller.close()
                self.senders.shutdown(cancel_futures=True)
        if self.errors:
            raise self.errors[0]
        if foreign:
            listed = "\n".join(f"{_label(e.job_name, e.state_index)} ({e.phase}): {e.message}" for e in foreign)
            raise WorkflowError(
                f"pydra errored jobs no executor failed:\n{listed}", [e.message for e in foreign], foreign
            )

    def _loop(self, graph: GraphRunner) -> None:
        while True:
            rnd = graph.ready()
            progressed = self._settle_round(rnd)
            progressed = self._dispatch(graph, rnd.dispatch) or progressed
            if progressed:
                continue
            if not self.open:
                if graph.done:
                    return
                if rnd.retry_after is None:
                    raise RuntimeError(graph.stuck() or "nothing can progress, but the run is not done")
            self._receive(graph, rnd.retry_after)

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
        """Send `jobs` and the twins to serve as one round; return whether any job was settled here instead."""
        grouped: dict[str, dict[tuple[int, ...], Job]] = {}
        settled = False
        for job in jobs:
            node, element = _senselab(job)
            if node is None:
                graph.complete(job, self._row(job))
            elif self._stopping():
                graph.complete(job, NotRun("the run was stopped before it was sent", held=self._held()))
            else:
                grouped.setdefault(node, {}).setdefault(element, job)
                continue
            settled = True
        twins, self.twins = self.twins, {}
        sends = [(n, js, False) for n, js in grouped.items()]
        sends += [] if self._stopping() else [(n, js, True) for n, js in twins.items()]
        items = [item for node, chosen, served in sends for item in self._fragments(graph, node, chosen, served)]
        if items:
            self.open.update(batch for batch, _ in items)
            self.senders.submit(contextvars.copy_context().run, self._send, items)
        return settled

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

    def _stopping(self) -> bool:
        """Whether no fragment is sent any more: the run is cancelled or held by a newer coordinator."""
        if self.cancel.is_set() and not any(isinstance(e, RunCancelled) for e in self.errors):
            self.errors.append(RunCancelled(f"run {self.placed.hash} was cancelled before every fragment was sent"))
        return self.cancel.is_set() or any(isinstance(e, (RunCancelled, StaleCoordinator)) for e in self.errors)

    def _held(self) -> bool:
        """Whether a newer coordinator stopped the run, which holds every job not yet sent."""
        return any(isinstance(e, StaleCoordinator) for e in self.errors)

    def _fragments(
        self, graph: GraphRunner, node: str, chosen: Mapping[tuple[int, ...], Job], served: bool
    ) -> list[tuple[_Batch, PlanFragment | BaseException]]:
        """Return the batches of `node`'s `chosen` jobs, each with its fragment (or the error forming it raised)."""
        elements = sorted(chosen)
        size = None if self.executor.splits_fragments else self.executor.fragment_size(self.placed, node)
        size = size or max(1, len(elements))
        made: list[tuple[_Batch, PlanFragment | BaseException]] = []
        for start in range(0, len(elements), size):
            batch = _Batch(node, {e: chosen[e] for e in elements[start : start + size]}, served)
            try:
                fragment = node_fragment(
                    self.placed, self.admission, node, upstream=self._upstream(graph, batch), elements=list(batch.jobs)
                )
            except Exception as error:  # noqa: BLE001 - the batch's refusal
                made.append((batch, error))
                continue
            if not self.executor.logs_claims:
                for job, key in zip(fragment.jobs, fragment_keys(fragment, self.placed, self.values), strict=True):
                    self.log("started", node, job.element, key=key)
            made.append((batch, fragment))
        return made

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

    def _send(self, items: Sequence[tuple[_Batch, PlanFragment | BaseException]]) -> None:
        """Submit a round's fragments in one `submit_many`, following each handle and posting each refusal."""
        formed = [(batch, f) for batch, f in items if isinstance(f, PlanFragment)]
        for batch, error in items:
            if not isinstance(error, PlanFragment):
                self.inbox.put(("refused", batch, error))
        if not formed:
            return
        try:
            taken: Sequence[Handle | BaseException] = self.executor.submit_many([f for _, f in formed])
        except BaseException as error:  # noqa: BLE001 - every fragment's refusal, a cancellation included
            taken = [error] * len(formed)
        for (batch, _), handle in zip(formed, taken, strict=True):
            if isinstance(handle, Handle):
                self.poller.follow(handle, batch)
            else:
                self.inbox.put(("refused", batch, handle))

    def _receive(self, graph: GraphRunner, timeout: float | None) -> None:
        """Apply the next message, waiting up to `timeout` (None: until one comes), then every message waiting."""
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
        kind, batch, payload = message
        if kind == "result":
            self._given(graph, batch, typing.cast(ResultRef, payload))
            return
        self.open.discard(batch)
        missing = [e for e in batch.jobs if e not in batch.given]
        if missing:
            error = payload or RuntimeError(f"the executor gave node {batch.node!r} no result for elements {missing}")
            self._unsent(graph, batch, missing, typing.cast(BaseException, error))

    def _given(self, graph: GraphRunner, batch: _Batch, ref: ResultRef) -> None:
        """Settle an element's result; report it to pydra unless pydra settled the job as a twin."""
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
        """Give each job of `elements` its outcome for `error`, which kept it from a result (`_Driver`)."""
        if isinstance(error, (FutureCancelled, *_CANCELLED)):
            cancelled = RunCancelled(f"run {self.placed.hash}: the submission of node {batch.node!r} was cancelled")
            cancelled.__cause__ = error
            error = cancelled
        if not isinstance(error, Exception):
            raise error
        held = isinstance(error, (RunCancelled, *Interrupted))
        outcome = NotRun(str(error), held=isinstance(error, Interrupted)) if held else Failed(error)
        failure = None if held else FaultRecord.from_exception(error)
        if not isinstance(error, LicenseRefused) and not any(e is error for e in self.errors):
            self.errors.append(error)
        for element in elements:
            job = batch.jobs[element]
            if failure is not None:
                ref = ResultRef(job=self._job_ref(batch.node, element), key_digest=job.checksum, failure=failure)
                self.settle(batch.node, ref)
                self.first.setdefault(job.checksum, ref)
                self.refs[job] = ref
                self.reported.add(_label(job.name, job.state_index))
            if not batch.served:
                graph.complete(job, outcome)

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

        A run stopped by a held env or a newer coordinator (`Interrupted`) leaves them without one, for a later epoch.
        """
        if any(isinstance(e, Interrupted) for e in self.errors):
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


class PydraRunner:
    """Runs placed plans through an `Executor`: pydra's `GraphRunner` expands the graph and releases its jobs.

    `lease` is the profile's `lease` section; it is held and not passed to pydra. Rounds are submitted on `senders`
    threads, the sum of the profile's `max_workers` by default; their results are followed by one poll thread per
    run, so neither bounds the fragments in flight.
    """

    def __init__(
        self,
        *,
        profile: ExecutionProfile | None = None,
        work_dir: Path | None = None,
        senders: int | None = None,
        clock: Clock = SystemClock(),  # noqa: B008 - stateless
    ) -> None:
        """Run under `profile`'s job leases, pydra's files under `work_dir` (`<run dir>/private/pydra`).

        The poll thread's waits run on `clock`.

        Raises:
            ValueError: `senders` is below 1.
        """
        self.profile = profile if profile is not None else ExecutionProfile.load()
        self.work_dir, self._clock = work_dir, clock
        self.senders = (
            senders if senders is not None else sum(c.max_workers for c in self.profile.resource_classes.values())
        )
        if self.senders < 1:
            raise ValueError(f"a runner submits on at least one thread, got {self.senders}")

    def run_plan(  # noqa: PLR0913 - the plan, its executor and values, then keyword-only run inputs
        self,
        placed: Plan,
        executor: Executor,
        *,
        values: Mapping[InputIdentity, object],
        events: EventLog,
        store: SenselabStore | None = None,
        cancel: threading.Event | None = None,
    ) -> RunOutcome:
        """Run every job of `placed` through `executor`, admitted with `values`, and return what became of each.

        Each job is first logged `queued` with its key under this plan (`plan_keys`).

        A failed element fails only its descendants, which are reported `blocked`; every other element runs. A job
        sharing its key with another job of the run is that job's reuse, or its failure. A license refusal fails each
        job of its fragment and is not raised. Each job's `queued`, `started` and `finished` or `failed` event is
        appended to `events`. Once `cancel` is set no further fragment is sent: open fragments are cancelled and the
        run raises `RunCancelled`. While it runs, `attempt_id()` is the attempt of `events`. pydra's files are under
        `<work dir>/<attempt of events>`.

        Raises:
            ValueError: as `compile_plan`, or as `executor.admit`.
            RunCancelled: `cancel` was set before every fragment was sent, or a submission was cancelled.
            QueueHeld, StaleCoordinator: a held env or queue, or a newer coordinator, kept jobs from running; they
                have no terminal event.
            WorkflowError: pydra errored a job no executor failed.
            Exception: the first other error an executor call raised, or a result it did not give, after the run.
        """
        with attempt_scope(events.attempt):
            return self._run_plan(placed, executor, values, events, store, cancel)

    def _run_plan(  # noqa: PLR0913 - as `run_plan`
        self,
        placed: Plan,
        executor: Executor,
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
        admission = executor.admit(placed, values)
        cancel = cancel if cancel is not None else threading.Event()
        pool = self.profile.pool

        def poller(post: Callable[[Message], object]) -> _Poller:
            return _Poller(
                executor,
                cancel,
                post,
                first_s=pool.first_poll_s,
                max_s=pool.poll_s,
                cancel_check_s=pool.cancel_check_s,
                clock=self._clock,
            )

        driver = _Driver(placed, admission, values, executor, store, events, cancel, poller, self.senders)
        keys = plan_keys(placed, values)
        for node, element in driver.expected():
            driver.log("queued", node, element, key=keys.get((node, element)))
        try:
            driver.drive(compile_plan(placed))
        finally:
            executor.release(admission)
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
    placed: Plan, executor: Executor, *, values: Mapping[InputIdentity, object], events: EventLog
) -> RunOutcome:
    """Run `placed` through `executor` on a `PydraRunner` under the packaged default profile."""
    return PydraRunner().run_plan(placed, executor, values=values, events=events)
