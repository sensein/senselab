"""The plan runner: a placed plan compiled to a pydra workflow, its jobs sent through an `Executor` (`PydraRunner`).

`compile_plan` turns a plan into a pydra workflow: a `records` node split over the manifest's rows, one node per
graph node (a `map` edge reads the row, so every mapped node shares the one record axis, and a node with several
mapped upstreams is split over them zipped), and, for each node a `gather` edge reads, a copy of it combined over the
rows. Each node's checksum is the job's senselab key digest (`job_key`), so pydra hashes no senselab code.
`ExecutorWorker` runs pydra's jobs: it sends a node's ready elements to the `Executor` as fragments of the placement's
batch size (`node_fragment`) and raises on a failed element, so pydra fails only that element's descendants.
`SenselabStore` holds the run's completion records; no result is pickled.
"""

import asyncio
import copy
import typing
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from pydra.compose import python, workflow
from pydra.engine.result import Result
from pydra.engine.store import Lease, context_key
from pydra.engine.submitter import Submitter, WorkflowError
from pydra.utils.resources import ComputeResources
from pydra.utils.routing import tag_router
from pydra.workers.base import Worker

from senselab_core.kernel.graph import Edge, mapped_nodes, order, split
from senselab_core.kernel.identity import (
    NO_EQUIVALENCE,
    Equivalence,
    ExecutionContext,
    InputIdentity,
    ReusePolicy,
    acceptable,
    gather_identity,
    port_output_identity,
)
from senselab_core.runtime.execution.executor import Admission, Executor, FaultRecord, JobRef, ResultRef
from senselab_core.runtime.execution.in_process import job_key, node_fragment
from senselab_core.runtime.execution_settings import JobLease
from senselab_core.runtime.planner import Plan
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.revision import cache_root

RECORDS = "records"
"""The pydra node split over the manifest's rows; every mapped node reads its row from it."""
RECORD_FIELD = "row"
"""The records node's input, a row index; `records.row` is the split every gather combines over."""
REF = "ref"
"""The one output of every compiled node: the job's `ResultRef` as JSON (the row index, for the records node)."""
SENSELAB = "senselab"
"""The key of a pydra context under which a completion record carries its senselab context and backend."""


class ElementFailed(RuntimeError):
    """An element the executor failed; `failure` is its record."""

    def __init__(self, failure: FaultRecord) -> None:
        """Hold the failure and name it in the message."""
        super().__init__(f"{failure.type} ({failure.fault}): {failure.message}")
        self.failure = failure


@dataclass(frozen=True)
class JobEvent:
    """What became of one job: `finished` or `failed` as its executor reported, or `blocked` by a failed upstream."""

    kind: Literal["finished", "failed", "blocked"]
    job: JobRef
    key_digest: str | None = None
    failure: FaultRecord | None = None


class EventLog(Protocol):
    """Where a run's job events are appended, in the order they happen."""

    def append(self, event: JobEvent) -> None:
        """Record `event`."""


@dataclass(frozen=True)
class RunStatus:
    """A finished run: each node's results in element order, and the elements a failed upstream element blocked."""

    plan_hash: str
    results: Mapping[str, tuple[ResultRef, ...]]
    blocked: Mapping[str, tuple[tuple[int, ...], ...]]

    @property
    def done(self) -> bool:
        """Whether every job of the plan completed."""
        failed = any(r.failure is not None for refs in self.results.values() for r in refs)
        return not failed and not any(self.blocked.values())


class SenselabReuse:
    """pydra's `ReusePolicy` for completion records: `acceptable` under the run's policy on their senselab contexts.

    A record whose pydra context carries no senselab context (`SENSELAB`) is reused only in its own context's slot.
    """

    def __init__(self, policy: ReusePolicy, equivalence: Equivalence = NO_EQUIVALENCE) -> None:
        """Accept under `policy`, with the measured `equivalence` classes."""
        self.policy = policy
        self.equivalence = equivalence

    def accept(self, stored: Mapping[str, Any], current: Mapping[str, Any], task: object) -> bool:  # noqa: ARG002
        """Whether a record from `stored` serves `current`: both carry a senselab context `acceptable` admits."""
        a, b = stored.get(SENSELAB), current.get(SENSELAB)
        if not isinstance(a, Mapping) or not isinstance(b, Mapping):
            return False
        return acceptable(
            ExecutionContext.model_validate(a["context"]),
            ExecutionContext.model_validate(b["context"]),
            self.policy,
            backend=str(a["backend"]),
            equivalence=self.equivalence,
        )


class SenselabStore:
    """pydra's `ResultStore` over one run: completion records in memory by checksum and context slot, never pickled.

    A job's results live in senselab's store, which the executor serves under the run's reuse policy; this store
    holds what pydra reads to resolve a downstream input and to mark a job done. `cache_root` holds only the leases
    and the workflow job's own files.
    """

    def __init__(self, cache_root: Path, reuse: SenselabReuse) -> None:
        """Hold the run's records under `cache_root`, reusing another slot's record when `reuse` accepts it."""
        self.cache_root = Path(cache_root)
        self.reuse = reuse
        self._records: dict[str, dict[str, Result]] = {}

    def __getstate__(self) -> dict[str, Any]:
        """Return the state a pickled job carries: the root and policy, never the records."""
        return {"cache_root": self.cache_root, "reuse": self.reuse, "_records": {}}

    def cache_dir(self, checksum: str, context: Mapping[str, Any]) -> Path:
        """Return the directory of the job `checksum` in `context`."""
        return self.cache_root / checksum / str(context_key(context))

    def lookup(
        self,
        checksum: str,
        context: Mapping[str, Any],
        task: object,
        stored: list[tuple[Path, Mapping[str, Any]]] | None = None,  # noqa: ARG002
    ) -> Result | None:
        """Return the record of `checksum` in `context`'s slot, or of another slot `reuse` accepts; None if none."""
        slots = self._records.get(checksum, {})
        own = slots.get(context_key(context))
        if own is not None:
            return own
        for result in slots.values():
            if self.reuse.accept(result.context, context, task):
                reused = copy.copy(result)
                reused.cross_context, reused.reused_in = True, dict(context)
                return reused
        return None

    def stored_contexts(self, checksum: str) -> Iterator[tuple[Path, Mapping[str, Any]]]:
        """Yield the slot and context of each record of `checksum`."""
        for slot, result in sorted(self._records.get(checksum, {}).items()):
            yield self.cache_root / checksum / slot, result.context

    def lockfile(self, checksum: str, context: Mapping[str, Any]) -> Path:
        """Return the lock file of the job `checksum` in `context`."""
        return self.cache_root / f"{checksum}-{context_key(context)}.lock"

    def acquire(self, checksum: str, context: Mapping[str, Any]) -> Lease:
        """Return the lease of the job `checksum` in `context`."""
        return Lease(checksum, self.cache_dir(checksum, context), self.lockfile(checksum, context))

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


def _refs(value: object) -> list[ResultRef]:
    return sorted((_ref(v) for v in typing.cast(Sequence[object], value)), key=lambda r: r.job.element)


class _NodeKey:
    """The checksum of a compiled node's job: its senselab key digest, from the identities its inputs resolve to."""

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
                ports = [port_output_identity(r.key_digest, feed.source_port) for r in _refs(value)]
                found[feed.port] = gather_identity(ports)
            else:
                found[feed.port] = port_output_identity(_ref(value).key_digest, feed.source_port)
        return found

    def checksum(self, inputs: Mapping[str, object]) -> str:
        """Return the digest of the job's logical key (`job_key`)."""
        return job_key(self.placed, self.node, self.identities(inputs)).digest()


def _node_checksum(key: _NodeKey) -> Callable[[object, Mapping[str, object]], str]:
    def checksum(task: object, inputs: Mapping[str, object]) -> str:  # noqa: ARG001
        return key.checksum(inputs)

    return checksum


def _records_key(plan_hash: str) -> Callable[[object, Mapping[str, object]], str]:
    def checksum(task: object, inputs: Mapping[str, object]) -> str:  # noqa: ARG001
        return f"{RECORDS}-{plan_hash}-{inputs[RECORD_FIELD]}"

    return checksum


def _never(**inputs: object) -> object:
    """Stand in for a compiled node's function, which `ExecutorWorker` never calls."""
    raise RuntimeError(f"a compiled node is run by its executor, not called; got {sorted(inputs)}")


def _record(row: int) -> object:
    """Stand in for the records node's function, which `ExecutorWorker` answers itself."""
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
        ValueError: a `map` edge reads a node's output rather than a graph input.
    """
    graph = placed.logical.graph
    if any(e.kind == "map" and split(e.source)[0] != "input" for e in graph.edges):
        raise ValueError("a map edge reads a graph input; a fan-out over a node's output is not run")
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


@dataclass
class _Run:
    """One run's state, shared by its workers: the plan, admission, executor, batches waiting and outcomes."""

    placed: Plan
    admission: Admission
    executor: Executor
    store: SenselabStore
    events: EventLog
    pending: dict[str, list[tuple[Any, tuple[int, ...], asyncio.Future[ResultRef]]]] = field(default_factory=dict)
    scheduled: set[str] = field(default_factory=set)
    results: dict[str, dict[tuple[int, ...], ResultRef]] = field(default_factory=dict)
    errors: list[BaseException] = field(default_factory=list)
    sending: set[asyncio.Task[None]] = field(default_factory=set)


def _element(placed: Plan, node: str, job: Any) -> tuple[int, ...]:  # noqa: ANN401 - a pydra job
    """Return the senselab element of a job of `node`: its record's row, or its per-element upstream's element."""
    for feed in _feeds(placed, node):
        value = getattr(job.task, feed.port)
        if feed.source == "input" and feed.edge.kind == "map":
            return (int(typing.cast(Mapping[str, int], value)[RECORD_FIELD]),)
        if feed.source != "input" and feed.edge.kind == "plain" and _ref(value).job.element:
            return _ref(value).job.element
    return ()


class ExecutorWorker(Worker):  # type: ignore[misc]
    """A pydra worker sending a compiled node's jobs to a senselab `Executor`, in fragments of the batch size.

    Jobs of one node that become ready together are sent as fragments of at most the placement's batch size; a job
    returns when its element's result arrives and raises `ElementFailed` when the executor failed it.
    """

    _plugin_name = "senselab-executor"

    def __init__(self, run_state: _Run | None = None, resource_class: str | None = None) -> None:
        """Run the jobs of `run_state`, those routed to `resource_class` (None for the submitter's own worker)."""
        super().__init__()
        self.run_state = run_state
        self.resource_class = resource_class

    def __getstate__(self) -> dict[str, Any]:
        """Return the state a pickled job carries: no run, whose executor lives in this process."""
        return {"loop": None, "run_state": None, "resource_class": self.resource_class}

    async def run(self, job: Any, rerun: bool = False) -> Result:  # noqa: ANN401, ARG002, FBT001, FBT002
        """Return the job's result once its executor gave it; raise `ElementFailed` for a failed element."""
        run = self.run_state
        assert run is not None, "an unpickled ExecutorWorker has no run"
        node = getattr(type(job.task), "senselab_node", None)
        if node is None:
            outputs: object = {RECORD_FIELD: int(job.task.row)}
        else:
            outputs = (await self._submit(run, node, job)).model_dump(mode="json")
        result = Result(
            outputs=job.task.Outputs(**{REF: outputs}),
            runtime=None,
            errored=False,
            cache_dir=run.store.cache_dir(job.checksum, job.context),
            task=job.task,
            context=job.context,
        )
        run.store.record(job.checksum, result)
        return result

    async def _submit(self, run: _Run, node: str, job: Any) -> ResultRef:  # noqa: ANN401 - a pydra job
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ResultRef] = loop.create_future()
        run.pending.setdefault(node, []).append((job, _element(run.placed, node, job), future))
        if len(run.pending[node]) >= run.placed.placement.nodes[node].batch_size:
            self._flush(run, node)
        elif node not in run.scheduled:
            run.scheduled.add(node)
            loop.call_soon(self._flush, run, node)
        ref = await future
        if ref.failure is not None:
            raise ElementFailed(ref.failure)
        return ref

    def _flush(self, run: _Run, node: str) -> None:
        run.scheduled.discard(node)
        size = run.placed.placement.nodes[node].batch_size
        waiting = run.pending.pop(node, [])
        for start in range(0, len(waiting), size):
            sending = asyncio.ensure_future(self._send(run, node, waiting[start : start + size]))
            run.sending.add(sending)
            sending.add_done_callback(run.sending.discard)

    async def _send(self, run: _Run, node: str, batch: list[tuple[Any, tuple[int, ...], Any]]) -> None:
        futures = {element: future for _, element, future in batch}
        loop = asyncio.get_running_loop()

        def settle(ref: ResultRef) -> None:
            run.results.setdefault(node, {})[ref.job.element] = ref
            kind: Literal["finished", "failed"] = "finished" if ref.failure is None else "failed"
            run.events.append(JobEvent(kind, ref.job, ref.key_digest, ref.failure))
            futures[ref.job.element].set_result(ref)

        def submit() -> None:
            fragment = node_fragment(
                run.placed, run.admission, node, upstream=_upstream(run.placed, node, batch), elements=list(futures)
            )
            for ref in run.executor.results(run.executor.submit(fragment)):
                loop.call_soon_threadsafe(settle, ref)

        try:
            await asyncio.to_thread(submit)
        except Exception as error:  # noqa: BLE001 - every waiting job is failed with it, and the run re-raises it
            run.errors.append(error)
        missing = [element for element, future in futures.items() if not future.done()]
        if missing and not run.errors:
            run.errors.append(RuntimeError(f"the executor gave node {node!r} no result for elements {missing}"))
        for element in missing:
            futures[element].set_exception(run.errors[-1])


def _upstream(placed: Plan, node: str, batch: list[tuple[Any, tuple[int, ...], Any]]) -> dict[str, list[ResultRef]]:
    """Return each upstream node's results the jobs of `batch` read, from their resolved inputs."""
    found: dict[str, dict[tuple[int, ...], ResultRef]] = {}
    for job, _, _ in batch:
        for feed in _feeds(placed, node):
            if feed.source == "input":
                continue
            value = getattr(job.task, feed.port)
            refs = _refs(value) if feed.edge.kind == "gather" else [_ref(value)]
            found.setdefault(feed.source, {}).update({r.job.element: r for r in refs})
    return {source: list(refs.values()) for source, refs in found.items()}


class PydraRunner:
    """Runs placed plans through an `Executor`: pydra expands the graph and orders its jobs (on `raise_at_end`).

    `lease` is the profile's job-lease setting; it is handed to pydra's lease settings once the pinned pydra has them.
    """

    def __init__(self, *, profile: ExecutionProfile | None = None, work_dir: Path | None = None) -> None:
        """Run under `profile`'s job leases, pydra's files under `work_dir` (`cache_root()/runs/<plan>/pydra`)."""
        self.profile = profile if profile is not None else ExecutionProfile.load()
        self.lease: JobLease = self.profile.lease
        self.work_dir = work_dir

    def run_plan(
        self,
        placed: Plan,
        executor: Executor,
        *,
        values: Mapping[InputIdentity, object],
        events: EventLog,
        store: SenselabStore | None = None,
    ) -> RunStatus:
        """Run every job of `placed` through `executor`, admitted with `values`, and return what became of each.

        A failed element fails only its descendants, which are reported `blocked`; every other element runs.
        `store` holds the run's completion records; None means a fresh one under the work directory.

        Raises:
            ValueError: as `compile_plan`, or as `executor.admit`.
            WorkflowError: pydra failed a job, and the executor failed no element.
            Exception: the first error an executor call raised, or a result it did not give, after the run.
        """
        root = self.work_dir if self.work_dir is not None else cache_root() / "runs" / placed.hash / "pydra"
        store = store if store is not None else SenselabStore(root, SenselabReuse(ReusePolicy(placed.logical.reuse)))
        admission = executor.admit(placed, values)
        run = _Run(placed, admission, executor, store, events)
        classes = sorted({p.resource_class for p in placed.placement.nodes.values()})
        workers = {name: ExecutorWorker(run_state=run, resource_class=name) for name in classes}
        try:
            with Submitter(
                store=store,
                worker=ExecutorWorker(run_state=run),
                workers=workers,
                router=tag_router({name: name for name in classes}),
                on_error="raise_at_end",
            ) as submitter:
                submitter(compile_plan(placed), raise_errors=True)
        except WorkflowError as error:
            if run.errors:
                raise run.errors[0] from error
            if not _failed(run):
                raise
        finally:
            executor.release(admission)
        return _status(placed, run)


def _failed(run: _Run) -> bool:
    """Whether the executor failed any element of the run."""
    return any(r.failure is not None for refs in run.results.values() for r in refs.values())


def _status(placed: Plan, run: _Run) -> RunStatus:
    """Return the run's status; each element given no result is `blocked`, and appended to the events as such."""
    mapped = mapped_nodes(placed.logical.graph)
    rows = len(placed.manifest.rows)
    results, blocked = {}, {}
    for node in order(placed.logical.graph):
        every = [(i,) for i in range(rows)] if node in mapped else [()]
        got = run.results.get(node, {})
        results[node] = tuple(got[e] for e in every if e in got)
        blocked[node] = tuple(e for e in every if e not in got)
        for element in blocked[node]:
            run.events.append(JobEvent("blocked", JobRef(plan_hash=placed.hash, node=node, element=element)))
    return RunStatus(placed.hash, results, blocked)


def run_plan(
    placed: Plan, executor: Executor, *, values: Mapping[InputIdentity, object], events: EventLog
) -> RunStatus:
    """Run `placed` through `executor` on a `PydraRunner` under the packaged default profile."""
    return PydraRunner().run_plan(placed, executor, values=values, events=events)
