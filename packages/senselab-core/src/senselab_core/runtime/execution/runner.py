"""The plan runner: a placed plan compiled to a pydra workflow, its jobs sent through an `Executor` (`PydraRunner`).

`compile_plan` turns a plan into a pydra workflow: a `records` node split over the manifest's rows, one node per
graph node (a `map` edge reads the row, so every mapped node shares the one record axis, and a node with several
mapped upstreams is split over them zipped), and, for each node a `gather` edge reads, a copy of it combined over the
rows. Each node's checksum is the job's senselab key digest (`job_checksum`), so pydra hashes no senselab code.
`ExecutorWorker` runs pydra's jobs: it sends a node's ready elements to the `Executor` as fragments of the placement's
batch size (`node_fragment`), or as one fragment to an executor that splits fragments, and raises on a failed
element, so pydra fails only that element's descendants. A fragment is submitted and returns at once; one `Follower`
thread per run polls every open fragment and settles each element as its result lands.
`SenselabStore` holds the run's completion records; no result is pickled.
"""

import asyncio
import contextlib
import contextvars
import os
import re
import threading
import time
import traceback
import typing
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydra.compose import python, workflow
from pydra.engine.result import Result
from pydra.engine.store import Lease, context_key
from pydra.engine.submitter import Submitter, WorkflowError
from pydra.utils.resources import ComputeResources
from pydra.utils.routing import tag_router
from pydra.workers.base import Worker

from senselab_core.kernel.graph import Edge, fan_out_issues, mapped_nodes, order, split
from senselab_core.kernel.identity import InputIdentity, gather_identity, port_output_identity
from senselab_core.runtime.execution.epoch import StaleCoordinator
from senselab_core.runtime.execution.events import TERMINAL, EventKind, EventLog, run_dir_of
from senselab_core.runtime.execution.executor import (
    Admission,
    Executor,
    FaultRecord,
    Handle,
    JobRef,
    QueueHeld,
    ResultRef,
)
from senselab_core.runtime.execution.in_process import fragment_keys, job_checksum, node_fragment, plan_keys
from senselab_core.runtime.execution_settings import JobLease
from senselab_core.runtime.planner import Plan, check_mapped_outputs
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.revision import attempt_scope
from senselab_core.runtime.storage import PRIVATE_DIR_MODE

RECORDS = "records"
"""The pydra node split over the manifest's rows; every mapped node reads its row from it."""
RECORD_FIELD = "row"
"""The records node's input, a row index; `records.row` is the split every gather combines over."""
REF = "ref"
"""The one output of every compiled node: the job's `ResultRef` as JSON (the row index, for the records node)."""
NO_TELEMETRY = "NO_ET"
"""The environment variable whose presence stops pydra's version check from fetching anything."""
CANCEL_CHECK_S = 0.05
"""Seconds between a waiting follower's checks of the run's cancel event."""


class ElementFailed(RuntimeError):
    """An element the executor failed; `failure` is its record."""

    def __init__(self, failure: FaultRecord) -> None:
        """Hold the failure and name it in the message."""
        super().__init__(f"{failure.type} ({failure.fault}): {failure.message}")
        self.failure = failure


Interrupted = (QueueHeld, StaleCoordinator)
"""The errors that stop a run without failing its unsent jobs: a held env, a newer coordinator."""


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


class SenselabStore:
    """pydra's `ResultStore` over one run: completion records in memory by checksum and context slot, never pickled.

    A job's results live in senselab's store, which the executor serves under the run's reuse policy; this store
    holds what pydra reads to resolve a downstream input and to mark a job done, and `lookup` returns a record of its
    own context's slot only. `cache_root` holds only the leases and the workflow job's own files.
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
                ports = [port_output_identity(r.key_digest, feed.source_port) for r in _refs(value)]
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


@dataclass
class _Run:
    """One run's state, shared by its workers: the plan, admission and its values, executor, batches and outcomes."""

    placed: Plan
    admission: Admission
    values: Mapping[InputIdentity, object]
    executor: Executor
    store: SenselabStore
    events: EventLog
    cancel: threading.Event
    senders: ThreadPoolExecutor
    follower: "Follower"
    pending: dict[str, list[tuple[Any, tuple[int, ...], asyncio.Future[ResultRef]]]] = field(default_factory=dict)
    scheduled: set[str] = field(default_factory=set)
    results: dict[str, dict[tuple[int, ...], ResultRef]] = field(default_factory=dict)
    errors: list[BaseException] = field(default_factory=list)
    sending: set[asyncio.Task[None]] = field(default_factory=set)
    element_failures: set[str] = field(default_factory=set)
    ended: set[tuple[str, tuple[int, ...]]] = field(default_factory=set)

    def expected(self) -> list[tuple[str, tuple[int, ...]]]:
        """Return every job of the plan as (node, element): a mapped node's per record, any other once."""
        graph = self.placed.logical.graph
        mapped = mapped_nodes(graph)
        rows = len(self.placed.manifest.rows)
        return [(n, (i,)) for n in order(graph) if n in mapped for i in range(rows)] + [
            (n, ()) for n in order(graph) if n not in mapped
        ]

    def fragment_size(self, node: str) -> int | None:
        """Return the most jobs of `node` one fragment holds: its batch size, None for an executor that splits."""
        return None if self.executor.splits_fragments else self.placed.placement.nodes[node].batch_size

    def close(self) -> None:
        """Give every job with no terminal event its own: `cancelled` once cancelled, else `blocked`.

        A run stopped by a held env or a newer coordinator (`Interrupted`) leaves them without one, for a later epoch.
        """
        if any(isinstance(e, Interrupted) for e in self.errors):
            return
        for node, element in self.expected():
            if (node, element) not in self.ended:
                self.log("cancelled" if self.cancel.is_set() else "blocked", node, element)

    def fail_batch(self, node: str, elements: list[tuple[int, ...]], error: Exception) -> None:
        """Log `failed` for each of `elements` with no result: infrastructure, with the error and its notes."""
        failure = FaultRecord(
            fault="infrastructure",
            type=type(error).__name__,
            message="".join(traceback.format_exception_only(error)),
            gap=False,
            charged=False,
        )
        for element in elements:
            if (node, element) not in self.ended:
                self.log("failed", node, element, failure=failure)

    def send(self, node: str, batch: list[tuple[Any, tuple[int, ...], Any]]) -> Handle:
        """Submit the jobs of `batch` as one fragment and return its handle.

        Each job is logged `started` as it is sent, unless the executor logs its own claims (`logs_claims`).

        Raises:
            RunCancelled: the run was cancelled before the fragment was sent.
        """
        if self.cancel.is_set():
            raise RunCancelled(f"run {self.placed.hash} was cancelled before node {node!r} was sent")
        elements = [element for _, element, _ in batch]
        fragment = node_fragment(
            self.placed, self.admission, node, upstream=_upstream(self.placed, node, batch), elements=elements
        )
        if not self.executor.logs_claims:
            for job, key in zip(fragment.jobs, fragment_keys(fragment, self.placed, self.values), strict=True):
                self.log("started", node, job.element, key=key)
        return self.executor.submit(fragment)

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
            job=JobRef(plan_hash=self.placed.hash, node=node, element=element),
            key=key,
            placement_digest=placement.canonical_digest(),
            failure=failure,
            env=placement.env,
            output=output,
            reuse=reuse,
        )


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
    """A pydra worker sending a compiled node's jobs to a senselab `Executor`, by batch size or by ready set.

    Jobs of one node that become ready together are sent as fragments of at most the placement's batch size, or as one
    fragment per ready set to an executor that splits fragments (`splits_fragments`). A job returns when its element's
    result arrives and raises `ElementFailed` when the executor failed it.
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
            try:
                outputs = (await self._submit(run, node, job)).model_dump(mode="json")
            except ElementFailed:
                index = job.state_index
                run.element_failures.add(job.name if index is None else f"{job.name}({index})")
                raise
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
        element = _element(run.placed, node, job)
        run.pending.setdefault(node, []).append((job, element, future))
        size = run.fragment_size(node)
        if size is not None and len(run.pending[node]) >= size:
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
        waiting = run.pending.pop(node, [])
        size = run.fragment_size(node)
        if size is None:
            size = max(1, len(waiting))
        for start in range(0, len(waiting), size):
            sending = asyncio.ensure_future(self._send(run, node, waiting[start : start + size]))
            run.sending.add(sending)
            sending.add_done_callback(run.sending.discard)

    async def _send(self, run: _Run, node: str, batch: list[tuple[Any, tuple[int, ...], Any]]) -> None:
        """Dispatch `batch` as one fragment; once every result it gave is settled, fail the elements it gave none."""
        futures = {element: future for _, element, future in batch}

        def settle(ref: ResultRef) -> None:
            run.results.setdefault(node, {})[ref.job.element] = ref
            kind: EventKind = "failed" if ref.failure is not None else "cache_hit" if ref.served else "finished"
            run.log(
                kind,
                node,
                ref.job.element,
                key=ref.key_digest,
                failure=ref.failure,
                output=ref.output_identity,
                reuse=ref.served,
            )
            futures[ref.job.element].set_result(ref)

        try:
            error = await _dispatched(run, node, batch, settle)
        except BaseException as signal:
            for future in futures.values():
                if not future.done():
                    future.set_exception(signal)
            raise
        if error is not None:
            run.errors.append(error)
            if not isinstance(error, (RunCancelled, *Interrupted)):
                run.fail_batch(node, list(futures), error)
        missing = [element for element, future in futures.items() if not future.done()]
        if missing and not run.errors:
            run.errors.append(RuntimeError(f"the executor gave node {node!r} no result for elements {missing}"))
        for element in missing:
            futures[element].set_exception(run.errors[-1])


async def _dispatched(
    run: _Run, node: str, batch: list[tuple[Any, tuple[int, ...], Any]], settle: Callable[[ResultRef], None]
) -> Exception | None:
    """Submit `batch` on the run's senders, then await the follower, each result passed to `settle` on this loop.

    No thread is held while the fragment's results are awaited. Returns once every result given is settled, with the
    error the submission or the following raised, or None.
    """
    loop = asyncio.get_running_loop()
    ended: asyncio.Future[Exception | None] = loop.create_future()

    def end(error: Exception | None) -> None:
        if not ended.done():
            ended.set_result(error)

    def soon(callback: Callable[..., None], *args: object) -> None:
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(callback, *args)

    try:
        handle = await loop.run_in_executor(run.senders, contextvars.copy_context().run, run.send, node, batch)
    except Exception as raised:  # noqa: BLE001 - the caller fails the batch with it, and the run re-raises it
        return raised
    run.follower.follow(handle, lambda ref: soon(settle, ref), lambda error: soon(end, error))
    return await ended


class Follower:
    """One thread following every open handle of a run, `Executor.poll` per handle per round, with no thread each.

    Each result is passed to its handle's `give`, and `end` is called once per handle: with None once every result
    was given, else with the error its poll raised, or `RunCancelled` once `cancel` is set (the handle is cancelled
    through the executor after its landed results are given). A round that gives nothing waits `first_s`, doubling up
    to `max_s`; a new handle or `cancel` ends the wait. The thread runs in the context the follower was made in.
    """

    def __init__(self, executor: Executor, cancel: threading.Event, *, first_s: float, max_s: float) -> None:
        """Follow the handles of `executor`, stopping at `cancel`."""
        self.executor, self.cancel, self.first_s, self.max_s = executor, cancel, first_s, max_s
        self._open: dict[str, tuple[Handle, Callable[[ResultRef], object], Callable[[Exception | None], object]]] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._closed = False
        self._thread: threading.Thread | None = None
        self._context = contextvars.copy_context()

    def follow(
        self, handle: Handle, give: Callable[[ResultRef], object], end: Callable[[Exception | None], object]
    ) -> None:
        """Follow `handle` from the next round, which starts at once."""
        with self._lock:
            self._open[handle.id] = (handle, give, end)
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._context.run, args=(self._loop,), name="senselab-follower", daemon=True
                )
                self._thread.start()
        self._wake.set()

    def open_handles(self) -> int:
        """Return the handles followed and not yet ended."""
        with self._lock:
            return len(self._open)

    def _round(self) -> bool:
        """Poll every open handle once; return whether any result was given."""
        with self._lock:
            followed = list(self._open.values())
        gave = False
        for handle, give, end in followed:
            try:
                refs, done = self.executor.poll(handle)
            except Exception as error:  # noqa: BLE001 - ended with it; the runner re-raises it
                self._end(handle, end, error)
                continue
            for ref in refs:
                give(ref)
            gave = gave or bool(refs)
            if done:
                self._end(handle, end, None)
            elif self.cancel.is_set():
                self.executor.cancel(handle)
                self._end(handle, end, RunCancelled(f"the run was cancelled while {handle.id} was followed"))
        return gave

    def _end(self, handle: Handle, end: Callable[[Exception | None], object], error: Exception | None) -> None:
        with self._lock:
            self._open.pop(handle.id, None)
        end(error)

    def _loop(self) -> None:
        pause = self.first_s
        while not self._closed:
            self._wake.clear()
            if self._round():
                pause = self.first_s
            elif self.open_handles():
                self._nap(pause)
                pause = min(2 * pause, self.max_s)
            else:
                self._wake.wait()

    def _nap(self, seconds: float) -> None:
        """Wait up to `seconds`, ending early on a new handle, on `close`, or once `cancel` is set."""
        deadline = time.monotonic() + seconds
        while not self._wake.is_set() and not self.cancel.is_set() and (left := deadline - time.monotonic()) > 0:
            self._wake.wait(min(left, CANCEL_CHECK_S))

    def close(self) -> None:
        """Stop following; open handles are left as they are."""
        self._closed = True
        self._wake.set()
        if self._thread is not None:
            self._thread.join()


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

    `lease` is the profile's `lease` section; it is held and not passed to pydra. Fragments are submitted on
    `senders` threads, the sum of the profile's `max_workers` by default; their results are followed by one
    `Follower` thread per run, so neither bounds the fragments in flight.
    """

    def __init__(
        self, *, profile: ExecutionProfile | None = None, work_dir: Path | None = None, senders: int | None = None
    ) -> None:
        """Run under `profile`'s job leases, pydra's files under `work_dir` (`cache_root()/runs/<plan>/pydra`).

        Raises:
            ValueError: `senders` is below 1.
        """
        self.profile = profile if profile is not None else ExecutionProfile.load()
        self.lease: JobLease = self.profile.lease
        self.work_dir = work_dir
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

        A failed element fails only its descendants, which are reported `blocked`; every other element runs. Each
        job's `queued`, `started` and `finished` or `failed` event is appended to `events`. Once `cancel` is set no
        further fragment is sent: running fragments finish and the run raises `RunCancelled`. While it runs,
        `attempt_id()` is the attempt of `events`.

        Raises:
            ValueError: as `compile_plan`, or as `executor.admit`.
            RunCancelled: `cancel` was set before every fragment was sent.
            WorkflowError: pydra errored a job other than an element the executor failed, or named no job.
            BaseException: an exception pydra raised its `WorkflowError` over, such as a cancellation.
            Exception: the first error an executor call raised, or a result it did not give, after the run.
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
        root = self.work_dir if self.work_dir is not None else run_dir_of(placed.hash) / "pydra"
        _private_dir(root)
        store = store if store is not None else SenselabStore(root)
        admission = executor.admit(placed, values)
        cancel = cancel if cancel is not None else threading.Event()
        pool = self.profile.pool
        senders = ThreadPoolExecutor(self.senders, thread_name_prefix="senselab-send")
        follower = Follower(executor, cancel, first_s=pool.first_poll_s, max_s=pool.poll_s)
        run = _Run(placed, admission, values, executor, store, events, cancel, senders, follower)
        keys = plan_keys(placed, values)
        for node, element in run.expected():
            run.log("queued", node, element, key=keys.get((node, element)))
        classes = sorted({p.resource_class for p in placed.placement.nodes.values()})
        try:
            with (
                _no_telemetry(),
                _submitter(run, store, classes) as submitter,
            ):
                submitter(compile_plan(placed), raise_errors=True)
        except WorkflowError as error:
            if run.errors:
                raise run.errors[0] from error
            if error.__context__ is not None:
                raise error.__context__ from error
            if not (error.errors and all(_errored_job(entry) in run.element_failures for entry in error.errors)):
                raise
        finally:
            follower.close()
            senders.shutdown(cancel_futures=True)
            executor.release(admission)
            run.close()
        return _outcome(placed, run)


def _submitter(run: _Run, store: SenselabStore, classes: Sequence[str]) -> Submitter:
    """Return the pydra `Submitter` over `store` with one `ExecutorWorker` per resource class."""
    return Submitter(
        store=store,
        worker=ExecutorWorker(run_state=run),
        workers={name: ExecutorWorker(run_state=run, resource_class=name) for name in classes},
        router=tag_router({name: name for name in classes}),
        on_error="raise_at_end",
    )


def _private_dir(path: Path) -> None:
    """Create `path` and its missing parents, and set `path` to mode `0o700` whatever the umask."""
    path.mkdir(mode=PRIVATE_DIR_MODE, parents=True, exist_ok=True)
    path.chmod(PRIVATE_DIR_MODE)


def _errored_job(entry: str) -> str | None:
    """Return the job an entry of a `WorkflowError` names as having raised, or None for any other entry."""
    found = _ERRORED.match(entry)
    return found[1] if found is not None else None


_ERRORED = re.compile(r"^Job '([^']+)', (?!inputs could not be resolved)")


@contextlib.contextmanager
def _no_telemetry() -> Iterator[None]:
    """Set `NO_TELEMETRY` for the block, unless it is already set, and remove it after."""
    if NO_TELEMETRY in os.environ:
        yield
        return
    os.environ[NO_TELEMETRY] = "1"
    try:
        yield
    finally:
        os.environ.pop(NO_TELEMETRY, None)


def _outcome(placed: Plan, run: _Run) -> RunOutcome:
    """Return the run's outcome; each element given no result is `blocked`."""
    mapped = mapped_nodes(placed.logical.graph)
    rows = len(placed.manifest.rows)
    results, blocked = {}, {}
    for node in order(placed.logical.graph):
        every = [(i,) for i in range(rows)] if node in mapped else [()]
        got = run.results.get(node, {})
        results[node] = tuple(got[e] for e in every if e in got)
        blocked[node] = tuple(e for e in every if e not in got)
    return RunOutcome(placed.hash, results, blocked)


def run_plan(
    placed: Plan, executor: Executor, *, values: Mapping[InputIdentity, object], events: EventLog
) -> RunOutcome:
    """Run `placed` through `executor` on a `PydraRunner` under the packaged default profile."""
    return PydraRunner().run_plan(placed, executor, values=values, events=events)
