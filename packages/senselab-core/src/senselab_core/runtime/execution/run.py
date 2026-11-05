"""A run's lifecycle and its results: `start` a plan, follow it as a `Run`, read its `ResultSet`.

A run lives in `<cache root>/2x/runs/<plan hash>/`: `plan.json` (the plan's documents), `manifest.json` (the attempt's
state), `placements/<attempt>.json`, `events/<writer>.jsonl` (`EventLog`), `prov/` (the provenance documents), and
`dead/`, `queues/` and `epoch/` for the pool. A `Run` over a directory derives its status and its results from the
events and the result store alone, so it reads a run started by another process.
"""

import contextlib
import importlib.util
import json
import os
import threading
import time
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from senselab_core.kernel.dataset import InputManifest, Record
from senselab_core.kernel.graph import Graph, mapped_nodes, order, split
from senselab_core.kernel.plan import LogicalPlan
from senselab_core.runtime.execution.events import Event, EventLog, RunStatus, read_events, run_dir_of, status_from
from senselab_core.runtime.execution.executor import Executor, FaultRecord
from senselab_core.runtime.planner import ExecutorUnavailable, Plan
from senselab_core.runtime.provenance import ProvStore
from senselab_core.runtime.revision import attempt_id, cache_root
from senselab_core.runtime.store import ResultStore

PLAN_FILE = "plan.json"
MANIFEST_FILE = "manifest.json"
PLACEMENTS_DIR = "placements"
RESULTS_DIR = "results"
"""The default result store, `<cache root>/results`."""
POOL_DIRS = ("dead", "prov", "queues", "epoch")

_DONE = ("finished", "cache_hit")


class MissingExtra(ImportError):
    """An optional package a call needs is not installed; `extra` names it."""

    def __init__(self, extra: str) -> None:
        """Name the missing package."""
        super().__init__(f"this needs {extra!r}, which is not installed; install it to use this")
        self.extra = extra


@dataclass(frozen=True)
class Output:
    """What one record gave: each graph output's value by name, with the key digest and output identity it came from."""

    outputs: Mapping[str, object]
    keys: Mapping[str, str] = field(default_factory=dict)
    identities: Mapping[str, str] = field(default_factory=dict)


Item = tuple[Record, Output | FaultRecord]
"""One record and what it gave."""


class ResultSet:
    """The results of a run over records: `(record, output or fault)` in record order.

    `shared` holds the graph outputs of nodes that run once for every record (a gather and what follows it).
    """

    def __init__(self, items: Iterable[Item], shared: Mapping[str, Output | FaultRecord] | None = None) -> None:
        """Hold `items` in the order given."""
        self._items = tuple(items)
        self.shared: Mapping[str, Output | FaultRecord] = dict(shared or {})

    def __iter__(self) -> Iterator[Item]:
        """Iterate `(record, output or fault)` in record order."""
        return iter(self._items)

    def __len__(self) -> int:
        """Return the number of records."""
        return len(self._items)

    def __getitem__(self, index: int) -> Item:
        """Return the `index`-th record and what it gave."""
        return self._items[index]

    @property
    def failures(self) -> list[tuple[Record, FaultRecord]]:
        """The records that failed, with their faults."""
        return [(r, o) for r, o in self._items if isinstance(o, FaultRecord)]

    def outputs(self, port: str) -> list[object]:
        """Return output `port` of each record in record order, a record's `FaultRecord` standing in for its value.

        Raises:
            KeyError: a record that gave outputs has no output `port`.
        """
        return [o if isinstance(o, FaultRecord) else o.outputs[port] for _, o in self._items]

    def to_frame(self) -> Any:  # noqa: ANN401 - a pandas DataFrame
        """Return a data frame of the records: an `id` column, a column per output and a `fault` column.

        The index is the records' entities, one level per entity name; records with no entities are indexed by id.

        Raises:
            MissingExtra: pandas is not installed.
        """
        try:
            import pandas  # noqa: PLC0415 - optional
        except ImportError as err:
            raise MissingExtra("pandas") from err
        ports = sorted({p for _, o in self._items if isinstance(o, Output) for p in o.outputs})
        names = sorted({n for r, _ in self._items for n in r.entities})
        if names:
            index = pandas.MultiIndex.from_tuples(
                [tuple(r.entities.get(n) for n in names) for r, _ in self._items], names=names
            )
        else:
            index = pandas.Index([r.id for r, _ in self._items], name="id")
        columns: dict[str, list[object]] = {"id": [r.id for r, _ in self._items]}
        for port in ports:
            columns[port] = [o.outputs.get(port) if isinstance(o, Output) else None for _, o in self._items]
        columns["fault"] = [o.fault if isinstance(o, FaultRecord) else None for _, o in self._items]
        return pandas.DataFrame(columns, index=index)


def _write(path: Path, document: object) -> None:
    """Write `document` as JSON to `path` atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    staging.write_text(json.dumps(document, sort_keys=True, indent=1), encoding="utf-8")
    os.replace(staging, path)


def _blocked() -> FaultRecord:
    return FaultRecord(
        fault="infrastructure",
        type="NotRun",
        message="the job has no result: an upstream job failed, or the run stopped before it",
        gap=True,
        charged=False,
    )


class _Reader:
    """Reads the graph outputs of one record, or of the run, from the latest terminal event of each job."""

    def __init__(self, graph: Graph, events: Sequence[Event], store: ResultStore) -> None:
        self.graph = graph
        self.mapped = mapped_nodes(graph)
        self.store = store
        self.latest: dict[tuple[str, tuple[int, ...]], Event] = {}
        for event in events:
            if event.job is not None and event.kind in (*_DONE, "failed"):
                self.latest[(event.node, event.job.element)] = event
        self.held: dict[tuple[str, str], Mapping[str, object] | None] = {}

    def _failure(self, index: int | None) -> FaultRecord:
        """Return the failure of a job of record `index` (any, for None) that failed, else a not-run fault."""
        for _, event in sorted(self.latest.items()):
            if event.kind == "failed" and event.failure is not None and event.job is not None:
                if index is None or event.job.element == (index,):
                    return event.failure
        return _blocked()

    def gave(self, name: str, index: int | None) -> tuple[str, str, object] | FaultRecord:
        """Return graph output `name` of record `index` as (key digest, output identity, value), or why it is not."""
        node, port = split(self.graph.outputs[name])
        event = self.latest.get((node, (index,) if index is not None and node in self.mapped else ()))
        if event is None or event.kind == "failed":
            return event.failure if event is not None and event.failure is not None else self._failure(index)
        assert event.key is not None and event.output is not None
        wanted = (event.key, event.output)
        if wanted not in self.held:
            self.held[wanted] = self.store.outputs_of(*wanted)
        found = self.held[wanted]
        if found is None or port not in found:
            return FaultRecord(
                fault="infrastructure",
                type="MissingOutput",
                message=f"output {port!r} of {node!r} is not in the result store",
                gap=True,
                charged=False,
            )
        return event.key, event.output, found[port]

    def collect(self, names: Sequence[str], index: int | None) -> Output | FaultRecord:
        """Return the graph outputs `names` of record `index` (the run's, for None), or the first fault.

        A record any of whose jobs failed, or has none finished, gives a fault even when its outputs are held.
        """
        for node in order(self.graph):
            if (index is None) != (node in self.mapped):
                event = self.latest.get((node, () if index is None else (index,)))
                if event is None or event.kind == "failed":
                    return event.failure if event is not None and event.failure is not None else self._failure(index)
        values: dict[str, object] = {}
        keys: dict[str, str] = {}
        identities: dict[str, str] = {}
        for name in names:
            found = self.gave(name, index)
            if isinstance(found, FaultRecord):
                return found
            keys[name], identities[name], values[name] = found
        return Output(values, keys, identities)


class Run:
    """A run of a plan over the directory `run_dir`: its status, its wait, its results and its cancel.

    A run `start` made runs in a background thread; one opened over a directory (`Run(run_dir)`) reads what is on
    disk, written by any process. `store` is where the results are read from, by default the one the manifest names.
    """

    def __init__(self, run_dir: Path, *, store: ResultStore | None = None, plan: Plan | None = None) -> None:
        """Open the run in `run_dir`."""
        self.run_dir = Path(run_dir)
        self._store = store
        self._plan = plan
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._error: BaseException | None = None

    @property
    def plan(self) -> Plan | None:
        """The plan this run was started from in this process, None for a run opened over a directory."""
        return self._plan

    def manifest(self) -> dict[str, Any]:
        """Return the manifest of the latest attempt, `{}` if the run has none."""
        try:
            found = json.loads((self.run_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        return found if isinstance(found, dict) else {}

    def events(self) -> list[Event]:
        """Return the run's events, from every writer, in order."""
        return read_events(self.run_dir)

    def status(self) -> RunStatus:
        """Return the status derived from the events; `done` once the latest attempt finished."""
        found = status_from(self.events())
        state = self.manifest().get("state")
        return RunStatus(found.counts, found.faults, found.broken, state not in (None, "running"), found.nodes)

    def wait(self, timeout: float | None = None) -> RunStatus:
        """Wait up to `timeout` seconds for the run's thread to finish and return its status.

        A run opened over a directory has no thread here; its status is returned at once.

        Raises:
            BaseException: the error that stopped the run, other than its cancellation.
        """
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
            if not thread.is_alive() and self._error is not None:
                raise self._error
        return self.status()

    @property
    def cancelled(self) -> threading.Event:
        """The event `cancel` sets, which the runner reads."""
        return self._cancel

    def fail(self, error: BaseException) -> None:
        """Hold `error`, the one that stopped the run, for `wait` to raise."""
        self._error = error

    def launch(self, thread: threading.Thread) -> None:
        """Start `thread`, the run's body, and hold it for `wait`."""
        self._thread = thread
        thread.start()

    def cancel(self) -> None:
        """Stop releasing jobs: fragments already sent finish or are left to their lease; no other is sent."""
        self._cancel.set()

    def results(self) -> ResultSet:
        """Return what each record gave, from the events and the result store.

        A record whose job failed, or has no result, gives its `FaultRecord`.

        Raises:
            FileNotFoundError: the run has no `plan.json`.
        """
        document = json.loads((self.run_dir / PLAN_FILE).read_text(encoding="utf-8"))
        graph = LogicalPlan.model_validate(document["logical"]).graph
        manifest = InputManifest.model_validate(document["manifest"])
        records = self._plan.records if self._plan is not None else ()
        if len(records) != len(manifest.rows):
            records = tuple(Record(id=row.id, entities=row.entities) for row in manifest.rows)
        reader = _Reader(graph, self.events(), self._results_store())
        per_record = [n for n in order(graph) if n in reader.mapped]
        outputs = sorted(graph.outputs)
        mine = [name for name in outputs if split(graph.outputs[name])[0] in per_record]
        items = [(record, reader.collect(mine, i)) for i, record in enumerate(records)]
        shared = {name: reader.collect([name], None) for name in outputs if name not in mine}
        return ResultSet(items, shared)

    def _results_store(self) -> ResultStore:
        if self._store is not None:
            return self._store
        named = self.manifest().get("store")
        return ResultStore(Path(named) if isinstance(named, str) else cache_root() / RESULTS_DIR)


def _executor(
    placed: Plan, executor: Executor | None, store: ResultStore | None
) -> tuple[Executor, ResultStore | None]:
    """Return the executor to run `placed` on and the store its results are read from."""
    if executor is not None:
        return executor, store if store is not None else getattr(executor, "store", None)
    if placed.execution is None or placed.config is None:
        raise ValueError("a plan with no execution profile or run config needs an executor to run on")
    if placed.execution.substrate.kind != "local":
        raise ExecutorUnavailable(placed.execution.substrate.kind, "Pool")
    from senselab_core.runtime.execution.in_process import InProcess  # noqa: PLC0415 - imports the planner

    store = store if store is not None else ResultStore(cache_root() / RESULTS_DIR, provenance=ProvStore())
    return InProcess(store=store, profile=placed.execution, run=placed.config), store


def _prepare(placed: Plan, run_dir: Path, store: ResultStore | None) -> dict[str, Any]:
    """Write the plan, this attempt's placement and its manifest into `run_dir`, and return the manifest."""
    attempt = attempt_id()
    _write(
        run_dir / PLAN_FILE,
        {
            "plan_hash": placed.hash,
            "logical": placed.logical.model_dump(mode="json"),
            "manifest": placed.manifest.model_dump(mode="json"),
            "placement": placed.placement.model_dump(mode="json"),
        },
    )
    _write(run_dir / PLACEMENTS_DIR / f"{attempt}.json", placed.placement.model_dump(mode="json"))
    for name in POOL_DIRS:
        (run_dir / name).mkdir(exist_ok=True)
    manifest: dict[str, Any] = {
        "plan_hash": placed.hash,
        "attempt": attempt,
        "state": "running",
        "started_ns": time.time_ns(),
        "finished_ns": None,
        "store": None if store is None else str(store.root),
        "error": None,
    }
    _write(run_dir / MANIFEST_FILE, manifest)
    return manifest


def start(placed: Plan, executor: Executor | None = None, *, store: ResultStore | None = None) -> Run:
    """Start `placed` on `executor` in a background thread and return its `Run`.

    With no executor, `InProcess` runs a plan under a local substrate, over `store` or the result store under the cache
    root, under the plan's own profile and run config. `store` is where the run's results are read from; for a given
    executor it is the executor's own `store` attribute, if it has one.

    Raises:
        ExecutorUnavailable: no executor is given and the substrate is not local.
        ValueError: no executor is given for a plan that carries no execution profile or run config, or as `admit`.
        MissingExtra: pydra is not installed.
    """
    if importlib.util.find_spec("pydra") is None:
        raise MissingExtra("pydra")
    from senselab_core.runtime.execution.runner import PydraRunner, RunCancelled  # noqa: PLC0415 - needs pydra

    executor, store = _executor(placed, executor, store)
    run_dir = run_dir_of(placed.hash)
    manifest = _prepare(placed, run_dir, store)
    run = Run(run_dir, store=store, plan=placed)
    log = EventLog(run_dir)
    runner = PydraRunner(profile=placed.execution, work_dir=run_dir / "pydra")

    def body() -> None:
        state, error = "error", None
        try:
            outcome = runner.run_plan(placed, executor, values=placed.values, events=log, cancel=run.cancelled)
            state = "done" if outcome.done else "failed"
        except RunCancelled:
            state = "cancelled"
        except BaseException as err:  # noqa: BLE001 - held for `wait`
            run.fail(err)
            error = f"{type(err).__name__}: {err}"
        finally:
            provenance = getattr(store, "provenance", None)
            if provenance is not None:
                with contextlib.suppress(OSError):
                    provenance.write(run_dir)
            _write(run_dir / MANIFEST_FILE, {**manifest, "state": state, "finished_ns": time.time_ns(), "error": error})

    run.launch(threading.Thread(target=body, name=f"senselab-run-{placed.hash[:8]}", daemon=True))
    return run
