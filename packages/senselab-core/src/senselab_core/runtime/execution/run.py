"""A run's lifecycle and its results: `start` a plan, follow it as a `Run`, read its `ResultSet`.

A run lives in `<cache root>/2x/runs/<plan hash>/`: `plan.json` (the plan's documents), `inputs.json` (what each
record's inputs are read from), `manifest.json` (the plan hash and the result store, the same for every attempt),
`placements/<attempt>.json` (one per attempt, never overwritten), `events/<writer>.jsonl` (`EventLog`), `prov/` (the
provenance documents), and `dead/`, `queues/` and `epoch/` for the pool. A `Run` over a directory derives its status
and its results from the events and the result store, and binds its input values from `inputs.json`, so it works from
a run another process started.
"""

import importlib.util
import json
import os
import re
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from senselab_core.kernel.dataset import InputManifest, Record
from senselab_core.kernel.graph import Graph, mapped_nodes, order, split
from senselab_core.kernel.identity import InputIdentity, ReusePolicy
from senselab_core.kernel.placement import PlanPlacement
from senselab_core.kernel.plan import LogicalPlan
from senselab_core.runtime.call import RunConfig, bind_input
from senselab_core.runtime.execution.events import (
    Event,
    EventLog,
    RunStatus,
    read_events,
    run_dir_of,
    status_from,
    terminal_events,
)
from senselab_core.runtime.execution.executor import Executor, FaultRecord
from senselab_core.runtime.planner import ExecutorUnavailable, Plan
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.provenance import ProvStore
from senselab_core.runtime.revision import cache_root, fresh_attempt_id
from senselab_core.runtime.store import ResultStore

PLAN_FILE = "plan.json"
INPUTS_FILE = "inputs.json"
MANIFEST_FILE = "manifest.json"
PLACEMENTS_DIR = "placements"
RESULTS_DIR = "results"
"""The default result store, `<cache root>/results`."""
POOL_DIRS = ("dead", "prov", "queues", "epoch")
POLL_S = 0.5
"""Seconds between reads of the event log while `Run.wait` waits for a run another process drives."""

_DONE = ("finished", "cache_hit")
_LIST = re.compile(r"^list\[(.+)\]$")


class MissingExtra(ImportError):
    """An optional package a call needs is not installed; `extra` names it."""

    def __init__(self, extra: str) -> None:
        """Name the missing package."""
        super().__init__(f"this needs {extra!r}, which is not installed; install it to use this")
        self.extra = extra


class InputChanged(ValueError):
    """An input read from disk no longer has the identity the plan was made with."""


@dataclass(frozen=True)
class Output:
    """What one record gave: each graph output's value by name, with the key digest and output identity it came from."""

    outputs: Mapping[str, object]
    keys: Mapping[str, str] = field(default_factory=dict)
    identities: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class NotRun:
    """A record some job of which did not run: `blocked` (by `because`), `cancelled` or still `pending`."""

    kind: Literal["blocked", "cancelled", "pending"]
    because: FaultRecord | None = None


Given = Output | FaultRecord | NotRun
"""What one record gave."""
Item = tuple[Record, Given]
"""One record and what it gave."""
Entry = tuple[Record, "Given | Callable[[], Given]"]


class ResultSet:
    """The results of a run over records: `(record, output, fault or not-run)` in record order, read when asked.

    An entry may be given as a callable, which is called once, on first access; a `Run`'s results read each record's
    outputs from the result store that way. `shared` holds the graph outputs of nodes that run once for every record
    (a gather and what follows it); `run` is the `Run` they belong to, if any.
    """

    def __init__(
        self,
        entries: Sequence[Entry],
        shared: Mapping[str, "Given | Callable[[], Given]"] | None = None,
        *,
        run: "Run | None" = None,
    ) -> None:
        """Hold `entries` in the order given."""
        self._entries = list(entries)
        self._shared = dict(shared or {})
        self.run = run

    def _given(self, index: int) -> Item:
        record, held = self._entries[index]
        if callable(held):
            held = held()
            self._entries[index] = (record, held)
        return record, held

    @property
    def shared(self) -> Mapping[str, Given]:
        """The graph outputs of the nodes that run once, by name."""
        for name, held in self._shared.items():
            if callable(held):
                self._shared[name] = held()
        return dict(self._shared)  # type: ignore[arg-type]

    def __iter__(self) -> Iterator[Item]:
        """Iterate `(record, what it gave)` in record order."""
        return (self._given(i) for i in range(len(self._entries)))

    def __len__(self) -> int:
        """Return the number of records."""
        return len(self._entries)

    def __getitem__(self, index: int) -> Item:
        """Return the `index`-th record and what it gave."""
        return self._given(index)

    @property
    def failures(self) -> list[tuple[Record, FaultRecord]]:
        """The records that failed, with their faults."""
        return [(r, o) for r, o in self if isinstance(o, FaultRecord)]

    def outputs(self, port: str) -> list[object]:
        """Return output `port` of each record in record order, a record's fault or `NotRun` standing in for it.

        Raises:
            KeyError: a record that gave outputs has no output `port`.
        """
        return [o.outputs[port] if isinstance(o, Output) else o for _, o in self]

    def to_frame(self) -> Any:  # noqa: ANN401 - a pandas DataFrame
        """Return a data frame of the records: an `id` column, a column per output, `fault` and `not_run` columns.

        The index is the records' entities, one level per entity name; records with no entities are indexed by id.

        Raises:
            MissingExtra: pandas is not installed.
        """
        try:
            import pandas  # noqa: PLC0415 - optional
        except ImportError as err:
            raise MissingExtra("pandas") from err
        items = list(self)
        ports = sorted({p for _, o in items if isinstance(o, Output) for p in o.outputs})
        names = sorted({n for r, _ in items for n in r.entities})
        if names:
            index = pandas.MultiIndex.from_tuples(
                [tuple(r.entities.get(n) for n in names) for r, _ in items], names=names
            )
        else:
            index = pandas.Index([r.id for r, _ in items], name="id")
        columns: dict[str, list[object]] = {"id": [r.id for r, _ in items]}
        for port in ports:
            columns[port] = [o.outputs.get(port) if isinstance(o, Output) else None for _, o in items]
        columns["fault"] = [o.fault if isinstance(o, FaultRecord) else None for _, o in items]
        columns["not_run"] = [o.kind if isinstance(o, NotRun) else None for _, o in items]
        return pandas.DataFrame(columns, index=index)


def _write(path: Path, document: object) -> None:
    """Write `document` as JSON to `path` atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    staging.write_text(json.dumps(document, sort_keys=True, indent=1, default=_source), encoding="utf-8")
    os.replace(staging, path)


def _source(value: object) -> str:
    """Return a path input as text; any other value that is not JSON cannot be run from disk."""
    if isinstance(value, os.PathLike):
        return str(os.fspath(value))
    raise ValueError(
        f"an input of type {type(value).__name__} cannot be written to the run directory; give a path or a JSON value"
    )


def load_plan(run_dir: Path) -> Plan:
    """Return the plan written in `run_dir`, with the sources of its inputs; it carries no profile or run config.

    Raises:
        FileNotFoundError: the run has no `plan.json`.
    """
    document = json.loads((Path(run_dir) / PLAN_FILE).read_text(encoding="utf-8"))
    inputs = Path(run_dir) / INPUTS_FILE
    return Plan(
        logical=LogicalPlan.model_validate(document["logical"]),
        manifest=InputManifest.model_validate(document["manifest"]),
        placement=PlanPlacement.model_validate(document["placement"]),
        hash=document["plan_hash"],
        sources=json.loads(inputs.read_text(encoding="utf-8")) if inputs.is_file() else {},
    )


def bind_inputs(run_dir: Path) -> dict[InputIdentity, object]:
    """Return the value of each input identity of the plan in `run_dir`, read from the sources in `inputs.json`.

    Every input is bound again (`bind_input`) and its identity must be the one the plan's manifest holds, so a file
    changed since planning is refused.

    Raises:
        InputChanged: an input has another identity than the manifest's.
        FileNotFoundError, KeyError: the run has no plan or no recorded sources.
    """
    placed = load_plan(run_dir)
    types = {
        port: (found[1] if (found := _LIST.match(declared)) else declared)
        for port, declared in placed.logical.graph.inputs.items()
    }
    rows, shared = placed.sources["rows"], placed.sources["shared"]
    values: dict[InputIdentity, object] = {}
    for row, read in zip(placed.manifest.rows, rows, strict=True):
        for port, expected in row.inputs.items():
            if port in shared and expected in values:
                continue
            identity, value, _ = bind_input(port, types[port], read[port] if port in read else shared[port])
            if identity != expected:
                raise InputChanged(f"input {port!r} of record {row.id!r} changed since the plan was made")
            values[identity] = value
    return values


class _Reader:
    """Reads the graph outputs of one record, or of the run, from the terminal event of each job."""

    def __init__(self, graph: Graph, events: Sequence[Event], store: ResultStore) -> None:
        self.graph = graph
        self.mapped = mapped_nodes(graph)
        self.store = store
        self.latest = terminal_events(events)
        self.held: dict[tuple[str, str], Mapping[str, object] | None] = {}

    def _because(self, index: int | None) -> FaultRecord | None:
        """Return the failure of a job of record `index` (any, for None) that failed, if one did."""
        for _, event in sorted(self.latest.items()):
            if event.kind == "failed" and event.failure is not None and event.job is not None:
                if index is None or event.job.element == (index,):
                    return event.failure
        return None

    def _stopped(self, event: Event | None, index: int | None) -> Given | None:
        """Return why the job whose terminal event is `event` gave nothing, or None if it gave something."""
        if event is None:
            return NotRun("pending")
        if event.kind == "failed":
            return event.failure
        if event.kind == "blocked":
            return NotRun("blocked", self._because(index))
        return NotRun("cancelled") if event.kind == "cancelled" else None

    def gave(self, name: str, index: int | None) -> tuple[str, str, object] | Given:
        """Return graph output `name` of record `index` as (key digest, output identity, value), or why it is not."""
        node, port = split(self.graph.outputs[name])
        event = self.latest.get((node, (index,) if index is not None and node in self.mapped else ()))
        stopped = self._stopped(event, index)
        if stopped is not None:
            return stopped
        assert event is not None and event.key is not None and event.output is not None
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

    def collect(self, names: Sequence[str], index: int | None) -> Given:
        """Return the graph outputs `names` of record `index` (the run's, for None), or why the record gave none.

        A record any of whose jobs failed, was blocked or cancelled gives that even when outputs are held.
        """
        for node in order(self.graph):
            if (index is None) != (node in self.mapped):
                stopped = self._stopped(self.latest.get((node, () if index is None else (index,))), index)
                if stopped is not None:
                    return stopped
        values: dict[str, object] = {}
        keys: dict[str, str] = {}
        identities: dict[str, str] = {}
        for name in names:
            found = self.gave(name, index)
            if isinstance(found, Output | FaultRecord | NotRun):
                return found
            keys[name], identities[name], values[name] = found
        return Output(values, keys, identities)


def _later(reader: _Reader, names: Sequence[str], index: int | None) -> Callable[[], Given]:
    """Return a read of graph outputs `names` of record `index`, to be made when the result is asked for."""
    return lambda: reader.collect(names, index)


class Run:
    """A run of a plan over the directory `run_dir`: its status, its wait, its results and its cancel.

    A run `start` made runs in a background thread; one opened over a directory (`Run(run_dir)`) reads what is on
    disk, written by any process. `store` is where the results are read from, by default the one the manifest names.
    """

    def __init__(self, run_dir: Path, *, store: ResultStore | None = None) -> None:
        """Open the run in `run_dir`."""
        self.run_dir = Path(run_dir)
        self._store = store
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._error: BaseException | None = None

    @property
    def plan(self) -> Plan:
        """The plan written in the run directory (`load_plan`)."""
        return load_plan(self.run_dir)

    def manifest(self) -> dict[str, Any]:
        """Return the manifest (plan hash and result store), `{}` if the run has none."""
        try:
            found = json.loads((self.run_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        return found if isinstance(found, dict) else {}

    def events(self) -> list[Event]:
        """Return the run's events, from every writer, in order."""
        return read_events(self.run_dir)

    def status(self) -> RunStatus:
        """Return the status derived from the events alone."""
        return status_from(self.events())

    def wait(self, timeout: float | None = None, *, poll: float = POLL_S) -> RunStatus:
        """Wait up to `timeout` seconds for the run to be done and return its status.

        A run started here is waited on through its thread; any run is then polled, reading the event log every `poll`
        seconds, until its events say it is done, so a run another process drives can be waited for.

        Raises:
            BaseException: the error that stopped a run started here, other than its cancellation.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
            if not thread.is_alive() and self._error is not None:
                raise self._error
        status = self.status()
        while not status.done and (deadline is None or time.monotonic() < deadline):
            time.sleep(poll if deadline is None else max(0.0, min(poll, deadline - time.monotonic())))
            status = self.status()
        return status

    @property
    def cancelled(self) -> threading.Event:
        """The event `cancel` sets, which the runner reads."""
        return self._cancel

    def fail(self, error: BaseException) -> None:
        """Hold `error`, the first one that stopped the run, for `wait` to raise."""
        if self._error is None:
            self._error = error

    def launch(self, thread: threading.Thread) -> None:
        """Start `thread`, the run's body, and hold it for `wait`."""
        self._thread = thread
        thread.start()

    def cancel(self) -> None:
        """Stop releasing jobs: fragments already sent finish or are left to their lease; no other is sent."""
        self._cancel.set()

    def results(self) -> ResultSet:
        """Return what each record gave, read from the result store as each is asked for.

        The events are read now; a record whose job failed, was blocked or cancelled gives that.

        Raises:
            FileNotFoundError: the run has no `plan.json`.
        """
        placed = self.plan
        graph = placed.logical.graph
        reader = _Reader(graph, self.events(), self._results_store())
        per_record = [n for n in order(graph) if n in reader.mapped]
        outputs = sorted(graph.outputs)
        mine = [name for name in outputs if split(graph.outputs[name])[0] in per_record]
        entries: list[Entry] = [
            (Record(id=row.id, entities=row.entities), _later(reader, mine, i))
            for i, row in enumerate(placed.manifest.rows)
        ]
        shared = {name: _later(reader, [name], None) for name in outputs if name not in mine}
        return ResultSet(entries, shared, run=self)

    def _results_store(self) -> ResultStore:
        if self._store is not None:
            return self._store
        named = self.manifest().get("store")
        return ResultStore(Path(named) if isinstance(named, str) else cache_root() / RESULTS_DIR)


def _executor(placed: Plan, executor: Executor | None, store: ResultStore | None) -> Executor:
    """Return the executor to run `placed` on: the one given, else `InProcess` for a local substrate.

    A plan read from disk carries no profile or run config: the packaged default profile is used if its digest is the
    one the plan was placed under.

    Raises:
        ExecutorUnavailable: no executor is given and the substrate is not local.
        ValueError: the default profile is not the one the plan was placed under.
    """
    if executor is not None:
        return executor
    profile = placed.execution if placed.execution is not None else ExecutionProfile.load()
    if profile.substrate.kind != "local":
        raise ExecutorUnavailable(profile.substrate.kind, "Pool")
    if profile.canonical_digest() != placed.placement.profile_digest:
        raise ValueError(f"plan {placed.hash} was placed under another execution profile; give the executor to run on")
    from senselab_core.runtime.execution.in_process import InProcess  # noqa: PLC0415 - imports the planner

    config = placed.config if placed.config is not None else RunConfig(reuse=ReusePolicy(placed.logical.reuse))
    return InProcess(store=store, profile=profile, run=config)


def _claim_attempt(run_dir: Path, placement: PlanPlacement) -> str:
    """Return a fresh attempt id and write its placement file, which is never replaced: a taken id gets `-<n>`."""
    directory = run_dir / PLACEMENTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    base, n = fresh_attempt_id(), 0
    while True:
        attempt = base if n == 0 else f"{base}-{n}"
        try:
            descriptor = os.open(directory / f"{attempt}.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            n += 1
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(placement.model_dump(mode="json"), sort_keys=True, indent=1))
        return attempt


def _prepare(placed: Plan, run_dir: Path, store: ResultStore | None) -> str:
    """Write the plan, its input sources and the manifest into `run_dir`, claim an attempt and return its id."""
    _write(
        run_dir / PLAN_FILE,
        {
            "plan_hash": placed.hash,
            "logical": placed.logical.model_dump(mode="json"),
            "manifest": placed.manifest.model_dump(mode="json"),
            "placement": placed.placement.model_dump(mode="json"),
        },
    )
    _write(run_dir / INPUTS_FILE, dict(placed.sources))
    for name in POOL_DIRS:
        (run_dir / name).mkdir(exist_ok=True)
    _write(run_dir / MANIFEST_FILE, {"plan_hash": placed.hash, "store": None if store is None else str(store.root)})
    return _claim_attempt(run_dir, placed.placement)


def start(placed: Plan, executor: Executor | None = None, *, store: ResultStore | None = None) -> Run:
    """Start `placed` on `executor` in a background thread, as a new attempt, and return its `Run`.

    With no executor, `InProcess` runs a plan under a local substrate, over `store` (default: the result store under
    the cache root) with the plan's own profile and run config. `store` is also where the run's results are read from,
    so it is given with any other executor that stores results. The input values are bound from the run directory
    (`bind_inputs`), so every input must be a path or a JSON value.

    Raises:
        ExecutorUnavailable: no executor is given and the substrate is not local.
        ValueError: an input cannot be written to the run directory, or as `admit`.
        InputChanged: an input file changed since the plan was made.
        MissingExtra: pydra is not installed.
    """
    if importlib.util.find_spec("pydra") is None:
        raise MissingExtra("pydra")
    from senselab_core.runtime.execution.runner import PydraRunner, RunCancelled  # noqa: PLC0415 - needs pydra

    provenance = ProvStore()
    if executor is None and store is None:
        store = ResultStore(cache_root() / RESULTS_DIR, provenance=provenance)
    executor = _executor(placed, executor, store)
    run_dir = run_dir_of(placed.hash)
    attempt = _prepare(placed, run_dir, store)
    values = bind_inputs(run_dir)
    run = Run(run_dir, store=store)
    log = EventLog(run_dir, attempt)
    runner = PydraRunner(profile=placed.execution, work_dir=run_dir / "pydra")
    written = provenance if store is None else store.provenance

    def body() -> None:
        try:
            runner.run_plan(placed, executor, values=values, events=log, cancel=run.cancelled)
        except BaseException as err:  # noqa: BLE001 - held for `wait`
            if not isinstance(err, RunCancelled):
                run.fail(err)
        try:
            written.write(run_dir)
        except Exception as err:  # noqa: BLE001 - held for `wait`
            run.fail(err)

    run.launch(threading.Thread(target=body, name=f"senselab-run-{placed.hash[:8]}", daemon=True))
    return run
