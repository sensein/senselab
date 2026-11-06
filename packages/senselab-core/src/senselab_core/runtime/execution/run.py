"""A run's lifecycle and its results: `start` a plan, follow it as a `Run`, read its `ResultSet`.

A run lives in `<cache root>/2x/runs/<plan hash>/`: `plan.json` (the logical plan and its input manifest, written
once), `inputs.json` (what each
record's inputs are read from, merged across attempts), `manifest.json` (the plan hash and the latest attempt's
result store),
`placements/<attempt>.json` (one per attempt, never overwritten), `attempts/<n>` (the attempt id claimed `n`-th),
`events/<writer>.jsonl` (`EventLog`), `prov/` (the
provenance documents), and `dead/`, `queues/` and `epoch/` for the pool. A `Run` over a directory derives its status
and its results from the events and the result store, and binds its input values from `inputs.json`, so it works from
a run another process started, unless an input was held in memory by the process that planned it.
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
from typing import Any, Literal, cast

from senselab_core.kernel.dataset import InputManifest, Record
from senselab_core.kernel.graph import Graph, mapped_nodes, order, split
from senselab_core.kernel.identity import InputIdentity, ReusePolicy
from senselab_core.kernel.placement import PlanPlacement
from senselab_core.kernel.plan import LogicalPlan
from senselab_core.kernel.signals import StaleSource
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
from senselab_core.runtime.execution.queue import RUN_QUEUES, QueueSet
from senselab_core.runtime.execution.records import add_plan_entity
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
ATTEMPTS_DIR = "attempts"
POOL_DIRS = ("dead", "prov", RUN_QUEUES, "epoch")
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


class InputChanged(StaleSource):
    """Inputs no longer of the identity the plan was made with, or unreadable; `changed` is (record, port)."""

    def __init__(self, changed: Sequence[tuple[str, str]]) -> None:
        """Name the records and ports whose inputs changed."""
        self.changed = tuple(changed)
        ports = sorted({port for _, port in self.changed})
        records = sorted({record for record, _ in self.changed})
        super().__init__(f"inputs {ports} of records {records[:5]} changed since the plan was made; plan them again")


class ConflictingSources(ValueError):
    """Two durable sources of one input read different contents; `conflicts` is (record, port), record `*` if shared."""

    def __init__(self, conflicts: Sequence[tuple[str, str]]) -> None:
        """Name the records and ports whose recorded and given sources differ."""
        self.conflicts = tuple(conflicts)
        super().__init__(
            f"inputs {sorted({p for _, p in self.conflicts})} of records {sorted({r for r, _ in self.conflicts})[:5]}"
            " are recorded with a source that now reads other content than the one given; this run is not that plan's"
        )


class InputsUnavailable(ValueError):
    """Inputs held in memory by the process that planned the run are not in this one; `missing` is (record, port)."""

    def __init__(self, missing: Sequence[tuple[str, str]]) -> None:
        """Name the records and ports whose inputs are missing."""
        self.missing = tuple(missing)
        ports = sorted({port for _, port in self.missing})
        records = sorted({record for record, _ in self.missing})
        super().__init__(
            f"inputs {ports} of records {records[:5]} were held in memory by the process that planned the run; run the"
            " plan in that process, or give them as paths or JSON values"
        )


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


def latest_placement(run_dir: Path) -> PlanPlacement:
    """Return the placement of the run's latest claimed attempt, `placements/<attempt>.json`.

    Raises:
        FileNotFoundError: the run has no claimed attempt.
    """
    directory = Path(run_dir) / ATTEMPTS_DIR
    claimed = sorted((int(p.name), p) for p in directory.iterdir() if p.name.isdigit()) if directory.is_dir() else []
    for _, path in reversed(claimed):
        attempt = path.read_text(encoding="utf-8")
        if attempt:
            placement = Path(run_dir) / PLACEMENTS_DIR / f"{attempt}.json"
            return PlanPlacement.model_validate(json.loads(placement.read_text(encoding="utf-8")))
    raise FileNotFoundError(f"run {Path(run_dir).name} has no claimed attempt, so no placement")


def load_plan(run_dir: Path) -> Plan:
    """Return the plan written in `run_dir` under its latest attempt's placement, with the sources of its inputs.

    It carries no profile or run config.

    Raises:
        FileNotFoundError: the run has no `plan.json`, or no claimed attempt.
    """
    document = json.loads((Path(run_dir) / PLAN_FILE).read_text(encoding="utf-8"))
    inputs = Path(run_dir) / INPUTS_FILE
    return Plan(
        logical=LogicalPlan.model_validate(document["logical"]),
        manifest=InputManifest.model_validate(document["manifest"]),
        placement=latest_placement(run_dir),
        hash=document["plan_hash"],
        sources=json.loads(inputs.read_text(encoding="utf-8")) if inputs.is_file() else {},
    )


def _memory(source: object) -> str | None:
    """Return the identity a `{"memory": <identity>}` source names, None for a path or JSON value."""
    if isinstance(source, Mapping) and set(source) == {"memory"} and isinstance(source["memory"], str):
        return source["memory"]
    return None


def _port_types(placed: Plan) -> dict[str, str]:
    """Return the type one record gives of each graph input of `placed`."""
    return {
        port: (found[1] if (found := _LIST.match(declared)) else declared)
        for port, declared in placed.logical.graph.inputs.items()
    }


def bind_sources(placed: Plan) -> dict[InputIdentity, object]:
    """Return the value of each input identity of `placed`, bound from its sources and its live values.

    A path or JSON source is bound again (`bind_input`) and must give the identity the manifest holds, so a file
    changed since planning is refused; a `{"memory": <identity>}` source takes the value `placed.live` holds. Every
    record is checked before anything is raised.

    Raises:
        InputChanged: inputs have another identity than the manifest's, or cannot be read; it names every one.
        InputsUnavailable: an input held in memory is not in `placed.live`.
        KeyError: the plan records no sources.
    """
    types = _port_types(placed)
    rows, shared = placed.sources["rows"], placed.sources["shared"]
    values: dict[InputIdentity, object] = {}
    missing: list[tuple[str, str]] = []
    changed: list[tuple[str, str]] = []
    for row, read in zip(placed.manifest.rows, rows, strict=True):
        for port, expected in row.inputs.items():
            if port in shared and expected in values:
                continue
            source = read[port] if port in read else shared[port]
            held = _memory(source)
            if held is not None and held == expected and held not in placed.live:
                missing.append((row.id, port))
                continue
            bound = _bound(port, types[port], source, placed.live) if held in (None, expected) else None
            if bound is None or bound[0] != expected:
                changed.append((row.id, port))
                continue
            values[expected] = bound[1]
    if changed:
        raise InputChanged(changed)
    if missing:
        raise InputsUnavailable(missing)
    return values


def _bound(port: str, kind: str, source: object, live: Mapping[InputIdentity, object]) -> tuple[str, object] | None:
    """Return the identity and value of an input read from `source`, None when its file cannot be read.

    A `{"memory": <identity>}` source is that identity's value in `live`, which holds it.
    """
    held = _memory(source)
    if held is not None:
        return held, live[held]
    try:
        identity, value, _ = bind_input(port, kind, source)
    except OSError:
        return None
    return identity, value


def bind_inputs(run_dir: Path) -> dict[InputIdentity, object]:
    """Return the value of each input identity of the plan in `run_dir` (`bind_sources` of `load_plan`).

    Raises:
        InputChanged, InputsUnavailable: as `bind_sources`; an input held in memory is never on disk.
        FileNotFoundError, KeyError: the run has no plan or no recorded sources.
    """
    return bind_sources(load_plan(run_dir))


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


def _claim_ordinal(run_dir: Path, attempt: str) -> int:
    """Return the next free ordinal `n` of the run's attempts, claimed by creating `attempts/<n>` holding `attempt`."""
    directory = run_dir / ATTEMPTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    ordinal = 1 + sum(1 for _ in directory.iterdir())
    while True:
        try:
            descriptor = os.open(directory / str(ordinal), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            ordinal += 1
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(attempt)
        return ordinal


def _claim_attempt(run_dir: Path, placement: PlanPlacement) -> tuple[str, int]:
    """Return a fresh attempt id and its ordinal, its placement file written and never replaced (a taken id: `-<n>`)."""
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
        return attempt, _claim_ordinal(run_dir, attempt)


def _write_once(path: Path, document: Mapping[str, object]) -> None:
    """Write `document` as JSON to `path` unless a file is there; one that is must name the same `plan_hash`.

    Raises:
        ValueError: `path` holds the document of another plan.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    staging.write_text(json.dumps(document, sort_keys=True, indent=1), encoding="utf-8")
    try:
        os.link(staging, path)
    except FileExistsError:
        held = json.loads(path.read_text(encoding="utf-8")).get("plan_hash")
        if held != document["plan_hash"]:
            raise ValueError(f"{path.parent.name} holds the plan {held}, not {document['plan_hash']}") from None
    finally:
        staging.unlink()


def _durable(source: object) -> object:
    """Return `source` as it is written to `inputs.json`: a path as text, a JSON value as itself."""
    return json.loads(json.dumps(source, default=_source))


def _kept(old: object, new: object, port: str, kind: str, expected: InputIdentity) -> object | None:
    """Return the source an input keeps, `old` recorded and `new` given; None when the two durable ones conflict.

    A durable source (a path or JSON value) is never replaced by a `{"memory": ...}` marker, and replaces one. Two
    different durable sources keep the new one unless the old one still reads, as another identity than `expected`.
    """
    if old is None or old == new or _memory(old) is not None:
        return new
    if _memory(new) is not None:
        return old
    try:
        identity = bind_input(port, kind, old)[0]
    except OSError:
        return new
    return new if identity == expected else None


def merged_sources(placed: Plan, run_dir: Path) -> dict[str, Any]:
    """Return the input sources of `placed` merged, per record and port, over those `run_dir` records (`_kept`).

    Raises:
        ConflictingSources: a recorded durable source reads another identity than the one the plan holds.
    """
    given = cast("dict[str, Any]", _durable(dict(placed.sources)))
    path = Path(run_dir) / INPUTS_FILE
    if not path.is_file():
        return given
    held = json.loads(path.read_text(encoding="utf-8"))
    types = _port_types(placed)
    rows, conflicts = placed.manifest.rows, []
    merged_rows = []
    for row, old, new in zip(rows, held["rows"], given["rows"], strict=True):
        kept = {p: _kept(old.get(p), s, p, types[p], row.inputs[p]) for p, s in new.items()}
        conflicts += [(row.id, p) for p, s in kept.items() if s is None]
        merged_rows.append(kept)
    shared = {p: _kept(held["shared"].get(p), s, p, types[p], rows[0].inputs[p]) for p, s in given["shared"].items()}
    conflicts += [("*", p) for p, s in shared.items() if s is None]
    if conflicts:
        raise ConflictingSources(conflicts)
    return {"rows": merged_rows, "shared": shared}


def _prepare(placed: Plan, run_dir: Path, store: ResultStore | None) -> tuple[str, int]:
    """Write the plan once, its input sources and the manifest into `run_dir`, then claim an attempt.

    Raises:
        ValueError: `run_dir` holds another plan.
        ConflictingSources: as `merged_sources`; nothing is written.
    """
    sources = merged_sources(placed, run_dir)
    _write_once(
        run_dir / PLAN_FILE,
        {
            "plan_hash": placed.hash,
            "logical": placed.logical.model_dump(mode="json"),
            "manifest": placed.manifest.model_dump(mode="json"),
        },
    )
    _write(run_dir / INPUTS_FILE, sources)
    for name in POOL_DIRS:
        (run_dir / name).mkdir(exist_ok=True)
    _write(run_dir / MANIFEST_FILE, {"plan_hash": placed.hash, "store": None if store is None else str(store.root)})
    return _claim_attempt(run_dir, placed.placement)


def start(
    placed: Plan,
    executor: Executor | None = None,
    *,
    store: ResultStore | None = None,
    values: Mapping[InputIdentity, object] | None = None,
) -> Run:
    """Start `placed` on `executor` in a background thread, as a new attempt, and return its `Run`.

    With no executor, `InProcess` runs a plan under a local substrate, over `store` (default: the result store under
    the cache root) with the plan's own profile and run config. `store` is also where the run's results are read from,
    so it is given with any other executor that stores results, and it is the store that executor writes: the plan's
    `senselab:Plan` entity is added to its provenance. The input values are bound (`bind_sources`) before anything
    is written: from their paths or JSON values, or from the values the plan holds in memory; `values` gives them
    already bound. The sources are merged into `inputs.json` (`merged_sources`), also before anything is written.
    Every item in the run's `queues/` with neither a slot nor an outcome is put again (`QueueSet.reconcile`).

    Raises:
        ExecutorUnavailable: no executor is given and the substrate is not local.
        ValueError: as `admit`.
        InputChanged: an input changed since the plan was made.
        InputsUnavailable: an input held in memory is not held by `placed` (a plan read from disk).
        ConflictingSources: a source the run records reads other content than the plan's.
        MissingExtra: pydra is not installed.
    """
    if importlib.util.find_spec("pydra") is None:
        raise MissingExtra("pydra")
    from senselab_core.runtime.execution.runner import PydraRunner, RunCancelled  # noqa: PLC0415 - needs pydra

    provenance = ProvStore()
    if executor is None and store is None:
        store = ResultStore(cache_root() / RESULTS_DIR, provenance=provenance)
    executor = _executor(placed, executor, store)
    values = bind_sources(placed) if values is None else values
    run_dir = run_dir_of(placed.hash)
    attempt, ordinal = _prepare(placed, run_dir, store)
    QueueSet(run_dir / RUN_QUEUES, placed.execution).reconcile()
    run = Run(run_dir, store=store)
    log = EventLog(run_dir, attempt, ordinal)
    runner = PydraRunner(profile=placed.execution, work_dir=run_dir / "pydra")
    written = provenance if store is None else store.provenance
    add_plan_entity(written, placed.hash)

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
