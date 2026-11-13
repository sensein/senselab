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

import contextlib
import dataclasses
import importlib.util
import json
import logging
import os
import re
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from senselab_worker.canonical import jcs_dumps

from senselab_core.kernel.dataset import InputManifest, Record
from senselab_core.kernel.formats import PLACEMENT_FORMAT, RETIRED_PLACEMENT_FORMATS, FormatVerdict, check
from senselab_core.kernel.graph import Graph, mapped_nodes, order, split
from senselab_core.kernel.identity import InputIdentity, ReusePolicy
from senselab_core.kernel.placement import PlanPlacement
from senselab_core.kernel.plan import LogicalPlan
from senselab_core.kernel.signals import StaleSource
from senselab_core.runtime.call import InputUnreadable, RunConfig, bind_input
from senselab_core.runtime.clock import Clock, SystemClock
from senselab_core.runtime.execution.epoch import EPOCH_DIR, Epoch, StaleCoordinator, run_lease
from senselab_core.runtime.execution.events import (
    Event,
    EventLog,
    RunStatus,
    read_events,
    run_dir_of,
    scan_events,
    status_from,
    terminal_events,
)
from senselab_core.runtime.execution.executor import Executor, FaultRecord
from senselab_core.runtime.execution.queue import RUN_QUEUES, QueueSet
from senselab_core.runtime.execution.records import add_plan_entity
from senselab_core.runtime.execution.telemetry import Actuals, Placements
from senselab_core.runtime.execution_settings import JobLease
from senselab_core.runtime.licenses import LicensePolicy, laxer_than, policy_digest, policy_document, policy_of
from senselab_core.runtime.locks import Lease, LeaseUnavailable
from senselab_core.runtime.planner import Plan, check_storage
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.provenance import ProvStore
from senselab_core.runtime.revision import cache_root, fresh_attempt_id
from senselab_core.runtime.site_placement import Placer
from senselab_core.runtime.storage import PRIVATE_DIR, private_dir, publish, write_atomically
from senselab_core.runtime.store import ResultStore

PLAN_FILE = "plan.json"
INPUTS_FILE = f"{PRIVATE_DIR}/inputs.json"
MANIFEST_FILE = f"{PRIVATE_DIR}/manifest.json"
PLACEMENTS_DIR = f"{PRIVATE_DIR}/placements"
PYDRA_DIR = f"{PRIVATE_DIR}/pydra"
"""The pydra runner's job and error pickles, which hold the store path and tracebacks."""
LICENSES_FILE = "licenses.json"
"""The run's license policy, written once at its first start (`run-licenses@1`)."""
RESULTS_DIR = "results"
"""The default result store, `<cache root>/results`."""
ATTEMPTS_DIR = "attempts"
POOL_DIRS = ("dead", "prov", RUN_QUEUES, EPOCH_DIR)
_LIST = re.compile(r"^list\[(.+)\]$")

logger = logging.getLogger("senselab")


class MissingExtra(ImportError):
    """An optional package a call needs is not installed; `extra` names it."""

    def __init__(self, extra: str) -> None:
        """Name the missing package."""
        super().__init__(f"this needs {extra!r}, which is not installed; install it to use this")
        self.extra = extra


class RunPrivate(PermissionError):
    """Another account's run: its private state (manifest, inputs, placement) is not readable here."""

    def __init__(self, run_dir: Path) -> None:
        """Name the run directory and the argument that names its result store."""
        super().__init__(
            f"run {Path(run_dir).name}'s private state belongs to another account; name its result store with "
            "Run(run_dir, store=<its result store>)"
        )
        self.run_dir = Path(run_dir)


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


class SourcesUnrecorded(ValueError):
    """The plan records no sources of its inputs, as one read from a run directory without its inputs file."""

    def __init__(self) -> None:
        """State that no input sources are recorded."""
        super().__init__("the plan records no input sources")


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
    write_atomically(path, json.dumps(document, sort_keys=True, indent=1, default=_source).encode())


def _source(value: object) -> str:
    """Return a path input as text; any other value that is not JSON cannot be run from disk."""
    if isinstance(value, os.PathLike):
        return str(os.fspath(value))
    raise ValueError(
        f"an input of type {type(value).__name__} cannot be written to the run directory; give a path or a JSON value"
    )


class PlacementFormatRetired(ValueError):
    """A run directory holds a placement of a format this build does not read: the run was made by an older build."""

    def __init__(self, found: str, path: Path) -> None:
        """Name the declared format and the run."""
        super().__init__(
            f"run {path.parent.parent.name} holds a placement of the retired format {found}, which this build does not"
            f" read ({PLACEMENT_FORMAT} is current); it was made by an older build, so plan and run it again"
        )
        self.found = found


def read_placement(path: Path) -> PlanPlacement:
    """Return the placement written at `path`.

    Raises:
        PlacementFormatRetired: it is of a retired format.
    """
    document = json.loads(path.read_text(encoding="utf-8"))
    found = document.get("format") if isinstance(document, dict) else None
    if check(PLACEMENT_FORMAT, found, RETIRED_PLACEMENT_FORMATS) is FormatVerdict.RETIRED:
        raise PlacementFormatRetired(str(found), path)
    return PlanPlacement.model_validate(document)


def latest_placement(run_dir: Path) -> PlanPlacement:
    """Return the placement of the run's latest claimed attempt, `placements/<attempt>.json`.

    Raises:
        FileNotFoundError: the run has no claimed attempt.
        PlacementFormatRetired: the placement is of a retired format.
    """
    directory = Path(run_dir) / ATTEMPTS_DIR
    claimed = sorted((int(p.name), p) for p in directory.iterdir() if p.name.isdigit()) if directory.is_dir() else []
    for _, path in reversed(claimed):
        attempt = path.read_text(encoding="utf-8")
        if attempt:
            placement = Path(run_dir) / PLACEMENTS_DIR / f"{attempt}.json"
            return read_placement(placement)
    raise FileNotFoundError(f"run {Path(run_dir).name} has no claimed attempt, so no placement")


def load_plan(run_dir: Path) -> Plan:
    """Return the plan written in `run_dir` under its latest attempt's placement, with the sources of its inputs.

    It carries no profile or run config.

    Raises:
        FileNotFoundError: the run has no `plan.json`, or no claimed attempt.
        RunPrivate: the run is another account's, so its placement and inputs are not readable.
    """
    document = json.loads((Path(run_dir) / PLAN_FILE).read_text(encoding="utf-8"))
    inputs = Path(run_dir) / INPUTS_FILE
    try:
        return Plan(
            logical=LogicalPlan.model_validate(document["logical"]),
            manifest=InputManifest.model_validate(document["manifest"]),
            placement=latest_placement(run_dir),
            hash=document["plan_hash"],
            sources=json.loads(inputs.read_text(encoding="utf-8")) if inputs.is_file() else {},
        )
    except PermissionError as err:
        raise RunPrivate(run_dir) from err


def _memory(source: object) -> str | None:
    """Return the identity a `{"memory": <identity>}` source names, None for a path or JSON value."""
    if isinstance(source, Mapping) and set(source) == {"memory"} and isinstance(source["memory"], str):
        return source["memory"]
    return None


def port_types(placed: Plan) -> dict[str, str]:
    """Return the type one record gives of each graph input of `placed`."""
    return {
        port: (found[1] if (found := _LIST.match(declared)) else declared)
        for port, declared in placed.logical.graph.inputs.items()
    }


def recorded_sources(placed: Plan) -> tuple[Sequence[Mapping[str, object]], Mapping[str, object]]:
    """Return the per-record sources of `placed`'s inputs and the shared ones.

    Raises:
        SourcesUnrecorded: the plan records no sources.
    """
    if "rows" not in placed.sources:
        raise SourcesUnrecorded
    return placed.sources["rows"], placed.sources["shared"]


def bind_sources(placed: Plan) -> dict[InputIdentity, object]:
    """Return the value of each input identity of `placed`, bound from its sources and its live values.

    A path or JSON source is bound again (`bind_input`) and must give the identity the manifest holds, so a file
    changed since planning is refused; a `{"memory": <identity>}` source takes the value `placed.live` holds. Every
    record is checked before anything is raised.

    Raises:
        InputChanged: inputs have another identity than the manifest's, or cannot be read; it names every one, and the
            first `InputUnreadable` is its cause.
        InputsUnavailable: an input held in memory is not in `placed.live`.
        SourcesUnrecorded: the plan records no sources.
    """
    types = port_types(placed)
    rows, shared = recorded_sources(placed)
    values: dict[InputIdentity, object] = {}
    missing: list[tuple[str, str]] = []
    changed: list[tuple[str, str]] = []
    unreadable: list[InputUnreadable] = []
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
            if isinstance(bound, InputUnreadable):
                unreadable.append(bound)
            if not isinstance(bound, tuple) or bound[0] != expected:
                changed.append((row.id, port))
                continue
            values[expected] = bound[1]
    if changed:
        raise InputChanged(changed) from next(iter(unreadable), None)
    if missing:
        raise InputsUnavailable(missing)
    return values


def _bound(
    port: str, kind: str, source: object, live: Mapping[InputIdentity, object]
) -> tuple[str, object] | InputUnreadable:
    """Return the identity and value of an input read from `source`, else why its file cannot be read.

    A `{"memory": <identity>}` source is that identity's value in `live`, which holds it.
    """
    held = _memory(source)
    if held is not None:
        return held, live[held]
    try:
        identity, value, _ = bind_input(port, kind, source)
    except InputUnreadable as err:
        return err
    return identity, value


def bind_inputs(run_dir: Path) -> dict[InputIdentity, object]:
    """Return the value of each input identity of the plan in `run_dir` (`bind_sources` of `load_plan`).

    Raises:
        InputChanged, InputsUnavailable, SourcesUnrecorded: as `bind_sources`; an input held in memory is never on disk.
        FileNotFoundError: the run has no plan.
    """
    return bind_sources(load_plan(run_dir))


class _Reader:
    """Reads the graph outputs of one record, or of the run, from the terminal event of each job."""

    def __init__(self, graph: Graph, events: Sequence[Event], store: ResultStore, epoch: int | None = None) -> None:
        self.graph = graph
        self.mapped = mapped_nodes(graph)
        self.store = store
        self.latest = terminal_events(events, epoch=epoch)
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
            return FaultRecord.gap_of("MissingOutput", f"output {port!r} of {node!r} is not in the result store")
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
    `epoch` is the epoch of the attempt `start` made, None for a run opened over a directory.
    """

    def __init__(
        self,
        run_dir: Path,
        *,
        store: ResultStore | None = None,
        profile: ExecutionProfile | None = None,
        clock: Clock = SystemClock(),  # noqa: B008 - stateless
    ) -> None:
        """Open the run in `run_dir`; `wait` and its liveness reads are timed on `clock`.

        `profile` is the one the run is under: its `pool.event_poll_s` paces `wait` and its `store` section opens the
        results store; None reads the packaged profile's.
        """
        self.run_dir = Path(run_dir)
        self._store, self._clock = store, clock
        self._profile = profile if profile is not None else ExecutionProfile.packaged()
        self._pool = self._profile.pool
        self._thread: threading.Thread | None = None
        self.epoch: int | None = None
        self._ended = threading.Event()
        self._cancel = threading.Event()
        self._error: BaseException | None = None

    @property
    def plan(self) -> Plan:
        """The plan written in the run directory (`load_plan`)."""
        return load_plan(self.run_dir)

    def manifest(self) -> dict[str, Any]:
        """Return the manifest (plan hash and result store), `{}` if the run has none.

        Raises:
            RunPrivate: the run is another account's, so its manifest is not readable.
        """
        try:
            found = json.loads((self.run_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except PermissionError as err:
            raise RunPrivate(self.run_dir) from err
        return found if isinstance(found, dict) else {}

    def events(self) -> list[Event]:
        """Return the run's events, from every writer, in order."""
        return read_events(self.run_dir)

    def status(self) -> RunStatus:
        """Return the status derived from the events under the run's epoch, and the lines refused.

        The run's epoch is the one its attempt holds (`epoch`), else the current one. Once a newer epoch is current,
        the status is of the events of the attempt's epoch and earlier, with `superseded` set.
        """
        current = self._epoch()
        mine = self.epoch if self.epoch is not None else current
        scan = scan_events(self.run_dir)
        superseded = mine is not None and current is not None and current > mine
        events = [e for e in scan.events if mine is None or e.epoch <= mine] if superseded else scan.events
        status = status_from(events, epoch=mine, quarantined=len(scan.quarantined))
        return dataclasses.replace(status, superseded=True) if superseded else status

    def _epoch(self) -> int | None:
        """Return the run's current epoch, None where none was taken."""
        return Epoch(self.run_dir).current() or None

    def wait(self, timeout: float | None = None, *, poll: float | None = None) -> RunStatus:
        """Wait up to `timeout` seconds for the run to be done and return its status.

        A run started here is waited on through its thread; any run is then polled, reading the event log every `poll`
        seconds (`pool.event_poll_s` by default), until its events say it is done, so a run another process drives
        can be waited for. An attempt a newer epoch stopped raises its `StaleCoordinator`. While no thread of
        this `Run` is alive, a run whose latest coordinator ended or whose lease is past its stale bound
        (`Epoch.stopped`), with no live claim on any of its queues, is not waited for further: its status is returned
        with `interrupted` set.

        Raises:
            BaseException: the error that stopped a run started here, other than its cancellation.
        """
        poll = self._pool.event_poll_s if poll is None else poll
        deadline = None if timeout is None else self._clock.now() + timeout
        thread = self._thread
        if thread is not None:
            if timeout is None:
                self._ended.wait()
            else:
                self._clock.wait(self._ended, timeout)
            if not thread.is_alive() and self._error is not None:
                raise self._error
        status = self.status()
        while not status.done and (deadline is None or self._clock.now() < deadline):
            if (thread is None or not thread.is_alive()) and self._stopped():
                return dataclasses.replace(self.status(), interrupted=True)
            self._clock.sleep(poll if deadline is None else max(0.0, min(poll, deadline - self._clock.now())))
            status = self.status()
        return status

    def _stopped(self) -> bool:
        """Return whether the latest coordinator stopped (`Epoch.stopped`) and no worker holds a live claim."""
        return (
            Epoch(self.run_dir, self._clock).stopped()
            and not QueueSet.for_run(self.run_dir, clock=self._clock).claimed_any()
        )

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
        threading.Thread(target=self._signal_end, args=(thread,), name="senselab-run-ended", daemon=True).start()

    def _signal_end(self, thread: threading.Thread) -> None:
        thread.join()
        self._ended.set()

    def cancel(self) -> None:
        """Stop releasing jobs: fragments already sent finish or are left to their lease; no other is sent."""
        self._cancel.set()

    def results(self) -> ResultSet:
        """Return what each record gave, read from the result store as each is asked for.

        The events are read now, under the run's epoch (`status`); a record whose job failed, was blocked or cancelled
        gives that, and an earlier attempt's result is served only under the key the attempt gives its job.

        Raises:
            FileNotFoundError: the run has no `plan.json`.
            StaleCoordinator: a newer epoch than this attempt's is current.
        """
        placed = self.plan
        graph = placed.logical.graph
        current = self._epoch()
        epoch = self.epoch if self.epoch is not None else current
        if epoch is not None and current is not None and current > epoch:
            raise StaleCoordinator(epoch, current)
        reader = _Reader(graph, self.events(), self._results_store(), epoch)
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
        return results_store(self._profile, self.manifest().get("store"))


def results_store(
    profile: ExecutionProfile | None, named: object = None, provenance: ProvStore | None = None
) -> ResultStore:
    """Return the result store at `named` (a manifest's `store`), else the cache's, under `profile`'s `store`."""
    root = Path(named) if isinstance(named, str) else cache_root() / RESULTS_DIR
    return ResultStore(root, provenance, settings=None if profile is None else profile.store)


def default_executor(placed: Plan, executor: Executor | None, store: ResultStore | None) -> Executor:
    """Return the executor to run `placed` on: the one given, else `InProcess` (local substrate) or `Pool` (slurm).

    A plan read from disk carries no profile or run config: the packaged default profile is used if its digest is the
    one the plan was placed under. The `Pool` is over `Slurm` and the run's own queues, bound to the run's epoch by
    `start`.

    Raises:
        ValueError: the default profile is not the one the plan was placed under.
    """
    if executor is not None:
        return executor
    profile = placed.execution if placed.execution is not None else ExecutionProfile.load()
    if profile.canonical_digest() != placed.placement.profile_digest:
        raise ValueError(f"plan {placed.hash} was placed under another execution profile; give the executor to run on")
    config = placed.config if placed.config is not None else RunConfig(reuse=ReusePolicy(placed.logical.reuse))
    if profile.substrate.kind == "local":
        from senselab_core.runtime.execution.in_process import InProcess  # noqa: PLC0415 - imports the planner

        return InProcess(store=store, profile=profile, run=config)
    from senselab_core.runtime.execution.pool import Pool  # noqa: PLC0415 - imports the planner
    from senselab_core.runtime.execution.substrates import substrate_of  # noqa: PLC0415 - as above

    if store is None:
        raise ValueError("a Pool over Slurm needs the result store it reads")
    run_dir = run_dir_of(placed.hash)
    return Pool(
        run_dir,
        queues=QueueSet.for_run(run_dir, profile, own=profile.lease),
        store=store,
        events=EventLog(run_dir, "a0", 1, epoch=0),
        profile=profile,
        substrate=substrate_of(profile, run_dir, placed.hash),
        epoch=0,
    )


def _record_placements(
    placed: Plan, executor: Executor, values: Mapping[InputIdentity, object], log: EventLog, epoch: int
) -> Callable[[], None]:
    """Append each class's placement to `log` for an executor that starts no scheduler jobs, as a `Pool` does itself.

    The explicit values and the hook apply to every executor; one with no workers to place records them as given.
    Return what closes the epoch: it appends each class's record again with the run's wall time beside the request.
    """
    from senselab_core.runtime.execution.pool import Pool  # noqa: PLC0415 - imports the planner

    execution = placed.execution
    if isinstance(executor, Pool) or execution is None:
        return lambda: None
    actuals = Actuals(SystemClock())
    ledger = Placements(
        log,
        lambda: placed.placement_request(values=values),
        actuals,
        lambda name: execution.resource_classes[name].device == "cuda",
    )
    placer = Placer(
        execution,
        placed.overrides,
        placed.hook,
        request=lambda: placed.placement_request(values=values),
        record=ledger.record,
        inert="this executor starts no scheduler jobs: the values are recorded, not applied",
    )
    names = sorted({p.resource_class for p in placed.placement.nodes.values()})
    for name in names:
        placer.for_class(name, epoch)
        actuals.began(name)

    def close() -> None:
        for name in names:
            actuals.landed(name, ())
        ledger.close()

    return close


def _claim_ordinal(run_dir: Path, attempt: str) -> int:
    """Return the next free ordinal `n` of the run's attempts, claimed by creating `attempts/<n>` holding `attempt`."""
    directory = run_dir / ATTEMPTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    ordinal = 1 + sum(1 for p in directory.iterdir() if p.name.isdigit())
    while not publish(directory / str(ordinal), attempt.encode()):
        ordinal += 1
    return ordinal


def _claim_attempt(run_dir: Path, placement: PlanPlacement) -> tuple[str, int]:
    """Return a fresh attempt id and its ordinal, its placement file written and never replaced (a taken id: `-<n>`)."""
    private_dir(run_dir / PRIVATE_DIR)
    directory = run_dir / PLACEMENTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    base, n = fresh_attempt_id(), 0
    while True:
        attempt = base if n == 0 else f"{base}-{n}"
        document = json.dumps(placement.model_dump(mode="json"), sort_keys=True, indent=1).encode()
        if publish(directory / f"{attempt}.json", document):
            return attempt, _claim_ordinal(run_dir, attempt)
        n += 1


def _write_once(path: Path, document: Mapping[str, object]) -> None:
    """Write `document` as JSON to `path` unless a file is there; one that is must name the same `plan_hash`.

    Raises:
        ValueError: `path` holds the document of another plan.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not publish(path, json.dumps(document, sort_keys=True, indent=1).encode()):
        held = json.loads(path.read_text(encoding="utf-8")).get("plan_hash")
        if held != document["plan_hash"]:
            raise ValueError(f"{path.parent.name} holds the plan {held}, not {document['plan_hash']}")


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
    except InputUnreadable:
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
    types = port_types(placed)
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


def record_sources(placed: Plan, run_dir: Path) -> None:
    """Merge the input sources of `placed` into `run_dir`'s `inputs.json` (`merged_sources`) under its lease.

    The lease, kept alive under the run's lease (`run_lease`, with the plan's profile's `lease` section among its
    epochs), is proven before the write, so two coordinators
    starting one plan merge one after the other and neither loses a durable source. A merge whose proof failed is
    made again, up to `lease.record_attempts` times.

    Raises:
        ConflictingSources: as `merged_sources`; nothing is written.
        LeaseUnavailable: the merge's lease was lost on every attempt.
        LeaseTimeout: another holder kept the lease for its whole timeout.
    """
    lease = run_lease(run_dir, own=_lease_of(placed))
    private_dir(run_dir / PRIVATE_DIR)
    for _ in range(lease.record_attempts):
        held = Lease(run_dir / INPUTS_FILE, settings=lease)
        with held, held.keep_alive(lease.renew_fraction):
            sources = merged_sources(placed, run_dir)
            if held.prove():
                _write(run_dir / INPUTS_FILE, sources)
                return
    raise LeaseUnavailable(run_dir / INPUTS_FILE, f"the lease on {run_dir / INPUTS_FILE} was lost on every merge")


def prepare(placed: Plan, run_dir: Path, store: ResultStore | None) -> tuple[str, int]:
    """Write the run's plan, input sources, manifest and license policy once, then claim an attempt.

    The digest of the stored policy is written beside the attempt's placement (`<attempt>.licenses`).

    Nothing here is visible to a reader of the run's epochs or status. Returns the attempt id and its ordinal.

    Raises:
        ValueError: `run_dir` holds another plan.
        ConflictingSources, LeaseUnavailable, LeaseTimeout: as `record_sources`; a source conflicts only with a
            recorded run, whose plan is already written, so nothing is written.
        LaxerLicensePolicy: the plan's policy allows what the run's stored policy refuses.
    """
    _write_once(run_dir / PLAN_FILE, _plan_document(placed))
    record_sources(placed, run_dir)
    for name in POOL_DIRS:
        (run_dir / name).mkdir(exist_ok=True)
    _write(run_dir / MANIFEST_FILE, {"plan_hash": placed.hash, "store": None if store is None else str(store.root)})
    digest = record_policy(run_dir, None if placed.config is None else placed.config.licenses)
    attempt, ordinal = _claim_attempt(run_dir, placed.placement)
    publish(run_dir / PLACEMENTS_DIR / f"{attempt}.licenses", digest.encode())
    return attempt, ordinal


def stored_policy(run_dir: Path) -> LicensePolicy | None:
    """Return the license policy the run in `run_dir` was started under, None when it has none or none was written.

    Raises:
        ValueError: `licenses.json` is not a `run-licenses@1` document.
    """
    try:
        return policy_of(json.loads((Path(run_dir) / LICENSES_FILE).read_text(encoding="utf-8")))
    except FileNotFoundError:
        return None


def record_policy(run_dir: Path, given: LicensePolicy | None) -> str:
    """Write `given` as the run's license policy unless one is written, and refuse one laxer than the stored.

    A policy at least as strict as the stored one is used for the attempt and does not replace it. Returns the digest
    of the stored policy.

    Raises:
        LaxerLicensePolicy: `given` allows what the stored policy refuses.
    """
    publish(Path(run_dir) / LICENSES_FILE, jcs_dumps(policy_document(given)))
    stored = stored_policy(run_dir)
    if (laxer := laxer_than(given, stored)) is not None:
        raise laxer
    return policy_digest(stored)


def checked_lease(run_dir: Path, own: JobLease, attribute_delay_s: float) -> JobLease:
    """Return the run's lease with `own` among its epochs, refused unless its beats stay below its stale bound.

    `attribute_delay_s` is the delay measured on the run's storage (`check_storage`).

    Raises:
        LeaseBoundError: the composite's bound fails with the larger of its delay and the one measured.
        RunLeaseUnreadable: a bound epoch's lease section is unreadable.
    """
    found = run_lease(run_dir, own)
    found.model_copy(update={"attribute_delay_s": max(found.attribute_delay_s, attribute_delay_s)}).check_bound()
    return found


def _lease_of(placed: Plan) -> JobLease:
    return (placed.execution if placed.execution is not None else ExecutionProfile.load()).lease


def _plan_document(placed: Plan) -> dict[str, object]:
    return {
        "plan_hash": placed.hash,
        "logical": placed.logical.model_dump(mode="json"),
        "manifest": placed.manifest.model_dump(mode="json"),
    }


def _check_store(placed: Plan, store: ResultStore | None) -> float:
    """Refuse a run whose result store lacks a primitive leases need (`check_storage`); return its attribute delay."""
    profile = placed.execution if placed.execution is not None else ExecutionProfile.load()
    return profile.lease.attribute_delay_s if store is None else check_storage(profile, store.root)


def _open(
    run_dir: Path,
    given: tuple[Plan, Executor, ProvStore],
    attempt: tuple[str, int],
    leases: tuple[JobLease, JobLease],
) -> tuple[contextlib.ExitStack, int, EventLog]:
    """Take the next epoch (not yet current), hold its coordinator lease, reconcile the queues and bind the executor.

    `given` is the plan, the executor and the provenance store the plan's entity is added to; `leases` are the plan's
    own lease section and the run's. Return what releases the epoch, the epoch and its event log; on a failure
    the epoch is released, marked ended, and the error raised.
    """
    placed, executor, written = given
    own, lease = leases
    epochs, held = Epoch(run_dir), contextlib.ExitStack()
    try:
        epoch = held.enter_context(epochs.opening(own))
        held.enter_context(epochs.held(epoch, lease))
        QueueSet.for_run(run_dir, placed.execution, own=own).reconcile()
        log = EventLog(run_dir, *attempt, epoch=epoch)
        executor.bind_epoch(epoch, log)
        add_plan_entity(written, placed.hash)
    except BaseException:
        held.close()
        raise
    return held, epoch, log


def _close_owned(executor: Executor, owned: bool, run: "Run", failed: bool) -> None:
    """Close an executor the run created, so provenance and `Run.fail` still run if the close fails.

    A close error is logged; it is handed to `run.fail` only when the run had not failed (`failed`) already.
    """
    if not (owned and hasattr(executor, "close")):
        return
    try:
        executor.close()
    except Exception as err:  # noqa: BLE001 - cleanup must not skip the run's bookkeeping
        logger.error("closing the executor of run %s failed: %s", run.run_dir.name, err)
        if not failed:
            run.fail(err)


def start(
    placed: Plan,
    executor: Executor | None = None,
    *,
    store: ResultStore | None = None,
    values: Mapping[InputIdentity, object] | None = None,
) -> Run:
    """Start `placed` on `executor` in a background thread, as a new attempt, and return its `Run`.

    With no executor, `InProcess` runs a plan under a local substrate, and a `Pool` over `Slurm` under a slurm one, over
    `store` (default: the result store under the cache root) with the plan's own profile and run config. `store` is also
    where the run's results are read from, so it is given with any other executor that stores results, and it is the
    store that executor writes: the plan's `senselab:Plan` entity is added to its provenance. The input values are bound
    (`bind_sources`) before anything is written: from their paths or JSON values, or from the values the plan holds in
    memory; `values` gives them already bound. The sources are merged into `inputs.json` under its lease
    (`record_sources`), also before anything else is written. Every item in the run's `queues/` with neither a slot nor
    an outcome is put again (`QueueSet.reconcile`). The run claims its attempt (`prepare`), then takes the next epoch,
    holds its coordinator lease until the run ends and hands the epoch to the executor (`bind_epoch`); the epoch is
    current only once its thread binds it (`Epoch.bind`), the last step before the plan runs.

    A node whose unit does not fit its class runs on the smallest class it fits (`Plan.fit`).

    Raises:
        Unplaceable: a node's unit fits no class from its own up; nothing is written.
        ValueError: as `admit`.
        InputChanged: an input changed since the plan was made.
        InputsUnavailable: an input held in memory is not held by `placed` (a plan read from disk).
        ConflictingSources: a source the run records reads other content than the plan's.
        LeaseUnavailable, LeaseTimeout: as `record_sources`.
        LeaseBoundError: the run's lease with this plan's section among its epochs fails the beat bound.
        RunLeaseUnreadable: a bound epoch's lease section is unreadable.
        MissingExtra: pydra is not installed.
    """
    if importlib.util.find_spec("pydra") is None:
        raise MissingExtra("pydra")
    from senselab_core.runtime.execution.runner import PydraRunner, RunCancelled  # noqa: PLC0415 - needs pydra

    placed = placed.fit()
    provenance = ProvStore()
    if executor is None and store is None:
        store = results_store(placed.execution, None, provenance)
    owned = executor is None
    executor = default_executor(placed, executor, store)
    delay = _check_store(placed, store)
    values = bind_sources(placed) if values is None else values
    run_dir = run_dir_of(placed.hash)
    attempt, ordinal = prepare(placed, run_dir, store)
    own = _lease_of(placed)
    lease = checked_lease(run_dir, own, delay)
    runner = PydraRunner(profile=placed.execution, work_dir=run_dir / PYDRA_DIR)
    run = Run(run_dir, store=store, profile=placed.execution)
    written = provenance if store is None else store.provenance
    epochs = Epoch(run_dir)
    held, epoch, log = _open(run_dir, (placed, executor, written), (attempt, ordinal), (own, lease))
    run.epoch = epoch

    def body() -> None:
        failed = False
        close_placements: Callable[[], None] = lambda: None  # noqa: E731 - replaced once the epoch is bound
        with held:
            try:
                epochs.bind(epoch, attempt, ordinal)
                close_placements = _record_placements(placed, executor, values, log, epoch)
                runner.run_plan(placed, executor, values=values, events=log, cancel=run.cancelled)
            except BaseException as err:  # noqa: BLE001 - held for `wait`
                if not isinstance(err, RunCancelled):
                    failed = True
                    run.fail(err)
            finally:
                close_placements()
                _close_owned(executor, owned, run, failed)
            try:
                written.write(run_dir)
            except Exception as err:  # noqa: BLE001 - held for `wait`
                run.fail(err)

    try:
        run.launch(threading.Thread(target=body, name=f"senselab-run-{placed.hash[:8]}", daemon=True))
    except BaseException:
        held.close()
        raise
    return run
