"""A run's event log and the status derived from it: `Event`, `EventLog`, `read_events` and `status_from`.

Every state change of a job, and each class's placement in an epoch, is one `run-event@2` line in
`<run>/events/<writer>.jsonl`; a writer appends to its own file only, so any number of processes log one run
without sharing a file. A run's status, its counts and its faults are computed from the job events alone.
"""

import heapq
import json
import os
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, ClassVar, Literal

from pydantic import Field, StringConstraints, model_validator

from senselab_core.kernel._base import Frozen, NonEmptyStr
from senselab_core.kernel.formats import RUN_EVENT_FORMAT, FormatTag, Formatted
from senselab_core.kernel.graph import NodeId
from senselab_core.kernel.identity import OutputIdentity
from senselab_core.runtime.execution.executor import FaultName, FaultRecord, Hex64, JobRef
from senselab_core.runtime.revision import NAMESPACE, cache_root
from senselab_core.runtime.site_placement import (
    ClassNeeds,
    ClassPlacement,
    CurrentValues,
    HookRef,
    PlacementAnswer,
    rendered,
)

EVENTS_DIR = "events"
"""The directory of a run holding one `<writer>.jsonl` file per writer."""

EventKind = Literal[
    "queued",
    "claimed",
    "started",
    "finished",
    "failed",
    "yielded",
    "cache_hit",
    "verify",
    "blocked",
    "cancelled",
    "placement",
]
"""What happened to a job, or `placement`, which is of the run and names no node."""
JobState = Literal["queued", "running", "done", "failed", "blocked", "cancelled"]
"""Where a job is, from its events."""
STATES: tuple[JobState, ...] = ("queued", "running", "done", "failed", "blocked", "cancelled")

_STATE: Mapping[str, JobState] = {
    "queued": "queued",
    "yielded": "queued",
    "claimed": "running",
    "started": "running",
    "finished": "done",
    "cache_hit": "done",
    "failed": "failed",
    "blocked": "blocked",
    "cancelled": "cancelled",
}
TERMINAL: Mapping[str, int] = {"finished": 3, "cache_hit": 3, "failed": 2, "blocked": 1, "cancelled": 0}
"""The kinds that end a job's state machine (queued, then claimed, started, then one of these), by precedence."""
RESULTS = ("finished", "cache_hit")
"""The terminal kinds that give a result."""


def run_dir_of(plan_hash: str, root: Path | None = None) -> Path:
    """Return the run directory of the plan `plan_hash`: `<cache root>/2x/runs/<plan hash>`."""
    return runs_root(root) / plan_hash


class ClassActuals(Frozen):
    """What a class's work took in an epoch, to read beside the request's `wall_s`, `gpu_hours` and memory.

    `wall_s` runs from the class's first put to its last outcome; `gpu_hours` is the lifetimes of its workers on a
    `cuda` class; `device_peak_bytes` is the largest device peak an `Observation` of the class reports. A figure is
    None where nothing measured it.
    """

    wall_s: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    gpu_hours: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    device_peak_bytes: Annotated[int, Field(ge=0)] | None = None


class PlacementRecord(Frozen):
    """How one resource class was placed in one epoch: the payload of a `placement` event.

    It holds the class's needs as the request gave them, the site's answer, the hook's reference, the verdict on the
    answer, the values the class was placed with and the layer that set each, and the `--exclude` list submitted.
    Partitions, accounts and nodes, the answer's `why` and the notes are rendered for the readers of the run
    (`rendered`). The epoch's end appends the class's record again with `closing` set: its actuals beside the request.
    """

    resource_class: NonEmptyStr
    request: ClassNeeds | None = None
    answer: PlacementAnswer | None = None
    hook: HookRef | None = None
    verdict: Literal["no-hook", "accepted", "refused", "failed"]
    note: str | None = None
    used: CurrentValues
    exclude: tuple[str, ...] = ()
    closing: ClassActuals | None = None


class Event(Formatted):
    """A `run-event@2` document: one state change of one job of one attempt, or a run-scoped `placement` of a class.

    `job` names the job (None for an event of the node as a whole), `key` its key digest, `attempt` the id of the
    attempt (run or resume) that wrote it, `ordinal` that attempt's place among the run's attempts (taken when it was
    claimed), `epoch` the coordinator epoch the attempt belongs to (0 for none), `writer` the writer whose file holds
    it, `seq` its position in that file, `placement_digest` the digest of the node's placement, `fault` the class of a
    failure, `failure` the record of it, `reuse` the policy that served a `cache_hit` (its value), `env` the env the job
    ran in, `output` the output identity of a `finished` or `cache_hit` job and `t_ns` the wall-clock nanoseconds of the
    event.
    """

    format: FormatTag = RUN_EVENT_FORMAT
    port_value: ClassVar[bool] = False
    kind: EventKind
    node: NodeId | None = None
    key: Hex64 | None = None
    job: JobRef | None = None
    attempt: NonEmptyStr
    ordinal: Annotated[int, Field(ge=1)]
    epoch: Annotated[int, Field(ge=0)] = 0
    writer: NonEmptyStr
    seq: Annotated[int, Field(ge=0)]
    placement_digest: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")] | None = None
    fault: FaultName | None = None
    failure: FaultRecord | None = None
    reuse: Literal["exact", "equivalent", "tolerant"] | None = None
    env: NonEmptyStr | None = None
    output: OutputIdentity | None = None
    placement: PlacementRecord | None = None
    t_ns: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def _check(self) -> "Event":
        if self.kind == "placement":
            if (
                self.placement is None
                or any(
                    v is not None
                    for v in (
                        self.node,
                        self.job,
                        self.key,
                        self.fault,
                        self.failure,
                        self.output,
                        self.reuse,
                        self.env,
                    )
                )
                or self.placement_digest is not None
            ):
                raise ValueError("a placement event carries its record alone: no node, job, key, fault, output or env")
            return self
        if self.node is None or self.placement is not None:
            raise ValueError("a job event names its node and carries no placement record")
        if self.job is not None and self.job.node != self.node:
            raise ValueError(f"event of node {self.node!r} names job {self.job.node!r}")
        if self.failure is not None and self.fault != self.failure.fault:
            raise ValueError("an event's fault is the class of its failure")
        if self.kind == "failed" and self.failure is None:
            raise ValueError("a failed event carries its failure")
        if self.kind != "failed" and (self.fault is not None or self.failure is not None):
            raise ValueError("only a failed event carries a fault")
        if self.reuse is not None and self.kind != "cache_hit":
            raise ValueError("only a cache_hit event carries a reuse policy")
        return self


class BrokenEnv(Frozen):
    """An env that failed `failures` times by infrastructure faults; `fault` is the latest."""

    env: NonEmptyStr
    fault: FaultRecord
    failures: Annotated[int, Field(ge=1)]


@dataclass(frozen=True)
class RunStatus:
    """Where a run is: its jobs by state, its failures by fault class, its broken envs and whether it is over.

    Derived from events alone (`status_from`). `done` is True once every job seen has a terminal event, and
    `cancelled` once any job was cancelled, so a cancelled run is never plain done. `last_event_ns` is the time of the
    newest event. `interrupted` is set by `Run.wait` alone, when the run is not done and its coordinator stopped.
    """

    counts: Mapping[JobState, int]
    faults: Mapping[str, int]
    broken: tuple[BrokenEnv, ...] = ()
    done: bool = False
    cancelled: bool = False
    last_event_ns: int | None = None
    interrupted: bool = False
    nodes: Mapping[str, Mapping[JobState, int]] = field(default_factory=dict)


def _latest(events: Sequence[Event]) -> list[Event]:
    """Return the events of a job's latest attempt: the highest `epoch`, then the highest `ordinal` within it."""
    latest = max((e.epoch, e.ordinal, e.attempt) for e in events)
    return [e for e in events if (e.epoch, e.ordinal, e.attempt) == latest]


def _superseded(event: Event, events: Sequence[Event]) -> bool:
    """Return whether `event` is a failure its writer then queued again: a retry within one attempt."""
    return event.kind == "failed" and any(
        e.kind == "queued" and e.writer == event.writer and e.seq > event.seq for e in events
    )


def _current(events: Sequence[Event], epoch: int | None) -> list[Event]:
    """Return the events of a job's latest attempt (`_latest`); none when it has no event of epoch `epoch` or later."""
    mine = _latest(events)
    return [] if epoch is not None and mine[0].epoch < epoch else mine


def _latest_terminal(events: Sequence[Event], epoch: int | None = None) -> Event | None:
    """Return the terminal event of a job's latest attempt that has precedence, else an earlier result under its key.

    A result (`finished`, `cache_hit`) outranks `failed`, which outranks `blocked`, then `cancelled`; among events of
    one kind the one later in its writer's file, then the greater writer, is taken. A `failed` its writer queued
    again is not terminal. While the latest attempt has no terminal event, a result an earlier attempt gave stands
    only when its key is the key the latest attempt's events give the job. With `epoch`, a job with no event of that
    epoch has no latest attempt (`_current`).
    """
    mine = _current(events, epoch)
    terminal = [e for e in mine if e.kind in TERMINAL and not _superseded(e, mine)]
    if not terminal:
        keys = {e.key for e in mine if e.key is not None}
        terminal = [e for e in events if e.kind in RESULTS and e.key in keys]
    return max(terminal, key=lambda e: (TERMINAL[e.kind], e.epoch, e.ordinal, e.writer, e.seq)) if terminal else None


def _job_state(events: Sequence[Event], epoch: int | None = None) -> JobState:
    """Return a job's state: its terminal state (`_latest_terminal`), else `running` or `queued` by counts.

    In the latest attempt a job is running while its claims (`claimed`, `started`) outnumber its yields and at least
    match its submissions (`queued`), so a retry queued after a failure is queued until it is claimed again. Counts
    read no clock and no order across writers.
    """
    found = _latest_terminal(events, epoch)
    if found is not None:
        return _STATE[found.kind]
    kinds = Counter(e.kind for e in _current(events, epoch))
    claims = kinds["claimed"] + kinds["started"]
    return "running" if claims > kinds["yielded"] and claims >= kinds["queued"] else "queued"


def _by_job(events: Iterable[Event]) -> dict[tuple[str, tuple[int, ...]], list[Event]]:
    """Return each job's state-changing events, in the order given."""
    jobs: dict[tuple[str, tuple[int, ...]], list[Event]] = {}
    for event in events:
        if event.kind in _STATE and event.node is not None:
            jobs.setdefault((event.node, event.job.element if event.job is not None else ()), []).append(event)
    return jobs


def terminal_events(events: Iterable[Event], *, epoch: int | None = None) -> dict[tuple[str, tuple[int, ...]], Event]:
    """Return, for each job `(node, element)` with one, the terminal event that states how it ended (under `epoch`)."""
    found = {job: _latest_terminal(each, epoch) for job, each in _by_job(events).items()}
    return {job: event for job, event in found.items() if event is not None}


def status_from(events: Iterable[Event], *, broken_after: int | None = None, epoch: int | None = None) -> RunStatus:
    """Return the status of a run from its `events`, each writer's in file order (`read_events`).

    A job's state is the terminal event of its latest attempt (highest `epoch`, then `ordinal`) with precedence
    (`TERMINAL`), a failure queued again excepted, else a result of an earlier attempt under the same key, else
    running or queued as the latest attempt's claims and submissions count (`_job_state`); `verify` changes none.
    `seq` is compared within one writer only and no clock order is assumed for terminal states. `faults` counts
    every `failed` event by its class; an env with `broken_after` infrastructure failures (the packaged profile's
    `infrastructure_replacements` when None) is `broken`. Cancelled and blocked jobs are not faults. With `epoch`
    (the run's current one), a job with no event of
    that epoch is queued.
    """
    if broken_after is None:
        from senselab_core.runtime.profiles import ExecutionProfile  # noqa: PLC0415 - read only when needed

        broken_after = ExecutionProfile.load().faults.infrastructure_replacements
    seen = [e for e in events if e.kind != "placement"]
    jobs = _by_job(seen)
    faults: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    latest: dict[str, FaultRecord] = {}
    for event in seen:
        if event.kind == "failed" and event.fault is not None:
            faults[event.fault] += 1
            if event.fault == "infrastructure" and event.env is not None and event.failure is not None:
                failures[event.env] += 1
                latest[event.env] = event.failure
    tally: Counter[str] = Counter()
    per_node: dict[str, Counter[str]] = {}
    for (node, _), found in jobs.items():
        state = _job_state(found, epoch)
        tally[state] += 1
        per_node.setdefault(node, Counter())[state] += 1
    broken = tuple(
        BrokenEnv(env=env, fault=latest[env], failures=n) for env, n in sorted(failures.items()) if n >= broken_after
    )
    return RunStatus(
        counts={s: tally[s] for s in STATES},
        faults=dict(faults),
        broken=broken,
        done=bool(jobs) and not tally["queued"] and not tally["running"],
        cancelled=tally["cancelled"] > 0,
        last_event_ns=max((e.t_ns for e in seen), default=None),
        nodes={n: {s: c[s] for s in STATES} for n, c in sorted(per_node.items())},
    )


class EventLog:
    """One writer's append-only log of a run: `<run dir>/events/<writer uuid>.jsonl`, written by this writer only.

    `attempt` is the id of the attempt every event of this log belongs to, `ordinal` its claim order and `epoch` its
    coordinator's epoch (0 for none), `clock` gives
    the `t_ns` of an event `record` makes, and each event gets the next `seq` of this writer. A writer used after a
    `fork` is a new writer in the child.
    """

    def __init__(
        self, run_dir: Path, attempt: str, ordinal: int, *, epoch: int = 0, clock: Callable[[], int] = time.time_ns
    ) -> None:
        """Log to `run_dir` as attempt `attempt`, the `ordinal`-th, of `epoch`; nothing is created until an append."""
        self.run_dir = Path(run_dir)
        self.attempt = attempt
        self.ordinal = ordinal
        self.epoch = epoch
        self.clock = clock
        self._writer = (os.getpid(), uuid.uuid4().hex)
        self._seq = 0
        self._lock = threading.Lock()

    @property
    def writer(self) -> str:
        """This process's writer uuid4 hex: the name of the file `append` writes."""
        if self._writer[0] != os.getpid():
            self._writer = (os.getpid(), uuid.uuid4().hex)
            self._seq = 0
        return self._writer[1]

    @property
    def path(self) -> Path:
        """The file this writer appends to."""
        return self.run_dir / EVENTS_DIR / f"{self.writer}.jsonl"

    def append(self, event: Event) -> None:
        """Append `event` as one line of this writer's file, in one write."""
        data = (json.dumps(event.model_dump(mode="json"), sort_keys=True, separators=(",", ":")) + "\n").encode()
        with self._lock:
            path = self.path
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                os.write(descriptor, data)
            finally:
                os.close(descriptor)

    def record(  # noqa: PLR0913 - the fields of an event
        self,
        kind: EventKind,
        node: str | None,
        *,
        job: JobRef | None = None,
        key: str | None = None,
        placement_digest: str | None = None,
        failure: FaultRecord | None = None,
        reuse: str | None = None,
        env: str | None = None,
        output: str | None = None,
        placement: PlacementRecord | None = None,
    ) -> Event:
        """Append the event of `kind` for `node` (None for a `placement`), stamped now with the next `seq`."""
        with self._lock:
            writer = self.writer
            seq, self._seq = self._seq, self._seq + 1
        event = Event(
            kind=kind,
            node=node,
            job=job,
            key=key,
            attempt=self.attempt,
            ordinal=self.ordinal,
            epoch=self.epoch,
            writer=writer,
            seq=seq,
            placement_digest=placement_digest,
            fault=None if failure is None else failure.fault,
            failure=failure,
            reuse=reuse,
            env=env,
            output=output,
            placement=placement,
            t_ns=self.clock(),
        )
        self.append(event)
        return event


def runs_root(root: Path | None = None) -> Path:
    """Return the directory holding every run: `<cache root>/2x/runs`."""
    return (cache_root() if root is None else Path(root)) / NAMESPACE / "runs"


def _lines(path: Path) -> Iterator[Event]:
    """Yield the events of one writer's file in order; a last line cut short by a crash is skipped."""
    lines = path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            yield Event.model_validate(json.loads(line))
        except ValueError:
            if number != len(lines) - 1:
                raise


def read_events(run_dir: Path) -> list[Event]:
    """Return every event of the run in `run_dir`: each writer's events in its file order, merged by time.

    The merge keeps every writer's own order whatever its clock did, so a clock stepping back reorders nothing in a
    file; across writers it is only an interleaving, which no status reads.

    Raises:
        ValueError: a line other than a file's last is not a `run-event@2` document.
    """
    streams = [_lines(path) for path in sorted((Path(run_dir) / EVENTS_DIR).glob("*.jsonl"))]
    return list(heapq.merge(*streams, key=lambda e: e.t_ns))


def placement_record(
    name: str, needs: ClassNeeds | None, hook: HookRef | None, placed: ClassPlacement
) -> PlacementRecord:
    """Return the record of class `name`'s `placed` values, every text in it rendered with `placed.names`."""

    def show(text: str) -> str:
        return rendered(text, placed.names)

    def current(found: CurrentValues) -> CurrentValues:
        return found.model_copy(update={"partition": None if found.partition is None else show(found.partition)})

    answer = placed.answer
    if answer is not None:
        answer = answer.model_copy(
            update={
                "classes": {
                    k: v.model_copy(
                        update={
                            "partition": None if v.partition is None else show(v.partition),
                            "exclude": None if v.exclude is None else tuple(show(n) for n in v.exclude),
                        }
                    )
                    for k, v in answer.classes.items()
                },
                "why": show(answer.why),
            }
        )
    return PlacementRecord(
        resource_class=name,
        request=None
        if needs is None
        else needs.model_copy(
            update={
                "current": current(needs.current),
                "unmeasured": tuple(u.model_copy(update={"why": show(u.why)}) for u in needs.unmeasured),
            }
        ),
        answer=answer,
        hook=hook,
        verdict=placed.verdict,
        note=None if placed.note is None else show(placed.note),
        used=current(placed.values),
        exclude=tuple(show(n) for n in placed.exclude),
    )


def placement_recorder(log: EventLog) -> Callable[[int, str, ClassNeeds | None, HookRef | None, ClassPlacement], None]:
    """Return what a `Placer` calls to append each placement to `log`, the epoch's event log, as a `placement` event."""

    def record(_: int, name: str, needs: ClassNeeds | None, hook: HookRef | None, placed: ClassPlacement) -> None:
        log.record("placement", None, placement=placement_record(name, needs, hook, placed))

    return record


def placements(events: Iterable[Event]) -> list[Event]:
    """Return the `placement` events among `events`, by epoch then class; a class's closing record follows its own."""
    found = [e for e in events if e.kind == "placement"]
    return sorted(found, key=lambda e: (e.epoch, e.placement.resource_class if e.placement else ""))
