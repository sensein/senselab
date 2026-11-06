"""A run's event log and the status derived from it: `Event`, `EventLog`, `read_events` and `status_from`.

Every state change of a job is one `run-event@1.0` line in `<run>/events/<writer>.jsonl`; a writer appends to its own
file only, so any number of processes log one run without sharing a file. A run's status, its counts and its
faults are computed from the events alone.
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

EVENTS_DIR = "events"
"""The directory of a run holding one `<writer>.jsonl` file per writer."""

EventKind = Literal[
    "queued", "claimed", "started", "finished", "failed", "yielded", "cache_hit", "verify", "blocked", "cancelled"
]
"""What happened to a job."""
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
TERMINAL = frozenset({"finished", "cache_hit", "failed", "blocked", "cancelled"})
"""The kinds that end a job's state machine: queued, then claimed, started, then one of these."""


def run_dir_of(plan_hash: str, root: Path | None = None) -> Path:
    """Return the run directory of the plan `plan_hash`: `<cache root>/2x/runs/<plan hash>`."""
    return runs_root(root) / plan_hash


class Event(Formatted):
    """A `run-event@1.0` document: one state change of one job of one attempt.

    `job` names the job (None for an event of the node as a whole), `key` its key digest, `attempt` the id of the
    attempt (run or resume) that wrote it, `seq` its position in its writer's file, `placement_digest` the digest of
    the node's placement, `fault` the class of a failure, `failure` the record of it, `reuse` the policy that served a
    `cache_hit` (its value), `env` the env the job ran in, `output` the output identity of a `finished` or
    `cache_hit` job and `t_ns` the wall-clock nanoseconds of the event.
    """

    format: FormatTag = RUN_EVENT_FORMAT
    port_value: ClassVar[bool] = False
    kind: EventKind
    node: NodeId
    key: Hex64 | None = None
    job: JobRef | None = None
    attempt: NonEmptyStr
    seq: Annotated[int, Field(ge=0)]
    placement_digest: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")] | None = None
    fault: FaultName | None = None
    failure: FaultRecord | None = None
    reuse: Literal["exact", "equivalent", "tolerant"] | None = None
    env: NonEmptyStr | None = None
    output: OutputIdentity | None = None
    t_ns: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def _check(self) -> "Event":
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
    newest event: a run that is not done and whose last event is old was left by a process that died.
    """

    counts: Mapping[JobState, int]
    faults: Mapping[str, int]
    broken: tuple[BrokenEnv, ...] = ()
    done: bool = False
    cancelled: bool = False
    last_event_ns: int | None = None
    nodes: Mapping[str, Mapping[JobState, int]] = field(default_factory=dict)


def _latest_terminal(events: Sequence[Event], started: Mapping[str, int]) -> Event | None:
    """Return a job's latest attempt's last terminal event by `seq`, or None if that attempt has none."""
    latest = max((e.attempt for e in events), key=lambda a: (started[a], a))
    terminal = [e for e in events if e.attempt == latest and e.kind in TERMINAL]
    return max(terminal, key=lambda e: (e.seq, e.t_ns)) if terminal else None


def _job_state(events: Sequence[Event], started: Mapping[str, int]) -> JobState:
    """Return a job's state: its latest terminal event's, else that of its latest attempt's last other event."""
    found = _latest_terminal(events, started)
    if found is not None:
        return _STATE[found.kind]
    latest = max((e.attempt for e in events), key=lambda a: (started[a], a))
    mine = [e for e in events if e.attempt == latest and e.kind in _STATE]
    return _STATE[mine[-1].kind] if mine else "queued"


def _by_job(events: Iterable[Event]) -> tuple[dict[tuple[str, tuple[int, ...]], list[Event]], dict[str, int]]:
    """Return each job's state-changing events in order, and the time each attempt's first event was written."""
    started: dict[str, int] = {}
    jobs: dict[tuple[str, tuple[int, ...]], list[Event]] = {}
    for event in events:
        started[event.attempt] = min(started.get(event.attempt, event.t_ns), event.t_ns)
        if event.kind in _STATE:
            jobs.setdefault((event.node, event.job.element if event.job is not None else ()), []).append(event)
    return jobs, started


def terminal_events(events: Iterable[Event]) -> dict[tuple[str, tuple[int, ...]], Event]:
    """Return, for each job `(node, element)` with one, the terminal event that states how it ended."""
    jobs, started = _by_job(events)
    found = {job: _latest_terminal(each, started) for job, each in jobs.items()}
    return {job: event for job, event in found.items() if event is not None}


def status_from(events: Iterable[Event], *, broken_after: int | None = None) -> RunStatus:
    """Return the status of a run from its `events`, each writer's in file order (`read_events`).

    A job's state is its latest attempt's last terminal event (by `seq` in its writer), else its last non-terminal
    event's state; `verify` changes none. No order across writers is assumed. `faults` counts every `failed` event by
    its class; an env with `broken_after` infrastructure failures (the packaged profile's
    `infrastructure_replacements` when None) is `broken`. Cancelled and blocked jobs are not faults.
    """
    if broken_after is None:
        from senselab_core.runtime.profiles import ExecutionProfile  # noqa: PLC0415 - read only when needed

        broken_after = ExecutionProfile.load().faults.infrastructure_replacements
    seen = list(events)
    jobs, started = _by_job(seen)
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
        state = _job_state(found, started)
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

    `attempt` is the id of the attempt every event of this log belongs to, `clock` gives the `t_ns` of an event
    `record` makes, and each event gets the next `seq` of this writer. A writer used after a `fork` is a new writer in
    the child.
    """

    def __init__(self, run_dir: Path, attempt: str, *, clock: Callable[[], int] = time.time_ns) -> None:
        """Log to `run_dir` as attempt `attempt`; nothing is created until the first append."""
        self.run_dir = Path(run_dir)
        self.attempt = attempt
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
        node: str,
        *,
        job: JobRef | None = None,
        key: str | None = None,
        placement_digest: str | None = None,
        failure: FaultRecord | None = None,
        reuse: str | None = None,
        env: str | None = None,
        output: str | None = None,
    ) -> Event:
        """Append the event of `kind` for `node`, stamped now with the next `seq`, and return it."""
        with self._lock:
            _ = self.writer
            seq, self._seq = self._seq, self._seq + 1
        event = Event(
            kind=kind,
            node=node,
            job=job,
            key=key,
            attempt=self.attempt,
            seq=seq,
            placement_digest=placement_digest,
            fault=None if failure is None else failure.fault,
            failure=failure,
            reuse=reuse,
            env=env,
            output=output,
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
        ValueError: a line other than a file's last is not a `run-event@1` document.
    """
    streams = [_lines(path) for path in sorted((Path(run_dir) / EVENTS_DIR).glob("*.jsonl"))]
    return list(heapq.merge(*streams, key=lambda e: e.t_ns))
