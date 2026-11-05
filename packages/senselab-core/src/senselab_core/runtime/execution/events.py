"""A run's event log and the status derived from it: `Event`, `EventLog`, `read_events` and `status_from`.

Every state change of a job is one `run-event@1.0` line in `<run>/events/<writer>.jsonl`; a writer appends to its own
file only, so any number of processes log one run without sharing a file. A run's status, its counts and its
faults are computed from the events alone.
"""

import json
import os
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
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

EventKind = Literal["queued", "claimed", "started", "finished", "failed", "yielded", "cache_hit", "verify"]
"""What happened to a job."""
JobState = Literal["queued", "running", "done", "failed"]
"""Where a job is, from its latest event."""

_STATE: Mapping[str, JobState] = {
    "queued": "queued",
    "yielded": "queued",
    "claimed": "running",
    "started": "running",
    "finished": "done",
    "cache_hit": "done",
    "failed": "failed",
}


def run_dir_of(plan_hash: str, root: Path | None = None) -> Path:
    """Return the run directory of the plan `plan_hash`: `<cache root>/2x/runs/<plan hash>`."""
    return (cache_root() if root is None else Path(root)) / NAMESPACE / "runs" / plan_hash


class Event(Formatted):
    """A `run-event@1.0` document: one state change of one job of one attempt.

    `job` names the job (None for an event of the node as a whole), `key` its key digest, `attempt` the submission it
    belongs to, `placement_digest` the digest of the node's placement, `fault` the class of a failure, `failure` the
    record of it, `reuse` the policy that served a `cache_hit` (its value), `env` the env the job ran in, `output` the
    output identity of a `finished` or `cache_hit` job and `t_ns` the wall-clock nanoseconds of the event.
    """

    format: FormatTag = RUN_EVENT_FORMAT
    port_value: ClassVar[bool] = False
    kind: EventKind
    node: NodeId
    key: Hex64 | None = None
    job: JobRef | None = None
    attempt: Annotated[int, Field(ge=1)] = 1
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
        if self.kind == "failed" and self.fault is None:
            raise ValueError("a failed event states its fault class")
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

    Derived from events (`status_from`); `done` is True once every job seen is `done` or `failed`.
    """

    counts: Mapping[JobState, int]
    faults: Mapping[str, int]
    broken: tuple[BrokenEnv, ...] = ()
    done: bool = False
    nodes: Mapping[str, Mapping[JobState, int]] = field(default_factory=dict)


def status_from(events: Iterable[Event], *, broken_after: int | None = None) -> RunStatus:
    """Return the status of a run from its `events`, given in order.

    A job's state is that of its latest event (`verify` changes none); `faults` counts every `failed` event by its
    class; an env with `broken_after` infrastructure failures (the packaged profile's `infrastructure_replacements`
    when None) is `broken`.
    """
    if broken_after is None:
        from senselab_core.runtime.profiles import ExecutionProfile  # noqa: PLC0415 - read only when needed

        broken_after = ExecutionProfile.load().faults.infrastructure_replacements
    state: dict[tuple[str, tuple[int, ...]], JobState] = {}
    faults: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    latest: dict[str, FaultRecord] = {}
    for event in events:
        moved = _STATE.get(event.kind)
        if moved is not None:
            state[(event.node, event.job.element if event.job is not None else ())] = moved
        if event.kind == "failed" and event.fault is not None:
            faults[event.fault] += 1
            if event.fault == "infrastructure" and event.env is not None and event.failure is not None:
                failures[event.env] += 1
                latest[event.env] = event.failure
    states: tuple[JobState, ...] = ("queued", "running", "done", "failed")
    tally = Counter(state.values())
    per_node: dict[str, Counter[str]] = {}
    for (node, _), moved in state.items():
        per_node.setdefault(node, Counter())[moved] += 1
    broken = tuple(
        BrokenEnv(env=env, fault=latest[env], failures=n) for env, n in sorted(failures.items()) if n >= broken_after
    )
    return RunStatus(
        counts={s: tally[s] for s in states},
        faults=dict(faults),
        broken=broken,
        done=bool(state) and not tally["queued"] and not tally["running"],
        nodes={n: {s: c[s] for s in states} for n, c in sorted(per_node.items())},
    )


class EventLog:
    """One writer's append-only log of a run: `<run dir>/events/<writer uuid>.jsonl`, written by this writer only.

    `clock` gives the `t_ns` of an event `record` makes. A writer used after a `fork` is a new writer in the child.
    """

    def __init__(self, run_dir: Path, *, clock: Callable[[], int] = time.time_ns) -> None:
        """Log to `run_dir`; nothing is created until the first append."""
        self.run_dir = Path(run_dir)
        self.clock = clock
        self._writer = (os.getpid(), uuid.uuid4().hex)
        self._lock = threading.Lock()

    @property
    def writer(self) -> str:
        """This process's writer uuid4 hex: the name of the file `append` writes."""
        if self._writer[0] != os.getpid():
            self._writer = (os.getpid(), uuid.uuid4().hex)
        return self._writer[1]

    @property
    def path(self) -> Path:
        """The file this writer appends to."""
        return self.run_dir / EVENTS_DIR / f"{self.writer}.jsonl"

    def append(self, event: Event) -> None:
        """Append `event` as one line of this writer's file, flushed before returning."""
        line = json.dumps(event.model_dump(mode="json"), sort_keys=True, separators=(",", ":")) + "\n"
        with self._lock:
            path = self.path
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()

    def record(  # noqa: PLR0913 - the fields of an event
        self,
        kind: EventKind,
        node: str,
        *,
        job: JobRef | None = None,
        key: str | None = None,
        attempt: int = 1,
        placement_digest: str | None = None,
        failure: FaultRecord | None = None,
        reuse: str | None = None,
        env: str | None = None,
        output: str | None = None,
    ) -> Event:
        """Append the event of `kind` for `node`, stamped now, and return it; `failure` gives its fault class."""
        event = Event(
            kind=kind,
            node=node,
            job=job,
            key=key,
            attempt=attempt,
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


def read_events(run_dir: Path) -> list[Event]:
    """Return every event of the run in `run_dir`, from all writers, ordered by time, then writer, then line.

    A writer's last line cut short by a crash is skipped; any other line that is not an event raises.

    Raises:
        ValueError: a line other than a file's last is not a `run-event@1` document.
    """
    found: list[tuple[int, str, int, Event]] = []
    for path in sorted((Path(run_dir) / EVENTS_DIR).glob("*.jsonl")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                event = Event.model_validate(json.loads(line))
            except ValueError:
                if number == len(lines) - 1:
                    continue
                raise
            found.append((event.t_ns, path.name, number, event))
    return [e for *_, e in sorted(found, key=lambda f: f[:3])]
