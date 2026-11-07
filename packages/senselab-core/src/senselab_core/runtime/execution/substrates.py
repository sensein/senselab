"""Substrates: what starts the pool workers that serve a run's queues, and `scale`, which sizes them from queue depth.

A `Substrate` starts workers for a queue, lists those alive, counts them and stops given ones. `Local` runs each worker
as a `senselab execute` process under the supervisor that launched it; `Slurm` submits one `sbatch` job per worker, with
the class's partition, gres, CPUs, memory and time, named by plan hash, run, attempt, epoch and a token of its own, and
never on an excluded node. A queue is named by its resource class, with `#<model identity>` for a resident model;
every worker serves one queue.

`scale` is called by the coordinator each round, with one `live()` listing: per queue, the workers wanted are the jobs
waiting on the queue (`Queue.work`) over `pool.quantum_items`, within the class's `max_workers` and the QOS CPU cap over
the class's CPUs per worker. A surplus is stopped, pending workers first, running ones only once the queue has no
unresolved item.

A worker marks itself ready (`mark_ready`) once it can reach the run and its store. `Fleet` counts a worker that left
`live()` without a mark as a failed start of its class, backs the class's next start off, and reports the class broken
after `faults.infrastructure_replacements` consecutive failures; it then starts no more workers of the class.
"""

import logging
import math
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from senselab_core.runtime.execution.queue import RUN_QUEUES, QueueSet
from senselab_core.runtime.execution.supervisor import Supervisor, current
from senselab_core.runtime.execution_settings import ResourceClass, SlurmCommands
from senselab_core.runtime.profiles import ExecutionProfile, expand

logger = logging.getLogger("senselab")

SBATCH, SQUEUE, SCANCEL = "sbatch", "squeue", "scancel"
"""The Slurm commands, looked up on `PATH`."""
LIVE_STATES = "PENDING,CONFIGURING,RUNNING,COMPLETING"
"""The job states `Slurm.live` counts."""
STOP_GRACE_S = 30.0
"""Seconds a local worker is given after SIGTERM before it is killed."""
PROFILE_PREFIX = "profile-"
"""A run's own profile is written as `<PROFILE_PREFIX><digest prefix>.json` where a worker cannot load it by name."""
WORKERS_DIR = "workers"
"""`<run dir>/workers/<worker id>` is a worker's mark that it started."""
EXIT_UNREACHABLE = 75
"""The exit status of a worker that cannot reach its run directory or store."""
SQUEUE_FIELDS = 4
"""The fields of a `squeue` row: id, state, comment, name."""
SQUEUE_FORMAT = "--format=%i|%T|%k|%j"
"""The `squeue` row format: id, state, comment, name."""
TOKEN_HEX = 12
"""Hex digits of the token that names one intended worker's job: 48 random bits, a format width."""
EPOCH_SUFFIX = re.compile(r"-e(\d+)$")


class SubstrateError(RuntimeError):
    """A substrate's command failed; the scaler counts it against the substrate's budget."""


class SlurmError(SubstrateError):
    """A Slurm command kept failing; the message gives the command and what it printed."""


class StoreUnreachable(OSError):
    """A worker cannot reach the run directory or the result store it was started for."""


@dataclass(frozen=True)
class WorkerHandle:
    """A started worker: the queue it serves, its id (a process id, or a Slurm job id) and whether it still waits."""

    queue: str
    id: str
    pending: bool = False


@dataclass(frozen=True)
class BrokenClass:
    """A resource class whose workers failed to start `failures` times in a row; `reason` is the latest cause.

    The class `*` is the substrate itself, which kept failing.
    """

    resource_class: str
    failures: int
    reason: str


class Substrate(Protocol):
    """What starts and stops the pool workers serving a run's queues."""

    def start(self, resource_class: str, count: int) -> list[WorkerHandle]:
        """Start up to `count` workers of queue `resource_class`; return those started."""

    def live(self) -> list[WorkerHandle]:
        """Return every worker of this run alive, pending ones included."""

    def running(self, resource_class: str) -> int:
        """Return the workers of queue `resource_class` alive, pending ones included."""

    def stop(self, handles: Sequence[WorkerHandle]) -> None:
        """Stop the workers `handles`."""


def class_of(queue: str) -> str:
    """Return the resource class of queue `queue`: its name before any `#<model identity>`."""
    return queue.partition("#")[0]


def worker_command(run_dir: Path, profile_argument: str) -> list[str]:
    """Return the `senselab execute` command of the run in `run_dir`, without its `--classes`."""
    pool = str(Path(run_dir) / RUN_QUEUES)
    return [sys.executable, "-m", "senselab_core.cli", "execute", "--pool", pool, "--profile", profile_argument]


def worker_id() -> str:
    """Return this process's worker id: its Slurm job id, else its process id."""
    return os.environ.get("SLURM_JOB_ID") or str(os.getpid())


def mark_ready(run_dir: Path) -> None:
    """Record that this worker started: it reached the run and its store."""
    directory = Path(run_dir) / WORKERS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    (directory / worker_id()).touch()


class ExternalWorkers:
    """A `Substrate` that starts nothing: workers are started on their own (`senselab execute --pool`)."""

    def start(self, resource_class: str, count: int) -> list[WorkerHandle]:  # noqa: ARG002
        """Start nothing."""
        return []

    def live(self) -> list[WorkerHandle]:
        """Return no worker."""
        return []

    def running(self, resource_class: str) -> int:  # noqa: ARG002
        """Return 0."""
        return 0

    def stop(self, handles: Sequence[WorkerHandle]) -> None:
        """Stop nothing."""


class Local:
    """A `Substrate` of local processes: each worker is `command` with `--classes <queue>`, in its own group.

    At most `max_workers` are alive at once, across queues. A process is launched through the supervisor in force
    when it starts, and stopped and released through that one.
    """

    def __init__(self, max_workers: int, *, command: Sequence[str]) -> None:
        """Run at most `max_workers` workers, each `command` plus `--classes <queue>`.

        Raises:
            ValueError: `max_workers` is below 1.
        """
        if max_workers < 1:
            raise ValueError(f"a local substrate runs at least one worker, got {max_workers}")
        self.max_workers = max_workers
        self.command = list(command)
        self._procs: dict[str, tuple[str, subprocess.Popen[bytes], Supervisor]] = {}
        self._lock = threading.Lock()

    def _alive(self) -> dict[str, tuple[str, "subprocess.Popen[bytes]", Supervisor]]:
        for key, (_, proc, supervisor) in list(self._procs.items()):
            if proc.poll() is not None:
                supervisor.released(proc.pid)
                del self._procs[key]
        return self._procs

    def start(self, resource_class: str, count: int) -> list[WorkerHandle]:
        """Start workers of `resource_class`, as many as `count` and the room under `max_workers` allow."""
        with self._lock:
            room = self.max_workers - len(self._alive())
            handles = []
            for _ in range(max(0, min(count, room))):
                supervisor = current()
                proc = supervisor.popen([*self.command, "--classes", resource_class])
                self._procs[str(proc.pid)] = (resource_class, proc, supervisor)
                handles.append(WorkerHandle(resource_class, str(proc.pid)))
            return handles

    def live(self) -> list[WorkerHandle]:
        """Return the workers alive."""
        with self._lock:
            return [WorkerHandle(queue, key) for key, (queue, _, _) in self._alive().items()]

    def running(self, resource_class: str) -> int:
        """Return the workers of `resource_class` alive."""
        return sum(1 for h in self.live() if h.queue == resource_class)

    def stop(self, handles: Sequence[WorkerHandle]) -> None:
        """SIGTERM each worker of `handles`, which commits what finished; kill one that outlasts `STOP_GRACE_S`."""
        with self._lock:
            found = [self._procs.pop(h.id) for h in handles if h.id in self._procs]
        for _, proc, _ in found:
            if proc.poll() is None:
                proc.terminate()
        for _, proc, supervisor in found:
            try:
                proc.wait(STOP_GRACE_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            supervisor.released(proc.pid)


def _minutes(seconds: float) -> int:
    return max(1, math.ceil(seconds / 60))


def _rows(out: str) -> list[tuple[str, str, str, str]]:
    """Return the `(id, state, comment, name)` of each row of `squeue` output in `SQUEUE_FORMAT`."""
    rows = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split("|", SQUEUE_FIELDS - 1)]
        if len(parts) == SQUEUE_FIELDS:
            rows.append((parts[0], parts[1], parts[2], parts[3]))
    return rows


class Slurm:
    """A `Substrate` of Slurm jobs: one `sbatch` job per worker, within a QOS CPU cap, never on an excluded node.

    A job is named `senselab-<plan hash prefix>-<run digest>-<attempt>-<token>-e<epoch>`, the token drawn per intended
    worker, and carries its queue as its comment. The jobs of this run, whatever their attempt or epoch, are those of
    the user (`squeue --me`) whose name starts `senselab-<plan hash prefix>-<run digest>-`. `live` cancels those of an
    earlier epoch, ignores those of a later one, and returns this epoch's. The class's `partition`, `gres`, `cpus`,
    `mem_bytes` and `time_s` become the job's `--partition`, `--gres`, `--cpus-per-task`, `--mem` and `--time`; `${VAR}`
    in a partition, a gres or an excluded node is expanded from the environment where the job is submitted. A Slurm
    command runs within `commands.timeout_s` and is tried `commands.attempts` times, `commands.retry_first_s` apart and
    doubling, before `SlurmError`. A submission that fails or times out is looked up by its name before it is tried
    again; a job found under that name is the one submitted.

    Attributes:
        attempt: The attempt id the jobs are named by; the pool sets it to its attempt's.
        epoch: The coordinator epoch the jobs are named by; the pool sets it to its own.
    """

    def __init__(  # noqa: PLR0913 - the classes and cap, then keyword-only submission settings
        self,
        classes: Mapping[str, ResourceClass],
        qos_cpu_cap: int | None,
        *,
        command: Sequence[str],
        plan_hash: str,
        run_dir: Path,
        attempt: str = "a0",
        epoch: int = 0,
        exclude_nodes: Sequence[str] = (),
        log_dir: Path | None = None,
        commands: SlurmCommands,
    ) -> None:
        """Submit workers of `classes`, each `command` plus `--classes <queue>`, at most `qos_cpu_cap` CPUs alive."""
        self.classes = dict(classes)
        self.qos_cpu_cap = qos_cpu_cap
        self.command = list(command)
        self.plan_hash = plan_hash
        self.run_dir = Path(run_dir)
        self.attempt = attempt
        self.epoch = epoch
        self.exclude_nodes = tuple(exclude_nodes)
        self.log_dir = log_dir
        self.commands = commands
        self._cache: list[WorkerHandle] | None = None

    @property
    def prefix(self) -> str:
        """The start of the name of every job of this run."""
        run = uuid.uuid5(uuid.NAMESPACE_URL, str(self.run_dir.resolve())).hex[:8]
        return f"senselab-{self.plan_hash[:12]}-{run}-"

    def job_name(self, token: str) -> str:
        """Return the name of this coordinator's job drawn under `token`."""
        return f"{self.prefix}{self.attempt}-{token}-e{self.epoch}"

    def _class(self, queue: str) -> ResourceClass:
        try:
            return self.classes[class_of(queue)]
        except KeyError:
            raise ValueError(
                f"queue {queue!r} is of no resource class of this substrate: {sorted(self.classes)}"
            ) from None

    def sbatch_args(self, queue: str, name: str) -> list[str]:
        """Return the `sbatch` command line of one worker of `queue`, named `name`.

        Raises:
            ValueError: the queue's class is unknown, or names a variable that is not set.
        """
        grant = self._class(queue)
        args = [SBATCH, "--parsable", f"--job-name={name}", f"--comment={queue}"]
        if grant.partition is not None:
            args.append(f"--partition={expand(grant.partition)}")
        if grant.gres is not None:
            args.append(f"--gres={expand(grant.gres)}")
        if grant.cpus is not None:
            args.append(f"--cpus-per-task={grant.cpus}")
        if grant.mem_bytes is not None:
            args.append(f"--mem={math.ceil(grant.mem_bytes / 2**20)}M")
        if grant.time_s is not None:
            args.append(f"--time={_minutes(grant.time_s)}")
        if self.exclude_nodes:
            args.append("--exclude=" + ",".join(expand(n) for n in self.exclude_nodes))
        if self.log_dir is not None:
            args.append(f"--output={self.log_dir}/%x-%j.out")
        return [*args, "--wrap", shlex.join([*self.command, "--classes", queue])]

    def _try(self, argv: Sequence[str]) -> tuple[str | None, str]:
        """Run `argv` once; return its output, or None and why it failed."""
        timeout = self.commands.timeout_s
        try:
            done = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=timeout)  # noqa: S603
        except subprocess.TimeoutExpired:
            return None, f"timed out after {timeout} s"
        if done.returncode == 0:
            return done.stdout, ""
        return None, f"exited {done.returncode}: {(done.stderr or done.stdout).strip()}"

    def _run(self, argv: Sequence[str], recover: Callable[[], str | None] | None = None) -> str:
        """Run `argv` until it succeeds, at most `commands.attempts` times, `commands.retry_first_s` apart and doubling.

        Without `recover` the command must be idempotent. With it, each failed try is followed, after the wait, by
        `recover()`, which returns the output the try had when it took effect after all; that ends the tries.

        Raises:
            SlurmError: no try succeeded and `recover` found nothing.
        """
        wait, last = self.commands.retry_first_s, ""
        for attempt in range(self.commands.attempts):
            out, last = self._try(argv)
            if out is not None:
                return out
            if recover is None and attempt + 1 == self.commands.attempts:
                break
            time.sleep(wait)
            wait *= 2
            if recover is not None and (found := recover()) is not None:
                return found
        raise SlurmError(f"{shlex.join(argv)} {last}")

    def _submit(self, queue: str) -> str:
        """Submit one worker of `queue` under a name of its own; return its job id.

        A failed try is followed by a lookup of the name (`squeue --me --name`); a job found is the one submitted.

        Raises:
            SlurmError: no try submitted a job found under its name, or the lookup failed, so the outcome is unknown.
        """
        name = self.job_name(uuid.uuid4().hex[:TOKEN_HEX])

        def lookup() -> str | None:
            out = self._run([SQUEUE, "--me", "--noheader", "--states=all", f"--name={name}", SQUEUE_FORMAT])
            return next((job for job, _, _, job_name in _rows(out) if job_name == name), None)

        return self._run(self.sbatch_args(queue, name), recover=lookup).strip().split(";")[0]

    def live(self) -> list[WorkerHandle]:
        """Return this epoch's live jobs of the run, from one `squeue`; cancel those of an earlier epoch."""
        out = self._run([SQUEUE, "--me", "--noheader", f"--states={LIVE_STATES}", SQUEUE_FORMAT])
        mine, stale = [], []
        for job, state, comment, name in _rows(out):
            if not name.startswith(self.prefix):
                continue
            found = EPOCH_SUFFIX.search(name)
            epoch = int(found[1]) if found else 0
            handle = WorkerHandle(comment, job, pending=state == "PENDING")
            if epoch == self.epoch:
                mine.append(handle)
            elif epoch < self.epoch:
                stale.append(handle)
        if stale:
            self._run([SCANCEL, *(h.id for h in stale)])
        self._cache = mine
        return mine

    def _cached(self) -> list[WorkerHandle]:
        return self._cache if self._cache is not None else self.live()

    def running(self, resource_class: str) -> int:
        """Return the live jobs of queue `resource_class`, from the last listing."""
        return sum(1 for h in self._cached() if h.queue == resource_class)

    def start(self, resource_class: str, count: int) -> list[WorkerHandle]:
        """Submit workers of `resource_class`, as many as `count` and the room under `qos_cpu_cap` allow.

        Raises:
            ValueError: the queue's class is unknown, or names a variable that is not set.
            SlurmError: `sbatch` kept failing.
        """
        self._class(resource_class)
        cached = self._cached()
        if self.qos_cpu_cap is not None:
            used = sum(self._cpus(h.queue) for h in cached)
            count = min(count, (self.qos_cpu_cap - used) // self._cpus(resource_class))
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        handles = []
        for _ in range(max(0, count)):
            handles.append(WorkerHandle(resource_class, self._submit(resource_class), pending=True))
            cached.append(handles[-1])
        return handles

    def _cpus(self, queue: str) -> int:
        return self._class(queue).cpus or 1

    def stop(self, handles: Sequence[WorkerHandle]) -> None:
        """Cancel the jobs of `handles`."""
        if handles:
            self._run([SCANCEL, *(h.id for h in handles)])
            gone = {h.id for h in handles}
            self._cache = [h for h in self._cached() if h.id not in gone]


class Fleet:
    """A `Substrate` that records the workers its substrate starts, budgets their starts and lists them once a round.

    A worker that left `live()` without having marked itself ready (`mark_ready`, in `workers_dir`) and that this
    fleet did not stop is a failed start of its class: the class's next start waits `first_s`, doubling per failure up
    to `max_s`, and after `budget` failures in a row the class is broken (`broken`) and starts no more workers.
    A worker that marked itself ready ends the run of failures. `fail_substrate` counts the same way the failures of
    the substrate's own commands, under the class `*`; `substrate_due` says when the substrate may be used again.

    Attributes:
        broken: Each broken class (and `*`), once.
    """

    def __init__(  # noqa: PLR0913 - the substrate, then keyword-only budget and clock
        self,
        substrate: Substrate,
        *,
        workers_dir: Path | None,
        budget: int,
        first_s: float,
        max_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Wrap `substrate`; `budget` consecutive failed starts of a class break it."""
        self.substrate = substrate
        self.workers_dir = workers_dir
        self.budget, self.first_s, self.max_s, self._clock = budget, first_s, max_s, clock
        self.broken: dict[str, BrokenClass] = {}
        self.starts: Counter[str] = Counter()
        self._tracked: dict[str, WorkerHandle] = {}
        self._failures: Counter[str] = Counter()
        self._not_before: dict[str, float] = {}
        self._snapshot: list[WorkerHandle] | None = None

    def reset(self) -> None:
        """Forget every failure and every broken class, as a new epoch does."""
        self.broken.clear()
        self._failures.clear()
        self._not_before.clear()

    def _ready(self, handle: WorkerHandle) -> bool:
        return self.workers_dir is not None and (self.workers_dir / handle.id).exists()

    def _fail(self, resource_class: str, reason: str) -> None:
        self._failures[resource_class] += 1
        n = self._failures[resource_class]
        self._not_before[resource_class] = self._clock() + min(self.first_s * 2 ** (n - 1), self.max_s)
        if n > self.budget and resource_class not in self.broken:
            self.broken[resource_class] = BrokenClass(resource_class, n, reason)
            logger.error("workers of %s are broken after %d failed starts: %s", resource_class, n, reason)

    def fail_substrate(self, err: Exception) -> None:
        """Count a failure of the substrate's own commands; the substrate is broken after `budget` in a row."""
        self._fail("*", str(err))

    def substrate_due(self) -> bool:
        """Whether the substrate may be used now: it is not broken, and its latest failure's wait has passed."""
        return "*" not in self.broken and self._clock() >= self._not_before.get("*", 0.0)

    def substrate_ok(self) -> None:
        """End the run of the substrate's failures."""
        self._failures["*"] = 0

    def broken_for(self, queue: str) -> BrokenClass | None:
        """Return the break that stops starts of `queue`: its class's, else the substrate's."""
        return self.broken.get(class_of(queue)) or self.broken.get("*")

    def live(self) -> list[WorkerHandle]:
        """List the substrate's workers once, and judge the tracked workers that left it."""
        found = self.substrate.live()
        ids = {h.id for h in found}
        for handle in list(self._tracked.values()):
            if handle.id in ids:
                if self._ready(handle):
                    self._failures[class_of(handle.queue)] = 0
                continue
            del self._tracked[handle.id]
            if self._ready(handle):
                self._failures[class_of(handle.queue)] = 0
            else:
                self._fail(
                    class_of(handle.queue), f"a worker of {handle.queue} (id {handle.id}) exited before it started"
                )
        self._snapshot = list(found)
        return found

    def start(self, resource_class: str, count: int) -> list[WorkerHandle]:
        """Start through the substrate unless the class is broken or backing off; record the workers.

        Once the class has a failed start on record, at most `budget + 1` less its failures less its started workers
        not yet ready (pending jobs excluded) are started, so a command that fails at once is run at most `budget + 1`
        times; before that only `max_workers` and the QOS cap bound starts.
        """
        cls = class_of(resource_class)
        if self.broken_for(resource_class) is not None or self._clock() < self._not_before.get(cls, 0.0):
            return []
        if self._failures[cls] > 0:
            pending = {h.id for h in self._snapshot or [] if h.pending}
            unproven = sum(
                1
                for h in self._tracked.values()
                if class_of(h.queue) == cls and h.id not in pending and not self._ready(h)
            )
            count = min(count, self.budget + 1 - self._failures[cls] - unproven)
            if count <= 0:
                return []
        started = self.substrate.start(resource_class, count)
        for handle in started:
            if self.workers_dir is not None:
                (self.workers_dir / handle.id).unlink(missing_ok=True)
            self._tracked[handle.id] = handle
        self.starts[cls] += len(started)
        return started

    def running(self, resource_class: str) -> int:
        """Return the workers of the queue in the last listing."""
        if self._snapshot is None:
            self.live()
        return sum(1 for h in self._snapshot or [] if h.queue == resource_class)

    def stop(self, handles: Sequence[WorkerHandle]) -> None:
        """Stop `handles` through the substrate and forget them."""
        self.substrate.stop(handles)
        gone = {h.id for h in handles}
        for key in gone:
            self._tracked.pop(key, None)
        if self._snapshot is not None:
            self._snapshot = [h for h in self._snapshot if h.id not in gone]

    def close(self) -> None:
        """Stop every worker still alive that this fleet started."""
        try:
            alive = {h.id for h in self.substrate.live()}
        except SubstrateError:
            alive = set(self._tracked)
        self.stop([h for h in self._tracked.values() if h.id in alive])


def scale(
    queues: QueueSet,
    substrate: Substrate,
    profile: ExecutionProfile,
    *,
    surplus_since: dict[str, float] | None = None,
    now: float | None = None,
) -> None:
    """Start or stop the workers each queue of `queues` needs, from one `live()` listing of `substrate`.

    A queue wants `ceil(work / pool.quantum_items)` workers, `work` being the jobs of its unresolved items. Fewer alive
    start workers, within the room left in the class (`max_workers` less the workers alive on its queues) and, under a
    QOS cap, the CPUs left over the class's CPUs per worker. More alive stop the surplus, pending workers first; running
    ones are stopped only once the queue has no unresolved item, so none is working. With `surplus_since` (the caller
    keeps it between rounds) a surplus is stopped only once it has lasted `2 × pool.poll_s`, so a queue that briefly
    empties between items keeps its workers; without it a surplus is stopped at once. `now` is the clock,
    `time.monotonic` by default.
    """
    names = queues.names()
    live = substrate.live()
    alive = {name: [h for h in live if h.queue == name] for name in names}
    cap = profile.substrate.qos_cpu_cap

    def cpus(name: str) -> int:
        grant = profile.resource_classes.get(class_of(name))
        return grant.cpus or 1 if grant is not None else 1

    used = sum(len(hs) * cpus(name) for name, hs in alive.items())
    for name in sorted(names):
        grant = profile.resource_classes.get(class_of(name))
        if grant is None:
            continue
        work = queues.queue(name).work()
        wanted = math.ceil(work / profile.pool.quantum_items)
        have = len(alive[name])
        if have > wanted:
            if surplus_since is not None:
                at = time.monotonic() if now is None else now
                if at - surplus_since.setdefault(name, at) < 2 * profile.pool.poll_s:
                    continue
                del surplus_since[name]
            victims = sorted(alive[name], key=lambda h: not h.pending)
            victims = victims if work == 0 else [h for h in victims if h.pending]
            substrate.stop(victims[: have - wanted])
            continue
        if surplus_since is not None:
            surplus_since.pop(name, None)
        room = grant.max_workers - sum(len(hs) for other, hs in alive.items() if class_of(other) == class_of(name))
        if cap is not None:
            room = min(room, (cap - used) // cpus(name))
        count = min(wanted - have, room)
        if count > 0:
            started = substrate.start(name, count)
            used += len(started) * cpus(name)
            alive[name].extend(started)


def profile_argument(profile: ExecutionProfile, run_dir: Path) -> str:
    """Return what `senselab execute --profile` takes to load `profile`: its packaged name, else a file written.

    A profile that differs from the packaged one of its name is written to `<run dir>/profile-<digest prefix>.json`,
    so two coordinators with different profiles never share a file.
    """
    digest = profile.canonical_digest()
    try:
        if ExecutionProfile.load(profile.name).canonical_digest() == digest:
            return profile.name
    except (FileNotFoundError, ValueError):
        pass
    path = Path(run_dir) / f"{PROFILE_PREFIX}{digest[:16]}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{os.getpid()}")
    staging.write_text(profile.model_dump_json(), encoding="utf-8")
    staging.replace(path)
    return str(path)


def substrate_of(profile: ExecutionProfile, run_dir: Path, plan_hash: str) -> Substrate:
    """Return the substrate `profile` names for the run in `run_dir` of plan `plan_hash`.

    `local` is `Local` over the sum of the classes' `max_workers`; `slurm` is `Slurm` over the profile's classes,
    QOS cap and excluded nodes, logging under `<run dir>/slurm`.
    """
    command = worker_command(run_dir, profile_argument(profile, run_dir))
    spec = profile.substrate
    if spec.kind == "local":
        return Local(sum(c.max_workers for c in profile.resource_classes.values()), command=command)
    if spec.commands is None:
        raise ValueError("a slurm substrate needs its commands settings")
    return Slurm(
        profile.resource_classes,
        spec.qos_cpu_cap,
        command=command,
        plan_hash=plan_hash,
        run_dir=Path(run_dir),
        exclude_nodes=spec.exclude_nodes,
        log_dir=Path(run_dir) / "slurm",
        commands=spec.commands,
    )


__all__ = [
    "EXIT_UNREACHABLE",
    "PROFILE_PREFIX",
    "WORKERS_DIR",
    "BrokenClass",
    "ExternalWorkers",
    "Fleet",
    "Local",
    "Slurm",
    "SlurmError",
    "StoreUnreachable",
    "Substrate",
    "SubstrateError",
    "WorkerHandle",
    "class_of",
    "mark_ready",
    "profile_argument",
    "scale",
    "substrate_of",
    "worker_command",
    "worker_id",
]
