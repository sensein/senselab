"""Substrates: what starts the pool workers that serve a run's queues, and `scale`, which sizes them from queue depth.

A `Substrate` starts workers for a queue, lists those alive, counts them and stops given ones. `Local` runs each worker
as a `senselab execute` process under the supervisor that launched it, a `cuda` class's on a visible device of its own;
`Slurm` submits one `sbatch` job per worker, with the class's partition, gres, CPUs, memory and time, named by plan
hash, run, attempt, epoch and a token of its own, and never on an excluded node. A queue is named by its resource
class, with `#<model identity>` for a resident model; every worker serves one queue.

`scale` is called by the coordinator each round, with one `live()` listing: per queue, the workers its waiting jobs
(`Queue.work`) want, as the coordinator sizes them, within the class's `max_workers` and the QOS CPU cap over the
class's CPUs per worker. A surplus is stopped, pending workers first, running ones only once the queue has no
unresolved item.

A worker marks itself ready (`mark_ready`) once it can reach the run and its store; the mark names its host by a digest,
its process and its `HardwareClass` per resource class, and writes no hardware index. `Fleet` counts a worker that left
`live()` without a mark as a failed start of its class, backs the class's next start off, and reports the class broken
after `faults.infrastructure_replacements` consecutive failures; it then starts no more workers of the class.
"""

import logging
import math
import os
import re
import secrets
import shlex
import subprocess
import sys
import threading
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from senselab_worker.canonical import jcs_dumps, jcs_loads

from senselab_core.kernel.declare import BackendSpec
from senselab_core.kernel.formats import WORKER_READY_FORMAT, FormatVerdict, check, reading
from senselab_core.runtime.calibration import HardwareClass
from senselab_core.runtime.clock import Clock, SystemClock
from senselab_core.runtime.config import expand
from senselab_core.runtime.execution.queue import PROBE_MARK, RUN_QUEUES, QueueSet
from senselab_core.runtime.execution.supervisor import Supervisor, current
from senselab_core.runtime.execution_settings import ResourceClass, SlurmCommands, is_service_class
from senselab_core.runtime.hardware import HardwareInfo, host_token, probe_hardware, process_started, visible_devices
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.site_placement import FIELDS, Placer, rendered, site_names
from senselab_core.runtime.storage import PRIVATE_DIR, private_dir

logger = logging.getLogger("senselab")

SBATCH, SQUEUE, SCANCEL = "sbatch", "squeue", "scancel"
"""The Slurm commands, looked up on `PATH`."""
CLIENTS = frozenset({SBATCH, SQUEUE, SCANCEL})
"""The only Slurm clients `Slurm` runs; it refuses any other (`UndeclaredClient`)."""
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
    """A substrate's command failed; the scaler counts it against the substrate's budget.

    Attributes:
        started: The workers the failing `start` had started before it failed.
    """

    def __init__(self, message: str, started: Sequence["WorkerHandle"] = ()) -> None:
        """Fail with `message`, after starting `started`."""
        super().__init__(message)
        self.started = tuple(started)


class SlurmError(SubstrateError):
    """A Slurm command kept failing, or could not be run; the message gives the command and why.

    Attributes:
        site_message: What `sbatch` said when it exited, the scheduler's reply to a submission; None for a command
            that timed out or could not be run, and for any other command.
    """

    def __init__(self, message: str, started: Sequence["WorkerHandle"] = (), site_message: str | None = None) -> None:
        """Fail with `message`, after starting `started`; `site_message` is the scheduler's reply to `sbatch`."""
        super().__init__(message, started)
        self.site_message = site_message


class UndeclaredClient(RuntimeError):
    """`Slurm` was asked to run a command that is not one of its declared clients (`CLIENTS`)."""


class SiteRefused(SlurmError):
    """The scheduler refused a submission: the class, the `sbatch` line, the site's message and who set each value.

    The text is rendered as for any record other users read (`rendered`): accounts, partitions, nodes and paths
    appear as labels; the operator's own text is logged.

    Attributes:
        resource_class: The class whose worker was refused.
        layers: The layer, `profile`, `override` or `hook`, that set each placement value of the class.
    """

    def __init__(  # noqa: PLR0913 - the refusal's four parts, the names to render, the workers started before it
        self,
        resource_class: str,
        request_line: str,
        site_message: str,
        layers: Mapping[str, str],
        names: Mapping[str, str],
        started: Sequence["WorkerHandle"] = (),
    ) -> None:
        """Refuse with the `sbatch` line and the site's message, rendered, and name the inputs that change it."""
        set_by = ", ".join(f"{k} by the {v}" for k, v in layers.items())
        text = (
            f"the site refused the submission of class {resource_class!r}: {request_line}: {site_message} "
            f"(values set {set_by}; change them with --placement or the placement hook, then resume)"
        )
        super().__init__(rendered(text, names), started, rendered(site_message, names))
        self.resource_class, self.layers = resource_class, dict(layers)


class LocalError(SubstrateError):
    """A local worker process could not be launched."""


class StoreUnreachable(OSError):
    """A worker cannot reach the run directory or the result store it was started for."""


@dataclass(frozen=True)
class WorkerHandle:
    """A started worker: the queue it serves, its id (a process id, or a Slurm job id), whether it still waits.

    `name` is a Slurm job's name; None for a process.
    """

    queue: str
    id: str
    pending: bool = False
    name: str | None = None


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
    """Return the resource class of queue `queue`: its name before any `#<model identity>` or probe mark."""
    return queue.partition("#")[0].partition(PROBE_MARK)[0]


def worker_command(run_dir: Path, profile_argument: str) -> list[str]:
    """Return the `senselab execute` command of the run in `run_dir`, without its `--classes`."""
    pool = str(Path(run_dir) / RUN_QUEUES)
    return [sys.executable, "-m", "senselab_core.cli", "execute", "--pool", pool, "--profile", profile_argument]


_NONCE = secrets.token_hex(4)
"""This process's nonce: a reused process id on one host still names another worker."""


def worker_id() -> str:
    """Return this process's worker id: its Slurm job id (when set), an opaque host token, its process id and a nonce.

    Workers of one allocation share the job id, so it alone does not tell them apart; the process id does on one host,
    the host token between hosts and the nonce across a process id reused. The id names no host, account or path.
    """
    job = os.environ.get("SLURM_JOB_ID")
    return "-".join(([job] if job else []) + [host_token()[:8], str(os.getpid()), _NONCE])


@dataclass(frozen=True)
class QueueReady:
    """What a worker says of one queue it serves: its `HardwareClass` (None: it names none) and each backend's env.

    An env is its `env_entry`, or `{"unbuilt": "build-failed" | "unreadable"}`.
    """

    hardware: HardwareClass | None
    envs: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class ReadyMark:
    """A `worker-ready@1.0` mark: its host's token, its process id and start, and what it says per queue."""

    host: str
    pid: int
    started: str | None
    queues: Mapping[str, QueueReady]


def backend_key(backend: BackendSpec) -> str:
    """Return the name a ready mark gives `backend`'s env: its task and backend, which fix its env and overlay."""
    return f"{backend.task}~{backend.name}"


def mark_ready(run_dir: Path, queues: Mapping[str, QueueReady] | None = None) -> None:
    """Record that this worker started: it reached the run and its store, and what it says of each queue it serves."""
    directory = Path(run_dir) / WORKERS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    document = {
        "format": WORKER_READY_FORMAT,
        "host": host_token(),
        "pid": os.getpid(),
        "started": process_started(os.getpid()),
        "queues": {
            name: {
                "hardware": None if ready.hardware is None else ready.hardware.model_dump(mode="json"),
                "envs": {k: dict(v) for k, v in ready.envs.items()},
            }
            for name, ready in (queues or {}).items()
        },
    }
    staged = directory / f".{worker_id()}.{os.getpid()}"
    staged.write_bytes(jcs_dumps(document))
    staged.replace(directory / worker_id())


def read_mark(path: Path) -> ReadyMark | None:
    """Return the `worker-ready@1.0` mark at `path`; None when it is unreadable or of another format (logged once)."""
    try:
        document = jcs_loads(Path(path).read_bytes())
        if not isinstance(document, dict):
            return None
        if check(WORKER_READY_FORMAT, document.get("format")) is not FormatVerdict.ACCEPTED:
            _say_once(
                ("mark", str(document.get("format"))),
                "a worker mark is %r; %r is read",
                document.get("format"),
                WORKER_READY_FORMAT,
            )
            return None
        with reading(WORKER_READY_FORMAT, document["format"]):
            return ReadyMark(
                document["host"],
                document["pid"],
                document["started"],
                {
                    name: QueueReady(
                        None if found["hardware"] is None else HardwareClass.model_validate(found["hardware"]),
                        {k: dict(v) for k, v in found["envs"].items()},
                    )
                    for name, found in document["queues"].items()
                },
            )
    except (OSError, ValueError, KeyError, TypeError):
        return None


_SAID: set[tuple[str, ...]] = set()


def _say_once(key: tuple[str, ...], message: str, *args: object) -> None:
    if key not in _SAID:
        _SAID.add(key)
        logger.warning(message, *args)


class Peers:
    """The other pool workers of a run alive on this host, from the marks in its `workers` directory.

    A mark never changes once written, so each is read once; a listing and a liveness check per worker remain. A
    worker is alive while its process is, and, where `/proc` says when it started, the same process.

    Attributes:
        reads: The marks read so far.
    """

    def __init__(self, run_dir: Path) -> None:
        """Watch the workers of the run in `run_dir`."""
        self._directory = Path(run_dir) / WORKERS_DIR
        self._marks: dict[str, ReadyMark] = {}
        self.reads = 0

    def alive(self) -> dict[int, ReadyMark]:
        """Return each other worker of this host alive now, by pid; this process is not one."""
        names = [p.name for p in self._directory.iterdir()] if self._directory.is_dir() else []
        for name in names:
            if name not in self._marks and not name.startswith("."):
                self.reads += 1
                if (mark := read_mark(self._directory / name)) is not None:
                    self._marks[name] = mark
        here, mine = host_token(), os.getpid()
        return {
            m.pid: m
            for n, m in self._marks.items()
            if n in names and m.host == here and m.pid != mine and _alive(m.pid, m.started)
        }


def _alive(pid: int, started: str | None) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return started is not None and process_started(pid) == started
    return started is None or process_started(pid) in (None, started)


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

    At most `max_workers` are alive at once, across queues. A worker of a `cuda` class (`cuda`, by name) is given one
    of `devices` no other live worker holds, as its `CUDA_VISIBLE_DEVICES`, so no more run than there are devices. A
    process is launched through the supervisor in force when it starts, and stopped and released through that one.
    """

    def __init__(
        self, max_workers: int, *, command: Sequence[str], cuda: Collection[str] = (), devices: Sequence[str] = ()
    ) -> None:
        """Run at most `max_workers` workers, each `command` plus `--classes <queue>`; `cuda` classes on `devices`.

        Raises:
            ValueError: `max_workers` is below 1.
        """
        if max_workers < 1:
            raise ValueError(f"a local substrate runs at least one worker, got {max_workers}")
        self.max_workers = max_workers
        self.command = list(command)
        self.cuda = frozenset(cuda)
        self.devices = tuple(devices)
        self._procs: dict[str, _Process] = {}
        self._lock = threading.Lock()

    def _alive(self) -> dict[str, "_Process"]:
        for key, held in list(self._procs.items()):
            if held.proc.poll() is not None:
                held.supervisor.released(held.proc.pid)
                del self._procs[key]
        return self._procs

    def start(self, resource_class: str, count: int) -> list[WorkerHandle]:
        """Start workers of `resource_class`, as many as `count`, the room under `max_workers` and free devices allow.

        Raises:
            LocalError: a process could not be launched, those launched before it kept and listed; or `resource_class`
                is a `cuda` class and no device is visible.
        """
        if class_of(resource_class) in self.cuda and not self.devices:
            raise LocalError(f"no device is visible to run {resource_class!r} on (CUDA_VISIBLE_DEVICES, nvidia-smi)")
        with self._lock:
            alive = self._alive()
            room = self.max_workers - len(alive)
            held = {p.device for p in alive.values()}
            free: list[str | None] = [d for d in self.devices if d not in held]
            if class_of(resource_class) not in self.cuda:
                free = [None] * max(0, room)
            handles: list[WorkerHandle] = []
            for device in free[: max(0, min(count, room))]:
                supervisor = current()
                argv = [*self.command, "--classes", resource_class]
                env = None if device is None else {**os.environ, "CUDA_VISIBLE_DEVICES": device}
                try:
                    proc = supervisor.popen(argv, env=env)
                except OSError as err:
                    raise LocalError(f"could not launch {shlex.join(argv)}: {err}", handles) from err
                self._procs[str(proc.pid)] = _Process(resource_class, proc, supervisor, device)
                handles.append(WorkerHandle(resource_class, str(proc.pid)))
            return handles

    def live(self) -> list[WorkerHandle]:
        """Return the workers alive."""
        with self._lock:
            return [WorkerHandle(p.queue, key) for key, p in self._alive().items()]

    def running(self, resource_class: str) -> int:
        """Return the workers of `resource_class` alive."""
        return sum(1 for h in self.live() if h.queue == resource_class)

    def stop(self, handles: Sequence[WorkerHandle]) -> None:
        """SIGTERM each worker of `handles`, which commits what finished; kill one that outlasts `STOP_GRACE_S`."""
        with self._lock:
            found = [self._procs.pop(h.id) for h in handles if h.id in self._procs]
        for held in found:
            if held.proc.poll() is None:
                held.proc.terminate()
        for held in found:
            try:
                held.proc.wait(STOP_GRACE_S)
            except subprocess.TimeoutExpired:
                held.proc.kill()
                held.proc.wait()
            held.supervisor.released(held.proc.pid)


@dataclass(frozen=True)
class _Process:
    """A local worker: its queue, process, the supervisor it was launched through, and its device (None for none)."""

    queue: str
    proc: "subprocess.Popen[bytes]"
    supervisor: Supervisor
    device: str | None


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
        clock: Clock = SystemClock(),  # noqa: B008 - stateless
        placer: Placer | None = None,
    ) -> None:
        """Submit workers of `classes`, each `command` plus `--classes <queue>`, at most `qos_cpu_cap` CPUs alive.

        The waits between a command's tries run on `clock`. With a `placer` each class's partition, time, worker
        cap and excluded nodes are its placement for the epoch (`Placer.for_class`), asked for before the class's
        first submission; without one they are the class's own.
        """
        self.classes = dict(classes)
        self.qos_cpu_cap = qos_cpu_cap
        self.command = list(command)
        self.plan_hash = plan_hash
        self.run_dir = Path(run_dir)
        self.attempt = attempt
        self.epoch = epoch
        self.exclude_nodes = tuple(exclude_nodes)
        self.log_dir = log_dir
        self.commands, self._clock, self.placer = commands, clock, placer
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

    def _granted(self, queue: str) -> tuple[ResourceClass, tuple[str, ...], Mapping[str, str], Mapping[str, str]]:
        """Return the class of `queue` as it is placed this epoch, its excluded nodes, each value's layer, its names."""
        base = self._class(queue)
        if self.placer is None:
            names = site_names([base.partition], self.exclude_nodes)
            return base, self.exclude_nodes, dict.fromkeys(FIELDS, "profile"), names
        placed = self.placer.for_class(class_of(queue), self.epoch)
        return placed.grant(base), placed.exclude, placed.values.layers, placed.names

    def sbatch_args(self, queue: str, name: str) -> list[str]:
        """Return the `sbatch` command line of one worker of `queue`, named `name`, under its placement.

        Raises:
            ValueError: the queue's class is unknown, or names a variable that is not set.
            PlacementAnswerRefused: the site's hook answered with a value that could not run the work.
        """
        grant, excluded, _, _ = self._granted(queue)
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
        if excluded:
            args.append("--exclude=" + ",".join(expand(n) for n in excluded))
        if self.log_dir is not None:
            args.append(f"--output={self.log_dir}/%x-%j.out")
        return [*args, "--wrap", shlex.join([*self.command, "--classes", queue])]

    def _try(self, argv: Sequence[str]) -> tuple[str | None, str, str | None]:
        """Run `argv` once; return its output, or None, why it failed and what the command said when it exited.

        Raises:
            UndeclaredClient: `argv[0]` is not one of `CLIENTS`.
        """
        if not argv or argv[0] not in CLIENTS:
            raise UndeclaredClient(
                f"{argv[0] if argv else '(none)'!r} is not one of the Slurm clients {sorted(CLIENTS)}"
            )
        timeout = self.commands.timeout_s
        try:
            done = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=timeout)  # noqa: S603
        except subprocess.TimeoutExpired:
            return None, f"timed out after {timeout} s", None
        except OSError as err:
            return None, f"could not run: {err}", None
        if done.returncode == 0:
            return done.stdout, "", None
        said = (done.stderr or done.stdout).strip()
        return None, f"exited {done.returncode}: {said}", said

    def _run(self, argv: Sequence[str], recover: Callable[[], str | None] | None = None) -> str:
        """Run `argv` until it succeeds, at most `commands.attempts` times, `commands.retry_first_s` apart and doubling.

        Without `recover` the command must be idempotent. With it, each failed try is followed, after the wait, by
        `recover()`, which returns the output the try had when it took effect after all; that ends the tries.

        Raises:
            SlurmError: no try succeeded and `recover` found nothing.
        """
        wait, last, said = self.commands.retry_first_s, "", None
        for attempt in range(self.commands.attempts):
            out, last, said = self._try(argv)
            if out is not None:
                return out
            if recover is None and attempt + 1 == self.commands.attempts:
                break
            self._clock.sleep(wait)
            wait *= 2
            if recover is not None and (found := recover()) is not None:
                return found
        raise SlurmError(f"{shlex.join(argv)} {last}", site_message=said if argv[0] == SBATCH else None)

    def _submit(self, queue: str) -> WorkerHandle:
        """Submit one worker of `queue` under a name of its own; return its pending handle.

        A failed try is followed by a lookup of the name (`squeue --me --name`); a job found is the one submitted.

        Raises:
            SlurmError: no try submitted a job found under its name, or the lookup failed, so the outcome is unknown.
        """
        name = self.job_name(uuid.uuid4().hex[:TOKEN_HEX])

        def lookup() -> str | None:
            out = self._run([SQUEUE, "--me", "--noheader", "--states=all", f"--name={name}", SQUEUE_FORMAT])
            return next((job for job, _, _, job_name in _rows(out) if job_name == name), None)

        args = self.sbatch_args(queue, name)
        try:
            job = self._run(args, recover=lookup).strip().split(";")[0]
        except SlurmError as err:
            if err.site_message is None:
                raise
            _, _, layers, names = self._granted(queue)
            raise SiteRefused(class_of(queue), shlex.join(args), err.site_message, layers, names) from err
        return WorkerHandle(queue, job, pending=True, name=name)

    def live(self) -> list[WorkerHandle]:
        """Return this epoch's live jobs of the run, from one `squeue`; cancel those of an earlier epoch."""
        out = self._run([SQUEUE, "--me", "--noheader", f"--states={LIVE_STATES}", SQUEUE_FORMAT])
        mine, stale = [], []
        for job, state, comment, name in _rows(out):
            if not name.startswith(self.prefix):
                continue
            found = EPOCH_SUFFIX.search(name)
            epoch = int(found[1]) if found else 0
            handle = WorkerHandle(comment, job, pending=state == "PENDING", name=name)
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
            PlacementAnswerRefused: the site's hook answered with a value that could not run the work.
            SlurmError: `sbatch` kept failing; it carries the workers submitted before it (`started`).
            SiteRefused: the scheduler refused the submission; names the `sbatch` line, its reply and the layers.
        """
        self._granted(resource_class)
        cached = self._cached()
        if self.qos_cpu_cap is not None:
            used = sum(self._cpus(h.queue) for h in cached)
            count = min(count, (self.qos_cpu_cap - used) // self._cpus(resource_class))
        if self.log_dir is not None:
            private_dir(self.log_dir)
        handles: list[WorkerHandle] = []
        for _ in range(max(0, count)):
            try:
                handles.append(self._submit(resource_class))
            except SlurmError as err:
                err.started = tuple(handles)
                raise
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
        clock: Clock = SystemClock(),  # noqa: B008 - stateless
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
        self._began: dict[str, tuple[str, float]] = {}
        self._lived: defaultdict[str, float] = defaultdict(float)

    def reset(self) -> None:
        """Forget every failure, every broken class and the worker lifetimes, as a new epoch does."""
        self.broken.clear()
        self._failures.clear()
        self._not_before.clear()
        self._lived.clear()
        now = self._clock.now()
        self._began = {i: (c, now) for i, (c, _) in self._began.items()}

    def worker_seconds(self, resource_class: str) -> float:
        """Return the seconds the workers of `resource_class` have lived since the epoch began, as `live` saw them.

        A worker lives from the first `live` that lists it started (not pending) to the first that does not list it.
        """
        now = self._clock.now()
        return self._lived[resource_class] + sum(now - b for c, b in self._began.values() if c == resource_class)

    def _ready(self, handle: WorkerHandle) -> bool:
        return self.workers_dir is not None and (self.workers_dir / handle.id).exists()

    def _fail(self, resource_class: str, reason: str) -> None:
        self._failures[resource_class] += 1
        n = self._failures[resource_class]
        self._not_before[resource_class] = self._clock.now() + min(self.first_s * 2 ** (n - 1), self.max_s)
        if n > self.budget and resource_class not in self.broken:
            self.broken[resource_class] = BrokenClass(resource_class, n, reason)
            logger.error("workers of %s are broken after %d failed starts: %s", resource_class, n, reason)

    def fail_substrate(self, err: Exception) -> None:
        """Count a failure of the substrate's commands: against the class a `SiteRefused` names, else the substrate.

        A class, or the substrate (`*`), is broken after `budget` failures in a row.
        """
        self._fail(err.resource_class if isinstance(err, SiteRefused) else "*", str(err))

    def substrate_due(self) -> bool:
        """Whether the substrate may be used now: it is not broken, and its latest failure's wait has passed."""
        return "*" not in self.broken and self._clock.now() >= self._not_before.get("*", 0.0)

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
        now = self._clock.now()
        for handle in found:
            if not handle.pending:
                self._began.setdefault(handle.id, (class_of(handle.queue), now))
        for gone in [i for i in self._began if i not in ids]:
            cls, began = self._began.pop(gone)
            self._lived[cls] += now - began
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
        if self.broken_for(resource_class) is not None or self._clock.now() < self._not_before.get(cls, 0.0):
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
        try:
            started = self.substrate.start(resource_class, count)
        except SubstrateError as err:
            self._track(cls, err.started)
            raise
        self._track(cls, started)
        return started

    def _track(self, cls: str, started: Sequence[WorkerHandle]) -> None:
        for handle in started:
            if self.workers_dir is not None:
                (self.workers_dir / handle.id).unlink(missing_ok=True)
            self._tracked[handle.id] = handle
        self.starts[cls] += len(started)

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

    def substrate_wait(self) -> float:
        """Return the seconds until the substrate may be used again; 0 when it may be now or is broken."""
        return max(0.0, self._not_before.get("*", 0.0) - self._clock.now())

    def close(self) -> None:
        """Stop every worker the substrate lists, late duplicates included; when it cannot list, every one tracked."""
        try:
            alive = self.substrate.live()
        except SubstrateError:
            alive = list(self._tracked.values())
        self.stop(alive)


def scale(  # noqa: PLR0913, C901 - the queues and substrate, then keyword-only state; one pass over each queue
    queues: QueueSet,
    substrate: Substrate,
    profile: ExecutionProfile,
    *,
    workers: Callable[[str, int], int],
    surplus_since: dict[str, float] | None = None,
    clock: Clock = SystemClock(),  # noqa: B008 - stateless
    granted: Callable[[str], ResourceClass] | None = None,
) -> None:
    """Start or stop the workers each queue of `queues` needs, from one `live()` listing of `substrate`.

    A queue wants `workers(name, work)` workers, `work` being the jobs of its unresolved items. Fewer alive
    start workers, within the room left in the class (`max_workers` less the workers alive on its queues) and, under a
    QOS cap, the CPUs left over the class's CPUs per worker. More alive stop the surplus, pending workers first; running
    ones are stopped only once the queue has no unresolved item, so none is working, and never on a service class,
    whose workers exit when their consumers stop. With `surplus_since` (the caller
    keeps it between rounds) a surplus is stopped only once it has lasted `2 × pool.poll_s`, so a queue that briefly
    empties between items keeps its workers; without it a surplus is stopped at once. The surplus is timed on `clock`.
    `granted` gives a class as it is placed (`max_workers` is read from it, only for a queue that wants to start
    workers); by default the profile's.
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
        wanted = workers(name, work)
        have = len(alive[name])
        if have > wanted:
            if surplus_since is not None:
                at = clock.now()
                if at - surplus_since.setdefault(name, at) < 2 * profile.pool.poll_s:
                    continue
                del surplus_since[name]
            victims = sorted(alive[name], key=lambda h: not h.pending)
            victims = victims if work == 0 and not is_service_class(name) else [h for h in victims if h.pending]
            substrate.stop(victims[: have - wanted])
            continue
        if surplus_since is not None:
            surplus_since.pop(name, None)
        if wanted == have:
            continue
        limit = (granted(name) if granted is not None else grant).max_workers
        room = limit - sum(len(hs) for other, hs in alive.items() if class_of(other) == class_of(name))
        if cap is not None:
            room = min(room, (cap - used) // cpus(name))
        count = min(wanted - have, room)
        if count > 0:
            started = substrate.start(name, count)
            used += len(started) * cpus(name)
            alive[name].extend(started)


def profile_argument(profile: ExecutionProfile, run_dir: Path) -> str:
    """Return what `senselab execute --profile` takes to load `profile`: its packaged name, else a file written.

    A profile that differs from the packaged one of its name is written to
    `<run dir>/private/profile-<digest prefix>.json`, so two coordinators with different profiles never share a file.
    """
    digest = profile.canonical_digest()
    try:
        if ExecutionProfile.load(profile.name).canonical_digest() == digest:
            return profile.name
    except (FileNotFoundError, ValueError):
        pass
    path = private_dir(Path(run_dir) / PRIVATE_DIR) / f"{PROFILE_PREFIX}{digest[:16]}.json"
    staging = path.with_name(f".{path.name}.{os.getpid()}")
    staging.write_text(profile.model_dump_json(), encoding="utf-8")
    staging.replace(path)
    return str(path)


def substrate_of(
    profile: ExecutionProfile,
    run_dir: Path,
    plan_hash: str,
    *,
    hardware: HardwareInfo | None = None,
) -> Substrate:
    """Return the substrate `profile` names for the run in `run_dir` of plan `plan_hash`.

    `local` is `Local` over the sum of the classes' `max_workers`, its `cuda` classes on the GPUs this process sees
    (`hardware`, probed when None); `slurm` is `Slurm` over the profile's classes, QOS cap and excluded nodes, logging
    under `<run dir>/private/slurm`.
    """
    command = worker_command(run_dir, profile_argument(profile, run_dir))
    spec = profile.substrate
    if spec.kind == "local":
        classes = profile.resource_classes
        return Local(
            sum(c.max_workers for c in classes.values()),
            command=command,
            cuda=[name for name, c in classes.items() if c.device == "cuda"],
            devices=visible_devices(hardware or probe_hardware()),
        )
    if spec.commands is None:
        raise ValueError("a slurm substrate needs its commands settings")
    return Slurm(
        profile.resource_classes,
        spec.qos_cpu_cap,
        command=command,
        plan_hash=plan_hash,
        run_dir=Path(run_dir),
        exclude_nodes=spec.exclude_nodes,
        log_dir=Path(run_dir) / PRIVATE_DIR / "slurm",
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
    "LocalError",
    "SiteRefused",
    "Slurm",
    "SlurmError",
    "StoreUnreachable",
    "Substrate",
    "SubstrateError",
    "WorkerHandle",
    "class_of",
    "Peers",
    "ReadyMark",
    "mark_ready",
    "read_mark",
    "profile_argument",
    "scale",
    "substrate_of",
    "worker_command",
    "worker_id",
]
