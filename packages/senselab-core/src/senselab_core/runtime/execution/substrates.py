"""Substrates: what starts the pool workers that serve a run's queues, and `scale`, which sizes them from queue depth.

A `Substrate` starts workers for a queue, counts those alive and stops given ones. `Local` runs each worker as a
`senselab execute` process under the supervisor in force; `Slurm` submits one `sbatch` job per worker, with the
class's partition, gres, CPUs, memory and time, named by plan hash and attempt id, and never on an excluded node.
A queue is named by its resource class, with `#<model identity>` for a resident model; every worker serves one queue.
`scale` is called by the coordinator each round: per queue, the workers wanted are the queue's depth over
`pool.quantum_items`, within the class's `max_workers` and the QOS CPU cap over the class's CPUs per worker. A
surplus worker is not stopped: a worker exits once its queue has stayed empty.
"""

import math
import os
import shlex
import subprocess
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from senselab_core.runtime.execution.queue import RUN_QUEUES, QueueSet
from senselab_core.runtime.execution.supervisor import current
from senselab_core.runtime.execution_settings import ResourceClass
from senselab_core.runtime.profiles import ExecutionProfile, expand

SBATCH, SQUEUE, SCANCEL = "sbatch", "squeue", "scancel"
"""The Slurm commands, looked up on `PATH`."""
LIVE_STATES = "PENDING,CONFIGURING,RUNNING,COMPLETING"
"""The job states `Slurm.running` counts."""
STOP_GRACE_S = 30.0
"""Seconds a local worker is given after SIGTERM before it is killed."""
PROFILE_FILE = "profile.json"
"""A run's own profile, written where a worker cannot load it by name."""


class SlurmError(RuntimeError):
    """A Slurm command failed; the message gives the command and what it printed."""


@dataclass(frozen=True)
class WorkerHandle:
    """A started worker: the queue it serves and its id (a process id, or a Slurm job id)."""

    queue: str
    id: str


class Substrate(Protocol):
    """What starts and stops the pool workers serving a run's queues."""

    def start(self, resource_class: str, count: int) -> list[WorkerHandle]:
        """Start up to `count` workers of queue `resource_class`; return those started."""

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


class ExternalWorkers:
    """A `Substrate` that starts nothing: workers are started on their own (`senselab execute --pool`)."""

    def start(self, resource_class: str, count: int) -> list[WorkerHandle]:  # noqa: ARG002
        """Start nothing."""
        return []

    def running(self, resource_class: str) -> int:  # noqa: ARG002
        """Return 0."""
        return 0

    def stop(self, handles: Sequence[WorkerHandle]) -> None:
        """Stop nothing."""


class Local:
    """A `Substrate` of local processes: each worker is `command` with `--classes <queue>`, in its own group.

    At most `max_workers` are alive at once, across queues. Processes are launched through the supervisor in force.
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
        self._procs: dict[str, tuple[str, subprocess.Popen[bytes]]] = {}
        self._lock = threading.Lock()

    def _alive(self) -> dict[str, tuple[str, "subprocess.Popen[bytes]"]]:
        for key, (_, proc) in list(self._procs.items()):
            if proc.poll() is not None:
                current().released(proc.pid)
                del self._procs[key]
        return self._procs

    def start(self, resource_class: str, count: int) -> list[WorkerHandle]:
        """Start workers of `resource_class`, as many as `count` and the room under `max_workers` allow."""
        with self._lock:
            room = self.max_workers - len(self._alive())
            handles = []
            for _ in range(max(0, min(count, room))):
                proc = current().popen([*self.command, "--classes", resource_class])
                self._procs[str(proc.pid)] = (resource_class, proc)
                handles.append(WorkerHandle(resource_class, str(proc.pid)))
            return handles

    def running(self, resource_class: str) -> int:
        """Return the workers of `resource_class` alive."""
        with self._lock:
            return sum(1 for queue, _ in self._alive().values() if queue == resource_class)

    def stop(self, handles: Sequence[WorkerHandle]) -> None:
        """SIGTERM each worker of `handles`, which commits what finished; kill one that outlasts `STOP_GRACE_S`."""
        with self._lock:
            found = [self._procs.pop(h.id) for h in handles if h.id in self._procs]
        for _, proc in found:
            if proc.poll() is None:
                proc.terminate()
        for _, proc in found:
            try:
                proc.wait(STOP_GRACE_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            current().released(proc.pid)


def _minutes(seconds: float) -> int:
    return max(1, math.ceil(seconds / 60))


class Slurm:
    """A `Substrate` of Slurm jobs: one `sbatch` job per worker, within a QOS CPU cap, never on an excluded node.

    A job is named `senselab-<plan hash prefix>-<attempt>` and carries its queue as its comment, so the workers of a
    queue are those `squeue` lists under that name. The class's `partition`, `gres`, `cpus`, `mem_bytes` and `time_s`
    become the job's `--partition`, `--gres`, `--cpus-per-task`, `--mem` and `--time`; `${VAR}` in a partition, a gres
    or an excluded node is expanded from the environment where the job is submitted.

    Attributes:
        attempt: The attempt id the jobs are named by; the pool sets it to its attempt's.
    """

    def __init__(  # noqa: PLR0913 - the classes and cap, then keyword-only submission settings
        self,
        classes: Mapping[str, ResourceClass],
        qos_cpu_cap: int | None,
        *,
        command: Sequence[str],
        plan_hash: str,
        attempt: str = "a0",
        exclude_nodes: Sequence[str] = (),
        log_dir: Path | None = None,
    ) -> None:
        """Submit workers of `classes`, each `command` plus `--classes <queue>`, at most `qos_cpu_cap` CPUs alive."""
        self.classes = dict(classes)
        self.qos_cpu_cap = qos_cpu_cap
        self.command = list(command)
        self.plan_hash = plan_hash
        self.attempt = attempt
        self.exclude_nodes = tuple(exclude_nodes)
        self.log_dir = log_dir

    @property
    def job_name(self) -> str:
        """The name of this plan's and attempt's jobs."""
        return f"senselab-{self.plan_hash[:12]}-{self.attempt}"

    def _class(self, queue: str) -> ResourceClass:
        try:
            return self.classes[class_of(queue)]
        except KeyError:
            raise ValueError(
                f"queue {queue!r} is of no resource class of this substrate: {sorted(self.classes)}"
            ) from None

    def sbatch_args(self, queue: str) -> list[str]:
        """Return the `sbatch` command line of one worker of `queue`.

        Raises:
            ValueError: the queue's class is unknown, or names a variable that is not set.
        """
        grant = self._class(queue)
        args = [SBATCH, "--parsable", f"--job-name={self.job_name}", f"--comment={queue}"]
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

    @staticmethod
    def _run(argv: Sequence[str]) -> str:
        done = subprocess.run(argv, capture_output=True, text=True, check=False)  # noqa: S603
        if done.returncode != 0:
            raise SlurmError(f"{shlex.join(argv)} exited {done.returncode}: {(done.stderr or done.stdout).strip()}")
        return done.stdout

    def _queued(self) -> list[tuple[str, str]]:
        """Return the live jobs of this plan and attempt as (job id, queue)."""
        out = self._run([SQUEUE, "--noheader", f"--name={self.job_name}", f"--states={LIVE_STATES}", "--format=%i|%k"])
        pairs = (line.split("|", 1) for line in out.splitlines() if "|" in line)
        return [(job.strip(), queue.strip()) for job, queue in pairs]

    def running(self, resource_class: str) -> int:
        """Return the live jobs of queue `resource_class`."""
        return sum(1 for _, queue in self._queued() if queue == resource_class)

    def start(self, resource_class: str, count: int) -> list[WorkerHandle]:
        """Submit workers of `resource_class`, as many as `count` and the room under `qos_cpu_cap` allow.

        Raises:
            ValueError: the queue's class is unknown, or names a variable that is not set.
            SlurmError: `sbatch` failed.
        """
        self._class(resource_class)
        if self.qos_cpu_cap is not None:
            used = sum(self._cpus(queue) for _, queue in self._queued())
            count = min(count, (self.qos_cpu_cap - used) // self._cpus(resource_class))
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        handles = []
        for _ in range(max(0, count)):
            job = self._run(self.sbatch_args(resource_class)).strip().split(";")[0]
            handles.append(WorkerHandle(resource_class, job))
        return handles

    def _cpus(self, queue: str) -> int:
        return self._class(queue).cpus or 1

    def stop(self, handles: Sequence[WorkerHandle]) -> None:
        """Cancel the jobs of `handles`."""
        if handles:
            self._run([SCANCEL, *(h.id for h in handles)])


class Fleet:
    """A `Substrate` that records the workers its substrate starts, so `close` can stop them all."""

    def __init__(self, substrate: Substrate) -> None:
        """Wrap `substrate`."""
        self.substrate = substrate
        self._handles: list[WorkerHandle] = []

    def start(self, resource_class: str, count: int) -> list[WorkerHandle]:
        """Start through the substrate and record the workers."""
        started = self.substrate.start(resource_class, count)
        self._handles.extend(started)
        return started

    def running(self, resource_class: str) -> int:
        """Return the substrate's count."""
        return self.substrate.running(resource_class)

    def stop(self, handles: Sequence[WorkerHandle]) -> None:
        """Stop `handles` through the substrate and forget them."""
        self.substrate.stop(handles)
        self._handles = [h for h in self._handles if h not in handles]

    def close(self) -> None:
        """Stop every worker started through this fleet."""
        self.stop(list(self._handles))


def scale(queues: QueueSet, substrate: Substrate, profile: ExecutionProfile) -> None:
    """Start the workers each queue of `queues` needs, within its class's `max_workers` and the QOS CPU cap.

    A queue wants `ceil(depth / pool.quantum_items)` workers; the room left in its class is `max_workers` less the
    workers alive on the class's queues, and under a QOS cap the CPUs left over the class's CPUs per worker.
    """
    names = queues.names()
    alive = {name: substrate.running(name) for name in names}
    cap = profile.substrate.qos_cpu_cap

    def cpus(name: str) -> int:
        grant = profile.resource_classes.get(class_of(name))
        return grant.cpus or 1 if grant is not None else 1

    used = sum(n * cpus(name) for name, n in alive.items())
    for name in sorted(names):
        grant = profile.resource_classes.get(class_of(name))
        if grant is None:
            continue
        depth = queues.queue(name).depth()
        wanted = math.ceil(depth / profile.pool.quantum_items) - alive[name]
        room = grant.max_workers - sum(n for other, n in alive.items() if class_of(other) == class_of(name))
        if cap is not None:
            room = min(room, (cap - used) // cpus(name))
        count = min(wanted, room)
        if count > 0:
            started = substrate.start(name, count)
            alive[name] += len(started)
            used += len(started) * cpus(name)


def profile_argument(profile: ExecutionProfile, run_dir: Path) -> str:
    """Return what `senselab execute --profile` takes to load `profile`: its packaged name, else a file written.

    A profile that differs from the packaged one of its name is written to `<run dir>/profile.json`.
    """
    try:
        if ExecutionProfile.load(profile.name).canonical_digest() == profile.canonical_digest():
            return profile.name
    except (FileNotFoundError, ValueError):
        pass
    path = Path(run_dir) / PROFILE_FILE
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
    return Slurm(
        profile.resource_classes,
        spec.qos_cpu_cap,
        command=command,
        plan_hash=plan_hash,
        exclude_nodes=spec.exclude_nodes,
        log_dir=Path(run_dir) / "slurm",
    )


__all__ = [
    "PROFILE_FILE",
    "ExternalWorkers",
    "Fleet",
    "Local",
    "Slurm",
    "SlurmError",
    "Substrate",
    "WorkerHandle",
    "class_of",
    "profile_argument",
    "scale",
    "substrate_of",
    "worker_command",
]
