"""senselab on pydra's warm pool: its pool tasks, its profile as worker classes and budgets, and `PoolSession`.

A pool job is one senselab job: `HostCall` or `WorkerCall` with the backend's model worker as its provided field and
the `JobDoc` as its input, its checksum the job's key digest. Its one output is the job's stored outputs.
`PoolSession` owns a `WarmPool` and the `Submitter(worker=pool)` its jobs are made under; whoever made it closes it,
so its workers and their models stay resident from one run to the next, and several runs may use it at once.
"""

import atexit
import contextlib
import json
import logging
import os
import re
import secrets
import threading
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydra.compose import python
from pydra.engine.job import Job as PydraJob
from pydra.engine.lease import LeaseSettings
from pydra.engine.resident import Provided
from pydra.engine.submitter import Submitter
from pydra.utils.resources import ComputeResources
from pydra.workers.pool.calibration import DirectoryCalibration
from pydra.workers.pool.classes import WorkerClass
from pydra.workers.pool.faults import FaultBudgets
from pydra.workers.pool.faults import FaultRecord as PoolFault
from pydra.workers.pool.fit import MeasuredFit
from pydra.workers.pool.pool import WarmPool
from pydra.workers.pool.settings import PoolSettings
from pydra.workers.pool.substrate import LocalSubstrate, SlurmSubstrate, Substrate
from senselab_worker.faults import FaultClass, ProtocolError, classification

from senselab_core.runtime.execution.executor import FaultRecord
from senselab_core.runtime.execution.outputs import as_stored, as_wire
from senselab_core.runtime.execution.portable import OUTPUTS, JobDoc, open_store, ran
from senselab_core.runtime.execution.providers import HOST, WORKER, Loader, Model, SenselabHost, SenselabWorker
from senselab_core.runtime.execution.workers import WorkerLoadFailed
from senselab_core.runtime.execution_settings import ResourceClass
from senselab_core.runtime.licenses import LicenseRefused
from senselab_core.runtime.measures import resolve_roots
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.revision import cache_root
from senselab_core.runtime.site_placement import ClassPlacement
from senselab_core.runtime.store import output_identity
from senselab_core.runtime.worker_client import WorkerFatal, WorkerLost

FAULT_MARK = "senselab-fault@1 "
"""The prefix of an `ItemFailed` message: the item's fault record follows as JSON."""
logger = logging.getLogger("senselab")

CALIBRATION_DIR = "pydra-directory-calibration@1"
"""Under a measures root, the directory the pool's `DirectoryCalibration` keeps its samples and unfit records in."""
POOLS_DIR = "2x/pools"
"""Under the cache root, the directory each process's default sessions are opened in (`default_session`)."""
_GRES = re.compile(r"^gpu:(?:(?P<type>[\w.-]+):)?(?P<count>[0-9]+)$")


class ItemFailed(RuntimeError):
    """The backend failed the job's item; `failure` is its record, with its fault class."""

    def __init__(self, failure: FaultRecord) -> None:
        """Hold the failure and carry it in the message."""
        super().__init__(FAULT_MARK + json.dumps(failure.model_dump(mode="json"), sort_keys=True))
        self.failure = failure

    def __reduce__(self) -> tuple[type["ItemFailed"], tuple[FaultRecord]]:
        """Pickle by the record."""
        return (type(self), (self.failure,))


def _call(worker: Model, job: JobDoc) -> dict[str, object]:
    """Send `job`'s one item to its model worker and return its outputs as the store holds them.

    Raises:
        ItemFailed: the backend failed the item.
    """
    started = datetime.now(timezone.utc)
    call = job.job.call
    params = {
        **call.params.model_dump(mode="json"),
        "backend_params": job.placement.backend_params.model_dump(mode="json"),
    }
    values = job.job.values
    [result] = worker.run([{"id": "0", "inputs": dict(values) if job.host else as_wire(values)}], params)
    if "error" in result:
        raise ItemFailed(FaultRecord.of(result["error"]))
    _, outputs = as_stored(call.task, as_wire(result["outputs"]), job.job.anchors)
    ran(job.key.digest(), started, output_identity(outputs))
    return outputs


def _key_digest(task: object, inputs: Mapping[str, Any]) -> str:  # noqa: ARG001 - the key is in the job
    doc = inputs["job"]
    assert isinstance(doc, JobDoc)
    return doc.key.digest()


def _host_call(worker: Provided[object, HOST], job: object) -> object:
    assert isinstance(job, JobDoc)
    return _call(worker, job)


def _worker_call(worker: Provided[object, WORKER], job: object) -> object:
    assert isinstance(job, JobDoc)
    return _call(worker, job)


HostCall = python.define(_host_call, outputs={OUTPUTS: object}, name="HostCall", checksum=_key_digest)
"""A senselab job of a host backend, its model loaded in the pool worker's process (`senselab-host@3`)."""
WorkerCall = python.define(_worker_call, outputs={OUTPUTS: object}, name="WorkerCall", checksum=_key_digest)
"""A senselab job of an isolated or server backend, over its worker process or server (`senselab-worker@3`)."""


def classify(exc: BaseException, phase: str) -> PoolFault:
    """Senselab's fault classifier for the pool: an item's own record, the worker's verdicts, then the fault table.

    Every exception the table does not classify is an infrastructure fault (a gap), never the job's own.
    """
    name = type(exc).__name__
    message = str(exc)[:2000]
    if isinstance(exc, ItemFailed):
        return _of(exc.failure.fault_class, exc.failure.gap, name, exc.failure.message, phase)
    if isinstance(exc, LicenseRefused):
        return PoolFault("policy", name, message, phase)
    if isinstance(exc, WorkerLoadFailed):
        return _of(*exc.carried_class, name, message, "load")
    if isinstance(exc, WorkerFatal) and exc.from_load:
        return _of(*exc.carried_class, name, message, "load")
    if isinstance(exc, (WorkerLost, ProtocolError, WorkerFatal)):
        return PoolFault("infrastructure", name, message, phase, reason="crash")
    return _of(*classification(exc), name, message, phase)


def _of(fault: FaultClass, gap: bool, name: str, message: str, phase: str) -> PoolFault:
    if fault is FaultClass.RESOURCE:
        return PoolFault("resource", name, message, phase, reason="oom")
    if fault is FaultClass.CONTENT and phase == "call":
        return PoolFault("content", name, message, phase)
    return PoolFault("infrastructure", name, message, phase, gap=gap)


def _gpus(resource: ResourceClass) -> tuple[int, str | None]:
    """Return a class's GPUs and GPU type: from its `gres` when it names one, its `gpu_type` otherwise; an `mps` one.

    Raises:
        ValueError: `gres` is not `gpu:[<type>:]<n>`, or names another type than `gpu_type`.
    """
    if resource.device == "mps":
        return 1, None
    if resource.device != "cuda":
        return 0, None
    if resource.gres is None:
        return 1, resource.gpu_type
    found = _GRES.match(resource.gres)
    if found is None:
        raise ValueError(f"gres {resource.gres!r} is not gpu:[<type>:]<count>")
    kind = found["type"] or resource.gpu_type
    if resource.gpu_type is not None and kind != resource.gpu_type:
        raise ValueError(f"gres {resource.gres!r} names another GPU type than gpu_type {resource.gpu_type!r}")
    return int(found["count"]), kind


def worker_classes(
    profile: ExecutionProfile, placements: Mapping[str, ClassPlacement] | None = None
) -> list[WorkerClass]:
    """Return the profile's resource classes as pydra worker classes: devices, typed GPUs, memory, time, partition.

    A class in `placements` takes its placed partition, time, worker cap and excluded nodes (`ClassPlacement.grant`);
    any other its profile's values and the profile's excluded nodes. A class names its partition and excluded nodes
    as substrate arguments; nothing names a QOS. Each is tagged with its device, so a job is placed on classes of its
    own device alone.
    """
    found = []
    placed = placements or {}
    for name, given in profile.resource_classes.items():
        resource = placed[name].grant(given) if name in placed else given
        exclude = placed[name].exclude if name in placed else profile.substrate.exclude_nodes
        gpus, gpu_type = _gpus(resource)
        args = {"partition": resource.partition} if resource.partition else {}
        if exclude:
            args["exclude"] = ",".join(exclude)
        found.append(
            WorkerClass(
                name=name,
                device=resource.device,
                gpus=gpus,
                gpu_type=gpu_type,
                cpus=resource.cpus,
                mem=resource.mem_bytes,
                time=None if resource.time_s is None else int(resource.time_s),
                max_workers=resource.max_workers,
                substrate_args=args,
                tags=frozenset({resource.device}),
                quantum_jobs=profile.pool.quantum_items,
                quantum_s=profile.pool.quantum_s,
            )
        )
    return found


def request(profile: ExecutionProfile, resource_class: str) -> ComputeResources:
    """Return what a job placed on `resource_class` asks: its device, GPUs and type, and its memory; never less."""
    resource = profile.resource_classes[resource_class]
    gpus, gpu_type = _gpus(resource)
    mem = None if resource.mem_bytes is None else f"{max(1, resource.mem_bytes // 1024)}K"
    return ComputeResources(gpus=gpus or None, gpu_type=gpu_type if gpus else None, mem=mem, tags=(resource.device,))


def budgets(profile: ExecutionProfile) -> FaultBudgets:
    """Return the profile's fault budgets as the pool's: a failed start counts against the infrastructure budget."""
    faults = profile.faults
    infrastructure = max(1, faults.infrastructure_replacements)
    return FaultBudgets(
        content_attempts=faults.content_attempts,
        resource_moves=faults.resource_replacements,
        infrastructure_replacements=infrastructure,
        crashes_charged=faults.crashes_charged,
        start_failures=infrastructure,
    )


def settings(profile: ExecutionProfile) -> PoolSettings:
    """Return the profile's pool settings: its lease, polls, wall target and unmeasured workers; no re-check."""
    lease = profile.lease
    timeout = max(lease.wait_s, lease.stale_after_s + lease.attribute_delay_s)
    return PoolSettings(
        lease=LeaseSettings(
            stale_after=lease.stale_after_s,
            renew_fraction=lease.renew_fraction,
            attribute_delay=lease.attribute_delay_s,
            timeout=timeout,
        ),
        poll_s=profile.pool.poll_s,
        idle_polls=profile.pool.idle_polls,
        call_timeout=profile.timeouts.request_s,
        target_wall_s=profile.sizing.target_wall_s,
        unmeasured_workers=profile.calibration.unmeasured_workers,
        recheck_fraction=0.0,
    )


def substrate_of(profile: ExecutionProfile) -> Substrate:
    """Return the substrate that starts the profile's workers: processes on this host, or one Slurm job each."""
    if profile.substrate.kind == "local":
        return LocalSubstrate()
    commands = profile.substrate.commands
    timeouts = {} if commands is None else {"submit_timeout": commands.timeout_s, "squeue_timeout": commands.timeout_s}
    return SlurmSubstrate(per_user_cpu_cap=profile.substrate.qos_cpu_cap, **timeouts)


def calibration_root(profile: ExecutionProfile) -> Path | None:
    """Return where every session of the profile keeps the pool's calibration: under its first writable measures root.

    The roots are the profile's `calibration.measures_roots` (`resolve_roots`), so a later session, over any pool
    root, reads the samples and unfit records an earlier one wrote. None (logged) when no root is writable.
    """
    for root in resolve_roots(profile.calibration.measures_roots, cache_root()):
        found = root / CALIBRATION_DIR
        try:
            found.mkdir(parents=True, exist_ok=True)
        except OSError as err:
            logger.info("measures root %s takes no pool calibration: %s", root, err)
            continue
        if os.access(found, os.W_OK):
            return found
    logger.warning("no measures root is writable; the pool's calibration is kept under its session root")
    return None


class PoolSession:
    """A warm pool whoever makes it closes: its `WarmPool`, its store and the submitter its jobs are made under.

    `root` holds the pool (`<root>/pool`) and the jobs' work directories (`<root>/private/work`), on storage every
    node sees; `store` is the result store's root. The pool's calibration is shared by every session of the profile
    (`calibration_root`). The pool's workers and their models stay resident until `close`, whatever runs use it, one
    after another or at once.
    """

    def __init__(  # noqa: PLR0913 - the profile and two roots, then keyword-only pool parts
        self,
        profile: ExecutionProfile,
        root: Path,
        store: Path,
        *,
        substrate: Substrate | None = None,
        pool_settings: PoolSettings | None = None,
        placements: Mapping[str, ClassPlacement] | None = None,
    ) -> None:
        """Open the pool at `root` over the store at `store`, its workers started by `substrate` (the profile's).

        Its worker classes are the profile's, each class in `placements` placed with those values (`worker_classes`).
        """
        self.profile = profile
        self.classes = worker_classes(profile, placements)
        self.root = Path(root).absolute()
        work = self.root / "private" / "work"
        work.mkdir(parents=True, exist_ok=True, mode=0o700)
        (self.root / "private").chmod(0o700)
        self.store = open_store(Path(store), work)
        calibration = DirectoryCalibration(calibration_root(profile) or self.root / "pool" / "calibration")
        self.calibration = calibration
        self.providers = (SenselabWorker(), SenselabHost())
        self.pool = WarmPool(
            self.root / "pool",
            self.classes,
            substrate if substrate is not None else substrate_of(profile),
            store=self.store,
            fit=MeasuredFit(calibration, headroom=profile.sizing.memory_headroom),
            calibration=calibration,
            faults=budgets(profile),
            classify=classify,
            settings=pool_settings if pool_settings is not None else settings(profile),
        )
        self.submitter = Submitter(worker=self.pool, providers=list(self.providers), telemetry=False)

    @property
    def epoch(self) -> int:
        """The pool's coordinator epoch."""
        assert self.pool.epoch is not None
        return int(self.pool.epoch)

    def job(self, doc: JobDoc, loader: Loader, name: str) -> PydraJob:
        """Return the pool job of `doc`, its model worker loaded by `loader`, asking for its plan class's resources."""
        task = (WorkerCall if loader.provider == WORKER else HostCall)(worker=loader.spec, job=doc)
        resources = request(self.profile, doc.placement.resource_class)
        return PydraJob(task, submitter=self.submitter, name=name, resources=resources)

    def close(self) -> None:
        """Close the pool, stopping the workers it started."""
        with contextlib.suppress(Exception):
            self.submitter.close()
        self.pool.close()

    def __enter__(self) -> "PoolSession":
        """Return the session, closed when the block ends."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close the session."""
        self.close()


_DEFAULTS: dict[tuple[str, str, str], PoolSession] = {}
_DEFAULTS_LOCK = threading.Lock()


def default_session(
    profile: ExecutionProfile, store: Path, placements: Mapping[str, ClassPlacement] | None = None
) -> PoolSession:
    """Return this process's session for `profile` over `store`, its classes placed by `placements`, opened once.

    Its root is a directory of its own under `<cache root>/2x/pools`; it is closed at exit. Every run started without
    a session, and every call, uses the one of its profile, store and worker classes, so a model stays resident from
    one to the next for the profile's idle window.
    """
    key = (profile.canonical_digest(), str(Path(store).absolute()), repr(worker_classes(profile, placements)))
    with _DEFAULTS_LOCK:
        found = _DEFAULTS.get(key)
        if found is None:
            if not _DEFAULTS:
                atexit.unregister(close_default_sessions)
                atexit.register(close_default_sessions)
            root = cache_root() / POOLS_DIR / f"{os.getpid()}-{secrets.token_hex(4)}"
            found = _DEFAULTS[key] = PoolSession(profile, root, Path(store), placements=placements)
    return found


def close_default_sessions() -> None:
    """Close every default session this process opened (`default_session`), stopping their workers."""
    with _DEFAULTS_LOCK:
        opened = list(_DEFAULTS.values())
        _DEFAULTS.clear()
    for found in opened:
        try:
            found.close()
        except Exception as err:  # noqa: BLE001 - one failed close must not keep the others open
            logger.warning("closing the default pool session at %s failed: %s", found.root, err)


@contextlib.contextmanager
def session(profile: ExecutionProfile, root: Path, store: Path) -> Iterator[PoolSession]:
    """Open a `PoolSession` for the block and close it after."""
    opened = PoolSession(profile, root, store)
    try:
        yield opened
    finally:
        opened.close()


def item_failure(text: str) -> FaultRecord | None:
    """Return the fault record an `ItemFailed` carried in `text` (an error message or traceback), or None."""
    at = text.rfind(FAULT_MARK)
    if at < 0:
        return None
    line = text[at + len(FAULT_MARK) :].splitlines()[0]
    try:
        return FaultRecord.model_validate(json.loads(line))
    except ValueError:
        return None


def coordinator_live(root: Path) -> bool:
    """Whether the newest coordinator of the pool at `root` is live: not ended, its lease held and not stale."""
    from pydra.engine.lease import LeaseTimeout  # noqa: PLC0415 - read only here
    from pydra.workers.pool.pool import _open_existing  # noqa: PLC0415 - the pool's own reader of its header

    if not (Path(root) / "pool.json").is_file():
        return False
    with _open_existing(root) as pool:
        epochs = pool.epochs()
        if not epochs or (Path(root) / "coordinator" / str(epochs[-1]) / "ended").exists():
            return False
        lease = pool.coordinator_lease(epochs[-1], beat="external")
        try:
            wait = lease.try_take()
        except LeaseTimeout:
            return True
        if wait is None:
            lease.__exit__(None, None, None)
            return False
        return True
