"""Pydra pool helpers for tests and benchmarks.

A pool served on threads (`served`, `Served`, `POOL_PROFILE`), pydra's batch probe and memory bound over measured
values (`probe_size`, `memory_bound`), and a pydra cache migration (`migrate_pickle_cache`).
"""

import contextlib
import statistics
import threading
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydra.compose import python
from pydra.engine.portable.migrate import migrate
from pydra.engine.portable.store import PortableStore
from pydra.engine.store import PickleStore
from pydra.engine.submitter import Submitter
from pydra.utils.resources import ComputeResources
from pydra.workers.pool.calibration import MemoryFit, TimeRecord
from pydra.workers.pool.classes import WorkerClass
from pydra.workers.pool.fit import MeasuredFit, probe_batch
from pydra.workers.pool.placement import FitRequest
from pydra.workers.pool.substrate import ExternalSubstrate
from pydra.workers.pool.worker import Worker

from senselab_core.runtime.execution.warm import PoolSession, classify, settings, worker_classes
from senselab_core.runtime.execution.workers import release_process_caches
from senselab_core.runtime.profiles import ExecutionProfile

POOL_PROFILE = ExecutionProfile.load(
    "laptop",
    {
        "devices": ["cpu"],
        "resource_classes": {"cpu": {"device": "cpu", "mem_bytes": 2**30, "max_workers": 1}},
        "pool": {"poll_s": 0.05, "quantum_items": 8, "quantum_s": 60, "idle_polls": 40},
        "lease": {"stale_after_s": 4.0, "renew_fraction": 0.2, "attribute_delay_s": 0, "wait_s": 10.0},
    },
)
"""A one-class CPU profile with fast polls and a short lease."""


class Served:
    """Pool workers of a session run on threads of this process, started by `serve`, stopped by `close`.

    Each class `serve` was called with is kept: while a queue of the class has waiting tickets and no worker of the
    class is live, a watcher thread starts another.
    """

    def __init__(self, session: PoolSession) -> None:
        """Serve `session`'s pool."""
        self.session = session
        self.threads: list[threading.Thread] = []
        self.workers: list[Worker] = []
        self.codes: list[int] = []
        self.kept: dict[str, WorkerClass] = {}
        self._live: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._watcher: threading.Thread | None = None

    def serve(self, name: str | None = None) -> None:
        """Start one worker of class `name` (the profile's first) on a thread and keep the class served."""
        classes = {c.name: c for c in worker_classes(self.session.profile)}
        cls: WorkerClass = classes[name] if name is not None else next(iter(classes.values()))
        with self._lock:
            self.kept[cls.name] = cls
            self._start(cls)
            if self._watcher is None:
                self._watcher = threading.Thread(target=self._watch, name="pool-watcher", daemon=True)
                self._watcher.start()

    def _start(self, cls: WorkerClass) -> None:
        worker = Worker(
            self.session.pool.root,
            cls,
            classify=classify,
            settings=settings(self.session.profile),
            device_memory=lambda: None,
        )
        thread = threading.Thread(target=lambda: self.codes.append(worker.run()), name=f"pool-worker-{cls.name}")
        thread.start()
        self.workers.append(worker)
        self.threads.append(thread)
        self._live[cls.name] = thread

    def waiting(self, name: str) -> int:
        """Return an upper bound on the tickets of class `name`'s queues without a final outcome for every job."""
        root = self.session.pool.pool
        return sum(root.depth(q) for q in root.queues() if q == name or q.startswith(name + "#"))

    def _watch(self) -> None:
        poll_s = self.session.profile.pool.poll_s
        while not self._stop.wait(poll_s):
            with self._lock:
                if self._stop.is_set():
                    return
                for name, cls in self.kept.items():
                    if not self._live[name].is_alive() and self.waiting(name) > 0:
                        self._start(cls)

    def stop_watching(self) -> None:
        """Stop starting workers; return once the watcher has exited."""
        with self._lock:
            self._stop.set()
        if self._watcher is not None:
            self._watcher.join()

    def close(self) -> None:
        """Stop the watcher, then every worker: each exits once its current claim ends."""
        self.stop_watching()
        for worker in self.workers:
            worker.stop.set()
        profile = self.session.profile
        bound = profile.pool.poll_s + profile.lease.stale_after_s
        for thread in self.threads:
            thread.join(bound)


@contextlib.contextmanager
def served(
    tmp_path: Path,
    profile: ExecutionProfile = POOL_PROFILE,
    workers: int = 1,
    store: Path | None = None,
    *,
    here: bool = True,
) -> Iterator[tuple[PoolSession, Served]]:
    """Open a session over `tmp_path` whose pool workers run here, `workers` of the first class, closed after.

    Its results are in the store at `store` (`<tmp_path>/store` when None). `here` false makes the session one whose
    workers its coordinator treats as on other hosts, as a Slurm pool's are. The process model caches are emptied
    before and after, so a test's loads are its own.
    """
    release_process_caches()
    root = store if store is not None else tmp_path / "store"
    session = PoolSession(profile, tmp_path / "session", root, substrate=ExternalSubstrate(), here=here)
    pool = Served(session)
    for _ in range(workers):
        pool.serve()
    try:
        yield session, pool
    finally:
        pool.stop_watching()
        session.close()
        pool.close()
        release_process_caches()


def probe_size(per_unit_s: Mapping[int, Sequence[float]], bound: int, samples: int, gain: float) -> int:
    """Return pydra's `probe_batch` size from seconds per unit measured at each batch size (median per size)."""
    times = [TimeRecord(b, statistics.median(v), len(v)) for b, v in per_unit_s.items() if v]
    return int(probe_batch(times, bound, samples, gain))


@dataclass(frozen=True)
class _OneFit:
    """A pydra `Calibration` holding one memory fit."""

    fit: MemoryFit

    def memory(self, key: str, hardware: object, task: str | None = None) -> MemoryFit:  # noqa: ARG002
        """Return the fit."""
        return self.fit

    def unfit(self, key: str) -> list[object]:  # noqa: ARG002
        """Return no unfit record."""
        return []

    def times(self, key: str, hardware: object, task: str | None = None) -> list[TimeRecord]:  # noqa: ARG002
        """Return no time record."""
        return []


@dataclass(frozen=True)
class DeviceFit:
    """Device memory measured for one residency: resident bytes, and the peak above them as `m0 + m1 × units`."""

    resident: int
    m0: float
    m1: float | None
    n: int


def memory_bound(fit: DeviceFit, device_bytes: int, profile: ExecutionProfile, longest: float) -> int:
    """Return pydra's `MeasuredFit.memory_bound` of `fit` on a one-GPU class of `device_bytes` under `profile`.

    The class's `quantum_jobs` is the profile's `quantum_items` and the headroom its `memory_headroom`; `longest` is
    the longest job's units.
    """
    cls = WorkerClass(
        name="measured", device="cuda", gpus=1, gpu_mem=device_bytes, quantum_jobs=profile.pool.quantum_items
    )
    request = FitRequest(residency=(), resources=ComputeResources(), units=longest, key="measured")
    rule = MeasuredFit(_OneFit(MemoryFit(fit.resident, fit.m0, fit.m1, fit.n)), headroom=profile.sizing.memory_headroom)
    return int(rule.memory_bound(request, cls, [longest]))


def _double(x: int) -> int:
    return 2 * x


def migrate_pickle_cache(cache: Path, store: PortableStore) -> list[tuple[str, str]]:
    """Run one task into a pydra `PickleStore` at `cache`, migrate the cache into `store`, return each slot's status."""
    with Submitter(worker="debug", store=PickleStore(cache), telemetry=False) as submitter:
        submitter(python.define(_double)(x=3))
    return [(s.checksum, s.status) for s in migrate(cache, store, trust_source=True).slots]
