"""A pool session whose workers run on threads of this process, for tests: `served`, `Served` and `POOL_PROFILE`."""

import contextlib
import threading
from collections.abc import Iterator
from pathlib import Path

from pydra.workers.pool.classes import WorkerClass
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
    """Pool workers of a session run on threads of this process, started by `serve`, joined by `close`."""

    def __init__(self, session: PoolSession) -> None:
        """Serve `session`'s pool."""
        self.session = session
        self.threads: list[threading.Thread] = []
        self.codes: list[int] = []

    def serve(self, name: str | None = None) -> None:
        """Start one worker of class `name` (the profile's first) on a thread; it exits once idle."""
        classes = {c.name: c for c in worker_classes(self.session.profile)}
        cls: WorkerClass = classes[name] if name is not None else next(iter(classes.values()))
        worker = Worker(
            self.session.pool.root,
            cls,
            classify=classify,
            settings=settings(self.session.profile),
            device_memory=lambda: None,
        )
        thread = threading.Thread(target=lambda: self.codes.append(worker.run()), name=f"pool-worker-{cls.name}")
        thread.start()
        self.threads.append(thread)

    def close(self) -> None:
        """Wait for every worker to exit: once idle, within its idle polls and one lease's staleness."""
        profile = self.session.profile
        bound = profile.pool.idle_polls * profile.pool.poll_s + profile.lease.stale_after_s
        for thread in self.threads:
            thread.join(bound)


@contextlib.contextmanager
def served(
    tmp_path: Path, profile: ExecutionProfile = POOL_PROFILE, workers: int = 1, store: Path | None = None
) -> Iterator[tuple[PoolSession, Served]]:
    """Open a session over `tmp_path` whose pool workers run here, `workers` of the first class, closed after.

    Its results are in the store at `store` (`<tmp_path>/store` when None). The process model caches are emptied
    before and after, so a test's loads are its own.
    """
    release_process_caches()
    root = store if store is not None else tmp_path / "store"
    session = PoolSession(profile, tmp_path / "session", root, substrate=ExternalSubstrate())
    pool = Served(session)
    for _ in range(workers):
        pool.serve()
    try:
        yield session, pool
    finally:
        session.close()
        pool.close()
        release_process_caches()
