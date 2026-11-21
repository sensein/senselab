"""A senselab pool session over an in-process pool worker on a thread: the warm pool tests' scaffold."""

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

PROFILE = ExecutionProfile.load(
    "laptop",
    {
        "devices": ["cpu"],
        "resource_classes": {"cpu": {"device": "cpu", "mem_bytes": 2**30, "max_workers": 1}},
        "pool": {"poll_s": 0.05, "first_poll_s": 0.01, "quantum_items": 8, "quantum_s": 60, "idle_polls": 40},
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
        """Wait for every worker to exit."""
        for thread in self.threads:
            thread.join(timeout=120)


@contextlib.contextmanager
def served(
    tmp_path: Path, profile: ExecutionProfile = PROFILE, workers: int = 1
) -> Iterator[tuple[PoolSession, Served]]:
    """Open a session over `tmp_path` whose pool workers run here, `workers` of the first class, closed after."""
    release_process_caches()
    session = PoolSession(profile, tmp_path / "session", tmp_path / "store", substrate=ExternalSubstrate())
    pool = Served(session)
    for _ in range(workers):
        pool.serve()
    try:
        yield session, pool
    finally:
        session.close()
        pool.close()
        release_process_caches()
