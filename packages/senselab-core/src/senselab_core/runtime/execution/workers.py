"""The workers a process runs jobs on: pooled per backend, env and model, bounded, and the servers they run over."""

import collections
import contextlib
import importlib
import threading
from collections.abc import Callable, Iterator
from typing import Any, Protocol

from senselab_worker.canonical import jcs_dumps
from senselab_worker.model_cache import ModelCache, ModelKey

from senselab_core.kernel._base import value_digest
from senselab_core.kernel.declare import BackendSpec
from senselab_core.kernel.identity import ResolvedModel
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.host_worker import HostWorker
from senselab_core.runtime.network import network_variables
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.vllm_server import (
    Attestation,
    AttestationMismatch,
    StaleAttestation,
    VllmServer,
    expected_attestation,
)
from senselab_core.runtime.worker_client import WorkerClient

PRECISION = "default"
"""The precision a job records: the backend's own; no backend declares another yet."""

Worker = HostWorker | WorkerClient
"""A worker a job is sent to: in this process, or a worker process."""


class Server(Protocol):
    """A running model server: what it attests, how a message names it, its live context fields, and `stop`."""

    label: str

    @property
    def attestation(self) -> Attestation | None:
        """What the running server attests; None once stopped."""

    def context_fields(self) -> dict[str, Any]:
        """Return the context fields of the running server; raises `StaleAttestation` once it changed."""

    def stop(self) -> None:
        """Stop the server."""


ServerStart = Callable[[BackendSpec, EnvHandle, ResolvedModel, ExecutionProfile], Server]
"""Start the server of a backend, env and model under a profile and return it running."""


def start_vllm(backend: BackendSpec, env: EnvHandle, model: ResolvedModel, profile: ExecutionProfile) -> Server:
    """Start the `VllmServer` of `backend`'s server settings over `env` and `model`, under `profile`'s env roots.

    Raises:
        VllmStartError, LeaseTimeout, LeaseUnavailable: the server did not start.
    """
    assert backend.server is not None
    root = profile.env_root("server")
    server = VllmServer(
        env,
        model,
        backend.server.settings,
        start_lock_dir=root / "locks",
        startup_timeout_s=backend.server.startup_timeout_s,
        start_margin_s=profile.envs.server_start_margin_s,
        lease=profile.lease,
        log_dir=root / "logs",
        network=profile.network,
    )
    return server.__enter__()


def expected_of(backend: BackendSpec, env: EnvHandle, model: ResolvedModel | None) -> Attestation:
    """Return the attestation a server of this backend must give, computed without starting it.

    Raises:
        ValueError: there is no model.
        VllmStartError: the env's lock pins no vllm.
    """
    if model is None or backend.server is None:
        raise ValueError(f"backend {backend.name!r} serves a model through a server, and the call resolved none")
    return expected_attestation(env, model, backend.server.settings)


def worker_init(device: str, model: ResolvedModel | None) -> dict[str, Any]:
    """Return a worker's init: the device, the dtype (`PRECISION`) and, with a model, its source, id and identity.

    A worker loads a model through its `ModelCache` under that identity, device and dtype.
    """
    init: dict[str, Any] = {"device": device, "dtype": PRECISION}
    if model is not None:
        init["model"] = {"source": model.ref.source, "id": model.ref.id, "identity": model.identity}
    return init


_CACHES: dict[int | None, ModelCache] = {}
_CACHES_LOCK = threading.Lock()


def process_cache(budget: int | None) -> ModelCache:
    """Return this process's model cache of `budget` bytes (None for no budget), made on first use."""
    with _CACHES_LOCK:
        if budget not in _CACHES:
            _CACHES[budget] = ModelCache(budget if budget is not None else 2**63 - 1)
        return _CACHES[budget]


class WorkerSet:
    """The pooled workers of a process, one per backend, env, init and budget or network, and the servers.

    At most `bound` pooled workers are kept: once more exist, the least recently used idle ones are closed. A
    worker is held by one job at a time. Host workers keep their state in this process's model cache of their budget
    (`process_cache`), shared by every set. A server is started once per (env, backend, model); its start holds a
    lease of that triple alone, so starts of other servers proceed. Servers are not counted in the bound.
    """

    def __init__(self, bound: int, *, start_server: ServerStart = start_vllm) -> None:
        """Keep at most `bound` pooled workers; servers start through `start_server`.

        Raises:
            ValueError: `bound` is below 1.
        """
        if bound < 1:
            raise ValueError(f"a worker set holds at least one worker, got bound {bound}")
        self.bound = bound
        self._start = start_server
        self._lock = threading.Lock()
        self._pool: collections.OrderedDict[tuple[str, ...], Worker] = collections.OrderedDict()
        self._held: dict[tuple[str, ...], threading.Lock] = {}
        self._busy: collections.Counter[tuple[str, ...]] = collections.Counter()
        self._servers: dict[tuple[str, str, str], Server] = {}
        self._leases: dict[tuple[str, str, str], threading.Lock] = {}

    def tighten(self, bound: int) -> None:
        """Lower the bound to `bound` when that is lower, closing the idle workers beyond it; a higher one is ignored.

        Raises:
            ValueError: `bound` is below 1.
        """
        if bound < 1:
            raise ValueError(f"a worker set holds at least one worker, got bound {bound}")
        with self._lock:
            self.bound = min(self.bound, bound)
        self._evict()

    def pooled(self) -> list[Worker]:
        """Return the pooled workers, least recently used first."""
        with self._lock:
            return list(self._pool.values())

    @contextlib.contextmanager
    def checkout(  # noqa: PLR0913 - the backend, env and model, then keyword-only placement
        self,
        backend: BackendSpec,
        env: EnvHandle,
        model: ResolvedModel | None,
        *,
        device: str,
        profile: ExecutionProfile,
        expected: Attestation | None = None,
    ) -> Iterator[Worker]:
        """Hold the worker of this backend, env, model and device until the block ends; other holders wait.

        A server backend's worker runs in this process over its running server, which must attest `expected`. A dead
        pooled worker is closed and replaced.

        Raises:
            ValueError: a server backend without `expected`; a budgeted host backend without `state_bytes`.
            VllmStartError, LeaseTimeout, LeaseUnavailable, AttestationMismatch: as `server`.
            WorkerFatal: the worker failed to start or load.
        """
        init = worker_init(device, model)
        if backend.env.kind == "server":
            if expected is None or model is None:
                raise ValueError(f"backend {backend.name!r} is a server backend; its expected attestation is needed")
            yield self.server_worker(backend, init, self.server(backend, env, model, profile, expected))
            return
        host = backend.env.kind == "host"
        variant = str(profile.model_cache_budget_bytes) if host else profile.network
        key = (backend.env.kind, env.python, backend.name, backend.worker, jcs_dumps(init).decode(), variant)
        with self._lock:
            held = self._held.setdefault(key, threading.Lock())
            self._busy[key] += 1
        try:
            with held:
                yield self._worker(key, backend, env, init, model, profile)
        finally:
            with self._lock:
                self._busy[key] -= 1
                if self._busy[key] == 0:
                    del self._busy[key]
            self._evict()

    def _worker(  # noqa: PLR0913 - the pool key and what a new worker is built from
        self,
        key: tuple[str, ...],
        backend: BackendSpec,
        env: EnvHandle,
        init: dict[str, Any],
        model: ResolvedModel | None,
        profile: ExecutionProfile,
    ) -> Worker:
        with self._lock:
            worker = self._pool.get(key)
        if isinstance(worker, HostWorker) or (worker is not None and worker.alive):
            with self._lock:
                self._pool.move_to_end(key)
            return worker
        if worker is not None:
            worker.close()
        worker = self._new_worker(backend, env, init, model, profile)
        with self._lock:
            self._pool[key] = worker
        self._evict()
        return worker

    def _evict(self) -> None:
        """Close the least recently used idle workers while more than `bound` are pooled."""
        closing: list[Worker] = []
        with self._lock:
            idle = [k for k in self._pool if self._busy[k] == 0]
            while len(self._pool) > self.bound and idle:
                closing.append(self._pool.pop(idle.pop(0)))
        for worker in closing:
            if isinstance(worker, WorkerClient):
                worker.close()

    def _new_worker(
        self,
        backend: BackendSpec,
        env: EnvHandle,
        init: dict[str, Any],
        model: ResolvedModel | None,
        profile: ExecutionProfile,
    ) -> Worker:
        if backend.env.kind != "host":
            return WorkerClient(
                env.python,
                backend.worker,
                init=init,
                variables=network_variables(profile.network),
                load_timeout=profile.timeouts.load_s,
                request_timeout=profile.timeouts.request_s,
            )
        budget = profile.model_cache_budget_bytes
        identity = model.identity if model is not None else value_digest("backend", {"name": backend.name})
        key: ModelKey = (backend.name, identity, init["device"], PRECISION)
        return HostWorker(
            backend.worker, init=init, key=key, cache=process_cache(budget), size_of=_size_of(backend.worker, budget)
        )

    def server(
        self,
        backend: BackendSpec,
        env: EnvHandle,
        model: ResolvedModel,
        profile: ExecutionProfile,
        expected: Attestation,
    ) -> Server:
        """Return the running server of this env, backend and model, starting it (or replacing a dead one) if needed.

        The start holds the lease of this env, backend and model only.

        Raises:
            VllmStartError, LeaseTimeout, LeaseUnavailable: as the server start.
            AttestationMismatch: the server attests something other than `expected`.
        """
        key = (env.python, backend.name, model.identity)
        with self._lock:
            lease = self._leases.setdefault(key, threading.Lock())
        with lease:
            with self._lock:
                server = self._servers.get(key)
            if server is not None:
                try:
                    server.context_fields()
                    return server
                except StaleAttestation:
                    with self._lock:
                        self._servers.pop(key, None)
                    server.stop()
            server = self._start(backend, env, model, profile)
            if server.attestation != expected:
                attested = server.attestation
                server.stop()
                raise AttestationMismatch(
                    f"the server of {server.label} attests {attested}, the lock declares {expected}"
                )
            with self._lock:
                self._servers[key] = server
            return server

    @staticmethod
    def server_worker(backend: BackendSpec, init: dict[str, Any], server: Server) -> HostWorker:
        """Return the in-process worker of a server backend, given `server` as `init["server"]`."""
        return HostWorker(
            backend.worker,
            init={**init, "server": server},
            key=(backend.name, init["model"]["identity"], init["device"], PRECISION),
            cache=ModelCache(2**63 - 1),
            size_of=lambda _: 0,
        )

    def close(self) -> None:
        """Stop every pooled worker process and every server; a later checkout starts its own again."""
        with self._lock:
            workers, servers = list(self._pool.values()), list(self._servers.values())
            self._pool.clear()
            self._servers.clear()
        for worker in workers:
            if isinstance(worker, WorkerClient):
                worker.close()
        for server in servers:
            server.stop()


def _size_of(module: str, budget: int | None) -> Callable[[object], int]:
    """Return how a backend's state is sized: 0 bytes with no budget, else by its module's `state_bytes`."""
    if budget is None:
        return lambda _: 0
    size_of = getattr(importlib.import_module(module), "state_bytes", None)
    if size_of is None:
        raise ValueError(f"backend module {module!r} defines no state_bytes, so a model-cache budget cannot hold it")
    return size_of  # type: ignore[no-any-return]
