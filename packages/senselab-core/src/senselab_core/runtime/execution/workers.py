"""The workers a process runs jobs on: pooled per backend, env and model, bounded, and the servers they run over.

A server is this set's own, or, with a registry, a registered service it consumes (`services`).
"""

import collections
import contextlib
import importlib
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Protocol, TypeVar

from senselab_worker.canonical import jcs_dumps
from senselab_worker.faults import InfrastructureFault
from senselab_worker.model_cache import ModelCache, ModelKey
from senselab_worker.serve import backend_init

from senselab_core.kernel._base import value_digest
from senselab_core.kernel.declare import BackendSpec
from senselab_core.kernel.identity import Attestation, ResolvedModel
from senselab_core.runtime.call import packaged
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.execution.services import ServiceLease, ServiceRegistry
from senselab_core.runtime.host_worker import HostWorker
from senselab_core.runtime.locks import LeaseTimeout, LeaseUnavailable
from senselab_core.runtime.network import network_variables
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.vllm_server import (
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
    """A running model server: what it attests, how a message names it, its endpoint, live context fields, `stop`."""

    label: str

    @property
    def url(self) -> str | None:
        """Where the running server answers; None once stopped."""

    @property
    def attestation(self) -> Attestation | None:
        """What the running server attests; None once stopped."""

    def context_fields(self) -> dict[str, Any]:
        """Return the context fields of the running server; raises `StaleAttestation` once it changed."""

    def stop(self) -> None:
        """Stop the server."""


class WorkerLoadFailed(InfrastructureFault):
    """A worker did not start or load its backend: an infrastructure fault of its env; `__cause__` is what it raised."""


Started = TypeVar("Started")

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


def expected_of(backend: BackendSpec, lock: Path | None, model: ResolvedModel | None) -> Attestation:
    """Return the attestation a server of this backend in an env built from `lock` must give, without starting it.

    Raises:
        ValueError: there is no model.
        VllmStartError: the lock pins no vllm.
    """
    if model is None or backend.server is None:
        raise ValueError(f"backend {backend.name!r} serves a model through a server, and the call resolved none")
    return expected_attestation(lock, model, backend.server.settings)


def service_of(backend: BackendSpec, model: ResolvedModel | None) -> Attestation | None:
    """Return what the service answering a call of `backend` on `model` attests, from its packaged env lock.

    None for a backend that is not a server backend. A call's key takes it (`Call.service`); the registry entry a
    call consumes, or the server it starts, attests it.

    Raises:
        LookupError: no declaring package names the backend's env lock.
        ValueError, VllmStartError: as `expected_of`.
    """
    if backend.server is None:
        return None
    return expected_of(backend, packaged().env_lock(backend), model)


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

    With a `registry`, a server is consumed from the registry while a live service attests what it must, and started
    otherwise; a set that `provide`s registers each server it starts there instead, and consumes none.
    """

    def __init__(
        self,
        bound: int,
        *,
        start_server: ServerStart = start_vllm,
        registry: ServiceRegistry | None = None,
        provide: bool = False,
    ) -> None:
        """Keep at most `bound` pooled workers; servers start through `start_server`.

        Raises:
            ValueError: `bound` is below 1, or `provide` without a registry.
        """
        if bound < 1:
            raise ValueError(f"a worker set holds at least one worker, got bound {bound}")
        if provide and registry is None:
            raise ValueError("a worker set that provides services needs their registry")
        self.bound = bound
        self._start = start_server
        self._registry, self._provide = registry, provide
        self._provided: dict[tuple[str, str, str], ServiceLease] = {}
        self._lock = threading.Lock()
        self._pool: collections.OrderedDict[tuple[str, ...], Worker] = collections.OrderedDict()
        self._held: dict[tuple[str, ...], threading.Lock] = {}
        self._busy: collections.Counter[tuple[str, ...]] = collections.Counter()
        self._holders: dict[tuple[str, ...], int] = {}
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
            LeaseTimeout, LeaseUnavailable: a server's start lease is busy or cannot be read.
            WorkerLoadFailed: a worker or a server failed to start or load, whatever else it raised
                (`VllmStartError` and `AttestationMismatch` from a server start among them).
        """
        init = worker_init(device, model)
        if backend.env.kind == "server":
            if expected is None or model is None:
                raise ValueError(f"backend {backend.name!r} is a server backend; its expected attestation is needed")
            yield _loaded(
                lambda: self.server_worker(backend, init, self.server(backend, env, model, profile, expected))
            )
            return
        host = backend.env.kind == "host"
        variant = str(profile.model_cache_budget_bytes) if host else profile.network
        named = (backend.task, backend.name, backend.version, backend.worker)
        key = (backend.env.kind, env.python, *named, jcs_dumps(init).decode(), variant)
        with self._lock:
            held = self._held.setdefault(key, threading.Lock())
            self._busy[key] += 1
        try:
            with held:
                with self._lock:
                    self._holders[key] = threading.get_ident()
                try:
                    yield self._worker(key, backend, env, init, model, profile)
                finally:
                    with self._lock:
                        self._holders.pop(key, None)
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
            return _loaded(
                lambda: WorkerClient(
                    env.python,
                    backend.worker,
                    init=init,
                    variables=network_variables(profile.network),
                    load_timeout=profile.timeouts.load_s,
                    request_timeout=profile.timeouts.request_s,
                )
            )
        budget = profile.model_cache_budget_bytes
        key = host_model_key(backend, env, model, init["device"])
        size_of = _size_of(backend.worker, budget)
        return _loaded(
            lambda: HostWorker(
                backend.worker, init=backend_init(init), key=key, cache=process_cache(budget), size_of=size_of
            )
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

        The start holds the lease of this env, backend and model only. A consuming set returns a live registered
        service attesting `expected` (`ConsumerLease`) instead; a providing set registers what it starts.

        Raises:
            VllmStartError, LeaseTimeout, LeaseUnavailable: as the server start or its registration.
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
                    self._retire(key, server)
            registry = self._registry
            if registry is not None and not self._provide and registry.lookup(expected) is not None:
                server = registry.consume(expected)
            else:
                server = self._started(backend, env, model, profile, expected)
                if self._provide:
                    assert registry is not None and server.url is not None
                    try:
                        self._provided[key] = registry.register(expected, server.url)
                    except BaseException:
                        server.stop()
                        raise
            with self._lock:
                self._servers[key] = server
            return server

    def _started(
        self,
        backend: BackendSpec,
        env: EnvHandle,
        model: ResolvedModel,
        profile: ExecutionProfile,
        expected: Attestation,
    ) -> Server:
        server = self._start(backend, env, model, profile)
        if server.attestation != expected:
            attested = server.attestation
            server.stop()
            raise AttestationMismatch(f"the server of {server.label} attests {attested}, the lock declares {expected}")
        return server

    def _retire(self, key: tuple[str, str, str], server: Server) -> None:
        """Withdraw the registration of `server` if this set provides it, then stop it."""
        with self._lock:
            provided = self._provided.pop(key, None)
        if provided is not None:
            provided.close()
        server.stop()

    def provided(self) -> list[ServiceLease]:
        """Return the registrations of the servers this set started and provides."""
        with self._lock:
            return list(self._provided.values())

    @staticmethod
    def server_worker(backend: BackendSpec, init: dict[str, Any], server: Server) -> HostWorker:
        """Return the in-process worker of a server backend, given `server` as `init["server"]`."""
        return HostWorker(
            backend.worker,
            init={**backend_init(init), "server": server},
            key=(backend.name, init["model"]["identity"], init["device"], PRECISION),
            cache=ModelCache(2**63 - 1),
            size_of=lambda _: 0,
        )

    def abandon(self, thread: int) -> None:
        """Close every pooled worker process the thread `thread` holds now, so its `run` raises `WorkerLost`.

        A later checkout starts another. A host worker runs in this process and cannot be stopped; it is left to finish.
        """
        with self._lock:
            held = [
                (k, w) for k, w in self._pool.items() if self._holders.get(k) == thread and isinstance(w, WorkerClient)
            ]
            for key, _ in held:
                del self._pool[key]
        for _, worker in held:
            worker.close()

    def close(self) -> None:
        """Stop every pooled worker process and every server, withdrawing what it provides; a checkout starts again."""
        with self._lock:
            workers, servers = list(self._pool.values()), list(self._servers.items())
            self._pool.clear()
            self._servers.clear()
        for worker in workers:
            if isinstance(worker, WorkerClient):
                worker.close()
        for key, server in servers:
            self._retire(key, server)


def host_model_key(backend: BackendSpec, env: EnvHandle, model: ResolvedModel | None, device: str) -> ModelKey:
    """Return the process `ModelCache` key of a host backend, as (backend code, identity, device, dtype).

    The backend code is `<name>:<worker module>@<version>`, so backends of different code never share a loaded state.
    The identity is the model's, so two tasks on one backend code and one model share it; without a model it is the
    digest of the backend's task, name, version and env.
    """
    identity = (
        model.identity
        if model is not None
        else value_digest(
            "backend", {"task": backend.task, "name": backend.name, "version": backend.version, "env": env.python}
        )
    )
    return (f"{backend.name}:{backend.worker}@{backend.version}", identity, device, PRECISION)


def _loaded(start: Callable[[], Started]) -> Started:
    """Return the worker `start` starts and loads.

    Raises:
        LeaseTimeout, LeaseUnavailable: a start lease is busy or cannot be read, as raised.
        WorkerLoadFailed: whatever else the start or load raised, as its cause.
    """
    try:
        return start()
    except (LeaseTimeout, LeaseUnavailable):
        raise
    except Exception as exc:
        failed = WorkerLoadFailed(f"{type(exc).__name__}: {exc}")
        for note in getattr(exc, "__notes__", ()):
            failed.add_note(note)
        raise failed from exc


def _size_of(module: str, budget: int | None) -> Callable[[object], int]:
    """Return how a backend's state is sized: 0 bytes with no budget, else by its module's `state_bytes`."""
    if budget is None:
        return lambda _: 0
    size_of = getattr(importlib.import_module(module), "state_bytes", None)
    if size_of is None:
        raise ValueError(f"backend module {module!r} defines no state_bytes, so a model-cache budget cannot hold it")
    return size_of  # type: ignore[no-any-return]
