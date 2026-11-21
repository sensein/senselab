"""What starts a backend's worker: its init, the process model cache, a server and the worker over it."""

import importlib
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, Protocol, TypeVar

from senselab_worker.faults import FaultClass, at_load, classification
from senselab_worker.model_cache import ModelCache, ModelKey
from senselab_worker.serve import backend_init

from senselab_core.kernel._base import value_digest
from senselab_core.kernel.declare import BackendSpec
from senselab_core.kernel.identity import Attestation, ResolvedModel
from senselab_core.runtime.call import packaged
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.hardware import measured_budget, probe_hardware
from senselab_core.runtime.host_worker import HostWorker
from senselab_core.runtime.locks import LeaseTimeout, LeaseUnavailable
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.vllm_server import (
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


class WorkerLoadFailed(Exception):  # noqa: N818 - a fault named for what failed, as its callers know it
    """A worker did not start or load its backend; `__cause__` is what it raised.

    Its fault class and gap are its cause's (`carried_class`), so a load that ran out of memory is a resource fault,
    re-placed on a larger class; a content fault stated at load is the environment's (`faults.at_load`), and with no
    cause it is an infrastructure fault of its env. It is raised by a checkout, never by a request to a loaded worker,
    so it is never split.
    """

    @property
    def carried_class(self) -> tuple[FaultClass, bool]:
        """The fault class of the cause and whether the table decided it."""
        cause = self.__cause__
        return at_load(*classification(cause)) if cause is not None else (FaultClass.INFRASTRUCTURE, False)

    from_load: ClassVar[bool] = True


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
        start_margin_s=profile.envs.server_start_margin_s,
        lease=profile.lease,
        vllm=profile.vllm,
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


_SERVERS = ModelCache(2**63 - 1)  # invariant: the largest signed 64-bit size, read as no bound
"""The state and request count of each server instance this process consumes."""
_CACHES: dict[int | None, ModelCache] = {}
_CACHES_LOCK = threading.Lock()


def _attested(server: Server) -> dict[str, Any]:
    attestation = server.attestation
    return {} if attestation is None else attestation.model_dump(mode="json")


def process_cache(budget: int | None, device_query_s: float | None = None) -> ModelCache:
    """Return this process's model cache of `budget` bytes, made on first use; None measures it (`measured_budget`).

    The measure reads the devices within `device_query_s` seconds (`sizing.device_query_s`).
    """
    with _CACHES_LOCK:
        if budget not in _CACHES:
            _CACHES[budget] = ModelCache(
                budget if budget is not None else measured_budget(probe_hardware(timeout_s=device_query_s))
            )
        return _CACHES[budget]


def server_worker(backend: BackendSpec, init: dict[str, Any], server: Server) -> HostWorker:
    """Return the in-process worker of a server backend, given `server` as `init["server"]`.

    Its requests are counted per attested server instance, in this process's server cache: a second consumer of a
    server another has used reports no load.
    """
    instance = value_digest("server", {"url": server.url, "attestation": _attested(server)})
    return HostWorker(
        backend.worker,
        init={**backend_init(init), "server": server},
        key=(backend.name, instance, init["device"], PRECISION),
        cache=_SERVERS,
        size_of=lambda _: 0,
    )


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


def loaded(start: Callable[[], Started]) -> Started:
    """Return the worker `start` starts and loads.

    Raises:
        LeaseTimeout, LeaseUnavailable: a start lease is busy or cannot be read, as raised.
        WorkerLoadFailed: whatever else the start or load raised, as its cause; it has the cause's fault class.
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


def size_of_state(backend: BackendSpec) -> Callable[[object], int]:
    """Return how a backend's state counts against the model-cache budget: by its module's `state_bytes`.

    A backend that declares no `models` and no `state_bytes` holds none, and counts 0.

    Raises:
        ValueError: the backend declares `models` and its module no `state_bytes`.
    """
    size_of: Callable[[object], int] | None = getattr(importlib.import_module(backend.worker), "state_bytes", None)
    if size_of is not None:
        return size_of
    if backend.models:
        raise ValueError(
            f"backend {backend.name!r} serves models and its module {backend.worker!r} defines no state_bytes, "
            "so the model-cache budget cannot hold it"
        )
    return lambda _: 0


def release_process_caches() -> None:
    """Drop every state the process model caches hold, so a later load loads again."""
    with _CACHES_LOCK:
        caches = list(_CACHES.values())
    for cache in caches:
        cache.clear()
