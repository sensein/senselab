"""senselab's pydra providers: a backend's model worker, loaded once and resident in a warm pydra worker.

`senselab-worker@3` owns a process: a `WorkerClient` speaking wire protocol v3 to a worker in the backend's locked
env, or, for a server backend, a server private to the pydra worker (a live registered service attesting the same
is consumed instead). `senselab-host@3` loads a host backend in the pydra worker's own process, its state in the
process model cache. A provided field's spec (`Loader.spec`) is the job's own loader as JSON: its backend, env, model,
device, profile and attestation. A provider holds no loader; it rebuilds the one a spec carries (`Loader.of`), so a
job's payload carries its own loader and no other.
"""

import functools
import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import senselab_worker
from senselab_worker import WIRE_VERSION
from senselab_worker.digest import file_digest, identity_digest
from senselab_worker.serve import backend_init

from senselab_core.kernel._base import value_digest
from senselab_core.kernel.declare import BackendSpec
from senselab_core.kernel.identity import Attestation, ResolvedModel
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.execution.services import service_registry
from senselab_core.runtime.execution.workers import (
    Server,
    ServerStart,
    WorkerLoadFailed,
    host_model_key,
    loaded,
    process_cache,
    server_worker,
    size_of_state,
    start_vllm,
    worker_init,
)
from senselab_core.runtime.host_worker import HostWorker
from senselab_core.runtime.network import network_variables
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.teardown import run_all
from senselab_core.runtime.vllm_server import AttestationMismatch
from senselab_core.runtime.worker_client import WorkerClient

WORKER = "senselab-worker@3"
"""The process-owning provider: an isolated backend's worker process, or a server backend's server."""
HOST = "senselab-host@3"
"""The in-process provider: a host backend in the pydra worker's own process."""


@dataclass(frozen=True)
class Loader:
    """What loads one backend's worker: the backend, its env, the model, the device, the profile and the attestation.

    `expected` is what a server backend's server must attest; None for any other backend.
    """

    backend: BackendSpec
    env: EnvHandle
    model: ResolvedModel | None
    device: str
    profile: ExecutionProfile
    expected: Attestation | None = None

    @property
    def provider(self) -> str:
        """The name of the provider that loads it: `HOST` for a host backend, else `WORKER`."""
        return HOST if self.backend.env.kind == "host" else WORKER

    @property
    def spec(self) -> dict[str, Any]:
        """The provided field's JSON spec: the loader as JSON (`of` rebuilds it), and its digest as `id`."""
        env = self.env
        loader = {
            "backend": {
                **self.backend.model_dump(mode="json", exclude={"params"}),
                "params": f"{self.backend.params.__module__}:{self.backend.params.__qualname__}",
            },
            "env": {
                "python": env.python,
                "python_version": env.python_version,
                "lock_sha": env.lock_sha,
                "wheel_set": env.wheel_set,
                "torch_index": env.torch_index,
                "lock": None if env.lock is None else str(env.lock),
            },
            "model": None if self.model is None else self.model.model_dump(mode="json"),
            "device": self.device,
            "profile": self.profile.model_dump(mode="json"),
            "expected": None if self.expected is None else self.expected.model_dump(mode="json"),
        }
        return {"loader": loader, "id": value_digest("loader", loader)}

    @classmethod
    def of(cls, spec: Mapping[str, Any]) -> "Loader":
        """Return the loader a provided field's spec carries; the backend's params class is imported by name.

        Raises:
            ValueError: the spec carries no loader, or one that does not validate.
        """
        try:
            given = spec["loader"]
            env = dict(given["env"])
            module, _, qualname = given["backend"]["params"].partition(":")
            params = functools.reduce(getattr, qualname.split("."), importlib.import_module(module))
            return cls(
                backend=BackendSpec.model_validate({**given["backend"], "params": params}),
                env=EnvHandle(**{**env, "lock": None if env["lock"] is None else Path(env["lock"])}),
                model=None if given["model"] is None else ResolvedModel.model_validate(given["model"]),
                device=given["device"],
                profile=ExecutionProfile.model_validate(given["profile"]),
                expected=None if given["expected"] is None else Attestation.model_validate(given["expected"]),
            )
        except (KeyError, TypeError, ImportError, AttributeError) as err:
            raise ValueError(f"a provided field's spec carries no loader: {err}") from None


@functools.cache
def source_digest() -> str:
    """Return the digest of the senselab-core and senselab-worker source files: the code a load runs under."""
    listed = []
    for package in (Path(__file__).parents[2], Path(senselab_worker.__file__).parent):
        for path in sorted(package.rglob("*.py")):
            listed.append([f"{package.name}/{path.relative_to(package).as_posix()}", file_digest("file@1", path)])
    return identity_digest("core-runtime@1", listed)


class Model:
    """A loaded backend worker: `run` sends one request; `close` stops what the load started, and waits."""

    def __init__(self, worker: HostWorker | WorkerClient, stops: tuple[Callable[[], None], ...] = ()) -> None:
        """Hold `worker`; `stops` are run after it closes (a private server's stop)."""
        self.worker = worker
        self._stops = stops

    def run(self, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
        """Send `items` with `params` as one request and return one result per item."""
        return self.worker.run(items, params)

    def close(self) -> None:
        """Stop the worker process and any server it started, waiting for each."""
        closing = (self.worker.close,) if isinstance(self.worker, WorkerClient) else ()
        run_all(*closing, *self._stops)


class _Provider:
    """The provider protocol over the `Loader` each spec carries."""

    name: ClassVar[str]

    def load(self, spec: Mapping[str, Any], device: object) -> Model:  # noqa: ARG002 - the spec names the device
        """Load the worker `spec` names and return it.

        Raises:
            WorkerLoadFailed: the worker or server failed to start or load, as its cause.
            ValueError: the spec carries no loader.
        """
        raise NotImplementedError

    def close(self, obj: object) -> None:
        """Stop what `load` started, and wait."""
        if isinstance(obj, Model):
            obj.close()

    def footprint(self, spec: Mapping[str, Any]) -> None:  # noqa: ARG002 - no declared figure
        """Return no declared footprint: placement reads the calibration."""
        return

    def measure(self, obj: object) -> None:  # noqa: ARG002 - the pool measures its own processes
        """Return no figure of its own: the pool attributes memory to its processes."""
        return

    def version(self) -> str:
        """The loader's code: the provider, the wire protocol and the senselab-core build."""
        return f"{self.name} wire@{WIRE_VERSION} {source_digest()}"


class SenselabWorker(_Provider):
    """`senselab-worker@3`: an isolated backend's worker process, or a server backend over a server.

    `start` starts a server backend's server (a module-level function, so the provider pickles with a job).
    """

    name: ClassVar[str] = WORKER

    def __init__(self, start: ServerStart = start_vllm) -> None:
        """Start servers with `start`."""
        self.start = start

    def load(self, spec: Mapping[str, Any], device: object) -> Model:  # noqa: ARG002 - the spec names the device
        """Start the worker process (or the server and its in-process worker) `spec` names and load its model.

        Raises:
            WorkerLoadFailed: the worker or server failed to start or load, as its cause.
            ValueError: the spec carries no loader.
        """
        found = Loader.of(spec)
        init = worker_init(found.device, found.model)
        if found.backend.env.kind == "server":
            server, stops = server_of(found, self.start)
            return Model(server_worker(found.backend, init, server), stops)
        profile = found.profile
        return Model(
            loaded(
                lambda: WorkerClient(
                    found.env.python,
                    found.backend.worker,
                    init=init,
                    variables=network_variables(profile.network),
                    load_timeout=profile.timeouts.load_s,
                    request_timeout=profile.timeouts.request_s,
                    settings=profile.worker,
                )
            )
        )


def server_of(found: Loader, start: ServerStart) -> tuple[Server, tuple[Callable[[], None], ...]]:
    """Return the server of `found`: a live registered service attesting its expectation, else one started here.

    A server started here is registered on the host's registry for other pool workers to consume, and withdrawn once
    stopped.

    Raises:
        WorkerLoadFailed: the server did not start, or attests other than expected.
        LeaseTimeout: another worker is starting this server.
    """
    if found.model is None or found.expected is None:
        raise WorkerLoadFailed(
            f"backend {found.backend.name!r} is a server backend; its model and attestation are needed"
        )
    model, expected = found.model, found.expected
    registry = service_registry(found.profile)
    consumer = registry.consume(expected)
    if registry.lookup(expected) is not None:
        return consumer, (consumer.stop,)
    consumer.stop()
    service = registry.reserve(expected)
    try:
        server = loaded(lambda: start(found.backend, found.env, model, found.profile))
        if server.attestation != expected:
            attested = server.attestation
            server.stop()
            raise WorkerLoadFailed(f"server of {server.label}") from AttestationMismatch(
                f"the server of {server.label} attests {attested}, the lock declares {expected}"
            )
        try:
            assert server.url is not None
            service.publish(server.url)
        except BaseException:
            server.stop()
            raise
    except BaseException:
        service.close()
        raise
    return server, (server.stop, service.close)


class SenselabHost(_Provider):
    """`senselab-host@3`: a host backend in this process, its state in the process model cache."""

    name: ClassVar[str] = HOST

    def load(self, spec: Mapping[str, Any], device: object) -> Model:  # noqa: ARG002 - the spec names the device
        """Load the host backend `spec` names in this process.

        Raises:
            WorkerLoadFailed: the backend failed to load, as its cause.
            ValueError: the spec carries no loader.
        """
        found = Loader.of(spec)
        init = worker_init(found.device, found.model)
        profile = found.profile
        return Model(
            loaded(
                lambda: HostWorker(
                    found.backend.worker,
                    init=backend_init(init),
                    key=host_model_key(found.backend, found.env, found.model, found.device),
                    cache=process_cache(profile.model_cache_budget_bytes, profile.sizing.device_query_s),
                    size_of=size_of_state(found.backend),
                )
            )
        )
