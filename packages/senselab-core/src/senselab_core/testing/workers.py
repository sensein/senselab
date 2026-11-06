"""Backend workers on their catalogue models, started as the job engine starts them: in their env, or over a server."""

from collections.abc import Mapping
from typing import Any

from senselab_worker.faults import FaultClass

from senselab_core.kernel.declare import BackendSpec
from senselab_core.kernel.identity import ModelRef, ResolvedModel
from senselab_core.kernel.license import License
from senselab_core.runtime.call import packaged
from senselab_core.runtime.execution.workers import PRECISION, WorkerSet, expected_of
from senselab_core.runtime.hardware import probe_hardware
from senselab_core.runtime.host_worker import HostWorker
from senselab_core.runtime.licenses import license_record
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.worker_client import WorkerClient

TYPED = (FaultClass.CONTENT, FaultClass.RESOURCE)
"""The fault classes an edge input may fail with: never an infrastructure fault."""

_MIT = license_record("MIT", source="test lock", verified={"date": "2026-11-04", "how": "manual"}, gated=False)


def mit_lock_records(backend: BackendSpec) -> Mapping[str, License | None]:  # noqa: ARG001 - a LockRecords
    """Return lock records giving every env one MIT-licensed package, for a test `Packaged` whose hits are served."""
    return {"pkg": _MIT}


def run_alone(
    worker: WorkerClient | HostWorker, items: Mapping[str, dict[str, Any]], params: dict[str, Any]
) -> dict[str, Any]:
    """Run each item of `items` alone and return the outputs of those that succeed, by name.

    Raises:
        AssertionError: an item fails with other than a content or resource fault.
    """
    outputs = {}
    for name, item in items.items():
        [result] = worker.run([item], params)
        if "error" in result:
            assert result["error"]["fault"] in TYPED, (name, result["error"])
        else:
            outputs[name] = result["outputs"]
    return outputs


class RealWorkers:
    """Start workers through the packaged env opener under `profile`; `close` stops every one started.

    One worker is live at a time: starting a worker first stops those started before it. Each worker loads its
    backend's first model (else its task's catalogue default, when there is one) at the packaged catalogue's pin,
    on `cuda` when the host has a GPU, else `cpu`.
    """

    def __init__(self, profile: ExecutionProfile, device: str | None = None) -> None:
        """Start nothing yet; `device` defaults to `cuda` when the host has a GPU, else `cpu`."""
        self.profile = profile
        self.device = device or ("cuda" if probe_hardware().gpus else "cpu")
        self.started: list[WorkerClient] = []
        self.servers = WorkerSet(1)

    def __call__(self, task: str, name: str) -> WorkerClient | HostWorker:
        """Stop the workers started before, then return a started worker of backend `name` of `task`.

        Faults are injected per request, by `params["_fault"]`.

        Raises:
            ValueError: `task` has no backend `name`.
            As `ensure_env` and `WorkerClient`.
        """
        found = [b for b in packaged().registry.backends(task) if b.name == name]
        if not found:
            raise ValueError(f"{task!r} has no backend {name!r}")
        backend = found[0]
        self.close()
        init: dict[str, Any] = {"device": self.device, "dtype": PRECISION}
        entry = packaged().catalogue.backend_entry(backend)
        if entry is not None:
            init["model"] = {"source": "hf", "id": entry.id, "identity": entry.revision}
        env = packaged().open_env(backend, self.profile)
        if backend.env.kind == "server":
            if entry is None:
                raise ValueError(f"backend {name!r} serves a model through a server, and none is catalogued for it")
            model = ResolvedModel(
                ref=ModelRef.parse(f"{entry.id}@{entry.revision}"), identity=entry.revision, license=entry.license
            )
            server = self.servers.server(backend, env, model, self.profile, expected_of(backend, env, model))
            return self.servers.server_worker(backend, init, server)
        worker = WorkerClient(env.python, backend.worker, init=init)
        self.started.append(worker)
        return worker

    def close(self) -> None:
        """Stop every worker started, and the server a server backend's worker runs over."""
        for worker in self.started:
            worker.close()
        self.started.clear()
        self.servers.close()
