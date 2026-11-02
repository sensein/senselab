"""Backend workers on their catalogue models, each started as `run_call` starts it: in its env, or over its server."""

from collections.abc import Mapping
from typing import Any

from senselab_worker.faults import FaultClass

from senselab_core.kernel.identity import ModelRef, ResolvedModel
from senselab_core.runtime.call import packaged, start_server_worker, stop_servers
from senselab_core.runtime.hardware import probe_hardware
from senselab_core.runtime.in_process import InProcessWorker
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.worker_client import WorkerClient

TYPED = (FaultClass.CONTENT, FaultClass.RESOURCE)
"""The fault classes an edge input may fail with: never an infrastructure fault."""


def run_alone(
    worker: WorkerClient | InProcessWorker, items: Mapping[str, dict[str, Any]], params: dict[str, Any]
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

    def __init__(self, profile: ExecutionProfile) -> None:
        """Start nothing yet."""
        self.profile = profile
        self.device = "cuda" if probe_hardware().gpus else "cpu"
        self.started: list[WorkerClient] = []
        self.serving = False

    def __call__(self, task: str, name: str, faults: Mapping[str, Any] | None = None) -> WorkerClient | InProcessWorker:
        """Stop the workers started before, then return a started worker of backend `name` of `task`.

        `faults` (the `FakeFaults` fields) go in its init.

        Raises:
            ValueError: `task` has no backend `name`.
            As `ensure_env` and `WorkerClient`.
        """
        found = [b for b in packaged().registry.backends(task) if b.name == name]
        if not found:
            raise ValueError(f"{task!r} has no backend {name!r}")
        backend = found[0]
        self.close()
        init: dict[str, Any] = {"device": self.device, "faults": dict(faults or {})}
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
            server_worker = start_server_worker(backend, env, model, self.profile, init)
            self.serving = True
            return server_worker
        worker = WorkerClient(env.python, backend.worker, init=init)
        self.started.append(worker)
        return worker

    def close(self) -> None:
        """Stop every worker started, and the server a server backend's worker runs over."""
        for worker in self.started:
            worker.close()
        self.started.clear()
        if self.serving:
            stop_servers()
            self.serving = False
