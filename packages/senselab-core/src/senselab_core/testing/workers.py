"""Backend workers on their catalogue models, each started in the env its backend declares."""

from collections.abc import Mapping
from typing import Any

from senselab_worker.faults import FaultClass

from senselab_core.runtime.call import packaged
from senselab_core.runtime.hardware import probe_hardware
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.worker_client import WorkerClient

TYPED = (FaultClass.CONTENT, FaultClass.RESOURCE)
"""The fault classes an edge input may fail with: never an infrastructure fault."""


def run_alone(worker: WorkerClient, items: Mapping[str, dict[str, Any]], params: dict[str, Any]) -> dict[str, Any]:
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

    Each worker loads its backend's first model at the packaged catalogue's pin, on `cuda` when the host has a GPU,
    else `cpu`.
    """

    def __init__(self, profile: ExecutionProfile) -> None:
        """Start nothing yet."""
        self.profile = profile
        self.device = "cuda" if probe_hardware().gpus else "cpu"
        self.started: list[WorkerClient] = []

    def __call__(self, task: str, name: str, faults: Mapping[str, Any] | None = None) -> WorkerClient:
        """Return a started worker of backend `name` of `task`, with `faults` (the `FakeFaults` fields) in its init.

        Raises:
            ValueError: `task` has no backend `name`.
            As `ensure_env` and `WorkerClient`.
        """
        found = [b for b in packaged().registry.backends(task) if b.name == name]
        if not found:
            raise ValueError(f"{task!r} has no backend {name!r}")
        backend = found[0]
        entry = packaged().catalogue.entry(backend.models[0])
        init = {
            "model": {"source": "hf", "id": entry.id, "identity": entry.revision},
            "device": self.device,
            "faults": dict(faults or {}),
        }
        worker = WorkerClient(packaged().open_env(backend, self.profile).python, backend.worker, init=init)
        self.started.append(worker)
        return worker

    def close(self) -> None:
        """Stop every worker started."""
        for worker in self.started:
            worker.close()
        self.started.clear()
