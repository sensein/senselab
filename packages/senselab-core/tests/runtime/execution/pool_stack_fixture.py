"""A one-node plan over host backends for pool tests: its task, registry, packaged stack and placed plan."""

import contextlib
import os
import platform
from collections.abc import Iterator, Sequence
from pathlib import Path

from planted_objects_fixture import MIRROR_INDEX
from senselab_core.kernel.dataset import Record, ValueInput
from senselab_core.kernel.declare import BackendSpec, Env, Registry, TaskParams, TaskSpec
from senselab_core.kernel.graph import Edge, Graph, Node
from senselab_core.kernel.identity import InputIdentity
from senselab_core.runtime.call import Packaged, RunConfig, bind_input, use_packaged
from senselab_core.runtime.catalogue import Catalogue
from senselab_core.runtime.datasets import RecordSet
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.execution_settings import JobLease
from senselab_core.runtime.licenses import license_record
from senselab_core.runtime.planner import Plan, plan
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.units import Unresolved
from senselab_core.testing.interpreter import source_tree_python
from senselab_core.testing.measures import FixedUnit


def unresolving(profile: ExecutionProfile) -> FixedUnit:
    """Return a resolver whose every unit is unresolved for a final cause, so every unit runs the safe values."""
    return FixedUnit(profile, Unresolved("the host names no hardware class"))


def lease_of(placed: Plan) -> JobLease:
    """Return the lease section of the profile `placed` was placed under."""
    assert placed.execution is not None
    return placed.execution.lease


BACKEND = "backends.sizes_backend_fixture"
"""The host backend every task here is served by: it records request sizes and fails inputs `fail` and `gap`."""


def scoring(name: str) -> TaskSpec:
    """Return a transform task `name` from one JSON input `text` to one output `value`."""
    return TaskSpec(
        name=name,
        version="1",
        kind="transform",
        inputs={"text": "json"},
        outputs={"value": "sha256"},
        params=TaskParams,
    )


def registry(*tasks: TaskSpec, worker: str = BACKEND, name: str = "dsp", env: Env | None = None) -> Registry:
    """Return a registry of `tasks`, each served by one batch-invariant backend `name` running `worker` in `env`.

    `env` is the host env when None.
    """
    found = Registry()
    for task in tasks:
        found.add_task(task)
        found.add_backend(
            BackendSpec(
                task=task.name,
                name=name,
                version="1",
                stack="dsp",
                env=env if env is not None else Env.host(),
                devices=("cpu",),
                batch_invariant=True,
                worker=worker,
                load_stub={},
            )
        )
    return found


def host_env(backend: BackendSpec, profile: ExecutionProfile) -> EnvHandle:  # noqa: ARG001 - an EnvOpener
    """Return this interpreter as every backend's env, under a fixed lock."""
    return EnvHandle(
        python=source_tree_python(),
        python_version=platform.python_version(),
        lock_sha="3" * 64,
        wheel_set="4" * 64,
        torch_index=MIRROR_INDEX,
    )


@contextlib.contextmanager
def packaged(found: Registry, cache: Path) -> Iterator[None]:
    """Use `found` with no catalogue, every env this interpreter under an MIT lock, and `SENSELAB_CACHE` at `cache`."""
    previous_cache = os.environ.get("SENSELAB_CACHE")
    os.environ["SENSELAB_CACHE"] = str(cache)
    mit = license_record("MIT", source="lock", verified={"date": "2026-11-06", "how": "pep639"}, gated=False)
    catalogue = Catalogue.load("format: catalogue@2.0\nentries: []\n")
    previous = use_packaged(Packaged(catalogue, host_env, found, lambda _: {"p": mit}))
    try:
        yield
    finally:
        use_packaged(previous)
        if previous_cache is None:
            os.environ.pop("SENSELAB_CACHE", None)
        else:
            os.environ["SENSELAB_CACHE"] = previous_cache


def planned(
    task: TaskSpec, profile: ExecutionProfile, texts: Sequence[str]
) -> tuple[Plan, dict[InputIdentity, object]]:
    """Return the plan mapping `task` over one record per text, placed under `profile`, and its input values."""
    graph = Graph(
        inputs={"text": "list[json]"},
        nodes=(Node(id="a", task=task.name, task_version="1"),),
        edges=(Edge(source="input.text", target="a.text", kind="map"),),
        outputs={"value": "a.value"},
    )
    rows = [Record(id=f"r{i}", inputs={"text": ValueInput(value=t)}) for i, t in enumerate(texts)]
    placed = plan(graph, RecordSet(rows), config=RunConfig(), execution=profile)
    return placed, {bind_input("x", "json", t)[0]: t for t in texts}
