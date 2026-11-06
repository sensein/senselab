"""A one-node plan over host backends for pool tests: its task, registry, packaged stack and placed plan."""

import contextlib
import platform
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from senselab_core.kernel.dataset import Record, ValueInput
from senselab_core.kernel.declare import BackendSpec, Env, Registry, TaskParams, TaskSpec
from senselab_core.kernel.graph import Edge, Graph, Node
from senselab_core.kernel.identity import InputIdentity
from senselab_core.runtime.call import Packaged, RunConfig, bind_input, use_packaged
from senselab_core.runtime.catalogue import Catalogue
from senselab_core.runtime.datasets import RecordSet
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.licenses import license_record
from senselab_core.runtime.planner import Plan, plan
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.testing.interpreter import source_tree_python

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


def registry(*tasks: TaskSpec, worker: str = BACKEND) -> Registry:
    """Return a registry of `tasks`, each served by one batch-invariant host backend `dsp` running `worker`."""
    found = Registry()
    for task in tasks:
        found.add_task(task)
        found.add_backend(
            BackendSpec(
                task=task.name,
                name="dsp",
                version="1",
                stack="dsp",
                env=Env.host(),
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
        torch_index=None,
    )


@contextlib.contextmanager
def packaged(found: Registry, cache: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Use `found` with no catalogue, every env this interpreter under an MIT lock, and the cache at `cache`."""
    monkeypatch.setenv("SENSELAB_CACHE", str(cache))
    mit = license_record("MIT", source="lock", verified={"date": "2026-11-06", "how": "pep639"}, gated=False)
    catalogue = Catalogue.load("format: catalogue@2.0\nentries: []\n")
    previous = use_packaged(Packaged(catalogue, host_env, found, lambda _: {"p": mit}))
    try:
        yield
    finally:
        use_packaged(previous)


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
