"""`RealWorkers`: a backend's worker starts under the python of the env the packaged opener returns."""

import json
import platform
import stat
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
import senselab_core.runtime.fake_backend as fake
from senselab_core.kernel.declare import BackendSpec, Env, Registry, ServerConfig
from senselab_core.runtime.call import Packaged, packaged, use_packaged
from senselab_core.runtime.catalogue import Catalogue
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.vllm_server import VllmStartError
from senselab_core.runtime.worker_client import WorkerClient
from senselab_core.testing.interpreter import source_tree_python
from senselab_core.testing.workers import RealWorkers, run_alone
from senselab_worker.fake import FAULT_KEY
from senselab_worker.faults import FaultClass

pytestmark = pytest.mark.logic

TOY = "toyorg/toy"
REVISION = "a" * 40
BACKEND = BackendSpec(
    task=fake.ECHO.name,
    name="fake-isolated",
    version="1",
    stack="isolated",
    models=(TOY,),
    env=Env.isolated("fake"),
    devices=("cpu",),
    batch_invariant=True,
    worker=fake.WORKER_MODULE,
    load_stub={},
)
CATALOGUE = Catalogue.load(
    f"""
format: catalogue@1.0
entries:
- id: {TOY}
  revision: {REVISION}
  tasks: [{fake.ECHO.name}]
  backends: [fake-isolated]
  stacks: [isolated]
  default_for: [{fake.ECHO.name}]
  license:
    expression: MIT
    source: toy card
    verified: {{date: 2026-11-01, how: hf-card}}
    gated: false
"""
)


@pytest.fixture
def env_python(tmp_path: Path) -> Iterator[Path]:
    """The packaged opener returns an env whose python records its arguments to `<tmp>/argv.json`, then runs."""
    python = tmp_path / "env" / "bin" / "python"
    python.parent.mkdir(parents=True)
    record = f"import json, sys; json.dump(sys.argv[1:], open({str(tmp_path / 'argv.json')!r}, 'w'))"
    python.write_text(f'#!/bin/sh\n"{sys.executable}" -c "{record}" "$@"\nexec "{source_tree_python()}" "$@"\n')
    python.chmod(python.stat().st_mode | stat.S_IXUSR)
    registry = Registry()
    registry.add_task(fake.ECHO)
    registry.add_backend(BACKEND)

    def opener(backend: BackendSpec, profile: ExecutionProfile) -> EnvHandle:  # noqa: ARG001 - an EnvOpener
        return EnvHandle(
            python=str(python),
            python_version=platform.python_version(),
            lock_sha="1" * 64,
            wheel_set="2" * 64,
            torch_index=None,
        )

    previous = use_packaged(Packaged(CATALOGUE, opener, registry))
    yield python
    use_packaged(previous)


def test_worker_runs_in_the_opened_env_on_the_catalogue_pin(env_python: Path, tmp_path: Path) -> None:
    """The worker is launched by the env's python, not this process; faults reach it per request; `close` stops it.

    `run_alone` keeps the outputs of the items that succeed and lets a content fault pass.
    """
    workers = RealWorkers(ExecutionProfile.load())
    worker = workers(fake.ECHO.name, "fake-isolated")
    assert isinstance(worker, WorkerClient)
    faulted = {FAULT_KEY: {"fail_items": ["b"]}}
    results = worker.run([{"id": i, "inputs": {"payload": i}} for i in "ab"], faulted)
    alone = run_alone(worker, {i: {"id": i, "inputs": {"payload": i}} for i in "ab"}, faulted)
    workers.close()
    assert "outputs" in results[0] and results[1]["error"]["fault"] is FaultClass.CONTENT and list(alone) == ["a"]
    assert fake.WORKER_MODULE in json.loads((tmp_path / "argv.json").read_text()) and not worker.alive
    with pytest.raises(ValueError, match="no backend"):
        workers(fake.ECHO.name, "absent")


def test_starting_a_worker_stops_the_one_before(env_python: Path) -> None:  # noqa: ARG001 - the fixture's effect
    """Two models never share the device: the second start stops the first worker."""
    workers = RealWorkers(ExecutionProfile.load())
    first = workers(fake.ECHO.name, "fake-isolated")
    second = workers(fake.ECHO.name, "fake-isolated")
    assert isinstance(first, WorkerClient) and isinstance(second, WorkerClient)
    assert not first.alive and second.alive and workers.started == [second]
    workers.close()


def _opened_by(seen: list[BackendSpec], python: Path, backend: BackendSpec) -> None:
    registry = Registry()
    registry.add_task(fake.ECHO)
    registry.add_backend(backend)

    def opener(b: BackendSpec, profile: ExecutionProfile) -> EnvHandle:  # noqa: ARG001 - an EnvOpener
        seen.append(b)
        return EnvHandle(str(python), platform.python_version(), "1" * 64, "2" * 64, None)

    use_packaged(Packaged(CATALOGUE, opener, registry))


def test_worker_sources_reach_the_opener_that_places_them(env_python: Path) -> None:
    """RealWorkers asks the packaged opener, which places `worker_sources`, for the backend as declared."""
    placed = BACKEND.model_copy(update={"worker_sources": {fake.WORKER_MODULE: "worker.py"}})
    seen: list[BackendSpec] = []
    previous = packaged()
    _opened_by(seen, env_python, placed)
    workers = RealWorkers(ExecutionProfile.load())
    workers(fake.ECHO.name, "fake-isolated")
    workers.close()
    use_packaged(previous)
    assert [b.worker_sources for b in seen] == [placed.worker_sources]


def test_server_backend_starts_over_its_server_not_in_the_env_python(env_python: Path, tmp_path: Path) -> None:
    """A server backend's worker runs in this process over its attested server: the env's python is never launched."""
    server = BackendSpec(
        task=fake.ECHO.name,
        name="fake-server",
        version="1",
        stack="server",
        models=(TOY,),
        env=Env.server("fake"),
        devices=("cpu",),
        batch_invariant=True,
        worker=fake.WORKER_MODULE,
        load_stub={},
        server=ServerConfig(settings={}, startup_timeout_s=1.0),
    )
    previous = packaged()
    _opened_by([], env_python, server)
    workers = RealWorkers(ExecutionProfile.load())
    try:
        with pytest.raises(VllmStartError, match="pins no vllm"):
            workers(fake.ECHO.name, "fake-server")
    finally:
        use_packaged(previous)
    assert not (tmp_path / "argv.json").exists()
