"""`RealWorkers`: a backend's worker starts under the python of the env the packaged opener returns."""

import json
import platform
import stat
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
import senselab_core.runtime.fake_backend as fake
from senselab_core.kernel.declare import BackendSpec, Env, Registry
from senselab_core.runtime.call import Packaged, use_packaged
from senselab_core.runtime.catalogue import Catalogue
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.testing.workers import RealWorkers, run_alone
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
    python.write_text(f'#!/bin/sh\n"{sys.executable}" -c "{record}" "$@"\nexec "{sys.executable}" "$@"\n')
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
    """The worker is launched by the env's python, not this process; faults reach it; `close` stops it.

    `run_alone` keeps the outputs of the items that succeed and lets a content fault pass.
    """
    workers = RealWorkers(ExecutionProfile.load())
    worker = workers(fake.ECHO.name, "fake-isolated", {"fail_items": ["b"]})
    results = worker.run([{"id": i, "inputs": {"payload": i}} for i in "ab"], {})
    alone = run_alone(worker, {i: {"id": i, "inputs": {"payload": i}} for i in "ab"}, {})
    workers.close()
    assert "outputs" in results[0] and results[1]["error"]["fault"] is FaultClass.CONTENT and list(alone) == ["a"]
    assert fake.WORKER_MODULE in json.loads((tmp_path / "argv.json").read_text()) and not worker.alive
    with pytest.raises(ValueError, match="no backend"):
        workers(fake.ECHO.name, "absent")


def test_starting_a_worker_stops_the_one_before(env_python: Path) -> None:  # noqa: ARG001 - the fixture's effect
    """Two models never share the device: the second start stops the first worker."""
    workers = RealWorkers(ExecutionProfile.load())
    first = workers(fake.ECHO.name, "fake-isolated")
    second = workers(fake.ECHO.name, "fake-isolated", {"fail_items": ["b"]})
    assert not first.alive and second.alive and workers.started == [second]
    workers.close()
