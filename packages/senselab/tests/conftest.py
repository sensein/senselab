"""Shared fixtures of the senselab tests."""

import os
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.testing.workers import RealWorkers


@pytest.fixture
def real_worker(site_profile: Callable[..., ExecutionProfile]) -> Iterator[RealWorkers]:
    """Backend workers on their catalogue models, in the envs of `SENSELAB_TEST_PROFILE`."""
    workers = RealWorkers(site_profile())
    yield workers
    workers.close()


@pytest.fixture(autouse=True)
def importable_by_pool_workers(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Let pool worker processes import the test's own module and its fixtures, which a job's payload may name."""
    here = str(Path(str(request.node.fspath)).parent)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([here, *filter(None, [os.environ.get("PYTHONPATH")])]))
