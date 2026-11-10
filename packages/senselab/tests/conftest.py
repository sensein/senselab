"""Shared fixtures of the senselab tests."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.testing.workers import RealWorkers

CLUSTER = (
    next(p for p in Path(__file__).resolve().parents if p.name == "packages")
    / "senselab-core/tests/data/profiles/cluster.yaml"
)


@pytest.fixture
def real_worker() -> Iterator[RealWorkers]:
    """Backend workers on their catalogue models, in the envs the packaged opener builds under the `cluster` fixture."""
    workers = RealWorkers(ExecutionProfile.load(CLUSTER))
    yield workers
    workers.close()
