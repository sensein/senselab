"""Shared fixtures of the senselab tests."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.testing.workers import RealWorkers

CLUSTER = (
    next(p for p in Path(__file__).resolve().parents if p.name == "packages")
    / "senselab-core/tests/data/profiles/cluster.yaml"
)
SITE = Path(os.environ.get("SENSELAB_TEST_PROFILE") or CLUSTER)


@pytest.fixture
def real_worker() -> Iterator[RealWorkers]:
    """Backend workers on their catalogue models, in the envs of `SENSELAB_TEST_PROFILE`, else the fixture."""
    workers = RealWorkers(ExecutionProfile.load(SITE))
    yield workers
    workers.close()
