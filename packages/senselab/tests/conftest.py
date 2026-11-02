"""Shared fixtures of the senselab tests."""

from collections.abc import Iterator

import pytest
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.testing.workers import RealWorkers


@pytest.fixture
def real_worker() -> Iterator[RealWorkers]:
    """Backend workers on their catalogue models, in the envs the packaged opener builds under `orcd`."""
    workers = RealWorkers(ExecutionProfile.load("orcd"))
    yield workers
    workers.close()
