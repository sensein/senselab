"""Shared fixtures of the senselab tests."""

from collections.abc import Callable, Iterator

import pytest
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.testing.workers import RealWorkers


@pytest.fixture
def real_worker(site_profile: Callable[..., ExecutionProfile]) -> Iterator[RealWorkers]:
    """Backend workers on their catalogue models, in the envs of `SENSELAB_TEST_PROFILE`."""
    workers = RealWorkers(site_profile())
    yield workers
    workers.close()
