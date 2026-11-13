"""The fake echo task as the packaged stack, over a cache root of its own, for the exit-harness tests."""

import platform
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from senselab_core.kernel.declare import REGISTRY, BackendSpec, TaskSpec
from senselab_core.runtime import fake_backend
from senselab_core.runtime.call import Packaged, use_packaged
from senselab_core.runtime.catalogue import Catalogue
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.testing.workers import mit_lock_records


def this_interpreter(backend: BackendSpec, profile: ExecutionProfile) -> EnvHandle:  # noqa: ARG001 - an EnvOpener
    """Open every backend's env as this interpreter."""
    return EnvHandle(sys.executable, platform.python_version(), "1" * 64, "2" * 64, None)


@pytest.fixture
def fake_stack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TaskSpec]:
    """The fake echo task as the packaged stack, over a cache root of its own."""
    monkeypatch.setenv("SENSELAB_CACHE", str(tmp_path / "cache"))
    previous = use_packaged(Packaged(Catalogue(entries=()), this_interpreter, REGISTRY, mit_lock_records))
    yield fake_backend.ECHO
    use_packaged(previous)
