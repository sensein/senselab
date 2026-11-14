"""Makes every interpreter with this directory on PYTHONPATH read the fake stack: the ladder and its pool workers."""

import platform

from senselab_core.kernel.declare import REGISTRY, BackendSpec
from senselab_core.runtime.call import Packaged, use_packaged
from senselab_core.runtime.catalogue import Catalogue
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.fake_backend import ECHO, FAKE  # noqa: F401 - declares the stack in REGISTRY
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.testing.interpreter import source_tree_python
from senselab_core.testing.workers import mit_lock_records


def _host(backend: BackendSpec, profile: ExecutionProfile) -> EnvHandle:  # noqa: ARG001 - an EnvOpener
    """Return this interpreter as the fake backend's env, under a fixed lock."""
    return EnvHandle(source_tree_python(), platform.python_version(), "3" * 64, "4" * 64, None)


use_packaged(Packaged(Catalogue(entries=()), _host, REGISTRY, mit_lock_records))
