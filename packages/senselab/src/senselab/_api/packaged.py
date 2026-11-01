"""The packaged catalogue and env locks, handed to `run_call` when `senselab` is imported."""

from importlib import resources
from pathlib import Path

from senselab_core.kernel.declare import BackendSpec
from senselab_core.runtime.call import Packaged, use_packaged
from senselab_core.runtime.catalogue import Catalogue
from senselab_core.runtime.envs import EnvHandle, EnvLockError, ensure_env
from senselab_core.runtime.profiles import ExecutionProfile

_DATA = resources.files("senselab").joinpath("data")
CATALOGUE = Catalogue.load(_DATA.joinpath("catalogue.yaml").read_text("utf-8"))
"""The packaged catalogue, `senselab/data/catalogue.yaml`."""


def open_env(backend: BackendSpec, profile: ExecutionProfile) -> EnvHandle:
    """Open the env `backend` declares from its packaged lock, `data/env_locks/<name>.txt`, under the profile's root.

    Raises:
        EnvLockError: a host backend, which declares no requirements to check against the senselab lock.
        As `ensure_env` otherwise.
    """
    env = backend.env
    if env.kind == "host":
        raise EnvLockError(f"host backend {backend.name!r} declares no requirements to check against the senselab lock")
    lock = Path(str(_DATA.joinpath("env_locks", f"{env.name}.txt")))
    return ensure_env(env, lock, profile.env_root(env.kind), settings=profile.envs)


PACKAGED = Packaged(CATALOGUE, open_env)
use_packaged(PACKAGED)
