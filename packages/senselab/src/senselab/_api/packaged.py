"""The packaged catalogue and env locks, handed to `run_call` when `senselab` is imported."""

from importlib import resources
from pathlib import Path

from senselab_core.kernel.declare import BackendSpec
from senselab_core.runtime.call import Packaged, use_packaged
from senselab_core.runtime.catalogue import Catalogue
from senselab_core.runtime.envs import EnvHandle, ensure_env
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.testing.lock_licenses import lock_records_from

_DATA = resources.files("senselab").joinpath("data")
CATALOGUE = Catalogue.load(_DATA.joinpath("catalogue.yaml").read_text("utf-8"))
"""The packaged catalogue, `senselab/data/catalogue.yaml`."""
LOCK_RECORDS = lock_records_from(_DATA.joinpath("env_licenses.yaml").read_text("utf-8"))
"""The license records of each env lock's packages, `senselab/data/env_licenses.yaml`, for `contract_cases`."""


def host_lock() -> Path | None:
    """Return the senselab lock, the `uv.lock` beside the `pyproject.toml` of the checkout this package runs from.

    None when no checkout lies above the package, as in an installed wheel.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "uv.lock").is_file():
            return parent / "uv.lock"
    return None


def open_env(backend: BackendSpec, profile: ExecutionProfile) -> EnvHandle:
    """Open the env `backend` declares.

    A host backend is checked against the senselab lock, or the installed packages when the install ships none; any
    other is built from `data/env_locks/`.

    Raises:
        As `ensure_env`.
    """
    env = backend.env
    if env.kind == "host":
        lock = host_lock()
        root = Path(".") if lock is None else lock.parent
        return ensure_env(env, lock, root, settings=profile.envs, requirements=backend.requirements)
    env_lock = Path(str(_DATA.joinpath("env_locks", f"{env.name}.txt")))
    workers = {module: Path(source) for module, source in backend.worker_sources.items()}
    return ensure_env(env, env_lock, profile.env_root(env.kind), settings=profile.envs, workers=workers)


PACKAGED = Packaged(CATALOGUE, open_env)
use_packaged(PACKAGED)
