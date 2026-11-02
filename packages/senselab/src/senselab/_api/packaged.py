"""The packaged catalogue, env locks, tolerance profiles and env licenses; the `senselab.packages` `register`."""

import importlib
from importlib import resources
from pathlib import Path

from senselab_core.kernel.declare import BackendSpec, Registry, declare_module
from senselab_core.runtime.call import Packaged
from senselab_core.runtime.catalogue import Catalogue
from senselab_core.runtime.envs import EnvHandle, ensure_env
from senselab_core.runtime.licenses import packaged_lock_records
from senselab_core.runtime.profiles import ExecutionProfile

from senselab._api.declarations import DECLARATION_MODULES

_DATA = resources.files("senselab").joinpath("data")
CATALOGUE = Catalogue.load(_DATA.joinpath("catalogue.yaml").read_text("utf-8"))
"""The packaged catalogue, `senselab/data/catalogue.yaml`."""


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
        return ensure_env(
            env, lock, root, settings=profile.envs, requirements=backend.requirements, network=profile.network
        )
    env_lock = Path(str(_DATA.joinpath("env_locks", f"{env.name}.txt")))
    workers = {module: Path(source) for module, source in backend.worker_sources.items()}
    return ensure_env(
        env, env_lock, profile.env_root(env.kind), settings=profile.envs, workers=workers, network=profile.network
    )


PACKAGED = Packaged(
    CATALOGUE, open_env, tolerance=_DATA.joinpath("tolerance"), lock_records=packaged_lock_records("senselab")
)


def register(registry: Registry) -> Packaged:
    """Declare the tasks and backends of `DECLARATION_MODULES` in `registry` and return `PACKAGED`.

    Raises:
        As `declare_module`.
    """
    for name in DECLARATION_MODULES:
        declare_module(registry, importlib.import_module(name))
    return PACKAGED
