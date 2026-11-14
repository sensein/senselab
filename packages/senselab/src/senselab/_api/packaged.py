"""The packaged catalogue, env locks and env licenses; the `senselab.packages` `register`."""

import importlib
from importlib import resources
from pathlib import Path

from senselab_core.kernel.declare import BackendSpec, Registry, declare_module
from senselab_core.runtime.call import EnvUnreadable, Packaged
from senselab_core.runtime.catalogue import Catalogue
from senselab_core.runtime.envs import (
    CudaCompatibilityError,
    EnvHandle,
    EnvLockError,
    ManagedPythonMissing,
    current_env,
    ensure_env,
    env_built,
)
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
            env,
            lock,
            root,
            settings=profile.envs,
            lease=profile.lease,
            requirements=backend.requirements,
            network=profile.network,
        )
    env_lock, workers = _built_from(backend)
    return ensure_env(
        env,
        env_lock,
        profile.env_root(env.kind),
        settings=profile.envs,
        lease=profile.lease,
        workers=workers,
        network=profile.network,
    )


def env_lock(backend: BackendSpec) -> Path:
    """Return the lock under `data/env_locks/` the isolated or server env `backend` declares is built from."""
    return _built_from(backend)[0]


def _built_from(backend: BackendSpec) -> tuple[Path, dict[str, Path]]:
    """Return the lock an isolated or server env is built from and the worker modules overlaid on it."""
    env_lock = Path(str(_DATA.joinpath("env_locks", f"{backend.env.name}.txt")))
    return env_lock, {module: Path(source) for module, source in backend.worker_sources.items()}


def is_built(backend: BackendSpec, profile: ExecutionProfile) -> bool:
    """Return whether the env `backend` declares is built under `profile`; a host env always is.

    Raises:
        As `env_built`.
    """
    if backend.env.kind == "host":
        return True
    env_lock, workers = _built_from(backend)
    return env_built(backend.env, env_lock, profile.env_root(backend.env.kind), settings=profile.envs, workers=workers)


def read_env(backend: BackendSpec, profile: ExecutionProfile) -> EnvHandle | None:
    """Return the handle of the env `backend` declares if it is built under `profile`, else None; builds nothing.

    A host env is always built: its handle is this interpreter's.

    Raises:
        EnvUnreadable: for any cause `current_env` or, for a host env, `open_env` fails.
    """
    try:
        if backend.env.kind == "host":
            return open_env(backend, profile)
        env_lock, workers = _built_from(backend)
        root = profile.env_root(backend.env.kind)
        return current_env(backend.env, env_lock, root, settings=profile.envs, workers=workers)
    except (EnvLockError, ManagedPythonMissing, CudaCompatibilityError, OSError, RuntimeError, LookupError) as err:
        raise EnvUnreadable(str(err)) from err


PACKAGED = Packaged(
    CATALOGUE,
    open_env,
    lock_records=packaged_lock_records("senselab"),
    env_built=is_built,
    env_read=read_env,
    env_lock=env_lock,
)


def register(registry: Registry) -> Packaged:
    """Declare the tasks and backends of `DECLARATION_MODULES` in `registry` and return `PACKAGED`.

    Raises:
        As `declare_module`.
    """
    for name in DECLARATION_MODULES:
        declare_module(registry, importlib.import_module(name))
    return PACKAGED
