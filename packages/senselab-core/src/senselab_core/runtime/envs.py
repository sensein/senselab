"""Locked envs: `ensure_env` builds an `Env.isolated` or `Env.server` env from its lock and describes `Env.host`.

An isolated env is a uv venv on a uv-managed interpreter (`--python-preference only-managed`, the lock's named
version, `t` for free-threaded), installed from its lock: a lock with `torch:` pins installs them from the PyTorch
wheel index routed to the host's CUDA (`pick_torch_index`), then the lock body with `--no-deps`. Under `root`, env
`<name>` lives in `<name>[-<tag>]-<uuid>/`, and `<name>[-<tag>].json` (`env-record@1.0`) names the current build;
both are written under a `Lease` on `<name>[-<tag>]`. A build is never moved or removed here, so a broken lease
costs a duplicate build, never a mixed env, and a superseded build stays until an explicit clean-up. The worker
runtime is copied into every env as an overlay that leaves its wheel set unchanged.
"""

import errno
import logging
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import sysconfig
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import senselab_worker
from pydantic import ValidationError
from senselab_worker.digest import file_digest, identity_digest

from senselab_core.kernel.declare import Env
from senselab_core.kernel.formats import FormatTag, Formatted
from senselab_core.runtime.hardware import _run
from senselab_core.runtime.locks import TIMEOUT_S, Lease
from senselab_core.runtime.profiles import Envs
from senselab_core.runtime.storage import write_atomically

logger = logging.getLogger("senselab")

PYTORCH_INDEX_BASE = "https://download.pytorch.org/whl"
CPU_INDEX = f"{PYTORCH_INDEX_BASE}/cpu"
CUDA_INDEXES: dict[str, tuple[int, int]] = {
    "cu130": (13, 0),
    "cu128": (12, 8),
    "cu126": (12, 6),
    "cu124": (12, 4),
    "cu121": (12, 1),
}
"""The PyTorch CUDA wheel indexes, newest first."""
DEFAULT_MAX_CUDA = CUDA_INDEXES["cu128"]
"""The CUDA ceiling of a lock that names no `torch-index:`."""
PYTHON_PREFERENCE = "only-managed"
ENV_RECORD_FORMAT = "env-record@1.0"
WHEEL_SET_TAG = "wheel-set@1"
_PYTHON_VERSION = re.compile(r"^\d+\.\d+t?$")
_SMI_CUDA = re.compile(r"CUDA Version:\s*(\d+)\.(\d+)")
_NVCC_CUDA = re.compile(r"release\s+(\d+)\.(\d+)")
_NO_WHEEL = re.compile(
    r"(?:no matching distribution(?: found)?(?: for)?|could not find a (?:version|distribution)"
    r"(?: that satisfies)?(?: the requirement)?)\s+`([^`]+)`",
    re.IGNORECASE,
)
_NO_INTERPRETER = re.compile(r"no interpreter found|no download found", re.IGNORECASE)
_OVERLAY_STAMP = "senselab_worker.runtime"


class EnvLockError(ValueError):
    """A lock file without the header fields an env is built from."""


class ManagedPythonMissing(RuntimeError):
    """No uv-managed interpreter of the version an env needs; the message names `uv python install <version>`.

    Attributes:
        version: The interpreter version, `t` included for free-threaded.
    """

    def __init__(self, version: str, detail: str) -> None:
        """Record `version`; `detail` says what was tried."""
        super().__init__(f"{detail}; install one with `uv python install {version}`")
        self.version = version


class CudaCompatibilityError(RuntimeError):
    """The routed PyTorch index serves no wheel for the lock's torch pins."""

    def __init__(self, host: "HostCuda", index: "TorchIndex", packages: list[str]) -> None:
        """Name the host CUDA, the index tried and the packages it lacks."""
        cuda = ".".join(map(str, host.version)) if host.version else "none"
        super().__init__(
            f"no wheel for {packages} on {index.url} (host CUDA {cuda}, from {host.source}); set the execution "
            f"profile's envs.torch_index to another index, e.g. {CPU_INDEX}"
        )


@dataclass(frozen=True)
class HostCuda:
    """The host's CUDA version, `None` when no probe reports one, and the probe that reported it."""

    version: tuple[int, int] | None
    source: Literal["nvidia-smi", "nvcc", "none"]


@dataclass(frozen=True)
class TorchIndex:
    """The PyTorch wheel index torch is installed from: its URL, tag (`cu128`, `cpu`, `profile`) and source."""

    url: str
    tag: str
    source: Literal["probe", "profile"]


@dataclass(frozen=True)
class EnvLock:
    """A lock file's header: the interpreter version, the torch pins and the newest CUDA index they are served on."""

    path: Path
    python: str
    torch_pins: tuple[str, ...]
    max_cuda: tuple[int, int]

    @classmethod
    def read(cls, path: Path) -> "EnvLock":
        """Parse the `# key: value` header of the lock at `path`.

        Raises:
            EnvLockError: `python:` or `torch:` is missing, or a value is not one the builder knows.
        """
        header: dict[str, str] = {}
        for line in Path(path).read_text("utf-8").splitlines():
            if not line.startswith("#"):
                break
            key, sep, value = line[1:].partition(":")
            if sep:
                header.setdefault(key.strip(), value.strip())
        missing = [k for k in ("python", "torch") if k not in header]
        if missing:
            raise EnvLockError(f"lock {path} has no {missing} header line")
        if not _PYTHON_VERSION.fullmatch(header["python"]):
            raise EnvLockError(f"lock {path} names python {header['python']!r}, not <major>.<minor>[t]")
        ceiling = header.get("torch-index", "none")
        if ceiling not in (*CUDA_INDEXES, "cpu", "none"):
            raise EnvLockError(f"lock {path} names torch-index {ceiling!r}; known: {[*CUDA_INDEXES, 'cpu']}")
        torch = header["torch"]
        return cls(
            path=Path(path),
            python=header["python"],
            torch_pins=tuple(torch.split()) if torch != "none" else (),
            max_cuda=CUDA_INDEXES.get(ceiling, (0, 0) if ceiling == "cpu" else DEFAULT_MAX_CUDA),
        )


@dataclass(frozen=True)
class EnvHandle:
    """A ready env: its interpreter and the identities an execution context records.

    `lock_sha` is the lock file's sha256, `wheel_set` the digest of the installed distributions, `torch_index` the
    URL torch came from (None for a torch-free env), `python_version` the interpreter's (`t` for free-threaded).
    """

    python: str
    python_version: str
    lock_sha: str
    wheel_set: str
    torch_index: str | None

    def context_fields(self) -> dict[str, Any]:
        """Return the `ExecutionContext` fields this env decides: `env_lock`, `wheel_set` and `driver`."""
        driver = {"torch_index": self.torch_index} if self.torch_index else {}
        return {"env_lock": self.lock_sha, "wheel_set": self.wheel_set, "driver": driver}


class EnvRecord(Formatted):
    """The current build of one env under a root, at `<root>/<name>[-<tag>].json`."""

    format: FormatTag = ENV_RECORD_FORMAT
    directory: str
    lock_sha: str
    torch_index: str | None
    python_version: str
    wheel_set: str


@dataclass(frozen=True)
class Uv:
    """The `uv` command line every build runs through."""

    command: tuple[str, ...]

    @classmethod
    def find(cls) -> "Uv":
        """Return `uv` from `PATH`, else `~/.local/bin/uv` or `~/.cargo/bin/uv`.

        Raises:
            FileNotFoundError: none of them is an executable.
        """
        home = Path.home()
        for candidate in (shutil.which("uv"), home / ".local" / "bin" / "uv", home / ".cargo" / "bin" / "uv"):
            if candidate and os.access(candidate, os.X_OK):
                return cls((str(candidate),))
        raise FileNotFoundError("uv not found on PATH, in ~/.local/bin or in ~/.cargo/bin")

    def __call__(self, *args: str) -> subprocess.CompletedProcess[str]:
        """Run `uv <args>`, capturing its output; the caller reads `returncode`."""
        return subprocess.run([*self.command, *args], capture_output=True, text=True, check=False)


def detect_host_cuda(*, run: Callable[[list[str]], str] = _run) -> HostCuda:
    """Return the CUDA version `nvidia-smi` reports, else the one `nvcc --version` reports, else none.

    `run` executes a command and returns its stdout, raising when it cannot; tests inject it.
    """
    for command, pattern, source in (
        (["nvidia-smi"], _SMI_CUDA, "nvidia-smi"),
        (["nvcc", "--version"], _NVCC_CUDA, "nvcc"),
    ):
        try:
            match = pattern.search(run(command))
        except (OSError, subprocess.SubprocessError):
            continue
        if match:
            return HostCuda((int(match[1]), int(match[2])), source)  # type: ignore[arg-type]
    return HostCuda(None, "none")


def pick_torch_index(
    host: HostCuda, *, override: str | None = None, max_cuda: tuple[int, int] | None = None
) -> TorchIndex:
    """Return `override` when set, else the newest CUDA index at or below the host's CUDA capped at `max_cuda`.

    `max_cuda` defaults to `DEFAULT_MAX_CUDA`; a host without CUDA, or older than every index, gets the CPU index.
    """
    if override:
        return TorchIndex(override, "profile", "profile")
    if host.version is not None:
        effective = min(host.version, max_cuda or DEFAULT_MAX_CUDA)
        for tag, version in CUDA_INDEXES.items():
            if version <= effective:
                return TorchIndex(f"{PYTORCH_INDEX_BASE}/{tag}", tag, "probe")
    return TorchIndex(CPU_INDEX, "cpu", "probe")


def ensure_env(  # noqa: PLR0913 - the env, its lock, root and profile settings, and two injectable boundaries
    env: Env,
    lock: Path,
    root: Path,
    *,
    settings: Envs,
    uv: Uv | None = None,
    host_cuda: HostCuda | None = None,
) -> EnvHandle:
    """Return the env `env` built from `lock` under `root`, building it once across processes.

    For `Env.host()`, `lock` is the senselab lock and the env is this interpreter's, which must be uv-managed.

    Args:
        env: The env a backend declares.
        lock: The lock file the env is installed from.
        root: The directory the profile's `env_roots` gives for `env.kind`; unused for a host env.
        settings: The profile's `envs`: `torch_index` replaces the probed index when set, and a build's lease is
            broken once older than `lease_stale_after_s`; a waiter waits that long plus `locks.TIMEOUT_S`.
        uv: The `uv` command; `Uv.find()` when None.
        host_cuda: The host's CUDA; probed with `detect_host_cuda` when None and the lock pins torch.

    Raises:
        ManagedPythonMissing: no uv-managed interpreter of the needed version, or the host's is not uv-managed.
        CudaCompatibilityError: the routed index has no wheel for the lock's torch pins.
        EnvLockError: the lock's header lacks what a build needs.
        RuntimeError: another `uv` step failed; the message carries uv's stderr.
        LeaseTimeout, LeaseUnavailable: as `Lease`.
    """
    uv = uv or Uv.find()
    if env.kind == "host":
        return _host_env(lock, uv)
    parsed = EnvLock.read(lock)
    index = None
    if parsed.torch_pins:
        host_cuda = host_cuda or detect_host_cuda()
        index = pick_torch_index(host_cuda, override=settings.torch_index, max_cuda=parsed.max_cuda)
    stem = f"{env.name}-{index.tag}" if index else str(env.name)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock_sha = file_digest("file@1", parsed.path)
    wanted = (lock_sha, index.url if index else None)
    current = _read_record(root, stem)
    if (
        current is not None
        and (current.lock_sha, current.torch_index) == wanted
        and _overlaid(root / current.directory)
    ):
        return _handle(root, current)
    stale_after = settings.lease_stale_after_s
    with Lease(root / stem, stale_after=stale_after, timeout=stale_after + TIMEOUT_S):
        current = _read_record(root, stem)
        if current is None or (current.lock_sha, current.torch_index) != wanted:
            current = _build(root, stem, parsed, lock_sha, index, uv, host_cuda)
            write_atomically(root / f"{stem}.json", current.model_dump_json().encode())
        _overlay(root / current.directory)
    return _handle(root, current)


def _handle(root: Path, record: EnvRecord) -> EnvHandle:
    return EnvHandle(
        python=str(_python(root / record.directory)),
        python_version=record.python_version,
        lock_sha=record.lock_sha,
        wheel_set=record.wheel_set,
        torch_index=record.torch_index,
    )


def _read_record(root: Path, stem: str) -> EnvRecord | None:
    try:
        record = EnvRecord.model_validate_json((root / f"{stem}.json").read_bytes())
    except FileNotFoundError:
        return None
    except (OSError, ValidationError) as err:
        logger.warning("env record %s.json is unreadable and is rebuilt: %s", stem, err)
        return None
    if not re.fullmatch(rf"{re.escape(stem)}-[0-9a-f]{{32}}", record.directory):
        logger.warning("env record %s.json names %r, not a build of %s; it is rebuilt", stem, record.directory, stem)
        return None
    return record if (root / record.directory).is_dir() else None


def _host_env(lock: Path, uv: Uv) -> EnvHandle:
    suffix = "t" if sysconfig.get_config_var("Py_GIL_DISABLED") else ""
    managed = _check(uv("python", "dir"), "find the uv-managed interpreters").stdout.strip()
    if not managed or not Path(sys.base_prefix).resolve().is_relative_to(Path(managed).resolve()):
        raise ManagedPythonMissing(
            f"{sys.version_info.major}.{sys.version_info.minor}{suffix}",
            f"the host interpreter {sys.executable} is not uv-managed",
        )
    return EnvHandle(
        python=sys.executable,
        python_version=platform.python_version() + suffix,
        lock_sha=file_digest("file@1", lock),
        wheel_set=_wheel_set(Path(sysconfig.get_paths()["purelib"])),
        torch_index=None,
    )


def _build(  # noqa: PLR0913 - one build's inputs
    root: Path,
    stem: str,
    lock: EnvLock,
    lock_sha: str,
    index: TorchIndex | None,
    uv: Uv,
    host_cuda: HostCuda | None,
) -> EnvRecord:
    directory = root / f"{stem}-{uuid.uuid4().hex}"
    logger.info("building env %s on Python %s (torch index %s)", stem, lock.python, index.tag if index else "none")
    try:
        created = uv("venv", "--python", lock.python, "--python-preference", PYTHON_PREFERENCE, str(directory))
        if created.returncode and _NO_INTERPRETER.search(created.stderr):
            raise ManagedPythonMissing(
                lock.python,
                f"uv has no managed Python {lock.python} "
                f"(--python-preference {PYTHON_PREFERENCE}): {created.stderr.strip()}",
            )
        _check(created, f"create a venv on Python {lock.python}")
        python = str(_python(directory))
        if index is not None:
            torch = uv("pip", "install", "--index-url", index.url, "--python", python, *lock.torch_pins)
            missing = _NO_WHEEL.findall(torch.stderr)
            if torch.returncode and missing:
                raise CudaCompatibilityError(host_cuda or HostCuda(None, "none"), index, list(dict.fromkeys(missing)))
            _check(torch, "install torch")
        body = ("pip", "install", "--python", python, "--no-deps", "--no-sources", "--requirement", str(lock.path))
        _check(uv(*body), "install the lock")
        _group_readable(directory)
        version = re.search(r"^version(?:_info)?\s*=\s*(\S+)", (directory / "pyvenv.cfg").read_text(), re.MULTILINE)
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    return EnvRecord(
        directory=directory.name,
        lock_sha=lock_sha,
        torch_index=index.url if index else None,
        python_version=(version[1] if version else lock.python.rstrip("t"))
        + ("t" if lock.python.endswith("t") else ""),
        wheel_set=_wheel_set(_site(directory)),
    )


def _check(result: subprocess.CompletedProcess[str], what: str) -> subprocess.CompletedProcess[str]:
    if result.returncode:
        raise RuntimeError(f"uv could not {what} (exit {result.returncode}): {result.stderr.strip()}")
    return result


def _python(directory: Path) -> Path:
    return directory / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def _site(directory: Path) -> Path:
    found = (
        [directory / "Lib" / "site-packages"]
        if sys.platform == "win32"
        else list(directory.glob("lib/python*/site-packages"))
    )
    if len(found) != 1:
        raise RuntimeError(f"env {directory.name} has {len(found)} site-packages directories, not one")
    return found[0]


def _wheel_set(site: Path) -> str:
    """Return `identity_digest("wheel-set@1", [[name, version], ...])` over the `*.dist-info` names in `site`."""
    installed = sorted(p.name.removesuffix(".dist-info").rpartition("-")[::2] for p in site.glob("*.dist-info"))
    return identity_digest(WHEEL_SET_TAG, [list(pair) for pair in installed])


def _overlaid(directory: Path) -> bool:
    try:
        return (_site(directory) / _OVERLAY_STAMP).read_text() == senselab_worker.runtime_identity()
    except (OSError, RuntimeError):
        return False


def _overlay(directory: Path) -> None:
    """Copy this worker runtime into the env's site-packages unless the copy there is already this one."""
    if _overlaid(directory):
        return
    site = _site(directory)
    source = Path(senselab_worker.__file__).parent
    fresh, old = site / f".senselab_worker-{uuid.uuid4().hex}", site / f".senselab_worker-{uuid.uuid4().hex}"
    shutil.copytree(source, fresh, ignore=shutil.ignore_patterns("__pycache__"))
    try:
        (site / "senselab_worker").rename(old)
    except FileNotFoundError:
        pass
    fresh.rename(site / "senselab_worker")
    shutil.rmtree(old, ignore_errors=True)
    write_atomically(site / _OVERLAY_STAMP, senselab_worker.runtime_identity().encode())


def _group_readable(directory: Path) -> None:
    """Give the group read on every file and traverse on every directory, execute where the owner has it."""
    for here, _, files in os.walk(directory):
        for path, is_dir in ((Path(here), True), *((Path(here) / f, False) for f in files)):
            try:
                mode = path.lstat().st_mode
                if stat.S_ISLNK(mode):
                    continue
                exec_bit = stat.S_IXGRP if is_dir or mode & stat.S_IXUSR else 0
                path.chmod(mode | stat.S_IRGRP | exec_bit)
            except OSError as err:
                if err.errno not in (errno.EPERM, errno.EACCES):
                    raise


__all__ = [
    "CPU_INDEX",
    "CUDA_INDEXES",
    "DEFAULT_MAX_CUDA",
    "ENV_RECORD_FORMAT",
    "PYTHON_PREFERENCE",
    "CudaCompatibilityError",
    "EnvHandle",
    "EnvLock",
    "EnvLockError",
    "EnvRecord",
    "HostCuda",
    "ManagedPythonMissing",
    "TorchIndex",
    "Uv",
    "detect_host_cuda",
    "ensure_env",
    "pick_torch_index",
]
